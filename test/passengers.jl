# Every record of a satellite goes to `step_loop`, whichever of its signals it belongs
# to: the driver's close the loops, a passenger's leave a scalar loop alone, and the
# vector loop decodes its decoding signal's. One host loop drives every estimator.

# A record of `signal` ending at `sample_index`, `n` samples long, with the prompt `p`.
passenger_output(p, n, sample_index) =
    CorrelatorOutput(EarlyPromptLateCorrelator(SVector{3,ComplexF64}(0.5p, p, 0.5p), 0.5), n, sample_index, 0.0)

@testset "A passenger record leaves $(nameof(typeof(estimator))) alone" for estimator in (
    ConventionalAssistedPLLAndDLL(),
    NCOReferencedPLLAndDLL(),
)
    driver, passenger = GPSL5Q(), GPSL5I()
    fs = 25e6Hz
    state = init_estimator_state(estimator, driver, 100.0Hz, 0.1Hz)
    output = passenger_output(cis(0.3), 25_000, 25_000)
    record = LoopRecord(passenger, output.correlator, cis(0.2), output, 1, fs; prn = 3)
    # The state as it was, and the command in force where this fold's would land.
    @test step_loop(estimator, state, record, FixedNCOWord(120.0, 0.12), NO_LANDING_SAMPLE) ==
          (state, 120.0Hz, 0.12Hz)
    timeline = NCOTimeline()
    reset_timeline!(timeline, 100.0, 0.1)
    schedule_word!(timeline, 50_000, 130.0, 0.13)
    @test step_loop(estimator, state, record, timeline, Int64(60_000)) == (state, 130.0Hz, 0.13Hz)
    @test step_loop(estimator, state, record, timeline, NO_LANDING_SAMPLE) == (state, 100.0Hz, 0.1Hz)
    # The driver's record steps the loop.
    driver_record = LoopRecord(driver, output.correlator, cis(0.2), output, 1, fs; prn = 3)
    stepped, = step_loop(estimator, state, driver_record, FixedNCOWord(100.0, 0.1), NO_LANDING_SAMPLE)
    @test stepped != state
end

# The satellite of the host loop below: a driver and its passengers, in the order a host
# folds a tick's records, one record per code block at 4 MHz, every one on the same
# replica. A data signal's prompt carries a pseudo-random bit per symbol; each signal's
# prompt sits at its carrier-phase offset against the driver's.
struct HostSignal{S,L}
    signal::S
    state::Base.RefValue{L}
    samples_per_record::Int
end

HostSignal(signal) = HostSignal(
    signal,
    Ref(SignalLoopState(signal)),
    round(Int, 4e6 * get_code_length(signal) / ustrip(Hz, get_code_frequency(signal))),
)

function host_prompt(host::HostSignal, driver, sample_index)
    rate = ustrip(Hz, get_data_frequency(host.signal))
    symbol = iszero(rate) ? 0 : floor(Int, sample_index / 4e6 * rate - 1e-9)
    bit = iszero(rate) || isodd(hash(symbol) >> 7) ? 1.0 : -1.0
    bit * cis(get_carrier_phase_offset(host.signal) - get_carrier_phase_offset(driver))
end

# The host loop, written once for every estimator: for every record of every signal,
# fold it into the signal's state, build the record from that state, step the
# satellite's one estimator state with it and apply the returned Dopplers. Returns the
# Dopplers the driver's records produced, and the final estimator state.
function run_host!(estimator, driver::HostSignal, passengers::Tuple, prn, num_ms)
    signals = (driver, passengers...)
    state = init_estimator_state(estimator, driver.signal, 100.0Hz, 0.0Hz)
    words = FixedNCOWord(100.0, 0.0)
    driver_dopplers = Tuple{typeof(1.0Hz),typeof(1.0Hz)}[]
    for ms = 1:num_ms, host in signals
        sample_index = 4000ms
        sample_index % host.samples_per_record == 0 || continue
        n = host.samples_per_record
        output = passenger_output(n * host_prompt(host, driver.signal, sample_index), n, sample_index)
        previous = host.state[].last_filtered_prompt
        host.state[], _, filtered, blocks = apply_record(host.state[], host.signal, prn, output, 4e6Hz,
            1e-6 / Hz, true, get_carrier_phase_offset(driver.signal))
        record = LoopRecord(host.signal, filtered, previous, output, blocks, 4e6Hz;
            prn, signal_state = host.state[])
        state, carrier, code = step_loop(estimator, state, record, words, NO_LANDING_SAMPLE)
        host === driver && push!(driver_dopplers, (carrier, code))
        host === driver || @test (carrier, code) == (words.carrier_doppler * Hz, words.code_doppler * Hz)
        words = FixedNCOWord(ustrip(Hz, carrier), ustrip(Hz, code))
        empty!(get_soft_bits(host.state[]))
    end
    driver_dopplers, state
end

@testset "One host loop runs $name" for (name, make, driver, decoding) in (
    ("a scalar loop", () -> ConventionalAssistedPLLAndDLL(), GPSL1C_P(), GPSL1CA()),
    ("the NCO-referenced loop", () -> NCOReferencedPLLAndDLL(), GPSL1C_P(), GPSL1CA()),
    ("a plain vector loop", () -> VectorPLLAndDLL(GPSL1CA()), GPSL1CA(), GPSL1CA()),
    ("a pilot + data vector loop", () -> VectorPLLAndDLL(GPSL1C_P() => GPSL1CA()), GPSL1C_P(), GPSL1CA()),
)
    # Two passengers of the same band at the same chip rate, the second neither the
    # driver nor decoded: GPS L1C-D with the C/A pair, the L1C pilot with plain C/A.
    others = driver isa GPSL1CA ? (GPSL1C_P(), GPSL1C_D()) : (GPSL1CA(), GPSL1C_D())
    alone_estimator = make()
    alone, _ = run_host!(alone_estimator, HostSignal(driver), (), 7, 600)
    estimator = make()
    with_passengers, state = run_host!(estimator, HostSignal(driver), map(HostSignal, others), 7, 600)
    # The passengers change nothing of the driver's loop, bit for bit.
    @test with_passengers == alone
    @test length(alone) == 600 * 4000 ÷ HostSignal(driver).samples_per_record
    if estimator isa VectorPLLAndDLL
        # The vector loop decodes the decoding signal: its bit clock found the edges and
        # its decoder took the bits, and the satellite is known by its driver.
        report = satellite_report(estimator, driver, 7)
        @test report.bit_synced
        @test report.decoder.prn == 7
        @test report.decoder isa typeof(TrackingLoops.GNSSDecoderState(decoding, 7))
        @test isfinite(report.cn0_dbhz)
        @test satellite_report(estimator, GPSL1C_D(), 7) === nothing
        @test estimator.navigation.registrations == 1
        @test navigation_cycle(estimator) >= 5
        # Without its decoding signal the satellite never syncs.
        @test satellite_report(alone_estimator, driver, 7).bit_synced == (driver isa typeof(decoding))
    else
        # A scalar loop neither decodes nor solves.
        @test navigation_solution(estimator) === nothing
        @test satellite_report(estimator, driver, 7) === nothing
    end
end
