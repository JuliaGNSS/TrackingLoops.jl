# Vector tracking of satellites tracked on a dataless pilot that drives, with its data
# component as a passenger: the engine decodes the bits from the passenger's records,
# and with `combine_signals` fuses both signals' readings into the filter's
# measurements. From records alone, on the pipeline harness of `vector_simulation.jl`.
#
# The fixtures broadcast GPS L1 C/A LNAV, so the pilot here is a dataless copy of
# L1 C/A on the same code and carrier, the C/A signal its data passenger.

using GNSSDecoder: GPSL1CAData
import PositionVelocityTime

struct TestL1Pilot{C<:AbstractMatrix} <: GNSSSignals.AbstractGPSSignal{C}
    codes::C
    lut::GNSSSignals.SignalLUT
end

TestL1Pilot() = (ca = GPSL1CA(); TestL1Pilot(ca.codes, ca.lut))

for f in (:get_modulation, :get_carrier_phase_offset, :get_band, :get_code_length, :get_code_frequency)
    @eval GNSSSignals.$f(::Type{<:TestL1Pilot}) = GNSSSignals.$f(GPSL1CA)
end
GNSSSignals.get_relative_power(::Type{<:TestL1Pilot}) = GNSSSignals.get_relative_power(GPSL1CA)
GNSSSignals.get_signal_name(::Type{<:TestL1Pilot}) = "test L1 pilot"
GNSSSignals.get_secondary_code(::TestL1Pilot) = GNSSSignals.NoSecondaryCode()
GNSSSignals.get_data_frequency(::Type{<:TestL1Pilot}) = 0Hz
PositionVelocityTime.correct_by_group_delay(
    decoder::GNSSDecoderState{<:GPSL1CAData},
    ::TestL1Pilot,
    t,
) = t - decoder.data.T_GD
TrackingLoops.get_default_correlator(::TestL1Pilot, num_ants::NumAnts = NumAnts(1)) =
    get_default_correlator(GPSL1CA(), num_ants)

@testset "A vector loop's signal groups" begin
    # A satellite tracked on several signals lists them, the driver first; the bits are
    # decoded from the first that carries any.
    v = VectorPLLAndDLL((GalileoE1C(), GalileoE1B()), GPSL1CA(), (GalileoE1B(), GalileoE1C()))
    pilot_driven, single, data_driven = v.navigation.groups
    @test pilot_driven.signal isa GalileoE1C
    @test pilot_driven.passengers isa Tuple{GalileoE1B}
    @test pilot_driven.data_signal isa GalileoE1B
    @test !TL._drives_data_signal(pilot_driven)
    @test single.passengers === () && TL._drives_data_signal(single)
    @test data_driven.data_signal isa GalileoE1B && TL._drives_data_signal(data_driven)
    @test all(slot -> length(slot.passengers) == 1, pilot_driven.slots)
    # The filter ranges on the drivers.
    @test v.navigation.layout.signal_id_by_group == [:GalileoE1C, :GPSL1CA, :GalileoE1B]

    @test_throws ArgumentError VectorPLLAndDLL(GalileoE1C())
    @test_throws ArgumentError VectorPLLAndDLL((GalileoE1C(),))
    @test_throws ArgumentError VectorPLLAndDLL((GPSL1CA(), GPSL5I()))
    @test_throws ArgumentError VectorPLLAndDLL((GalileoE1B(), GalileoE1B()))
    @test_throws ArgumentError VectorPLLAndDLL((GalileoE1C(), GalileoE1B()), GalileoE1C())
    @test_throws ArgumentError VectorPLLAndDLL(())
    # One satellite's signals come from one constellation.
    @test_throws ArgumentError VectorPLLAndDLL((GPSL1CA(), GalileoE1B()))

    # An estimator that takes no passenger records leaves the state as it is.
    alone = VectorPLLAndDLL(GPSL1CA())
    @test !takes_passenger_records(alone)
    state = init_estimator_state(alone, GPSL1CA(), 0.0Hz, 0.0Hz)
    correlator = update_accumulator(get_default_correlator(GPSL5I()), SVector(0.5, 1.0, 0.5) .* cis(0.1))
    record = LoopRecord(GPSL5I(), correlator, 0.0im, 5000, 5000, 5000, 1, 5e6Hz; prn = 3)
    @test fold_passenger_record(alone, state, record, _NO_WORD; driver_signal = GPSL1CA()) === state
end

# A record of `signal` for satellite `prn`, ending at `sample_index`, 1 ms at 5 MHz.
function group_record(signal, p, taps, sample_index; previous_prompt = 0.0im, prn = 3, cn0 = NaN)
    correlator = update_accumulator(get_default_correlator(signal), p .* SVector(taps...))
    LoopRecord(signal, correlator, previous_prompt, 5000, sample_index, sample_index, 1, 5e6Hz;
        prn, cn0)
end

# The bytes one passenger record folded into a pilot-driven L5 satellite allocates.
fold_and_measure(estimator, state, record) = @allocated fold_passenger_record(
    estimator,
    state,
    record,
    _NO_WORD;
    driver_signal = estimator.navigation.groups[1].signal,
    differential_group_delay_chips = 0.01,
)

@testset "Passenger records in the navigation engine, $(combine ? "" : "not ")combining" for combine in (false, true)
    taps = (0.5, 1.0, 0.4)
    v = VectorPLLAndDLL((GPSL5Q(), GPSL5I()); combine_signals = combine, cycle_time = 10ms)
    group = only(v.navigation.groups)
    state = init_estimator_state(v, GPSL5Q(), 0.0Hz, 0.0Hz)
    passenger(sample_index; kw...) = group_record(GPSL5I(), cis(0.1), taps, sample_index; previous_prompt = cis(0.05), kw...)
    fold(state, record; delay = 0.01) = fold_passenger_record(v, state, record, _NO_WORD;
        driver_signal = GPSL5Q(), differential_group_delay_chips = delay)

    # Before its first driver record the satellite has no slot: the passenger reaches
    # the inner loop only, when combining.
    early = fold(state, passenger(5000))
    @test (early.inner.signal_combining_sums.pll.weight > 0s) == combine
    @test all(slot -> !slot.occupied, group.slots)
    state, = step_loop(v, state, group_record(GPSL5Q(), cis(0.0), taps, 5000), _NO_WORD, NO_LANDING_SAMPLE)
    slot = group.slots[state.slot]
    @test slot.occupied && slot.prn == 3

    # The data passenger of a pilot driver runs the bit clock, combining or not.
    state = fold(state, passenger(10000))
    @test slot.data_last_end_sample == 10000
    @test slot.last_end_sample == 5000
    @test only(slot.passengers).integration_time ≈ 1ms

    # In the vector loop and combining, its readings wait in the state for the driver's
    # next record.
    in_loop = TL._enable_vector_tracking(state)
    in_loop = fold(in_loop, passenger(50000))
    r = only(in_loop.passenger_readings)
    @test r.pending_code == (combine ? (1, last(r.pending_code)) : (0, 0.0))
    @test first(r.pending_carrier) == (combine ? 1 : 0)
    @test r.code == (0, 0.0)
    expected_dll = dll_disc(GPSL5I(), passenger(50000).filtered_correlator, 0.0Hz, 5e6Hz) + 0.01
    combine && @test last(r.pending_code) ≈ expected_dll
    # Unknown group delay: the rate reading only. Without a previous prompt: the code
    # reading only.
    in_loop = fold(in_loop, passenger(51000); delay = NaN)
    in_loop = fold(in_loop, group_record(GPSL5I(), cis(0.1), taps, 52000))
    r = only(in_loop.passenger_readings)
    @test first(r.pending_code) == (combine ? 2 : 0)
    @test first(r.pending_carrier) == (combine ? 2 : 0)
    # The driver's next record moves them into its cycle, as its own readings.
    driver = group_record(GPSL5Q(), cis(0.0), taps, 55000; previous_prompt = cis(0.0))
    stepped, = TL._step_satellite(v, in_loop, driver, _NO_WORD, NO_LANDING_SAMPLE)
    r = only(stepped.passenger_readings)
    @test r.pending_code == (0, 0.0) && r.pending_carrier == (0, 0.0Hz)
    @test first(r.code) == (combine ? 2 : 0) && first(r.carrier) == (combine ? 2 : 0)
    # The snapshot takes the cycle's readings; one folded after the driver record waits
    # for the next.
    stepped = fold(stepped, passenger(56000))
    snapshotted = only(TL._reset_discriminator_accumulators(stepped).passenger_readings)
    @test snapshotted.code == (0, 0.0)
    @test first(snapshotted.pending_code) == (combine ? 1 : 0)
    # Leaving the vector loop drops them, and out of it nothing is counted.
    @test only(TL._disable_vector_tracking(stepped).passenger_readings) == TL.PassengerReadings()
    @test only(fold(state, passenger(58000)).passenger_readings) == TL.PassengerReadings()

    # Without allocating.
    record = passenger(57500)
    fold_and_measure(v, in_loop, record)
    @test fold_and_measure(v, in_loop, record) == 0
    # A record of a signal that is not the driver's passenger is refused.
    @test_throws ArgumentError fold(state, group_record(GPSL1CA(), cis(0.1), taps, 59000))
end

@testset "The filter fuses every signal's readings by their own variance" begin
    v = VectorPLLAndDLL((GPSL5Q(), GPSL5I()); combine_signals = true)
    slot = first(only(v.navigation.groups).slots)
    state = TL._enable_vector_tracking(init_estimator_state(v, GPSL5Q(), 0.0Hz, 0.0Hz))
    # The driver read all 100 records of 1 ms of a 0.1 s cycle.
    state = SatVectorPLLAndDLL(state; code_discr_acc = (100, 2.0), carrier_discr_acc = (100, 200.0Hz))
    slot.cn0_dbhz = 45.0
    slot.coherent_integration_time = 0.001s
    slot.early_late_spacing = 0.5
    T, chip_length, wavelength = 0.1, 29.3, 0.25
    cn0 = TL.linear_cn0_floor(45.0)
    own_code = TL._pseudorange_noise_variance(cn0, 0.001, 0.5, chip_length, 0.1)
    own_rate = TL._pseudorange_rate_noise_variance(cn0, 0.001, wavelength, 0.1)
    # Without passenger readings: the driver's own, bit for bit, at the variance of the
    # span its readings cover.
    code, rate, code_variance, rate_variance, has_code, has_rate =
        TL._member_measurements(slot, state, T, chip_length, wavelength)
    @test (code, rate) ==
          (TL.accumulated_code_discriminator(state, T), TL.accumulated_carrier_discriminator(state))
    @test code_variance ≈ own_code && rate_variance ≈ own_rate
    @test has_code && has_rate
    # Readings for a quarter of the cycle only: four times the code variance, sixteen
    # times the rate variance, as the rate reads the phase over a quarter of the span.
    partial = SatVectorPLLAndDLL(state; code_discr_acc = (25, 0.5), carrier_discr_acc = (25, 50.0Hz))
    partial_measurements = TL._member_measurements(slot, partial, T, chip_length, wavelength)
    @test partial_measurements[3] ≈ 4 * own_code
    @test partial_measurements[4] ≈ 16 * own_rate

    # A passenger at a known C/N₀ with readings for 40 of the cycle's records: weighted
    # by its inverse variance over the 40 ms they cover.
    estimator = MomentsCN0Estimator(100)
    for k = 1:100
        estimator = update(estimator, cis(0.01k) * (1 + 0.05 * (-1)^k))
    end
    passenger(host_cn0; factor = 1.0) = TL.VTPassenger(estimator, host_cn0, NaN, 0.001s, 0.5, factor)
    # The state with the passenger's cycle readings.
    with_readings(state, code, carrier) = SatVectorPLLAndDLL(
        state;
        passenger_readings = (TL.PassengerReadings((0, 0.0), (0, 0.0Hz), code, carrier),),
    )
    state = with_readings(state, (40, 1.6), (40, 40.0Hz))
    slot.passengers[1] = passenger(NaN)
    code, rate, code_variance, rate_variance, has_code, has_rate =
        TL._member_measurements(slot, state, T, chip_length, wavelength)
    p_cn0 = TL.linear_cn0_floor(TL._capped_cn0_dbhz(estimator, 0.001s))
    p_code = TL._pseudorange_noise_variance(p_cn0, 0.001, 0.5, chip_length, 0.04)
    p_rate = TL._pseudorange_rate_noise_variance(p_cn0, 0.001, wavelength, 0.04)
    advance = TL.code_phase_advance(state, T)
    @test code ≈ ((-0.02 + advance) / own_code + (-0.04 + advance) / p_code) / (1 / own_code + 1 / p_code)
    @test rate ≈ (2.0 / own_rate + 1.0 / p_rate) / (1 / own_rate + 1 / p_rate)
    @test code_variance ≈ 1 / (1 / own_code + 1 / p_code)
    @test rate_variance ≈ 1 / (1 / own_rate + 1 / p_rate)
    @test has_code && has_rate
    # A passenger tracked with the VEML correlator (a BOC signal) is weighted by the
    # BPSK model's variance scaled by its correlator's factor.
    slot.passengers[1] = passenger(NaN; factor = 0.4)
    @test TL._member_measurements(slot, state, T, chip_length, wavelength)[3] ≈
          1 / (1 / own_code + 1 / (0.4 * p_code))
    @test TL._dll_variance_factor(get_default_correlator(GalileoE1B())) == 0.4
    @test TL._dll_variance_factor(get_default_correlator(GPSL1CA())) == 1.0
    # The host's C/N₀ estimate, where it gave one, replaces the passenger's own.
    slot.passengers[1] = passenger(30.0)
    hosted_code = TL._pseudorange_noise_variance(TL.linear_cn0_floor(30.0), 0.001, 0.5, chip_length, 0.04)
    @test TL._member_measurements(slot, state, T, chip_length, wavelength)[3] ≈
          1 / (1 / own_code + 1 / hosted_code)

    # A member whose driver has no reading this cycle is measured on its passenger's.
    slot.passengers[1] = passenger(NaN)
    empty_driver = SatVectorPLLAndDLL(state; code_discr_acc = (0, 0.0), carrier_discr_acc = (0, 0.0Hz))
    code, rate, code_variance, rate_variance, has_code, has_rate =
        TL._member_measurements(slot, empty_driver, T, chip_length, wavelength)
    @test has_code && has_rate
    @test code ≈ -0.04 + advance
    @test rate ≈ 1.0
    @test code_variance ≈ p_code
    # A cycle without an FLL reading of any signal withholds the rate measurement
    # only; the code measurement stands. A passenger's FLL reading alone is one.
    no_fll = SatVectorPLLAndDLL(state; carrier_discr_acc = (0, 0.0Hz))
    @test TL._member_measurements(slot, with_readings(no_fll, (40, 1.6), (0, 0.0Hz)), T,
        chip_length, wavelength)[5:6] == (true, false)
    passenger_rate = TL._member_measurements(slot, with_readings(no_fll, (0, 0.0), (40, 40.0Hz)),
        T, chip_length, wavelength)
    @test passenger_rate[5:6] == (true, true)
    @test passenger_rate[2] ≈ 1.0
    # With no reading of any signal, the member is withheld.
    @test TL._member_measurements(slot, with_readings(empty_driver, (0, 0.0), (0, 0.0Hz)), T,
        chip_length, wavelength)[5:6] == (false, false)
end

# The pipeline receiver on pilot-driven satellites, their C/A data components the
# passengers.
function PairedPipelineReceiver(; combine_signals, num_sats = typemax(Int), cn0_dbhz = 45.0, lead = 8.0)
    decoders, states = fixture_decoders(GPSL1CA())
    fix = calc_pvt(SignalGroup(GPSL1CA(), states); approximate_year = 2021)
    t0 = next_subframe1_start(maximum(calc_corrected_time, states)) - lead
    truth = SimTruth(; position = SVector(fix.position.x, fix.position.y, fix.position.z), t0)
    pilot = TestL1Pilot()
    estimator = VectorPLLAndDLL((pilot, GPSL1CA()); combine_signals, approximate_year = 2021,
        enable_ionospheric_correction = false, enable_tropospheric_correction = false)
    sats = [SimSat(pilot, decoder, estimator, truth, t0; cn0_dbhz,
        stream = LNAVStream(decoder.data)) for decoder in decoders[1:min(num_sats, end)]]
    for sat in sats
        sat.next_end_sample = next_block_end(sat, 0)
    end
    receiver = PipelineReceiver(sats, estimator, truth, 0, zeros(Int, length(sats)), Ref(false), Ref(0))
    # The data components' previous prompts, by satellite.
    receiver, Dict(sat.decoder.prn => complex(0.0) for sat in sats)
end

# One record of a pilot-driven satellite: the data component's, carrying the bits,
# folded as the passenger, then the pilot's, without them, stepped; each with its own
# noise and previous prompt, on the one replica.
function paired_record!(receiver::PipelineReceiver, sat::SimSat, start_sample, data_prompts)
    sample_index = sat.next_end_sample
    num_samples = sample_index - start_sample
    words = sim_words(sat, NO_LANDING_SAMPLE)
    t_end = sim_time(receiver, sample_index)
    data_correlator, data_prompt, = simulate_correlator(sat, receiver.truth, t_end, num_samples, sample_index, words)
    stream = sat.stream
    sat.stream = nothing
    correlator, prompt, code_error, f_true, carrier, code =
        simulate_correlator(sat, receiver.truth, t_end, num_samples, sample_index, words)
    sat.stream = stream
    dt = num_samples / ustrip(Hz, SIM_FS)
    code_phase = mod((sat.replica_time + dt * (1 + code / sat.code_frequency)) * sat.code_frequency,
        get_code_length(sat.signal))
    prn = sat.decoder.prn
    data_record = LoopRecord(GPSL1CA(), data_correlator, data_prompts[prn], num_samples,
        sample_index, sample_index, 1, SIM_FS; prn, code_phase)
    state = fold_passenger_record(receiver.estimator, sat.state, data_record, words;
        driver_signal = sat.signal, differential_group_delay_chips = 0.0)
    record = LoopRecord(sat.signal, correlator, sat.previous_prompt, num_samples,
        sample_index, sample_index, 1, SIM_FS; prn, code_phase)
    sat.state, new_carrier, new_code = step_loop(receiver.estimator, state, record, words, NO_LANDING_SAMPLE)
    sat.previous_prompt = prompt
    data_prompts[prn] = data_prompt
    advance_replica!(sat, f_true, carrier, code, dt, new_carrier, new_code,
        start_sample, sample_index, NO_LANDING_SAMPLE)
    sat.next_end_sample = next_block_end(sat, sample_index)
    code_error
end

function run_paired_pipeline(; combine_signals, duration = 36.0, kwargs...)
    rx, data_prompts = PairedPipelineReceiver(; combine_signals, kwargs...)
    results, _, diverged = run_pipeline!(rx, duration;
        record! = (receiver, sat, start) -> paired_record!(receiver, sat, start, data_prompts),
        driver_signal = first(rx.sats).signal)
    rx, results, diverged
end

@testset "A dataless pilot drives, its data passenger is decoded, $(combine ? "" : "not ")combined" for combine in (false, true)
    rx, results, diverged = run_paired_pipeline(; combine_signals = combine)
    @test !diverged
    @test takes_passenger_records(rx.estimator)
    # The bit clock and decoder ran on the passenger's records.
    for slot in pipeline_slots(rx)
        @test slot.bit_buffer.found
        @test TL.is_decoding_completed_for_positioning(slot.running_decoder)
        @test slot.in_lock && slot.pvt_ready
    end
    seed = findfirst(r -> r.status.enabled, results)
    @test seed !== nothing
    @test 26.0 < results[seed].time - rx.truth.t0 < 27.0
    @test position_error(rx, results[seed]) < 10.0
    # The solution reports the satellites by the pilot, the signal the filter ranges on.
    @test all(key -> first(key) == :TestL1Pilot, keys(results[end].pvt.sats))
    tail = results[seed+50:end]
    @test all(r -> r.status.running && r.status.num_members == length(rx.sats), tail)
    @test all(r -> length(r.measured) == length(rx.sats), tail)
    @test pipeline_errors(rx, tail) < 3.0
    @test pipeline_code_errors(tail) < 0.05
    # Combining, both signals' readings reach the filter, each counted in its cycle.
    slots = pipeline_slots(rx)
    @test all(slot -> first(only(slot.estimator_state.passenger_readings).code) > 0, slots) ==
          combine
    @test all(slot -> first(only(slot.estimator_state.passenger_readings).carrier) > 0, slots) ==
          combine
end

@testset "Fusing the data passenger shrinks the filter's measurement noise" begin
    # Two signals of equal power at the same C/N₀: the fused measurements carry about
    # half the driver's variance, and so the filter's position uncertainty shrinks.
    alone, alone_results, = run_paired_pipeline(; combine_signals = false)
    combined, combined_results, = run_paired_pipeline(; combine_signals = true)
    alone_std = ustrip(u"m", navigation_status(alone.estimator).position_std)
    combined_std = ustrip(u"m", navigation_status(combined.estimator).position_std)
    @test combined_std < 0.85 * alone_std
    members = combined.vt.buffers.members
    @test !isempty(members)
    alone_members = Dict(m.prn => m for m in alone.vt.buffers.members)
    paired = filter(m -> haskey(alone_members, m.prn), members)
    @test !isempty(paired)
    @test all(m -> 0.35 < m.code_variance / alone_members[m.prn].code_variance < 0.65, paired)
    @test all(m -> 0.35 < m.rate_variance / alone_members[m.prn].rate_variance < 0.65, paired)
end

@testset "The engine reads the host's C/N₀ where the records carry it" begin
    taps = (0.5, 1.0, 0.4)
    v = VectorPLLAndDLL((GPSL5Q(), GPSL5I()); combine_signals = true, cycle_time = 10ms)
    group = only(v.navigation.groups)
    state = init_estimator_state(v, GPSL5Q(), 0.0Hz, 0.0Hz)
    driver = group_record(GPSL5Q(), cis(0.0), taps, 5000; cn0 = 38.5dBHz)
    @test driver.cn0 == 38.5
    state, = step_loop(v, state, driver, _NO_WORD, NO_LANDING_SAMPLE)
    slot = group.slots[state.slot]
    @test slot.host_cn0_dbhz == 38.5
    @test TL._cn0_dbhz(slot.host_cn0_dbhz, NaN, slot.cn0_estimator, 1ms) == 38.5
    # Capped as the engine's own estimate is.
    @test TL._cn0_dbhz(120.0, NaN, slot.cn0_estimator, 1ms) == TL.MAX_CN0_DBHZ
    state = fold_passenger_record(v, state, group_record(GPSL5I(), cis(0.1), taps, 10000; cn0 = 36.0),
        _NO_WORD; driver_signal = GPSL5Q(), differential_group_delay_chips = 0.0)
    @test TL._passenger_cn0_dbhz(only(slot.passengers)) == 36.0
    # Without it, the engine's own estimate.
    state, = step_loop(v, state, group_record(GPSL5Q(), cis(0.0), taps, 10000), _NO_WORD, NO_LANDING_SAMPLE)
    @test isnan(slot.host_cn0_dbhz)
    # The scalar loops do not read it.
    scalar = ConventionalAssistedPLLAndDLL()
    scalar_state = init_estimator_state(scalar, GPSL5Q(), 0.0Hz, 0.0Hz)
    @test step_loop(scalar, scalar_state, driver, _NO_WORD, NO_LANDING_SAMPLE) ==
          step_loop(scalar, scalar_state, group_record(GPSL5Q(), cis(0.0), taps, 5000), _NO_WORD,
              NO_LANDING_SAMPLE)
end

@testset "A change of record length restarts the engine's own C/N₀" begin
    # Prompts of two lengths carry two noise scales, which one integration time cannot
    # turn into a C/N₀: the estimate restarts, holding the old one while it refills.
    prompt(k) = cis(0.1k) * (1 + 0.1 * (-1)^k)
    estimator = MomentsCN0Estimator(10)
    for k = 1:10
        estimator = update(estimator, prompt(k))
    end
    before = TL._capped_cn0_dbhz(estimator, 1ms)
    held, same = TL._restart_cn0(NaN, estimator, 1.0ms, 1.1ms)
    @test isnan(held) && same === estimator
    held, fresh = TL._restart_cn0(NaN, estimator, 1.0ms, 20.0ms)
    @test held == before && length(fresh) == 0
    fresh = update(fresh, prompt(1))
    @test TL._cn0_dbhz(NaN, held, fresh, 20ms) == before
    for k = 2:10
        fresh = update(fresh, prompt(k))
    end
    @test TL._cn0_dbhz(NaN, held, fresh, 20ms) == TL._capped_cn0_dbhz(fresh, 20ms)
    # The host's estimate wins either way, and an empty estimator holds nothing.
    @test TL._cn0_dbhz(35.0, held, fresh, 20ms) == 35.0
    @test isnan(first(TL._restart_cn0(NaN, MomentsCN0Estimator(10), 1.0ms, 4.0ms)))
end
