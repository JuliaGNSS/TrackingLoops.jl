using TrackingLoopFilters: filter_loop, ThirdOrderBilinearLF
const TL = TrackingLoops

# The satellite's own loop under the vector loop, without the navigation engine around
# it: what `step_loop` runs after the engine's part.
step_satellite(estimator::VectorPLLAndDLL, args...) = TL._step_satellite(estimator, args...)
step_satellite(estimator, args...) = step_loop(estimator, args...)

# One record at `sample_index` with the given raw accumulators, normalised the
# way `apply_record` does before the loop filters see it (the post-correlation
# filter is the identity for the first prompt).
function vector_record(signal, accumulators, num_samples, fs; previous_prompt = complex(0.0, 0.0), sample_index = num_samples)
    correlator = update_accumulator(get_default_correlator(signal), SVector(accumulators...))
    output = CorrelatorOutput(correlator, num_samples, sample_index)
    normalized = normalize(correlator, num_samples, get_code_amplitude(signal))
    LoopRecord(signal, normalized, previous_prompt, output, 1, fs)
end

@testset "Vector PLL and DLL state" begin
    estimator = VectorPLLAndDLL(GPSL1CA())
    @test estimator.inner isa ConventionalPLLAndDLL{ThirdOrderAssistedBilinearLF}
    state = init_estimator_state(estimator, GPSL1CA(), 500.0Hz, 100.0Hz)
    @test state isa SatVectorPLLAndDLL
    @test state.inner == init_estimator_state(ConventionalAssistedPLLAndDLL(), GPSL1CA(), 500.0Hz, 100.0Hz)
    @test state.vt_on == false
    @test state.code_discr_acc == (0, 0.0)
    @test state.carrier_phase_acc == TL.CarrierPhaseAccumulator()
    @test all(isnan, state.previous_replica_half)
    @test isnan(first(state.previous_phase_error))
    @test state.code_freq_update == 0.0Hz
    @test state.carrier_freq_update == 0.0Hz
    @test state.code_freq_update_history == (0.0Hz, 0.0Hz, 0.0Hz)
    @test state.code_update_landing_lead == 0.0s
    @test estimator_state_type(estimator, GPSL1CA()) === typeof(state)
    @test TL._mean_code_discriminator(state) === nothing
    @test TL._mean_carrier_doppler(state) === nothing
end

@testset "Vector PLL and DLL seeding" begin
    # Auto bandwidths resolve from the driver signal, sized exactly like the
    # inner (scalar fallback) loop.
    state = init_estimator_state(VectorPLLAndDLL(GPSL1CA()), GPSL1CA(), 100.0Hz, 0.0Hz)
    @test state.inner.bandwidths.wide_carrier == 50.0Hz
    @test state.inner.bandwidths.narrow_carrier == 18.0Hz
    @test state.inner.bandwidths.fll_assist == 5.0Hz
    @test state.inner.bandwidths.code == 1.0Hz
    @test state.inner.bandwidths.vector_fll == 1.0Hz ==
          default_vector_fll_loop_filter_bandwidth(GPSL1CA())
    @test state.inner.init_carrier_doppler == 100.0Hz
    @test carrier_loop_stage(state) == FLL_ASSISTED_PLL
    explicit = ConventionalAssistedPLLAndDLL(; narrow_carrier_loop_filter_bandwidth = 12.0Hz, code_loop_filter_bandwidth = 1.0Hz)
    explicit_state = init_estimator_state(VectorPLLAndDLL(GPSL1CA(); inner = explicit), GPSL1CA(), 100.0Hz, 0.0Hz)
    @test explicit_state.inner.bandwidths.narrow_carrier == 12.0Hz
    @test explicit_state.inner.bandwidths.code == 1.0Hz
    bumped = VectorPLLAndDLL(GPSL1CA(); inner = ConventionalPLLAndDLL(explicit; narrow_carrier_loop_filter_bandwidth = 15.0Hz))
    @test bumped.inner.narrow_carrier_loop_filter_bandwidth == 15.0Hz
    @test bumped.inner.code_loop_filter_bandwidth == 1.0Hz
    referenced = init_estimator_state(VectorPLLAndDLL(GalileoE1B(); inner = NCOReferencedPLLAndDLL()), GalileoE1B(), 0.0Hz, 0.0Hz)
    @test referenced.inner isa SatNCOReferencedPLLAndDLL
    @test referenced.inner.bandwidths.narrow_carrier ==
          default_narrow_carrier_loop_filter_bandwidth(GalileoE1B())
    # The vector FLL's bandwidth is the vector estimator's, resolved into the inner
    # state, for either inner loop; the inner loop's own bandwidths stay.
    for inner in (explicit, NCOReferencedPLLAndDLL())
        estimator = VectorPLLAndDLL(GPSL1CA(); inner, vector_fll_loop_filter_bandwidth = 0.5Hz)
        vector_state = init_estimator_state(estimator, GPSL1CA(), 100.0Hz, 0.0Hz)
        inner_state = init_estimator_state(inner, GPSL1CA(), 100.0Hz, 0.0Hz)
        @test vector_state.inner.bandwidths.vector_fll == 0.5Hz
        b = inner_state.bandwidths
        @test vector_state.inner.bandwidths ==
              LoopBandwidths(b.wide_carrier, b.narrow_carrier, b.fll_assist, b.code, 0.5Hz)
    end
end

@testset "The vector FLL runs at its own bandwidth, capped by the cycle" begin
    # One record in the vector loop with a carrier correction alone: the FLL branch at
    # the configured bandwidth, or at `0.1 / cycle_time` on a longer cycle.
    signal = GPSL1CA()
    fs = 5e6Hz
    record = vector_record(signal, (1000.0 + 10im, 2000.0 + 20im, 750.0 + 10im), 5000, fs)
    words = FixedNCOWord(100.0, 0.0)
    pll_discriminator = pll_disc(signal, record.filtered_correlator)
    for (cycle_time, vector_fll, applied) in
        ((100.0ms, 1.0Hz, 1.0Hz), (100.0ms, 0.4Hz, 0.4Hz), (100.0ms, 3.0Hz, 1.0Hz), (1.0s, 1.0Hz, 0.1Hz))
        estimator = VectorPLLAndDLL(signal; cycle_time, vector_fll_loop_filter_bandwidth = vector_fll)
        state = init_estimator_state(estimator, signal, 100.0Hz, 0.0Hz)
        state = TL._set_vector_corrections(TL._enable_vector_tracking(state), 0.0Hz, 5.0Hz)
        _, carrier_doppler, _ = step_satellite(estimator, state, record, words, NO_LANDING_SAMPLE)
        expected, _ = filter_loop(ThirdOrderAssistedBilinearLF(), (pll_discriminator, 5.0Hz), 1.0ms, (18.0Hz, applied))
        @test carrier_doppler ≈ 100.0Hz + expected
        @test effective_vector_fll_loop_filter_bandwidth(vector_fll, cycle_time) ≈ applied
    end
end

@testset "A vector loop needs an FLL-assisted inner loop" begin
    @test_throws ArgumentError VectorPLLAndDLL(GPSL1CA(); inner = ConventionalPLLAndDLL())
    @test_throws ArgumentError VectorPLLAndDLL(GPSL1CA(); inner = ConventionalPLLAndDLL(ThirdOrderBilinearLF))
    @test_throws ArgumentError VectorPLLAndDLL(GPSL1CA(); inner = VectorPLLAndDLL(GPSL1CA()))
    @test VectorPLLAndDLL(GPSL1CA(); inner = ConventionalAssistedPLLAndDLL()) isa VectorPLLAndDLL
    @test VectorPLLAndDLL(GPSL1CA(); inner = NCOReferencedPLLAndDLL()) isa VectorPLLAndDLL
end

@testset "Out of the vector loop it is the inner loop, with $(nameof(typeof(inner)))" for inner in (
    ConventionalAssistedPLLAndDLL(),
    NCOReferencedPLLAndDLL(),
)
    # Under an NCO delay too: the delayed simulation of `estimators.jl`.
    for d in (0, 3)
        @test simulate_delayed_loop(VectorPLLAndDLL(GPSL1CA(); inner), d; steps = 300) ==
              simulate_delayed_loop(inner, d; steps = 300)
    end
    # And record by record on one fixed word: nothing is accumulated and the
    # corrections are never written.
    signal = GPSL1CA()
    fs = 5e6Hz
    vector = VectorPLLAndDLL(GPSL1CA(); inner)
    vector_state = TL._set_vector_corrections(init_estimator_state(vector, signal, 100.0Hz, 0.1Hz), 0.5Hz, 2.0Hz)
    inner_state = init_estimator_state(inner, signal, 100.0Hz, 0.1Hz)
    record = vector_record(signal, (1000.0 + 10im, 2000.0 + 20im, 750.0 + 10im), 5000, fs; previous_prompt = cis(0.1))
    words = FixedNCOWord(100.0, 0.1)
    new_vector_state, vector_carrier, vector_code = step_satellite(vector, vector_state, record, words, NO_LANDING_SAMPLE)
    new_inner_state, inner_carrier, inner_code = step_satellite(inner, inner_state, record, words, NO_LANDING_SAMPLE)
    @test vector_carrier == inner_carrier
    @test vector_code == inner_code
    @test new_vector_state.inner == new_inner_state
    @test new_vector_state.code_discr_acc == (0, 0.0)
    @test new_vector_state.carrier_phase_acc == TL.CarrierPhaseAccumulator()
    # The replica's last half record and its prompt's phase error are kept, for the
    # span a later vector loop opens with: half of the 1 ms record on the 100 Hz word.
    @test new_vector_state.previous_replica_half == (100.0 * 0.0005, 0.0005s)
    @test new_vector_state.previous_phase_error == (pll_disc(signal, record.filtered_correlator), Int8(0))
    @test new_vector_state.code_freq_update == 0.5Hz
    @test new_vector_state.carrier_freq_update == 2.0Hz
    @test vector_carrier != 100.0Hz
end

@testset "Vector loop closure applies the NCO corrections, with $(nameof(typeof(inner)))" for inner in (
    ConventionalAssistedPLLAndDLL(),
    NCOReferencedPLLAndDLL(),
)
    sampling_frequency = 5e6Hz
    gpsl1 = GPSL1CA()
    carrier_doppler = 100.0Hz
    init_code_doppler = carrier_doppler * get_code_center_frequency_ratio(gpsl1)
    num_samples = 5000
    accumulators = (1000.0 + 10im, 2000.0 + 20im, 750.0 + 10im)
    nav_carrier_freq_update = 5.0Hz
    nav_code_freq_update = -0.25Hz

    estimator = VectorPLLAndDLL(GPSL1CA(); inner)
    state = init_estimator_state(estimator, gpsl1, carrier_doppler, init_code_doppler)
    state = TL._enable_vector_tracking(state)
    state = TL._set_vector_corrections(state, nav_code_freq_update, nav_carrier_freq_update)
    record = vector_record(gpsl1, accumulators, num_samples, sampling_frequency)
    words = FixedNCOWord(ustrip(Hz, carrier_doppler), ustrip(Hz, init_code_doppler))
    state, new_carrier_doppler, new_code_doppler = step_satellite(estimator, state, record, words, NO_LANDING_SAMPLE)

    normalized_correlator = update_accumulator(
        get_default_correlator(gpsl1),
        SVector(accumulators...) ./ num_samples,
    )
    integration_time = num_samples / sampling_frequency
    pll_discriminator = pll_disc(gpsl1, normalized_correlator)
    dll_discriminator = dll_disc(gpsl1, normalized_correlator, init_code_doppler, sampling_frequency)
    # The FLL branch is driven by the vector loop's carrier update, at its own 1 Hz
    # beside the 18 Hz PLL.
    expected_carrier_freq_update, _ = filter_loop(
        ThirdOrderAssistedBilinearLF(),
        (pll_discriminator, nav_carrier_freq_update),
        integration_time,
        (18.0Hz, 1.0Hz),
    )
    @test new_carrier_doppler == carrier_doppler + expected_carrier_freq_update
    # The code Doppler follows the vector loop's update directly, plus the
    # carrier filter's own aiding; the code loop filter is frozen.
    @test new_code_doppler ==
          init_code_doppler +
          nav_code_freq_update +
          expected_carrier_freq_update * get_code_center_frequency_ratio(gpsl1)
    @test state.inner.code_loop_filter == SecondOrderBilinearLF()
    # The DLL discriminator is accumulated. Without a previous prompt there is no
    # frequency measurement, so no carrier span opens: not a span of zero rotation.
    @test state.code_discr_acc == (1, dll_discriminator)
    @test state.carrier_phase_acc == TL.CarrierPhaseAccumulator()
    @test TL._mean_code_discriminator(state) == dll_discriminator
    @test TL._mean_carrier_doppler(state) === nothing
    @test !TL.has_code_and_carrier_measurements(state)
    # The replica's second half record is kept for the next span.
    @test state.previous_replica_half == (ustrip(Hz, carrier_doppler) * 0.0005, 0.0005s)
    # The corrections survive the step untouched.
    @test state.code_freq_update == nav_code_freq_update
    @test state.carrier_freq_update == nav_carrier_freq_update

    state = TL._reset_code_and_carrier_accumulators(state)
    @test state.code_discr_acc == (0, 0.0)
    @test state.carrier_phase_acc == TL.CarrierPhaseAccumulator()
    @test state.previous_replica_half == (ustrip(Hz, carrier_doppler) * 0.0005, 0.0005s)
    @test TL._mean_code_discriminator(state) === nothing
    @test TL._mean_carrier_doppler(state) === nothing
    @test state.vt_on
    @test state.code_freq_update == nav_code_freq_update
end

@testset "The vector loop accumulates the incoming carrier phase between prompts" begin
    signal = GPSL1CA()
    fs = 5e6Hz
    estimator = VectorPLLAndDLL(GPSL1CA())
    state = TL._enable_vector_tracking(init_estimator_state(estimator, signal, 100.0Hz, 0.1Hz))
    previous = cis(0.2)
    # Prompts 0.02, 0.05 and 0.03 cycles ahead of the replica.
    prompts = cis.(2π .* (0.02, 0.05, 0.03))
    record(p, previous, k; kwargs...) = LoopRecord(signal, loop_epl(0.5p, p, 0.5p), previous, 5000, 5000k, 5000k, 1, fs; kwargs...)
    records = [record(prompts[k], k == 1 ? previous : prompts[k - 1], k) for k = 1:3]
    # The first record has no replica half record before it, so it only leaves its own,
    # and its prompt's phase error.
    state, = step_satellite(estimator, state, records[1], FixedNCOWord(100.0, 0.1), NO_LANDING_SAMPLE)
    @test state.carrier_phase_acc == TL.CarrierPhaseAccumulator()
    @test state.previous_replica_half == (0.05, 0.0005s)
    @test state.previous_phase_error[1] ≈ 0.02
    @test state.previous_phase_error[2] == 0
    # Each later one adds the span between the two prompts' centres: the replica's
    # advance over it, half a record on each word, plus the change of the phase error.
    state, = step_satellite(estimator, state, records[2], FixedNCOWord(120.0, 0.1), NO_LANDING_SAMPLE)
    state, = step_satellite(estimator, state, records[3], FixedNCOWord(140.0, 0.1), NO_LANDING_SAMPLE)
    acc = state.carrier_phase_acc
    @test acc.count == 2
    @test acc.cycles ≈ (100.0 + 120.0) * 0.0005 + (120.0 + 140.0) * 0.0005 + (0.03 - 0.02)
    @test acc.duration ≈ 0.002s
    # The spans run from the first record's centre (0.5 ms) to the third's (2.5 ms).
    doppler, time = TL._mean_carrier_doppler(state)
    @test doppler ≈ 120.0Hz + 0.01 / 0.002 * Hz
    @test time ≈ 0.0015s
    @test TL.accumulated_carrier_doppler(state, 0.003) == (ustrip(Hz, doppler), 0.003 - ustrip(s, time))
    @test state.previous_replica_half == (140.0 * 0.0005, 0.0005s)
    # A record without a previous prompt (a changed carrier phase polarity) does not
    # follow on from the one before it, so its span is left out; the next one starts
    # from its half record and its phase error.
    no_prompt = record(prompts[3], 0.0, 4)
    skipped, = step_satellite(estimator, state, no_prompt, FixedNCOWord(160.0, 0.1), NO_LANDING_SAMPLE)
    @test skipped.carrier_phase_acc == acc
    @test skipped.previous_replica_half == (160.0 * 0.0005, 0.0005s)
    @test skipped.previous_phase_error[1] ≈ 0.03
    # Nor does one whose phase error was read with another polarity, even with a
    # previous prompt: the phase errors refer to phases half a cycle apart.
    other_polarity = record(prompts[3], prompts[3], 5; carrier_phase_polarity = 1)
    skipped, = step_satellite(estimator, skipped, other_polarity, FixedNCOWord(160.0, 0.1), NO_LANDING_SAMPLE)
    @test skipped.carrier_phase_acc == acc
    @test skipped.previous_phase_error[1] ≈ 0.03
    @test skipped.previous_phase_error[2] == 1
end

# Step a vector-loop satellite through records whose prompts lie `phases[k]` cycles
# ahead of a replica on the fixed `word` (Hz), 1 ms records at 4 MHz, read with
# `polarity`. Returns the carrier phase it accumulated (cycles) and the spans.
function accumulate_prompt_phases(phases; word = 250.0, polarity = 0)
    signal = GPSL1CA()
    fs = 4e6Hz
    estimator = VectorPLLAndDLL(signal)
    state = TL._enable_vector_tracking(init_estimator_state(estimator, signal, word * Hz, 0.0Hz))
    previous = complex(0.0)
    for (k, phase) in enumerate(phases)
        p = 0.8cis(2π * phase)
        output = CorrelatorOutput(loop_epl(0.5p, p, 0.5p), 4000, 4000k)
        record = LoopRecord(signal, output.correlator, previous, output, 1, fs; carrier_phase_polarity = polarity)
        state, = step_satellite(estimator, state, record, FixedNCOWord(word, 0.0), NO_LANDING_SAMPLE)
        previous = p
    end
    state.carrier_phase_acc
end

@testset "Prompt noise across the discriminator's range leaves no cycle behind, polarity $polarity" for (polarity, jump) in (
    (0, 0.27),   # a data signal, Costas: past the FLL's quarter cycle
    (1, 0.55),   # a pilot, four-quadrant: past the FLL's half cycle
)
    # The incoming carrier gains 0.004 cycles on the replica per record, and the noise
    # turns one prompt `jump` cycles away from the one before it and back: more than
    # the FLL's rotation reads before it wraps, by half a cycle for the Costas FLL and
    # a whole one for the four-quadrant FLL. Every phase error itself stays within the
    # PLL discriminator's range.
    true_phases = 0.004 .* (0:9)
    noise = [0.0, 0.01, -0.12, -0.12 + jump, 0.02, -0.01, 0.0, 0.03, -0.02, 0.01]
    if polarity != 0
        noise[3] = -0.28
        noise[4] = -0.28 + jump
    end
    phases = true_phases .+ noise
    @test all(abs.(phases) .< (polarity == 0 ? 0.25 : 0.5))
    @test abs(phases[4] - phases[3]) > (polarity == 0 ? 0.25 : 0.5)
    acc = accumulate_prompt_phases(phases; polarity)
    @test acc.count == 9
    @test acc.duration ≈ 0.009s
    # The replica's advance over the spans plus the incoming signal's advance against
    # it, at the noise of the two end prompts only.
    @test acc.cycles ≈ 250.0 * 0.009 + phases[end] - phases[1] atol = 1e-9
end

@testset "The vector loop accumulates the carrier phase in Hz from a MHz sampling frequency" begin
    # A MHz sampling frequency must not reach the accumulator in its own unit. The
    # FLL reading it once took did, and the accumulator refused it as soon as the
    # satellite was in the vector loop; the replica's half records are formed from
    # the sampling frequency too.
    signal = GPSL1CA()
    estimator = VectorPLLAndDLL(GPSL1CA())
    function accumulated(fs, num_samples)
        state = TL._enable_vector_tracking(init_estimator_state(estimator, signal, 100.0Hz, 0.1Hz))
        for k = 1:2
            record = vector_record(signal, (1000.0 + 10im, 2000.0 + 400im, 750.0 + 10im), 5000, fs;
                previous_prompt = cis(0.2), sample_index = num_samples * k)
            state, = step_satellite(estimator, state, record, FixedNCOWord(100.0, 0.1), NO_LANDING_SAMPLE)
        end
        state.carrier_phase_acc
    end
    acc = accumulated(5.0u"MHz", 5000)
    @test acc.count == 1
    # Two half records of 100 Hz; the prompts are alike.
    @test acc.cycles ≈ 0.1
    reference = accumulated(5e6Hz, 5000)
    @test acc.cycles ≈ reference.cycles
    @test acc.duration ≈ reference.duration
    @test acc.time_moment ≈ reference.time_moment
end

@testset "The NCO-referenced vector loop keeps its landing prediction" begin
    # In the vector loop, with the delay the prediction has something to do,
    # and without it the NCO-referenced loop is the conventional one.
    signal = GPSL1CA()
    fs = 4e6Hz
    function run(inner, landing_offset)
        estimator = VectorPLLAndDLL(GPSL1CA(); inner)
        state = init_estimator_state(estimator, signal, 100.0Hz, 0.1Hz)
        state = TL._set_vector_corrections(TL._enable_vector_tracking(state), 0.1Hz, 1.0Hz)
        timeline = NCOTimeline()
        reset_timeline!(timeline, 100.0, 0.1)
        previous = complex(0.0, 0.0)
        carriers = typeof(1.0Hz)[]
        # The replica's advance over each record, from the words it ran on.
        record_cycles = Float64[]
        for k = 1:50
            p = cis(0.3 + 0.01k)
            output = CorrelatorOutput(loop_epl(0.5p, p, 0.5p), 4000, 4000k)
            record = LoopRecord(signal, output.correlator, previous, output, 1, fs)
            landing = landing_offset == 0 ? NO_LANDING_SAMPLE : Int64(4000k + landing_offset)
            push!(record_cycles, first(mean_nco_word(timeline, 4000k - 4000, 4000k)) * 0.001)
            state, carrier, code = step_satellite(estimator, state, record, timeline, landing)
            schedule_word!(timeline, 4000k + landing_offset, ustrip(Hz, carrier), ustrip(Hz, code))
            promote_words!(timeline, 4000k - 4000)
            push!(carriers, carrier)
            previous = p
        end
        # The spans run from the first record's centre to the last one's; every word
        # lands on a record boundary, so each half record is half the record's advance.
        replica = sum(record_cycles) - (record_cycles[1] + record_cycles[end]) / 2
        carriers, state, state.carrier_phase_acc.cycles - replica
    end
    referenced, referenced_state = run(NCOReferencedPLLAndDLL(), 0)
    conventional, conventional_state = run(ConventionalAssistedPLLAndDLL(), 0)
    @test referenced == conventional
    @test referenced_state.code_discr_acc == conventional_state.code_discr_acc
    @test referenced_state.carrier_phase_acc == conventional_state.carrier_phase_acc
    predicted, predicted_state, predicted_rotation = run(NCOReferencedPLLAndDLL(), 12000)
    control, control_state, control_rotation = run(NCOReferencedPLLAndDLL(; predict_landing = false), 12000)
    @test predicted != control
    # The accumulated phase errors are the prompts' raw ones, before any prediction
    # to the landing sample, so the prediction does not reach them; the replica's
    # advance is the words' that really ran.
    @test predicted_rotation ≈ control_rotation atol = 1e-12
    @test predicted_rotation != 0.0
    # The first record has no previous prompt.
    @test predicted_state.carrier_phase_acc.count == 49
end

@testset "Vector tracking state management" begin
    estimator = VectorPLLAndDLL(GPSL1CA())
    state = init_estimator_state(estimator, GPSL1CA(), 100.0Hz, 0.1Hz)
    joined = TL._enable_vector_tracking(state)
    @test joined.vt_on
    # Re-enabling a member changes nothing, so the members can be enabled every
    # cycle.
    steered = SatVectorPLLAndDLL(
        TL._set_vector_corrections(joined, 1.0Hz, 10.0Hz, 0.02s);
        code_discr_acc = (3, 0.3),
        carrier_phase_acc = TL.CarrierPhaseAccumulator(3, 0.3, 0.003s, 0.0s^2),
        previous_replica_half = (0.1, 0.0005s),
        previous_phase_error = (0.02, Int8(1)),
    )
    @test TL._enable_vector_tracking(steered) === steered
    # The corrections take over one another.
    @test steered.code_freq_update == 1.0Hz
    @test steered.carrier_freq_update == 10.0Hz
    @test steered.code_freq_update_history == (1.0Hz, 0.0Hz, 0.0Hz)
    @test steered.code_update_landing_lead == 0.02s
    next = TL._set_vector_corrections(steered, 2.0Hz, 20.0Hz)
    @test next.code_freq_update_history == (2.0Hz, 1.0Hz, 0.0Hz)
    @test TL._set_vector_corrections(next, 3.0Hz, 0.0Hz).code_freq_update_history ==
          (3.0Hz, 2.0Hz, 1.0Hz)
    @test next.code_freq_update == 2.0Hz
    @test next.code_update_landing_lead == 0.0s
    @test next.code_discr_acc == (3, 0.3)
    # Disabling zeroes everything the vector loop owned, so a re-enabled
    # satellite never runs on stale values.
    released = TL._disable_vector_tracking(next)
    @test !released.vt_on
    @test released.inner == next.inner
    @test released.code_discr_acc == (0, 0.0)
    @test released.carrier_phase_acc == TL.CarrierPhaseAccumulator()
    @test released.code_freq_update == 0.0Hz
    @test released.carrier_freq_update == 0.0Hz
    @test released.code_freq_update_history == (0.0Hz, 0.0Hz, 0.0Hz)
    @test released.code_update_landing_lead == 0.0s
    # The replica's last half record and its last prompt's phase error are the
    # replica's: they survive the hand-over both ways.
    @test released.previous_replica_half == (0.1, 0.0005s)
    @test TL._enable_vector_tracking(released).previous_replica_half == (0.1, 0.0005s)
    @test released.previous_phase_error == (0.02, Int8(1))
    @test TL._enable_vector_tracking(released).previous_phase_error == (0.02, Int8(1))
    @test TL._enable_vector_tracking(released).code_freq_update == 0.0Hz
end

@testset "Resetting keeps the vector flag and stops applying the corrections" begin
    estimator = VectorPLLAndDLL(GPSL1CA(); inner = ConventionalAssistedPLLAndDLL(; narrow_carrier_loop_filter_bandwidth = 12.0Hz))
    state = init_estimator_state(estimator, GPSL1CA(), 100.0Hz, 0.1Hz)
    record = vector_record(GPSL1CA(), (1000.0 + 10im, 2000.0 + 200im, 750.0 + 10im), 5000, 5e6Hz; previous_prompt = cis(0.1))
    state = TL._set_vector_corrections(TL._enable_vector_tracking(state), 0.5Hz, 4.0Hz)
    state, = step_satellite(estimator, state, record, FixedNCOWord(100.0, 0.1), NO_LANDING_SAMPLE)
    state = TL._set_vector_corrections(state, 0.6Hz, 4.0Hz, 0.01s)
    @test state.inner.carrier_loop_filter != ThirdOrderAssistedBilinearLF()
    reset = reset_estimator_state(estimator, state, 150.0Hz, 0.2Hz)
    @test reset isa typeof(state)
    @test reset.vt_on
    @test reset.code_discr_acc == (0, 0.0)
    @test reset.carrier_phase_acc == TL.CarrierPhaseAccumulator()
    @test reset.previous_replica_half == state.previous_replica_half
    @test reset.previous_phase_error == state.previous_phase_error
    # The re-seeded Dopplers carry the last correction, so it is no longer applied…
    @test reset.code_freq_update == 0.0Hz
    @test reset.carrier_freq_update == 0.0Hz
    # …but the replica is still steered by it, so the next code measurement is moved to
    # the epoch exactly as without the reset.
    @test reset.code_freq_update_history == (0.6Hz, 0.5Hz, 0.0Hz)
    @test reset.code_update_landing_lead == 0.01s
    @test TrackingLoops.code_phase_advance(reset, 0.1) == TrackingLoops.code_phase_advance(state, 0.1)
    @test TrackingLoops.code_phase_advance(reset, 0.1) != 0.0
    @test reset.inner.carrier_loop_filter == ThirdOrderAssistedBilinearLF()
    @test reset.inner.bandwidths.narrow_carrier == 12.0Hz
    # In the vector loop the carrier loop stays at the narrow PLL.
    @test carrier_loop_stage(reset) == NARROW_PLL
    @test reset.inner.init_carrier_doppler == 150.0Hz
    @test reset.inner.init_code_doppler == 0.2Hz
    # With nothing applied, the code Doppler is the re-seeded one plus the carrier aiding.
    _, carrier_doppler, code_doppler =
        step_satellite(estimator, reset, record, FixedNCOWord(150.0, 0.2), NO_LANDING_SAMPLE)
    @test code_doppler ≈ 0.2Hz + (carrier_doppler - 150.0Hz) * get_code_center_frequency_ratio(GPSL1CA())
end

@testset "Releasing re-seeds the inner loop, with $(nameof(typeof(inner)))" for inner in (
    ConventionalAssistedPLLAndDLL(; narrow_carrier_loop_filter_bandwidth = 12.0Hz),
    NCOReferencedPLLAndDLL(; narrow_carrier_loop_filter_bandwidth = 12.0Hz),
)
    estimator = VectorPLLAndDLL(GPSL1CA(); inner)
    state = init_estimator_state(estimator, GPSL1CA(), 100.0Hz, 0.1Hz)
    record = vector_record(GPSL1CA(), (1000.0 + 10im, 2000.0 + 200im, 750.0 + 10im), 5000, 5e6Hz; previous_prompt = cis(0.1))
    state = TL._set_vector_corrections(TL._enable_vector_tracking(state), 0.5Hz, 4.0Hz, 0.01s)
    state, = step_satellite(estimator, state, record, FixedNCOWord(100.0, 0.1), NO_LANDING_SAMPLE)
    released = TL._release_from_vector_tracking(state, 150.0Hz, 0.2Hz)
    @test released isa typeof(state)
    @test !released.vt_on
    # The inner loop starts over from the handed-over replica, with its own
    # configuration kept: the per-satellite state carries it. Unlike a reset, the
    # release restarts the staging.
    reset_inner = reset_estimator_state(inner, state.inner, 150.0Hz, 0.2Hz)
    @test released.inner == Base.typename(typeof(reset_inner)).wrapper(
        reset_inner;
        staging = TL.CarrierLoopStaging(typeof(reset_inner.carrier_loop_filter)),
    )
    @test reset_inner.staging == state.inner.staging
    @test released.inner.init_carrier_doppler == 150.0Hz
    @test released.inner.init_code_doppler == 0.2Hz
    @test released.inner.bandwidths.narrow_carrier == 12.0Hz
    # The carrier loop pulls in afresh, FLL-assisted, after the narrow PLL it ran
    # at in the vector loop.
    @test carrier_loop_stage(state) == NARROW_PLL
    @test carrier_loop_stage(released) == FLL_ASSISTED_PLL
    # Nothing the vector loop owned survives.
    @test released.code_discr_acc == (0, 0.0)
    @test released.carrier_phase_acc == TL.CarrierPhaseAccumulator()
    @test released.previous_replica_half == state.previous_replica_half
    @test released.previous_phase_error == state.previous_phase_error
    @test released.code_freq_update == 0.0Hz
    @test released.carrier_freq_update == 0.0Hz
    @test released.code_freq_update_history == (0.0Hz, 0.0Hz, 0.0Hz)
    # Out of the vector loop it steps exactly as the re-seeded inner loop.
    @test step_satellite(estimator, released, record, FixedNCOWord(150.0, 0.2), NO_LANDING_SAMPLE)[2:3] ==
          step_satellite(inner, released.inner, record, FixedNCOWord(150.0, 0.2), NO_LANDING_SAMPLE)[2:3]
end

# ─────────────────────────────────────────────────────────────────────────────
# The pseudorange-rate observable of a satellite in the vector loop, on a carrier of
# known phase: the loop's own carrier filter tracking it record by record, and the
# accumulators read and emptied at every navigation epoch, as the engine's snapshot
# does — before the record that crosses the epoch.

using Random: Xoshiro, randn
using Statistics: mean, std

const RATE_SIGNAL = GPSL1CA()
const RATE_FS = 4e6
const RATE_WAVELENGTH = TL.SPEED_OF_LIGHT / ustrip(Hz, get_center_frequency(RATE_SIGNAL))

# Track the carrier of phase `phase(t)` (cycles) with records of `record_samples`,
# the first ending at `first_end` (samples), up to `duration` seconds, the epochs every
# `cycle` seconds. Each prompt is the mean of the carrier against the replica over its
# record, with complex noise of `noise_std` per component, read with the carrier phase
# polarity `polarity`. Returns per epoch `E` the
# mean Doppler the satellite accumulated (Hz), how long before `E` it refers to (s), the
# carrier phase it accumulated (cycles) and the spans' length (s), and the centres of the
# first and the last record whose spans the measurements cover.
function track_carrier(phase, doppler0; record_samples = 16000, first_end = 16000, cycle = 0.1,
        duration = 2.0, noise_std = 0.0, rng = Xoshiro(1), polarity = 0)
    estimator = VectorPLLAndDLL(RATE_SIGNAL)
    state = TL._enable_vector_tracking(init_estimator_state(estimator, RATE_SIGNAL, doppler0 * Hz, 0.0Hz))
    carrier = doppler0
    replica_phase = phase(0.0) # cycles, at the start of the record ahead
    previous_prompt = complex(0.0)
    epochs = NamedTuple[]
    epoch = 1
    record_end = first_end
    record_start = 0
    previous_center = NaN
    last_center = NaN
    while record_end / RATE_FS <= duration
        t0, t1 = record_start / RATE_FS, record_end / RATE_FS
        # The engine's snapshot: the accumulators up to the epoch, emptied before the
        # record that crosses it.
        if t1 > epoch * cycle
            doppler, offset = TL.accumulated_carrier_doppler(state, epoch * cycle)
            acc = state.carrier_phase_acc
            push!(epochs, (; epoch = epoch * cycle, doppler, offset, cycles = acc.cycles,
                duration = ustrip(s, acc.duration), count = acc.count, last_center = previous_center))
            state = TL._reset_code_and_carrier_accumulators(state)
            epoch += 1
        end
        n = 64
        prompt = sum(cis(2π * (phase(t0 + (i - 0.5) / n * (t1 - t0)) -
                                replica_phase - carrier * (i - 0.5) / n * (t1 - t0))) for i = 1:n) / n
        prompt += noise_std * complex(randn(rng), randn(rng))
        output = CorrelatorOutput(loop_epl(0.5prompt, prompt, 0.5prompt), record_end - record_start, record_end)
        record = LoopRecord(RATE_SIGNAL, output.correlator, previous_prompt, output, 1, RATE_FS * Hz;
            carrier_phase_polarity = polarity)
        state, new_carrier, = TL._step_satellite(estimator, state, record,
            FixedNCOWord(carrier, 0.0), NO_LANDING_SAMPLE)
        replica_phase += carrier * (t1 - t0)
        carrier = ustrip(Hz, new_carrier)
        previous_prompt = prompt
        previous_center = (t0 + t1) / 2
        record_start = record_end
        record_end += record_samples
    end
    epochs
end

@testset "The rate is the mean Doppler over the cycle, at the instant it refers to" begin
    # A line-of-sight acceleration of 5 m/s², on 4 ms records ending 1.3 ms past the
    # epochs. The Doppler the replica ends a cycle on lags the ramp, and the cycle's mean
    # is not the end-of-cycle Doppler: it is the Doppler at the mean of the spans'
    # midpoints, half a cycle and half a record before the epoch — exactly so under a
    # constant acceleration, whatever the loop's lag.
    acceleration = 5.0
    ramp = acceleration / RATE_WAVELENGTH # Hz/s
    doppler(t) = 1200.0 + ramp * t
    phase(t) = 0.3 + 1200.0 * t + ramp * t^2 / 2
    epochs = track_carrier(phase, 1200.0; first_end = 16000 + 5200, duration = 3.0)
    for e in epochs[11:end]
        # The spans run from the centre of the last record before the previous epoch to
        # that of the last one before this epoch: 2 ms + 1.3 ms… before each.
        @test e.offset ≈ 0.1 / 2 + 0.002 + 0.0027 atol = 1e-9
        @test RATE_WAVELENGTH * abs(e.doppler - doppler(e.epoch - e.offset)) < 0.001
        # Half a cycle and more away from the epoch's Doppler: what a prediction at the
        # epoch would mistake for a rate error.
        @test RATE_WAVELENGTH * (doppler(e.epoch) - e.doppler) ≈ acceleration * e.offset atol = 0.001
    end
end

@testset "The rate measurements are continuous across cycles" begin
    # Noisy prompts at 45 dB-Hz, 4.1 ms records, so the 100 ms cycles take 24 or 25 of
    # them, under a slow Doppler ramp. The cycles' spans tile the time between the first
    # and the last measured prompt, and the carrier phase they accumulated sums to the
    # incoming phase there, up to the noise of the two end prompts: no walk, however
    # many cycles.
    cn0 = 10^4.5
    record_samples = 16400
    T_coh = record_samples / RATE_FS
    ramp = 0.5 / RATE_WAVELENGTH
    phase(t) = 0.1 + 800.0 * t + ramp * t^2 / 2
    epochs = track_carrier(phase, 800.0; record_samples, first_end = record_samples, duration = 60.0,
        noise_std = sqrt(1 / (2 * cn0 * T_coh)), rng = Xoshiro(7))
    measured = epochs[11:end]
    @test all(e -> e.count in (24, 25), measured)
    @test any(e -> e.count == 24, measured) && any(e -> e.count == 25, measured)
    first_center = epochs[10].last_center
    integrated_error = Float64[]
    tiling_error = 0.0
    total_cycles = 0.0
    total_duration = 0.0
    for e in measured
        total_cycles += e.cycles
        total_duration += e.duration
        tiling_error = max(tiling_error, abs(total_duration - (e.last_center - first_center)))
        push!(integrated_error,
            RATE_WAVELENGTH * (total_cycles - (phase(e.last_center) - phase(first_center))))
    end
    @test tiling_error < 1e-9
    # Two prompt phases' worth of noise, 2.8 mm (1σ), from the first cycle to the 590th.
    σ_φ² = 1 / (2 * cn0 * T_coh) * (1 + 1 / (2 * cn0 * T_coh))
    σ_integrated = RATE_WAVELENGTH * sqrt(2σ_φ²) / 2π
    @test maximum(abs, integrated_error) < 5σ_integrated
    @test std(integrated_error[1:end÷2]) ≈ std(integrated_error[end÷2:end]) rtol = 0.3
    # Each cycle's error then has the variance the filter weights the rate with, and
    # consecutive cycles share a boundary prompt: a lag-1 autocorrelation of −1/2.
    residuals = [RATE_WAVELENGTH * (e.cycles / e.duration -
                 (phase(e.last_center) - phase(e.last_center - e.duration)) / e.duration) for e in measured]
    @test std(residuals) / mean(RATE_WAVELENGTH * sqrt(2σ_φ²) / (2π * e.duration) for e in measured) ≈ 1.0 atol = 0.15
    r = residuals .- mean(residuals)
    @test sum(r[1:end-1] .* r[2:end]) / sum(abs2, r) ≈ -0.5 atol = 0.1
end

@testset "A weak satellite's rates have no cycle outliers, polarity $polarity at $cn0_dbhz dB-Hz" for (polarity, cn0_dbhz) in (
    (0, 37.0), # a data signal, Costas
    (1, 34.0), # a pilot, four-quadrant
)
    # 1 ms records: the prompt noise turns consecutive prompts by more than the FLL's
    # range now and then, a quarter cycle for the Costas FLL and half a cycle for the
    # four-quadrant one. Such a turn must not leave half a cycle (λ/2 over the 100 ms
    # cycle, 6–7σ) or a whole one in the cycle's rate: the residuals stay Gaussian at
    # the modelled variance, with the −1/2 lag-1 autocorrelation of shared boundary
    # prompts.
    cn0 = 10^(cn0_dbhz / 10)
    record_samples = 4000
    T_coh = record_samples / RATE_FS
    phase(t) = 0.1 + 800.0 * t + 0.2 / RATE_WAVELENGTH * t^2 / 2
    epochs = track_carrier(phase, 800.0; record_samples, first_end = record_samples, duration = 60.0,
        noise_std = sqrt(1 / (2 * cn0 * T_coh)), rng = Xoshiro(11), polarity)
    measured = epochs[11:end]
    σ_φ² = 1 / (2 * cn0 * T_coh) * (1 + 1 / (2 * cn0 * T_coh))
    σ_rate = mean(RATE_WAVELENGTH * sqrt(2σ_φ²) / (2π * e.duration) for e in measured)
    residuals = [RATE_WAVELENGTH * (e.cycles / e.duration -
                 (phase(e.last_center) - phase(e.last_center - e.duration)) / e.duration) for e in measured]
    @test maximum(abs, residuals) < 5σ_rate
    @test std(residuals) / σ_rate ≈ 1.0 atol = 0.15
    r = residuals .- mean(residuals)
    @test sum(r[1:end-1] .* r[2:end]) / sum(abs2, r) ≈ -0.5 atol = 0.1
end
