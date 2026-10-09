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
    @test state.carrier_discr_acc == (0, 0.0Hz)
    @test state.code_freq_update == 0.0Hz
    @test state.carrier_freq_update == 0.0Hz
    @test state.code_freq_update_history == (0.0Hz, 0.0Hz, 0.0Hz)
    @test state.code_update_landing_lead == 0.0s
    @test estimator_state_type(estimator, GPSL1CA()) === typeof(state)
    @test TL._mean_code_discriminator(state) === nothing
    @test TL._mean_carrier_discriminator(state) === nothing
end

@testset "Vector PLL and DLL seeding" begin
    # Auto bandwidths resolve from the driver signal, sized exactly like the
    # inner (scalar fallback) loop.
    state = init_estimator_state(VectorPLLAndDLL(GPSL1CA()), GPSL1CA(), 100.0Hz, 0.0Hz)
    @test state.inner.bandwidths.wide_carrier == 50.0Hz
    @test state.inner.bandwidths.code == 1.0Hz
    @test state.inner.init_carrier_doppler == 100.0Hz
    explicit = ConventionalAssistedPLLAndDLL(; wide_carrier_loop_filter_bandwidth = 12.0Hz, code_loop_filter_bandwidth = 1.0Hz)
    explicit_state = init_estimator_state(VectorPLLAndDLL(GPSL1CA(); inner = explicit), GPSL1CA(), 100.0Hz, 0.0Hz)
    @test explicit_state.inner.bandwidths.wide_carrier == 12.0Hz
    @test explicit_state.inner.bandwidths.code == 1.0Hz
    bumped = VectorPLLAndDLL(GPSL1CA(); inner = ConventionalPLLAndDLL(explicit; wide_carrier_loop_filter_bandwidth = 15.0Hz))
    @test bumped.inner.wide_carrier_loop_filter_bandwidth == 15.0Hz
    @test bumped.inner.code_loop_filter_bandwidth == 1.0Hz
    referenced = init_estimator_state(VectorPLLAndDLL(GalileoE1B(); inner = NCOReferencedPLLAndDLL()), GalileoE1B(), 0.0Hz, 0.0Hz)
    @test referenced.inner isa SatNCOReferencedPLLAndDLL
    @test referenced.inner.bandwidths.wide_carrier == default_narrow_carrier_loop_filter_bandwidth(GalileoE1B())
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
    @test new_vector_state.carrier_discr_acc == (0, 0.0Hz)
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
    pll_discriminator = pll_disc(gpsl1, normalized_correlator)  # cycles
    dll_discriminator = dll_disc(gpsl1, normalized_correlator, init_code_doppler, sampling_frequency)
    # The FLL branch is driven by the vector loop's carrier update.
    expected_carrier_freq_update, _ = filter_loop(
        ThirdOrderAssistedBilinearLF(),
        (pll_discriminator, nav_carrier_freq_update),
        integration_time,
        18.0Hz,
    )
    @test new_carrier_doppler == carrier_doppler + expected_carrier_freq_update
    # The code Doppler follows the vector loop's update directly, plus the
    # carrier filter's own aiding; the code loop filter is frozen.
    @test new_code_doppler ==
          init_code_doppler +
          nav_code_freq_update +
          expected_carrier_freq_update * get_code_center_frequency_ratio(gpsl1)
    @test state.inner.code_loop_filter == SecondOrderBilinearLF()
    # The DLL discriminator is accumulated; the FLL one is not, since there is no
    # previous prompt to read it against.
    @test state.code_discr_acc == (1, dll_discriminator)
    @test state.carrier_discr_acc == (0, 0.0Hz)
    @test TL._mean_code_discriminator(state) == dll_discriminator
    @test TL._mean_carrier_discriminator(state) === nothing
    # The corrections survive the step untouched.
    @test state.code_freq_update == nav_code_freq_update
    @test state.carrier_freq_update == nav_carrier_freq_update

    state = TL._reset_discriminator_accumulators(state)
    @test state.code_discr_acc == (0, 0.0)
    @test state.carrier_discr_acc == (0, 0.0Hz)
    @test TL._mean_code_discriminator(state) === nothing
    @test TL._mean_carrier_discriminator(state) === nothing
    @test state.vt_on
    @test state.code_freq_update == nav_code_freq_update
end

@testset "The vector loop accumulates the raw FLL discriminator" begin
    signal = GPSL1CA()
    fs = 5e6Hz
    estimator = VectorPLLAndDLL(GPSL1CA())
    state = TL._enable_vector_tracking(init_estimator_state(estimator, signal, 100.0Hz, 0.1Hz))
    previous = cis(0.2)
    record = vector_record(signal, (1000.0 + 10im, 2000.0 + 400im, 750.0 + 10im), 5000, fs; previous_prompt = previous)
    fll = fll_disc(signal, record.filtered_correlator, previous, 5000 / fs)
    @test fll != 0.0Hz
    state, = step_satellite(estimator, state, record, FixedNCOWord(100.0, 0.1), NO_LANDING_SAMPLE)
    state, = step_satellite(estimator, state, record, FixedNCOWord(100.0, 0.1), NO_LANDING_SAMPLE)
    @test state.carrier_discr_acc == (2, fll + fll)
    @test TL._mean_carrier_discriminator(state) == (fll + fll) / 2
end

@testset "The vector loop leaves records without a previous prompt out of the FLL accumulator" begin
    # Counted, their 0 Hz would bias the mean: three readings of `fll` would
    # average as 3/4 of it.
    signal = GPSL1CA()
    fs = 5e6Hz
    estimator = VectorPLLAndDLL(GPSL1CA())
    state = TL._enable_vector_tracking(init_estimator_state(estimator, signal, 100.0Hz, 0.1Hz))
    accumulators = (1000.0 + 10im, 2000.0 + 400im, 750.0 + 10im)
    first_record = vector_record(signal, accumulators, 5000, fs)
    record = vector_record(signal, accumulators, 5000, fs; previous_prompt = cis(0.2))
    fll = fll_disc(signal, record.filtered_correlator, cis(0.2), 5000 / fs)
    words = FixedNCOWord(100.0, 0.1)
    state, = step_satellite(estimator, state, first_record, words, NO_LANDING_SAMPLE)
    for _ = 1:3
        state, = step_satellite(estimator, state, record, words, NO_LANDING_SAMPLE)
    end
    @test state.code_discr_acc[1] == 4
    @test state.carrier_discr_acc == (3, fll + fll + fll)
    @test TL._mean_carrier_discriminator(state) ≈ fll
end

@testset "The vector loop accumulates the FLL discriminator in Hz from a MHz sampling frequency" begin
    # The FLL reading took the sampling frequency's unit, and the `Hz`-typed
    # accumulator refused it as soon as the satellite was in the vector loop.
    signal = GPSL1CA()
    estimator = VectorPLLAndDLL(GPSL1CA())
    function accumulated(fs)
        state = TL._enable_vector_tracking(init_estimator_state(estimator, signal, 100.0Hz, 0.1Hz))
        record = vector_record(signal, (1000.0 + 10im, 2000.0 + 400im, 750.0 + 10im), 5000, fs; previous_prompt = cis(0.2))
        state, = step_satellite(estimator, state, record, FixedNCOWord(100.0, 0.1), NO_LANDING_SAMPLE)
        state.carrier_discr_acc
    end
    count, carrier_sum = accumulated(5.0u"MHz")
    @test count == 1
    @test carrier_sum != 0.0Hz
    @test carrier_sum ≈ last(accumulated(5e6Hz))
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
        for k = 1:50
            p = cis(0.3 + 0.01k)
            output = CorrelatorOutput(loop_epl(0.5p, p, 0.5p), 4000, 4000k)
            record = LoopRecord(signal, output.correlator, previous, output, 1, fs)
            landing = landing_offset == 0 ? NO_LANDING_SAMPLE : Int64(4000k + landing_offset)
            state, carrier, code = step_satellite(estimator, state, record, timeline, landing)
            schedule_word!(timeline, 4000k + landing_offset, ustrip(Hz, carrier), ustrip(Hz, code))
            promote_words!(timeline, 4000k - 4000)
            push!(carriers, carrier)
            previous = p
        end
        carriers, state
    end
    referenced, referenced_state = run(NCOReferencedPLLAndDLL(), 0)
    conventional, conventional_state = run(ConventionalAssistedPLLAndDLL(), 0)
    @test referenced == conventional
    @test referenced_state.code_discr_acc == conventional_state.code_discr_acc
    @test referenced_state.carrier_discr_acc == conventional_state.carrier_discr_acc
    predicted, predicted_state = run(NCOReferencedPLLAndDLL(), 12000)
    control, control_state = run(NCOReferencedPLLAndDLL(; predict_landing = false), 12000)
    @test predicted != control
    # The accumulated FLL is the raw discriminator, before any re-basing onto
    # the landing word, so the prediction does not reach it.
    @test predicted_state.carrier_discr_acc == control_state.carrier_discr_acc
    # The first of the 50 records has no previous prompt and is not counted.
    @test predicted_state.carrier_discr_acc[1] == 49
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
        carrier_discr_acc = (3, 1.5Hz),
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
    @test released.carrier_discr_acc == (0, 0.0Hz)
    @test released.code_freq_update == 0.0Hz
    @test released.carrier_freq_update == 0.0Hz
    @test released.code_freq_update_history == (0.0Hz, 0.0Hz, 0.0Hz)
    @test released.code_update_landing_lead == 0.0s
    @test TL._enable_vector_tracking(released).code_freq_update == 0.0Hz
end

@testset "Resetting keeps the vector flag and stops applying the corrections" begin
    estimator = VectorPLLAndDLL(GPSL1CA(); inner = ConventionalAssistedPLLAndDLL(; wide_carrier_loop_filter_bandwidth = 12.0Hz))
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
    @test reset.carrier_discr_acc == (0, 0.0Hz)
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
    @test reset.inner.bandwidths.wide_carrier == 12.0Hz
    @test reset.inner.init_carrier_doppler == 150.0Hz
    @test reset.inner.init_code_doppler == 0.2Hz
    # With nothing applied, the code Doppler is the re-seeded one plus the carrier aiding.
    _, carrier_doppler, code_doppler =
        step_satellite(estimator, reset, record, FixedNCOWord(150.0, 0.2), NO_LANDING_SAMPLE)
    @test code_doppler ≈ 0.2Hz + (carrier_doppler - 150.0Hz) * get_code_center_frequency_ratio(GPSL1CA())
end

@testset "Releasing re-seeds the inner loop, with $(nameof(typeof(inner)))" for inner in (
    ConventionalAssistedPLLAndDLL(; wide_carrier_loop_filter_bandwidth = 12.0Hz),
    NCOReferencedPLLAndDLL(; wide_carrier_loop_filter_bandwidth = 12.0Hz),
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
    # configuration kept: the per-satellite state carries it.
    @test released.inner == reset_estimator_state(inner, state.inner, 150.0Hz, 0.2Hz)
    @test released.inner.init_carrier_doppler == 150.0Hz
    @test released.inner.init_code_doppler == 0.2Hz
    @test released.inner.bandwidths.wide_carrier == 12.0Hz
    # Nothing the vector loop owned survives.
    @test released.code_discr_acc == (0, 0.0)
    @test released.carrier_discr_acc == (0, 0.0Hz)
    @test released.code_freq_update == 0.0Hz
    @test released.carrier_freq_update == 0.0Hz
    @test released.code_freq_update_history == (0.0Hz, 0.0Hz, 0.0Hz)
    # Out of the vector loop it steps exactly as the re-seeded inner loop.
    @test step_satellite(estimator, released, record, FixedNCOWord(150.0, 0.2), NO_LANDING_SAMPLE)[2:3] ==
          step_satellite(inner, released.inner, record, FixedNCOWord(150.0, 0.2), NO_LANDING_SAMPLE)[2:3]
end
