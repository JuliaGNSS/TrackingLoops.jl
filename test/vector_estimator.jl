using TrackingLoopFilters: filter_loop, ThirdOrderBilinearLF

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
    estimator = VectorPLLAndDLL()
    @test estimator.inner isa ConventionalPLLAndDLL{ThirdOrderAssistedBilinearLF}
    state = init_estimator_state(estimator, GPSL1CA(), 500.0Hz, 100.0Hz)
    @test state isa SatVectorPLLAndDLL
    @test state.inner == init_estimator_state(ConventionalAssistedPLLAndDLL(), GPSL1CA(), 500.0Hz, 100.0Hz)
    @test state.vt_on == false
    @test state.code_discr_acc == (0, 0.0)
    @test state.carrier_discr_acc == (0, 0.0Hz)
    @test state.code_freq_update == 0.0Hz
    @test state.carrier_freq_update == 0.0Hz
    @test state.previous_code_freq_update == 0.0Hz
    @test state.code_update_landing_lead == 0.0s
    @test estimator_state_type(estimator, GPSL1CA()) === typeof(state)
    @test mean_code_discriminator(state) === nothing
    @test mean_carrier_discriminator(state) === nothing
end

@testset "Vector PLL and DLL seeding" begin
    # Auto bandwidths resolve from the driver signal, sized exactly like the
    # inner (scalar fallback) loop.
    state = init_estimator_state(VectorPLLAndDLL(), GPSL1CA(), 100.0Hz, 0.0Hz)
    @test state.inner.carrier_loop_filter_bandwidth == 18.0Hz
    @test state.inner.code_loop_filter_bandwidth == 1.0Hz
    @test state.inner.init_carrier_doppler == 100.0Hz
    explicit = ConventionalAssistedPLLAndDLL(; carrier_loop_filter_bandwidth = 12.0Hz, code_loop_filter_bandwidth = 1.0Hz)
    explicit_state = init_estimator_state(VectorPLLAndDLL(explicit), GPSL1CA(), 100.0Hz, 0.0Hz)
    @test explicit_state.inner.carrier_loop_filter_bandwidth == 12.0Hz
    @test explicit_state.inner.code_loop_filter_bandwidth == 1.0Hz
    bumped = VectorPLLAndDLL(ConventionalPLLAndDLL(explicit; carrier_loop_filter_bandwidth = 15.0Hz))
    @test bumped.inner.carrier_loop_filter_bandwidth == 15.0Hz
    @test bumped.inner.code_loop_filter_bandwidth == 1.0Hz
    referenced = init_estimator_state(VectorPLLAndDLL(NCOReferencedPLLAndDLL()), GalileoE1B(), 0.0Hz, 0.0Hz)
    @test referenced.inner isa SatNCOReferencedPLLAndDLL
    @test referenced.inner.carrier_loop_filter_bandwidth == default_carrier_loop_filter_bandwidth(GalileoE1B())
end

@testset "A vector loop needs an FLL-assisted inner loop" begin
    @test_throws ArgumentError VectorPLLAndDLL(ConventionalPLLAndDLL())
    @test_throws ArgumentError VectorPLLAndDLL(ConventionalPLLAndDLL(ThirdOrderBilinearLF))
    @test_throws ArgumentError VectorPLLAndDLL(VectorPLLAndDLL())
    @test VectorPLLAndDLL(ConventionalAssistedPLLAndDLL()) isa VectorPLLAndDLL
    @test VectorPLLAndDLL(NCOReferencedPLLAndDLL()) isa VectorPLLAndDLL
end

@testset "Out of the vector loop it is the inner loop, with $(nameof(typeof(inner)))" for inner in (
    ConventionalAssistedPLLAndDLL(),
    NCOReferencedPLLAndDLL(),
)
    # Under an NCO delay too: the delayed simulation of `estimators.jl`.
    for d in (0, 3)
        @test simulate_delayed_loop(VectorPLLAndDLL(inner), d; steps = 300) ==
              simulate_delayed_loop(inner, d; steps = 300)
    end
    # And record by record on one fixed word: nothing is accumulated and the
    # corrections are never written.
    signal = GPSL1CA()
    fs = 5e6Hz
    vector = VectorPLLAndDLL(inner)
    vector_state = set_vector_corrections(init_estimator_state(vector, signal, 100.0Hz, 0.1Hz), 0.5Hz, 2.0Hz)
    inner_state = init_estimator_state(inner, signal, 100.0Hz, 0.1Hz)
    record = vector_record(signal, (1000.0 + 10im, 2000.0 + 20im, 750.0 + 10im), 5000, fs; previous_prompt = cis(0.1))
    words = FixedNCOWord(100.0, 0.1)
    new_vector_state, vector_carrier, vector_code = step_loop(vector, vector_state, record, words, NO_LANDING_SAMPLE)
    new_inner_state, inner_carrier, inner_code = step_loop(inner, inner_state, record, words, NO_LANDING_SAMPLE)
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

    estimator = VectorPLLAndDLL(inner)
    state = init_estimator_state(estimator, gpsl1, carrier_doppler, init_code_doppler)
    state = enable_vector_tracking(state)
    state = set_vector_corrections(state, nav_code_freq_update, nav_carrier_freq_update)
    record = vector_record(gpsl1, accumulators, num_samples, sampling_frequency)
    words = FixedNCOWord(ustrip(Hz, carrier_doppler), ustrip(Hz, init_code_doppler))
    state, new_carrier_doppler, new_code_doppler = step_loop(estimator, state, record, words, NO_LANDING_SAMPLE)

    normalized_correlator = update_accumulator(
        get_default_correlator(gpsl1),
        SVector(accumulators...) ./ num_samples,
    )
    integration_time = num_samples / sampling_frequency
    pll_discriminator = pll_disc(gpsl1, normalized_correlator)
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
    # The discriminators are accumulated; the FLL one is zero here, with count
    # one, since there is no previous prompt.
    @test state.code_discr_acc == (1, dll_discriminator)
    @test state.carrier_discr_acc == (1, 0.0Hz)
    @test mean_code_discriminator(state) == dll_discriminator
    @test mean_carrier_discriminator(state) == 0.0Hz
    # The corrections survive the step untouched.
    @test state.code_freq_update == nav_code_freq_update
    @test state.carrier_freq_update == nav_carrier_freq_update

    state = reset_discriminator_accumulators(state)
    @test state.code_discr_acc == (0, 0.0)
    @test state.carrier_discr_acc == (0, 0.0Hz)
    @test mean_code_discriminator(state) === nothing
    @test mean_carrier_discriminator(state) === nothing
    @test state.vt_on
    @test state.code_freq_update == nav_code_freq_update
end

@testset "The vector loop accumulates the raw FLL discriminator" begin
    signal = GPSL1CA()
    fs = 5e6Hz
    estimator = VectorPLLAndDLL()
    state = enable_vector_tracking(init_estimator_state(estimator, signal, 100.0Hz, 0.1Hz))
    previous = cis(0.2)
    record = vector_record(signal, (1000.0 + 10im, 2000.0 + 400im, 750.0 + 10im), 5000, fs; previous_prompt = previous)
    fll = fll_disc(signal, record.filtered_correlator, previous, 5000 / fs)
    @test fll != 0.0Hz
    state, = step_loop(estimator, state, record, FixedNCOWord(100.0, 0.1), NO_LANDING_SAMPLE)
    state, = step_loop(estimator, state, record, FixedNCOWord(100.0, 0.1), NO_LANDING_SAMPLE)
    @test state.carrier_discr_acc == (2, fll + fll)
    @test mean_carrier_discriminator(state) == (fll + fll) / 2
end

@testset "The NCO-referenced vector loop keeps its landing prediction" begin
    # In the vector loop, with the delay the prediction has something to do,
    # and without it the NCO-referenced loop is the conventional one.
    signal = GPSL1CA()
    fs = 4e6Hz
    function run(inner, landing_offset)
        estimator = VectorPLLAndDLL(inner)
        state = init_estimator_state(estimator, signal, 100.0Hz, 0.1Hz)
        state = set_vector_corrections(enable_vector_tracking(state), 0.1Hz, 1.0Hz)
        timeline = NCOTimeline()
        reset_timeline!(timeline, 100.0, 0.1)
        previous = complex(0.0, 0.0)
        carriers = typeof(1.0Hz)[]
        for k = 1:50
            p = cis(0.3 + 0.01k)
            output = CorrelatorOutput(loop_epl(0.5p, p, 0.5p), 4000, 4000k)
            record = LoopRecord(signal, output.correlator, previous, output, 1, fs)
            landing = landing_offset == 0 ? NO_LANDING_SAMPLE : Int64(4000k + landing_offset)
            state, carrier, code = step_loop(estimator, state, record, timeline, landing)
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
    @test predicted_state.carrier_discr_acc[1] == 50
end

@testset "Vector tracking state management" begin
    estimator = VectorPLLAndDLL()
    state = init_estimator_state(estimator, GPSL1CA(), 100.0Hz, 0.1Hz)
    joined = enable_vector_tracking(state)
    @test joined.vt_on
    # Re-enabling a member changes nothing, so the members can be enabled every
    # cycle.
    steered = SatVectorPLLAndDLL(
        set_vector_corrections(joined, 1.0Hz, 10.0Hz, 0.02s);
        code_discr_acc = (3, 0.3),
        carrier_discr_acc = (3, 1.5Hz),
    )
    @test enable_vector_tracking(steered) === steered
    # The corrections take over one another.
    @test steered.code_freq_update == 1.0Hz
    @test steered.carrier_freq_update == 10.0Hz
    @test steered.previous_code_freq_update == 0.0Hz
    @test steered.code_update_landing_lead == 0.02s
    next = set_vector_corrections(steered, 2.0Hz, 20.0Hz)
    @test next.previous_code_freq_update == 1.0Hz
    @test next.code_freq_update == 2.0Hz
    @test next.code_update_landing_lead == 0.0s
    @test next.code_discr_acc == (3, 0.3)
    # Disabling zeroes everything the vector loop owned, so a re-enabled
    # satellite never runs on stale values.
    released = disable_vector_tracking(next)
    @test !released.vt_on
    @test released.inner == next.inner
    @test released.code_discr_acc == (0, 0.0)
    @test released.carrier_discr_acc == (0, 0.0Hz)
    @test released.code_freq_update == 0.0Hz
    @test released.carrier_freq_update == 0.0Hz
    @test released.previous_code_freq_update == 0.0Hz
    @test released.code_update_landing_lead == 0.0s
    @test enable_vector_tracking(released).code_freq_update == 0.0Hz
end

@testset "Resetting keeps the vector flag and zeroes the corrections" begin
    estimator = VectorPLLAndDLL(ConventionalAssistedPLLAndDLL(; carrier_loop_filter_bandwidth = 12.0Hz))
    state = init_estimator_state(estimator, GPSL1CA(), 100.0Hz, 0.1Hz)
    record = vector_record(GPSL1CA(), (1000.0 + 10im, 2000.0 + 200im, 750.0 + 10im), 5000, 5e6Hz; previous_prompt = cis(0.1))
    state = set_vector_corrections(enable_vector_tracking(state), 0.5Hz, 4.0Hz, 0.01s)
    state, = step_loop(estimator, state, record, FixedNCOWord(100.0, 0.1), NO_LANDING_SAMPLE)
    state = set_vector_corrections(state, 0.6Hz, 4.0Hz)
    @test state.inner.carrier_loop_filter != ThirdOrderAssistedBilinearLF()
    reset = reset_estimator_state(estimator, state, 150.0Hz, 0.2Hz)
    @test reset isa typeof(state)
    @test reset.vt_on
    @test reset.code_discr_acc == (0, 0.0)
    @test reset.carrier_discr_acc == (0, 0.0Hz)
    @test reset.code_freq_update == 0.0Hz
    @test reset.carrier_freq_update == 0.0Hz
    @test reset.previous_code_freq_update == 0.0Hz
    @test reset.code_update_landing_lead == 0.0s
    @test reset.inner.carrier_loop_filter == ThirdOrderAssistedBilinearLF()
    @test reset.inner.carrier_loop_filter_bandwidth == 12.0Hz
    @test reset.inner.init_carrier_doppler == 150.0Hz
    @test reset.inner.init_code_doppler == 0.2Hz
end
