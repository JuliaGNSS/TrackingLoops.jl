# Signal combining: passengers' discriminators mixed into the driver's loops.
# Ported from Tracking.jl's `test/signal_combining.jl`; the end-to-end tests
# stay there.

using TrackingLoopFilters:
    ThirdOrderAssistedBilinearLF, ThirdOrderBilinearLF, SecondOrderBilinearLF, filter_loop
using Accessors: @set

const SC = TrackingLoops

# A record of `signal` whose taps are `taps` turned by the prompt `p`.
function combining_record(signal, p, taps; n = 5000, fs = 5e6Hz, previous_prompt = 0.0im, sample_index = n)
    correlator = update_accumulator(get_default_correlator(signal), p .* SVector(taps...))
    LoopRecord(signal, correlator, previous_prompt, n, sample_index, sample_index, 1, fs)
end

const _NO_WORD = FixedNCOWord(0.0, 0.0)

@testset "Signal combining is an estimator setting, off by default" begin
    @test !TrackingLoops.combines_signals(ConventionalPLLAndDLL())
    @test !TrackingLoops.combines_signals(ConventionalAssistedPLLAndDLL())
    @test !TrackingLoops.combines_signals(NCOReferencedPLLAndDLL())
    @test !TrackingLoops.combines_signals(VectorPLLAndDLL(GPSL1CA()))
    estimator = ConventionalAssistedPLLAndDLL(; combine_signals = true)
    @test TrackingLoops.combines_signals(estimator)
    # The vector loop has its own switch, which covers its scalar fallback too: its
    # inner loop is built without one.
    vector = VectorPLLAndDLL(GPSL1CA(); combine_signals = true)
    @test TrackingLoops.combines_signals(vector)
    @test !TrackingLoops.combines_signals(vector.inner)
    @test !TrackingLoops.combines_signals(VectorPLLAndDLL(GPSL1CA(); combine_signals = false))
    @test TrackingLoops.combines_signals(
        VectorPLLAndDLL(GPSL1CA(); inner = ConventionalAssistedPLLAndDLL(), combine_signals = true),
    )
    @test_throws ArgumentError VectorPLLAndDLL(GPSL1CA(); inner = estimator)
    @test_throws ArgumentError VectorPLLAndDLL(GPSL1CA(); inner = NCOReferencedPLLAndDLL(), combine_signals = true)
    @test !TrackingLoops.combines_signals(VectorPLLAndDLL(GPSL1CA(); inner = NCOReferencedPLLAndDLL()))
    # Whether a host hands passenger records over: a combining loop, and a vector loop
    # with passengers, combining or not.
    @test !takes_passenger_records(ConventionalAssistedPLLAndDLL())
    @test takes_passenger_records(estimator)
    @test !takes_passenger_records(VectorPLLAndDLL(GPSL1CA()))
    @test takes_passenger_records(vector)
    @test takes_passenger_records(VectorPLLAndDLL((GalileoE1C(), GalileoE1B())))
    # A data driver decodes its own bits: without combining it needs no passenger.
    @test !takes_passenger_records(VectorPLLAndDLL((GalileoE1B(), GalileoE1C())))
    @test takes_passenger_records(
        VectorPLLAndDLL((GalileoE1B(), GalileoE1C()); combine_signals = true),
    )
    # The keyword-update constructor keeps it unless told otherwise.
    @test TrackingLoops.combines_signals(ConventionalPLLAndDLL(estimator; code_loop_filter_bandwidth = 2.0Hz))
    @test !TrackingLoops.combines_signals(ConventionalPLLAndDLL(estimator; combine_signals = false))
    # An estimator that does not combine leaves the state as it is.
    state = init_estimator_state(ConventionalAssistedPLLAndDLL(), GPSL5Q(), 0.0Hz, 0.0Hz)
    record = combining_record(GPSL5I(), cis(0.1), (0.5, 1.0, 0.5))
    @test fold_passenger_record(ConventionalAssistedPLLAndDLL(), state, record, _NO_WORD; driver_signal = GPSL5Q()) ===
          state
    nco_state = init_estimator_state(NCOReferencedPLLAndDLL(), GPSL5Q(), 0.0Hz, 0.0Hz)
    @test fold_passenger_record(NCOReferencedPLLAndDLL(), nco_state, record, _NO_WORD; driver_signal = GPSL5Q()) ===
          nco_state
end

@testset "Weights and the weighted mean" begin
    T = 1 / 250Hz  # 4 ms
    @test SC._discriminator_weight(GalileoE1B(), T) ≈ 0.5 * 0.004s
    @test SC._fll_discriminator_weight(GalileoE1B(), T) ≈ 0.5 * (0.004s)^3
    @test SC._discriminator_weight(GPSL1CA(), T) ≈ 0.7079457843841379 * 0.004s
    # Without passengers the driver's own reading, bit for bit.
    @test SC._weighted_mean(0.1234567, 0.002s, SC.WeightedSum(0.0s, 0.0s)) === 0.1234567
    @test SC._weighted_mean(0.1, 0.002s, SC.WeightedSum(0.002s * 0.3, 0.002s)) ≈ 0.2
    @test SC._weighted_mean(0.1, 0.002s, SC.WeightedSum(0.006s * 0.3, 0.006s)) ≈ 0.25
    # The FLL's mean stays in Hz.
    @test SC._weighted_mean(1.0Hz, 1.0s^3, SC.WeightedSum(3.0Hz * 1.0s^3, 1.0s^3)) === 2.0Hz
    # A four-quadrant driver reading is combined only within the passengers'
    # two-quadrant range.
    @test SC._gated_mean(0.2, 1.0s, SC.WeightedSum(0.1s, 1.0s), 0.25) ≈ 0.15
    @test SC._gated_mean(0.3, 1.0s, SC.WeightedSum(0.1s, 1.0s), 0.25) === 0.3
end

@testset "A passenger record's contribution" begin
    T = 1ms
    estimator = ConventionalAssistedPLLAndDLL(; combine_signals = true)
    state = init_estimator_state(estimator, GPSL5Q(), 0.0Hz, 0.0Hz)
    # L5I sits a quarter cycle from L5Q, which the loops lock onto the real axis:
    # its PLL is read on the driver's frame.
    record = combining_record(GPSL5I(), cis(π / 2 + 0.1), (0.5, 1.0, 0.4); previous_prompt = cis(π / 2))
    # 10 ns at 10.23 Mchip/s: the later passenger reads the driver's error minus
    # 0.1023 chip, which the offset adds back.
    combined = fold_passenger_record(
        estimator,
        state,
        record,
        _NO_WORD;
        driver_signal = GPSL5Q(),
        differential_group_delay_chips = 0.1023,
    )
    sums = combined.signal_combining_sums
    weight = 0.5 * 0.001s
    @test sums.pll.weight ≈ weight
    @test sums.pll.sum / sums.pll.weight ≈ 0.1 / 2π
    @test sums.fll.weight ≈ weight * (0.001s)^2
    @test sums.fll.sum / sums.fll.weight ≈ 0.1 / (2π * 0.001) * Hz
    @test sums.dll.weight ≈ weight
    @test sums.dll.sum / sums.dll.weight ≈
          dll_disc(GPSL5I(), record.filtered_correlator, 0.0Hz, 5e6Hz) + 0.1023
    # Everything else in the state is untouched.
    @test (@set combined.signal_combining_sums = SignalCombiningSums()) == state

    # No FLL without a previous prompt, no DLL without a known group delay.
    partial =
        fold_passenger_record(
            estimator,
            state,
            combining_record(GPSL5I(), cis(π / 2 + 0.1), (0.5, 1.0, 0.4)),
            _NO_WORD;
            driver_signal = GPSL5Q(),
        ).signal_combining_sums
    @test iszero(partial.fll.weight) && iszero(partial.dll.weight) && partial.pll.weight > 0s

    # No FLL once it is no longer formed: after the FLL drop, or without an
    # FLL-assisted carrier filter.
    locked = @set state.staging.stage = WIDE_PLL
    @test iszero(
        fold_passenger_record(estimator, locked, record, _NO_WORD; driver_signal = GPSL5Q()).signal_combining_sums.fll.weight,
    )
    plain_estimator = ConventionalPLLAndDLL(; combine_signals = true)
    plain = init_estimator_state(plain_estimator, GPSL5Q(), 0.0Hz, 0.0Hz)
    @test iszero(
        fold_passenger_record(plain_estimator, plain, record, _NO_WORD; driver_signal = GPSL5Q()).signal_combining_sums.fll.weight,
    )

    # In the vector loop only into the PLL; out of it as the inner loop.
    vector = VectorPLLAndDLL((GPSL5Q(), GPSL5I()); combine_signals = true)
    vector_state = init_estimator_state(vector, GPSL5Q(), 0.0Hz, 0.0Hz)
    in_vt = fold_passenger_record(
        vector,
        SC._enable_vector_tracking(vector_state),
        record,
        _NO_WORD;
        driver_signal = GPSL5Q(),
        differential_group_delay_chips = 0.1023,
    ).inner.signal_combining_sums
    @test in_vt.pll == sums.pll
    @test iszero(in_vt.fll.weight) && iszero(in_vt.dll.weight)
    out_of_vt = fold_passenger_record(
        vector,
        vector_state,
        record,
        _NO_WORD;
        driver_signal = GPSL5Q(),
        differential_group_delay_chips = 0.1023,
    ).inner.signal_combining_sums
    @test out_of_vt == sums

    # Pending sums whose passenger record ended at or before the driver record's start
    # belong to a driver record that never came: the step drops them.
    @test sums.first_end_sample == record.sample_index
    driver(sample_index) = combining_record(GPSL5Q(), cis(0.3), (0.5, 1.0, 0.5);
        sample_index, previous_prompt = cis(0.2))
    carrier(state, record) = step_loop(estimator, state, record, _NO_WORD, NO_LANDING_SAMPLE)[2]
    @test carrier(combined, driver(10000)) == carrier(state, driver(10000))
    @test carrier(combined, driver(5000)) != carrier(state, driver(5000))
end

@testset "A passenger's FLL reading, two- or four-quadrant" begin
    signal = GPSL5Q()
    n = 5000
    fs = 5e6Hz
    taps = (0.5, 1.0, 0.5)
    previous = cis(0.0)
    T = n / fs
    # 0.6 rad over one 1 ms record: within the two-quadrant range, both alike.
    within = combining_record(signal, cis(0.6), taps; n, fs, previous_prompt = previous)
    @test SC._passenger_fll_reading(within, true) ≈ SC._passenger_fll_reading(within, false)
    # 2.0 rad: the two-quadrant reading folds it, the four-quadrant one does not.
    beyond = combining_record(signal, cis(2.0), taps; n, fs, previous_prompt = previous)
    @test SC._passenger_fll_reading(beyond, true) ≈ uconvert(Hz, 2.0 / (2π * T))
    @test SC._passenger_fll_reading(beyond, false) ≈ uconvert(Hz, (2.0 - π) / (2π * T))
    # Combining reads it two-quadrant even on a wiped-off record.
    wiped = LoopRecord(signal, beyond.filtered_correlator, previous, n, n, n, 1, fs;
        polarity = 1)
    sums = SC._add_passenger_discriminators(SignalCombiningSums(), wiped, _NO_WORD,
        SC._ALL_LOOPS, signal, NaN)
    @test sums.fll.sum / sums.fll.weight ≈ uconvert(Hz, (2.0 - π) / (2π * T))
end

@testset "Several passengers add up" begin
    estimator = ConventionalAssistedPLLAndDLL(; combine_signals = true)
    state = init_estimator_state(estimator, GPSL1C_P(), 0.0Hz, 0.0Hz)
    fold(state, record) = fold_passenger_record(estimator, state, record, _NO_WORD; driver_signal = GPSL1C_P())
    veml = (0.2, 0.7, 1.0, 0.7, 0.2)
    d_record = combining_record(GPSL1C_D(), cis(0.1), veml; n = 50000)
    ca_records = (
        combining_record(GPSL1CA(), cis(0.2), (0.5, 1.0, 0.5)),
        combining_record(GPSL1CA(), cis(0.25), (0.5, 1.0, 0.5); previous_prompt = cis(0.2), sample_index = 10000),
    )
    d_only = fold(state, d_record).signal_combining_sums
    ca_only = foldl(fold, ca_records; init = state).signal_combining_sums
    both = foldl(fold, ca_records; init = fold(state, d_record)).signal_combining_sums
    @test d_only.pll.weight ≈ SC._discriminator_weight(GPSL1C_D(), 10ms)
    @test ca_only.pll.weight ≈ 2 * SC._discriminator_weight(GPSL1CA(), 1ms)
    # Each passenger adds its own records, whatever the others do.
    @test both.pll.weight ≈ d_only.pll.weight + ca_only.pll.weight
    @test both.pll.sum ≈ d_only.pll.sum + ca_only.pll.sum
    # Only L1 C/A's second record has a previous prompt for the FLL.
    @test iszero(d_only.fll.weight)
    @test both.fll.weight ≈ ca_only.fll.weight > 0s^3
    @test both.fll.sum ≈ ca_only.fll.sum
end

@testset "The driver's step closes on the passengers' means" begin
    T = 1 / 250Hz
    fs = 16.368e6Hz
    n = 65472
    estimator = ConventionalAssistedPLLAndDLL(; combine_signals = true)
    alone_estimator = ConventionalAssistedPLLAndDLL()
    state = init_estimator_state(estimator, GalileoE1C(), 0.0Hz, 0.0Hz)
    driver = combining_record(GalileoE1C(), cis(0.2), (0.6, 0.8, 1.0, 0.6, 0.4); n, fs, previous_prompt = cis(0.1))
    step(state) = step_loop(estimator, state, driver, _NO_WORD, NO_LANDING_SAMPLE)
    w = 0.5 * 0.004s
    # Pending sums of passenger records ending with the driver record.
    pending(pll, fll, dll) = SignalCombiningSums(pll, fll, dll, n)
    with_sums(pll, fll, dll) = @set state.signal_combining_sums = pending(pll, fll, dll)
    no_pll = SC.WeightedSum(0.0s, 0.0s)
    no_fll = SC.WeightedSum(0.0Hz * 0.0s^3, 0.0s^3)

    # No passenger pending: exactly the loop without combining.
    alone = step_loop(alone_estimator, state, driver, _NO_WORD, NO_LANDING_SAMPLE)
    @test step(state) == alone

    # The DLL reads the weighted mean of the driver's and the passengers'.
    own_dll = dll_disc(GalileoE1C(), driver.filtered_correlator, 0.0Hz, fs)
    stepped, = step(with_sums(no_pll, no_fll, SC.WeightedSum(3w * 0.05, 3w)))
    expected = last(filter_loop(SecondOrderBilinearLF(), (own_dll + 3 * 0.05) / 4, T, 1.0Hz))
    @test getfield(stepped.code_loop_filter, 1) ≈ getfield(expected, 1)
    # The sums are consumed.
    @test stepped.signal_combining_sums === SignalCombiningSums()

    # The PLL and FLL take theirs too.
    @test step(with_sums(SC.WeightedSum(w * 0.01, w), no_fll, no_pll))[2] != alone[2]
    fll_pending = SC.WeightedSum(w * (0.004s)^2 * 7.0Hz, w * (0.004s)^2)
    @test step(with_sums(no_pll, fll_pending, no_pll))[2] != alone[2]
    # Not into an FLL that is no longer formed.
    locked = @set state.staging.stage = WIDE_PLL
    @test step(@set locked.signal_combining_sums = pending(no_pll, fll_pending, no_pll))[2] ==
          step(locked)[2]

    # In the vector loop: into the PLL only, and the satellite's accumulators keep the
    # driver's own DLL and FLL readings (the passengers' reach the filter through the
    # engine, see "vector_signal_groups.jl").
    vector = VectorPLLAndDLL(GalileoE1B(); combine_signals = true)
    vt = SC._enable_vector_tracking(init_estimator_state(vector, GalileoE1C(), 0.0Hz, 0.0Hz))
    mixed = pending(
        SC.WeightedSum(w * 0.01, w),
        fll_pending,
        SC.WeightedSum(w * 0.05, w),
    )
    vt_step(state) = step_satellite(vector, state, driver, _NO_WORD, NO_LANDING_SAMPLE)
    combined_vt, combined_carrier, = vt_step(@set vt.inner.signal_combining_sums = mixed)
    alone_vt, alone_carrier, = vt_step(vt)
    @test combined_carrier != alone_carrier
    @test combined_vt.code_discr_acc == alone_vt.code_discr_acc
    @test combined_vt.carrier_discr_acc == alone_vt.carrier_discr_acc
    @test first(
        vt_step(@set vt.inner.signal_combining_sums = pending(no_pll, fll_pending, SC.WeightedSum(w * 0.05, w))),
    ).inner.carrier_loop_filter == alone_vt.inner.carrier_loop_filter
end

@testset "A four-quadrant driver reading beyond the two-quadrant range is not combined" begin
    # A synced pilot driver reads its PLL four-quadrant; beyond ±1/4 cycle a
    # two-quadrant passenger would fold the error, so its pending PLL sum is left
    # out of that step, and joins within the range.
    estimator = ConventionalAssistedPLLAndDLL(; combine_signals = true)
    state = init_estimator_state(estimator, GPSL5Q(), 0.0Hz, 0.0Hz)
    record(phase) = LoopRecord(
        GPSL5Q(),
        update_accumulator(get_default_correlator(GPSL5Q()), cis(phase) .* SVector(0.5, 1.0, 0.5)),
        cis(phase - 0.01),
        5000,
        5000,
        5000,
        1,
        5e6Hz;
        polarity = 1,
    )
    pending = @set state.signal_combining_sums.pll = SC.WeightedSum(0.001s * 0.1, 0.001s)
    step(state, phase) = step_loop(estimator, state, record(phase), _NO_WORD, NO_LANDING_SAMPLE)
    # 2.0 rad, about 0.32 cycle: beyond the range, the driver's own reading.
    @test step(pending, 2.0)[2] == step(state, 2.0)[2]
    # 0.2 rad: within it, combined.
    @test step(pending, 0.2)[2] != step(state, 0.2)[2]
end

# One passenger record folded and one driver record stepped, the state kept in a
# `Ref` so the measurement sees only what the two calls allocate (see
# `allocations.jl` for why the testset's own variables are kept out of it).
function fold_and_step!(state_ref, estimator, passenger, driver, words)
    st = fold_passenger_record(
        estimator,
        state_ref[],
        passenger,
        words;
        # The driver's own signal: constructing one builds its code table.
        driver_signal = driver.signal,
        differential_group_delay_chips = 0.1,
    )
    st, _, _ = step_loop(estimator, st, driver, words, NO_LANDING_SAMPLE)
    state_ref[] = st
    nothing
end

@testset "Signal combining adds no allocation" begin
    estimator = ConventionalAssistedPLLAndDLL(; combine_signals = true)
    state = init_estimator_state(estimator, GPSL5Q(), 0.0Hz, 0.0Hz)
    passenger = combining_record(GPSL5I(), cis(0.1), (0.5, 1.0, 0.5); previous_prompt = cis(0.05))
    driver = combining_record(GPSL5Q(), cis(0.2), (0.5, 1.0, 0.5); previous_prompt = cis(0.1))
    state_ref = Ref(state)
    fold_and_step!(state_ref, estimator, passenger, driver, _NO_WORD)
    @test (@allocated fold_and_step!(state_ref, estimator, passenger, driver, _NO_WORD)) == 0
    @test isempty(
        AllocCheck.check_allocs(
            step_loop,
            Tuple{typeof(estimator),typeof(state),typeof(driver),FixedNCOWord,Int64};
            ignore_throw = true,
        ),
    )
    @test isempty(
        AllocCheck.check_allocs(
            SC._with_passenger_record,
            Tuple{typeof(state),typeof(passenger),FixedNCOWord,typeof(SC._ALL_LOOPS),typeof(driver.signal),Float64};
            ignore_throw = true,
        ),
    )
    # The same with the very-early-prompt-late correlator of a BOC signal pair.
    veml_state = init_estimator_state(estimator, GalileoE1C(), 0.0Hz, 0.0Hz)
    taps = (0.3, 0.6, 1.0, 0.6, 0.3)
    veml_passenger = combining_record(GalileoE1B(), cis(0.1), taps; n = 16368, fs = 4.092e6Hz,
        previous_prompt = cis(0.05))
    veml_driver = combining_record(GalileoE1C(), cis(0.2), taps; n = 16368, fs = 4.092e6Hz,
        previous_prompt = cis(0.1))
    @test get_default_correlator(GalileoE1C()) isa VeryEarlyPromptLateCorrelator
    veml_ref = Ref(veml_state)
    fold_and_step!(veml_ref, estimator, veml_passenger, veml_driver, _NO_WORD)
    @test (@allocated fold_and_step!(veml_ref, estimator, veml_passenger, veml_driver, _NO_WORD)) == 0
end
