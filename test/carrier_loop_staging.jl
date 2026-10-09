# The carrier loop's staging: FLL-assisted until the phase-lock indicator reads
# lock, a wide pure PLL until it has held lock for longer, a narrow pure PLL
# after, and four-quadrant discriminators on a wiped-off prompt. Ported from
# Tracking.jl's `test/carrier_loop_staging.jl`; the end-to-end tracking tests
# stay there.

using TrackingLoopFilters: ThirdOrderAssistedBilinearLF, ThirdOrderBilinearLF
using Accessors: @set, setproperties
using Random: Xoshiro
using Statistics: mean, median

_staging_correlator(p) = EarlyPromptLateCorrelator(SVector(p / 2, p, p / 2), 0.5)

# One 1 ms GPS L5Q record through `estimator`'s loop at a fixed zero replica
# word; returns the carrier update and the new state.
function staging_step(
    estimator,
    state,
    prompt,
    previous_prompt,
    polarity = 0,
)
    record = LoopRecord(
        GPSL5Q(),
        _staging_correlator(prompt),
        previous_prompt,
        5000,
        5000,
        5000,
        1,
        5e6Hz;
        polarity,
    )
    # `step_satellite` (from `vector_estimator.jl`) is `step_loop` but for the
    # vector estimator, whose own loop it runs without the navigation engine.
    state, carrier_doppler, _ = step_satellite(
        estimator,
        state,
        record,
        FixedNCOWord(0.0, 0.0),
        NO_LANDING_SAMPLE,
    )
    carrier_doppler - state_init_carrier_doppler(state), state
end
state_init_carrier_doppler(state) = state.init_carrier_doppler
state_init_carrier_doppler(state::SatVectorPLLAndDLL) = state.inner.init_carrier_doppler

@testset "The sync's sign" begin
    # The sync polarity times secondary chip 0, which the pre-sync replica
    # carries on every block: +1 for GPS L5Q, -1 for every Galileo E1C PRN.
    synced(polarity) =
        setproperties(BitBuffer{UInt32}(), (; found = true, polarity = Int8(polarity)))
    @test @inferred(get_sync_polarity(GPSL5Q(), synced(1), 1)) === Int8(1)
    @test get_sync_polarity(GPSL5Q(), synced(-1), 1) === Int8(-1)
    @test get_sync_polarity(GalileoE1C(), synced(1), 1) === Int8(-1)
    @test get_sync_polarity(GalileoE1C(), synced(-1), 11) === Int8(1)
    # None where there is no such sync: before it, on data, and on the pilots
    # without a secondary code, which stay two-quadrant.
    @test get_sync_polarity(GalileoE1C(), BitBuffer{UInt32}(), 1) === Int8(0)
    @test get_sync_polarity(GPSL1CA(), synced(1), 1) === Int8(0)
    @test get_sync_polarity(GPSL5I(), synced(1), 1) === Int8(0)
    @test get_sync_polarity(GalileoE5aQP(), synced(1), 1) === Int8(0)
    @test get_sync_polarity(GPSL2CL(), BitBuffer{UInt32}(), 1) === Int8(0)
end

@testset "Phase-lock indicator time constant and threshold" begin
    @test TrackingLoops._phase_lock_time_constant(1ms) == 0.1s
    @test TrackingLoops._phase_lock_time_constant(4ms) == 0.1s
    @test TrackingLoops._phase_lock_time_constant(10ms) == 0.25s
    @test TrackingLoops._phase_lock_time_constant(20ms) == 0.5s
    @test phase_lock_indicator_threshold(GPSL5Q(), 1ms) == 0.5
end

@testset "Phase-lock indicator" begin
    update(indicator, prompt, T) =
        TrackingLoops._update_phase_lock(indicator, prompt, T, GPSL5Q())
    T = 1ms
    # The indicator after `n` records of the prompts `prompt(k)`.
    function indicator_after(prompt, n; T = T)
        indicator = TrackingLoops.PhaseLockIndicator()
        for k = 1:n
            indicator = @inferred update(indicator, prompt(k), T)
        end
        indicator
    end
    @test isnan(phase_lock_indicator(TrackingLoops.PhaseLockIndicator()))
    # Nothing until the averages span the 0.1 s time constant: 100 records.
    @test isnan(phase_lock_indicator(indicator_after(k -> cis(0.1), 99)))
    @test !isnan(phase_lock_indicator(indicator_after(k -> cis(0.1), 100)))

    # A clean prompt at a fixed phase: cos 2φ.
    locked = indicator_after(k -> 3.0 * cis(0.1), 110)
    @test phase_lock_indicator(locked) ≈ cos(0.2)
    # Data bits and secondary-code chips flip the prompt's sign: it is blind to it.
    flips = indicator_after(k -> (isodd(k ÷ 7) ? -3.0 : 3.0) * cis(0.1), 200)
    @test phase_lock_indicator(flips) ≈ cos(0.2)
    # A spinning phase reads about zero.
    spinning = indicator_after(k -> cis(2π * 20.0 * k * 1e-3), 200)
    @test abs(phase_lock_indicator(spinning)) < 0.05

    # Normalised by the averaged signal power, it reads the phase, not the SNR: at
    # 0 dB per record a locked loop's median reading stays near 1, where
    # Σ(I² − Q²) / Σ(I² + Q²) reads 1/2. Single readings scatter widely at that SNR
    # (a small power estimate inflates them), which the hold absorbs.
    rng = Xoshiro(1)
    indicator = TrackingLoops.PhaseLockIndicator()
    readings = Float64[]
    ratio = 0.0
    power = 0.0
    for k = 1:50_000
        p = 1.0 + (randn(rng) + im * randn(rng)) / sqrt(2)
        indicator = update(indicator, p, T)
        ratio += real(p)^2 - imag(p)^2
        power += abs2(p)
        k % 100 == 0 && k > 100 && push!(readings, phase_lock_indicator(indicator))
    end
    @test median(readings) ≈ 1.0 atol = 0.1
    @test ratio / power ≈ 0.5 atol = 0.02

    # A record of another length restarts the averages; the sample-rounding jitter
    # of a record's length does not.
    longer = update(locked, 20.0 * cis(0.1), 20ms)
    @test longer.num_records == 1 && isnan(phase_lock_indicator(longer))
    @test update(locked, cis(0.1), 1.0002ms).num_records == locked.num_records + 1

    # The hold runs from the first reading at or above the threshold, and breaks at
    # the first below it: the 100th record of a locked prompt is its first reading.
    @test indicator_after(k -> 3.0 * cis(0.1), 100).hold == 1ms
    @test locked.hold ≈ 11ms
    @test update(locked, 3.0 * cis(0.1), T).hold ≈ 12ms
    @test spinning.hold == 0.0s
    @test TrackingLoops.PhaseLockIndicator().hold == 0.0s
    # A record of another length restarts the averages and so the hold.
    @test longer.hold == 0.0s
    # Restarting the hold keeps the averages.
    restarted = TrackingLoops._restart_hold(locked)
    @test restarted.hold == 0.0s
    @test phase_lock_indicator(restarted) == phase_lock_indicator(locked)
    @test TrackingLoops._held(100ms, 1, T) && !TrackingLoops._held(99ms, 1, T)
end

@testset "The wide stage of the FLL-assisted filter is the pure PLL, with $(nameof(typeof(assisted_estimator)))" for assisted_estimator in
                                                                                                                (
    ConventionalAssistedPLLAndDLL(),
    NCOReferencedPLLAndDLL(),
)
    # The plain PLL at the assisted loop's own wide bandwidth (the
    # NCO-referenced loop's default is narrower).
    plain_estimator = ConventionalPLLAndDLL(;
        wide_carrier_loop_filter_bandwidth = init_estimator_state(assisted_estimator, GPSL5Q(), 0.0Hz, 0.0Hz).bandwidths.wide_carrier,
    )
    @test carrier_loop_stage(init_estimator_state(plain_estimator, GPSL5Q(), 0.0Hz, 0.0Hz)) == WIDE_PLL
    @test carrier_loop_stage(init_estimator_state(assisted_estimator, GPSL5Q(), 0.0Hz, 0.0Hz)) ==
          FLL_ASSISTED_PLL
    wide(estimator) = @set init_estimator_state(estimator, GPSL5Q(), 0.0Hz, 0.0Hz).staging.stage =
        WIDE_PLL
    assisted = wide(assisted_estimator)
    plain = wide(plain_estimator)
    previous_prompt = cis(0.0)
    for k = 1:20
        prompt = cis(0.3 * sin(k))
        assisted_update, assisted = staging_step(assisted_estimator, assisted, prompt, previous_prompt)
        plain_update, plain = staging_step(plain_estimator, plain, prompt, previous_prompt)
        @test assisted_update == plain_update
        previous_prompt = prompt
    end
    @test assisted.carrier_loop_filter.x1 == plain.carrier_loop_filter.x1
    @test assisted.carrier_loop_filter.x2 == plain.carrier_loop_filter.x2

    # While FLL-assisted the FLL branch reads the FLL discriminator, at the FLL's
    # own bandwidth.
    assisting = init_estimator_state(assisted_estimator, GPSL5Q(), 0.0Hz, 0.0Hz)
    with_fll, _ = staging_step(assisted_estimator, assisting, cis(0.3), cis(0.0))
    without_fll, _ = staging_step(assisted_estimator, assisting, cis(0.3), 0.0im)
    @test with_fll != without_fll
    wider_fll = @set assisting.bandwidths.fll_assist = 10.0Hz
    @test staging_step(assisted_estimator, wider_fll, cis(0.3), cis(0.0))[1] != with_fll
    no_fll = @set assisting.bandwidths.fll_assist = 0.0Hz
    @test staging_step(assisted_estimator, no_fll, cis(0.3), cis(0.0))[1] == without_fll

    # Resetting restarts the staging.
    reset = reset_estimator_state(assisted_estimator, assisted, 0.0Hz, 0.0Hz)
    @test carrier_loop_stage(reset) == FLL_ASSISTED_PLL
    @test reset.staging.phase_lock === TrackingLoops.PhaseLockIndicator()
    @test reset.staging.phase_lock.hold == 0.0s
end

@testset "The stage sets the carrier bandwidth" begin
    estimator = ConventionalAssistedPLLAndDLL()
    state = init_estimator_state(estimator, GPSL5Q(), 0.0Hz, 0.0Hz)
    bandwidth(state, T) =
        TrackingLoops._carrier_bandwidth(state.bandwidths, carrier_loop_stage(state), T)
    @test bandwidth(state, 1ms) == 50.0Hz
    @test bandwidth(state, 4ms) ≈ 22.5Hz
    @test bandwidth(state, 10ms) ≈ 9.0Hz
    @test bandwidth((@set state.staging.stage = WIDE_PLL), 1ms) == 50.0Hz
    narrow = @set state.staging.stage = NARROW_PLL
    @test bandwidth(narrow, 1ms) == 18.0Hz
    @test bandwidth(narrow, 4ms) ≈ 10.0Hz
    @test bandwidth(narrow, 10ms) ≈ 4.0Hz
    @test TrackingLoops._fll_assist_bandwidth(state.bandwidths, 1ms) == 5.0Hz
    @test TrackingLoops._fll_assist_bandwidth(state.bandwidths, 10ms) ≈ 2.0Hz
end

@testset "Scalar loop: staged on phase lock, four-quadrant from the sync" begin
    # A closed loop on a synced pilot's prompt, locked half a cycle off, which
    # the sync's sign of -1 accounts for.
    integration_time = 1 / 1000Hz
    true_doppler = 5.0Hz
    estimator = ConventionalAssistedPLLAndDLL()
    state = init_estimator_state(estimator, GPSL5Q(), 0.0Hz, 0.0Hz)
    φ = 0.4  # signal minus replica phase, rad
    previous_prompt = 0.0im
    max_error = 0.0
    freq_update = 0.0Hz
    stages = CarrierLoopStage[]
    for k = 1:3000
        prompt = -cis(φ)
        freq_update, state = staging_step(estimator, state, prompt, previous_prompt, -1)
        push!(stages, carrier_loop_stage(state))
        carrier_loop_stage(state) == NARROW_PLL &&
            (max_error = max(max_error, abs(rem2pi(φ, RoundNearest))))
        previous_prompt = prompt
        φ += 2π * Float64((true_doppler - freq_update) * integration_time)
    end
    # The FLL dropped once the phase-lock indicator, reading from record 100 on,
    # has held lock for 0.1 s; the loop narrowed once it has held for another
    # 0.4 s from there.
    first_wide = findfirst(==(WIDE_PLL), stages)
    first_narrow = findfirst(==(NARROW_PLL), stages)
    @test first_wide == 199
    @test first_narrow == 599
    @test carrier_loop_stage(state) == NARROW_PLL
    @test phase_lock_indicator(state) > 0.99
    @test max_error < 0.2
    @test abs(rem2pi(φ, RoundNearest)) < 0.01
    # Pure PLL from the FLL drop on: the FLL branch adds nothing.
    @test freq_update ≈ true_doppler atol = 0.01Hz

    # The Costas PLL locks half a cycle off on the same prompt, and without the
    # wipe-off the FLL stays two-quadrant: the loop runs, but the sign is lost.
    costas = init_estimator_state(estimator, GPSL5Q(), 0.0Hz, 0.0Hz)
    φ = 0.4 + π
    previous_prompt = 0.0im
    for k = 1:3000
        prompt = -cis(φ)
        freq_update, costas = staging_step(estimator, costas, prompt, previous_prompt)
        previous_prompt = prompt
        φ += 2π * Float64((true_doppler - freq_update) * integration_time)
    end
    @test abs(rem2pi(φ, RoundNearest)) > π - 0.01
end

@testset "NCO-referenced landing prediction keeps the four-quadrant range" begin
    # A phase error past a quarter cycle (in cycles, as `pll_disc` reads it) stays
    # past it under a known polarity, and is folded by the Costas range without
    # one.
    estimator = NCOReferencedPLLAndDLL()
    state = init_estimator_state(estimator, GPSL5Q(), 0.0Hz, 0.0Hz)
    words = FixedNCOWord(0.0, 0.0)
    predict(polarity) = TrackingLoops._predict_landing_phase_error(
        0.4,
        state,
        words,
        2500.0,
        0,
        1ms,
        5e6Hz,
        polarity,
    )
    @test predict(1) ≈ 0.4
    @test predict(-1) ≈ 0.4
    @test predict(0) ≈ 0.4 - 0.5
end

@testset "Vector loop: FLL-assisted, four-quadrant from the sync" begin
    # The vector loop decodes its signal's data, so it is built for GPS L5I; the
    # satellite's own loop reads the signal off the (L5Q) record.
    estimator = VectorPLLAndDLL(GPSL5I())
    vt_state = TrackingLoops._enable_vector_tracking(
        init_estimator_state(estimator, GPSL5Q(), 0.0Hz, 0.0Hz),
    )
    # A 120° advance in 1 ms: 333 Hz four-quadrant, -167 Hz two-quadrant.
    _, wiped_off = staging_step(estimator, vt_state, cis(2π / 3), cis(0.0), -1)
    _, with_flips = staging_step(estimator, vt_state, cis(2π / 3), cis(0.0), 0)
    @test TrackingLoops._mean_carrier_discriminator(wiped_off) ≈ (333 + 1 / 3) * 1Hz
    @test TrackingLoops._mean_carrier_discriminator(with_flips) ≈ -(166 + 2 / 3) * 1Hz
    # Not staged in the vector loop: it stays FLL-assisted.
    for _ = 1:2000
        _, vt_state = staging_step(estimator, vt_state, cis(0.01), cis(0.0))
    end
    @test carrier_loop_stage(vt_state) == FLL_ASSISTED_PLL
    @test vt_state.inner.staging.phase_lock === TrackingLoops.PhaseLockIndicator()

    # Out of the vector loop it stages like the scalar loop.
    fallback = @set vt_state.vt_on = false
    for _ = 1:1500
        _, fallback = staging_step(estimator, fallback, cis(0.01), cis(0.0))
    end
    @test carrier_loop_stage(fallback) == NARROW_PLL
    @test phase_lock_indicator(fallback) ≈ cos(0.02)
end
