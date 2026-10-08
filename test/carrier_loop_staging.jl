# The carrier loop's staging: FLL-assisted until frequency lock, a pure PLL
# after, and four-quadrant discriminators on a wiped-off prompt. Ported from
# Tracking.jl's `test/carrier_loop_staging.jl`; the end-to-end tracking tests
# stay there.

using TrackingLoopFilters: ThirdOrderAssistedBilinearLF, ThirdOrderBilinearLF
using Accessors: @set, setproperties

_staging_correlator(p) = EarlyPromptLateCorrelator(SVector(p / 2, p, p / 2), 0.5)
_locked() = FrequencyLockIndicator(0.0Hz * 0.0s, 0.0s, true)

# One 1 ms GPS L5Q record through `estimator`'s loop at a fixed zero replica
# word; returns the carrier update and the new state.
function staging_step(
    estimator,
    state,
    prompt,
    previous_prompt,
    wiped_off = false,
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
        wiped_off,
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

@testset "Wiped-off prompts and the sync's sign" begin
    # Data signals never are.
    @test !has_wiped_off_prompt(GPSL1CA(), BitBuffer{UInt32}())
    @test !has_wiped_off_prompt(GPSL1CA(), setproperties(BitBuffer{UInt32}(), (; found = true)))
    # Pilots with a secondary code once synced to it ...
    @test !has_wiped_off_prompt(GalileoE1C(), BitBuffer{UInt32}())
    @test has_wiped_off_prompt(GalileoE1C(), setproperties(BitBuffer{UInt32}(), (; found = true)))
    @test !has_wiped_off_prompt(GPSL5Q(), BitBuffer{UInt32}())
    @test has_wiped_off_prompt(GPSL5Q(), setproperties(BitBuffer{UInt32}(), (; found = true)))
    # ... and pilots without one from the start.
    @test has_wiped_off_prompt(GalileoE5aQP(), BitBuffer{UInt32}())
    @test has_wiped_off_prompt(GPSL2CL(), BitBuffer{UInt32}())

    # The sync polarity times secondary chip 0, which the pre-sync replica
    # carries on every block: +1 for GPS L5Q, -1 for every Galileo E1C PRN.
    synced(polarity) =
        setproperties(BitBuffer{UInt32}(), (; found = true, polarity = Int8(polarity)))
    @test @inferred(get_sync_polarity(GPSL5Q(), synced(1), 1)) === Int8(1)
    @test get_sync_polarity(GPSL5Q(), synced(-1), 1) === Int8(-1)
    @test get_sync_polarity(GalileoE1C(), synced(1), 1) === Int8(-1)
    @test get_sync_polarity(GalileoE1C(), synced(-1), 11) === Int8(1)
    # None where there is no such sync: before it, on data, and on the pilots
    # without a secondary code, which keep the Costas PLL.
    @test get_sync_polarity(GalileoE1C(), BitBuffer{UInt32}(), 1) === Int8(0)
    @test get_sync_polarity(GPSL5I(), synced(1), 1) === Int8(0)
    @test get_sync_polarity(GalileoE5aQP(), synced(1), 1) === Int8(0)
    @test get_sync_polarity(GPSL2CL(), BitBuffer{UInt32}(), 1) === Int8(0)
end

@testset "Frequency lock indicator" begin
    update = TrackingLoops._update_frequency_lock
    signal = GPSL5Q()
    T = 1 / 1000Hz
    # Records until lock is declared, for a constant FLL reading.
    function records_to_lock(fll; integration_time = T, max_records = 10_000)
        indicator = FrequencyLockIndicator()
        for n = 1:max_records
            indicator = @inferred update(indicator, signal, fll, cis(0.0), integration_time)
            indicator.locked && return n
        end
        nothing
    end
    # Decided at the end of each 0.5 s window.
    @test records_to_lock(1.0Hz) in 499:501
    @test records_to_lock(-2.9Hz) in 499:501
    @test records_to_lock(1.0Hz; integration_time = 1 / 50Hz) in 24:26
    @test isnothing(records_to_lock(3.5Hz))
    @test isnothing(records_to_lock(-3.5Hz))

    # On long records the window spans at least four of them, and the threshold
    # shrinks to a quarter of the FLL's ±1/(4T) range, which 3 Hz would exceed:
    # GPS L2 CL's 1.5 s records read at most 0.17 Hz.
    @test frequency_lock_window(signal, T) == 0.5s
    @test frequency_lock_threshold(signal, T) == 3.0Hz
    @test frequency_lock_window(GPSL2CL(), 1.5s) == 6.0s
    @test frequency_lock_threshold(GPSL2CL(), 1.5s) ≈ 1 / 24 * 1Hz
    @test isnothing(records_to_lock(0.1Hz; integration_time = 1.5s, max_records = 100))
    @test records_to_lock(0.03Hz; integration_time = 1.5s) == 4

    # It is the window mean that counts: noise around zero averages out, a
    # residual error does not.
    function after_window(fll)
        indicator = FrequencyLockIndicator()
        for k = 1:500
            indicator = update(indicator, signal, fll(k) * 1Hz, cis(0.0), T)
        end
        indicator
    end
    @test after_window(k -> isodd(k) ? 20.0 : -20.0).locked
    @test after_window(k -> isodd(k) ? 24.0 : -16.0) === FrequencyLockIndicator()

    # A record without a previous prompt has no FLL reading, and lock latches.
    @test update(FrequencyLockIndicator(), signal, 0.0Hz, 0.0im, T) ===
          FrequencyLockIndicator()
    @test update(_locked(), signal, 10.0Hz, cis(0.0), T) === _locked()
end

@testset "Frequency-locked FLL-assisted filter is the pure PLL, with $(nameof(typeof(assisted_estimator)))" for assisted_estimator in
                                                                                                                (
    ConventionalAssistedPLLAndDLL(),
    NCOReferencedPLLAndDLL(),
)
    plain_estimator = ConventionalPLLAndDLL()
    locked(estimator) = TrackingLoops._stepped_state(
        init_estimator_state(estimator, GPSL5Q(), 0.0Hz, 0.0Hz),
        nothing,
        nothing,
        NaN,
        _locked(),
    )
    assisted = locked(assisted_estimator)
    plain = locked(plain_estimator)
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

    # Before lock the FLL branch reads the FLL discriminator.
    unlocked = init_estimator_state(assisted_estimator, GPSL5Q(), 0.0Hz, 0.0Hz)
    with_fll, _ = staging_step(assisted_estimator, unlocked, cis(0.3), cis(0.0))
    without_fll, _ = staging_step(assisted_estimator, unlocked, cis(0.3), 0.0im)
    @test with_fll != without_fll

    # Resetting restarts the staging.
    @test reset_estimator_state(assisted_estimator, assisted, 0.0Hz, 0.0Hz).frequency_lock ===
          FrequencyLockIndicator()
end

@testset "Scalar loop: four-quadrant PLL from the sync, FLL dropped at frequency lock" begin
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
    for k = 1:3000
        prompt = -cis(φ)
        freq_update, state = staging_step(estimator, state, prompt, previous_prompt, true, -1)
        state.frequency_lock.locked &&
            (max_error = max(max_error, abs(rem2pi(φ, RoundNearest))))
        previous_prompt = prompt
        φ += 2π * Float64((true_doppler - freq_update) * integration_time)
    end
    @test state.frequency_lock.locked
    @test max_error < 0.2
    @test abs(rem2pi(φ, RoundNearest)) < 0.01
    # Pure PLL from lock on: the FLL branch adds nothing to the frequency state.
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
    _, wiped_off = staging_step(estimator, vt_state, cis(2π / 3), cis(0.0), true, -1)
    _, with_flips = staging_step(estimator, vt_state, cis(2π / 3), cis(0.0), false, 0)
    @test TrackingLoops._mean_carrier_discriminator(wiped_off) ≈ (333 + 1 / 3) * 1Hz
    @test TrackingLoops._mean_carrier_discriminator(with_flips) ≈ -(166 + 2 / 3) * 1Hz
    # No frequency lock indicator in the vector loop: it stays FLL-assisted.
    for _ = 1:2000
        _, vt_state = staging_step(estimator, vt_state, cis(0.01), cis(0.0))
    end
    @test vt_state.inner.frequency_lock === FrequencyLockIndicator()

    # Out of the vector loop it stages like the scalar loop.
    fallback = @set vt_state.vt_on = false
    for _ = 1:1500
        _, fallback = staging_step(estimator, fallback, cis(0.01), cis(0.0))
    end
    @test fallback.inner.frequency_lock.locked
end
