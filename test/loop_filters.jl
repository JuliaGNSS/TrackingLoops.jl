# Loop-filter bandwidth rules and the per-record filter updates. Ported from
# the bandwidth parts of Tracking.jl's `test/conventional_pll_and_dll.jl`.

using Unitful: ms
using TrackingLoopFilters: filter_loop, ThirdOrderBilinearLF
using Random: MersenneTwister
using Statistics: median

@testset "Doppler aiding" begin
    gpsl1 = GPSL1CA()
    carrier_freq, code_freq = @inferred aid_dopplers(gpsl1, 10Hz, 1Hz, 2Hz, -0.5Hz)
    @test carrier_freq == 10Hz + 2Hz
    @test code_freq == 1Hz + 2Hz / 1540 - 0.5Hz
end

@testset "Default loop bandwidths" begin
    # Flat bandwidths for every signal: the PLL's 50 Hz wide and 18 Hz narrow, the
    # FLL-assist 5 Hz and the code 1 Hz.
    for signal in (GPSL1CA(), GPSL1C_P(), GalileoE1B(), GPSL5I())
        @test default_wide_carrier_loop_filter_bandwidth(signal) == 50.0Hz
        @test default_narrow_carrier_loop_filter_bandwidth(signal) == 18.0Hz
        @test default_fll_assist_loop_filter_bandwidth(signal) == 5.0Hz
        @test default_code_loop_filter_bandwidth(signal) == 1.0Hz
    end
    @test default_narrow_carrier_loop_filter_bandwidth(GPSL1CA()) isa typeof(1.0Hz)
    # A state's bandwidths: the signal's defaults, the generic ones without a signal.
    @test LoopBandwidths(GPSL1CA()) == LoopBandwidths() ==
          LoopBandwidths(50.0Hz, 18.0Hz, 5.0Hz, 1.0Hz)
    @test LoopBandwidths(GPSL1CA(); wide_carrier = 18.0Hz).wide_carrier == 18.0Hz
    @test LoopBandwidths(GPSL1CA(); wide_carrier = 18.0Hz).narrow_carrier == 18.0Hz
end

@testset "The FLL-assist bandwidth is capped by its own stability product" begin
    effective = effective_fll_assist_loop_filter_bandwidth
    @test effective(5.0Hz, 1ms) == 5.0Hz
    @test effective(5.0Hz, 4ms) == 5.0Hz
    @test effective(5.0Hz, 10ms) ≈ 2.0Hz
    @test effective(5.0Hz, 20ms) ≈ 1.0Hz
    @test effective(5.0Hz, 20ms) * 20ms ≈ MAX_FLL_ASSIST_LOOP_BANDWIDTH_TIME_PRODUCT
    @test @inferred(effective(5.0Hz, 5000 / 5e6Hz)) isa typeof(1.0Hz)
end

# The carrier bandwidth, wide or narrow, is capped by its stability product, not
# scaled by the block count.
@testset "carrier loop bandwidth is capped, not scaled, by the integration length" begin
    bw = 18.0Hz
    @test effective_carrier_loop_filter_bandwidth(bw, 1ms) == bw
    @test effective_carrier_loop_filter_bandwidth(bw, 4ms) == bw
    @test effective_carrier_loop_filter_bandwidth(bw, 5ms) ≈ bw
    @test effective_carrier_loop_filter_bandwidth(bw, 10ms) ≈ 9.0Hz
    @test effective_carrier_loop_filter_bandwidth(bw, 20ms) ≈ 4.5Hz
    for integration_time in (10ms, 20ms, 100ms, 1500ms)
        @test effective_carrier_loop_filter_bandwidth(bw, integration_time) *
              integration_time ≈ MAX_CARRIER_LOOP_BANDWIDTH_TIME_PRODUCT
    end
    # An explicit bandwidth below the cap is used verbatim.
    @test effective_carrier_loop_filter_bandwidth(2.0Hz, 20ms) == 2.0Hz
    @test @inferred(effective_carrier_loop_filter_bandwidth(bw, 5000 / 5e6Hz)) isa
          typeof(1.0Hz)
end

# The DLL bandwidth is an absolute value, not a per-primary-period reference: a
# longer coherent integration must not narrow it; only its own `BL·Δt`
# stability product may cap it.
@testset "code loop bandwidth is not narrowed by the integration length" begin
    bw = 1.0Hz
    l1ca_period = 1ms

    # The cap starts binding only past 18 ms.
    @test effective_code_loop_filter_bandwidth(bw, l1ca_period) == bw
    @test effective_code_loop_filter_bandwidth(bw, 10 * l1ca_period) == bw
    @test effective_code_loop_filter_bandwidth(bw, 20 * l1ca_period) ≈ 0.9Hz

    # Where it binds it holds the stability product, not a block ratio.
    for num_blocks in (20, 100, 1500)
        integration_time = num_blocks * l1ca_period
        @test effective_code_loop_filter_bandwidth(bw, integration_time) *
              integration_time ≈ MAX_CODE_LOOP_BANDWIDTH_TIME_PRODUCT
    end

    # An explicit bandwidth below the cap is used verbatim, at any length.
    @test effective_code_loop_filter_bandwidth(0.25Hz, 20 * l1ca_period) == 0.25Hz
end

@testset "Carrier and code frequency updates run the discriminators through the filter" begin
    gpsl1 = GPSL1CA()
    sampling_frequency = 5e6Hz
    integration_time = 1ms
    correlator = EarlyPromptLateCorrelator(
        SVector(1000.0 + 10im, 2000.0 + 20im, 750.0 + 10im),
        0.5,
    )
    previous_prompt = 1900.0 + 50im

    # FLL-assisted third-order PLL: fed the (PLL, FLL) discriminator pair.
    assisted = ThirdOrderAssistedBilinearLF()
    update_assisted, filter_assisted = @inferred calculate_carrier_frequency_update(
        gpsl1,
        assisted,
        correlator,
        previous_prompt,
        integration_time,
        18.0Hz,
    )
    expected = filter_loop(
        assisted,
        # The phase error in cycles, the FLL error in Hz.
        (
            pll_disc(gpsl1, correlator),
            fll_disc(gpsl1, correlator, previous_prompt, integration_time),
        ),
        integration_time,
        18.0Hz,
    )
    @test update_assisted == expected[1]
    @test filter_assisted == expected[2]

    # Any other loop filter: the PLL discriminator alone.
    plain = SecondOrderBilinearLF()
    update_plain, filter_plain = @inferred calculate_carrier_frequency_update(
        gpsl1,
        plain,
        correlator,
        previous_prompt,
        integration_time,
        18.0Hz,
    )
    expected = filter_loop(plain, pll_disc(gpsl1, correlator), integration_time, 18.0Hz)
    @test update_plain == expected[1]
    @test filter_plain == expected[2]
    @test update_plain != 0.0Hz

    # Code loop: the DLL discriminator through the code filter.
    code_filter = SecondOrderBilinearLF()
    update_code, filter_code = @inferred calculate_code_frequency_update(
        gpsl1,
        code_filter,
        correlator,
        0.0Hz,
        sampling_frequency,
        integration_time,
        1.0Hz,
    )
    expected = filter_loop(
        code_filter,
        dll_disc(gpsl1, correlator, 0.0Hz, sampling_frequency),
        integration_time,
        1.0Hz,
    )
    @test update_code == expected[1]
    @test filter_code == expected[2]
    @test update_code != 0.0Hz
end

@testset "The carrier loop has the configured noise bandwidth" begin
    # Closed loop with white phase noise σ_n: the NCO jitter must be
    # σ² = σ_n² · 2 · BL · T. Fed radians, the loop was ≈5.6× wider.
    gpsl1 = GPSL1CA()
    T = 1ms
    bandwidth = 18.0Hz
    σ_n = 0.05
    rng = MersenneTwister(1)
    loop_filter = ThirdOrderBilinearLF()
    φ = 0.0  # signal phase minus NCO phase, rad
    acc = 0.0
    n = 200_000
    settle = 1_000
    for i = 1:n
        prompt = cis(φ + σ_n * randn(rng))
        correlator = EarlyPromptLateCorrelator(SVector(prompt, prompt, prompt), 0.5)
        freq_update, loop_filter = calculate_carrier_frequency_update(
            gpsl1,
            loop_filter,
            correlator,
            prompt,
            T,
            bandwidth,
        )
        φ -= 2π * Float64(freq_update * T)
        i > settle && (acc += φ^2)
    end
    effective_bandwidth = acc / (n - settle) / σ_n^2 / (2 * ustrip(s, T))
    @test 0.85 < effective_bandwidth / ustrip(Hz, bandwidth) < 1.15
end

# A GPS L1 C/A record of `num_code_blocks` blocks whose taps all sit at the prompt `p`.
staging_record(p, num_code_blocks = 1) = LoopRecord(
    GPSL1CA(),
    EarlyPromptLateCorrelator(SVector(p / 2, p, p / 2), 0.5),
    complex(0.0),
    4000num_code_blocks,
    4000num_code_blocks,
    4000num_code_blocks,
    num_code_blocks,
    4e6Hz,
)

@testset "The phase-lock indicator reads cos 2φ, also at a low C/N₀" begin
    gpsl1 = GPSL1CA()
    # Its time constant: 0.1 s, at least 25 records, from the nominal record length.
    @test TrackingLoops._phase_lock_time_constant(gpsl1, 1) == 100
    @test TrackingLoops._phase_lock_time_constant(gpsl1, 20) == 25
    @test TrackingLoops._phase_lock_time_constant(GalileoE1B(), 1) == 25
    @test TrackingLoops._phase_lock_time_constant(GalileoE5aQP(), 31) == 50
    # A plain PLL's staging, whose count of locked records no FLL drop restarts.
    advance(staging, prompts) = foldl(
        (staging, p) -> TrackingLoops._advance_staging(staging, staging_record(p)),
        prompts;
        init = staging,
    )
    fresh = CarrierLoopStaging(ThirdOrderBilinearLF())
    # No reading until the averages span their time constant, 0.1 s of 1 ms records.
    @test isnan(advance(fresh, fill(complex(1.0), 99)).phase_lock_indicator)
    @test advance(fresh, fill(complex(1.0), 100)).phase_lock_indicator ≈ 1.0
    # A constant phase error φ reads cos 2φ.
    @test advance(fresh, fill(cis(π / 6), 200)).phase_lock_indicator ≈ cos(π / 3)
    # Locked at a low C/N₀ (30 dB-Hz: |P|²/σ² = 1 per 1 ms record), the
    # moments-based power still normalises it to about 1, if noisily, …
    rng = MersenneTwister(3)
    amplitude = sqrt(10^3.0 * 1e-3)
    locked = [advance(fresh, [amplitude + randn(rng, ComplexF64) for _ = 1:500]).phase_lock_indicator for _ = 1:50]
    @test 0.8 < median(locked) < 1.2
    @test count(>(0.5), locked) >= 45
    # … and a spinning phase about 0.
    spinning = [
        advance(fresh, [amplitude * cis(0.3k) + randn(rng, ComplexF64) for k = 1:500]).phase_lock_indicator for
        _ = 1:50
    ]
    @test abs(median(spinning)) < 0.1
    @test count(>(0.5), spinning) <= 2
    # For how many records it has read lock.
    held = advance(fresh, fill(complex(1.0), 300))
    @test held.num_locked_records == 201
end

@testset "The carrier loop's stages only advance" begin
    gpsl1 = GPSL1CA()
    staging = CarrierLoopStaging(ThirdOrderAssistedBilinearLF())
    @test staging.stage == FLL_ASSISTED_PLL
    @test isnan(staging.phase_lock_indicator)
    @test CarrierLoopStaging(ThirdOrderBilinearLF()).stage == WIDE_PLL
    stages = CarrierLoopStage[]
    for k = 1:1000
        staging = TrackingLoops._advance_staging(staging, staging_record(cis(0.1)))
        push!(stages, staging.stage)
    end
    # The first reading at the 100th record, after one time constant (0.1 s), the
    # FLL dropped once it has read lock for one time constant, the loop narrowed
    # four after that.
    @test findfirst(==(WIDE_PLL), stages) == 199
    @test findfirst(==(NARROW_PLL), stages) == 599
    @test issorted(stages)
    # Narrowed, the last stage, it no longer counts how long it has read lock.
    @test staging.phase_lock_indicator ≈ cos(0.2)
    @test staging.num_locked_records == 0
    # Lock lost, the narrow PLL stays.
    for k = 1:500
        staging = TrackingLoops._advance_staging(staging, staging_record(cis(0.3k)))
    end
    @test staging.phase_lock_indicator < 0.5
    @test staging.stage == NARROW_PLL
end
