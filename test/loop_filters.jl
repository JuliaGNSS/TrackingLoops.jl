# Loop bandwidth rules and the per-integration filter updates. Ported from
# the bandwidth parts of Tracking.jl's `test/conventional_pll_and_dll.jl`.

using Unitful: ms
using TrackingLoopFilters: filter_loop, ThirdOrderBilinearLF
using Random: MersenneTwister
using Statistics: median

@testset "Doppler aiding" begin
    gpsl1 = GPSL1CA()
    carrier_freq, code_freq = @inferred calc_carrier_aided_dopplers(gpsl1, 10Hz, 1Hz, 2Hz, -0.5Hz)
    @test carrier_freq == 10Hz + 2Hz
    @test code_freq == 1Hz + 2Hz / 1540 - 0.5Hz
end

@testset "Default loop bandwidths" begin
    # Flat bandwidths for every signal: the PLL's 50 Hz wide and 18 Hz narrow, the
    # FLL-assist 5 Hz and the code 1 Hz.
    for signal in (GPSL1CA(), GPSL1C_P(), GalileoE1B(), GPSL5I())
        @test default_wide_carrier_loop_bandwidth(signal) == 50.0Hz
        @test default_narrow_carrier_loop_bandwidth(signal) == 18.0Hz
        @test default_fll_assist_loop_bandwidth(signal) == 5.0Hz
        @test default_code_loop_bandwidth(signal) == 1.0Hz
    end
    @test default_narrow_carrier_loop_bandwidth(GPSL1CA()) isa typeof(1.0Hz)
    # A state's bandwidths: the signal's defaults, the generic ones without a signal.
    @test LoopBandwidths(GPSL1CA()) == LoopBandwidths() ==
          LoopBandwidths(50.0Hz, 18.0Hz, 5.0Hz, 1.0Hz)
    @test LoopBandwidths(GPSL1CA(); wide_carrier = 18.0Hz).wide_carrier == 18.0Hz
    @test LoopBandwidths(GPSL1CA(); wide_carrier = 18.0Hz).narrow_carrier == 18.0Hz
end

@testset "The FLL-assist bandwidth is capped by its own stability product" begin
    effective = calc_effective_fll_assist_loop_bandwidth
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
    @test calc_effective_carrier_loop_bandwidth(bw, 1ms) == bw
    @test calc_effective_carrier_loop_bandwidth(bw, 4ms) == bw
    @test calc_effective_carrier_loop_bandwidth(bw, 5ms) ≈ bw
    @test calc_effective_carrier_loop_bandwidth(bw, 10ms) ≈ 9.0Hz
    @test calc_effective_carrier_loop_bandwidth(bw, 20ms) ≈ 4.5Hz
    for integration_time in (10ms, 20ms, 100ms, 1500ms)
        @test calc_effective_carrier_loop_bandwidth(bw, integration_time) *
              integration_time ≈ MAX_CARRIER_LOOP_BANDWIDTH_TIME_PRODUCT
    end
    # An explicit bandwidth below the cap is used verbatim.
    @test calc_effective_carrier_loop_bandwidth(2.0Hz, 20ms) == 2.0Hz
    @test @inferred(calc_effective_carrier_loop_bandwidth(bw, 5000 / 5e6Hz)) isa
          typeof(1.0Hz)
end

# The DLL bandwidth is an absolute value, not a per-primary-period reference: a
# longer coherent integration must not narrow it; only its own `BL·Δt`
# stability product may cap it.
@testset "code loop bandwidth is not narrowed by the integration length" begin
    bw = 1.0Hz
    l1ca_period = 1ms

    # The cap starts binding only past 18 ms.
    @test calc_effective_code_loop_bandwidth(bw, l1ca_period) == bw
    @test calc_effective_code_loop_bandwidth(bw, 10 * l1ca_period) == bw
    @test calc_effective_code_loop_bandwidth(bw, 20 * l1ca_period) ≈ 0.9Hz

    # Where it binds it holds the stability product, not a block ratio.
    for num_blocks in (20, 100, 1500)
        integration_time = num_blocks * l1ca_period
        @test calc_effective_code_loop_bandwidth(bw, integration_time) *
              integration_time ≈ MAX_CODE_LOOP_BANDWIDTH_TIME_PRODUCT
    end

    # An explicit bandwidth below the cap is used verbatim, at any length.
    @test calc_effective_code_loop_bandwidth(0.25Hz, 20 * l1ca_period) == 0.25Hz
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
        correlator_bank = EarlyPromptLateCorrelatorBank(SVector(prompt, prompt, prompt), 0.5)
        freq_update, loop_filter =
            filter_loop(loop_filter, pll_disc(gpsl1, correlator_bank), T, bandwidth)
        φ -= 2π * Float64(freq_update * T)
        i > settle && (acc += φ^2)
    end
    effective_bandwidth = acc / (n - settle) / σ_n^2 / (2 * ustrip(s, T))
    @test 0.85 < effective_bandwidth / ustrip(Hz, bandwidth) < 1.15
end

# A GPS L1 C/A integration of `num_code_blocks` blocks whose correlators all sit at the prompt `p`.
staging_integration(p, num_code_blocks = 1) = LoopRecord(
    GPSL1CA(),
    EarlyPromptLateCorrelatorBank(SVector(p / 2, p, p / 2), 0.5),
    complex(0.0),
    4000num_code_blocks,
    4000num_code_blocks,
    4000num_code_blocks,
    num_code_blocks,
    4e6Hz,
)

@testset "The phase-lock indicator reads cos 2φ, also at a low C/N₀" begin
    gpsl1 = GPSL1CA()
    # Its time constant: 0.1 s, at least 25 integrations, from the nominal integration length.
    @test TrackingLoops._phase_lock_time_constant(gpsl1, 1) == 100
    @test TrackingLoops._phase_lock_time_constant(gpsl1, 20) == 25
    @test TrackingLoops._phase_lock_time_constant(GalileoE1B(), 1) == 25
    @test TrackingLoops._phase_lock_time_constant(GalileoE5aQP(), 31) == 50
    # A plain PLL's staging, whose count of locked integrations no FLL drop restarts.
    advance(staging, prompts) = foldl(
        (staging, p) -> TrackingLoops._advance_staging(staging, staging_integration(p)),
        prompts;
        init = staging,
    )
    fresh = TrackingLoops.CarrierLoopStaging(ThirdOrderBilinearLF)
    # No reading until the averages span their time constant, 0.1 s of 1 ms integrations.
    @test isnan(advance(fresh, fill(complex(1.0), 99)).lock_detector.indicator)
    @test advance(fresh, fill(complex(1.0), 100)).lock_detector.indicator ≈ 1.0
    # A constant phase error φ reads cos 2φ.
    @test advance(fresh, fill(cis(π / 6), 200)).lock_detector.indicator ≈ cos(π / 3)
    # Locked at a low C/N₀ (30 dB-Hz: |P|²/σ² = 1 per 1 ms integration), the
    # moments-based power still normalises it to about 1, if noisily, …
    rng = MersenneTwister(3)
    amplitude = sqrt(10^3.0 * 1e-3)
    locked = [advance(fresh, [amplitude + randn(rng, ComplexF64) for _ = 1:500]).lock_detector.indicator for _ = 1:50]
    @test 0.8 < median(locked) < 1.2
    @test count(>(0.5), locked) >= 45
    # … and a spinning phase about 0.
    spinning = [
        advance(fresh, [amplitude * cis(0.3k) + randn(rng, ComplexF64) for k = 1:500]).lock_detector.indicator for
        _ = 1:50
    ]
    @test abs(median(spinning)) < 0.1
    @test count(>(0.5), spinning) <= 2
    # For how many integrations it has read lock.
    held = advance(fresh, fill(complex(1.0), 300))
    @test held.num_locked_integrations == 201
end

@testset "A component's prompt is de-rotated against the driver's carrier phase" begin
    derotation(driver, signal) =
        TrackingLoops._carrier_phase_derotation(get_carrier_phase_offset(driver), signal)
    # A quadrature component turns by ∓90° onto the real axis.
    @test derotation(GPSL5Q(), GPSL5I()) ≈ -im
    @test real(derotation(GPSL5Q(), GPSL5I()) * 5im) ≈ 5
    @test derotation(GalileoE5aQ(), GalileoE5aI()) ≈ im
    @test derotation(GPSL5I(), GPSL5Q()) ≈ im
    @test real(derotation(GPSL5I(), GPSL5Q()) * -5im) ≈ 5
    @test derotation(GalileoE5aI(), GalileoE5aQ()) ≈ -im
    # The driver itself and an in-phase component are left exactly as they are.
    for (driver, signal) in (
        (GPSL5Q(), GPSL5Q()),
        (GalileoE5aQ(), GalileoE5aQ()),
        (GPSL5I(), GPSL5I()),
        (GPSL1CA(), GPSL1CA()),
        (GalileoE1C(), GalileoE1B()),
    )
        @test derotation(driver, signal) === 1.0 + 0.0im
    end
end

@testset "The carrier loop's stages only advance" begin
    gpsl1 = GPSL1CA()
    staging = TrackingLoops.CarrierLoopStaging(ThirdOrderAssistedBilinearLF)
    @test staging.stage == FLL_ASSISTED_PLL
    @test isnan(staging.lock_detector.indicator)
    @test TrackingLoops.CarrierLoopStaging(ThirdOrderBilinearLF).stage == WIDE_PLL
    stages = CarrierLoopStage[]
    for k = 1:1000
        staging = TrackingLoops._advance_staging(staging, staging_integration(cis(0.1)))
        push!(stages, staging.stage)
    end
    # The first reading at the 100th integration, after one time constant (0.1 s), the
    # FLL dropped once it has read lock for one time constant, the loop narrowed
    # four after that.
    @test findfirst(==(WIDE_PLL), stages) == 199
    @test findfirst(==(NARROW_PLL), stages) == 599
    @test issorted(stages)
    # Narrowed, the last stage, it no longer counts how long it has read lock.
    @test staging.lock_detector.indicator ≈ cos(0.2)
    @test staging.num_locked_integrations == 0
    # Lock lost, the narrow PLL stays.
    for k = 1:500
        staging = TrackingLoops._advance_staging(staging, staging_integration(cis(0.3k)))
    end
    @test staging.lock_detector.indicator < 0.5
    @test staging.stage == NARROW_PLL
end
