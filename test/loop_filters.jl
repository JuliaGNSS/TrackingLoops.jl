# Loop-filter bandwidth rules and the per-record filter updates. Ported from
# the bandwidth parts of Tracking.jl's `test/conventional_pll_and_dll.jl`.

using Unitful: ms
using TrackingLoopFilters: filter_loop, ThirdOrderBilinearLF
using Random: MersenneTwister

@testset "Doppler aiding" begin
    gpsl1 = GPSL1CA()
    carrier_freq, code_freq = @inferred aid_dopplers(gpsl1, 10Hz, 1Hz, 2Hz, -0.5Hz)
    @test carrier_freq == 10Hz + 2Hz
    @test code_freq == 1Hz + 2Hz / 1540 - 0.5Hz
end

@testset "Default loop bandwidths" begin
    # A flat 50 Hz pull-in, 18 Hz tracking and 5 Hz FLL carrier and 1 Hz code bandwidth for every signal.
    for signal in (GPSL1CA(), GPSL1C_P(), GalileoE1B(), GPSL5I())
        @test default_wide_carrier_loop_filter_bandwidth(signal) == 50.0Hz
        @test default_code_loop_filter_bandwidth(signal) == 1.0Hz
    end
    @test default_wide_carrier_loop_filter_bandwidth(GPSL1CA()) isa typeof(1.0Hz)
end

# The carrier bandwidth is capped by its stability product, not scaled by the
# block count.
@testset "carrier loop bandwidth is capped, not scaled, by the integration length" begin
    bw = 18.0Hz
    @test wide_cap(bw, 1ms) == bw
    @test wide_cap(bw, 4ms) == bw
    @test wide_cap(bw, 5ms) ≈ bw
    @test wide_cap(bw, 10ms) ≈ 9.0Hz
    @test wide_cap(bw, 20ms) ≈ 4.5Hz
    for integration_time in (10ms, 20ms, 100ms, 1500ms)
        @test wide_cap(bw, integration_time) *
              integration_time ≈ TrackingLoops._MAX_WIDE_CARRIER_LOOP_BANDWIDTH_TIME_PRODUCT
    end
    # An explicit bandwidth below the cap is used verbatim.
    @test wide_cap(2.0Hz, 20ms) == 2.0Hz
    @test @inferred(wide_cap(bw, 5000 / 5e6Hz)) isa
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
    # Its FLL path at the signal's FLL-assist default, as `step_loop` runs it.
    discriminators = (
        pll_disc(gpsl1, correlator),
        fll_disc(gpsl1, correlator, previous_prompt, integration_time),
    )
    fll_assist = TrackingLoops._capped_bandwidth(
        default_fll_assist_loop_filter_bandwidth(gpsl1),
        integration_time,
        TrackingLoops._MAX_FLL_ASSIST_LOOP_BANDWIDTH_TIME_PRODUCT,
    )
    # The phase error in cycles, the FLL error in Hz.
    expected = filter_loop(assisted, discriminators, integration_time, (18.0Hz, fll_assist))
    @test update_assisted == expected[1]
    @test filter_assisted == expected[2]
    explicit, = calculate_carrier_frequency_update(gpsl1, assisted, correlator, previous_prompt,
        integration_time, 18.0Hz; fll_assist_loop_bandwidth = 2.0Hz)
    @test explicit == filter_loop(assisted, discriminators, integration_time, (18.0Hz, 2.0Hz))[1]

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
