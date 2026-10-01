# Loop-filter bandwidth rules and the per-record filter updates. Ported from
# the bandwidth parts of Tracking.jl's `test/conventional_pll_and_dll.jl`.

using Unitful: ms
using TrackingLoopFilters: filter_loop

@testset "Doppler aiding" begin
    gpsl1 = GPSL1CA()
    carrier_freq, code_freq = @inferred aid_dopplers(gpsl1, 10Hz, 1Hz, 2Hz, -0.5Hz)
    @test carrier_freq == 10Hz + 2Hz
    @test code_freq == 1Hz + 2Hz / 1540 - 0.5Hz
end

@testset "Default loop bandwidths" begin
    # Carrier: `BL·T = 0.018` against the primary code period.
    @test default_carrier_loop_filter_bandwidth(GPSL1CA()) ≈ 18.0Hz
    @test default_carrier_loop_filter_bandwidth(GPSL5I()) ≈ 18.0Hz
    @test default_carrier_loop_filter_bandwidth(GPSL1C_P()) ≈ 1.8Hz
    @test default_carrier_loop_filter_bandwidth(GalileoE1B()) ≈ 4.5Hz
    @test default_carrier_loop_filter_bandwidth(GPSL1CA()) isa typeof(1.0Hz)
    # Code: a flat 1 Hz for every signal.
    for signal in (GPSL1CA(), GPSL1C_P(), GalileoE1B(), GPSL5I())
        @test default_code_loop_filter_bandwidth(signal) == 1.0Hz
    end
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
              integration_time ≈ MAX_LOOP_BANDWIDTH_TIME_PRODUCT
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
