# Discriminators. Ported from Tracking.jl's `test/discriminators.jl`.

using Unitful: ms, MHz
using TrackingLoops: _boc11_envelope, _boc11_envelope_slope, _veml_discriminator_slope

# A correlator whose taps all sit at the prompt `p`.
_prompt_correlator(p) = EarlyPromptLateCorrelator(SVector(p / 2, p, p / 2), 0.5)

@testset "PLL discriminator" begin
    correlator_minus60off = EarlyPromptLateCorrelator(
        SVector(-0.5 + sqrt(3) / 2im, -1 + sqrt(3) * 1im, -0.5 + sqrt(3) / 2im),
        0.5,
    )
    correlator_0off =
        EarlyPromptLateCorrelator(SVector(0.5 + 0.0im, 1 + 0.0im, 0.5 + 0.0im), 0.5)
    correlator_plus60off = EarlyPromptLateCorrelator(
        SVector(0.5 + sqrt(3) / 2im, 1 + sqrt(3) * 1im, 0.5 + sqrt(3) / 2im),
        0.5,
    )
    gpsl1 = GPSL1CA()
    @test @inferred(pll_disc(gpsl1, correlator_minus60off)) == -π / 3  #-60°
    @test @inferred(pll_disc(gpsl1, correlator_0off)) == 0
    @test @inferred(pll_disc(gpsl1, correlator_plus60off)) == π / 3  #+60°

    # With the prompt's sign given, the four-quadrant discriminator agrees with
    # the Costas one within ±π/2; beyond, the Costas one folds the error by π
    # while it does not.
    for correlator in (correlator_0off, correlator_plus60off)
        @test @inferred(pll_disc(gpsl1, correlator; polarity = 1)) ≈
              pll_disc(gpsl1, correlator)
    end
    # An unknown sign (0) reads with the Costas discriminator.
    @test @inferred(pll_disc(gpsl1, correlator_minus60off; polarity = 0)) ==
          pll_disc(gpsl1, correlator_minus60off)
    # `correlator_minus60off`'s prompt sits at +120°, which Costas reads as -60°.
    @test @inferred(pll_disc(gpsl1, correlator_minus60off; polarity = 1)) ≈ 2π / 3
    @test pll_disc(gpsl1, _prompt_correlator(cis(-2π / 3)); polarity = 1) ≈ -2π / 3
    @test pll_disc(gpsl1, _prompt_correlator(-1.0 + 0.0im)) == 0
    @test pll_disc(gpsl1, _prompt_correlator(-1.0 + 0.0im); polarity = 1) ≈ π
    # A negative polarity reads the error around the inverted prompt.
    @test @inferred(pll_disc(gpsl1, _prompt_correlator(-cis(0.2)); polarity = -1)) ≈ 0.2
    @test pll_disc(gpsl1, _prompt_correlator(-cis(2.5)); polarity = -1) ≈ 2.5
    @test pll_disc(gpsl1, _prompt_correlator(-cis(0.2)); polarity = 1.0) ≈ 0.2 - π
end

@testset "FLL discriminator" begin
    correlator_minus60off = EarlyPromptLateCorrelator(
        SVector(-0.5 + sqrt(3) / 2im, -1 + sqrt(3) * 1im, -0.5 + sqrt(3) / 2im),
        0.5,
    )
    correlator_0off =
        EarlyPromptLateCorrelator(SVector(0.5 + 0.0im, 1 + 0.0im, 0.5 + 0.0im), 0.5)
    correlator_plus60off = EarlyPromptLateCorrelator(
        SVector(0.5 + sqrt(3) / 2im, 1 + sqrt(3) * 1im, 0.5 + sqrt(3) / 2im),
        0.5,
    )
    correlator_plus90off =
        EarlyPromptLateCorrelator(SVector(0.0 + 0.5im, 0.0 + 1.0im, 0.0 + 0.5im), 0.5)
    correlator_empty = EarlyPromptLateCorrelator()
    gpsl1 = GPSL1CA()
    @test @inferred(
        fll_disc(gpsl1, correlator_0off, get_prompt(correlator_minus60off), 1ms)
    ) == (166 + 2 / 3) * 1Hz
    @test @inferred(fll_disc(gpsl1, correlator_0off, get_prompt(correlator_0off), 1ms)) ==
          0.0Hz
    @test @inferred(
        fll_disc(gpsl1, correlator_0off, get_prompt(correlator_plus60off), 1ms)
    ) == -(166 + 2 / 3) * 1Hz
    @test @inferred(
        fll_disc(gpsl1, correlator_0off, get_prompt(correlator_plus90off), 1ms)
    ) == -250Hz
    # No previous prompt yet: no frequency error.
    @test @inferred(fll_disc(gpsl1, correlator_0off, get_prompt(correlator_empty), 1ms)) ==
          0Hz
    # An integration time from a sampling frequency in MHz still yields Hz.
    in_mhz = @inferred fll_disc(
        gpsl1,
        correlator_0off,
        get_prompt(correlator_minus60off),
        5000 / 5.0MHz,
    )
    @test in_mhz isa typeof(1.0Hz)
    @test in_mhz ≈ (166 + 2 / 3) * 1Hz
    @test fll_disc(gpsl1, correlator_0off, get_prompt(correlator_empty), 5000 / 5.0MHz) isa
          typeof(1.0Hz)
    # The four-quadrant discriminator pulls in over ±1 / (2T) = ±500 Hz instead
    # of ±250 Hz; within ±250 Hz the two agree. `correlator_minus60off`'s prompt
    # sits at +120°, an advance of -120° to `correlator_0off`'s.
    @test @inferred(
        fll_disc(
            gpsl1,
            correlator_0off,
            get_prompt(correlator_minus60off),
            1ms;
            four_quadrant = true,
        )
    ) ≈ -(333 + 1 / 3) * 1Hz
    @test fll_disc(gpsl1, correlator_0off, cis(-2π / 3), 1ms) ≈ -(166 + 2 / 3) * 1Hz
    @test fll_disc(gpsl1, correlator_0off, cis(-2π / 3), 1ms; four_quadrant = true) ≈
          (333 + 1 / 3) * 1Hz
    @test fll_disc(gpsl1, correlator_0off, cis(2π / 3), 1ms; four_quadrant = true) ≈
          -(333 + 1 / 3) * 1Hz
    @test abs(fll_disc(gpsl1, correlator_0off, -1.0 + 0.0im, 1ms; four_quadrant = true)) ≈
          500Hz
    # Which sign the two prompts share cancels.
    @test fll_disc(
        gpsl1,
        _prompt_correlator(-1.0 + 0.0im),
        -cis(-2π / 3),
        1ms;
        four_quadrant = true,
    ) ≈ (333 + 1 / 3) * 1Hz
    @test fll_disc(
        gpsl1,
        correlator_0off,
        get_prompt(correlator_empty),
        1ms;
        four_quadrant = true,
    ) == 0Hz
    # In Hz whatever the unit of the integration time.
    @test @inferred(
        fll_disc(gpsl1, correlator_0off, cis(-2π / 3), 1e-3 / Hz; four_quadrant = true)
    ) isa typeof(1.0Hz)
end

@testset "DLL discriminator" begin
    gpsl1 = GPSL1CA()
    sampling_frequency = get_code_frequency(gpsl1) * 4

    very_early_correlator =
        EarlyPromptLateCorrelator(SVector(1.0 + 0.0im, 0.5 + 0.0im, 0.0 + 0.0im), 0.5)
    early_correlator =
        EarlyPromptLateCorrelator(SVector(0.75 + 0.0im, 0.75 + 0.0im, 0.25 + 0.0im), 0.5)
    prompt_correlator =
        EarlyPromptLateCorrelator(SVector(0.5 + 0.0im, 1 + 0.0im, 0.5 + 0.0im), 0.5)
    late_correlator =
        EarlyPromptLateCorrelator(SVector(0.25 + 0.0im, 0.75 + 0.0im, 0.75 + 0.0im), 0.5)
    very_late_correlator =
        EarlyPromptLateCorrelator(SVector(0.0 + 0.0im, 0.5 + 0.0im, 1.0 + 0.0im), 0.5)

    @test @inferred(
        get_early_late_sample_spacing(
            prompt_correlator,
            sampling_frequency,
            get_code_frequency(gpsl1),
        )
    ) == 4

    @test @inferred(dll_disc(gpsl1, very_early_correlator, 0.0Hz, sampling_frequency)) ==
          -0.5
    @test @inferred(dll_disc(gpsl1, early_correlator, 0.0Hz, sampling_frequency)) == -0.25
    @test @inferred(dll_disc(gpsl1, prompt_correlator, 0.0Hz, sampling_frequency)) == 0
    @test @inferred(dll_disc(gpsl1, late_correlator, 0.0Hz, sampling_frequency)) == 0.25
    @test @inferred(dll_disc(gpsl1, very_late_correlator, 0.0Hz, sampling_frequency)) == 0.5
end

@testset "BOC(1,1) envelope model of the VEML calibration" begin
    # Main peak: 1 at the origin, zero crossing at 1/3 chip; side-lobe vertex of
    # 1/2 at half a chip; zero from one chip on.
    @test _boc11_envelope(0.0) == 1.0
    @test _boc11_envelope(1 / 6) ≈ 0.5
    @test _boc11_envelope(0.5) ≈ 0.5
    @test _boc11_envelope(0.75) ≈ 0.25
    @test _boc11_envelope(1.5) == 0.0
    # Slopes are the mean of the one-sided derivatives, so a knot gets the mean.
    @test _boc11_envelope_slope(0.1) == -3.0
    @test _boc11_envelope_slope(1 / 3) == 0.0
    @test _boc11_envelope_slope(0.4) == 3.0
    @test _boc11_envelope_slope(0.5) == 1.0
    @test _boc11_envelope_slope(0.75) == -1.0
    @test _boc11_envelope_slope(1.0) == -0.5
    @test _boc11_envelope_slope(2.0) == 0.0
    # For the default ±0.15/±0.6 taps: 4 / (2 − 3·0.15 − 0.6).
    @test _veml_discriminator_slope(0.15, 0.6) ≈ 4 / (2 - 3 * 0.15 - 0.6)
    # A flat tap layout: the inner taps' falling and the outer taps' rising
    # envelope cancel.
    @test _veml_discriminator_slope(0.1, 0.4) == 0.0
end

@testset "VEML discriminator" begin
    signal = GalileoE1B()
    sampling_frequency = 20MHz
    # Symmetric taps: no code error.
    centred = update_accumulator(
        VeryEarlyPromptLateCorrelator(),
        SVector{5,ComplexF64}(0.2, 0.7, 1.0, 0.7, 0.2),
    )
    @test @inferred(dll_disc(signal, centred, 0.0Hz, sampling_frequency)) == 0.0
    # More energy on the early side (higher index) reads a positive error.
    early_heavy = update_accumulator(
        VeryEarlyPromptLateCorrelator(),
        SVector{5,ComplexF64}(0.1, 0.6, 1.0, 0.8, 0.3),
    )
    late_heavy = update_accumulator(
        VeryEarlyPromptLateCorrelator(),
        SVector{5,ComplexF64}(0.3, 0.8, 1.0, 0.6, 0.1),
    )
    e = dll_disc(signal, early_heavy, 0.0Hz, sampling_frequency)
    l = dll_disc(signal, late_heavy, 0.0Hz, sampling_frequency)
    @test e ≈ -l
    @test e > 0
    # A tap layout whose S-curve is flat (inner taps on the falling main peak,
    # outer taps on the rising side lobe, ±2/±8 samples at 20 MHz) cannot be
    # calibrated and returns the raw ratio instead of dividing by zero.
    flat = update_accumulator(
        VeryEarlyPromptLateCorrelator(;
            preferred_early_late_to_prompt_code_shift = 0.1,
            preferred_very_early_late_to_prompt_code_shift = 0.4,
        ),
        SVector{5,ComplexF64}(0.1, 0.6, 1.0, 0.8, 0.3),
    )
    @test dll_disc(signal, flat, 0.0Hz, sampling_frequency) ≈
          (0.3 + 0.8 - 0.1 - 0.6) / (0.3 + 0.8 + 0.1 + 0.6)
end

@testset "VEML discriminator refuses taps quantized off the correlation peak" begin
    signal = GalileoE1B()
    correlator = update_accumulator(
        VeryEarlyPromptLateCorrelator(),
        SVector{5,ComplexF64}(0.2, 0.7, 1.0, 0.7, 0.2),
    )
    # Sampled at the code rate, a sample is a chip: the default ±0.15/±0.6 chip
    # taps both round to one sample, i.e. one chip, where the envelope is zero.
    @test_throws ArgumentError dll_disc(signal, correlator, 0.0Hz, 1.023MHz)
    # Twice the code rate puts the taps at half a chip: finite, and zero for
    # symmetric taps.
    @test dll_disc(signal, correlator, 0.0Hz, 2.046MHz) == 0
end

# A chips-calibrated discriminator has an S-curve slope of 1 at the origin: sweep
# a true code offset τ over the linear region, build the accumulators by
# correlating the (noiseless) code against tap replicas at the correlator's
# actual sample-quantized offsets, and least-squares fit `dll_disc`'s output
# against τ.
function dll_disc_s_curve_slope(
    signal,
    correlator,
    sampling_frequency;
    prn = 1,
    offsets = -0.05:0.005:0.05,
)
    code_frequency = get_code_frequency(signal)
    chips_per_sample = upreferred(code_frequency / sampling_frequency)
    samples_per_code = round(Int, get_code_length(signal) / chips_per_sample)
    shifts = get_correlator_sample_shifts(correlator, sampling_frequency, code_frequency)
    phases = (0:(samples_per_code-1)) .* chips_per_sample
    discriminator = map(offsets) do offset
        incoming = get_code.(signal, phases .+ offset, prn)
        accumulators = map(shifts) do shift
            replica = get_code.(signal, phases .+ shift * chips_per_sample, prn)
            complex(sum(incoming .* replica) / samples_per_code, 0.0)
        end
        dll_disc(
            signal,
            update_accumulator(correlator, SVector(accumulators)),
            0.0Hz,
            sampling_frequency,
        )
    end
    sum(offsets .* discriminator) / sum(abs2, offsets)
end

@testset "DLL discriminator S-curve slope is 1 (calibrated in chips)" begin
    veml = VeryEarlyPromptLateCorrelator()
    # EPL on GPS L1 C/A: the (2 - d) / 2 normalization.
    @test dll_disc_s_curve_slope(GPSL1CA(), EarlyPromptLateCorrelator(), 20MHz) ≈ 1.0 atol =
        0.02
    # VEML on BOC(1,1) — the normalization's own correlation model — at two
    # sampling rates that quantize the taps differently.
    @test dll_disc_s_curve_slope(GalileoE1B_BOC11(), veml, 20MHz) ≈ 1.0 atol = 0.02
    @test dll_disc_s_curve_slope(GalileoE1B_BOC11(), veml, 79.5MHz) ≈ 1.0 atol = 0.02
    # VEML on the full modulations: the residual gain error is the modulation
    # mismatch against the BOC(1,1) model.
    @test dll_disc_s_curve_slope(GalileoE1B(), veml, 20MHz) ≈ 1.0 atol = 0.1
    @test dll_disc_s_curve_slope(GPSL1C_P(), veml, 20MHz) ≈ 1.0 atol = 0.1
    @test dll_disc_s_curve_slope(GalileoE1C(), veml, 20MHz) ≈ 1.0 atol = 0.2
end
