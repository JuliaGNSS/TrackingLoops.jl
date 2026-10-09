"""
$(SIGNATURES)

Code phase error in chips from the noncoherent early-minus-late envelope normalized
discriminator `(2 - d) / 2 * (E - L) / (E + L)` for early-late spacing `d` in chips
(`1/2 * (E - L) / (E + L)` at 1 chip).

See: Kaplan & Hegarty, "Understanding GPS: Principles and Applications", 2nd ed.,
Table 5.5; GNSS-SDR tracking_discriminators.cc.
"""
function dll_disc(
    signal::AbstractGNSSSignal,
    correlator::EarlyPromptLateCorrelator,
    code_doppler,
    sampling_frequency,
)
    code_frequency = code_doppler + get_code_frequency(signal)
    code_phase_delta = code_frequency / sampling_frequency
    E = abs(get_early(correlator))
    L = abs(get_late(correlator))
    distance_between_early_and_late =
        get_early_late_sample_spacing(correlator, sampling_frequency, code_frequency) *
        code_phase_delta
    (2 - distance_between_early_and_late) / 2 * (E - L) / (E + L)
end

# Piecewise-linear sine-BOC(1,1) autocorrelation envelope |R(τ)| (infinite bandwidth,
# long-code approximation): the main peak falls from 1 with slope −3 to the zero crossing
# at 1/3 chips, the (negative) side lobe's envelope rises with slope +3 to 1/2 at
# 0.5 chips and falls with slope −1 to zero at 1 chip.
_boc11_envelope(offset) =
    offset < 1 / 3 ? 1 - 3 * offset :
    offset < 1 / 2 ? 3 * offset - 1 : offset < 1 ? 1 - offset : 0.0

# Mean of the one-sided derivatives, so a tap exactly on a knot (e.g. the side-lobe
# vertex at 0.5 chips under coarse sampling) gets the slope its ± excursions produce.
function _boc11_envelope_slope(offset)
    right = offset < 1 / 3 ? -3.0 : offset < 1 / 2 ? 3.0 : offset < 1 ? -1.0 : 0.0
    left = offset <= 1 / 3 ? -3.0 : offset <= 1 / 2 ? 3.0 : offset <= 1 ? -1.0 : 0.0
    (left + right) / 2
end

# VEML S-curve slope at the origin for inner and outer tap offsets in chips: each tap
# pair contributes −2·slope·τ to the numerator and 2·|R| to the denominator.
_veml_discriminator_slope(inner_offset, outer_offset) =
    -(_boc11_envelope_slope(inner_offset) + _boc11_envelope_slope(outer_offset)) /
    (_boc11_envelope(inner_offset) + _boc11_envelope(outer_offset))

"""
$(SIGNATURES)

Code phase error in chips from the noncoherent very-early-minus-late envelope
normalized discriminator `(VE + E - VL - L) / (VE + E + VL + L)` for BOC(1,1)-dominant
signals (Galileo E1, GPS L1C), divided by its S-curve slope so the output is in chips.
The slope (`4 / (2 − 3·0.15 − 0.6) ≈ 4.2` for the default ±0.15/±0.6 chip taps) is
evaluated on the piecewise-linear sine-BOC(1,1) envelope at the sample-quantized tap
offsets, as GNSS-SDR's `CalculateSlopeAbs` on `SinBocCorrelationFunction` does.
Against full CBOC/TMBOC a residual gain error remains; the discriminator is linear
only within the inner tap offset.

Throws an `ArgumentError` if both quantized tap offsets are one chip or more (slope
undefined), which a coarse sampling rate can cause for shifts just below one chip.

Raw discriminator form from GNSS-SDR's Galileo E1 DLL/PLL VEML tracking:
<https://gnss-sdr.org/docs/sp-blocks/tracking/#implementation-galileo_e1_dll_pll_veml_tracking>
"""
function dll_disc(
    signal::AbstractGNSSSignal,
    correlator::VeryEarlyPromptLateCorrelator,
    code_doppler,
    sampling_frequency,
)
    code_frequency = code_doppler + get_code_frequency(signal)
    code_phase_delta = upreferred(code_frequency / sampling_frequency)
    inner_offset =
        calc_preferred_code_shift_to_sample_shift(
            correlator.preferred_early_late_to_prompt_code_shift,
            sampling_frequency,
            code_frequency,
        ) * code_phase_delta
    outer_offset =
        calc_preferred_code_shift_to_sample_shift(
            correlator.preferred_very_early_late_to_prompt_code_shift,
            sampling_frequency,
            code_frequency,
        ) * code_phase_delta
    # Constant message: this path is compiled with `juliac --trim`.
    min(inner_offset, outer_offset) < 1 || throw(
        ArgumentError(
            "VEML dll_disc: both tap offsets are one chip or more at this sampling " *
            "frequency, where the BOC(1,1) correlation peak has vanished; " *
            "use code shifts below one chip.",
        ),
    )
    slope = _veml_discriminator_slope(inner_offset, outer_offset)
    VE = abs(get_very_early(correlator))
    E = abs(get_early(correlator))
    L = abs(get_late(correlator))
    VL = abs(get_very_late(correlator))
    raw = (VE + E - VL - L) / (VE + E + VL + L)
    # A locally flat S-curve cannot be calibrated; functional layouts have slope > 0.
    slope == 0 ? raw : raw / slope
end

"""
$(SIGNATURES)

Carrier phase error in cycles (see [`calculate_carrier_frequency_update`](@ref)).
With `polarity = 0` (default) it is the two-quadrant Costas discriminator
`atan(Q / I) / 2π`: insensitive to data or secondary-code sign flips, range ±1/4
cycle. With the prompt's sign as `polarity` (±1) it is the four-quadrant
`atan(Q, I) / 2π` of `polarity * prompt`, range ±1/2 cycle, worth up to 6 dB of
tracking threshold. A wrong `polarity` reads as half a cycle of error, so pass ±1
only for a dataless signal, synced to its secondary code where it has one.

See: Kaplan & Hegarty, "Understanding GPS: Principles and Applications", 2nd ed.,
Tables 5.2 and 5.3.
"""
function pll_disc(signal::AbstractGNSSSignal, correlator; polarity::Real = 0)
    p = get_prompt(correlator)
    iszero(polarity) && return atan(imag(p) / real(p)) / 2π
    q = polarity * p
    atan(imag(q), real(q)) / 2π
end

"""
$(SIGNATURES)

Carrier frequency error in `Hz`, whatever unit the integration time carries, from
the rotation between the previous and current prompt; zero without a previous
prompt. By default the two-quadrant
`atan(cross / dot)`, insensitive to a sign flip between the prompts, range
±1 / (4 · `integration_time`). With `four_quadrant = true` the four-quadrant
`atan(cross, dot)`, range ±1 / (2 · `integration_time`); it needs both prompts to
share their sign (either one), as for [`pll_disc`](@ref)'s `polarity`.

See: Kaplan & Hegarty, "Understanding GPS: Principles and Applications", 2nd ed.,
Table 5.4.
"""
function fll_disc(
    signal::AbstractGNSSSignal,
    correlator,
    previous_prompt,
    integration_time;
    four_quadrant::Bool = false,
)
    if previous_prompt == 0
        return uconvert(Hz, 0.0/integration_time)
    end

    current_prompt = get_prompt(correlator)

    result = conj(previous_prompt) * current_prompt
    cross = imag(result)
    dot = real(result)

    # Two-quadrant: `cross / dot` is ±Inf where `dot` is zero, which `atan` maps to ±π/2.
    rotation = four_quadrant ? atan(cross, dot) : atan(cross / dot)
    return uconvert(Hz, rotation / (2 * pi * integration_time))
end
