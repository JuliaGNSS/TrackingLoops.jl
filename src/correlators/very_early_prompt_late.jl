"""
$(SIGNATURES)

Correlator bank with very-early, early, prompt, late and very-late correlators.

`preferred_early_late_to_prompt_spacing` is the requested distance in chips from
prompt to each early/late correlator (`0.15` by default).
`preferred_very_early_late_to_prompt_spacing` is the distance from prompt to each
very-early/very-late correlator (`0.6` by default). These are distances from prompt,
not the full separation of either pair. Each is rounded to whole samples as for
[`EarlyPromptLateCorrelatorBank`](@ref).
"""
struct VeryEarlyPromptLateCorrelatorBank{M,T} <: AbstractEarlyPromptLateCorrelatorBank{M}
    correlator_outputs::SVector{5,T}
    preferred_early_late_to_prompt_spacing::Float64
    preferred_very_early_late_to_prompt_spacing::Float64
    function VeryEarlyPromptLateCorrelatorBank(
        correlator_outputs::AbstractVector{SVector{M,T}},
        preferred_early_late_to_prompt_spacing,
        preferred_very_early_late_to_prompt_spacing,
    ) where {M,T<:Complex}
        new{M,SVector{M,T}}(
            correlator_outputs,
            preferred_early_late_to_prompt_spacing,
            preferred_very_early_late_to_prompt_spacing,
        )
    end
    function VeryEarlyPromptLateCorrelatorBank(
        correlator_outputs::AbstractVector{T},
        preferred_early_late_to_prompt_spacing,
        preferred_very_early_late_to_prompt_spacing,
    ) where {T<:Complex}
        new{1,T}(
            correlator_outputs,
            preferred_early_late_to_prompt_spacing,
            preferred_very_early_late_to_prompt_spacing,
        )
    end
end

"""
$(SIGNATURES)

Construct a zeroed [`VeryEarlyPromptLateCorrelatorBank`](@ref); default spacings from
GNSS-SDR's Galileo E1 VEML tracking:
<https://gnss-sdr.org/docs/sp-blocks/tracking/#implementation-galileo_e1_dll_pll_veml_tracking>

Throws an `ArgumentError` if both spacings from prompt are one chip or more, where the
BOC(1,1) peak the VEML [`dll_disc`](@ref) is calibrated on has vanished.
"""
function VeryEarlyPromptLateCorrelatorBank(;
    num_antennas::NumAntennas = NumAntennas(1),
    preferred_early_late_to_prompt_spacing = 0.15,
    preferred_very_early_late_to_prompt_spacing = 0.6,
)
    min(
        preferred_early_late_to_prompt_spacing,
        preferred_very_early_late_to_prompt_spacing,
    ) < 1 || throw(
        ArgumentError(
            "VeryEarlyPromptLateCorrelatorBank: the early/late-to-prompt spacing " *
            "($preferred_early_late_to_prompt_spacing chips) or the very-early/very-late-to-prompt " *
            "spacing ($preferred_very_early_late_to_prompt_spacing chips) must be " *
            "below one chip; the VEML discriminator is calibrated on the BOC(1,1) " *
            "correlation peak, which has vanished from one chip on.",
        ),
    )
    VeryEarlyPromptLateCorrelatorBank(
        zero_correlator_outputs(num_antennas, NumCorrelators(5)),
        preferred_early_late_to_prompt_spacing,
        preferred_very_early_late_to_prompt_spacing,
    )
end

"""
$(SIGNATURES)

Get very early correlator output
"""
function get_very_early(correlator_bank::VeryEarlyPromptLateCorrelatorBank)
    correlator_outputs = get_correlator_outputs(correlator_bank)
    correlator_output_index = min(length(correlator_outputs), get_prompt_index(correlator_bank) + 2)
    correlator_outputs[correlator_output_index]
end

"""
$(SIGNATURES)

Get very late correlator output
"""
function get_very_late(correlator_bank::VeryEarlyPromptLateCorrelatorBank)
    correlator_output_index = max(1, get_prompt_index(correlator_bank) - 2)
    get_correlator_outputs(correlator_bank)[correlator_output_index]
end

"""
$(SIGNATURES)

Update the correlator bank with new correlator outputs
"""
function with_correlator_outputs(correlator_bank::VeryEarlyPromptLateCorrelatorBank, correlator_outputs)
    VeryEarlyPromptLateCorrelatorBank(
        correlator_outputs,
        correlator_bank.preferred_early_late_to_prompt_spacing,
        correlator_bank.preferred_very_early_late_to_prompt_spacing,
    )
end

"""
$(SIGNATURES)

Replica shifts relative to the prompt in samples, from latest to earliest replica.
"""
function calc_correlator_sample_offsets(
    correlator_bank::VeryEarlyPromptLateCorrelatorBank,
    sampling_frequency,
    code_frequency,
)
    early_late_sample_shift = calc_correlator_sample_offset(
        correlator_bank.preferred_early_late_to_prompt_spacing,
        sampling_frequency,
        code_frequency,
    )
    very_early_late_sample_shift = calc_correlator_sample_offset(
        correlator_bank.preferred_very_early_late_to_prompt_spacing,
        sampling_frequency,
        code_frequency,
    )
    SVector(
        -very_early_late_sample_shift,
        -early_late_sample_shift,
        0,
        early_late_sample_shift,
        very_early_late_sample_shift,
    )
end
