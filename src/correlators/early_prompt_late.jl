"""
$(SIGNATURES)

Correlator bank with early, prompt and late correlators.

`preferred_early_late_to_prompt_spacing` is the requested distance in chips from
prompt to each early/late correlator, rounded to whole samples for the sampling
and code frequency, at least one sample. The default `0.5` places early and late
half a chip from prompt, for a total early–late spacing of one chip before rounding.
"""
struct EarlyPromptLateCorrelatorBank{M,T} <: AbstractEarlyPromptLateCorrelatorBank{M}
    correlator_outputs::SVector{3,T}
    preferred_early_late_to_prompt_spacing::Float64
    function EarlyPromptLateCorrelatorBank(
        correlator_outputs::AbstractVector{SVector{M,T}},
        preferred_early_late_to_prompt_spacing,
    ) where {M,T<:Complex}
        new{M,SVector{M,T}}(correlator_outputs, preferred_early_late_to_prompt_spacing)
    end
    function EarlyPromptLateCorrelatorBank(
        correlator_outputs::AbstractVector{T},
        preferred_early_late_to_prompt_spacing,
    ) where {T<:Complex}
        new{1,T}(correlator_outputs, preferred_early_late_to_prompt_spacing)
    end
end

"""
$(SIGNATURES)

Construct a zeroed [`EarlyPromptLateCorrelatorBank`](@ref).
"""
function EarlyPromptLateCorrelatorBank(;
    num_ants::NumAnts = NumAnts(1),
    preferred_early_late_to_prompt_spacing = 0.5,
)
    EarlyPromptLateCorrelatorBank(
        zero_correlator_outputs(num_ants, NumCorrelators(3)),
        preferred_early_late_to_prompt_spacing,
    )
end

"""
$(SIGNATURES)

Update the correlator bank with new correlator outputs
"""
function with_correlator_outputs(correlator_bank::EarlyPromptLateCorrelatorBank, correlator_outputs)
    EarlyPromptLateCorrelatorBank(
        correlator_outputs,
        correlator_bank.preferred_early_late_to_prompt_spacing,
    )
end

"""
$(SIGNATURES)

Replica shifts relative to the prompt in samples, from latest to earliest replica.
"""
function calc_correlator_sample_offsets(
    correlator_bank::EarlyPromptLateCorrelatorBank,
    sampling_frequency,
    code_frequency,
)
    sample_shift = calc_correlator_sample_offset(
        correlator_bank.preferred_early_late_to_prompt_spacing,
        sampling_frequency,
        code_frequency,
    )
    SVector(-sample_shift, 0, sample_shift)
end
