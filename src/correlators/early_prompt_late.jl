"""
$(SIGNATURES)

Correlator with early, prompt and late taps. The code shift is a preferred shift:
rounded to whole samples for the sampling and code frequency, at least one.
"""
struct EarlyPromptLateCorrelator{M,T} <: AbstractEarlyPromptLateCorrelator{M}
    accumulators::SVector{3,T}
    preferred_early_late_to_prompt_code_shift::Float64
    function EarlyPromptLateCorrelator(
        accumulators::AbstractVector{SVector{M,T}},
        preferred_early_late_to_prompt_code_shift,
    ) where {M,T<:Complex}
        new{M,SVector{M,T}}(accumulators, preferred_early_late_to_prompt_code_shift)
    end
    function EarlyPromptLateCorrelator(
        accumulators::AbstractVector{T},
        preferred_early_late_to_prompt_code_shift,
    ) where {T<:Complex}
        new{1,T}(accumulators, preferred_early_late_to_prompt_code_shift)
    end
end

"""
$(SIGNATURES)

Construct a zeroed [`EarlyPromptLateCorrelator`](@ref).
"""
function EarlyPromptLateCorrelator(;
    num_ants::NumAnts = NumAnts(1),
    preferred_early_late_to_prompt_code_shift = 0.5,
)
    EarlyPromptLateCorrelator(
        get_initial_accumulator(num_ants, NumAccumulators(3)),
        preferred_early_late_to_prompt_code_shift,
    )
end

"""
$(SIGNATURES)

Update the correlator with new accumulators
"""
function update_accumulator(correlator::EarlyPromptLateCorrelator, accumulators)
    EarlyPromptLateCorrelator(
        accumulators,
        correlator.preferred_early_late_to_prompt_code_shift,
    )
end

"""
$(SIGNATURES)

Replica shifts relative to the prompt in samples, from latest to earliest replica.
"""
function get_correlator_sample_shifts(
    correlator::EarlyPromptLateCorrelator,
    sampling_frequency,
    code_frequency,
)
    sample_shift = calc_preferred_code_shift_to_sample_shift(
        correlator.preferred_early_late_to_prompt_code_shift,
        sampling_frequency,
        code_frequency,
    )
    SVector(-sample_shift, 0, sample_shift)
end
