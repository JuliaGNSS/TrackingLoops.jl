"""
$(SIGNATURES)

VeryEarlyPromptLateCorrelator holding a user defined number of correlation values.
The code is shifted in samples. Hence, the specified code shift is actually a
preferred code shift, because depending on sampling frequency and
code frequency the specified code shift might not be the actual code shift. It is as
close as possible, though. The algorithm makes sure that at least one sample is shifted.
"""
struct VeryEarlyPromptLateCorrelator{M,T} <: AbstractEarlyPromptLateCorrelator{M}
    accumulators::SVector{5,T}
    preferred_early_late_to_prompt_code_shift::Float64
    preferred_very_early_late_to_prompt_code_shift::Float64
    function VeryEarlyPromptLateCorrelator(
        accumulators::AbstractVector{SVector{M,T}},
        preferred_early_late_to_prompt_code_shift,
        preferred_very_early_late_to_prompt_code_shift,
    ) where {M,T<:Complex}
        new{M,SVector{M,T}}(
            accumulators,
            preferred_early_late_to_prompt_code_shift,
            preferred_very_early_late_to_prompt_code_shift,
        )
    end
    function VeryEarlyPromptLateCorrelator(
        accumulators::AbstractVector{T},
        preferred_early_late_to_prompt_code_shift,
        preferred_very_early_late_to_prompt_code_shift,
    ) where {T<:Complex}
        new{1,T}(
            accumulators,
            preferred_early_late_to_prompt_code_shift,
            preferred_very_early_late_to_prompt_code_shift,
        )
    end
end

"""
$(SIGNATURES)

VeryEarlyPromptLateCorrelator constructor without parameters and some default
parameters, taken from GNSS-SDR's Galileo E1 DLL/PLL VEML tracking:
<https://gnss-sdr.org/docs/sp-blocks/tracking/#implementation-galileo_e1_dll_pll_veml_tracking>

Throws an `ArgumentError` if both code shifts are one chip or more: the BOC(1,1)
correlation peak the VEML discriminator ([`dll_disc`](@ref)) is calibrated on has
vanished there, so no tap would see the signal.
"""
function VeryEarlyPromptLateCorrelator(;
    num_ants::NumAnts = NumAnts(1),
    preferred_early_late_to_prompt_code_shift = 0.15,
    preferred_very_early_late_to_prompt_code_shift = 0.6,
)
    min(
        preferred_early_late_to_prompt_code_shift,
        preferred_very_early_late_to_prompt_code_shift,
    ) < 1 || throw(
        ArgumentError(
            "VeryEarlyPromptLateCorrelator: the early/late code shift " *
            "($preferred_early_late_to_prompt_code_shift chips) or the " *
            "very-early/very-late code shift " *
            "($preferred_very_early_late_to_prompt_code_shift chips) must be below " *
            "one chip; the VEML discriminator is calibrated on the BOC(1,1) " *
            "correlation peak, which has vanished from one chip on.",
        ),
    )
    VeryEarlyPromptLateCorrelator(
        get_initial_accumulator(num_ants, NumAccumulators(5)),
        preferred_early_late_to_prompt_code_shift,
        preferred_very_early_late_to_prompt_code_shift,
    )
end

"""
$(SIGNATURES)

Get very early correlator
"""
function get_very_early(correlator::VeryEarlyPromptLateCorrelator)
    accumulators = get_accumulators(correlator)
    accumulator_index = min(length(accumulators), get_prompt_index(correlator) + 2)
    accumulators[accumulator_index]
end

"""
$(SIGNATURES)

Get very late correlator
"""
function get_very_late(correlator::VeryEarlyPromptLateCorrelator)
    accumulator_index = max(1, get_prompt_index(correlator) - 2)
    get_accumulators(correlator)[accumulator_index]
end

"""
$(SIGNATURES)

Update the correlator with new accumulators
"""
function update_accumulator(correlator::VeryEarlyPromptLateCorrelator, accumulators)
    VeryEarlyPromptLateCorrelator(
        accumulators,
        correlator.preferred_early_late_to_prompt_code_shift,
        correlator.preferred_very_early_late_to_prompt_code_shift,
    )
end

"""
$(SIGNATURES)

Calculate the replica phase offset required for the correlator with
respect to the prompt correlator, expressed in samples. The shifts are
ordered from latest to earliest replica.
"""
function get_correlator_sample_shifts(
    correlator::VeryEarlyPromptLateCorrelator,
    sampling_frequency,
    code_frequency,
)
    early_late_sample_shift = calc_preferred_code_shift_to_sample_shift(
        correlator.preferred_early_late_to_prompt_code_shift,
        sampling_frequency,
        code_frequency,
    )
    very_early_late_sample_shift = calc_preferred_code_shift_to_sample_shift(
        correlator.preferred_very_early_late_to_prompt_code_shift,
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
