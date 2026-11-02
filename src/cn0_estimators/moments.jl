"""
$(SIGNATURES)

    MomentsCN0Estimator(; num_integrations = 100)

Estimate C/N₀ from the second and fourth moments of the latest `num_integrations` prompts.
"""
struct MomentsCN0Estimator <: AbstractCN0Estimator
    prompt_buffer::Vector{ComplexF64}
    write_index::Int
    num_filled::Int
end

function MomentsCN0Estimator(; num_integrations::Integer = 100)
    num_integrations > 0 || throw(ArgumentError("`num_integrations` must be positive"))
    MomentsCN0Estimator(zeros(ComplexF64, num_integrations), 0, 0)
end

length(estimator::MomentsCN0Estimator) = estimator.num_filled
get_prompt_buffer(estimator::MomentsCN0Estimator) = estimator.prompt_buffer
get_write_index(estimator::MomentsCN0Estimator) = estimator.write_index

"""
$(SIGNATURES)

Buffers the prompts such that they can be used to estimate the CN0.
Writes the prompt into the estimator's buffer and returns the updated estimator,
which replaces `estimator`.
The moment ratio needs no side information, so this estimator implements only
the prompt-only form of `TrackingLoops.update!`; see
[`update!(::AbstractCN0Estimator, ::Any, ::CN0UpdateContext)`](@ref) for the
extension point that also receives the navigation-bit context.
"""
function update!(estimator::MomentsCN0Estimator, prompt)
    buffer_length = length(estimator.prompt_buffer)
    next_index = mod(get_write_index(estimator), buffer_length) + 1
    estimator.prompt_buffer[next_index] = prompt
    num_filled = min(length(estimator) + 1, buffer_length)
    MomentsCN0Estimator(estimator.prompt_buffer, next_index, num_filled)
end

"""
$(SIGNATURES)

Estimates the CN0 based on the struct `MomentsCN0Estimator`.
"""
function estimate_cn0(estimator::MomentsCN0Estimator, coherent_integration_time)
    length(estimator) == 0 && return 0.0dBHz
    prompt_buffer = get_prompt_buffer(estimator)
    M₂ = 1 / length(estimator) * sum(abs2, prompt_buffer)
    M₄ = 1 / length(estimator) * sum(x -> abs2(x)^2, prompt_buffer)
    Pd = sqrt(abs(2 * M₂^2 - M₄))
    noise_power = abs(M₂ - Pd)
    SNR = Pd / noise_power
    dBHz(SNR / coherent_integration_time)
end
