"""
$(SIGNATURES)

    MomentsCN0Estimator(; num_averaged_integrations = 100)

Estimate C/N₀ from the second and fourth moments of the latest `num_averaged_integrations` prompts.
"""
struct MomentsCN0Estimator <: AbstractCN0Estimator
    prompt_buffer::Vector{ComplexF64}
    newest_index::Int
    num_buffered::Int
end

function MomentsCN0Estimator(; num_averaged_integrations::Integer = 100)
    num_averaged_integrations > 0 || throw(ArgumentError("`num_averaged_integrations` must be positive"))
    MomentsCN0Estimator(zeros(ComplexF64, num_averaged_integrations), 0, 0)
end

length(estimator::MomentsCN0Estimator) = estimator.num_buffered
get_prompt_buffer(estimator::MomentsCN0Estimator) = estimator.prompt_buffer
get_newest_index(estimator::MomentsCN0Estimator) = estimator.newest_index

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
    next_index = mod(get_newest_index(estimator), buffer_length) + 1
    estimator.prompt_buffer[next_index] = prompt
    num_buffered = min(length(estimator) + 1, buffer_length)
    MomentsCN0Estimator(estimator.prompt_buffer, next_index, num_buffered)
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

reset_cn0_estimator!(e::MomentsCN0Estimator) =
    (fill!(e.prompt_buffer, zero(ComplexF64)); MomentsCN0Estimator(e.prompt_buffer, 0, 0))
