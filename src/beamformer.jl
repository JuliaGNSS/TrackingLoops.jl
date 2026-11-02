"""
$(SIGNATURES)

Abstract beamformer for the prompt signal — the linear combiner that
reduces a correlator output's per-antenna values to the single complex value the
discriminators and the C/N₀ estimator see.

An implementation must provide two methods:

  - [`get_beamforming_weights`](@ref)`(beamformer, ::NumAntennas{M})` — the combining weights `w` this
    beamformer currently applies. A plain `ComplexF64` at `M == 1`, an
    `SVector{M,ComplexF64}` above.
  - [`update!`](@ref)`(beamformer, prompt)` — return the beamformer to use for the next
    integration, which is where an adaptive beamformer recomputes its weights.

The contract is deliberately **linear in the antennas**: the combined correlator output is
`wᴴ·b`. That is what lets the C/N₀ path stay honest without any per-satellite
noise state. The noise reference measures one spatial covariance `R̂` per signal
(see [`AbstractNoiseDensityEstimator`](@ref)), and each satellite reduces it to its own
scalar floor through its own weights, `N₀ = wᴴR̂w` — exact for any fixed `w`,
because `E[|wᴴn|²] = wᴴRw`. A beamformer that combined its antennas non-linearly
would have no such closed form, and its C/N₀ would silently be a ratio of two
different noise scales.
"""
abstract type AbstractBeamformer end

"""
$(SIGNATURES)

This is the default beamformer. For a single antenna channel it
returns the prompt value as is. For multi antenna systems it selects the **last**
antenna's value — the noise reference's covariance reduces to that antenna's own
floor under these weights, so the two sides of the C/N₀ ratio agree.
"""
struct DefaultBeamformer <: AbstractBeamformer end

"""
$(SIGNATURES)

Update the beamformer with the latest prompt and return the
updated beamformer, which replaces `beamformer`. This is the extension point for custom
[`AbstractBeamformer`](@ref)s — e.g. a beamformer adapting its weights.
"""
update!(beamformer::DefaultBeamformer, prompt) = beamformer

"""
$(SIGNATURES)

The linear combining weights `w` that `beamformer` currently applies to a correlator
output, as a `ComplexF64` for `NumAntennas(1)` and an `SVector{M,ComplexF64}` for
`NumAntennas(M)`. The combined correlator output is `wᴴ·b`, and the matching post-combination noise
floor is `wᴴR̂w` — see [`AbstractBeamformer`](@ref).

Required for every [`AbstractBeamformer`](@ref); there is no fallback,
because a beamformer whose weights are unknown cannot be given a correct C/N₀.

# Examples

A beamformer that averages the antenna elements:

```julia
struct MyBeamformer <: AbstractBeamformer end
TrackingLoops.update!(f::MyBeamformer, prompt) = f
TrackingLoops.get_beamforming_weights(::MyBeamformer, ::NumAntennas{1}) = 1.0 + 0.0im
TrackingLoops.get_beamforming_weights(::MyBeamformer, ::NumAntennas{M}) where {M} =
    SVector{M,ComplexF64}(ntuple(_ -> 1 / M + 0.0im, M))
```
"""
function get_beamforming_weights end

# A plain `ComplexF64` at `M == 1`, matching single-antenna correlator outputs (see
# `correlator_output_type`); `one(ComplexF64)` makes combining and noise reduction
# bit-identical no-ops there.
get_beamforming_weights(::DefaultBeamformer, ::NumAntennas{1}) = one(ComplexF64)
get_beamforming_weights(::DefaultBeamformer, ::NumAntennas{M}) where {M} =
    SVector{M,ComplexF64}(ntuple(i -> i == M ? one(ComplexF64) : zero(ComplexF64), M))

# Combine one correlator output through the weights: `wᴴ·b`. The conjugation is on
# `w`, matching `_reduce_noise_density`'s `wᴴRw` — the two must agree or the
# C/N₀ ratio's numerator and denominator would use different combiners.
@inline _combine_antennas(w::Number, correlator_output::Number) = conj(w) * correlator_output
@inline _combine_antennas(w::StaticVector, correlator_output::StaticVector) = w'correlator_output

# The same, for every correlator output of a correlator bank at once. A separate small function on
# purpose: inline in the large `_process_correlator_bank_output!`, the weight-capturing closure is
# heap-allocated per integration, breaking the per-integration allocation checks
# (`test/allocations.jl`).
@inline _beamform_correlator_bank(correlator_bank, weights) = with_correlator_outputs(
    correlator_bank,
    map(correlator_output -> _combine_antennas(weights, correlator_output), get_correlator_outputs(correlator_bank)),
)

# Reduce a measured noise floor to the scalar one that *this* combiner sees.
# `nothing` (no noise estimator configured) passes through, keeping the C/N₀
# context's type parameter `Nothing`, so `NoiseReferencedCN0Estimator`'s `update` throws
# the wiring-mistake error at the first integration. The scalar case is
# `|w|²·N₀`; the matrix case `wᴴRw` needs no extra `‖w‖²` factor, because the
# covariance already carries the antennas' relative scale.
@inline _reduce_noise_density(::Nothing, w) = nothing
@inline _reduce_noise_density(density::Number, w::Number) = abs2(w) * density
@inline _reduce_noise_density(R::StaticMatrix, w::StaticVector) = real(w' * R * w)
