# Noise estimation

A noise estimator measures the post-correlation noise density `N₀` of one
signal, which a [`NoiseRefCN0Estimator`](@ref) divides each record's prompt power
by. It is fed [`NoiseObservation`](@ref)s — from a correlator tap on a PRN that
is not transmitted ([`noise_observation_from_correlator`](@ref)) or from raw
samples ([`noise_observation_from_samples`](@ref)) — with
[`append_noise_observation!`](@ref) or [`update_noise!`](@ref), and read with
[`get_noise_density`](@ref). [`CorrelatorNoiseEstimator`](@ref) keeps a sliding
window of observations; subtype [`AbstractNoiseEstimator`](@ref) for another
source.

```@autodocs
Modules = [TrackingLoops]
Pages = ["noise_estimators/noise_estimator.jl", "noise_estimators/window.jl"]
Private = false
```
