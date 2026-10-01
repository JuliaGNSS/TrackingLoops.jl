# C/N₀ estimation

A C/N₀ estimator is advanced once per record with the record's prompt
(`TrackingLoops.update`, called by [`apply_record`](@ref)) and read with
[`estimate_cn0`](@ref). The package provides:

- [`MomentsCN0Estimator`](@ref) — the moments method on a window of prompts;
- [`NWPRCN0Estimator`](@ref) — the narrow-to-wideband power ratio, summing
  prompts coherently over one navigation symbol once the bit grid is known;
- [`NoiseRefCN0Estimator`](@ref) — divides the prompt power by a noise density
  measured independently of the signal (see [Noise estimation](@ref));
- [`NoCN0Estimator`](@ref) — for a component whose C/N₀ is not needed.

[`default_cn0_estimator`](@ref) says which one a signal gets by default. A
custom estimator subtypes [`AbstractCN0Estimator`](@ref) and implements
`TrackingLoops.update` and [`estimate_cn0`](@ref), plus
[`requires_noise_density`](@ref) if it reads a measured noise floor.

```@autodocs
Modules = [TrackingLoops]
Pages = [
    "cn0_estimators/cn0_estimator.jl",
    "cn0_estimators/moments.jl",
    "cn0_estimators/nwpr.jl",
    "cn0_estimators/noise_ref.jl",
    "cn0_estimators/no_cn0.jl",
]
Private = false
```
