```@meta
CurrentModule = TrackingLoops
```

# C/N₀ estimation

A C/N₀ estimator is advanced once per record with the record's prompt
(`TrackingLoops.update`, called by [`apply_record`](@ref)) and read with
[`estimate_cn0`](@ref). The package provides:

- [`MomentsCN0Estimator`](@ref) — the moments method on a window of prompts;
- [`NWPRCN0Estimator`](@ref) — the narrow-to-wideband power ratio, summing
  prompts coherently over short windows, aligned to the bit grid once known;
- [`NoiseRefCN0Estimator`](@ref) — divides the prompt power by a noise density
  measured independently of the signal (see [Noise estimation](@ref));
- [`NoCN0Estimator`](@ref) — for a component whose C/N₀ is not needed.

[`default_cn0_estimator`](@ref TrackingLoops.default_cn0_estimator) says which one a signal gets by default.

```@docs
TrackingLoops.AbstractCN0Estimator
TrackingLoops.CN0UpdateContext
TrackingLoops.default_cn0_estimator(::AbstractGNSSSignal, ::Int)
TrackingLoops.requires_noise_density(::Type{<:AbstractCN0Estimator})
TrackingLoops.update(::AbstractCN0Estimator, ::Any, ::CN0UpdateContext)
MomentsCN0Estimator
estimate_cn0(::MomentsCN0Estimator, ::Any)
TrackingLoops.update(::MomentsCN0Estimator, ::Any)
NWPRCN0Estimator
NWPRCN0Estimator(::AbstractGNSSSignal)
NWPRCN0Estimator()
estimate_cn0(::NWPRCN0Estimator, ::Any)
TrackingLoops.update(::NWPRCN0Estimator, ::Any, ::CN0UpdateContext)
TrackingLoops.update(::NWPRCN0Estimator, ::Any)
NoiseRefCN0Estimator
NoiseRefCN0Estimator()
estimate_cn0(::NoiseRefCN0Estimator, ::Any)
TrackingLoops.requires_noise_density(::Type{NoiseRefCN0Estimator})
TrackingLoops.update(::NoiseRefCN0Estimator, ::Any, ::CN0UpdateContext)
TrackingLoops.update(::NoiseRefCN0Estimator, ::Any)
NoCN0Estimator
estimate_cn0(::NoCN0Estimator, ::Any)
TrackingLoops.update(::NoCN0Estimator, ::Any)
```
