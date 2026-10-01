# Correlators and discriminators

A correlator holds the accumulators of one integration: one complex value per
tap and antenna. [`EarlyPromptLateCorrelator`](@ref) has the early, prompt and
late taps a DLL needs; [`VeryEarlyPromptLateCorrelator`](@ref) adds a very-early
and a very-late tap, e.g. for multipath monitoring or BOC tracking. A
[`CorrelatorOutput`](@ref) is one completed record as a correlator hands it to
the loop: the correlator together with the sample span it was integrated over.

The discriminators [`pll_disc`](@ref), [`fll_disc`](@ref) and
[`dll_disc`](@ref) turn a (filtered) correlator into the carrier-phase,
carrier-frequency and code-phase errors the loop filters act on. Before that,
the [`DefaultPostCorrFilter`](@ref) combines the antennas of a multi-antenna
correlator into one; subtype [`AbstractPostCorrFilter`](@ref) for beamforming.

## Correlators

```@autodocs
Modules = [TrackingLoops]
Pages = ["correlators/correlator.jl", "correlators/early_prompt_late.jl", "correlators/very_early_prompt_late.jl"]
Private = false
```

## Discriminators

```@autodocs
Modules = [TrackingLoops]
Pages = ["discriminators.jl"]
Private = false
```

## Post-correlation filter

```@autodocs
Modules = [TrackingLoops]
Pages = ["post_corr_filter.jl"]
Private = false
```
