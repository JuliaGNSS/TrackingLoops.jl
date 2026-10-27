```@meta
CurrentModule = TrackingLoops
```

# Correlators and discriminators

A correlator holds the accumulators of one integration: one complex value per
tap and antenna. [`EarlyPromptLateCorrelator`](@ref) has the early, prompt and
late taps a DLL needs; [`VeryEarlyPromptLateCorrelator`](@ref) adds a very-early
and a very-late tap, e.g. for multipath monitoring or BOC tracking. A
[`CorrelatorOutput`](@ref) is one completed record as a correlator hands it to
the loop: the correlator together with the sample span it was integrated over.

The discriminators [`pll_disc`](@ref TrackingLoops.pll_disc), [`fll_disc`](@ref TrackingLoops.fll_disc) and
[`dll_disc`](@ref TrackingLoops.dll_disc) turn a (filtered) correlator into the carrier-phase,
carrier-frequency and code-phase errors the loop filters act on. Before that,
the [`DefaultPostCorrFilter`](@ref) combines the antennas of a multi-antenna
correlator into one; subtype [`AbstractPostCorrFilter`](@ref TrackingLoops.AbstractPostCorrFilter) for beamforming.

## Correlators

```@docs
TrackingLoops.AbstractCorrelator
TrackingLoops.AbstractEarlyPromptLateCorrelator
CorrelatorOutput
TrackingLoops.apply(::F, ::AbstractCorrelator) where {F}
TrackingLoops.calc_preferred_code_shift_to_sample_shift(::Any, ::Any, ::Any)
get_accumulators(::AbstractCorrelator)
get_early(::AbstractEarlyPromptLateCorrelator)
TrackingLoops.get_early_late_sample_spacing(::AbstractEarlyPromptLateCorrelator, ::Any, ::Any)
TrackingLoops.get_initial_accumulator(::NumAnts, ::NumAccumulators{M}) where {M}
get_late(::AbstractEarlyPromptLateCorrelator)
TrackingLoops.get_num_accumulators(::AbstractCorrelator)
get_num_ants(::AbstractCorrelator{M}) where {M}
get_prompt(::AbstractCorrelator)
TrackingLoops.get_prompt_index(::AbstractCorrelator)
TrackingLoops.normalize(::AbstractCorrelator, ::Any, ::Any = 1)
EarlyPromptLateCorrelator
EarlyPromptLateCorrelator()
TrackingLoops.get_correlator_sample_shifts(::EarlyPromptLateCorrelator, ::Any, ::Any)
TrackingLoops.update_accumulator(::EarlyPromptLateCorrelator, ::Any)
VeryEarlyPromptLateCorrelator
VeryEarlyPromptLateCorrelator()
TrackingLoops.get_correlator_sample_shifts(::VeryEarlyPromptLateCorrelator, ::Any, ::Any)
TrackingLoops.get_very_early(::VeryEarlyPromptLateCorrelator)
TrackingLoops.get_very_late(::VeryEarlyPromptLateCorrelator)
TrackingLoops.update_accumulator(::VeryEarlyPromptLateCorrelator, ::Any)
```

## Discriminators

```@docs
TrackingLoops.dll_disc(::AbstractGNSSSignal, ::EarlyPromptLateCorrelator, ::Any, ::Any)
TrackingLoops.dll_disc(::AbstractGNSSSignal, ::VeryEarlyPromptLateCorrelator, ::Any, ::Any)
TrackingLoops.fll_disc(::AbstractGNSSSignal, ::Any, ::Any, ::Any)
TrackingLoops.pll_disc(::AbstractGNSSSignal, ::Any)
```

## Post-correlation filter

```@docs
TrackingLoops.AbstractPostCorrFilter
DefaultPostCorrFilter
TrackingLoops.get_weights
TrackingLoops.update(::DefaultPostCorrFilter, ::Any)
```
