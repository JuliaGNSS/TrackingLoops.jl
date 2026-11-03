```@meta
CurrentModule = TrackingLoops
```

# Correlator banks and discriminators

A correlator bank holds the correlator outputs of one integration: one complex value per
correlator and antenna. [`EarlyPromptLateCorrelatorBank`](@ref) has the early, prompt and
late correlators a DLL needs; [`VeryEarlyPromptLateCorrelatorBank`](@ref) adds a very-early
and a very-late correlator, e.g. for multipath monitoring or BOC tracking. A
[`CorrelatorBankOutput`](@ref) is one completed integration as a correlator bank hands it to
the loop: the correlator bank together with the sample span it was integrated over.

The discriminators [`pll_disc`](@ref TrackingLoops.pll_disc), [`fll_disc`](@ref TrackingLoops.fll_disc) and
[`dll_disc`](@ref TrackingLoops.dll_disc) turn a (filtered) correlator bank into the carrier-phase,
carrier-frequency and code-phase errors the loop filters act on. Before that,
the [`DefaultPostCorrFilter`](@ref) combines the antennas of a multi-antenna
correlator bank into one; subtype [`AbstractPostCorrFilter`](@ref TrackingLoops.AbstractPostCorrFilter) for beamforming.

## Correlator banks

The spacing keywords give the requested distance **from prompt**, in chips:
`preferred_early_late_to_prompt_spacing` for early and late, and
`preferred_very_early_late_to_prompt_spacing` for very early and very late.
For example, an early/late-to-prompt spacing of `0.5` puts each correlator half a
chip from prompt, for a full early–late spacing of one chip. The requested positions
are rounded to whole samples. `calc_correlator_sample_offsets` returns those
signed sample offsets; `calc_early_late_sample_spacing` returns their full separation.

```@docs
TrackingLoops.AbstractCorrelatorBank
TrackingLoops.AbstractEarlyPromptLateCorrelatorBank
CorrelatorBankOutput
TrackingLoops.apply(::F, ::AbstractCorrelatorBank) where {F}
TrackingLoops.calc_correlator_sample_offset(::Any, ::Any, ::Any)
get_correlator_outputs(::AbstractCorrelatorBank)
get_early(::AbstractEarlyPromptLateCorrelatorBank)
TrackingLoops.calc_early_late_sample_spacing(::AbstractEarlyPromptLateCorrelatorBank, ::Any, ::Any)
TrackingLoops.zero_correlator_outputs(::NumAnts, ::NumCorrelators{M}) where {M}
get_late(::AbstractEarlyPromptLateCorrelatorBank)
TrackingLoops.get_num_correlators(::AbstractCorrelatorBank)
get_num_ants(::AbstractCorrelatorBank{M}) where {M}
get_prompt(::AbstractCorrelatorBank)
TrackingLoops.get_prompt_index(::AbstractCorrelatorBank)
TrackingLoops.normalize(::AbstractCorrelatorBank, ::Any, ::Any = 1)
EarlyPromptLateCorrelatorBank
EarlyPromptLateCorrelatorBank()
TrackingLoops.calc_correlator_sample_offsets(::EarlyPromptLateCorrelatorBank, ::Any, ::Any)
TrackingLoops.with_correlator_outputs(::EarlyPromptLateCorrelatorBank, ::Any)
VeryEarlyPromptLateCorrelatorBank
VeryEarlyPromptLateCorrelatorBank()
TrackingLoops.calc_correlator_sample_offsets(::VeryEarlyPromptLateCorrelatorBank, ::Any, ::Any)
TrackingLoops.get_very_early(::VeryEarlyPromptLateCorrelatorBank)
TrackingLoops.get_very_late(::VeryEarlyPromptLateCorrelatorBank)
TrackingLoops.with_correlator_outputs(::VeryEarlyPromptLateCorrelatorBank, ::Any)
```

## Discriminators

```@docs
TrackingLoops.dll_disc(::AbstractGNSSSignal, ::EarlyPromptLateCorrelatorBank, ::Any, ::Any)
TrackingLoops.dll_disc(::AbstractGNSSSignal, ::VeryEarlyPromptLateCorrelatorBank, ::Any, ::Any)
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
