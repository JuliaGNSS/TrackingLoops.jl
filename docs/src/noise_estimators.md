```@meta
CurrentModule = TrackingLoops
```

# Noise estimation

A noise estimator measures the post-correlation noise density `N₀` of one
signal, which a [`NoiseRefCN0Estimator`](@ref) divides each record's prompt power
by. It is fed [`NoiseObservation`](@ref TrackingLoops.NoiseObservation)s — from a correlator tap on a PRN that
is not transmitted ([`noise_observation_from_correlator`](@ref)) or from raw
samples ([`noise_observation_from_samples`](@ref TrackingLoops.noise_observation_from_samples)) — with
[`append_noise_observation!`](@ref) or [`update_noise!`](@ref TrackingLoops.update_noise!), and read with
[`get_noise_density`](@ref TrackingLoops.get_noise_density). [`CorrelatorNoiseEstimator`](@ref) keeps a sliding
window of observations; subtype [`AbstractNoiseEstimator`](@ref TrackingLoops.AbstractNoiseEstimator) for another
source.

## Why per signal and not per RF band

What a record actually divides by is the **post-correlation** density. For an
interferer of PSD `S_I(f)` on top of the thermal floor that is

```
N₀,eff = N₀,thermal + ∫ S_I(f) · |G(f)|² df
```

— the spectral separation coefficient, weighted by the *despreading modulation's*
own spectrum `|G(f)|²`. Two signals sharing one band, antenna and front end
therefore see different floors as soon as the interference is not white, and
not by a little: BPSK(1) peaks at DC with a null at ±1.023 MHz, BOC(1,1) is the
reverse, so a CW tone at band centre is rejected by Galileo E1B and lands
squarely in GPS L1 C/A, while a tone at ±1.023 MHz does the opposite. Front-end
tilt, filter roll-off at the band edge and adjacent-band leakage all colour the
floor the same way, more quietly.

Because [`CorrelatorNoiseEstimator`](@ref) *despreads* rather than metering
power, keying by signal makes that integral **measured rather than modelled**:
the reference runs the consumer's own code, so its spectral weighting is the
consumer's by construction.

It also matches the hardware: a noise channel is a tracking channel with a wrong
PRN, configured with a code, so per signal is what an FPGA builds anyway.

## Why a density and not a power

For a correlator normalised the way `TrackingLoops.normalize` normalises — by
`num_integrated_samples * code_amplitude` — white input noise of per-sample variance
`σ²` gives `E|P|² = σ²/N = N₀/T`. So `N₀ = σ²/f_s` is **independent of the
integration time**, which is what lets one per-signal figure serve records of any
length. It is stored as a Unitful quantity of dimension `1/Hz`, so the
consumer's `⟨|P|²⟩/N₀ − 1/T` is dimension-checked rather than trusted.

## Filling the reference

A [`CorrelatorNoiseEstimator`](@ref) plans its own measurement:
[`noise_despreads`](@ref TrackingLoops.noise_despreads) splits a slice of samples into
code-period sub-integrations, rotates the PRN and draws each one's random code phase and
carrier. A host only despreads each [`NoiseDespread`](@ref TrackingLoops.NoiseDespread)
through [`noise_correlator`](@ref TrackingLoops.noise_correlator)'s taps and appends them
with [`append_despread_noise!`](@ref TrackingLoops.append_despread_noise!). Tracking.jl
does so on its kernels when [`update_noise!`](@ref TrackingLoops.update_noise!) calls its
[`despread_noise!`](@ref TrackingLoops.despread_noise!); a hardware host programs a
noise channel with the same plan:

```julia
correlator = TrackingLoops.noise_correlator(estimator)
despreads = TrackingLoops.noise_despreads(
    estimator,
    signal,
    sample_range,
    sampling_frequency,
    intermediate_frequency,
)
for despread in despreads
    taps = despread_on_noise_channel(correlator, despread)
    TrackingLoops.append_despread_noise!(estimator, despread, taps)
end
```

```@docs
TrackingLoops.AbstractNoiseEstimator
TrackingLoops.NoiseDensity
TrackingLoops.NoiseEstimators
TrackingLoops.NoiseObservation
TrackingLoops.NoiseUpdateContext
append_noise_observation!(::AbstractNoiseEstimator, ::NoiseObservation)
TrackingLoops.get_noise_density(::AbstractNoiseEstimator)
TrackingLoops.noise_density_and_ready(::AbstractNoiseEstimator)
TrackingLoops.noise_density_type(::AbstractNoiseEstimator)
TrackingLoops.noise_observation(::Complex, ::Any, ::Any)
noise_observation_from_correlator(::Any, ::Any, ::Any, ::Any)
TrackingLoops.noise_observation_from_samples(::Any, ::Any, ::Any)
TrackingLoops.noise_window_filling(::AbstractNoiseEstimator)
TrackingLoops.noise_window_looks(::AbstractNoiseEstimator)
TrackingLoops.update_noise!(::AbstractNoiseEstimator, ::Any, ::Any, ::Any)
CorrelatorNoiseEstimator
CorrelatorNoiseEstimator()
TrackingLoops.NoiseDespread
TrackingLoops.append_despread_noise!(::CorrelatorNoiseEstimator, ::NoiseDespread, ::Any)
append_noise_observation!(::CorrelatorNoiseEstimator{D, T}, ::NoiseObservation) where {D, T}
TrackingLoops.despread_noise!
TrackingLoops.get_noise_density(::CorrelatorNoiseEstimator)
TrackingLoops.noise_correlator(::CorrelatorNoiseEstimator)
TrackingLoops.noise_despreads(::CorrelatorNoiseEstimator, ::AbstractGNSSSignal, ::AbstractUnitRange{<:Integer}, ::Any, ::Any)
TrackingLoops.update_noise!(::CorrelatorNoiseEstimator, ::Any, ::AbstractUnitRange{<:Integer}, ::NoiseUpdateContext)
```
