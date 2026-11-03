```@meta
CurrentModule = TrackingLoops
```

# Noise estimation

A noise estimator measures the post-correlation noise density `N₀` of one
signal, which a [`NoiseRefCN0Estimator`](@ref) divides each record's prompt power
by. It is fed [`NoiseObservation`](@ref TrackingLoops.NoiseObservation)s — from a correlator tap on a PRN that
is not transmitted ([`noise_observation_from_correlator`](@ref TrackingLoops.noise_observation_from_correlator)) or from raw
samples ([`noise_observation_from_samples`](@ref TrackingLoops.noise_observation_from_samples)) — with
[`append_noise_observation!`](@ref) or [`update_noise!`](@ref TrackingLoops.update_noise!), and read with
[`get_noise_density`](@ref TrackingLoops.get_noise_density). [`CorrelatorNoiseEstimator`](@ref) keeps a sliding
window of observations; subtype [`AbstractNoiseEstimator`](@ref TrackingLoops.AbstractNoiseEstimator) for another
source.

## Why per signal and not per RF band

What a coherent integration actually divides by is the **post-correlation** density. For an
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

For a correlator bank normalised the way `TrackingLoops.normalize` normalises — by
`num_integrated_samples * code_amplitude` — white input noise of per-sample variance
`σ²` gives `E|P|² = σ²/N = N₀/T`. So `N₀ = σ²/f_s` is **independent of the
integration time**, which is what lets one per-signal figure serve coherent integrations of any
length. It is stored as a Unitful quantity of dimension `1/Hz`, so the
consumer's `⟨|P|²⟩/N₀ − 1/T` is dimension-checked rather than trusted.

## Filling the reference

A [`CorrelatorNoiseEstimator`](@ref) plans its own measurement:
[`noise_despreads`](@ref TrackingLoops.noise_despreads) splits a slice of samples into
code-period despreads, rotates the PRN and draws each one's random code phase and
carrier. A host only despreads each [`NoiseDespread`](@ref TrackingLoops.NoiseDespread)
through [`get_noise_correlator_bank`](@ref TrackingLoops.get_noise_correlator_bank)'s correlator outputs and appends them
with [`append_despread_noise!`](@ref TrackingLoops.append_despread_noise!). Tracking.jl
does so on its kernels when [`update_noise!`](@ref TrackingLoops.update_noise!) calls its
[`despread_noise!`](@ref TrackingLoops.despread_noise!):

```julia
correlator_bank = TrackingLoops.get_noise_correlator_bank(estimator)
despreads = TrackingLoops.noise_despreads(
    estimator,
    signal,
    sample_range,
    sampling_frequency,
    intermediate_frequency,
)
for despread in despreads
    correlator_outputs = despread_on_kernel(correlator_bank, despread)
    TrackingLoops.append_despread_noise!(estimator, despread, correlator_outputs)
end
```

## A hardware noise channel

A hardware host cannot redraw per code period: each new probe is a re-arm of its noise
channel, and the device dumps on its own grid. So it draws one
[`NoiseProbe`](@ref TrackingLoops.NoiseProbe) per arm with
[`draw_noise_probe`](@ref TrackingLoops.draw_noise_probe), programs the channel with its
PRN, code phase, code frequency and carrier, and appends every dump over the dump's own
sample count with [`append_noise_correlator_outputs!`](@ref TrackingLoops.append_noise_correlator_outputs!):

```julia
probe = TrackingLoops.draw_noise_probe(estimator, signal, intermediate_frequency)
arm_noise_channel!(device, probe)                       # the host's driver call
for dump in noise_channel_dumps(device)                 # until the next re-arm
    TrackingLoops.append_noise_correlator_outputs!(estimator, probe, dump.correlator_outputs,
        dump.num_integrated_samples, sampling_frequency)
end
```

The rotation advances on the next draw: the PRN is carried in the newest observation.
How often to redraw is the host's trade-off. A probe parked over a whole window that
happens to line up with a visible but untracked satellite (a hit needs the code phase
within a correlator, about `3 × 1.5 / 1023 ≈ 0.44 %` per such satellite, and the carrier within
about `1/T` of its Doppler, about a fifth of a ±5 kHz dither at `T = 1 ms`) raises every
observation it covers by about `T·(C/N₀)/3 ≈ 10.5·N₀` at 45 dB-Hz, so `N̂₀` by up to
≈ 10 dB for that window. Redrawing `n` times per window lets one hit bias only `1/n` of
it, at `n` times the chance of some hit: a few redraws per window already cap a rare
hit at a few dB, at the cost of the dumps lost to each re-arm. A device that can restart
its code generator at a random phase by itself redraws without a re-arm.

```@docs
TrackingLoops.AbstractNoiseEstimator
TrackingLoops.NoiseDensity
TrackingLoops.NoiseEstimators
TrackingLoops.NoiseObservation
TrackingLoops.NoiseUpdateContext
append_noise_observation!(::AbstractNoiseEstimator, ::NoiseObservation)
TrackingLoops.get_noise_density(::AbstractNoiseEstimator)
TrackingLoops.get_noise_density_and_ready(::AbstractNoiseEstimator)
TrackingLoops.noise_density_type(::AbstractNoiseEstimator)
TrackingLoops.noise_observation(::Complex, ::Any, ::Any)
TrackingLoops.noise_observation_from_correlator(::Any, ::Any, ::Any, ::Any)
TrackingLoops.noise_observation_from_samples(::Any, ::Any, ::Any)
TrackingLoops.get_noise_window_filling(::AbstractNoiseEstimator)
TrackingLoops.noise_window_looks(::AbstractNoiseEstimator)
TrackingLoops.update_noise!(::AbstractNoiseEstimator, ::Any, ::Any, ::Any, ::Any)
CorrelatorNoiseEstimator
CorrelatorNoiseEstimator()
TrackingLoops.NoiseProbe
TrackingLoops.draw_noise_probe(::CorrelatorNoiseEstimator, ::AbstractGNSSSignal, ::Any)
TrackingLoops.append_noise_correlator_outputs!(::CorrelatorNoiseEstimator, ::NoiseProbe, ::Any, ::Integer, ::Any)
TrackingLoops.NoiseDespread
TrackingLoops.append_despread_noise!(::CorrelatorNoiseEstimator, ::NoiseDespread, ::Any)
append_noise_observation!(::CorrelatorNoiseEstimator{D, T}, ::NoiseObservation) where {D, T}
TrackingLoops.despread_noise!
TrackingLoops.get_noise_density(::CorrelatorNoiseEstimator)
TrackingLoops.get_noise_correlator_bank(::CorrelatorNoiseEstimator)
TrackingLoops.noise_despreads(::CorrelatorNoiseEstimator, ::AbstractGNSSSignal, ::AbstractUnitRange{<:Integer}, ::Any, ::Any)
TrackingLoops.update_noise!(::CorrelatorNoiseEstimator, ::Any, ::Integer, ::Integer, ::NoiseUpdateContext)
```
