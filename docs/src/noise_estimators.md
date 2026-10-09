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

## Why per signal and not per RF band

What a record actually divides by is the **post-correlation** density. For an
interferer of PSD `S_I(f)` on top of the thermal floor that is

```
N₀,eff = N₀,thermal + ∫ S_I(f) · |G(f)|² df
```

— the spectral separation coefficient, weighted by the *despreading modulation's*
own spectrum `|G(f)|²`. Two signals sharing one band, one antenna and one front
end therefore see different floors the moment the interference is not white, and
the difference is not small: BPSK(1) has its peak at DC and a null at
±1.023 MHz, BOC(1,1) is the reverse, so a CW tone at band centre is rejected by
Galileo E1B and lands squarely in GPS L1 C/A, while a tone at ±1.023 MHz does the
opposite. Front-end tilt, filter roll-off at the band edge and adjacent-band
leakage all colour the floor the same way, more quietly.

Because [`CorrelatorNoiseEstimator`](@ref) *despreads* rather than metering
power, keying by signal makes that integral **measured rather than modelled**:
the reference runs the consumer's own code, so its spectral weighting is the
consumer's by construction. This is the same argument that makes the reference
backend-free — it traverses the identical path as the prompt — extended from the
quantiser to the interference environment.

It also matches the hardware. A noise channel is a tracking channel with a wrong
PRN, and a tracking channel is configured with a code; per-signal is what an FPGA
would build anyway.

## Why a density and not a power

For a correlator normalised the way `TrackingLoops.normalize` normalises — by
`integrated_samples * code_amplitude` — white input noise of per-sample variance
`σ²` gives `E|P|² = σ²/N = N₀/T`. So `N₀ = σ²/f_s` is **independent of the
integration time**, which is what lets one per-signal figure serve records of any
length. It is stored as a Unitful quantity of dimension `1/Hz`, so the
consumer's `⟨|P|²⟩/N₀ − 1/T` is dimension-checked rather than trusted.

```@autodocs
Modules = [TrackingLoops]
Pages = ["noise_estimators/noise_estimator.jl", "noise_estimators/window.jl"]
Private = false
```
