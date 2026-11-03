```@meta
CurrentModule = TrackingLoops
```

# TrackingLoops.jl

The arithmetic of a GNSS tracking loop, one coherent integration at a time.

A tracking loop takes the correlator outputs of one integration — early, prompt
and late correlator outputs for one satellite — and turns them into what the next
integration should run with: a carrier Doppler, a code Doppler, a navigation bit
when one has completed, and a C/N₀ estimate. This package is that step, and
nothing around it. It does not correlate, it does not read samples, and it does
not know where its integrations come from or where its Doppler commands go.

That makes it usable from anywhere an integration shows up:

- in a software receiver, where the same process correlates and closes the loop
  ([Tracking.jl](https://github.com/JuliaGNSS/Tracking.jl) is built on it);
- in a dedicated loop process next to an FPGA correlator, where integrations arrive
  over DMA and the corrections are written to NCO registers
  ([HardwareLoopCore.jl](https://github.com/JuliaGNSS/HardwareLoopCore.jl) is
  built on it, and compiles it with `juliac --trim` into a small, allocation-free
  executable);
- in an analysis script that replays logged integrations.

Because all of these run the same code, a loop tuned or debugged on one of them
behaves identically on the others.

## Installation

```julia
using Pkg
Pkg.add("TrackingLoops")
```

TrackingLoops runs on Linux, macOS and FreeBSD. It does not install on Windows
because the navigation-message decoder it depends on
([GNSSDecoder.jl](https://github.com/JuliaGNSS/GNSSDecoder.jl)) needs Aff3ct,
which has no Windows build.

## One loop iteration

Per satellite you hold a tracking-loop state and, per tracked signal of
that satellite, a [`PostCorrelationState`](@ref). For every coherent integration
([`CorrelatorBankOutput`](@ref)) the correlator bank produced, of every signal,
[`process_correlator_bank_output!`](@ref) processes the integration through the signal's post-correlation filter, C/N₀
estimator and bit synchronizer, and [`step_loop!`](@ref) turns it into the Dopplers
the next replica runs with. The loop state knows its satellite's driver
signal, the one whose integrations close the loops; an integration of another signal of
the satellite returns the command in force, so the same four calls serve every
integration and every loop:

```julia
using TrackingLoops, GNSSSignals, Unitful

signal = GPSL1CA()
fs = 4e6u"Hz"
tracking_loop = ConventionalAssistedPLLAndDLL()
loop_state = init_loop_state(tracking_loop, signal, carrier_doppler, code_doppler)  # one per satellite
loop = PostCorrelationState(signal)      # bit synchronizer, C/N₀ estimator, post-correlation filter

# For every coherent integration `correlator_bank_output::CorrelatorBankOutput` the correlator bank produced:
loop, prompt, filtered, code_periods, overshoot =
    process_correlator_bank_output!(loop, signal, prn, correlator_bank_output, fs, noise_density, noise_density_ready)
overshoot && @warn "integration crossed a navigation-bit boundary; bit sync restarted"
integration = LoopRecord(signal, filtered, output, loop, fs; prn)
loop_state, carrier_doppler, code_doppler =
    step_loop!(tracking_loop, loop_state, integration, ConstantNCOCommand(carrier_hz, code_hz), NO_LANDING_SAMPLE)
# program the next replica with carrier_doppler and code_doppler
```

Read the results off the state as they become available:
[`estimate_cn0`](@ref) on `loop.cn0_estimator`,
[`is_bit_or_secondary_code_synced`](@ref) and [`get_soft_bits`](@ref) on
`loop.bit_synchronizer`.

Everything is a plain value or a small mutable state that is preallocated once,
so a loop can be stepped for hours without allocating — provided the consumer
drains the demodulated soft bits ([`get_soft_bits`](@ref)) as they arrive; the soft-bit
vector has room for 64 of them before its vector grows.

## Contents of this manual

- [Correlator banks and discriminators](@ref) — the integration types, their accessors,
  the discriminators and the post-correlation filter.
- [Tracking loops and loop filters](@ref) — the loops behind
  [`step_loop!`](@ref), their bandwidth rules, and the post-correlation processing
  [`process_correlator_bank_output!`](@ref).
- [NCO timeline](@ref) — what a hardware NCO ran and will run.
- [Bit and secondary-code synchronisation](@ref) — the bit synchronizer and the
  per-signal sync detectors.
- [C/N₀ estimation](@ref) — the moments, NWPR and noise-referenced estimators.
- [Noise estimation](@ref) — the noise-density window a noise-referenced
  estimator reads.
- [Vector tracking](@ref) — the loop a navigation filter takes over, and
  the filter that closes every satellite's loops at once.
- [Internals](@ref) — the internal functions and types, for those extending
  the package.

Every name of the API is documented in one of these pages, and only those are part of
the API that the version number makes promises about. The names a caller writes
are exported; the rest — the building blocks of custom correlator banks, tracking loops and
hosts, the per-satellite states and the values results carry — are written
qualified, `TrackingLoops.name`. An enum's type is written
qualified, as `TrackingLoops.CarrierLoopStage`, and so are values a caller rarely
writes, as `TrackingLoops.SYNC_LOST`.

## Package-wide definitions

```@docs
TrackingLoops
```

```@docs
AbstractTrackingLoop
TrackingLoops.NumCorrelators
NumAnts
TrackingLoops.update
```
