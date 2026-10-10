# TrackingLoops.jl

The arithmetic of a GNSS tracking loop, one correlator record at a time.

A tracking loop takes the correlator outputs of one integration — early, prompt
and late accumulators for one satellite — and turns them into what the next
integration should run with: a carrier Doppler, a code Doppler, a navigation bit
when one has completed, and a C/N₀ estimate. This package is that step, and
nothing around it. It does not correlate, it does not read samples, and it does
not know where its records come from or where its Doppler commands go.

That makes it usable from anywhere a record shows up:

- in a software receiver, where the same process correlates and closes the loop
  ([Tracking.jl](https://github.com/JuliaGNSS/Tracking.jl) is built on it);
- in a dedicated loop process next to an FPGA correlator, where records arrive
  over DMA and the corrections are written to NCO registers
  ([HardwareLoopCore.jl](https://github.com/JuliaGNSS/HardwareLoopCore.jl) is
  built on it, and compiles it with `juliac --trim` into a small, allocation-free
  executable);
- in an analysis script that replays logged records.

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

Per satellite you hold a Doppler-estimator state and, per tracked signal of
that satellite, a [`SignalLoopState`](@ref). For every correlator record
([`CorrelatorOutput`](@ref)) the correlator produced, of every signal,
[`apply_record`](@ref) folds the record into the signal's prompt filter, C/N₀
estimator and bit buffer, and [`step_loop`](@ref) turns it into the Dopplers
the next replica runs with. The estimator state knows its satellite's driver
signal, the one whose records close the loops; a record of another signal of
the satellite returns the command in force, so the same four calls serve every
record and every estimator:

```julia
using TrackingLoops, GNSSSignals, Unitful

signal = GPSL1CA()
fs = 4e6u"Hz"
estimator = ConventionalAssistedPLLAndDLL()
state = init_estimator_state(estimator, signal, carrier_doppler, code_doppler)  # one per satellite
loop = SignalLoopState(signal)              # bit buffer, C/N₀ estimator, prompt filter

# For every correlator record `output::CorrelatorOutput` the correlator produced:
previous_prompt = loop.last_filtered_prompt   # read before `apply_record` replaces it
loop, prompt, filtered, blocks, overshoot =
    apply_record(loop, signal, prn, output, fs, noise_density, noise_density_ready)
overshoot && @warn "record crossed a navigation-bit boundary; bit sync restarted"
record = LoopRecord(signal, filtered, previous_prompt, output, loop, fs; prn)
state, carrier_doppler, code_doppler =
    step_loop(estimator, state, record, FixedNCOWord(carrier_hz, code_hz), NO_LANDING_SAMPLE)
# program the next replica with carrier_doppler and code_doppler
```

Read the results off the state as they become available:
[`estimate_cn0`](@ref) on `loop.cn0_estimator`,
[`has_bit_or_secondary_code_been_found`](@ref) and [`get_soft_bits`](@ref) on
`loop.bit_buffer`.

Everything is a plain value or a small mutable state that is preallocated once,
so a loop can be stepped for hours without allocating — provided the consumer
drains the decoded soft bits ([`get_soft_bits`](@ref)) as they arrive; the bit
buffer has room for 64 of them before its vector grows.

## Contents of this manual

- [Correlators and discriminators](@ref) — the record types, their accessors,
  the discriminators and the post-correlation filter.
- [Doppler estimators and loop filters](@ref) — the estimators behind
  [`step_loop`](@ref), their bandwidth rules, and the per-record fold
  [`apply_record`](@ref).
- [NCO timeline](@ref) — what a hardware NCO ran and will run.
- [Bit and secondary-code synchronisation](@ref) — the bit buffer and the
  per-signal sync detectors.
- [C/N₀ estimation](@ref) — the moments, NWPR and noise-referenced estimators.
- [Noise estimation](@ref) — the noise-density window a noise-referenced
  estimator reads.
- [Vector tracking](@ref) — the estimator a navigation filter takes over, and
  the filter that closes every satellite's loops at once.
- [Internals](@ref) — the unexported functions and types, for those extending
  the package.

Every exported name is documented in one of these pages. Only the exported
names are part of the public API that the version number makes promises about.

## Package-wide definitions

```@docs
TrackingLoops
```

```@autodocs
Modules = [TrackingLoops]
Pages = ["TrackingLoops.jl"]
Private = false
```
