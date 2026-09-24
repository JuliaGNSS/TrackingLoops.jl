[![CI](https://github.com/JuliaGNSS/TrackingLoops.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/JuliaGNSS/TrackingLoops.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/JuliaGNSS/TrackingLoops.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaGNSS/TrackingLoops.jl)

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

## What is in it

- **Correlators** — `EarlyPromptLateCorrelator`, `VeryEarlyPromptLateCorrelator`
  for one or several antennas, with their accumulator accessors and the sample
  shifts each tap is placed at.
- **Discriminators and loop filters** — `pll_disc`, `fll_disc`, `dll_disc`, the
  bandwidth rules that keep a loop stable for a given integration time, and the
  code-Doppler aiding from the carrier.
- **Doppler estimators** — `ConventionalPLLAndDLL`, the FLL-assisted
  `ConventionalAssistedPLLAndDLL`, and `NCOReferencedPLLAndDLL`, which stays
  stable when the correction it computes only takes effect several records
  later, as it does with a hardware NCO. All three are stepped with one call,
  `step_loop(estimator, state, record, words, landing_sample)`.
- **NCO timelines** — `NCOTimeline` records which replica frequency ran over
  which samples, so a record can be attributed to the word it was really
  produced under rather than the one that was last requested.
- **Bit and secondary-code synchronisation** — the `BitBuffer` with the
  bit-edge and overlay-code detectors for every GPS, Galileo and BeiDou signal
  GNSSSignals.jl models.
- **C/N₀ estimation** — moments, NWPR and noise-reference estimators, plus the
  noise-density window a noise-referenced estimator reads.
- **The per-record fold** — `apply_record` advances a signal component's prompt
  filter, C/N₀ estimator and bit buffer in one step, so every caller does it the
  same way.

Everything is a plain value or a small mutable state that is preallocated once,
so a loop can be stepped for hours without allocating — provided the consumer
drains the decoded soft bits (`get_soft_bits`) as they arrive; the bit buffer
has room for 64 of them before its vector grows.

## Example

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
record = LoopRecord(signal, filtered, previous_prompt, output, blocks, fs)
state, carrier_doppler, code_doppler =
    step_loop(estimator, state, record, FixedNCOWord(carrier_hz, code_hz), NO_LANDING_SAMPLE)
# program the next replica with carrier_doppler and code_doppler
```

See the docstrings of `step_loop`, `apply_record`, `NCOTimeline` and the
estimators for the full signatures.
