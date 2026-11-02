[![Docs: stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaGNSS.github.io/TrackingLoops.jl/stable)
[![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaGNSS.github.io/TrackingLoops.jl/dev)
[![CI](https://github.com/JuliaGNSS/TrackingLoops.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/JuliaGNSS/TrackingLoops.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/JuliaGNSS/TrackingLoops.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaGNSS/TrackingLoops.jl)
[![semantic-release](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--release-e10079.svg)](https://github.com/semantic-release/semantic-release)

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

## What is in it

- **Correlator banks** — `EarlyPromptLateCorrelatorBank`, `VeryEarlyPromptLateCorrelatorBank`
  for one or several antennas, with their correlator output accessors and the sample
  shifts each correlator is placed at.
- **Discriminators and loop filters** — `TrackingLoops.pll_disc`, `TrackingLoops.fll_disc`
  and `TrackingLoops.dll_disc`, the bandwidth rules that keep a loop stable for a given integration time, and the
  code-Doppler aiding from the carrier.
- **Tracking loops** — `ScalarTrackingLoop`, the FLL-assisted
  `ConventionalAssistedPLLAndDLL`, and `DelayCompensatedTrackingLoop`, which stays
  stable when the correction it computes only takes effect several integrations
  later, as it does with a hardware NCO. All three are stepped with one call,
  `step_loop!(tracking_loop, loop_state, integration, nco_history, application_sample_index)`.
- **NCO timelines** — `NCOTimeline` records which replica frequency ran over
  which samples, so an integration can be attributed to the command it was really
  produced under rather than the one that was last requested.
- **Bit and secondary-code synchronisation** — the `TrackingLoops.BitSynchronizer` with the
  bit-edge and overlay-code detectors for every GPS, Galileo and BeiDou signal
  GNSSSignals.jl models.
- **C/N₀ estimation** — moments, NWPR and noise-reference estimators, plus the
  noise-density window a noise-referenced estimator reads.
- **Post-correlation processing** — `process_correlator_bank_output!` advances a signal component's prompt
  filter, C/N₀ estimator and bit synchronizer in one step, so every caller does it the
  same way.
- **Vector tracking** — `VectorTrackingLoop` is another tracking loop, stepped
  with the same `step_loop!`; its navigation processor does the
  whole pipeline from the integrations it is handed: it decodes every satellite's
  navigation bits, solves the PVT, and then closes every satellite's code and
  carrier loops at once with a navigation filter — an unscented Kalman filter
  over position, velocity, one clock per GNSS time system and the
  inter-frequency biases.

Everything is a plain value or a small mutable state that is preallocated once,
so a loop can be stepped for hours without allocating — provided the consumer
drains the demodulated soft bits (`get_soft_bits`) as they arrive; the bit synchronizer
has room for 64 of them before its vector grows.

## Example

```julia
using TrackingLoops, GNSSSignals, Unitful

signal = GPSL1CA()
fs = 4e6u"Hz"
tracking_loop = ConventionalAssistedPLLAndDLL()
loop_state = init_loop_state(tracking_loop, signal, carrier_doppler, code_doppler)  # one per satellite
signal_state = PostCorrelationState(signal)      # bit synchronizer, C/N₀ estimator, post-correlation filter

# For every coherent integration's `correlator_bank_output::CorrelatorBankOutput` the correlator bank produced:
previous_prompt = signal_state.last_filtered_prompt   # read before `process_correlator_bank_output!` replaces it
signal_state, prompt, filtered, code_periods, overshoot =
    process_correlator_bank_output!(signal_state, signal, prn, correlator_bank_output, fs, noise_density, noise_density_ready)
overshoot && @warn "integration crossed a navigation-bit boundary; bit sync restarted"
integration = LoopRecord(signal, filtered, previous_prompt, output, code_periods, fs)
loop_state, carrier_doppler, code_doppler =
    step_loop!(tracking_loop, loop_state, integration, ConstantNCOCommand(carrier_hz, code_hz), NO_LANDING_SAMPLE)
# program the next replica with carrier_doppler and code_doppler
```

See the [documentation](https://JuliaGNSS.github.io/TrackingLoops.jl/stable)
for the full API reference.

## Vector tracking

Build one `VectorTrackingLoop` for all the ranging signals the receiver tracks, and
step every satellite with it like any other loop. The integrations must name their
satellite (`prn`), report the replica's code phase at their end (`code_phase`
on the `CorrelatorBankOutput`) and share one time grid (`sample_offset`):

```julia
tracking_loop = VectorTrackingLoop(GPSL1CA(), GalileoE1B())   # inner = ConventionalAssistedPLLAndDLL()
loop_state = init_loop_state(tracking_loop, signal, carrier_doppler, code_doppler)  # one per satellite

# For every integration, as above, but naming the satellite:
integration = LoopRecord(signal, filtered, previous_prompt, output, code_periods, fs; prn, sample_offset)
loop_state, carrier_doppler, code_doppler = step_loop!(tracking_loop, loop_state, integration, nco_history, application_sample_index)

get_navigation_solution(tracking_loop)   # the latest `PVTSolution`
get_navigation_status(tracking_loop)     # what the latest navigation filter update did
get_satellite_report(tracking_loop, signal, prn)  # immutable snapshot
get_decoder(tracking_loop, signal, prn)   # borrowed, read-only decoder until the next integration
```

Each satellite runs its scalar loop until the filter takes it over. `process_correlator_bank_output!`
handles bit synchronization, demodulation and C/N₀ estimation. `step_loop!` decodes
the navigation data and snapshots host C/N₀ at every navigation epoch; the integration that brings the
last satellite past an epoch runs that epoch's update — the scalar PVT until its
first fix seeds the filter, a filter iteration after that — and each satellite
applies the update's admission, release and corrections on its next integration,
sized for where its command takes effect (at `application_sample_index` with a hardware NCO).
Nothing is logged: `get_navigation_status` reports the events, a satellite's report (`get_satellite_report`)
why it was handed back. Storage is preallocated for
`satellite_capacity_per_signal` satellites per signal and reused as satellites come
and go; once warm, integrations and updates allocate nothing and compile with
`juliac --trim=safe` (see `test/trim`).

## Platforms

TrackingLoops runs on Linux, macOS and FreeBSD. It does not install on Windows
because the navigation-message decoder it depends on (GNSSDecoder.jl) needs
Aff3ct, which has no Windows build.
