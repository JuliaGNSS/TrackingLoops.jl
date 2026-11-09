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

Correlator banks, discriminators and loop filters, the scalar tracking loops
(`ScalarTrackingLoop`, `DelayCompensatedTrackingLoop`), NCO timelines, bit and
secondary-code synchronisation, C/N₀ and noise-density estimation, post-correlation
processing, and vector tracking (`VectorTrackingLoop`, a navigation filter closing every
satellite's loops). Every loop is stepped with one call,

```julia
loop_state, carrier_doppler, code_doppler =
    step_loop!(tracking_loop, loop_state, post_correlation_state, nco_history, application_sample_index)
```

and a warm loop allocates nothing. The
[documentation](https://JuliaGNSS.github.io/TrackingLoops.jl/stable) walks through a
host's calls, each part of the package, the API reference and a glossary of its terms.

## Platforms

TrackingLoops runs on Linux, macOS and FreeBSD. It does not install on Windows
because the navigation-message decoder it depends on (GNSSDecoder.jl) needs
Aff3ct, which has no Windows build.
