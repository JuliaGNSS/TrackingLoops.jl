[![CI](https://github.com/JuliaGNSS/TrackingLoops.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/JuliaGNSS/TrackingLoops.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/JuliaGNSS/TrackingLoops.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaGNSS/TrackingLoops.jl)

# TrackingLoops.jl

The per-record tracking-loop arithmetic of [Tracking.jl](https://github.com/JuliaGNSS/Tracking.jl),
as a package of its own — and, on top of it, the loop core of a hardware
correlator's dedicated loop process.

Tracking.jl imports everything here (correlators, discriminators, loop filters,
bit and secondary-code sync, C/N₀ estimators, the delay-aware
`NCOReferencedPLLAndDLL` with its NCO word timelines) and re-exports it, so the
software receiver's `track!` is unchanged and bit-identical. Nothing here knows a device or a shared-memory
segment: the loop process's engine on top of it is HardwareLoopCore.jl.

The loop engine is HardwareLoopCore.jl, the receiver side of its protocol is
GNSSReceiver.jl's `RemoteHardwareLoop`, and the LiteX-M2SDR driver and the
`gnss_loop` executable are in GNSSM2SDR.jl's `M2SDRLoop/`. See GNSSReceiver.jl's
`docs/plans/2026-09-22-loop-process.md`.
