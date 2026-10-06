# Changelog

## [3.0.1](https://github.com/JuliaGNSS/TrackingLoops.jl/compare/v3.0.0...v3.0.1) (2026-10-06)


### Bug Fixes

* **vector:** keep the fixed seed for a singular fix design ([bf2d8df](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/bf2d8dfe50fa66bde62af319a394a06b7d46f439))
* **vector:** seed the filter covariance from the geometry of the scalar fix ([f01d865](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/f01d865d9d01bd539ac7eb1c753ab3f87f4d47c3)), closes [#16](https://github.com/JuliaGNSS/TrackingLoops.jl/issues/16)

# [3.0.0](https://github.com/JuliaGNSS/TrackingLoops.jl/compare/v2.0.0...v3.0.0) (2026-10-05)


* feat(vector)!: run the whole vector-tracking pipeline inside the estimator ([0531407](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/05314072560b6cf34d02014563db5ad20740581e))


### Bug Fixes

* **vector:** release a satellite that rejoins after being dropped while in the vector loop ([feb1e57](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/feb1e57402cd3579c7956acd702ea6476cfc1b8b))


### Features

* let every estimator report the navigation solution, its cycle and its satellites ([525ab35](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/525ab35c031b4b0927e8625e60826ecc15e8f12f))
* let records identify their satellite and replica code phase ([61fe804](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/61fe804288b4f47ec4c7b38df9d5695d91c16a49))


### BREAKING CHANGES

* `VectorPLLAndDLL(inner)` is now
`VectorPLLAndDLL(signals...; inner = ConventionalAssistedPLLAndDLL(), config,
cycle_time = 100ms, lock_cn0_threshold = 30dBHz, max_satellites_per_signal,
approximate_year, enable_ionospheric_correction,
enable_tropospheric_correction)`: list every ranging signal the satellites
use, and pass what used to go to `VectorTrackingState` here. The records it
is stepped with must carry `prn` and `code_phase` (`LoopRecord(...; prn,
sample_offset)`, `CorrelatorOutput(..., code_phase)`) on a time grid shared
by all satellites; a dataless pilot signal is rejected. `VTSat`,
`VTSignalGroup`, `VectorTrackingState` and `update_navigation!` are removed:
step every satellite through `step_loop`, and read
`navigation_solution(estimator)` and `navigation_status(estimator)` instead
of `update_navigation!`'s `(pvt, status)`, `release_reason(estimator,
signal, prn)` instead of `VTSat.release_reason`, and `member_sats(estimator)`
instead of `vt.member_sats`. `position_uncertainty` and `clock_uncertainty`
take the estimator. `decode_soft_bits!` is removed: the estimator decodes
the bits itself. `enable_vector_tracking`, `disable_vector_tracking`,
`set_vector_corrections`, `release_from_vector_tracking`,
`reset_discriminator_accumulators`, `mean_code_discriminator` and
`mean_carrier_discriminator` are no longer exported: the engine drives the
satellite states itself.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

# [2.0.0](https://github.com/JuliaGNSS/TrackingLoops.jl/compare/v1.1.2...v2.0.0) (2026-10-04)


* build!: depend on PositionVelocityTime, GNSSDecoder and KalmanFilters ([10d8103](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/10d810301a866e6012895e896fae1ba719c624ec))


### Bug Fixes

* **vector:** admit a satellite only once it stands a degree above the horizon ([2490df1](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/2490df16ff7a506a9c7fa589c51a37541da14dbd))
* **vector:** hand a released satellite over at the replica where its command lands ([aad163a](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/aad163a385351e4bec6c111e8f5b9372ea846006))
* **vector:** keep the code-phase advance right across resets and long NCO delays ([f81e5ec](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/f81e5ecefcc2a3ae5dffd9992352638335a58a18))
* **vector:** propagate each cycle by its measured length ([92503d6](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/92503d65457d07b294f6a3a4eb96ecf8c3581e2e))
* **vector:** report only the biases a cycle measured ([815c57f](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/815c57f7b3f9b9ea0dfaf4bd4a978d2fd9671384))
* **vector:** seed from a fresh fix without stale members or a stale epoch offset ([6f383cc](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/6f383ccff6bf802e2b67a82b56e75e63ea392e1d))


### Features

* add the per-record VectorPLLAndDLL around a chosen scalar loop ([f358026](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/f358026d2a22022f819fe8978dadcf00b183e5c2))
* add the vector-tracking filter model ([3c80f7c](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/3c80f7c370a7bcfb478ca647bbc30eed51ee73a3))
* close the loops with update_navigation! ([58d3dee](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/58d3deeb9b76f53bc5f16af1d20cc63c6c48ad4f))


### Performance Improvements

* **vector:** reuse the cycle's predictions when a correction lands at the epoch ([bc43700](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/bc437007c60522cce5eb8c17ee7f8a4bbe604c50))


### BREAKING CHANGES

* TrackingLoops no longer installs on Windows, because
GNSSDecoder's Aff3ct dependency has no Windows build. Linux, macOS and
FreeBSD are unaffected. Stay on TrackingLoops 1.x on Windows.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>

## [1.1.2](https://github.com/JuliaGNSS/TrackingLoops.jl/compare/v1.1.1...v1.1.2) (2026-10-03)


### Bug Fixes

* **discriminators:** refuse VEML taps that all miss the correlation peak ([d16a236](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/d16a23610d52ad6f2b97ad460c0b80d331accaad))

## [1.1.1](https://github.com/JuliaGNSS/TrackingLoops.jl/compare/v1.1.0...v1.1.1) (2026-09-30)

No changes to the package. Replaces the accidental 2.0.0 release.

# [1.1.0](https://github.com/JuliaGNSS/TrackingLoops.jl/compare/v1.0.1...v1.1.0) (2026-09-24)


### Bug Fixes

* **nco_timeline:** refuse a zero capacity and share mean_nco_word's boundary ([0cc577f](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/0cc577fff0f64c38c4d273dacc87e8cfe3eb73f7))
* **record:** empty every C/N₀ estimator when a channel is re-armed ([76d56f3](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/76d56f3217912591917bae52a65811e7d1ea565b))


### Features

* export CorrelatorNoiseEstimator ([d5d9530](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/d5d95302b5794670705410ab36985ade30dcf22a))
* **record:** report a bit-boundary overshoot from apply_record ([850e0ca](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/850e0cad3d6e39a928e566a90ebebdef7f927cea))

## [1.0.1](https://github.com/JuliaGNSS/TrackingLoops.jl/compare/v1.0.0...v1.0.1) (2026-09-23)


### Bug Fixes

* **docs:** name the owner of every cross-reference that leaves this package ([658a5cf](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/658a5cf7a3a9086e4ed861710498be577d1cc863))

# 1.0.0 (2026-09-23)


### Features

* the per-record tracking-loop arithmetic, extracted from Tracking.jl ([76c5269](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/76c5269c7ba37b44d5129fae7f1ec24146bb3848))
