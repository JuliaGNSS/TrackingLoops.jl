# Changelog

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
