# Changelog

# [6.1.0](https://github.com/JuliaGNSS/TrackingLoops.jl/compare/v6.0.0...v6.1.0) (2026-10-22)


### Features

* **vector:** report the carrier phase correction the residual PLL estimates ([2b82c9f](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/2b82c9fe2b967885fcd09b38fd98cc65f817baf4))

# [6.0.0](https://github.com/JuliaGNSS/TrackingLoops.jl/compare/v5.1.2...v6.0.0) (2026-10-21)


* feat(estimators)!: carry the bit clock's sync polarity on LoopRecord ([1abc7be](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/1abc7bef9fb3689ae0ddd0f8c6207ce949d8e2fa))
* feat(vector)!: set the carrier NCO from the navigation filter alone ([8f41ff3](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/8f41ff3bd9c1399fc30342e81c98578b1bef5da8))
* refactor(vector)!: rename member_sats to vector_loop_members ([8fe05b9](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/8fe05b920ad995673d009e88c359b524de9e4503))
* refactor(vector)!: rename VectorPLLAndDLL's cycle_time to vt_update_interval ([ad4e0a6](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/ad4e0a6f63945d23bdde95aa8ca4d12b239c8ef1))
* refactor(vector)!: rename VectorTracking to NavigationFilterConfig ([785e5f0](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/785e5f0de2061c85a6661ddcf29a84d4a65da746))
* refactor(vector)!: rename VTStatus to NavigationStatus and its enabled to seeded ([8938ac5](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/8938ac5805253a21d1e720d8de6a54f0fe934c6d))


### Bug Fixes

* **vector:** apply the elevation mask before the measurement update ([2813e9d](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/2813e9d058c8fa6fc6053e4f1756a7552da5d00e))


### Features

* **estimators:** report whether the carrier loop is phase-locked ([874f104](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/874f104dfc13d88b4464814e1f45c3cfb13175d6))


### BREAKING CHANGES

* member_sats is vector_loop_members. Call
vector_loop_members(estimator) where member_sats(estimator) was called;
the result is unchanged.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
* VectorTracking is NavigationFilterConfig and
VectorPLLAndDLL's config keyword is filter_config. Construct
NavigationFilterConfig(; ...) where VectorTracking(; ...) was
constructed and pass it as filter_config = ... where config = ... was
passed; filter_config = nothing still solves the scalar PVT only. The
keywords and the defaults are unchanged.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
* VectorPLLAndDLL's cycle_time keyword is
vt_update_interval. Pass vt_update_interval = … where cycle_time = … was
passed; the value and its meaning are unchanged.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
* VTStatus is NavigationStatus and its enabled field is
seeded. Replace VTStatus with NavigationStatus and status.enabled with
status.seeded; the other fields are unchanged.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
* the vector loop's FLL branch is gone, and with it
VectorPLLAndDLL's vector_fll_loop_filter_bandwidth keyword,
LoopBandwidths' vector_fll field and keyword,
default_vector_fll_loop_filter_bandwidth,
effective_vector_fll_loop_filter_bandwidth and
MAX_VECTOR_FLL_LOOP_BANDWIDTH_CYCLE_PRODUCT. Drop them; set the residual
PLL's bandwidth with VectorPLLAndDLL's residual_pll_loop_filter_bandwidth
keyword instead. While a satellite is in the vector loop,
carrier_loop_stage reports VECTOR_LOOP, a new CarrierLoopStage value,
where it reported NARROW_PLL (compare against VECTOR_LOOP there), and
phase_lock_indicator reads the residual PLL's indicator, on the prompts
it rotated, where it read the frozen inner loop's. A fallback releases,
as VT_FALLBACK, every satellite still in the vector loop, those that
missed the cycle's epoch too, which stayed in the loop before.
CarrierLoopStaging's fields changed: its phase-lock averages and
indicator are a PhaseLockDetector in its lock_detector field. Code that
reads them, or builds a staging field by field, follows;
CarrierLoopStaging(filter_type) is unchanged.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
* LoopRecord gains a last field, bit_polarity. A host
calling LoopRecord's full positional constructor passes it (0 before the
sync); the other constructors set it.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

## [5.1.2](https://github.com/JuliaGNSS/TrackingLoops.jl/compare/v5.1.1...v5.1.2) (2026-10-20)


### Bug Fixes

* **vector:** never lock a satellite stepped past an epoch out of the next ones ([610b7a7](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/610b7a793bac6e25e5d961f41c63861656eeb7da))

## [5.1.1](https://github.com/JuliaGNSS/TrackingLoops.jl/compare/v5.1.0...v5.1.1) (2026-10-18)


### Bug Fixes

* **vector:** predict the pseudorange rate as the derivative of the modelled pseudorange ([0e3eb0b](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/0e3eb0bb599d8e2a670ac9f68f0688573b4d1e49))

# [5.1.0](https://github.com/JuliaGNSS/TrackingLoops.jl/compare/v5.0.2...v5.1.0) (2026-10-16)


### Features

* **vector:** run the vector loop's FLL branch at its own 1 Hz bandwidth ([f033f58](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/f033f58246d080caa62e90ce1dc7240dd8f0d4c9))

## [5.0.2](https://github.com/JuliaGNSS/TrackingLoops.jl/compare/v5.0.1...v5.0.2) (2026-10-15)


### Bug Fixes

* **vector:** measure the pseudorange rate from the replica's phase advance over the cycle ([271eda3](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/271eda3ee0e88bc94be4153e2e6b14691eb9934b))

## [5.0.1](https://github.com/JuliaGNSS/TrackingLoops.jl/compare/v5.0.0...v5.0.1) (2026-10-15)


### Bug Fixes

* **vector:** require GNSSDecoder 5.0.3 so a B1C slot taken over by another satellite syncs on its PRN ([c064408](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/c0644081bec58479e432e902e9a8a9270a21e259)), closes [JuliaGNSS/GNSSDecoder.jl#105](https://github.com/JuliaGNSS/GNSSDecoder.jl/issues/105)

# [5.0.0](https://github.com/JuliaGNSS/TrackingLoops.jl/compare/v4.0.1...v5.0.0) (2026-10-14)


* feat(estimators)!: read a synced pilot's records with the four-quadrant discriminators ([0479192](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/047919249e30b8c007e1550fc5a2e0d0d92d7d78))
* feat(estimators)!: stage the carrier loop on phase lock with wide, narrow and FLL-assist bandwidths ([5541252](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/55412526b18844f3cd0c0e32475ddae1d41823b5))
* refactor(discriminators)!: return the PLL phase error in cycles ([24bd911](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/24bd9114eff13964635b42584ef342e9e39bb4a3))
* refactor(loop_filters)!: remove calculate_carrier_frequency_update ([8f92ab0](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/8f92ab0241c3aa65ee25e2081266138ba2bc2923))


### Features

* **discriminators:** add four-quadrant PLL and FLL discriminators ([4f28322](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/4f283227b865535322645f9fead24bbf9a786aa1))


### BREAKING CHANGES

* the carrier bandwidth is split by the stage it applies to.
The estimator keyword and field carrier_loop_filter_bandwidth of
ConventionalPLLAndDLL, ConventionalAssistedPLLAndDLL and
NCOReferencedPLLAndDLL becomes wide_carrier_loop_filter_bandwidth, next to
the new narrow_carrier_loop_filter_bandwidth and
fll_assist_loop_filter_bandwidth; their all-positional constructors take
(wide, narrow, fll_assist, code), and NCOReferencedPLLAndDLL's
predict_landing after them. default_carrier_loop_filter_bandwidth is
replaced by default_wide_carrier_loop_filter_bandwidth (50 Hz),
default_narrow_carrier_loop_filter_bandwidth (18 Hz) and
default_fll_assist_loop_filter_bandwidth (5 Hz); a method of the first does
not widen the NCO-referenced loop, which pulls in at the narrow default.
SatConventionalPLLAndDLL's and SatNCOReferencedPLLAndDLL's
carrier_loop_filter_bandwidth and code_loop_filter_bandwidth fields become
one field bandwidths::LoopBandwidths: read state.bandwidths.wide_carrier,
.narrow_carrier, .fll_assist and .code, and pass bandwidths =
LoopBandwidths(...) to SatConventionalPLLAndDLL's keyword constructors. Both
gain a trailing staging field, read with carrier_loop_stage(state) and
phase_lock_indicator(state), which their keyword constructors start; code
building them positionally must pass the bandwidths and the staging,
TrackingLoops.CarrierLoopStaging(typeof(carrier_loop_filter)) (not
exported). The loops no
longer run the PLL at one 18 Hz with the FLL fed on every record: they pull
in at 50 Hz FLL-assisted, drop the FLL and narrow to 18 Hz on phase lock, so
tracking results differ. For one PLL bandwidth throughout, pass the same
wide and narrow bandwidth. reset_estimator_state keeps the staging, so after
it the loop runs on at the stage it had reached; a release from the vector
loop restarts it. A wider loop tolerates
less command delay: at the default 50 Hz the conventional loop holds through
three records of delay, and at 85 Hz the NCO-referenced loop through five
instead of six once the FLL is dropped.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
* fold_record and apply_record take the pre-sync prompts of a
dataless signal with a secondary code (GPS L1C-P and L5Q, Galileo E1C,
E5a-Q, E5b-Q and E6-C, BeiDou B1C-P and B2a-Q) as primary-only, correlated
with a replica that carries no secondary code; a host whose pre-sync replica
carries a secondary chip on every block (as Tracking.jl's carries chip 0)
multiplies such a signal's pre-sync records by that chip, or the
four-quadrant PLL reads a half-cycle error for the PRNs whose chip is
negative. The new is_dataless_with_secondary_code(signal) picks those
signals, so that a host and the carrier phase polarity agree on them. Other
signals need no change. fold_record's positional argument
correlated_pre_sync and apply_record's keyword correlated_pre_sync are
replaced by the keyword overlay_wiped, which defaults to the bit buffer
having synced before the record: pass overlay_wiped = false where
correlated_pre_sync = true was passed, drop correlated_pre_sync = false, and
drop the positional argument from a fold_record call. The LoopRecord
constructor that takes a SignalLoopState no longer takes a previous prompt:
call LoopRecord(signal, filtered_correlator, output, state,
sampling_frequency) in place of LoopRecord(signal, filtered_correlator,
previous_prompt, output, state, sampling_frequency), and drop the read of
state.last_filtered_prompt before apply_record. LoopRecord gains the
trailing field carrier_phase_polarity::Int8, and SignalLoopState the
trailing fields last_previous_prompt::ComplexF64 and
last_carrier_phase_polarity::Int8; code calling their all-positional
constructors must append Int8(0), respectively a zero prompt and Int8(0).
The other LoopRecord constructors default the polarity. The new
SignalLoopState(state; bit_buffer, cn0_estimator, ...) replaces the fields
passed and keeps the others, so code that rebuilds a state to swap a part of
it can use it and need not list every field. A host that builds
records without a SignalLoopState passes carrier_phase_polarity as the ninth
value fold_record returns, and a zero previous prompt where it differs from
the previous record's, to get the four-quadrant discriminators; without it
it tracks as before. Code destructuring fold_record's result positionally
receives the additional value last. wrap_half_cycle is renamed
wrap_phase_error and no longer exported, as only the NCO-referenced loop
needs it: call TrackingLoops.wrap_phase_error(x), which folds into ±1/4
cycle as wrap_half_cycle(x) did, and TrackingLoops.wrap_phase_error(x;
polarity) with a nonzero polarity into ±1/2 cycle.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
* pll_disc returns the carrier phase error in cycles (±1/4
cycle) instead of radians (±π/2). Multiply its result by 2π where radians
are needed, and drop a division by 2π where it was fed to a loop filter.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
* calculate_carrier_frequency_update is removed. Step a
carrier loop through an estimator's step_loop, or call the loop filter
directly: filter_loop(carrier_loop_filter, phase_error, integration_time,
bandwidth) with the PLL discriminator's phase error in cycles, and with
(phase_error, fll_disc(...)) as the input of a ThirdOrderAssistedBilinearLF.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

## [4.0.1](https://github.com/JuliaGNSS/TrackingLoops.jl/compare/v4.0.0...v4.0.1) (2026-10-11)


### Bug Fixes

* **bit_buffer:** let the bit-edge noise scale forget a transient ([e7b36e1](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/e7b36e1ffe75497e81a991bd8ebacecc7598dd40))

# [4.0.0](https://github.com/JuliaGNSS/TrackingLoops.jl/compare/v3.0.1...v4.0.0) (2026-10-10)


* feat(vector)!: range on a pilot and decode its data component ([542765b](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/542765b3338b6a273cf7780a4e2a3623e61146e4)), closes [#36](https://github.com/JuliaGNSS/TrackingLoops.jl/issues/36)
* fix(loop_filters)!: give the carrier loop its configured bandwidth, by default a capped 18 Hz ([55e764f](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/55e764ffa8c6347aa5405afebd0bcbab44a8c339)), closes [Tracking.jl#244](https://github.com/Tracking.jl/issues/244) [#245](https://github.com/JuliaGNSS/TrackingLoops.jl/issues/245)


### Bug Fixes

* **discriminators:** return the FLL discriminator in Hz whatever the integration time's unit ([947ed42](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/947ed429c4e16c2325b252d9920b20cd114dfdfc))
* **vector:** leave another pair's data component to the satellite it rides on ([1d64a7a](https://github.com/JuliaGNSS/TrackingLoops.jl/commit/1d64a7a625e831c0f5824a8738692df6b0b3f941))


### BREAKING CHANGES

* VectorPLLAndDLL reads the bit sync, soft bits and C/N₀ from
the records, so build them with `LoopRecord(signal, filtered, previous_prompt,
output, state::SignalLoopState, fs; prn)` after `apply_record`, and call
`step_loop` on every record of a satellite. Its C/N₀ comes from the host's
estimator (NoiseRefCN0Estimator by default, out of lock until it has a noise
density); the `num_prompts_for_cn0_estimation` keyword is gone. A dataless
signal must be paired with its data component. SatConventionalPLLAndDLL and
SatNCOReferencedPLLAndDLL gain a trailing `driver` field, the key of the
driver signal `init_estimator_state` records (`0`, the keyword default, takes
every record to be the driver's); build states with `init_estimator_state`.
SignalLoopState and LoopRecord gained fields. A record of a signal the vector
estimator does not know is ignored unless it is the satellite's driver.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
* carrier loops now have the bandwidth they are configured
with, and the default is a flat 18 Hz capped at `0.09 / T_int`. Before,
a loop's actual noise bandwidth was about 5–6× the configured one (about
105 Hz for a 1 ms GPS L1 C/A loop), and the default was `0.018 / T`
divided by the number of coherently integrated code blocks. This applies
to `ConventionalPLLAndDLL`, `ConventionalAssistedPLLAndDLL`,
`NCOReferencedPLLAndDLL`, the inner loop of `VectorPLLAndDLL` and
`calculate_carrier_frequency_update`; `pll_disc` is unchanged.
`default_carrier_loop_filter_bandwidth` returns 18 Hz for every signal,
e.g. instead of 4.5 Hz for Galileo E1B and 1.8 Hz for GPS L1C, and a
carrier bandwidth (default or explicit) is no longer divided by N but
capped at `0.09 / T_int`: an explicit 18 Hz on GPS L1 C/A integrated
over 20 ms runs at 4.5 Hz instead of 0.9 Hz. The 1 ms loops are about
5–6× narrower than before, with less thermal jitter but less dynamic
tolerance, a smaller pull-in range and slower settling. Code that set
`carrier_loop_filter_bandwidth`, or relied on the defaults, for a given
dynamic behaviour must re-tune it; to get approximately the previous
loop back, configure 85 Hz, e.g.
`ConventionalAssistedPLLAndDLL(; carrier_loop_filter_bandwidth = 85.0Hz)`.
Code that overrode `default_carrier_loop_filter_bandwidth` assuming its
value would be divided by N must now return the bandwidth it wants at the
actual integration length. `MAX_LOOP_BANDWIDTH_TIME_PRODUCT` is removed:
use `MAX_CODE_LOOP_BANDWIDTH_TIME_PRODUCT` (the same 0.018) for the code
loop, and `MAX_CARRIER_LOOP_BANDWIDTH_TIME_PRODUCT` (0.09) with
`effective_carrier_loop_filter_bandwidth` for the carrier loop.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

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
