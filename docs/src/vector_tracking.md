```@meta
CurrentModule = TrackingLoops
```

# Vector tracking

In scalar tracking every satellite closes its own code and carrier loops from
its own discriminators. In vector tracking a navigation filter closes them all
at once: each satellite's accumulated DLL and FLL discriminators become
pseudorange and pseudorange-rate measurements, the filter fuses them into a
position, velocity and clock state and sets every satellite's NCOs from it (the
carrier to the predicted Doppler, the code by a correction). Weak or briefly
obscured satellites are carried through by the common solution instead of losing
lock on their own.

The filter is an unscented Kalman filter over position, velocity, one clock bias
per GNSS time system (all driven by the one oscillator's drift) and one
inter-frequency bias per band beyond the reference band. It follows the bias
model of [PositionVelocityTime.jl](https://github.com/JuliaGNSS/PositionVelocityTime.jl):
which biases a cycle can determine is decided per cycle, and where the
measurements cannot separate two clocks, a broadcast offset between their
systems (GGTO, BGTO) ties them as a pseudo-measurement. With `use_pseudorange_rates = true` (the
default) it also measures the pseudorange rates (VDFLL); with `false` it is a
VDLL, and the carrier NCOs' Doppler comes from the code measurements alone.

`NavigationFilterConfig` selects the dynamics with `motion_model_order = 2`
and `clock_model_order = 2` by default. Motion orders `1`, `2`, and `3` carry
position, position and velocity, and position, velocity and acceleration,
respectively. Clock order `1` carries per-system biases; order `2` also carries
the common oscillator drift. `acceleration_noise_std` describes platform
manoeuvres at every order; `h0` and `hm2` retain the oscillator's Allan-variance
coefficients.

## One tracking loop does it all

[`VectorTrackingLoop`](@ref) is a tracking loop: a host builds each satellite's
state with [`init_loop_state`](@ref) and steps it with [`step_loop!`](@ref) on
every integration. Everything vector tracking needs happens inside that step, from
what the integrations carry. All satellites stepped with one loop share its navigation
processor, which

- decodes each satellite's navigation data from the demodulated soft bits its
  integrations carry;
- snapshots each satellite at every navigation epoch (every `navigation_filter_update_interval`
  on the integrations' time grid): code phase from the last data-bit edge, decoder,
  Dopplers, C/N₀ and the discriminators accumulated since the last epoch;
- runs the epoch's navigation cycle on the integration that completes the last
  satellite's snapshot: the scalar PVT until the filter is seeded, a filter
  iteration after that;
- leaves the cycle's decisions (admission, release, commands) for each
  satellite to take up on its next integration, sized for where its command takes effect.

```julia
signal = GPSL1CA()
tracking_loop = VectorTrackingLoop(signal, GalileoE1C() => GalileoE1B())
loop_state = init_loop_state(tracking_loop, signal, carrier_doppler, code_doppler)
get_navigation_solution(tracking_loop)  # PVTSolution; get_navigation_status(tracking_loop)
```

Several constellations or bands share one filter: list every ranging signal at
construction. Each satellite runs the scalar loop `scalar_loop` until the filter takes
it over, and again once the filter lets it go. In the vector loop the code filter
is frozen, the filter sets the replica, and the measurements are accumulated for
the filter (see [The carrier in the vector loop](@ref vector-carrier)).

## Pilot + data pairs

A signal that carries navigation data is ranged on and decoded alike. A pilot
carries none, so it is given with its data component, `driver => decoding
signal`: the satellite ranges on the pilot, whose prompt has no data-bit
transitions, and decodes the data component (Galileo `GalileoE1C() =>
GalileoE1B()`, GPS `GPSL5Q() => GPSL5I()`, `GPSL1C_P() => GPSL1C_D()`,
`GPSL2CL() => GPSL2CM()`, BeiDou `BeiDouB1C_P() => BeiDouB1C_D()`).

- The **driver** closes the loops: its integrations carry the discriminators, the
  replica's code phase and the C/N₀ the measurements are weighted by.
- The **decoding signal** feeds the decoder: its integrations carry its prompts and
  where its bit clock stands. For a plain signal these are the driver's own
  integrations.

At an epoch the transmit time is the data bits the decoding signal has
counted plus the driver's code phase within the current bit, as
PositionVelocityTime builds it for a pair. The two codes are aligned at the
satellite, so their lengths may differ (L2 CL is 1.5 s, L2 CM 20 ms); the chip
rates must agree. A satellite is in lock when the driver's C/N₀ clears
`min_tracking_cn0` and the decoding signal holds its bit sync; a data
component too weak to decode never completes its decoder, so the satellite never
becomes ready for the PVT. The solution, the reports and the release reasons
are keyed by the driver.

The integrations of the two arrive independently and in any order: a satellite's
snapshot of an epoch is complete once an integration of each has crossed it, and the
cycle waits for that as for a late satellite.

## What the integrations must carry

The navigation processor runs no bit clock and no C/N₀ estimator of its own. The host holds a
[`PostCorrelationState`](@ref) per signal of every satellite (bit clock, C/N₀
estimator, post-correlation filter), advances it with [`process_correlator_bank_output!`](@ref) on every
integration, and steps the tracking loop on it:

```julia
post_correlation_state =
    process_correlator_bank_output!(post_correlation_state, correlator_bank_output, fs, noise_density, noise_density_ready)
loop_state, carrier_doppler, code_doppler =
    step_loop!(tracking_loop, loop_state, post_correlation_state, nco_history, application_sample_index)
```

The navigation processor reads from it whether the bit clock holds the sync and
whether the latest integration found or lost it, the code periods into the current bit,
exactly the demodulated soft bits that integration added (whether or not the host has
drained earlier ones), and the host's C/N₀
estimator (noise-referenced by default, which reads `-Inf` dB-Hz, out of lock, until the
host has a noise density). Beyond that, every [`PostCorrelationState`](@ref) handed to the
tracking loop must carry what only the correlator knows:

- `prn`: the satellite;
- `last_code_phase`: the replica's code phase in chips at the integration's end, from the
  [`CorrelatorBankOutput`](@ref). The end sample alone pins it only to within one
  sample (some 75 m at 4 MHz). Only its part past the nearest code-period
  boundary is read, so any wrap convention works; without it (`NaN`) the integration
  is taken to end on a code period boundary;
- a common time grid: `last_sample_index / sampling_frequency` must be the time since
  one origin shared by all satellites. A host whose correlator restarts its
  sample count passes the offset as `sample_index_offset` to
  [`process_correlator_bank_output!`](@ref).

`nothing` still means the command computed from an integration acts from the
integration's end; a hardware host passes the application sample (see
[Under a hardware NCO delay](@ref)).

## The lifecycle of a satellite

Vector tracking needs a decoded navigation message and a first position fix, so
every satellite starts on its scalar loop:

1. A satellite registers with the navigation processor on its first integration. Until vector
   tracking runs, each cycle solves the scalar PVT over the satellites in lock
   (C/N₀ at least `min_tracking_cn0`, bit sync found), decoded for positioning and
   healthy.
2. The first fix seeds the filter with its position and clock biases and the
   covariance its geometry gives them, so a fix of poor geometry (high DOP)
   starts the filter as uncertain as it is. The fix's satellites are added to the
   vector loop and take their first commands, computed at the seeded state, on
   their next integration. In the loop the filter sets a satellite's carrier NCO and
   its residual PLL follows the carrier, whatever [Carrier-loop staging](@ref)
   stage its scalar loop had reached.
3. While it runs, a satellite is admitted once it is decoded, healthy, in lock
   and 1° (`ADMISSION_ELEVATION`) above the horizon. A member out of lock (C/N₀
   or bit sync) stays in the loop, unmeasured, and is predicted through the
   outage. A member that misses an epoch sits that cycle out.
4. A member is released when it is no longer eligible ([`VT_INELIGIBLE`](@ref
   TrackingLoops.VectorTrackingReleaseReason); this includes a satellite without an integration for two cycles,
   whose slot is freed) or drops below the horizon
   ([`VT_BELOW_HORIZON`](@ref TrackingLoops.VectorTrackingReleaseReason)). Its scalar loop is re-seeded at
   the vector loop's estimate of the Doppler and its rate (see
   [The carrier in the vector loop](@ref vector-carrier)) and its carrier-loop
   staging restarts, so it pulls in afresh, FLL-assisted at the wide bandwidth.
5. After `measurement_deficit_timeout` of unsolvable cycles, or with no member
   left, every satellite in the loop, members that missed the epoch included, is
   released ([`VT_FALLBACK`](@ref TrackingLoops.VectorTrackingReleaseReason)) and the scalar PVT runs again
   until a fresh fix seeds the filter anew.

A re-acquired satellite (a fresh state on a PRN seen before) gets its old slot
back: its decoder restarts its sync but keeps its data, so it is ready again
after the next subframe rather than a whole frame once the host's bit clock has
found the bit edges again.

## Stepping the satellites

A cycle runs once every satellite that can still complete its epoch's snapshot
has, and a satellite that has gone two cycles without an integration of one of its
signals is dropped, so a host should step every signal of every satellite at
least once per half cycle (Tracking.jl's `track!` does, for any chunk shorter
than that). A satellite that falls behind holds a cycle up only until the others
reach the next epoch; then the cycle runs without it. A satellite already past
an epoch when it becomes pending (stepped earlier in the chunk than the integration
that ran the previous cycle, as after reacquisition) sits that epoch's cycle
out; the next pending epoch is the first no satellite is past. A host never has
to hold an integration back or order its channels: each integration is stepped as it
arrives.

## [The carrier in the vector loop](@id vector-carrier)

The navigation filter sets a member's carrier NCO directly: no loop filter and
no phase discriminator sit between them. So the FLL measures the incoming carrier
against a replica nothing else moves, and a residual PLL per satellite follows
the carrier on the prompts the NCO leaves, for the data.

`get_carrier_loop_stage(loop_state)` reports `VECTOR_LOOP` while the navigation filter
commands the satellite, and the scalar loop's stage otherwise.

- **The filter sets the carrier NCO.** Each integration is commanded the filter's
  predicted Doppler at the middle of the integration its command takes effect on: a ramp
  evaluated from the latest cycle's state at the application sample and one cycle later
  (velocity, acceleration where the motion model has one, clock drift and the
  satellite's motion). The code is commanded from that Doppler, carrier-aided,
  plus the filter's code correction.
- **A residual PLL follows the carrier.** Per satellite, a third-order software
  loop estimates the residual phase, Doppler and Doppler rate against the
  filter's ramp from the prompts and the commands the NCO ran, Costas or
  four-quadrant by the integration's `carrier_phase_polarity`. It never steers the
  NCO, and neither the command nor a new ramp moves its estimate of the incoming
  Doppler. Its only outputs are the phase the decoded prompts are rotated by, and
  its lock.
- **The rate measurement** is the replica's Doppler at the epoch plus the mean
  FLL discriminator over the cycle: the replica follows the filter's ramp, so its
  rise to the epoch matches the incoming carrier's, and no loop's jitter is in
  its command. The FLL compares consecutive prompts, so it needs no phase lock, and
  the measurements telescope across cycles (lag-1 correlation −½ without an NCO
  delay, an integrated error bounded by the boundary prompts' noise).
- **The navigation processor decodes the prompts it rotated.** The prompts turn at the
  filter's Doppler error, so the navigation processor rotates each of the decoding signal's
  prompts onto the incoming carrier by the residual PLL's phase (and a data
  component's carrier-phase offset from its pilot), sums them over each data
  bit the host's bit clock counts and decodes the sum's real part. A plain
  signal's prompt is rotated by the residual PLL's prediction for its integration, which
  that prompt did not correct; a data component's integrations, which a host processes
  after its pilot's, take the phase estimated at the pilot's integration of the same
  span. The host's bit synchronizer still keeps the bit clock (it counts code periods once
  synced), but its soft bits are not decoded while the satellite is a member, so
  the host's prompts need no rotation. Its C/N₀ estimator should read power only,
  as the default noise-referenced and the moment-based ones do: a coherent one
  ([`NWPRCN0Estimator`](@ref)) reads the turning prompts as a small loss.
- **Lock.** A member is measured while the host's C/N₀ clears
  `min_tracking_cn0` and its decoding signal holds the bit sync, as for any
  satellite. The residual PLL's own phase lock ([`is_phase_locked`](@ref TrackingLoops.is_phase_locked)) reads
  the indicator on the prompts it rotated, with the C/N₀ those moments read no
  more than 6 dB below `min_tracking_cn0`. While the host's C/N₀ is below
  `min_tracking_cn0` (no signal) it holds its residual at the filter's
  Doppler, so it does not integrate a fade's noise into tens of hertz and pulls
  in again from the filter's prediction once the signal is back; its own
  moments read up to the threshold on bare noise, so they do not decide that.
  With the signal there, in phase lock or not, its loop runs on and pulls in by
  itself.
- **Carrier phase.** The replica's carrier phase is the NCO's, which the
  filter sets, so it is not the incoming carrier's: a host that measures the
  carrier phase (RINEX `L`) adds [`calc_carrier_phase_correction`](@ref TrackingLoops.calc_carrier_phase_correction), the
  residual PLL's predicted residual at the sample it reads the replica at,
  whole cycles included (zero out of the vector loop and for the scalar
  loops). It steps by half a cycle where a pilot turns the discriminator
  four-quadrant and the residual PLL moves to its frame, and by the residual
  on a release.
- **Hand-over.** A satellite is added at its scalar loop's command, which its
  residual PLL starts from. It is released into its scalar loop at the residual
  PLL's estimate of the incoming Doppler and its rate (the ramp's plus the
  residual PLL's), which the third-order carrier filter starts from; out of lock,
  at the Doppler and rate the filter last commanded. Its staging restarts.

The residual PLL's bandwidth is [`VectorTrackingLoop`](@ref)'s
`residual_pll_loop_filter_bandwidth`, 10 Hz by default
([`default_residual_pll_loop_filter_bandwidth`](@ref TrackingLoops.default_residual_pll_loop_filter_bandwidth)), capped like the scalar
PLL's against the coherent integration time. At 10 Hz its thermal jitter is
`√(10/18) = 0.75` times the 18 Hz PLL's.

In the simulation of `test/vector_residual_pll.jl` (synthetic GPS L1 C/A and
Galileo E1 satellites, no oscillator phase noise):

| | |
|---|---|
| phase jitter of the residual PLL's prediction at 40 dB-Hz, vs. the 18 Hz scalar PLL | 0.77 (L1 C/A), 0.75 (E1) |
| FLL rate residuals, Galileo E1 at 37–45 dB-Hz (Costas) and 34 dB-Hz (four-quadrant), GPS L1 C/A at 45 dB-Hz | `σ/√R` 0.92–1.08, lag-1 −0.36 to −0.62, at most 4.7σ, integrated error below 3.5 cm |
| 0.5 g / 1 g manoeuvre at 40 dB-Hz | in lock throughout, residual Doppler up to 4.0 / 7.6 Hz, phase error up to 10° / 21° |

Driving each member's PLL through its FLL branch instead, which leaves the PLL's
integrators in charge of the replica's frequency, gave 1.47 (L1 C/A) and 1.08 (E1)
times the scalar PLL's prompt-phase jitter.

Not done yet: a receiver-clock phase common to all satellites, estimated at the
integration rate (vector phase lock), and a carrier-phase output for RTK.

## Under a hardware NCO delay

With a hardware correlator, the command computed from an integration takes effect only
when the NCO applies it. Pass the application sample and the channel's
[`NCOTimeline`](@ref) to `step_loop!` as `application_sample_index` and
`nco_history`, respectively. Each satellite then sizes its
commands for the moment they take effect, against the replica the timeline predicts
there, and a released satellite's scalar loop takes over from that replica. The
application sample may lie up to 2.5 cycles after the epoch.

## Reading the results

Nothing is logged. [`get_navigation_solution`](@ref) is the latest solution (the
scalar PVT's before the filter is seeded) and [`get_navigation_status`](@ref) the
[`VectorTrackingStatus`](@ref TrackingLoops.VectorTrackingStatus) of the latest cycle: its events and the filter's
position and clock uncertainties (also [`calc_position_std`](@ref TrackingLoops.calc_position_std) and
[`calc_clock_bias_std`](@ref TrackingLoops.calc_clock_bias_std)). A satellite's [`get_satellite_report`](@ref) says whether and why the
latest cycle released it (`release_reason`), and [`get_vector_tracking_members`](@ref) reports every
member, measured and coasted. [`get_navigation_cycle_index`](@ref) counts the cycles, so a
consumer polling it after each step reads each cycle's results, and
[`get_navigation_elapsed_time`](@ref) is the elapsed time of the latest navigation
cycle on the integrations' time grid. An unsuccessful solve retains the previous PVT
solution, so the elapsed time may be newer than the solution's fix.
[`get_satellite_report`](@ref) hands out what the navigation processor knows of a satellite
as an immutable snapshot of bit sync, C/N₀, measurement readiness and membership.
Its values remain stable after later integrations. [`get_decoder`](@ref) separately
borrows the navigation processor's decoder; consume or copy its contents before the next integration
changes its shared buffers.

Scalar loops answer these with `nothing`, so a host can ask whichever tracking
loop it was given:

```julia
cycle = get_navigation_cycle_index(tracking_loop)
if !isnothing(cycle) && cycle != last_cycle
    last_cycle = cycle              # a failed solve may retain the earlier PVT fix
    pvt = get_navigation_solution(tracking_loop)       # position, velocity, time, DOP, …
    report = get_satellite_report(tracking_loop, GPSL1CA(), prn)
end
```

The solution reuses its containers on the next cycle, so copy out values needed
later. Satellite reports are immutable snapshots and remain valid after later
integrations and calls.

## Storage

All storage (decoders and the filter's buffers) is allocated at construction for
`satellite_capacity_per_signal` satellites per ranging signal; a dropped satellite
leaves its slot to the next one, and only past that capacity does a group grow.
Once warm, integrations and cycles allocate nothing, and the tracking loop compiles with
`juliac --trim=safe`.

## Known limits

- A pair ranges on the driver alone: the data component's discriminators are
  not combined with the pilot's.
- The residual PLL is capped like the scalar PLL against the integration's
  integration time: at 20 ms integrations its 10 Hz runs at 4.5 Hz.
- On 1 ms integrations (GPS L1 C/A) the FLL's rate measurement breaks down below
  about 40 dB-Hz: a phase difference between two prompts wraps at the Costas
  discriminator's ±90° often enough to break the telescoping, and the residuals
  reach three times `√R`.
- Membership follows the host's C/N₀, which lags a sudden blockage: a cycle in
  which the signal is suddenly gone can still be measured with noise-only FLL
  discriminators.

## The tracking loop

```@docs
TrackingLoops.VectorLoopState
VectorTrackingLoop
init_loop_state(::VectorTrackingLoop, ::AbstractGNSSSignal, ::Any, ::Any)
reinit_loop_state(::VectorTrackingLoop, ::VectorLoopState, ::Any, ::Any)
step_loop!(::VectorTrackingLoop, ::VectorLoopState, ::PostCorrelationState, ::Any, ::Union{Nothing,Integer})
TrackingLoops.default_residual_pll_loop_filter_bandwidth()
```

## The navigation cycle

```@docs
TrackingLoops.VectorTrackingStatus
TrackingLoops.SatelliteReport
get_decoder
TrackingLoops.VectorTrackingReleaseReason
TrackingLoops.calc_clock_bias_std(::VectorTrackingLoop)
get_navigation_solution(::VectorTrackingLoop)
get_navigation_status(::VectorTrackingLoop)
TrackingLoops.calc_position_std(::VectorTrackingLoop)
get_vector_tracking_members(::VectorTrackingLoop)
NavigationFilterConfig
```
