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
which biases a cycle can determine is decided per cycle, and clocks collapse
onto a hub system's through the broadcast offset (GGTO, BGTO) when the
measurements cannot separate them. With `use_pseudorange_rates = true` (the
default) it also measures the pseudorange rates (VDFLL); with `false` it is a
VDLL, and the carrier NCOs' Doppler comes from the code measurements alone.

## One estimator does it all

[`VectorPLLAndDLL`](@ref) is a Doppler estimator like any other: a host builds
each satellite's state with [`init_estimator_state`](@ref) and steps it with
[`step_loop`](@ref) on every record of the satellite. Everything vector
tracking needs happens inside that step, from what the records carry. All
satellites stepped with one estimator share its navigation engine, which

- decodes each satellite's navigation data from the soft bits its records
  carry;
- snapshots each satellite at every navigation epoch (every `vt_update_interval`
  on the records' time grid): code phase from the last data-symbol edge, decoder,
  Dopplers, C/N₀ and the discriminators accumulated since the last epoch;
- runs the epoch's navigation cycle on the record that completes the last
  satellite's snapshot: the scalar PVT until the filter is seeded, a filter
  iteration after that;
- leaves the cycle's decisions (admission, release, commands) for each
  satellite to take up on its next record, sized for where its command lands.

```julia
estimator = VectorPLLAndDLL(GPSL1CA(), GalileoE1C() => GalileoE1B())   # owns decoders, filter and PVT
ts = TrackState(; signal = GPSL1CA(), doppler_estimator = estimator)
for chunk in chunks
    track!(chunk, ts, fs)              # decodes, solves, steers
end
navigation_solution(estimator)         # the PVTSolution; navigation_status(estimator)
```

Several constellations or bands share one filter: list every ranging signal at
construction. Each satellite runs the scalar loop `inner` until the filter takes
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

- The **driver** closes the loops: its records carry the discriminators, the
  replica's code phase and the C/N₀ the measurements are weighted by.
- The **decoding signal** feeds the decoder: its records carry its prompts and
  where its bit clock stands. For a plain signal these are the driver's own
  records.

At an epoch the transmit time is the data symbols the decoding signal has
counted plus the driver's code phase within the current symbol, as
PositionVelocityTime builds it for a pair. The two codes are aligned at the
satellite, so their lengths may differ (L2 CL is 1.5 s, L2 CM 20 ms); the chip
rates must agree. A satellite is in lock when the driver's C/N₀ clears
`lock_cn0_threshold` and the decoding signal holds its bit sync; a data
component too weak to decode never completes its decoder, so the satellite never
becomes ready for the PVT. The solution, the reports and the release reasons
are keyed by the driver.

The records of the two arrive independently and in any order: a satellite's
snapshot of an epoch is complete once a record of each has crossed it, and the
cycle waits for that as for a late satellite.

## What the records must carry

The engine runs no bit clock and no C/N₀ estimator of its own. The host holds a
[`SignalLoopState`](@ref) per signal of every satellite (bit clock, C/N₀
estimator, prompt filter), advances it with [`apply_record`](@ref) on every
record, and builds the record from it:

```julia
loop, prompt, filtered, blocks, overshoot =
    apply_record(loop, signal, prn, output, fs, noise_density, noise_density_ready, driver_phase)
record = LoopRecord(signal, filtered, output, loop, fs; prn)
state, carrier_doppler, code_doppler = step_loop(estimator, state, record, words, landing_sample)
```

The record summarises that state: whether the bit clock holds the sync and
whether this record found or lost it, the code blocks into the current symbol,
exactly the soft bits this record added (so the host may drain the bit buffer
whenever it likes), and the host's C/N₀ estimator (noise-referenced by default,
which reads `-Inf` dB-Hz, out of lock, until the host has a noise density).
Beyond that, every [`LoopRecord`](@ref) handed to the estimator must carry what
only the correlator knows:

- `prn`: the satellite;
- `code_phase`: the replica's code phase in chips at the record's end, from the
  [`CorrelatorOutput`](@ref). The end sample alone pins it only to within one
  sample (some 75 m at 4 MHz). Only its part past the nearest code-block
  boundary is read, so any wrap convention works; without it (`NaN`) the record
  is taken to end on a block boundary;
- a common time grid: `sample_index / sampling_frequency` must be the time since
  one origin shared by all satellites. A host whose correlator restarts its
  sample count passes the offset as `sample_offset` when it builds the record.

`NO_LANDING_SAMPLE` still means the command computed from a record acts from the
record's end; a hardware host passes the landing sample (see
[Under a hardware NCO delay](@ref)).

## The lifecycle of a satellite

Vector tracking needs a decoded navigation message and a first position fix, so
every satellite starts on its scalar loop:

1. A satellite registers with the engine on its first record. Until vector
   tracking runs, each cycle solves the scalar PVT over the satellites in lock
   (C/N₀ at least `lock_cn0_threshold`, bit sync found), decoded for positioning and
   healthy.
2. The first fix seeds the filter with its position and clock biases and the
   covariance its geometry gives them, so a fix of poor geometry (high DOP)
   starts the filter as uncertain as it is. The fix's satellites are added to the
   vector loop and take their first commands, computed at the seeded state, on
   their next record. In the loop the filter sets a satellite's carrier NCO and
   its residual PLL follows the carrier, whatever [Carrier-loop staging](@ref)
   stage its scalar loop had reached.
3. While it runs, a satellite is admitted once it is decoded, healthy, in lock
   and 1° (`ADMISSION_ELEVATION`) above the horizon. A member out of lock (C/N₀
   or bit sync) stays in the loop, unmeasured, and is predicted through the
   outage. A member that misses an epoch sits that cycle out.
4. A member is released when it is no longer eligible ([`VT_INELIGIBLE`](@ref
   VTReleaseReason); this includes a satellite without a record for two cycles,
   whose slot is freed) or drops below the horizon
   ([`VT_BELOW_HORIZON`](@ref VTReleaseReason)). Its scalar loop is re-seeded at
   the vector loop's estimate of the Doppler and its rate (see
   [The carrier in the vector loop](@ref vector-carrier)) and its carrier-loop
   staging restarts, so it pulls in afresh, FLL-assisted at the wide bandwidth.
5. After `insufficient_meas_timeout` of unsolvable cycles, or with no member
   left, every satellite in the loop, members that missed the epoch included, is
   released ([`VT_FALLBACK`](@ref VTReleaseReason)) and the scalar PVT runs again
   until a fresh fix seeds the filter anew.

A re-acquired satellite (a fresh state on a PRN seen before) gets its old slot
back: its decoder restarts its sync but keeps its data, so it is ready again
after the next subframe rather than a whole frame once the host's bit clock has
found the bit edges again.

## Stepping the satellites

A cycle runs once every satellite that can still complete its epoch's snapshot
has, and a satellite that has gone two cycles without a record of one of its
signals is dropped, so a host should step every signal of every satellite at
least once per half cycle (Tracking.jl's `track!` does, for any chunk shorter
than that). A satellite that falls behind holds a cycle up only until the others
reach the next epoch; then the cycle runs without it. A satellite already past
an epoch when it becomes pending (stepped earlier in the chunk than the record
that ran the previous cycle, as after a re-registration) sits that epoch's cycle
out; the next pending epoch is the first no satellite is past. A host never has
to hold a record back or order its channels: each record is stepped as it
arrives.

## [The carrier in the vector loop](@id vector-carrier)

The navigation filter sets a member's carrier NCO directly: no loop filter and
no phase discriminator sit between them. So the FLL measures the incoming carrier
against a replica nothing else moves, and a residual PLL per satellite follows
the carrier on the prompts the NCO leaves, for the data.

- **The filter sets the carrier NCO.** Each record is commanded the filter's
  predicted Doppler at the middle of the record its command lands on: a ramp
  evaluated from the latest cycle's state at the landing and one cycle later
  (velocity, acceleration where the motion model has one, clock drift and the
  satellite's motion). The code is commanded from that Doppler, carrier-aided,
  plus the filter's code correction.
- **A residual PLL follows the carrier.** Per satellite, a third-order software
  loop estimates the residual phase, Doppler and Doppler rate against the
  filter's ramp from the prompts and the words the NCO ran, Costas or
  four-quadrant by the record's `carrier_phase_polarity`. It never steers the
  NCO, and neither the command nor a new ramp moves its estimate of the incoming
  Doppler. Its only outputs are the phase the decoded prompts are rotated by, and
  its lock.
- **The rate measurement** is the replica's Doppler at the epoch plus the mean
  FLL discriminator over the cycle: the replica follows the filter's ramp, so its
  rise to the epoch matches the incoming carrier's, and no loop's jitter is in
  its word. The FLL compares consecutive prompts, so it needs no phase lock, and
  the measurements telescope across cycles (lag-1 correlation −½ without an NCO
  delay, an integrated error bounded by the boundary prompts' noise).
- **The engine decodes the prompts it rotated.** The prompts turn at the
  filter's Doppler error, so the engine rotates each of the decoding signal's
  prompts onto the incoming carrier by the residual PLL's phase (and a data
  component's carrier-phase offset from its pilot), sums them over each data
  symbol the host's bit clock counts and decodes the sum's real part. A plain
  signal's prompt is rotated by the residual PLL's prediction for its record, which
  that prompt did not correct; a data component's records, which a host folds
  after its pilot's, take the phase estimated at the pilot's record of the same
  span. The host's bit buffer still keeps the bit clock (it counts blocks once
  synced), but its soft bits are not decoded while the satellite is a member, so
  the host's prompts need no rotation. Its C/N₀ estimator should read power only,
  as the default noise-referenced and the moment-based ones do: a coherent one
  ([`NWPRCN0Estimator`](@ref)) reads the turning prompts as a small loss.
- **Lock.** A member is measured while the host's C/N₀ clears
  `lock_cn0_threshold` and its decoding signal holds the bit sync, as for any
  satellite. The residual PLL's own phase lock ([`phase_locked`](@ref)) reads
  the indicator on the prompts it rotated, with the C/N₀ those moments read no
  more than 6 dB below `lock_cn0_threshold`. While the host's C/N₀ is below
  `lock_cn0_threshold` (no signal) it holds its residual at the filter's
  Doppler, so it does not integrate a fade's noise into tens of hertz and pulls
  in again from the filter's prediction once the signal is back; its own
  moments read up to the threshold on bare noise, so they do not decide that.
  With the signal there, in phase lock or not, its loop runs on and pulls in by
  itself.
- **Carrier phase.** The replica's carrier phase is the NCO's, which the
  filter sets, so it is not the incoming carrier's: a host that measures the
  carrier phase (RINEX `L`) adds [`carrier_phase_correction`](@ref), the
  residual PLL's predicted residual at the sample it reads the replica at,
  whole cycles included (zero out of the vector loop and for the scalar
  loops). It steps by half a cycle where a pilot turns the discriminator
  four-quadrant and the residual PLL moves to its frame, and by the residual
  on a release.
- **Hand-over.** A satellite is added at its scalar loop's word, which its
  residual PLL starts from. It is released into its scalar loop at the residual
  PLL's estimate of the incoming Doppler and its rate (the ramp's plus the
  residual PLL's), which the third-order carrier filter starts from; out of lock,
  at the Doppler and rate the filter last commanded. Its staging restarts.

The residual PLL's bandwidth is [`VectorPLLAndDLL`](@ref)'s
`residual_pll_loop_filter_bandwidth`, 10 Hz by default
([`default_residual_pll_loop_filter_bandwidth`](@ref)), capped like the scalar
PLL's against the record's integration time. At 10 Hz its thermal jitter is
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
record rate (vector phase lock), and a carrier-phase output for RTK.

## Under a hardware NCO delay

With a hardware correlator, the command computed from a record takes effect only
when it lands at the NCO. Pass the landing sample to `step_loop` and the
channel's [`NCOTimeline`](@ref) as its words: each satellite then sizes its
commands for the moment they land, against the replica the timeline predicts
there, and a released satellite's scalar loop takes over from that replica. The
landing may lie up to 2.5 cycles after the epoch.

## Reading the results

Nothing is logged. [`navigation_solution`](@ref) is the latest solution (the
scalar PVT's before the filter is seeded) and [`navigation_status`](@ref) the
[`NavigationStatus`](@ref) of the latest cycle: its events and the filter's
position and clock uncertainties (also [`position_uncertainty`](@ref) and
[`clock_uncertainty`](@ref)). [`release_reason`](@ref) says whether and why the
latest cycle released a satellite, and [`vector_loop_members`](@ref) reports every
member, measured and coasted. [`navigation_cycle`](@ref) counts the cycles, so a
consumer polling it after each step reads every solution exactly once, and
[`navigation_epoch`](@ref) is the moment the latest solution describes, on the
records' time grid.
[`satellite_report`](@ref) hands out what the engine knows of a satellite
(decoder, bit sync, C/N₀, lock, PVT readiness, membership), so a receiver need
not decode or estimate the C/N₀ a second time.

Scalar loops answer these with `nothing`, so a host can ask whichever estimator
it was given:

```julia
cycle = navigation_cycle(estimator)
if cycle != last_cycle            # `nothing` for a scalar loop: never a new solution
    last_cycle = cycle
    pvt = navigation_solution(estimator)       # position, velocity, time, DOP, …
    report = satellite_report(estimator, GPSL1CA(), prn)
end
```

The returned objects are the estimator's own and reused by the next cycle or
call: copy out what is needed later.

## Storage

All storage (decoders and the filter's buffers) is allocated at construction for
`max_satellites_per_signal` satellites per ranging signal; a dropped satellite
leaves its slot to the next one, and only past that capacity does a group grow.
Once warm, records and cycles allocate nothing, and the estimator compiles with
`juliac --trim=safe`.

## Known limits

- A pair ranges on the driver alone: the data component's discriminators are
  not combined with the pilot's.
- The residual PLL is capped like the scalar PLL against the record's
  integration time: at 20 ms records its 10 Hz runs at 4.5 Hz.
- On 1 ms records (GPS L1 C/A) the FLL's rate measurement breaks down below
  about 40 dB-Hz: a phase difference between two prompts wraps at the Costas
  discriminator's ±90° often enough to break the telescoping, and the residuals
  reach three times `√R`.
- Membership follows the host's C/N₀, which lags a sudden blockage: a cycle in
  which the signal is suddenly gone can still be measured with noise-only FLL
  discriminators.
- HardwareLoopCore does not yet report the records' `prn` and `code_phase`.

## The estimator

```@autodocs
Modules = [TrackingLoops]
Pages = ["vector/estimator.jl", "vector/engine.jl", "vector/residual_pll.jl"]
Private = false
```

## The navigation cycle

```@autodocs
Modules = [TrackingLoops]
Pages = ["vector/tracking.jl", "vector/model.jl"]
Private = false
```
