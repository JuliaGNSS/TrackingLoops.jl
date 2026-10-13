# Vector tracking

In scalar tracking every satellite closes its own code and carrier loops from
its own discriminators. In vector tracking a navigation filter closes them all
at once: each satellite's accumulated DLL and FLL discriminators become
pseudorange and pseudorange-rate measurements, the filter fuses them into a
position, velocity and clock state, and the predicted line-of-sight dynamics
are fed back as per-satellite NCO corrections. Weak or briefly obscured
satellites are carried through by the common solution instead of losing lock on
their own.

The filter is an unscented Kalman filter over position, velocity, one clock bias
per GNSS time system (all driven by the one oscillator's drift) and one
inter-frequency bias per band beyond the reference band. It follows the bias
model of [PositionVelocityTime.jl](https://github.com/JuliaGNSS/PositionVelocityTime.jl):
which biases a cycle can determine is decided per cycle, and clocks collapse
onto a hub system's through the broadcast offset (GGTO, BGTO) when the
measurements cannot separate them. With `use_pseudorange_rates = true` (the
default) it also measures the pseudorange rates (VDFLL); with `false` it is a
VDLL, and the carrier corrections come from the filter's prediction alone.

## One estimator does it all

[`VectorPLLAndDLL`](@ref) is a Doppler estimator like any other: a host builds
each satellite's state with [`init_estimator_state`](@ref) and steps it with
[`step_loop`](@ref) on every record of the satellite. Everything vector
tracking needs happens inside that step, from what the records carry. All
satellites stepped with one estimator share its navigation engine, which

- decodes each satellite's navigation data from the soft bits its records
  carry;
- snapshots each satellite at every navigation epoch (every `cycle_time` on the
  records' time grid): code phase from the last data-symbol edge, decoder,
  Dopplers, C/N₀ and the discriminators accumulated since the last epoch;
- runs the epoch's navigation cycle on the record that completes the last
  satellite's snapshot: the scalar PVT until the filter is seeded, a filter
  iteration after that;
- leaves the cycle's decisions (admission, release, corrections) for each
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
it over, and again once the filter lets it go. In the vector loop the code
filter is frozen, the filter's corrections steer the replica, and the
discriminators are accumulated for the filter.

## Pilot + data pairs

A signal that carries navigation data is ranged on and decoded alike. A pilot
carries none, so it is given with its data component, `driver => decoding
signal`: the satellite ranges on the pilot, whose prompt has no data-bit
transitions, and decodes the data component (Galileo `GalileoE1C() =>
GalileoE1B()`, GPS `GPSL5Q() => GPSL5I()`, `GPSL1C_P() => GPSL1C_D()`,
`GPSL2CL() => GPSL2CM()`, BeiDou `BeiDouB1C_P() => BeiDouB1C_D()`).

- The **driver** closes the loops: its records carry the discriminators, the
  replica's code phase and the C/N₀ the measurements are weighted by.
- The **decoding signal** feeds the decoder: its records carry the soft bits
  and where its bit clock stands. For a plain signal these are the driver's own
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

1. A satellite joins the engine on its first record. Until vector tracking
   runs, each cycle solves the scalar PVT over the satellites in lock (C/N₀
   above `lock_cn0_threshold`, bit sync found), decoded for positioning and
   healthy.
2. The first fix seeds the filter with its position and clock biases and the
   covariance its geometry gives them, so a fix of poor geometry (high DOP)
   starts the filter as uncertain as it is. The fix's satellites join the vector
   loop and take their first corrections, computed at the seeded state, on their
   next record. In the loop a satellite runs its carrier loop at the narrow PLL
   bandwidth, whatever [Carrier-loop staging](@ref) stage its scalar loop had
   reached.
3. While it runs, a satellite is admitted once it is decoded, healthy, in lock
   and a degree above the horizon. A member out of lock stays in the loop,
   unmeasured, and is predicted through the outage. A member that misses an
   epoch sits that cycle out.
4. A member is released when it is no longer eligible ([`VT_INELIGIBLE`](@ref
   VTReleaseReason); this includes a satellite without a record for two cycles,
   whose slot is freed) or drops below the horizon
   ([`VT_BELOW_HORIZON`](@ref VTReleaseReason)). Its scalar loop is re-seeded
   from the replica at its landing and its carrier-loop staging restarts, so
   it pulls in afresh, FLL-assisted at the wide bandwidth.
5. After `insufficient_meas_timeout` of unsolvable cycles, or with no member
   left, every member is released ([`VT_FALLBACK`](@ref VTReleaseReason)) and
   the scalar PVT runs again until a fresh fix seeds the filter anew.

A re-acquired satellite (a fresh state on a PRN seen before) gets its old slot
back: its decoder restarts its sync but keeps its data, so it is ready again
after the next subframe rather than a whole frame once the host's bit clock has
found the bit edges again.

## Stepping the satellites

A cycle runs once every satellite has completed its epoch's snapshot, and a
satellite that has gone two cycles without a record of one of its signals is
dropped, so a host should step every signal of every satellite at least once per
half cycle (Tracking.jl's `track!` does, for any chunk shorter than that). A
satellite that falls behind holds a cycle up only until the others reach the
next epoch; then the cycle runs without it. A host never has to hold a record
back or order its channels: each record is stepped as it arrives.

## Under a hardware NCO delay

With a hardware correlator, the command computed from a record takes effect only
when it lands at the NCO. Pass the landing sample to `step_loop` and the
channel's [`NCOTimeline`](@ref) as its words: each satellite then sizes its
corrections for the moment its command lands, against the replica the timeline
predicts there, and a released satellite's scalar loop takes over from that
replica. The landing may lie up to 2.5 cycles after the epoch.

## Reading the results

Nothing is logged. [`navigation_solution`](@ref) is the latest solution (the
scalar PVT's before the filter is seeded) and [`navigation_status`](@ref) the
[`VTStatus`](@ref) of the latest cycle: its events and the filter's position and
clock uncertainties (also [`position_uncertainty`](@ref) and
[`clock_uncertainty`](@ref)). [`release_reason`](@ref) says whether and why a
satellite was released, and [`member_sats`](@ref) reports every member, measured
and coasted. [`navigation_cycle`](@ref) counts the cycles, so a consumer polling
it reads every solution once, and [`navigation_epoch`](@ref) is the moment the
latest solution describes. [`satellite_report`](@ref) hands out what the engine
knows of a satellite (decoder, bit sync, C/N₀, lock, PVT readiness, membership),
so a receiver need not decode or estimate the C/N₀ a second time.

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
- Lock is a C/N₀ threshold over the bit-synced satellites, not a full lock
  detector.
- HardwareLoopCore does not yet report the records' `prn` and `code_phase`.

## The estimator

```@autodocs
Modules = [TrackingLoops]
Pages = ["vector/estimator.jl", "vector/engine.jl"]
Private = false
```

## The navigation cycle

```@autodocs
Modules = [TrackingLoops]
Pages = ["vector/tracking.jl", "vector/model.jl"]
Private = false
```
