# Vector tracking

In scalar tracking every satellite closes its own code and carrier loops from
its own discriminators. In vector tracking a navigation filter closes them all
at once. Each satellite's accumulated DLL and FLL discriminators become
pseudorange and pseudorange-rate measurements. The filter fuses them into a
position, velocity and clock state and feeds the predicted line-of-sight
dynamics back as per-satellite NCO corrections. Weak or briefly obscured
satellites are carried through by the common solution instead of losing lock on
their own.

The filter is an unscented Kalman filter over position, velocity, one clock bias
per GNSS time system (all driven by the one oscillator's drift) and one
inter-frequency bias per band beyond the reference band. It follows the bias
model of [PositionVelocityTime.jl](https://github.com/JuliaGNSS/PositionVelocityTime.jl):
which biases a cycle can determine is decided per cycle, and clocks collapse
onto a hub system's through the broadcast offset (GGTO, BGTO) when the
measurements cannot separate them. With `use_pseudorange_rates = true` (the
default) it measures the pseudorange rates too (VDFLL). With `false` it is a VDLL,
and the carrier corrections come from the filter's prediction alone.

## One estimator does it all

[`VectorPLLAndDLL`](@ref) is a Doppler estimator like any other: a host builds
each satellite's state with [`init_estimator_state`](@ref) and steps it with
[`step_loop`](@ref) on every record of the satellite, as it steps any other
loop. Everything vector tracking needs happens inside that step, from what the
records carry. Every satellite stepped with one estimator shares its navigation
engine, which

- decodes each satellite's navigation data from the soft bits its records
  carry;
- snapshots each satellite at every navigation epoch (every `cycle_time` on the
  records' time grid): the replica's code phase from the last data-symbol edge,
  the decoder there, the Dopplers, the C/N₀ and the discriminators accumulated
  since the last epoch;
- runs the epoch's navigation cycle on the record that completes the last
  satellite's snapshot: the scalar PVT until the filter is seeded, a filter
  iteration after that;
- leaves the cycle's decisions — admission, release, the corrections — for each
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
construction. Each satellite runs the scalar loop `inner` —
`ConventionalAssistedPLLAndDLL()` (the default) or `NCOReferencedPLLAndDLL()` —
until the filter takes it over, and again once the filter lets it go. In the
vector loop the code filter is frozen, the filter's corrections steer the
replica, and the discriminators are accumulated for the filter.

## Pilot + data pairs

A signal that carries navigation data is ranged on and decoded alike. A pilot
carries none, so it is given with its data component, `driver => decoding
signal`: the satellite ranges on the pilot, whose prompt has no data-bit
transitions, and decodes the data component — Galileo `GalileoE1C() =>
GalileoE1B()`, GPS `GPSL5Q() => GPSL5I()`, `GPSL1C_P() => GPSL1C_D()`,
`GPSL2CL() => GPSL2CM()`, BeiDou `BeiDouB1C_P() => BeiDouB1C_D()`.

- The **driver** closes the loops: its records carry the discriminators, the
  replica's code phase and the C/N₀ the measurements are weighted by.
- The **decoding signal** feeds the decoder: its records carry the soft bits
  and where its bit clock stands. For a plain signal these are the driver's own
  records.

At an epoch the transmit time is the data symbols the decoding signal has
counted plus the driver's code phase within the current symbol, which is how
PositionVelocityTime builds it for a pair. The two codes are aligned at the
satellite, so their lengths may differ (L2 CL is 1.5 s, L2 CM 20 ms); the
chip rates must agree. A satellite is in lock when the driver's C/N₀ clears
`lock_cn0_threshold` and the decoding signal holds its bit sync; a data
component too weak to decode never completes its decoder, so it never makes the
satellite ready for the PVT. The solution, the reports and the release reasons
are keyed by the driver.

The records of the two arrive independently and in any order: a satellite's
snapshot of an epoch is complete once a record of each has crossed it, and the
cycle waits for that as it waits for a late satellite. A satellite is dropped
when either signal has gone two cycles without a record.

## What the records must carry

The engine runs no bit clock and no C/N₀ estimator of its own. The host holds a
[`SignalLoopState`](@ref) per signal of every satellite — its bit clock, C/N₀
estimator and prompt filter — advances it with [`apply_record`](@ref) on every
record, and builds the record from it:

```julia
previous_prompt = loop.last_filtered_prompt
loop, prompt, filtered, blocks, overshoot =
    apply_record(loop, signal, prn, output, fs, noise_density, noise_density_ready, driver_phase)
record = LoopRecord(signal, filtered, previous_prompt, output, loop, fs; prn)
state, carrier_doppler, code_doppler = step_loop(estimator, state, record, words, landing_sample)
```

The record summarises the state: whether the bit clock holds the sync and
whether this record found or lost it, the code blocks into the current symbol,
exactly the soft bits this record added — so the host may drain the bit buffer
after every record or once per call — and the C/N₀ estimator the host chose
(noise-referenced by default, which reads `-Inf` dB-Hz, out of lock, until the
host has a noise density). Beyond that, every [`LoopRecord`](@ref) handed to
the estimator must carry what only the correlator knows:

- `prn`: the satellite;
- `code_phase`: the replica's code phase in chips at the record's end, from the
  [`CorrelatorOutput`](@ref). The end sample alone pins it only to within one
  sample (some 75 m at 4 MHz). Only its part past the nearest code-block
  boundary is read, so any wrap convention works; without it (`NaN`) the record
  is taken to end on a block boundary;
- a common time grid: `sample_index / sampling_frequency` must be the time since
  one origin shared by all satellites. A host whose correlator restarts its
  sample count passes the offset as `sample_offset` when it builds the record.

`NO_LANDING_SAMPLE` keeps meaning that the command computed from a record acts
from the record's end; a hardware host passes the landing sample, and each
satellite sizes its corrections for that moment.

## The lifecycle of a satellite

Vector tracking needs a decoded navigation message and a first position fix, so
every satellite starts on its scalar loop:

1. A satellite joins the engine on its first record. Until vector tracking
   runs, each cycle solves the scalar PVT over the satellites that are in lock
   (C/N₀ above `lock_cn0_threshold`, bit sync found), decoded for positioning
   and healthy.
2. The first fix seeds the filter. The filter starts from the fix's position
   and clock biases, with the covariance the fix's geometry gives them, so a fix
   of poor geometry (a high DOP) starts the filter as uncertain as it is. The
   fix's satellites join the vector loop, and take their first corrections,
   computed at the seeded state, on their next record.
3. While it runs, a satellite is admitted once it is decoded, healthy, in lock
   and a degree above the horizon. A member out of lock stays in the loop,
   unmeasured, and is predicted through the outage. A member that misses an
   epoch sits that cycle out.
4. A member is released when it is no longer eligible ([`VT_INELIGIBLE`](@ref
   VTReleaseReason)) — which includes a satellite without a record for two
   cycles, whose slot is freed — or drops below the horizon
   ([`VT_BELOW_HORIZON`](@ref VTReleaseReason)). Its scalar loop is re-seeded
   from the replica at its landing.
5. After `insufficient_meas_timeout` of unsolvable cycles, or with no member
   left, every member is released ([`VT_FALLBACK`](@ref VTReleaseReason)) and
   the next cycle solves the scalar PVT again, until a fresh fix seeds the filter
   anew.

A re-acquired satellite — a fresh state on a PRN seen before — gets its old slot
back: its decoder restarts its sync but keeps the data it had decoded, so it is
ready again after the next subframe rather than a whole frame once the host's
bit clock has found the bit edges again.

## Stepping the satellites

A cycle runs once every satellite has completed its epoch's snapshot, and a
satellite that has gone two cycles without a record of one of its signals is
dropped. So a host should step every signal of every satellite of the estimator
at least once per half cycle. Tracking.jl's `track!` does, for any chunk
shorter than that. A satellite that falls behind holds a cycle up only until
the others reach the next epoch; then the cycle runs without it. A host never
has to hold a record back or order its channels: each record is stepped as it
arrives.

## Under a hardware NCO delay

With a hardware correlator, the command computed from a record only takes
effect when it lands at the NCO. Pass the landing sample to `step_loop`, and
the channel's [`NCOTimeline`](@ref) as its words: each satellite then sizes its
corrections for the moment its own command lands, against the replica the
timeline predicts there, and a released satellite's scalar loop takes over from
that replica. The landing may lie up to 2.5 cycles after the epoch.

## Reading the results

Nothing is logged. [`navigation_solution`](@ref) is the latest solution (the
scalar PVT's before the filter is seeded) and [`navigation_status`](@ref) the
[`VTStatus`](@ref) of the latest cycle: the events (the filter was seeded, fell
back, released a satellite) and the filter's position and clock uncertainties,
also at hand as [`position_uncertainty`](@ref) and
[`clock_uncertainty`](@ref). [`release_reason`](@ref) says whether and why the
latest cycle released a satellite, and [`member_sats`](@ref) reports every
member of the loop, measured and coasted.

[`navigation_cycle`](@ref) counts the cycles, so a consumer that polls it after
each step reads every solution exactly once, and [`navigation_epoch`](@ref) is
the moment the latest solution describes, on the records' time grid.
[`satellite_report`](@ref) hands out what the engine knows of a satellite — its
decoder, bit sync, C/N₀, lock, readiness for the PVT and membership — so a
receiver neither decodes the bits nor estimates the C/N₀ a second time.

Every estimator answers these: the scalar loops with `nothing`, so a host can
ask whichever estimator it was given.

```julia
cycle = navigation_cycle(estimator)
if cycle != last_cycle            # `nothing` for a scalar loop: never a new solution
    last_cycle = cycle
    pvt = navigation_solution(estimator)       # position, velocity, time, DOP, …
    report = satellite_report(estimator, GPSL1CA(), prn)
end
```

The solution, the per-member report and the satellite reports are the
estimator's own objects, reused by the next cycle or call: copy out what is
needed later.

## Storage

The estimator allocates the slots of `max_satellites_per_signal` satellites per
ranging signal at construction: decoders and the filter's buffers. A dropped satellite leaves its slot free with all its storage, for the
next one to reuse; only past that capacity does a group grow. Once warm, records
and cycles allocate nothing, and the whole estimator compiles with
`juliac --trim=safe`.

## Known limits

- A pair ranges on the driver alone: the data component's discriminators are
  not combined with the pilot's.
- Lock is a C/N₀ threshold over the bit-synced satellites, not a full lock
  detector.
- A host must report the record's `prn` and `code_phase`. HardwareLoopCore does
  not yet.

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
