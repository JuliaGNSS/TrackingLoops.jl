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
each satellite's state with [`init_estimator_state`](@ref) and steps it record
by record with [`step_loop`](@ref). Everything vector tracking needs happens
inside that step, from what the records carry. Every satellite stepped with one
estimator shares its navigation engine, which

- syncs to each satellite's navigation bits, decodes them and estimates its
  C/N₀, on the satellite's own bit clock;
- snapshots each satellite at every navigation epoch (every `cycle_time` on the
  records' time grid): the replica's code phase from the last data-symbol edge,
  the decoder there, the Dopplers, and the discriminators accumulated since the
  last epoch;
- runs the epoch's navigation cycle on the record that brings the last satellite
  past the epoch: the scalar PVT until the filter is seeded, a filter iteration
  after that;
- leaves the cycle's decisions — admission, release, the corrections — for each
  satellite to take up on its next record, sized for where its command lands.

```julia
estimator = VectorPLLAndDLL(GPSL1CA(), GalileoE1B())   # owns decoders, filter and PVT
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

## What the records must carry

The engine derives everything from the records except what only the correlator
knows, which every [`LoopRecord`](@ref) handed to the estimator must carry:

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
2. The first fix seeds the filter. The fix's satellites join the vector loop,
   and take their first corrections, computed at the seeded state, on their next
   record.
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
back: its bit clock restarts, and its decoder restarts its sync but keeps the
data it had decoded, so it is ready again after the next subframe rather than a
whole frame.

## Stepping the satellites

A cycle runs once every satellite has reached its epoch, and a satellite that
has gone two cycles without a record is dropped. So a host should step every
satellite of the estimator at least once per half cycle. Tracking.jl's `track!`
does, for any chunk shorter than that. A satellite that falls behind holds a
cycle up only until the others reach the next epoch; then the cycle runs
without it.

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

## Storage

The estimator allocates the slots of `max_satellites_per_signal` satellites per
signal at construction: decoders, bit clocks, C/N₀ estimators and the filter's
buffers. A dropped satellite leaves its slot free with all its storage, for the
next one to reuse; only past that capacity does a group grow. Once warm, records
and cycles allocate nothing, and the whole estimator compiles with
`juliac --trim=safe`.

## Known limits

- The engine decodes the bits of the signal it steps, so a dataless pilot
  (GPS L1C-P, Galileo E1C) cannot run vector tracking: the constructor rejects
  it.
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
