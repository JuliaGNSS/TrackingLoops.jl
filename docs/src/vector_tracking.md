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

## Several signals of a satellite

A satellite tracked on several signals of one band lists them as a group, the
driver first, in the order the host steps them:

```julia
estimator = VectorPLLAndDLL((GalileoE1C(), GalileoE1B()), GPSL1CA(); combine_signals = true)
```

A host that already keeps its satellites' signal groups builds the settings
alone, [`VectorTrackingSettings`](@ref) with the estimator's keywords, and
[`with_signal_groups`](@ref) builds the estimator for its groups, so they are
listed once; Tracking.jl's `TrackState` does this:

```julia
settings = VectorTrackingSettings(; combine_signals = true)
estimator = with_signal_groups(settings, (GalileoE1C(), GalileoE1B()), GPSL1CA())
```

The settings are no estimator: the host steps, and the results are read from, the
estimator `with_signal_groups` returned. Handed an estimator already built with its
signals, `with_signal_groups` returns it for any of its groups, in any order, and
throws for a group it was not built for.

The host steps the driver's records with [`step_loop`](@ref). Every passenger
record goes to [`fold_passenger_record`](@ref), which such an estimator takes
([`takes_passenger_records`](@ref) is `true`): in sample order, each before the
driver record it ends within. A passenger record carries the passenger's own
`prn` and `code_phase`, and the `fold_end` of the driver's fold (see
[Host contract](@ref)). The signals of a group are one constellation's and
share the driver's code rate and carrier frequency.

- **Decoding.** The engine decodes the bits of the group's first signal that
  carries navigation data: the driver, or for a dataless pilot driver its data
  passenger, whose records then run the satellite's bit clock and decoder. The
  filter ranges on the driver, and reports the satellite by it.
- **Combining** (`combine_signals = true`, the vector loop's own switch, as
  `ConventionalPLLAndDLL`'s is the scalar loop's; it covers the scalar fallback
  too, so `inner` is built without it). Out of the vector loop the passengers are
  combined as in the scalar loops ([Signal combining](@ref)); in it, into the
  PLL. The code and rate measurements the filter owns take every signal's DLL
  and raw FLL readings. A passenger's readings wait for the driver's next record
  and join its cycle, as the scalar loops combine them into that record: a
  reading moves by at most one driver record across an epoch. Each signal's
  cycle mean is weighted by its inverse variance, built from that signal's own
  C/N₀, coherent integration time and tap spacing, and from the span its
  readings cover that cycle, so a signal with readings for part of a cycle only
  weighs that much less. The fused variance is the inverse of the summed
  weights. The variances take every reading of a cycle at the signal's latest
  coherent integration time: in the one cycle where a signal's records change
  length (at bit sync, say) they are off by up to the ratio of the two lengths. Only the thermal noise averages
  down: the orbit, clock and atmosphere are common to every signal of a
  satellite. The code variance follows the BPSK early-minus-late model; for a
  signal tracked with the very-early-prompt-late correlator (Galileo E1, the
  BOC(1,1) family) it is scaled by 0.4, a first-order correction from a
  simulation of its discriminator (0.36 for CBOC, 0.43 for BOC(1,1)). At a low
  C/N₀ · T_coh the model overstates every signal's variance alike, as the
  normalised discriminators saturate. A passenger's DLL
  readings are taken only where its group delay relative to the driver is given
  (`differential_group_delay_chips`), referred to the driver's code phase by
  it. A passenger's FLL readings are read four-quadrant where its record has a
  polarity. A member with no code reading of any signal in a
  cycle is withheld from it, one with no rate reading from its rate row only.
  Without `combine_signals` the filter reads the driver's readings alone, and
  the passengers' records only decode the bits where the driver carries none.
- **Two weightings, by design.** Whatever closes a carrier loop on the combined
  readings — the scalar fallback out of the vector loop, and the PLL aiding in
  it — weights them by the ICD power split, exactly as the scalar loops do
  ([Signal combining](@ref)), so a satellite's loops do not change when it
  enters or leaves the vector loop. Only the navigation filter's code and rate
  measurements are weighted by inverse variance, from each signal's C/N₀: the
  filter needs their variances anyway, and the measured C/N₀ also covers what
  the power split does not, such as a passenger's own antenna gain.
- **C/N₀.** The engine weights the measurements and decides lock by each
  signal's C/N₀: the host's estimate where its records carry one (their `cn0`,
  see [`LoopRecord`](@ref)), its own estimate from the prompts otherwise. Its
  own restarts where a signal's records change length (at bit sync, say), as
  prompts of two lengths cannot share one estimate, and the estimate from before
  is kept until the restarted one has refilled.

## Staging and discriminators under vector tracking

A satellite's carrier loop stages as the scalar loop does only while it is out
of the vector loop: `inner` runs FLL-assisted at the wide bandwidth until
phase lock, then as a PLL alone, narrowed once lock has held (see
[`CarrierLoopStage`](@ref)). In the vector loop the FLL branch carries the
navigation filter's carrier correction, so it is never dropped there, the PLL
runs at the narrow bandwidth, and the stage and the phase-lock indicator are left
as they are. A satellite the filter takes over keeps `inner`'s state; one it
releases re-seeds `inner` from the Dopplers its replica runs at, which restarts
the staging.

The discriminators are the record's in both modes. Where the record's
`polarity` (see [`LoopRecord`](@ref)) says the replica wipes off every sign
modulation of its prompt (a pilot synced to its secondary code), the PLL and the
FLL read four-quadrant, otherwise two-quadrant. In the vector loop the PLL still
closes on that reading, and the raw FLL reading accumulated for the filter is
four-quadrant on such a record, so a pilot's rate measurement has the ±1/(2T) range rather than ±1/(4T).
A record without a previous prompt has no FLL reading and is left out of the
cycle's rate measurement; a cycle without any FLL reading withholds the
satellite's rate row and keeps its pseudorange row. Passengers combined into the
PLL read it two-quadrant, and only while the driver's own reading lies within
that range ([Signal combining](@ref)); the FLL readings the filter fuses are
the passengers' own, four-quadrant where the record has a polarity, as the
driver's.

## What the records must carry

The engine derives everything from the records except what only the correlator
knows: the satellite (`prn`), the replica's code phase at the record's end
(`code_phase`) and a time grid shared by all satellites. The
[Host contract](@ref) lists them with everything else a host owes the
estimators.

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
signal at construction: decoders, bit clocks, C/N₀ estimators and the filter's
buffers. A dropped satellite leaves its slot free with all its storage, for the
next one to reuse; only past that capacity does a group grow. Once warm, records
and cycles allocate nothing, and the whole estimator compiles with
`juliac --trim=safe`.

## Known limits

- A dataless pilot (GPS L1C-P, Galileo E1C) runs vector tracking only with its
  data component as a passenger, whose bits the engine decodes: on its own the
  constructor rejects it.
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
