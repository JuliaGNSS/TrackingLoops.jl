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
[`step_loop`](@ref). Everything vector tracking needs happens inside that step,
from what the records carry. All satellites stepped with one estimator share
its navigation engine, which

- syncs to each satellite's navigation bits, decodes them and estimates its
  C/N₀, on the satellite's own bit clock;
- snapshots each satellite at every navigation epoch (every `cycle_time` on the
  records' time grid): code phase from the last data-symbol edge, decoder,
  Dopplers and the discriminators accumulated since the last epoch;
- runs the epoch's navigation cycle on the record that brings the last satellite
  past the epoch: the scalar PVT until the filter is seeded, a filter iteration
  after that;
- leaves the cycle's decisions (admission, release, corrections) for each
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
construction. Each satellite runs the scalar loop `inner` until the filter takes
it over, and again once the filter lets it go. In the vector loop the code
filter is frozen, the filter's corrections steer the replica, and the
discriminators are accumulated for the filter.

## Several signals of a satellite

A satellite tracked on several signals of one band lists them as a group, the
driver first:

```julia
estimator = VectorPLLAndDLL((GalileoE1C(), GalileoE1B()), GPSL1CA(); combine_signals = true)
```

A host that keeps its satellites' signal groups builds the settings alone,
[`VectorTrackingSettings`](@ref) with the estimator's keywords, and
[`with_signal_groups`](@ref) builds the estimator for its groups, so they are
listed once, as Tracking.jl's `TrackState` does:

```julia
settings = VectorTrackingSettings(; combine_signals = true)
estimator = with_signal_groups(settings, (GalileoE1C(), GalileoE1B()), GPSL1CA())
```

The host steps the driver's records with [`step_loop`](@ref) and folds the
passengers' with [`fold_passenger_record`](@ref) (see [Host contract](@ref)). The
signals of a group are one constellation's and share the driver's code rate and
carrier frequency.
The engine decodes the bits of the group's first signal that carries navigation
data: the driver, or for a dataless pilot driver its data passenger, whose
records then run the satellite's bit clock and decoder. The filter ranges on the
driver and reports the satellite by it.

With `combine_signals = true` the passengers are combined as in the scalar loops
([Signal combining](@ref)) out of the vector loop, and into the PLL in it. The
filter's code and rate measurements instead fuse every signal's DLL and raw FLL
readings, a passenger's in the cycle of the driver record it ends within. Each
signal's cycle mean is weighted by its inverse variance, from its own C/N₀,
coherent integration time, tap spacing and the part of the cycle its readings
cover; the fused variance is the inverse of the summed weights. The variances
take every reading of a cycle at the signal's latest coherent integration time,
so in the one cycle where a signal's records change length (at bit sync, say)
they are off by up to the ratio of the two lengths. Only thermal noise averages
down: orbit, clock and atmosphere are common to all signals of a satellite. The
code variance follows the BPSK early-minus-late model, scaled by 0.4 for the
very-early-prompt-late correlator (Galileo E1, the BOC(1,1) family; a simulation
gave 0.36 for CBOC, 0.43 for BOC(1,1)). At a low C/N₀ · T_coh the model
overstates every signal's variance alike, as the normalised discriminators
saturate.

A passenger's DLL readings count only where its group delay relative to the
driver is given (`differential_group_delay_chips`), and are referred to the
driver's code phase by it; its FLL readings are four-quadrant where its record
has a polarity. A member with no code reading in a cycle is withheld from it, one
with no rate reading from its rate row only. Without `combine_signals` the filter
reads the driver alone, and the passengers' records only decode the bits where
the driver carries none.

The two weightings are deliberate. Whatever closes a carrier loop on combined
readings (the scalar fallback, and the PLL in the vector loop) weights by the
ICD power split as the scalar loops do, so a satellite's loops do not change
when it enters or leaves the vector loop. The filter's measurements are
weighted by measured C/N₀ because the filter needs their variances anyway, and
the measured C/N₀ also captures what the power split does not, such as a
passenger's own antenna gain. That C/N₀, which also decides lock, is the host's
estimate where the records carry one (`cn0`, see [`LoopRecord`](@ref)), the
engine's own from the prompts otherwise. The engine's own restarts where a
signal's records change length (at bit sync, say), keeping the estimate from
before until it has refilled.

## Staging and discriminators under vector tracking

Out of the vector loop, `inner` stages its carrier loop as the scalar loop does
([Carrier loop staging](@ref)). In the vector loop the FLL branch carries the
navigation filter's carrier correction, so it is never dropped, the PLL runs at
the narrow bandwidth, and the stage and the phase-lock indicator are frozen. A
satellite the filter takes over keeps `inner`'s state; one it releases re-seeds
`inner` from its replica's Dopplers, which restarts the staging.

The discriminators follow the record's `polarity` in both modes: where it says
the replica wipes off every sign modulation of the prompt (a pilot synced to its
secondary code) the PLL and the FLL read four-quadrant, otherwise two-quadrant
(see [`LoopRecord`](@ref)). So a pilot's rate measurement has the four-quadrant
FLL's ±1/(2T) range rather than ±1/(4T). A cycle without any FLL reading for a
satellite withholds only its rate row and keeps its pseudorange row.

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
   loop and take their first corrections on their next record.
3. While it runs, a satellite is admitted once it is decoded, healthy, in lock
   and a degree above the horizon. A member out of lock stays in the loop,
   unmeasured, and is predicted through the outage. A member that misses an
   epoch sits that cycle out.
4. A member is released when it is no longer eligible ([`VT_INELIGIBLE`](@ref
   VTReleaseReason)), which includes a satellite without a record for two
   cycles, whose slot is freed, or drops below the horizon
   ([`VT_BELOW_HORIZON`](@ref VTReleaseReason)). Its scalar loop is re-seeded
   from the replica at its landing.
5. After `insufficient_meas_timeout` of unsolvable cycles, or with no member
   left, every member is released ([`VT_FALLBACK`](@ref VTReleaseReason)) and
   the scalar PVT runs again until a fresh fix seeds the filter anew.

A re-acquired satellite (a fresh state on a PRN seen before) gets its old slot
back: its bit clock and decoder sync restart, but the decoder keeps its data, so
it is ready again after the next subframe rather than a whole frame.

## Stepping the satellites

A cycle runs once every satellite has reached its epoch, and a satellite without
a record for two cycles is dropped, so a host should step every satellite at
least once per half cycle (Tracking.jl's `track!` does, for any chunk shorter
than that). A satellite that falls behind holds a cycle up only until the
others reach the next epoch.

With a hardware correlator, pass the landing sample to `step_loop` and the
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

All storage (decoders, bit clocks, C/N₀ estimators, filter buffers) is
allocated at construction for `max_satellites_per_signal` satellites per signal;
a dropped satellite leaves its slot to the next one. Once warm, records and
cycles allocate nothing, and the estimator compiles with `juliac --trim=safe`.

## Known limits

- A dataless pilot (GPS L1C-P, Galileo E1C) runs vector tracking only with its
  data component as a passenger.
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
