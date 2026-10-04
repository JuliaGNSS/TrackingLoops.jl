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

The work is split in two:

- per record, [`VectorPLLAndDLL`](@ref) is stepped with the same
  [`step_loop`](@ref) as every other estimator. Out of the vector loop it is the
  scalar loop it wraps, bit for bit. In it, the code filter is frozen, the
  filter's corrections steer the replica, and the discriminators are accumulated
  for the filter;
- once per navigation cycle, [`update_navigation!`](@ref) runs the filter over
  every satellite and writes each member's corrections back into its estimator
  state.

The interface between the two is plain data: one [`VTSignalGroup`](@ref) of
[`VTSat`](@ref) slots per ranging signal, which the caller fills from its own
channels. It knows neither a correlator nor a device, so the software receiver
and the loop process of a hardware correlator drive the same code.

## The lifecycle of a satellite

Vector tracking needs a decoded navigation message and a first position fix, so
every satellite starts on its scalar loop:

1. Until vector tracking runs, each cycle solves the scalar PVT over the
   satellites marked `pvt_ready`.
2. The first fix seeds the filter. The fix's satellites that are still tracked
   join the vector loop, and their loops are closed a first time at the seeded
   state.
3. While it runs, a satellite is admitted once it is decoded, healthy, in lock
   and a degree above the horizon. A member out of lock stays in the loop,
   unmeasured, and is predicted through the outage.
4. A member is released when it is no longer eligible ([`VT_INELIGIBLE`](@ref
   VTReleaseReason)) or drops below the horizon ([`VT_BELOW_HORIZON`](@ref
   VTReleaseReason)). Its scalar loop is re-seeded from the replica it takes
   over from ([`release_from_vector_tracking`](@ref)).
5. After `insufficient_meas_timeout` of unsolvable cycles, or with no member
   left, every member is released ([`VT_FALLBACK`](@ref VTReleaseReason)) and
   the next cycle solves the scalar PVT again, until a fresh fix seeds the filter
   anew.

## Driving the loop

Wrap each satellite's scalar loop, `ConventionalAssistedPLLAndDLL()` or
`NCOReferencedPLLAndDLL()`, in a `VectorPLLAndDLL`:

```julia
estimator = VectorPLLAndDLL(ConventionalAssistedPLLAndDLL())  # or NCOReferencedPLLAndDLL()
state = init_estimator_state(estimator, signal, carrier_doppler, code_doppler)
```

Build one `VTSignalGroup` per signal, with as many slots as the receiver tracks
of it at most, and the filter once for the tuple of groups:

```julia
gps = VTSignalGroup(GPSL1CA(), [VTSat(decoder, state) for (decoder, state) in channels])
vt = VectorTrackingState(VectorTracking(), (gps,))
```

Once per navigation cycle, `cycle_time` after the previous one, fill the slots
and run the cycle:

```julia
for (sat, channel) in zip(gps.sats, channels)
    sat.active = true
    sat.decoder = channel.decoder          # keep it current with decode_soft_bits!
    sat.estimator_state = channel.state
    sat.code_phase = channel.code_phase    # the replica running at the cycle epoch
    sat.carrier_doppler = channel.carrier_doppler
    sat.code_doppler = channel.code_doppler
    sat.code_phase_at_landing = channel.code_phase  # no NCO delay
    sat.carrier_doppler_at_landing = channel.carrier_doppler
    sat.code_doppler_at_landing = channel.code_doppler
    sat.cn0_dbhz = channel.cn0
    sat.coherent_integration_time = channel.coherent_integration_time  # the last dump's
    sat.early_late_spacing = channel.early_late_spacing                # chips
    sat.in_lock = channel.in_lock
    sat.pvt_ready = channel.pvt_ready
end
pvt, status = update_navigation!(vt, (gps,), cycle_time)
# copy every `sat.estimator_state` back to its channel, and act on `sat.release_reason`
```

`cycle_time` is the measured interval, not the nominal one: the process model
propagates each cycle by its own length. `coherent_integration_time` and
`early_late_spacing` size the measurement noise, so they must be the
correlator's own. [`decode_soft_bits!`](@ref) feeds a channel's completed soft
bits to its decoder.

Several constellations or bands share one filter: pass one group per signal, in
the same order to [`VectorTrackingState`](@ref) and to every
[`update_navigation!`](@ref) call. With `config = nothing` the state only ever
solves the scalar PVT, so a receiver has one path whether vector tracking is
enabled or not.

## Under a hardware NCO delay

With a hardware correlator, the command computed at a cycle's epoch only takes
effect when it lands at the NCO, `landing_lead` later. Fill the epoch fields
with the replica that was actually running at the epoch (read from the
channel's [`NCOTimeline`](@ref), not the last command). Fill the landing fields
(`landing_lead`, `code_phase_at_landing`, `carrier_doppler_at_landing`,
`code_doppler_at_landing`) with the replica predicted at the landing under the
words already committed. The corrections are then sized for the moment each one
lands, and a released satellite's scalar loop takes over from that replica. The
landing may lie up to 2.5 cycles after the epoch. Without a delay, the landing
fields equal the epoch's and `landing_lead` is zero.

## Reporting

Nothing is logged. The [`VTStatus`](@ref) of each cycle carries the events (the
filter was seeded, fell back, released a satellite) and the filter's position
and clock uncertainties. Each `VTSat` carries its
[`release_reason`](@ref VTReleaseReason). The state keeps the latest solution
(`vt.pvt`) and, in `vt.member_sats`, every member of the loop, measured and
coasted.

Once warm, a cycle allocates nothing and compiles with `juliac --trim=safe`.

## The per-record estimator

```@autodocs
Modules = [TrackingLoops]
Pages = ["vector/estimator.jl"]
Private = false
```

## The navigation cycle

```@autodocs
Modules = [TrackingLoops]
Pages = ["vector/tracking.jl", "vector/model.jl"]
Private = false
```
