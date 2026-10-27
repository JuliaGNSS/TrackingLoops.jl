```@meta
CurrentModule = TrackingLoops
```

# NCO timeline

An [`NCOCommand`](@ref) describes physical carrier and code Doppler offsets in Hz.
Both offsets are stored as plain `Float64` values. Device adapters translate these
frequencies to their hardware representation.

A hardware correlator holds each command until the next one takes effect, often
several records after the record that motivated it. An [`NCOTimeline`](@ref)
tracks the applied command and commands scheduled at named device samples.
[`calc_mean_nco_command`](@ref) attributes a record covering `[a, b)` to its
actual replica frequencies. A command applied at `b` belongs to the next record.
The result is an `NCOCommand` with the fields `carrier_doppler` and `code_doppler`.
A software correlator can use [`ConstantNCOCommand`](@ref) for a command held
throughout a chunk.

For hardware with delayed feedback, schedule the predicted application sample,
confirm its actual sample when the device acknowledges execution, or cancel the
command if the device rejects it:

```julia
timeline = NCOTimeline(; capacity = 64)
reset_timeline!(timeline, 100.0, 0.1)
schedule_command!(timeline, 1000, 110.0, 0.11) || error("NCO timeline full")
confirm_command!(timeline, 1000, 1200)
(; carrier_doppler, code_doppler) = calc_mean_nco_command(timeline, 1000, 1400)
```

A queued command is named by the sample it was scheduled for, and keeps that name when
it is confirmed elsewhere. [`confirm_command!`](@ref) moves only the named command and
preserves every other one, also one it lands on: among commands at one sample the latest
confirmed is in effect, as the device keeps the newest write.
[`cancel_command!`](@ref) removes only the named rejected command. Neither throws:
feedback for a command no longer scheduled is a no-op, so the loop never stops on a late
or repeated acknowledgement.

Scheduling a fresh command replaces the pending suffix at or after its sample, as a
device keeps the newest command per sample.

Resolve hardware feedback before [`advance_timeline!`](@ref) retires the affected
history, and advance only when future queries will start at or beyond that sample.
Advancement trusts the recorded application samples. The fixed-capacity queue never
allocates and never rewrites its history: `schedule_command!` returns `false` for a
command that does not fit, and the host does not write that command to the device.
Size the queue for outstanding commands and retire consumed history regularly.

```@docs
NO_LANDING_SAMPLE
NCOCommand
ConstantNCOCommand
TrackingLoops.ScheduledNCOCommand
NCOTimeline
calc_mean_nco_command(::NCOTimeline, ::Real, ::Real)
get_nco_command(::NCOTimeline, ::Any)
advance_timeline!(::NCOTimeline, ::Any)
confirm_command!(::NCOTimeline, ::Any, ::Any)
cancel_command!(::NCOTimeline, ::Any)
reset_timeline!(::NCOTimeline, ::Any, ::Any)
schedule_command!(::NCOTimeline, ::Any, ::Any, ::Any)
```
