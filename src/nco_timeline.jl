# Physical NCO Doppler commands and their device-sample timeline.

"No explicit application sample: the integration end is used."
const NO_LANDING_SAMPLE = typemin(Int64)

"""
    NCOCommand(carrier_doppler_hz, code_doppler_hz)

Physical carrier and code Doppler offsets in Hz, stored as plain `Float64` values.
These are frequency offsets, independent of any device register encoding. Read them
by name: `command.carrier_doppler`, or `(; carrier_doppler, code_doppler) = command`.
"""
struct NCOCommand
    carrier_doppler::Float64
    code_doppler::Float64
end

"""
A physical NCO command queued in an [`NCOTimeline`](@ref), in effect from `sample_index`:
the sample it was scheduled for, which names it for the device's feedback, until
`confirmed`, then the one it really took effect at.
"""
struct ScheduledNCOCommand
    sample_index::Int64
    confirmed::Bool
    command::NCOCommand
end

"""
    NCOTimeline(; capacity = 64)

The physical NCO command in effect and commands scheduled at named device samples.
[`calc_mean_nco_command`](@ref) attributes integrations to their replica frequencies.
Scheduled commands occupy a fixed vector of `capacity` entries, so nothing
allocates; [`schedule_command!`](@ref) refuses a command that does not fit.

Hardware integrations may schedule predicted application samples, then use
[`confirm_command!`](@ref) to record actual application or [`cancel_command!`](@ref)
for rejection. Resolve that feedback before advancing past the affected history.
"""
mutable struct NCOTimeline
    applied_command::NCOCommand
    const scheduled_commands::Vector{ScheduledNCOCommand}
    num_scheduled_commands::Int
end

function NCOTimeline(; capacity::Integer = 64)
    capacity >= 1 || throw(ArgumentError("an NCOTimeline needs a capacity of at least 1"))
    NCOTimeline(NCOCommand(0.0, 0.0),
        [ScheduledNCOCommand(0, false, NCOCommand(0.0, 0.0)) for _ = 1:capacity], 0)
end

"A handover: start on the given Doppler offsets with no commands in flight."
function reset_timeline!(timeline::NCOTimeline, carrier_doppler_hz, code_doppler_hz)
    timeline.applied_command = NCOCommand(carrier_doppler_hz, code_doppler_hz)
    timeline.num_scheduled_commands = 0
    timeline
end

"""
    schedule_command!(timeline, sample_index, carrier_doppler_hz, code_doppler_hz)

Schedule physical Doppler offsets at `sample_index` and return `true`. Anything queued
at or past that sample is superseded: a device keeps the newest command per sample, and
a later command never targets an earlier sample. With the queue full, return `false`
and leave the timeline as it is: the host must not write that command to the device,
and a queue that stays full means [`advance_timeline!`](@ref) is not keeping up.
"""
function schedule_command!(timeline::NCOTimeline, sample_index, carrier_doppler_hz, code_doppler_hz)
    commands = timeline.scheduled_commands
    count = timeline.num_scheduled_commands
    @inbounds while count > 0 && commands[count].sample_index >= sample_index
        count -= 1
    end
    count < length(commands) || return false
    @inbounds commands[count+1] =
        ScheduledNCOCommand(Int64(sample_index), false, NCOCommand(carrier_doppler_hz, code_doppler_hz))
    timeline.num_scheduled_commands = count + 1
    true
end

# The unconfirmed command scheduled for `predicted_sample_index`, `0` for none. There is
# at most one: scheduling supersedes everything at or past the new sample.
function _pending_command_index(timeline::NCOTimeline, predicted_sample_index)
    @inbounds for i = timeline.num_scheduled_commands:-1:1
        scheduled = timeline.scheduled_commands[i]
        !scheduled.confirmed && scheduled.sample_index == predicted_sample_index && return i
    end
    0
end

function _remove_scheduled_command!(timeline::NCOTimeline, index::Int)
    count = timeline.num_scheduled_commands
    @inbounds for i = index:(count-1)
        timeline.scheduled_commands[i] = timeline.scheduled_commands[i+1]
    end
    timeline.num_scheduled_commands = count - 1
    timeline
end

"""
    confirm_command!(timeline, predicted_sample_index, actual_sample_index)

The command scheduled for `predicted_sample_index` really took effect at
`actual_sample_index`: move it there, preserving every other command. A pending command
between the two samples, or at the actual one, keeps its own sample, so its own feedback still finds it. Among commands at one sample the latest confirmed is in
effect, as the device keeps the newest write. A no-op when no unconfirmed command was
scheduled for `predicted_sample_index`. Confirm before [`advance_timeline!`](@ref)
retires it.
"""
function confirm_command!(timeline::NCOTimeline, predicted_sample_index, actual_sample_index)
    index = _pending_command_index(timeline, predicted_sample_index)
    index == 0 && return timeline
    commands = timeline.scheduled_commands
    @inbounds scheduled = commands[index]
    command = ScheduledNCOCommand(Int64(actual_sample_index), true, scheduled.command)
    # Remove and reinsert after every command at or before the actual sample; feedback
    # does not invoke suffix overriding.
    _remove_scheduled_command!(timeline, index)
    count = timeline.num_scheduled_commands
    insert_at = count + 1
    @inbounds while insert_at > 1 && commands[insert_at-1].sample_index > actual_sample_index
        commands[insert_at] = commands[insert_at-1]
        insert_at -= 1
    end
    @inbounds commands[insert_at] = command
    timeline.num_scheduled_commands = count + 1
    timeline
end

"""
    cancel_command!(timeline, predicted_sample_index)

Remove the command scheduled for `predicted_sample_index` that the device rejected,
preserving every other command. Cancel before advancing past its predicted
application. A no-op when no unconfirmed command was scheduled for
`predicted_sample_index`.
"""
function cancel_command!(timeline::NCOTimeline, predicted_sample_index)
    index = _pending_command_index(timeline, predicted_sample_index)
    index == 0 && return timeline
    _remove_scheduled_command!(timeline, index)
end

"""
    advance_timeline!(timeline, sample_index)

Fold commands at or before `sample_index` into the applied command and retire them.
Call only after hardware feedback is resolved and no future query starts before
that sample. Advancing relies on scheduled application times; it does not itself
verify device execution.
"""
function advance_timeline!(timeline::NCOTimeline, sample_index)
    commands = timeline.scheduled_commands
    count = timeline.num_scheduled_commands
    n = 0
    @inbounds for i = 1:count
        command = commands[i]
        command.sample_index <= sample_index || break
        timeline.applied_command = command.command
        n += 1
    end
    if n > 0
        @inbounds for i = 1:(count-n)
            commands[i] = commands[i+n]
        end
        timeline.num_scheduled_commands = count - n
    end
    timeline
end

"The physical NCO command in effect at device sample `sample_index`."
function get_nco_command(timeline::NCOTimeline, sample_index)
    command = timeline.applied_command
    @inbounds for i = 1:timeline.num_scheduled_commands
        scheduled = timeline.scheduled_commands[i]
        scheduled.sample_index <= sample_index || break
        command = scheduled.command
    end
    command
end

"""
    calc_mean_nco_command(commands, a, b) -> NCOCommand

Time-weighted physical Doppler offsets in effect over device samples `[a, b)`.
A command applied at `s` is in effect from `s` onwards. For `b <= a`, return the
command at `a`. Accepts an [`NCOTimeline`](@ref) or [`ConstantNCOCommand`](@ref).
"""
function calc_mean_nco_command(timeline::NCOTimeline, a::Real, b::Real)
    total = b - a
    total > 0 || return get_nco_command(timeline, a)
    (; carrier_doppler, code_doppler) = timeline.applied_command
    carrier_sum = 0.0
    code_sum = 0.0
    t = a
    @inbounds for i = 1:timeline.num_scheduled_commands
        scheduled = timeline.scheduled_commands[i]
        scheduled.sample_index >= b && break
        if scheduled.sample_index > t
            carrier_sum += carrier_doppler * (scheduled.sample_index - t)
            code_sum += code_doppler * (scheduled.sample_index - t)
            t = scheduled.sample_index
        end
        (; carrier_doppler, code_doppler) = scheduled.command
    end
    carrier_sum += carrier_doppler * (b - t)
    code_sum += code_doppler * (b - t)
    NCOCommand(carrier_sum / total, code_sum / total)
end

"""
    ConstantNCOCommand(carrier_doppler_hz, code_doppler_hz)
    ConstantNCOCommand(command::NCOCommand)

A physical command held constant for every queried span, as in a software receiver
within one chunk. See [`calc_mean_nco_command`](@ref).
"""
struct ConstantNCOCommand
    command::NCOCommand
end
ConstantNCOCommand(carrier_doppler_hz, code_doppler_hz) =
    ConstantNCOCommand(NCOCommand(carrier_doppler_hz, code_doppler_hz))
calc_mean_nco_command(command::ConstantNCOCommand, a::Real, b::Real) = command.command
get_nco_command(command::ConstantNCOCommand, sample_index) = command.command
