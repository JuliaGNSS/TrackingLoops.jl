# Entry point of the `juliac --trim=safe` check (see `check.jl`): vector tracking on
# the synthetic GPS and Galileo satellites of `../vector_simulation.jl`, from the
# scalar fix through a partial outage to a steady state, printed so the trimmed
# executable's output can be compared against a regular Julia session. It is the
# loop process's workload: the per-record `step_loop` of every channel and
# `update_navigation!` once per cycle.
using TrackingLoops, GNSSSignals, StaticArrays, Unitful, LinearAlgebra
using Unitful: Hz, s, ms

include(joinpath(@__DIR__, "..", "vector_simulation.jl"))

# Every channel of one group through one millisecond; the records end on the
# group's own code period.
function step_group!(channels, estimator, truth, k, sample_index, landing)
    n = record_ms(first(channels).signal)
    k % n == 0 || return nothing
    for sat in channels
        simulate_record!(sat, estimator, truth, truth.t0 + sample_index / 4e6,
            n * SAMPLES_PER_MS, sample_index, landing)
    end
    nothing
end

function fill_group!(channels, group, epoch, landing, outage::Bool)
    for (i, sat) in enumerate(channels)
        sat.in_view = !(outage && i <= 2)
        fill_vtsat!(group.sats[i], sat, epoch, landing)
    end
    nothing
end

function copy_back!(channels, group)
    for (i, sat) in enumerate(channels)
        sat.state = group.sats[i].estimator_state
    end
    nothing
end

# One `print` per value: a long `print(io, xs...)` is not specialised on its
# arguments' types, which leaves the call dynamic.
function report(io, cycle, pvt, status)
    print(io, "cycle ", cycle, ":")
    for x in (pvt.position.x, pvt.position.y, pvt.position.z, pvt.velocity.x,
        pvt.velocity.y, pvt.velocity.z, ustrip(pvt.time_correction),
        status.position_std.val, status.time_with_insufficient_meas.val)
        print(io, " ", x)
    end
    print(io, " ", length(pvt.sats))
    print(io, " ", status.num_members)
    print(io, " ", status.running)
    print(io, " ", status.enabled)
    t = pvt.time
    t === nothing || print(io, " ", t.second)
    println(io)
end

function (@main)(args::Vector{String})::Cint
    io = Core.stdout
    rx = SimReceiver(; signals = (GPSL1CA(), GalileoE1B()))
    channels, groups = rx.channels, rx.groups
    sample = 0
    for cycle = 1:60
        for k = 1:rx.cycle_ms
            sample_index = sample + k * SAMPLES_PER_MS
            step_group!(channels[1], rx.estimator, rx.truth, k, sample_index, NO_LANDING_SAMPLE)
            step_group!(channels[2], rx.estimator, rx.truth, k, sample_index, NO_LANDING_SAMPLE)
        end
        sample += rx.cycle_ms * SAMPLES_PER_MS
        outage = 20 <= cycle <= 30
        fill_group!(channels[1], groups[1], sample, NO_LANDING_SAMPLE, outage)
        fill_group!(channels[2], groups[2], sample, NO_LANDING_SAMPLE, outage)
        pvt, status = update_navigation!(rx.vt, groups, 0.1s)
        copy_back!(channels[1], groups[1])
        copy_back!(channels[2], groups[2])
        (cycle <= 3 || cycle % 10 == 0) && report(io, cycle, pvt, status)
    end
    return 0
end
