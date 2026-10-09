# Entry point of the `juliac --trim=safe` check (see `check.jl`): vector tracking from
# records alone, on the synthetic GPS L1 C/A satellites of `../vector_simulation.jl`
# that broadcast real navigation bits — bit sync, decoding, the scalar fix, the seed
# and a steady state through a partial outage — printed so the trimmed executable's
# output can be compared against a regular Julia session. It runs twice: on the C/A
# signal alone, and ranging on a pilot while decoding the C/A data component. It is
# the loop process's workload: `apply_record` and `step_loop` on every record of every
# signal, and nothing else.
using TrackingLoops, GNSSSignals, StaticArrays, Unitful, LinearAlgebra
using Unitful: Hz, s, ms

include(joinpath(@__DIR__, "..", "vector_simulation.jl"))

# Every channel's records that end within the millisecond ending at `sample`, of
# every signal.
function step_channels!(rx, sample, outage::Bool)
    for (i, sat) in enumerate(rx.sats)
        sat.in_view = !(outage && i <= 2)
        pipeline_records!(rx, i, sample)
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

# 33 s: the first fix comes 26.2 s in; a two-second outage of two satellites late on.
function run_receiver(io, rx)
    nav = rx.estimator.navigation
    last_cycle = nav.cycle_id
    sample = 0
    for _ = 1:33_000
        sample += SAMPLES_PER_MS
        t = sample / 4e6
        step_channels!(rx, sample, 29.0 <= t <= 31.0)
        if nav.cycle_id != last_cycle
            last_cycle = nav.cycle_id
            status = nav.status
            (status.enabled || last_cycle % 20 == 0) && report(io, last_cycle, nav.pvt, status)
        end
    end
    nothing
end

function (@main)(args::Vector{String})::Cint
    io = Core.stdout
    run_receiver(io, PipelineReceiver())
    println(io, "pilot + data:")
    run_receiver(io, PipelineReceiver(; signals = GPSL1C_P() => GPSL1CA()))
    return 0
end
