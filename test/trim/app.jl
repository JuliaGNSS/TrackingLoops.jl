# Entry point of the `juliac --trim=safe` check (see `check.jl`): vector tracking from
# records alone, on the synthetic GPS L1 C/A satellites of `../vector_simulation.jl`
# that broadcast real navigation bits — bit sync, decoding, the scalar fix, the seed
# and a steady state through a partial outage — printed so the trimmed executable's
# output can be compared against a regular Julia session. It is the loop process's
# workload: `apply_record` and `step_loop` on every record of every channel, and
# nothing else.
using TrackingLoops, GNSSSignals, StaticArrays, Unitful, LinearAlgebra
using TrackingLoops: default_cn0_estimator
using Unitful: Hz, s, ms

include(joinpath(@__DIR__, "..", "vector_simulation.jl"))

# Every channel's records that end within the millisecond ending at `sample`.
function step_channels!(rx, sample, outage::Bool)
    for (i, sat) in enumerate(rx.sats)
        sat.in_view = !(outage && i <= 2)
        step_satellite!(rx, i, sample)
    end
    nothing
end

# One `print` per value: a long `print(io, xs...)` is not specialised on its
# arguments' types, which leaves the call dynamic.
function report(io, cycle, pvt, status)
    print(io, "cycle ", cycle, ":")
    for x in (pvt.position.x, pvt.position.y, pvt.position.z, pvt.velocity.x,
        pvt.velocity.y, pvt.velocity.z,
        status.position_std.val, status.time_with_insufficient_meas.val)
        print(io, " ", x)
    end
    print(io, " ", length(pvt.observations))
    print(io, " ", status.num_members)
    print(io, " ", status.running)
    print(io, " ", status.seeded)
    t = pvt.time
    t === nothing || print(io, " ", t.second)
    println(io)
end

# The noise reference as a hardware host fills it: one probe per arm from the estimator's
# own `Xoshiro`, each dump's correlator outputs appended over its own sample count.
function report_noise(io)
    signal = GPSL1CA()
    estimator = CorrelatorNoiseEstimator()
    for k = 1:20
        probe = TrackingLoops.draw_noise_probe(estimator, signal, 0.0Hz)
        correlator_outputs = SVector(
            complex(probe.code_phase / 1023, 1.0),
            complex(1.0, ustrip(Hz, probe.carrier_frequency) / 5000),
            complex(0.5, 0.5),
        )
        TrackingLoops.append_noise_correlator_outputs!(estimator, probe, correlator_outputs, 3990 + k, 4e6Hz)
        print(io, "noise ", k)
        print(io, " ", probe.prn)
        print(io, " ", probe.code_phase)
        print(io, " ", ustrip(Hz, probe.carrier_frequency))
        println(io)
    end
    print(io, "noise density ", ustrip(Hz^-1, TrackingLoops.get_noise_density(estimator)))
    println(io)
end

function (@main)(args::Vector{String})::Cint
    io = Core.stdout
    rx = PipelineReceiver(Val(false))
    nav = rx.estimator.navigation
    last_cycle = nav.cycle_id
    sample = 0
    # 33 s: the first fix comes 26.2 s in; a two-second outage of two satellites late on.
    for _ = 1:33_000
        sample += SAMPLES_PER_MS
        t = sample / 4e6
        step_channels!(rx, sample, 29.0 <= t <= 31.0)
        if nav.cycle_id != last_cycle
            last_cycle = nav.cycle_id
            status = nav.status
            (status.seeded || last_cycle % 20 == 0) && report(io, last_cycle, nav.pvt, status)
        end
    end
    report_noise(io)
    return 0
end
