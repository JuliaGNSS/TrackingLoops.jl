# The loop process's navigation budget, on the simulated satellites of the tests
# (`test/vector_simulation.jl`): one navigation cycle, for GPS L1 C/A alone and with L2C
# (one inter-frequency bias), and the per-record step of every member between two cycles.
# Run with AirspeedVelocity's `benchpkg`, or directly:
#
#     julia --project=benchmark -e 'include("benchmark/benchmarks.jl"); run(SUITE)'
using BenchmarkTools, TrackingLoops, GNSSSignals, StaticArrays, Unitful
using TrackingLoops: default_cn0_estimator
using Unitful: Hz, dBHz, s, ms

# The host's C/N₀ estimator the simulated records carry, as in the tests: it reads the
# C/N₀ the record was built with.
struct ReportedCN0 <: TrackingLoops.AbstractCN0Estimator
    cn0_dbhz::Float64
end
TrackingLoops.estimate_cn0(estimator::ReportedCN0, integration_time) = estimator.cn0_dbhz * dBHz
with_cn0(record::LoopRecord, cn0_dbhz) = LoopRecord(
    (f === :cn0_estimator ? ReportedCN0(cn0_dbhz) : getfield(record, f) for f in fieldnames(LoopRecord))...,
)

include(joinpath(@__DIR__, "..", "test", "vector_simulation.jl"))

const SUITE = BenchmarkGroup()

# Every member's step, one signal's vector at a time; recursive, so that it unrolls over
# the signals' tuple.
step_members(estimator, ::Tuple{}) = nothing
function step_members(estimator, steps::Tuple)
    for (state, record, words) in first(steps)
        TrackingLoops._step_satellite(estimator, state, record, words, NO_LANDING_SAMPLE)
    end
    step_members(estimator, Base.tail(steps))
end

for (name, signals) in (("GPS L1 C/A", (GPSL1CA(),)), ("GPS L1 C/A and L2C", (GPSL1CA(), GPSL2CM())))
    # A running filter, and every slot filled for the next epoch as `run_cycle!` fills
    # them. Each sample runs the cycle on a copy, so every one starts from the same state.
    rx = SimReceiver(; signals)
    _, sample, _ = run_simulation!(rx, 5)
    rx.vt.running || error("the filter did not start on $name")
    epoch = sample + rx.cycle_ms * SAMPLES_PER_MS
    rx.vt.pending_epoch = round(Int, epoch / (rx.nominal_ms * SAMPLES_PER_MS))
    for (channels, group) in zip(rx.channels, rx.groups), (i, sat) in enumerate(channels)
        fill_slot!(group.slots[i], sat, epoch, NO_LANDING_SAMPLE, rx.vt)
    end
    now = epoch / ustrip(Hz, SIM_FS)
    num_satellites = sum(length, rx.channels)
    SUITE["navigation cycle"]["$num_satellites satellites, $name"] =
        @benchmarkable TrackingLoops._navigation_cycle!(nav, $now) setup = (nav = deepcopy($(rx.vt))) evals = 1

    # One record of every member, stepped from its state at the cycle's end as the
    # satellite's own loop steps it in the vector loop; one vector per signal, so that
    # each is stepped with its types known.
    steps = map(rx.channels) do channels
        map(channels) do sat
            num_samples = record_ms(sat.signal) * SAMPLES_PER_MS
            record_end = sample + num_samples
            words = sim_words(sat, NO_LANDING_SAMPLE)
            correlator, = simulate_correlator(sat, rx.truth, sim_time(rx, record_end), num_samples,
                record_end, words)
            record = with_cn0(LoopRecord(sat.signal, correlator, sat.previous_prompt, num_samples,
                record_end, record_end, 1, SIM_FS), 45.0)
            (sat.state, record, words)
        end
    end
    SUITE["member step"]["$num_satellites members, $name"] =
        @benchmarkable step_members($(rx.estimator), $steps)
end
