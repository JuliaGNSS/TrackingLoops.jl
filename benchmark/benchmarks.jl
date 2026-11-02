# The loop process's navigation budget, on the simulated satellites of the tests
# (`test/vector_simulation.jl`): one navigation cycle, for GPS L1 C/A alone and with L2C
# (one inter-frequency bias), and the per-integration step of every member between two cycles.
# Run with AirspeedVelocity's `benchpkg`, or directly:
#
#     julia --project=benchmark -e 'include("benchmark/benchmarks.jl"); run(SUITE)'
using BenchmarkTools, TrackingLoops, GNSSSignals, StaticArrays, Unitful
using TrackingLoops: default_cn0_estimator
using Unitful: Hz, dBHz, s, ms

# The host's C/N₀ estimator the simulated integrations carry, as in the tests: it reads the
# C/N₀ the integration was built with.
struct ReportedCN0 <: TrackingLoops.AbstractCN0Estimator
    cn0_dbhz::Float64
end
TrackingLoops.estimate_cn0(estimator::ReportedCN0, coherent_integration_time) = estimator.cn0_dbhz * dBHz
with_cn0(integration::CoherentIntegration, cn0_dbhz) = CoherentIntegration(
    (f === :cn0_estimator ? ReportedCN0(cn0_dbhz) : getfield(integration, f) for f in fieldnames(CoherentIntegration))...,
)

include(joinpath(@__DIR__, "..", "test", "vector_simulation.jl"))

const SUITE = BenchmarkGroup()

# Every member's step, one signal's vector at a time; recursive, so that it unrolls over
# the signals' tuple.
step_members(tracking_loop, ::Tuple{}) = nothing
function step_members(tracking_loop, steps::Tuple)
    for (state, integration, nco_history) in first(steps)
        TrackingLoops._step_satellite(tracking_loop, state, integration, nco_history, nothing)
    end
    step_members(tracking_loop, Base.tail(steps))
end

for (name, signals) in (("GPS L1 C/A", (GPSL1CA(),)), ("GPS L1 C/A and L2C", (GPSL1CA(), GPSL2CM())))
    # A running filter, and every slot filled for the next epoch as `run_cycle!` fills
    # them. Each sample runs the cycle on a copy, so every one starts from the same state.
    rx = SimReceiver(; signals)
    _, sample, _ = run_simulation!(rx, 5)
    rx.vt.vector_tracking_active || error("the filter did not start on $name")
    epoch = sample + rx.cycle_ms * SAMPLES_PER_MS
    rx.vt.pending_epoch_index = round(Int, epoch / (rx.nominal_ms * SAMPLES_PER_MS))
    for (channels, group) in zip(rx.channels, rx.groups), (i, sat) in enumerate(channels)
        fill_slot!(group.slots[i], sat, epoch, nothing, rx.vt)
    end
    now = epoch / ustrip(Hz, SIM_FS)
    num_satellites = sum(length, rx.channels)
    SUITE["navigation cycle"]["$num_satellites satellites, $name"] =
        @benchmarkable TrackingLoops._navigation_cycle!(nav, $now) setup = (nav = deepcopy($(rx.vt))) evals = 1

    # One integration of every member, stepped from its state at the cycle's end as the
    # satellite's own loop steps it in the vector loop; one vector per signal, so that
    # each is stepped with its types known.
    steps = map(rx.channels) do channels
        map(channels) do sat
            num_samples = integration_ms(sat.signal) * SAMPLES_PER_MS
            integration_end_sample_index = sample + num_samples
            nco_history = sim_nco_history(sat, nothing)
            correlator_bank, = simulate_correlator(sat, rx.truth, sim_time(rx, integration_end_sample_index), num_samples,
                integration_end_sample_index, nco_history)
            integration = with_cn0(CoherentIntegration(sat.signal, correlator_bank, sat.previous_prompt, num_samples,
                integration_end_sample_index, integration_end_sample_index, 1, SIM_FS), 45.0)
            (sat.state, integration, nco_history)
        end
    end
    SUITE["member step"]["$num_satellites members, $name"] =
        @benchmarkable step_members($(rx.tracking_loop), $steps)
end
