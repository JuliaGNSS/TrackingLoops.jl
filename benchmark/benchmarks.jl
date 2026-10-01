# The loop process's navigation budget: one `update_navigation!` cycle of 18 satellites
# on two bands (GPS L1 C/A and L5, one inter-frequency bias), from PositionVelocityTime's
# fixture satellites, against one cycle's worth of per-record `step_loop` for the same
# channels. Run with AirspeedVelocity's `benchpkg`, or directly:
#
#     julia --project=benchmark -e 'include("benchmark/benchmarks.jl"); run(SUITE)'
using BenchmarkTools, TrackingLoops, GNSSSignals, StaticArrays
using Unitful: Hz, s
using PositionVelocityTime:
    _precompile_states, _precompile_cnav, _PRECOMPILE_GPS_L1CA_STATES

const SUITE = BenchmarkGroup()

# The fixture satellites of one signal as filled `VTSat`s: in lock, in the vector
# loop, with a cycle of accumulated discriminators.
function fixture_group(signal, make_data)
    states = _precompile_states(signal, _PRECOMPILE_GPS_L1CA_STATES, make_data, GPSL1CA())
    estimator = VectorPLLAndDLL()
    sats = map(states) do state
        estimator_state = init_estimator_state(estimator, signal, state.carrier_doppler,
            state.carrier_doppler * get_code_center_frequency_ratio(signal))
        VTSat(state.decoder, estimator_state; active = true, code_phase = state.code_phase,
            carrier_doppler = state.carrier_doppler, code_doppler = 0.0Hz,
            code_phase_at_landing = state.code_phase,
            carrier_doppler_at_landing = state.carrier_doppler, cn0_dbhz = 45.0,
            in_lock = true, pvt_ready = true)
    end
    VTSignalGroup(signal, sats)
end

# One cycle's worth of accumulated discriminators on every member.
function accumulate!(groups)
    for group in groups, sat in group.sats
        sat.estimator_state = SatVectorPLLAndDLL(sat.estimator_state;
            code_discr_acc = (100, 0.01), carrier_discr_acc = (100, 5.0Hz))
    end
    groups
end

groups = (fixture_group(GPSL1CA(), identity), fixture_group(GPSL5I(), _precompile_cnav))
vt = VectorTrackingState(VectorTracking(), groups; approximate_year = 2021)
update_navigation!(vt, groups, 0.1s) # the scalar fix seeds the filter
@assert vt.running
accumulate!(groups)
update_navigation!(vt, groups, 0.1s)

SUITE["update_navigation!"]["18 satellites, 2 bands"] =
    @benchmarkable update_navigation!($vt, $groups, 0.1s) setup = accumulate!($groups) evals = 1

# The per-record loop the cycle sits on: 100 records of 1 ms for each of the 18 channels.
signal = GPSL1CA()
estimator = VectorPLLAndDLL()
state = enable_vector_tracking(init_estimator_state(estimator, signal, 100.0Hz, 0.1Hz))
correlator = EarlyPromptLateCorrelator(SVector{3,ComplexF64}(0.5, 1.0, 0.5), 0.5)
function records!(state, estimator, signal, correlator, n)
    for k = 1:n
        output = CorrelatorOutput(correlator, 4000, 4000k)
        record = LoopRecord(signal, correlator, cis(0.01), output, 1, 4e6Hz)
        state, = step_loop(estimator, state, record, FixedNCOWord(100.0, 0.1), NO_LANDING_SAMPLE)
    end
    state
end
SUITE["step_loop"]["18 channels × 100 records"] =
    @benchmarkable records!($state, $estimator, $signal, $correlator, 1800)
