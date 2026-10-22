"""
    TrackingLoops

The device-independent core of a GNSS tracking loop, shared by the software
receiver (Tracking.jl) and the allocation-free *loop process* of a hardware
correlator (HardwareLoopCore.jl):

  - correlator record types, discriminators and the post-correlation filter;
  - the navigation-bit buffer with per-signal bit-edge and secondary-code sync;
  - the C/N₀ estimators and the noise-density window they read;
  - loop-filter bandwidth rules and the Doppler estimators (conventional
    FLL-assisted PLL/DLL, delay-aware NCO-referenced loop) behind one per-record
    [`step_loop`](@ref);
  - [`apply_record`](@ref), the per-record fold of a signal component (prompt
    filter, C/N₀ estimator, bit buffer);
  - the [`NCOTimeline`](@ref): what a hardware NCO ran and will run;
  - vector tracking: [`VectorPLLAndDLL`](@ref) decodes every satellite's bits (a
    pilot + data pair's from the data component, ranging on the pilot), solves the
    PVT and closes all loops at once with a navigation filter.

Nothing here reads a raw sample or knows a device or a segment.
"""
module TrackingLoops

using BitIntegers
using Dates: year, now, UTC
using Dictionaries: Dictionary, set!
using GNSSDecoder:
    GNSSDecoder,
    GNSSDecoderState,
    decode!,
    reset_decoder_state!,
    get_broadcast_ephemeris,
    is_decoding_completed_for_positioning,
    is_sat_healthy
using Geodesy: ECEF, ENUfromECEF, wgs84
using KalmanFilters:
    KFTUIntermediate, UKFMUIntermediate, measurement_update!, time_update!
using LinearAlgebra: LinearAlgebra, dot, mul!, Symmetric
using PositionVelocityTime:
    PositionVelocityTime,
    BiasColumns,
    BiasLayout,
    BroadcastInterSystemBias,
    InterFrequencyBias,
    IonosphericModel,
    PVTSolution,
    PVTWorkspace,
    SPEED_OF_LIGHT,
    SatelliteInfo,
    SatelliteMeasurement,
    SatelliteState,
    SignalGroup,
    SupportedTimeSystem,
    ifb_layout,
    calc_dop!,
    calc_fit_covariance!,
    calc_leap_second_correction,
    calc_pseudorange,
    calc_design_matrix!,
    calc_course_over_ground,
    calc_pvt!,
    calc_pseudorange_rate,
    calc_satellite_clock_drift,
    calc_satellite_position_and_velocity,
    predict_pseudoranges!,
    collect_measurements!,
    decide_bias_layout!,
    dop_bias_columns,
    empty_keeping_capacity!,
    fit_bias_columns,
    calc_elevation_and_azimuth,
    num_parameters,
    predict_atmospheric_delays!,
    select_reference_system
using DocStringExtensions
using GNSSSignals
using SpecialFunctions: erfinv
using StaticArrays
using TrackingLoopFilters
using Unitful: upreferred, uconvert, ustrip, dimension, NoUnits, Hz, dBHz, ms, s, m
using Random: AbstractRNG, Xoshiro
import Base.zero, Base.length

# For the 1800-chip overlay-code searches of GPS L1C-P and BeiDou B1C-P.
BitIntegers.@define_integers 1800

export NumAnts,
    EarlyPromptLateCorrelator,
    VeryEarlyPromptLateCorrelator,
    CorrelatorOutput,
    get_early,
    get_prompt,
    get_late,
    get_accumulators,
    get_num_ants,
    get_default_correlator,
    DefaultPostCorrFilter,
    has_bit_or_secondary_code_been_found,
    is_dataless_with_secondary_code,
    get_soft_bits,
    MomentsCN0Estimator,
    NWPRCN0Estimator,
    NoCN0Estimator,
    NoiseRefCN0Estimator,
    estimate_cn0,
    CorrelatorNoiseEstimator,
    noise_observation_from_correlator,
    append_noise_observation!,
    default_wide_carrier_loop_filter_bandwidth,
    effective_carrier_loop_filter_bandwidth,
    FLL_ASSISTED_PLL,
    WIDE_PLL,
    NARROW_PLL,
    VECTOR_LOOP,
    carrier_loop_stage,
    AbstractDopplerEstimator,
    ConventionalPLLAndDLL,
    ConventionalAssistedPLLAndDLL,
    NCOReferencedPLLAndDLL,
    VectorPLLAndDLL,
    NavigationFilterConfig,
    VT_NOT_RELEASED,
    VT_INELIGIBLE,
    VT_BELOW_HORIZON,
    VT_FALLBACK,
    navigation_solution,
    navigation_status,
    navigation_cycle,
    navigation_epoch,
    satellite_report,
    release_reason,
    vector_loop_members,
    init_estimator_state,
    reset_estimator_state,
    LoopRecord,
    step_loop,
    SignalLoopState,
    apply_record,
    restart_bit_clock,
    reset_signal_state,
    NCOTimeline,
    FixedNCOWord,
    NO_LANDING_SAMPLE,
    reset_timeline!,
    schedule_word!,
    promote_words!,
    reschedule_word!,
    nco_word_at,
    mean_nco_word

const Maybe{T} = Union{T,Nothing}

"""
    NumAnts(n)
    NumAnts(correlator::AbstractCorrelator)
    NumAnts(estimator::AbstractNoiseEstimator)
    NumAnts(density_type::Type)

Type parameter wrapper for the number of antennas; create with `NumAnts(n)`. The other
forms read it off what carries one, as a constant of its type: a correlator's antenna
count, a noise estimator's (through [`noise_density_type`](@ref)), or a noise density
type's (`NumAnts(1)` for a scalar density, `NumAnts(M)` for an `M×M` covariance). A
host checks a noise estimator against its correlators with them, and specialises a
kernel on the count without `NumAnts(get_num_ants(c))`, which allocates per call.
"""
struct NumAnts{x} end

NumAnts(x) = NumAnts{x}()

"""
$(SIGNATURES)

Type parameter wrapper for the number of correlator accumulators; create with
`NumAccumulators(n)`.
"""
struct NumAccumulators{x} end

NumAccumulators(x) = NumAccumulators{x}()

"""
    update(x, prompt)

Advance a per-record state (C/N₀ estimator, post-correlation filter) by one prompt
and return the new state.
"""
function update end

"""
$(SIGNATURES)

Abstract supertype for Doppler estimators. Subtypes carry configuration; the
per-satellite state lives with the satellite. Hosts drive every estimator through
one interface:

  - [`init_estimator_state`](@ref)`(estimator, driver_signal, carrier_doppler,
    code_doppler)` builds a satellite's state, which records the driver: the
    signal whose records close the loops;
  - [`step_loop`](@ref)`(estimator, state, record, words, landing_sample)`
    folds one record into it and returns `(state, carrier_doppler,
    code_doppler)`. The host calls it on every record of every signal of the
    satellite (each built from that signal's [`SignalLoopState`](@ref)); the
    estimator picks out the ones it uses and, for the rest, returns the state
    unchanged and the command in force;
  - [`reset_estimator_state`](@ref)`(estimator, state, carrier_doppler,
    code_doppler)` re-seeds it from converged Dopplers, keeping what the
    estimator chooses to keep.

Implemented by [`ConventionalPLLAndDLL`](@ref), [`NCOReferencedPLLAndDLL`](@ref)
and [`VectorPLLAndDLL`](@ref), whose navigation engine also runs inside `step_loop`.
"""
abstract type AbstractDopplerEstimator end

include("bit_buffer.jl")
include("cn0_estimators/cn0_estimator.jl")
include("cn0_estimators/moments.jl")
include("cn0_estimators/no_cn0.jl")
include("cn0_estimators/nwpr.jl")
include("cn0_estimators/noise_ref.jl")
include("correlators/correlator.jl")
include("correlators/early_prompt_late.jl")
include("correlators/very_early_prompt_late.jl")
include("noise_estimators/noise_estimator.jl")
include("noise_estimators/window.jl")
include("discriminators.jl")
include("post_corr_filter.jl")
include("gps/l1ca.jl")
include("gps/l1c_d.jl")
include("gps/l1c_p.jl")
include("gps/l2c.jl")
include("gps/l5.jl")
include("galileo/e1b.jl")
include("galileo/e1c.jl")
include("galileo/e5a.jl")
include("galileo/e5a_qp.jl")
include("galileo/e5b.jl")
include("galileo/e6.jl")
include("beidou/b1i.jl")
include("beidou/b3i.jl")
include("beidou/b2a.jl")
include("beidou/b2b.jl")
include("beidou/b1c.jl")
include("sample_parameters.jl")
include("loop_filters.jl")
include("carrier_loop_staging.jl")
include("record.jl")
include("nco_timeline.jl")
include("estimators.jl")
include("vector/residual_pll.jl")
include("vector/estimator.jl")
include("vector/model.jl")
include("vector/tracking.jl")
include("vector/engine.jl")

end # module
