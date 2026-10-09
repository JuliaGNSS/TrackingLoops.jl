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
  - [`apply_record`](@ref), the per-record fold of a signal component;
  - the [`NCOTimeline`](@ref): what a hardware NCO ran and will run;
  - vector tracking: [`VectorPLLAndDLL`](@ref) decodes every satellite's bits,
    solves the PVT and closes all loops at once with a navigation filter.

Nothing here reads a raw sample or knows a device or a segment.
"""
module TrackingLoops

using BitIntegers
using Dates: year, now, UTC
using Dictionaries: Dictionary, set!
using GNSSDecoder:
    GNSSDecoder,
    GNSSDecoderState,
    SECONDS_PER_WEEK,
    decode!,
    reset_decoder_state!,
    is_decoding_completed_for_positioning,
    is_sat_healthy
using Geodesy: ECEF, ENUfromECEF, wgs84
using KalmanFilters: KFTUIntermediate, UKFMUIntermediate, measurement_update!, time_update!
using LinearAlgebra: LinearAlgebra, dot, mul!, cholesky!, issuccess, Symmetric
using PositionVelocityTime:
    PositionVelocityTime,
    BandLayoutScratch,
    BiasColumns,
    BroadcastTimeOffset,
    CANDIDATE_HUB_SYSTEMS,
    InterFrequencyBias,
    IonosphericModel,
    PVTSolution,
    PVTWorkspace,
    SPEED_OF_LIGHT,
    SatInfo,
    SatelliteMeasurement,
    SatelliteState,
    SignalGroup,
    SupportedTimeSystem,
    TAITime,
    band_ifb_layout,
    band_ifb_layout!,
    calc_DOP!,
    calc_H!,
    calc_course_over_ground,
    calc_line_of_sight,
    calc_pvt!,
    calc_satellite_clock_drift,
    calc_satellite_position_and_velocity,
    calc_steering_offset,
    calc_ρ_hat!,
    collect_measurement_rows!,
    day_of_year,
    empty_keeping_capacity!,
    fold_week_crossover,
    get_sat_enu,
    get_sat_position,
    get_sat_velocity,
    predict_atmospheric_delays!,
    time_scale_offset_to_gpst
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
    NumAccumulators,
    AbstractCorrelator,
    AbstractEarlyPromptLateCorrelator,
    EarlyPromptLateCorrelator,
    VeryEarlyPromptLateCorrelator,
    CorrelatorOutput,
    get_early,
    get_prompt,
    get_late,
    get_very_early,
    get_very_late,
    get_accumulators,
    get_num_accumulators,
    get_num_ants,
    get_prompt_index,
    get_early_late_sample_spacing,
    get_correlator_sample_shifts,
    update_accumulator,
    normalize,
    get_default_correlator,
    pll_disc,
    fll_disc,
    dll_disc,
    AbstractPostCorrFilter,
    DefaultPostCorrFilter,
    get_weights,
    update,
    BitBuffer,
    PhaseAccumulators,
    SyncResult,
    has_bit_or_secondary_code_been_found,
    get_soft_bits,
    get_code_block_buffer_type,
    detect_bit_or_secondary_code_sync,
    uses_soft_bit_edge_detection,
    uses_soft_secondary_code_detection,
    get_bit_edge_detection_confidence,
    get_bit_edge_or_secondary_code_tolerance,
    AbstractCN0Estimator,
    CN0UpdateContext,
    MomentsCN0Estimator,
    NWPRCN0Estimator,
    NoCN0Estimator,
    NoiseRefCN0Estimator,
    requires_noise_density,
    estimate_cn0,
    default_cn0_estimator,
    AbstractNoiseEstimator,
    CorrelatorNoiseEstimator,
    NoiseEstimators,
    NoiseObservation,
    NoiseDensity,
    noise_observation,
    noise_observation_from_correlator,
    noise_observation_from_samples,
    append_noise_observation!,
    update_noise!,
    get_noise_density,
    noise_density_type,
    noise_window_looks,
    max_num_code_blocks_to_integrate,
    default_num_code_blocks_to_integrate,
    calc_num_code_blocks_to_integrate,
    calc_num_code_blocks_for_bit_buffer,
    default_wide_carrier_loop_filter_bandwidth,
    default_code_loop_filter_bandwidth,
    effective_code_loop_filter_bandwidth,
    default_narrow_carrier_loop_filter_bandwidth,
    default_fll_assist_loop_filter_bandwidth,
    CarrierLoopStage,
    FLL_ASSISTED_PLL,
    WIDE_PLL,
    NARROW_PLL,
    phase_lock_indicator_threshold,
    phase_lock_indicator,
    carrier_loop_stage,
    aid_dopplers,
    calculate_carrier_frequency_update,
    calculate_code_frequency_update,
    AbstractDopplerEstimator,
    ConventionalPLLAndDLL,
    ConventionalAssistedPLLAndDLL,
    SatConventionalPLLAndDLL,
    NCOReferencedPLLAndDLL,
    SatNCOReferencedPLLAndDLL,
    VectorPLLAndDLL,
    with_signal_groups,
    VectorTrackingSettings,
    SatVectorPLLAndDLL,
    VectorTracking,
    VTStatus,
    VTReleaseReason,
    VT_NOT_RELEASED,
    VT_INELIGIBLE,
    VT_BELOW_HORIZON,
    VT_FALLBACK,
    navigation_solution,
    navigation_status,
    navigation_cycle,
    navigation_epoch,
    satellite_report,
    SatelliteReport,
    release_reason,
    member_sats,
    position_uncertainty,
    clock_uncertainty,
    init_estimator_state,
    reset_estimator_state,
    LoopRecord,
    step_loop,
    SignalLoopState,
    apply_record,
    restart_bit_clock,
    reset_signal_state,
    estimator_state_type,
    NCOTimeline,
    scheduled_words,
    FixedNCOWord,
    NO_LANDING_SAMPLE,
    reset_timeline!,
    schedule_word!,
    promote_words!,
    reschedule_word!,
    word_changes_within,
    nco_word_at,
    mean_nco_word,
    wrap_half_cycle,
    get_sync_polarity,
    SignalCombiningSums,
    takes_passenger_records,
    fold_passenger_record

const Maybe{T} = Union{T,Nothing}

"""
$(SIGNATURES)

Type parameter wrapper for the number of antennas; create with `NumAnts(n)`.
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
    code_doppler)` builds a satellite's state;
  - [`step_loop`](@ref)`(estimator, state, record, words, landing_sample)`
    folds one record into it and returns `(state, carrier_doppler,
    code_doppler)`;
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
include("record.jl")
include("nco_timeline.jl")
include("doppler_estimators/carrier_loop_staging.jl")
include("doppler_estimators/signal_combining.jl")
include("doppler_estimators/conventional.jl")
include("doppler_estimators/nco_referenced.jl")
include("doppler_estimators/interface.jl")
include("vector/estimator.jl")
include("vector/model.jl")
include("vector/state.jl")
include("vector/tracking.jl")
include("vector/engine.jl")
include("vector/passengers.jl")

end # module
