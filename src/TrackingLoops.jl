"""
    TrackingLoops

The device-independent core of a GNSS tracking loop, shared by the software
receiver (Tracking.jl) and the allocation-free *loop process* of a hardware
correlator (HardwareLoopCore.jl):

  - predetection integration types, discriminators and the beamformer;
  - the navigation-bit synchronizer with per-signal bit-edge and secondary-code sync;
  - the C/N₀ estimators and the noise-density window they read;
  - loop-filter bandwidth rules and the tracking loops (conventional
    FLL-assisted PLL/DLL, delay-aware NCO-referenced loop) behind one per-integration
    [`step_loop!`](@ref);
  - [`process_correlator_bank_output!`](@ref), the post-correlation processing of a signal component (prompt
    filter, C/N₀ estimator, bit synchronizer);
  - the [`NCOTimeline`](@ref): what a hardware NCO ran and will run;
  - vector tracking: [`VectorTrackingLoop`](@ref) decodes every satellite's bits (a
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
    has_positioning_data,
    is_sat_healthy
using Geodesy: ECEF, ENUfromECEF, wgs84
using KalmanFilters:
    KFTUIntermediate, UKFMUIntermediate, measurement_update!, time_update!
using LinearAlgebra: LinearAlgebra, dot, mul!, Symmetric
using PositionVelocityTime:
    PositionVelocityTime,
    BiasColumns,
    BiasLayout,
    applied_broadcast_inter_system_biases!,
    InterFrequencyBias,
    IonosphericModel,
    PVTSolution,
    PVTWorkspace,
    SPEED_OF_LIGHT,
    ObservationResiduals,
    PreparedObservation,
    SignalObservation,
    SupportedTimeSystem,
    ifb_layout,
    calc_dop!,
    calc_fit_covariance!,
    calc_pseudorange,
    calc_design_matrix!,
    calc_course_over_ground,
    calc_pvt!,
    calc_pseudorange_rate,
    calc_satellite_clock_drift,
    calc_satellite_position_and_velocity,
    predict_pseudoranges!,
    prepare_observations!,
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

export NumAntennas,
    EarlyPromptLateCorrelatorBank,
    VeryEarlyPromptLateCorrelatorBank,
    CorrelatorBankOutput,
    get_early,
    get_prompt,
    get_late,
    get_correlator_outputs,
    get_num_antennas,
    get_default_correlator_bank,
    DefaultBeamformer,
    is_bit_or_secondary_code_synced,
    is_dataless_with_secondary_code,
    get_soft_bits,
    MomentsCN0Estimator,
    NWPRCN0Estimator,
    NoCN0Estimator,
    NoiseReferencedCN0Estimator,
    estimate_cn0,
    CorrelatorNoiseEstimator,
    calc_noise_observation_from_correlator_bank,
    append_noise_observation!,
    default_wide_carrier_loop_filter_bandwidth,
    effective_carrier_loop_filter_bandwidth,
    FLL_ASSISTED_PLL,
    WIDE_PLL,
    NARROW_PLL,
    VECTOR_LOOP,
    get_carrier_loop_stage,
    AbstractTrackingLoop,
    ScalarTrackingLoop,
    ConventionalAssistedPLLAndDLL,
    DelayCompensatedTrackingLoop,
    VectorTrackingLoop,
    NavigationFilterConfig,
    VT_NOT_RELEASED,
    VT_INELIGIBLE,
    VT_BELOW_HORIZON,
    VT_FALLBACK,
    get_navigation_solution,
    get_navigation_status,
    get_navigation_cycle_index,
    get_navigation_elapsed_time,
    get_satellite_report,
    get_release_reason,
    get_vector_tracking_members,
    init_loop_state,
    reseed_loop_state,
    PredetectionIntegration,
    step_loop!,
    PostCorrelationState,
    process_correlator_bank_output!,
    reset_cn0_estimator!,
    reset_sync!,
    reset_post_correlation!,
    NCOTimeline,
    ConstantNCOCommand,
    NCOCommand,
    confirm_command!,
    cancel_command!,
    NO_LANDING_SAMPLE,
    reset_timeline!,
    schedule_command!,
    advance_timeline!,
    get_nco_command,
    calc_mean_nco_command

const Maybe{T} = Union{T,Nothing}

"""
    NumAntennas(n)
    NumAntennas(correlator_bank::AbstractCorrelatorBank)
    NumAntennas(estimator::AbstractNoiseDensityEstimator)
    NumAntennas(density_type::Type)

Type parameter wrapper for the number of antennas; create with `NumAntennas(n)`. The other
forms read it off what carries one, as a constant of its type: a correlator bank's antenna
count, a noise estimator's (through [`noise_density_type`](@ref)), or a noise density
type's (`NumAntennas(1)` for a scalar density, `NumAntennas(M)` for an `M×M` covariance). A
host checks a noise estimator against its correlator banks with them, and specialises a
kernel on the count without `NumAntennas(get_num_antennas(c))`, which allocates per call.
"""
struct NumAntennas{x} end

NumAntennas(x) = NumAntennas{x}()

"""
$(SIGNATURES)

Type parameter wrapper for the number of correlators; create with
`NumCorrelators(n)`.
"""
struct NumCorrelators{x} end

NumCorrelators(x) = NumCorrelators{x}()

"""
    update!(x, prompt)

Advance a per-integration state (C/N₀ estimator, beamformer) by one prompt
and return the updated state. A state may write its buffers in place, so the
returned state replaces the argument: discarding it loses the update.
"""
function update! end

"""
$(SIGNATURES)

Abstract supertype for tracking loops. Scalar loops carry configuration; a vector
loop also owns the shared navigation filter and decoders. Per-satellite loop state
lives with the satellite. Hosts drive every tracking loop through one interface:

  - [`init_loop_state`](@ref)`(tracking_loop, driver_signal, carrier_doppler,
    code_doppler)` builds a satellite's state, which records the driver: the
    signal whose integrations close the loops;
  - [`step_loop!`](@ref)`(tracking_loop, state, integration, nco_history, application_sample_index)`
    advances it by one integration and returns `(state, carrier_doppler,
    code_doppler)`. The host calls it on every integration of every signal of the
    satellite (each built from that signal's [`PostCorrelationState`](@ref)); the
    tracking loop picks out the ones it uses and, for the rest, returns the state
    unchanged and the command in force;
  - [`reseed_loop_state`](@ref)`(tracking_loop, state, carrier_doppler,
    code_doppler)` re-seeds it from converged Dopplers, keeping what the
    tracking loop chooses to keep.

Implemented by [`ScalarTrackingLoop`](@ref), [`DelayCompensatedTrackingLoop`](@ref)
and [`VectorTrackingLoop`](@ref), whose navigation engine also runs inside `step_loop!`.
"""
abstract type AbstractTrackingLoop end

include("bit_synchronizer.jl")
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
include("beamformer.jl")
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
include("post_correlation.jl")
include("nco_timeline.jl")
include("tracking_loops.jl")
include("vector/residual_pll.jl")
include("vector/loop_state.jl")
include("vector/model.jl")
include("vector/tracking.jl")
include("vector/engine.jl")

end # module
