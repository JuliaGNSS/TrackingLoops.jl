"""
    TrackingLoops

The device-independent core of a GNSS tracking loop, shared by the software
receiver (Tracking.jl) and the allocation-free *loop process* of a hardware
correlator (HardwareLoopCore.jl):

  - the correlator banks, the discriminators, the post-correlation filter and the
    post-correlation processing of a coherent integration;
  - the navigation-bit synchronizer with per-signal bit-edge and secondary-code sync;
  - the C/N₀ estimators and the noise-density window they read;
  - loop bandwidth rules and the tracking loops (conventional
    FLL-assisted PLL/DLL, delay-compensated loop) behind one per-integration
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
using LinearAlgebra: LinearAlgebra, dot, Symmetric
using PositionVelocityTime:
    PositionVelocityTime,
    BiasColumns,
    BiasLayout,
    ClockConstraint,
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
using Unitful: upreferred, uconvert, ustrip, NoUnits, Hz, dBHz, ms, s, m
using Random: AbstractRNG, Xoshiro
import Base.zero, Base.length

# For the 1800-chip overlay-code searches of GPS L1C-P and BeiDou B1C-P.
BitIntegers.@define_integers 1800

export NumAnts,
    EarlyPromptLateCorrelatorBank,
    VeryEarlyPromptLateCorrelatorBank,
    CorrelatorBankOutput,
    get_early,
    get_prompt,
    get_late,
    get_correlator_outputs,
    get_num_ants,
    get_default_correlator_bank,
    DefaultPostCorrFilter,
    is_bit_or_secondary_code_synced,
    is_dataless_with_secondary_code,
    get_soft_bits,
    drain_soft_bits!,
    get_secondary_code_phase,
    get_num_code_periods_into_bit,
    get_last_num_code_periods,
    get_bit_polarity,
    MomentsCN0Estimator,
    NWPRCN0Estimator,
    NoCN0Estimator,
    NoiseReferencedCN0Estimator,
    estimate_cn0,
    CorrelatorNoiseDensityEstimator,
    append_noise_observation!,
    FLL_ASSISTED_PLL,
    WIDE_PLL,
    NARROW_PLL,
    VECTOR_LOOP,
    get_decoder,
    get_signal_pairs,
    get_scalar_loop,
    get_carrier_loop_stage,
    AbstractTrackingLoop,
    ScalarTrackingLoop,
    DelayCompensatedTrackingLoop,
    VectorTrackingLoop,
    NavFilterConfig,
    get_navigation_solution,
    get_navigation_status,
    get_navigation_update_index,
    get_navigation_elapsed_time,
    get_satellite_report,
    get_vector_tracking_members,
    init_loop_state,
    reinit_loop_state,
    step_loop!,
    PostCorrelationState,
    process_correlator_bank_output!,
    reset_sync!,
    reset_post_correlation!,
    NCOTimeline,
    ConstantNCOCommand,
    NCOCommand,
    confirm_command!,
    cancel_command!,
    reset_timeline!,
    schedule_command!,
    advance_timeline!,
    get_nco_command,
    calc_mean_nco_command

const Maybe{T} = Union{T,Nothing}

"""
    NumAnts(n)
    NumAnts(correlator_bank::AbstractCorrelatorBank)
    NumAnts(estimator::AbstractNoiseDensityEstimator)
    NumAnts(density_type::Type)

Type parameter wrapper for the number of antennas; create with `NumAnts(n)`. The other
forms read it off what carries one, as a constant of its type: a correlator bank's antenna
count, a noise estimator's (through [`noise_density_type`](@ref)), or a noise density
type's (`NumAnts(1)` for a scalar density, `NumAnts(M)` for an `M×M` covariance). A
host checks a noise estimator against its correlator banks with them, and specialises a
kernel on the count without `NumAnts(get_num_ants(c))`, which allocates per call.
"""
struct NumAnts{M} end

NumAnts(M) = NumAnts{M}()

"""
$(SIGNATURES)

Type parameter wrapper for the number of correlators; create with
`NumCorrelators(n)`.
"""
struct NumCorrelators{K} end

NumCorrelators(K) = NumCorrelators{K}()

"""
    update!(estimator, prompt)

Advance a C/N₀ estimator by one prompt and return the updated estimator. An estimator
may write its buffers in place, so the returned estimator replaces the argument:
discarding it loses the update.
"""
function update! end

"""
    update(post_corr_filter, prompt)

Return the post-correlation filter to use for the next integration, updated with the
latest prompt. A post-correlation filter is not changed in place: the returned filter
replaces the argument.
"""
function update end

"""
$(SIGNATURES)

Abstract supertype for tracking loops. Scalar loops carry configuration; a vector
loop also owns the shared navigation filter and decoders. Per-satellite loop state
lives with the satellite. Hosts drive every tracking loop through one interface:

  - [`init_loop_state`](@ref)`(tracking_loop, driver_signal, carrier_doppler,
    code_doppler)` builds a satellite's state, which records the driver: the
    signal whose integrations close the loops;
  - [`step_loop!`](@ref)`(tracking_loop, state, post_correlation_state, nco_history,
    application_sample_index)` advances it by one integration, the latest of a signal's
    [`PostCorrelationState`](@ref), and returns `(state, carrier_doppler, code_doppler)`.
    The host calls it on every integration of every signal of the satellite; the
    tracking loop picks out the ones it uses and, for the rest, returns the state
    unchanged and the command in force;
  - [`reinit_loop_state`](@ref)`(tracking_loop, state, carrier_doppler,
    code_doppler)` re-initialises it at converged Dopplers, keeping what the
    tracking loop chooses to keep.

A custom tracking loop implements `init_loop_state` and `step_loop!`. `init_loop_state`
must be pure: no side effects, and a state type that depends only on the types of the
tracking loop and the driver signal, as a host also calls it for a template state
([`loop_state_type`](@ref TrackingLoops.loop_state_type)) and to re-initialise. The other
methods are optional: `reinit_loop_state` (a host may re-initialise with `init_loop_state`
instead), [`get_decoder`](@ref) (fallback `nothing`, for a loop without decoders) and
[`calc_carrier_phase_correction`](@ref TrackingLoops.calc_carrier_phase_correction)
(fallback `0.0`, for a loop that steers its replica onto the carrier).
[`get_carrier_loop_stage`](@ref), [`is_phase_locked`](@ref) and
[`get_phase_lock_indicator`](@ref) have no fallback: a custom loop implements them for a
host that reads them.

Implemented by [`ScalarTrackingLoop`](@ref), [`DelayCompensatedTrackingLoop`](@ref)
and [`VectorTrackingLoop`](@ref), whose navigation processor also runs inside `step_loop!`.
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
include("post_corr_filter.jl")
include("signals.jl")
include("integration_length.jl")
include("loop_filters.jl")
include("carrier_loop_staging.jl")
include("post_correlation.jl")
include("nco_timeline.jl")
include("tracking_loops.jl")
include("vector/residual_pll.jl")
include("vector/tracking_loop.jl")
include("vector/model.jl")
include("vector/navigation_processor.jl")
include("vector/satellite_step.jl")

end # module
