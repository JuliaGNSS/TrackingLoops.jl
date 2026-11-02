using Test
using TrackingLoops
using TrackingLoops:
    NumCorrelators, AbstractCorrelatorBank, AbstractEarlyPromptLateCorrelatorBank,
    get_very_early, get_very_late, get_num_correlators, get_prompt_index,
    calc_early_late_sample_spacing, calc_correlator_sample_offsets, with_correlator_outputs,
    normalize, pll_disc, fll_disc, dll_disc, AbstractBeamformer, get_beamforming_weights,
    update!, BitSynchronizer, SyncHypothesisAccumulators, SyncResult, prompt_sign_history_type,
    detect_sync, uses_soft_bit_edge_detection,
    uses_soft_secondary_code_detection, get_sync_detection_confidence,
    get_secondary_code_error_tolerance, AbstractCN0Estimator, CN0UpdateContext,
    requires_noise_density, default_cn0_estimator, AbstractNoiseDensityEstimator,
    NoiseEstimators, NoiseObservation, NoiseDensity, noise_observation,
    calc_noise_observation_from_samples, update_noise!, get_noise_density,
    noise_density_type, get_num_squared_magnitudes, max_num_code_periods_to_integrate,
    default_num_code_periods_to_integrate, calc_num_code_periods_to_integrate,
    calc_num_bit_synchronizer_code_periods, MAX_CARRIER_LOOP_BANDWIDTH_TIME_PRODUCT,
    MAX_CODE_LOOP_BANDWIDTH_TIME_PRODUCT, MAX_FLL_ASSIST_LOOP_BANDWIDTH_TIME_PRODUCT,
    default_narrow_carrier_loop_filter_bandwidth,
    default_fll_assist_loop_filter_bandwidth, default_code_loop_filter_bandwidth,
    default_residual_pll_loop_filter_bandwidth,
    effective_fll_assist_loop_filter_bandwidth, effective_code_loop_filter_bandwidth,
    LoopBandwidths, CarrierLoopStage, get_phase_lock_indicator, is_phase_locked,
    calc_carrier_phase_correction, calc_carrier_aided_dopplers, step_code_loop,
    ScalarLoopState, VectorLoopState,
    NavigationStatus, VectorReleaseReason, SatelliteReport, calc_position_std,
    calc_clock_bias_std, SyncEvent, SYNC_UNCHANGED, SYNC_FOUND, SYNC_LOST,
    loop_state_type, despread_noise!,
    NoiseUpdateContext, NoiseProbe, NoiseDespread, noise_despreads, next_noise_probe,
    noise_correlator_bank, append_noise_correlator_outputs!, append_despread_noise!, noise_density_and_ready, noise_window_filling
using GNSSSignals
using StaticArrays
using Unitful
using Unitful: Hz, dBHz, s, ms, @u_str
using AllocCheck
using LinearAlgebra: norm
using TrackingLoopFilters: ThirdOrderAssistedBilinearLF, SecondOrderBilinearLF

include("api.jl")
include("correlators.jl")
include("sample_parameters.jl")
include("noise_estimators.jl")
include("nco_timeline.jl")
include("tracking_loops.jl")
include("vector_loop_state.jl")
include("vector_model.jl")
include("vector_simulation.jl")
include("vector_tracking.jl")
include("vector_engine.jl")
include("vector_closed_loop.jl")
include("vector_pipeline.jl")
include("vector_residual_pll.jl")
include("post_correlation.jl")
include("signal_state.jl")
include("bit_synchronizer.jl")
include("signals.jl")
include("cn0_estimators.jl")
include("discriminators.jl")
include("beamformer.jl")
include("loop_filters.jl")
include("allocations.jl")
include("vector_allocations.jl")
