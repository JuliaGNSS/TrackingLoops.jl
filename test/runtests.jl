using Test
using TrackingLoops
using TrackingLoops:
    NumCorrelators, AbstractCorrelatorBank, AbstractEarlyPromptLateCorrelatorBank,
    get_very_early, get_very_late, get_num_correlators, get_prompt_index,
    calc_early_late_sample_spacing, calc_correlator_sample_offsets, set_correlator_outputs,
    normalize, pll_disc, fll_disc, dll_disc, AbstractPostCorrFilter, get_weights,
    update, BitSynchronizer, SyncHypothesisAccumulators, SyncResult, prompt_sign_history_type,
    detect_sync, uses_soft_bit_edge_detection,
    uses_soft_secondary_code_detection, get_sync_detection_confidence,
    get_secondary_code_error_tolerance, AbstractCN0Estimator, CN0UpdateContext,
    requires_noise_density, default_cn0_estimator, AbstractNoiseDensityEstimator,
    NoiseDensityEstimators, NoiseObservation, NoiseDensity, noise_observation,
    calc_noise_observation_from_samples, calc_noise_observation_from_correlator_bank,
    update_noise!, get_noise_density,
    noise_density_type, get_num_squared_magnitudes, max_num_code_periods_to_integrate,
    default_num_code_periods_to_integrate, calc_num_code_periods_to_integrate,
    calc_num_code_periods_for_bit_sync, MAX_CARRIER_LOOP_BANDWIDTH_TIME_PRODUCT,
    MAX_CODE_LOOP_BANDWIDTH_TIME_PRODUCT, MAX_FLL_ASSIST_LOOP_BANDWIDTH_TIME_PRODUCT,
    default_narrow_carrier_loop_bandwidth,
    default_fll_assist_loop_bandwidth, default_code_loop_bandwidth,
    default_residual_pll_loop_bandwidth,
    calc_effective_fll_assist_loop_bandwidth, calc_effective_code_loop_bandwidth,
    LoopBandwidths, CarrierLoopStage, get_phase_lock_indicator, is_phase_locked,
    calc_carrier_phase_correction, calc_carrier_aided_dopplers,
    ScalarLoopState, SatNCOReferencedPLLAndDLL, VectorLoopState,
    VectorTrackingStatus, VectorTrackingReleaseReason, VT_NOT_RELEASED, VT_INELIGIBLE,
    VT_BELOW_HORIZON, VT_FALLBACK, SatelliteReport, calc_position_std,
    calc_clock_bias_std, SyncEvent, SYNC_UNCHANGED, SYNC_FOUND, SYNC_LOST,
    loop_state_type, despread_noise!,
    NoiseUpdateContext, NoiseProbe, NoiseDespread, noise_despreads, draw_noise_probe,
    get_noise_correlator_bank, append_noise_correlator_outputs!, append_despread_noise!, get_noise_density_and_ready, get_noise_window_filling
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
include("vector_tracking_loop.jl")
include("vector_model.jl")
include("vector_simulation.jl")
include("vector_navigation_processor.jl")
include("vector_satellite_step.jl")
include("vector_closed_loop.jl")
include("vector_pipeline.jl")
include("vector_residual_pll.jl")
include("post_correlation.jl")
include("signal_state.jl")
include("bit_synchronizer.jl")
include("signals.jl")
include("cn0_estimators.jl")
include("discriminators.jl")
include("post_corr_filter.jl")
include("loop_filters.jl")
include("allocations.jl")
include("vector_allocations.jl")
