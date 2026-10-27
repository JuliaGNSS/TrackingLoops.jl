using Test
using TrackingLoops
using TrackingLoops:
    NumAccumulators, AbstractCorrelator, AbstractEarlyPromptLateCorrelator,
    get_very_early, get_very_late, get_num_accumulators, get_prompt_index,
    get_early_late_sample_spacing, get_correlator_sample_shifts, update_accumulator,
    normalize, pll_disc, fll_disc, dll_disc, AbstractPostCorrFilter, get_weights,
    update, BitBuffer, PhaseAccumulators, SyncResult, get_code_block_buffer_type,
    detect_bit_or_secondary_code_sync, uses_soft_bit_edge_detection,
    uses_soft_secondary_code_detection, get_bit_edge_detection_confidence,
    get_bit_edge_or_secondary_code_tolerance, AbstractCN0Estimator, CN0UpdateContext,
    requires_noise_density, default_cn0_estimator, AbstractNoiseEstimator,
    NoiseEstimators, NoiseObservation, NoiseDensity, noise_observation,
    noise_observation_from_samples, update_noise!, get_noise_density,
    noise_density_type, noise_window_looks, max_num_code_blocks_to_integrate,
    default_num_code_blocks_to_integrate, calc_num_code_blocks_to_integrate,
    calc_num_code_blocks_for_bit_buffer, MAX_CARRIER_LOOP_BANDWIDTH_TIME_PRODUCT,
    MAX_CODE_LOOP_BANDWIDTH_TIME_PRODUCT, MAX_FLL_ASSIST_LOOP_BANDWIDTH_TIME_PRODUCT,
    default_narrow_carrier_loop_filter_bandwidth,
    default_fll_assist_loop_filter_bandwidth, default_code_loop_filter_bandwidth,
    default_residual_pll_loop_filter_bandwidth,
    effective_fll_assist_loop_filter_bandwidth, effective_code_loop_filter_bandwidth,
    LoopBandwidths, CarrierLoopStage, phase_lock_indicator, phase_locked,
    carrier_phase_correction, aid_dopplers, calculate_code_frequency_update,
    SatConventionalPLLAndDLL, SatNCOReferencedPLLAndDLL, SatVectorPLLAndDLL,
    NavigationStatus, VTReleaseReason, SatelliteReport, position_uncertainty,
    clock_uncertainty, SyncChange, SYNC_UNCHANGED, SYNC_FOUND, SYNC_LOST,
    estimator_state_type, scheduled_words, word_changes_within, despread_noise!,
    NoiseUpdateContext, NoiseDespread, noise_despreads, noise_correlator,
    append_despread_noise!, noise_density_and_ready, noise_window_filling
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
include("estimators.jl")
include("vector_estimator.jl")
include("vector_model.jl")
include("vector_simulation.jl")
include("vector_tracking.jl")
include("vector_engine.jl")
include("vector_closed_loop.jl")
include("vector_pipeline.jl")
include("vector_residual_pll.jl")
include("record.jl")
include("signal_state.jl")
include("bit_buffer.jl")
include("signals.jl")
include("cn0_estimators.jl")
include("discriminators.jl")
include("post_corr_filter.jl")
include("loop_filters.jl")
include("allocations.jl")
include("vector_allocations.jl")
