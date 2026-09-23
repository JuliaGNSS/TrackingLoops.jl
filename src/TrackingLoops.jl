"""
    TrackingLoops

The device-independent core of a GNSS tracking loop, extracted from Tracking.jl
so that the same code closes the loops in the software receiver and in the
allocation-free *loop process* of a hardware correlator
(GNSSReceiver.jl, `docs/plans/2026-09-22-loop-process.md`):

  - the correlator record types and their accessors, the discriminators and
    the post-correlation filter;
  - the navigation-bit buffer with its bit-edge and secondary-code sync
    detectors, per signal;
  - the C/N₀ estimators and the noise-density window they read;
  - the loop filters' bandwidth rules and the Doppler estimators — the
    conventional FLL-assisted PLL/DLL and the delay-aware NCO-referenced
    loop — behind one per-record [`step_loop`](@ref) on a plain per-satellite state;
  - the per-record fold [`apply_record`](@ref) that advances a signal
    component's prompt filter, C/N₀ estimator and bit buffer identically on
    both paths;
  - the [`NCOTimeline`](@ref): what a hardware NCO ran and will run.

Tracking.jl depends on this package for its software correlator; the loop
process's engine (HardwareLoopCore.jl) depends on it without Tracking. Nothing
here reads a raw sample, and nothing here knows a device or a segment.
"""
module TrackingLoops

using BitIntegers
using DocStringExtensions
using GNSSSignals
using SpecialFunctions: erfinv
using StaticArrays
using TrackingLoopFilters
using Unitful: upreferred, uconvert, ustrip, dimension, NoUnits, Hz, dBHz, ms, s
using Random: AbstractRNG, Xoshiro
import Base.zero, Base.length

# 1800-bit exact-width unsigned for the 1800-chip overlay-code searches of
# GPS L1C-P and BeiDou B1C-P.
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
    MAX_LOOP_BANDWIDTH_TIME_PRODUCT,
    default_carrier_loop_filter_bandwidth,
    default_code_loop_filter_bandwidth,
    effective_code_loop_filter_bandwidth,
    aid_dopplers,
    calculate_carrier_frequency_update,
    calculate_code_frequency_update,
    AbstractDopplerEstimator,
    ConventionalPLLAndDLL,
    ConventionalAssistedPLLAndDLL,
    SatConventionalPLLAndDLL,
    NCOReferencedPLLAndDLL,
    SatNCOReferencedPLLAndDLL,
    init_estimator_state,
    reset_estimator_state,
    LoopRecord,
    step_loop,
    SignalLoopState,
    apply_record,
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
    wrap_half_cycle

const Maybe{T} = Union{T,Nothing}

"""
$(SIGNATURES)

Type parameter wrapper for specifying the number of antennas in the system.
Use `NumAnts(n)` to create an instance.
"""
struct NumAnts{x} end

NumAnts(x) = NumAnts{x}()

"""
$(SIGNATURES)

Type parameter wrapper for specifying the number of correlator accumulators.
Use `NumAccumulators(n)` to create an instance.
"""
struct NumAccumulators{x} end

NumAccumulators(x) = NumAccumulators{x}()

"""
    update(x, prompt)

Advance a per-record state — a C/N₀ estimator or a post-correlation filter —
with one prompt and return the new state (immutable update).
"""
function update end

"""
$(SIGNATURES)

Abstract supertype for Doppler estimators. Concrete subtypes carry estimator
configuration; the per-satellite state lives with the satellite — see
[`init_estimator_state`](@ref) and [`step_loop`](@ref).
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
include("estimators.jl")

end # module
