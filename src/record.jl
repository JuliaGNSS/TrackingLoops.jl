# ─────────────────────────────────────────────────────────────────────────────
# The per-record fold of one signal component (normalise, filter, C/N₀, bit buffer):
# `Tracking._apply_correlator_output`'s arithmetic on bare state, so a loop process
# without Tracking runs the same code and produces the same record history.
# ─────────────────────────────────────────────────────────────────────────────

"""
    SignalLoopState(signal; num_prompts_for_cn0_estimation = 100, cn0_estimator,
                    post_corr_filter)

The per-record state of one signal component on a satellite: bit buffer, C/N₀
estimator, post-correlation filter, the last filtered prompt (the FLL chains from
it), the last record's block count and the polarity it was correlated with
([`get_sync_polarity`](@ref)). Immutable, rebuilt per record by
[`apply_record`](@ref); the estimators own their buffers, so every component needs
its own instance. The state before a fold builds that record's [`LoopRecord`](@ref).
"""
struct SignalLoopState{B<:Unsigned,PCF<:AbstractPostCorrFilter,CN0<:AbstractCN0Estimator}
    bit_buffer::BitBuffer{B}
    cn0_estimator::CN0
    post_corr_filter::PCF
    last_filtered_prompt::ComplexF64
    last_num_code_blocks::Int
    last_polarity::Int8
end

function SignalLoopState(
    signal::AbstractGNSSSignal;
    num_prompts_for_cn0_estimation::Int = 100,
    cn0_estimator::AbstractCN0Estimator = default_cn0_estimator(
        signal,
        num_prompts_for_cn0_estimation,
    ),
    post_corr_filter::AbstractPostCorrFilter = DefaultPostCorrFilter(),
)
    bit_buffer = BitBuffer{get_code_block_buffer_type(signal)}()
    # So the first soft bits after sync do not grow the vector.
    sizehint!(bit_buffer.soft_bits, 64)
    # Seed the sync accumulators at the detector's length so no later record,
    # re-arm or bit-clock restart (they zero in place) allocates.
    if uses_soft_bit_edge_detection(signal)
        _seed_phase_accumulators!(
            bit_buffer.phase_acc,
            _calc_num_code_blocks_that_form_a_bit(signal),
        )
    elseif uses_soft_secondary_code_detection(signal)
        _seed_phase_accumulators!(bit_buffer.phase_acc, get_secondary_code_length(signal))
    end
    SignalLoopState(
        bit_buffer,
        cn0_estimator,
        post_corr_filter,
        complex(0.0, 0.0),
        1,
        Int8(0),
    )
end

has_bit_or_secondary_code_been_found(state::SignalLoopState) =
    has_bit_or_secondary_code_been_found(state.bit_buffer)
get_soft_bits(state::SignalLoopState) = get_soft_bits(state.bit_buffer)
estimate_cn0(state::SignalLoopState, integration_time) =
    estimate_cn0(state.cn0_estimator, integration_time)

# Fold the record into the C/N₀ estimator, or skip it while a required noise density
# is not measured yet. Both branches return the same type; `requires_noise_density`
# folds away at compile time.
@inline function _update_cn0_estimator(
    estimator::AbstractCN0Estimator,
    prompt,
    signal::AbstractGNSSSignal,
    bit_buffer::BitBuffer,
    num_code_blocks::Integer,
    bit_sync_usable::Bool,
    noise_density,
    noise_density_ready::Bool,
    integration_time,
)
    if !noise_density_ready && requires_noise_density(estimator)
        return estimator
    end
    # Positional, not keyword: this is the per-record site.
    update(
        estimator,
        prompt,
        CN0UpdateContext(
            signal,
            bit_buffer,
            num_code_blocks,
            bit_sync_usable,
            noise_density,
            integration_time,
        ),
    )
end

# Whether a pre-sync-correlated record's prompt is dropped (see `fold_record`).
@inline _drops_pre_sync_prompt(signal::AbstractGNSSSignal, correlated_pre_sync::Bool) =
    correlated_pre_sync && get_secondary_code_length(signal) > 1

# `get_sync_polarity` of a record correlated with `bit_buffer`'s state, or with the
# pre-sync replica if `correlated_pre_sync`, which leaves the secondary code on.
@inline _correlated_polarity(
    signal,
    bit_buffer::BitBuffer,
    prn,
    correlated_pre_sync::Bool,
) =
    _drops_pre_sync_prompt(signal, correlated_pre_sync) ? Int8(0) :
    get_sync_polarity(signal, bit_buffer, prn)

# One record into the bit buffer. Shared by `fold_record` and the vector loop's bit
# clock so the two cannot diverge.
@inline function _advance_bit_buffer(
    signal::AbstractGNSSSignal,
    prn::Integer,
    bit_buffer::BitBuffer,
    bit_block_count::Integer,
    bit_prompt,
    correlated_pre_sync::Bool,
)
    drop_prompt = _drops_pre_sync_prompt(signal, correlated_pre_sync)
    bit_buffer = buffer(
        signal,
        prn,
        bit_buffer,
        bit_block_count,
        drop_prompt ? zero(bit_prompt) : bit_prompt,
    )
    # The code-phase snap after this fold aligns the *upcoming* integration to
    # `bit_buffer.secondary_phase`.
    if correlated_pre_sync
        bit_buffer = _advance_secondary_phase(signal, bit_buffer, bit_block_count)
    end
    bit_buffer
end

"""
    fold_record(signal, prn, bit_buffer, cn0_estimator, post_corr_filter, output,
                sampling_frequency, noise_density, noise_density_ready,
                driver_carrier_phase_offset, correlated_pre_sync)
        -> (bit_buffer, cn0_estimator, post_corr_filter, prompt, filtered_correlator,
            bit_block_count, integrated_code_blocks, overshoot)

Apply one completed record to a signal component's bare state: normalize the raw
correlator by sample count and code amplitude, update and apply the
post-correlation filter, advance the C/N₀ estimator and the bit buffer.

The bit buffer is credited with the blocks actually integrated, recovered from the
sample count. `correlated_pre_sync = true` marks a record after a bit/secondary sync
found earlier in the same fold: for secondary-coded signals its prompt is kept out
of the coherent sum (the sync changed the replica under it), but its blocks are
credited and it moves the secondary-code anchor. `overshoot` is true when a
post-sync record carried the accumulator past the bit boundary, so the buffer
dropped sync and restarted its search (JuliaGNSS/Tracking.jl#238).
"""
@inline function fold_record(
    signal::AbstractGNSSSignal,
    prn::Integer,
    bit_buffer::BitBuffer,
    cn0_estimator::AbstractCN0Estimator,
    post_corr_filter::AbstractPostCorrFilter,
    output::CorrelatorOutput,
    sampling_frequency,
    noise_density,
    noise_density_ready::Bool,
    driver_carrier_phase_offset::Real,
    correlated_pre_sync::Bool,
)
    bit_buffer_before = bit_buffer
    normalized_correlator =
        normalize(output.correlator, output.integrated_samples, get_code_amplitude(signal))
    post_corr_filter = update(post_corr_filter, get_prompt(normalized_correlator))
    # Used twice: to combine the antennas and to reduce the noise covariance.
    weights = get_weights(post_corr_filter, _num_ants_val(normalized_correlator))
    filtered_correlator = _combine_correlator(normalized_correlator, weights)
    prompt = get_prompt(filtered_correlator)
    bit_block_count = calc_num_code_blocks_for_bit_buffer(
        signal,
        output.integrated_samples,
        sampling_frequency,
        has_bit_or_secondary_code_been_found(bit_buffer),
    )
    # Floor at 1 for the fractional-block record right after a sync phase snap.
    integrated_code_blocks = max(1, bit_block_count)
    # Onto the driver's (real) phase frame for sync search and bit accumulation.
    bit_prompt = prompt * _carrier_phase_derotation(driver_carrier_phase_offset, signal)
    drop_prompt = _drops_pre_sync_prompt(signal, correlated_pre_sync)
    scalar_noise_density = _reduce_noise_density(noise_density, weights)
    cn0_estimator = _update_cn0_estimator(
        cn0_estimator,
        prompt,
        signal,
        bit_buffer,
        bit_block_count,
        !drop_prompt,
        scalar_noise_density,
        noise_density_ready,
        output.integrated_samples / sampling_frequency,
    )
    bit_buffer = _advance_bit_buffer(
        signal,
        prn,
        bit_buffer,
        bit_block_count,
        bit_prompt,
        correlated_pre_sync,
    )
    overshoot =
        has_bit_or_secondary_code_been_found(bit_buffer_before) &&
        !has_bit_or_secondary_code_been_found(bit_buffer)
    return bit_buffer,
    cn0_estimator,
    post_corr_filter,
    prompt,
    filtered_correlator,
    bit_block_count,
    integrated_code_blocks,
    overshoot
end

"""
    apply_record(state::SignalLoopState, signal, prn, output, sampling_frequency,
                 noise_density, noise_density_ready,
                 driver_carrier_phase_offset = get_carrier_phase_offset(signal);
                 correlated_pre_sync = false)
        -> (state, prompt, filtered_correlator, integrated_code_blocks, overshoot)

`fold_record` on a [`SignalLoopState`](@ref). Returns the new state, the prompt,
the filtered correlator the discriminators read, the blocks the record covered and
whether it overshot the bit boundary (see `fold_record`). Nothing here logs; the
caller reports `overshoot` (Tracking.jl warns once per satellite). Build the
record's [`LoopRecord`](@ref) from `state`, the state before this fold.

`driver_carrier_phase_offset` is the carrier-phase offset of the satellite's
estimator-driver signal; the bit-buffer prompt is de-rotated against it. The
default (the signal's own) suits the driver; a passenger component (the data half
of a pilot/data pair) must be handed the driver's.
"""
@inline function apply_record(
    state::SignalLoopState,
    signal::AbstractGNSSSignal,
    prn::Integer,
    output::CorrelatorOutput,
    sampling_frequency,
    noise_density,
    noise_density_ready::Bool,
    driver_carrier_phase_offset::Real = get_carrier_phase_offset(signal);
    correlated_pre_sync::Bool = false,
)
    bit_buffer,
    cn0_estimator,
    post_corr_filter,
    prompt,
    filtered_correlator,
    _,
    integrated_code_blocks,
    overshoot = fold_record(
        signal,
        prn,
        state.bit_buffer,
        state.cn0_estimator,
        state.post_corr_filter,
        output,
        sampling_frequency,
        noise_density,
        noise_density_ready,
        driver_carrier_phase_offset,
        correlated_pre_sync,
    )
    new_state = SignalLoopState(
        bit_buffer,
        cn0_estimator,
        post_corr_filter,
        prompt,
        integrated_code_blocks,
        # As the record was correlated: before the fold that may sync it.
        _correlated_polarity(signal, state.bit_buffer, prn, correlated_pre_sync),
    )
    new_state, prompt, filtered_correlator, integrated_code_blocks, overshoot
end

"""
    restart_bit_clock(state::SignalLoopState) -> SignalLoopState

The state with a fresh, unsynchronised bit buffer (what a lost record costs),
everything else kept.
"""
restart_bit_clock(state::SignalLoopState) = SignalLoopState(
    _fresh_bit_buffer(state.bit_buffer),
    state.cn0_estimator,
    state.post_corr_filter,
    state.last_filtered_prompt,
    state.last_num_code_blocks,
    state.last_polarity,
)

# An unsynchronised bit buffer reusing `bb`'s vectors, emptied in place.
function _fresh_bit_buffer(bb::BitBuffer{B}) where {B}
    empty!(bb.soft_bits)
    BitBuffer{B}(
        zero(B),
        0,
        false,
        0,
        Int8(0),
        complex(0.0, 0.0),
        0,
        bb.soft_bits,
        _reset_phase_accumulators!(bb.phase_acc),
    )
end

"""
    reset_signal_state(state::SignalLoopState) -> SignalLoopState

The state a freshly armed channel starts from (no sync, no soft bits, empty C/N₀
estimator, no previous prompt), reusing the old vectors so a re-arm allocates
nothing. The post-correlation filter is kept: it is configuration, not history;
an adaptive filter must be reset by its owner.

A custom [`AbstractCN0Estimator`](@ref) with history needs a
`TrackingLoops._reset_cn0_estimator` method; without one this throws rather than
hand the new satellite the old one's C/N₀.
"""
reset_signal_state(state::SignalLoopState) = SignalLoopState(
    _fresh_bit_buffer(state.bit_buffer),
    _reset_cn0_estimator(state.cn0_estimator),
    state.post_corr_filter,
    complex(0.0, 0.0),
    1,
    Int8(0),
)

_reset_cn0_estimator(e::NoiseRefCN0Estimator) =
    (fill!(e.buffered_cn0, 0.0); NoiseRefCN0Estimator(e.num_records, e.buffered_cn0, 0, 0))
_reset_cn0_estimator(e::MomentsCN0Estimator) =
    (fill!(e.prompt_buffer, zero(ComplexF64)); MomentsCN0Estimator(e.prompt_buffer, 0, 0))
function _reset_cn0_estimator(e::NWPRCN0Estimator)
    fill!(e.buffered_narrowband_powers, 0.0)
    fill!(e.buffered_wideband_powers, 0.0)
    _with_window_state(
        e,
        _reset_cn0_estimator(e.fallback);
        ratio_current_index = 0,
        filled_ratio_length = 0,
        num_records_per_ratio = 0,
        ratios_are_bit_aligned = false,
    )
end
_reset_cn0_estimator(e::NoCN0Estimator) = e
_reset_cn0_estimator(e::AbstractCN0Estimator) = throw(
    ArgumentError(
        "this C/N₀ estimator has no `TrackingLoops._reset_cn0_estimator` " *
        "method; add one that empties its history",
    ),
)
