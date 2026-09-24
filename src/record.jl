# ─────────────────────────────────────────────────────────────────────────────
# The per-record fold of one signal component: normalise the record, filter the
# prompt, advance the C/N₀ estimator and the bit buffer. This is the arithmetic
# `Tracking._apply_correlator_output` runs on a `TrackedSignal`, lifted onto the
# component's bare state so a loop process without Tracking runs the same code
# — and so the record history the receiver mirrors is identical on both paths.
# ─────────────────────────────────────────────────────────────────────────────

"""
    SignalLoopState(signal; num_prompts_for_cn0_estimation = 100, cn0_estimator, post_corr_filter)

The per-record state of one signal component on a satellite: its bit buffer,
C/N₀ estimator, post-correlation filter, the last filtered prompt (which the
FLL discriminator chains from) and the block count of the last record. An
immutable value, rebuilt by [`apply_record`](@ref) per record; the estimators
buffer into vectors they own, so every component needs its own instance.
"""
struct SignalLoopState{B<:Unsigned,PCF<:AbstractPostCorrFilter,CN0<:AbstractCN0Estimator}
    bit_buffer::BitBuffer{B}
    cn0_estimator::CN0
    post_corr_filter::PCF
    last_filtered_prompt::ComplexF64
    last_num_code_blocks::Int
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
    # Room for the soft bits a fold can complete before its consumer drains
    # them, so the first bit after sync does not grow the vector.
    sizehint!(bit_buffer.soft_bits, 64)
    # Seed the sync detector's per-hypothesis accumulators now, at the length
    # the signal's detector uses, so the first record after an arm — and every
    # re-arm and bit-clock restart, which zero them in place — allocates nothing.
    if uses_soft_bit_edge_detection(signal)
        _seed_phase_accumulators!(bit_buffer.phase_acc, _calc_num_code_blocks_that_form_a_bit(signal))
    elseif uses_soft_secondary_code_detection(signal)
        _seed_phase_accumulators!(bit_buffer.phase_acc, get_secondary_code_length(signal))
    end
    SignalLoopState(
        bit_buffer,
        cn0_estimator,
        post_corr_filter,
        complex(0.0, 0.0),
        1,
    )
end

has_bit_or_secondary_code_been_found(state::SignalLoopState) =
    has_bit_or_secondary_code_been_found(state.bit_buffer)
get_soft_bits(state::SignalLoopState) = get_soft_bits(state.bit_buffer)
estimate_cn0(state::SignalLoopState, integration_time) =
    estimate_cn0(state.cn0_estimator, integration_time)

# Build the context and fold the record into one signal's CN0 estimator, or skip
# it where the estimator needs a density that signal has not measured yet. Both
# branches return the same concrete estimator type, so this stays type-stable
# and allocation-free; the `requires_noise_density` half of the condition folds
# away at compile time.
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

"""
    fold_record(signal, prn, bit_buffer, cn0_estimator, post_corr_filter, output,
                sampling_frequency, noise_density, noise_density_ready,
                driver_carrier_phase, correlated_pre_sync)
        -> (bit_buffer, cn0_estimator, post_corr_filter, prompt, filtered_correlator,
            bit_block_count, integrated_code_blocks, overshoot)

Apply one completed record to the bare per-record state of a signal component:
normalize the record's raw correlator by its sample count and code amplitude,
update and apply the post-correlation filter, advance the C/N₀ estimator and the
bit buffer. Returns the new state values plus the filtered correlator and block
counts the loop-filter step needs.

The bit accumulator is credited with the blocks *actually* integrated
(`calc_num_code_blocks_for_bit_buffer`), recovered from the record's sample
count. `correlated_pre_sync = true` marks a record that follows a bit/secondary
sync detected earlier in the same fold: its prompt is kept out of the coherent
bit accumulation for secondary-coded signals (the sync changed the replica
under it) but its blocks are always credited, and it moves the secondary-code
anchor along. `overshoot` reports a post-sync record that carried the bit
accumulator past the navigation-bit boundary, on which the bit buffer dropped
sync and restarted its search (issue #238). Everything here is the arithmetic
Tracking's per-record advance performs, in the same order.
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
    driver_carrier_phase::Real,
    correlated_pre_sync::Bool,
)
    bit_buffer_before = bit_buffer
    normalized_correlator =
        normalize(output.correlator, output.integrated_samples, get_code_amplitude(signal))
    post_corr_filter = update(post_corr_filter, get_prompt(normalized_correlator))
    # The filter's weights, read once and used twice: to combine the antennas
    # here, and to reduce the shared noise covariance to *this* satellite's
    # floor below.
    weights = get_weights(post_corr_filter, _num_ants_val(normalized_correlator))
    filtered_correlator = _combine_correlator(normalized_correlator, weights)
    prompt = get_prompt(filtered_correlator)
    bit_block_count = calc_num_code_blocks_for_bit_buffer(
        signal,
        output.integrated_samples,
        sampling_frequency,
        has_bit_or_secondary_code_been_found(bit_buffer),
    )
    # Blocks this record actually covered, for the driver's `1/N` carrier-
    # bandwidth scaling. The floor at 1 covers the fractional-block record right
    # after a sync phase-snap accumulator reset.
    integrated_code_blocks = max(1, bit_block_count)
    # De-rotate the prompt onto the driver's (real) phase frame before both the
    # sync search and the coherent bit accumulation.
    bit_prompt = prompt * _carrier_phase_derotation(driver_carrier_phase, signal)
    # Keep a pre-sync-correlated record's prompt out of the coherent sum where
    # the sync changed the replica under it (secondary-code wipe-off), but
    # always let it advance the accumulator's block count.
    drop_prompt = correlated_pre_sync && get_secondary_code_length(signal) > 1
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
    bit_buffer = buffer(
        signal,
        prn,
        bit_buffer,
        bit_block_count,
        drop_prompt ? zero(bit_prompt) : bit_prompt,
    )
    # Such a record also moves the secondary-code anchor: the code-phase snap
    # runs after this fold and aligns the *upcoming* integration to
    # `bit_buffer.secondary_phase`.
    if correlated_pre_sync
        bit_buffer = _advance_secondary_phase(signal, bit_buffer, bit_block_count)
    end
    # A post-sync record that carried the accumulator past the bit boundary
    # made `buffer` drop sync and restart the search: report it, so the caller
    # can say so (a receiver logs, a loop process publishes a status event).
    overshoot = has_bit_or_secondary_code_been_found(bit_buffer_before) &&
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
                 driver_carrier_phase = get_carrier_phase_offset(signal);
                 correlated_pre_sync = false)
        -> (state, prompt, filtered_correlator, integrated_code_blocks, overshoot)

`fold_record` on a [`SignalLoopState`](@ref): the new state (with the
filtered prompt as its `last_filtered_prompt`), the prompt, the filtered
correlator the discriminators read, the blocks the record covered, and
whether the record overshot the navigation-bit boundary — on which the bit
buffer dropped sync and restarted its search. Nothing here logs; report
`overshoot` the way the caller reports things (Tracking.jl warns once per
satellite).

`driver_carrier_phase` is the carrier-phase offset of the satellite's
estimator-driver signal (`get_carrier_phase_offset`), against which this
component's bit-buffer prompt is de-rotated. It defaults to the signal's own
offset — a no-op, right for the driver itself; a passenger component (the data
half of a pilot/data pair) must be handed the driver's.
"""
@inline function apply_record(
    state::SignalLoopState,
    signal::AbstractGNSSSignal,
    prn::Integer,
    output::CorrelatorOutput,
    sampling_frequency,
    noise_density,
    noise_density_ready::Bool,
    driver_carrier_phase::Real = get_carrier_phase_offset(signal);
    correlated_pre_sync::Bool = false,
)
    bit_buffer, cn0_estimator, post_corr_filter, prompt, filtered_correlator, _, integrated_code_blocks, overshoot =
        fold_record(
            signal,
            prn,
            state.bit_buffer,
            state.cn0_estimator,
            state.post_corr_filter,
            output,
            sampling_frequency,
            noise_density,
            noise_density_ready,
            driver_carrier_phase,
            correlated_pre_sync,
        )
    new_state = SignalLoopState(
        bit_buffer,
        cn0_estimator,
        post_corr_filter,
        prompt,
        integrated_code_blocks,
    )
    new_state, prompt, filtered_correlator, integrated_code_blocks, overshoot
end

"""
    restart_bit_clock(state::SignalLoopState) -> SignalLoopState

The state with a fresh, unsynchronised bit buffer — what a lost record costs a
satellite (its bit clock is rebuilt from the signal), everything else kept.
"""
restart_bit_clock(state::SignalLoopState) = SignalLoopState(
    _fresh_bit_buffer(state.bit_buffer),
    state.cn0_estimator,
    state.post_corr_filter,
    state.last_filtered_prompt,
    state.last_num_code_blocks,
)

# An unsynchronised bit buffer that owns `bb`'s vectors: the soft bits are
# emptied and the sync accumulators zeroed in place, so nothing is allocated.
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

The state a freshly armed channel starts from — no sync, empty soft bits, an
empty C/N₀ estimator, no previous prompt — reusing the vectors the previous
occupant's state owned, so a re-arm allocates nothing. The post-correlation
filter is kept as it is: it is configuration (a beamformer, say), not history,
and a filter that adapts must be reset by whoever owns it.

A custom [`AbstractCN0Estimator`](@ref) that carries history has to add a
method to `TrackingLoops._reset_cn0_estimator`; without one this throws rather
than hand the new satellite the old one's C/N₀.
"""
reset_signal_state(state::SignalLoopState) = SignalLoopState(
    _fresh_bit_buffer(state.bit_buffer),
    _reset_cn0_estimator(state.cn0_estimator),
    state.post_corr_filter,
    complex(0.0, 0.0),
    1,
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
