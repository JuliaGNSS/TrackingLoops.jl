# ─────────────────────────────────────────────────────────────────────────────
# The per-record fold of one signal component (normalise, filter, C/N₀, bit buffer):
# `Tracking._apply_correlator_output`'s arithmetic on bare state, so a loop process
# without Tracking runs the same code and produces the same record history.
# ─────────────────────────────────────────────────────────────────────────────

"""
    SyncChange

What one record did to a signal's bit clock (see [`SignalLoopState`](@ref) and
[`LoopRecord`](@ref)):

  - `SYNC_UNCHANGED`: neither found nor lost the bit or secondary-code sync;
  - `SYNC_FOUND`: found it;
  - `SYNC_LOST`: lost it — a post-sync record that overshot the navigation-bit
    boundary, on which the bit buffer restarted its search.
"""
@enum SyncChange::Int8 SYNC_UNCHANGED SYNC_FOUND SYNC_LOST

"""
    SignalLoopState(signal; num_prompts_for_cn0_estimation = 100, cn0_estimator,
                    post_corr_filter)

The per-record state of one signal component on a satellite: bit buffer, C/N₀
estimator, post-correlation filter, the last filtered prompt (the FLL chains from
it) and the last record's block count. Immutable, rebuilt per record by
[`apply_record`](@ref); the estimators own their buffers, so every component needs
its own instance: one per signal of every satellite, driver and passengers alike.

It also keeps what the last record did, for a [`LoopRecord`](@ref) built from it:
`last_sync_change` (a [`SyncChange`](@ref)), `last_num_new_soft_bits` (the soft bits
it appended), `last_previous_prompt` (the prompt the FLL compares its prompt with:
the one before, or `0` across a change of carrier phase polarity) and
`last_carrier_phase_polarity` (its prompt's sign where known, else `0`; see
[`LoopRecord`](@ref)).
"""
struct SignalLoopState{B<:Unsigned,PCF<:AbstractPostCorrFilter,CN0<:AbstractCN0Estimator}
    bit_buffer::BitBuffer{B}
    cn0_estimator::CN0
    post_corr_filter::PCF
    last_filtered_prompt::ComplexF64
    last_num_code_blocks::Int
    last_sync_change::SyncChange
    last_num_new_soft_bits::Int
    last_previous_prompt::ComplexF64
    last_carrier_phase_polarity::Int8
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
        _seed_phase_accumulators!(bit_buffer.phase_acc, get_code_blocks_per_bit(signal))
    elseif uses_soft_secondary_code_detection(signal)
        _seed_phase_accumulators!(bit_buffer.phase_acc, get_secondary_code_length(signal))
    end
    SignalLoopState(
        bit_buffer,
        cn0_estimator,
        post_corr_filter,
        complex(0.0, 0.0),
        1,
        SYNC_UNCHANGED,
        0,
        complex(0.0, 0.0),
        Int8(0),
    )
end

"""
    SignalLoopState(state::SignalLoopState; bit_buffer, cn0_estimator, post_corr_filter,
                    last_filtered_prompt, last_num_code_blocks, last_sync_change,
                    last_num_new_soft_bits, last_previous_prompt,
                    last_carrier_phase_polarity)

`state` with the fields passed replaced, every other one kept, so a host that
swaps a part of the state (a C/N₀ estimator, say) need not list the rest.
"""
function SignalLoopState(
    state::SignalLoopState;
    bit_buffer::Maybe{BitBuffer} = nothing,
    cn0_estimator::Maybe{AbstractCN0Estimator} = nothing,
    post_corr_filter::Maybe{AbstractPostCorrFilter} = nothing,
    last_filtered_prompt::Maybe{Complex} = nothing,
    last_num_code_blocks::Maybe{Integer} = nothing,
    last_sync_change::Maybe{SyncChange} = nothing,
    last_num_new_soft_bits::Maybe{Integer} = nothing,
    last_previous_prompt::Maybe{Complex} = nothing,
    last_carrier_phase_polarity::Maybe{Integer} = nothing,
)
    SignalLoopState(
        isnothing(bit_buffer) ? state.bit_buffer : bit_buffer,
        isnothing(cn0_estimator) ? state.cn0_estimator : cn0_estimator,
        isnothing(post_corr_filter) ? state.post_corr_filter : post_corr_filter,
        isnothing(last_filtered_prompt) ? state.last_filtered_prompt :
        ComplexF64(last_filtered_prompt),
        isnothing(last_num_code_blocks) ? state.last_num_code_blocks : Int(last_num_code_blocks),
        isnothing(last_sync_change) ? state.last_sync_change : last_sync_change,
        isnothing(last_num_new_soft_bits) ? state.last_num_new_soft_bits :
        Int(last_num_new_soft_bits),
        isnothing(last_previous_prompt) ? state.last_previous_prompt :
        ComplexF64(last_previous_prompt),
        isnothing(last_carrier_phase_polarity) ? state.last_carrier_phase_polarity :
        Int8(last_carrier_phase_polarity),
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

# Whether a record's prompt is kept out of a secondary-coded signal's coherent sum:
# one after the sync whose replica has not wiped the overlay yet (the sync changed
# the replica under it). Its blocks are credited all the same.
@inline _drops_unwiped_prompt(
    signal::AbstractGNSSSignal,
    synced_before::Bool,
    overlay_wiped::Bool,
) = synced_before && !overlay_wiped && get_secondary_code_length(signal) > 1

"""
    is_dataless_with_secondary_code(signal) -> Bool

Whether `signal` is dataless with a secondary code: GPS L1C-P and L5Q, Galileo E1C,
E5a-Q, E5b-Q and E6-C, BeiDou B1C-P and B2a-Q. Synced and correlated with the overlay
wiped, such a signal's prompt keeps the sign the sync read, so the four-quadrant
carrier discriminators apply. Its pre-sync prompts must be primary-only: a host
whose pre-sync replica carries a secondary chip uses this to pick the records to
multiply by that chip.
"""
is_dataless_with_secondary_code(signal::AbstractGNSSSignal) =
    iszero(get_bit_rate(signal)) && get_secondary_code_length(signal) > 1

# A record's carrier phase polarity (see `fold_record`). `bit_buffer` is the one the
# record was correlated with, before the fold. The sync read its polarity off the
# overlay's sign in the primary-only pre-sync prompts, the sign the prompt keeps once
# the overlay is wiped. The pilots without a secondary code (GPS L2 CL, Galileo
# E5a-QP) give `0` only for now: their prompts share a sign, so their FLL could be
# four-quadrant from the start and their PLL could take the sign the Costas loop
# holds, though no sync resolves it.
@inline _carrier_phase_polarity(
    signal::AbstractGNSSSignal,
    bit_buffer::BitBuffer,
    overlay_wiped::Bool,
) =
    is_dataless_with_secondary_code(signal) && overlay_wiped ? bit_buffer.polarity :
    Int8(0)

"""
    fold_record(signal, prn, bit_buffer, cn0_estimator, post_corr_filter, output,
                sampling_frequency, noise_density, noise_density_ready,
                driver_carrier_phase_offset;
                overlay_wiped = has_bit_or_secondary_code_been_found(bit_buffer))
        -> (bit_buffer, cn0_estimator, post_corr_filter, prompt, filtered_correlator,
            bit_block_count, integrated_code_blocks, overshoot, carrier_phase_polarity)

Apply one completed record to a signal component's bare state: normalize the raw
correlator by sample count and code amplitude, update and apply the
post-correlation filter, advance the C/N₀ estimator and the bit buffer.

The bit buffer is credited with the blocks actually integrated
(`calc_num_code_blocks_for_bit_buffer`), recovered from the sample count. Everything
here is Tracking's per-record arithmetic, in the same order.
`carrier_phase_polarity` is the sign of the record's prompt where the replica wiped
every sign modulation off it (a dataless signal with a secondary code, overlay
wiped), else `0`; a [`LoopRecord`](@ref) carries it to the carrier discriminators.

`overshoot`, `overlay_wiped` and the pre-sync contract are as for
[`apply_record`](@ref).
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
    driver_carrier_phase_offset::Real;
    overlay_wiped::Bool = has_bit_or_secondary_code_been_found(bit_buffer),
)
    bit_buffer_before = bit_buffer
    synced_before = has_bit_or_secondary_code_been_found(bit_buffer)
    normalized_correlator =
        normalize(output.correlator, output.num_integrated_samples, get_code_amplitude(signal))
    post_corr_filter = update(post_corr_filter, get_prompt(normalized_correlator))
    # Used twice: to combine the antennas and to reduce the shared noise covariance
    # to this satellite's density.
    weights = get_weights(post_corr_filter, NumAnts(normalized_correlator))
    filtered_correlator = _combine_correlator(normalized_correlator, weights)
    prompt = get_prompt(filtered_correlator)
    bit_block_count = calc_num_code_blocks_for_bit_buffer(
        signal,
        output.num_integrated_samples,
        sampling_frequency,
        synced_before,
    )
    # Floor at 1 for the fractional-block record right after a sync phase snap.
    integrated_code_blocks = max(1, bit_block_count)
    # Onto the driver's (real) phase frame for sync search and bit accumulation.
    bit_prompt = prompt * _carrier_phase_derotation(driver_carrier_phase_offset, signal)
    drop_prompt = _drops_unwiped_prompt(signal, synced_before, overlay_wiped)
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
        output.num_integrated_samples / sampling_frequency,
    )
    bit_buffer = buffer(
        signal,
        prn,
        bit_buffer,
        bit_block_count,
        drop_prompt ? zero(bit_prompt) : bit_prompt,
    )
    # An unwiped post-sync record moves the secondary-code anchor: the code-phase
    # snap after this fold aligns the *upcoming* integration to `bit_buffer.secondary_phase`.
    if synced_before && !overlay_wiped
        bit_buffer = _advance_secondary_phase(signal, bit_buffer, bit_block_count)
    end
    overshoot = has_bit_or_secondary_code_been_found(bit_buffer_before) &&
        !has_bit_or_secondary_code_been_found(bit_buffer)
    return bit_buffer,
    cn0_estimator,
    post_corr_filter,
    prompt,
    filtered_correlator,
    bit_block_count,
    integrated_code_blocks,
    overshoot,
    _carrier_phase_polarity(signal, bit_buffer_before, overlay_wiped)
end

"""
    apply_record(state::SignalLoopState, signal, prn, output, sampling_frequency,
                 noise_density, noise_density_ready,
                 driver_carrier_phase_offset = get_carrier_phase_offset(signal);
                 overlay_wiped = has_bit_or_secondary_code_been_found(state))
        -> (state, prompt, filtered_correlator, integrated_code_blocks, overshoot)

`fold_record` on a [`SignalLoopState`](@ref). Returns the new state, the prompt,
the filtered correlator the discriminators read, the blocks the record covered and
`overshoot`: whether a post-sync record carried the bit accumulator past the bit
boundary, so the buffer dropped sync and restarted its search (`SYNC_LOST`;
JuliaGNSS/Tracking.jl#238). The new state keeps the filtered prompt and what the
record did, for a [`LoopRecord`](@ref) built from it. Nothing here logs; the caller
reports `overshoot` (Tracking.jl warns once per satellite).

`driver_carrier_phase_offset` is the carrier-phase offset (cycles, `get_carrier_phase_offset`) of
the satellite's estimator-driver signal; the bit-buffer prompt is de-rotated against
it. The default (the signal's own) suits the driver; a passenger component (the data
half of a pilot/data pair) must be handed the driver's.

The pre-sync prompts of an [`is_dataless_with_secondary_code`](@ref) signal must be
primary-only, so that the sync's polarity is the prompt's sign once the overlay is
wiped. For any other signal the sync polarity only fixes the soft bits' sign, which
the Costas loop leaves ambiguous anyway, so a fixed secondary chip in the pre-sync
replica is fine.

`overlay_wiped` says whether the record's replica wiped the secondary code at the
synced phase; it defaults to the buffer having synced before the record. Pass
`false` for a record after a sync it was not correlated for yet (e.g. one found
earlier in the same fold): for a secondary-coded signal its prompt is kept out of
the coherent sum, though its blocks are credited, it moves the secondary-code anchor
and its polarity is `0`.
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
    overlay_wiped::Bool = has_bit_or_secondary_code_been_found(state.bit_buffer),
)
    was_synced = has_bit_or_secondary_code_been_found(state.bit_buffer)
    num_soft_bits = Base.length(state.bit_buffer.soft_bits)
    bit_buffer,
    cn0_estimator,
    post_corr_filter,
    prompt,
    filtered_correlator,
    _,
    integrated_code_blocks,
    overshoot,
    carrier_phase_polarity = fold_record(
        signal,
        prn,
        state.bit_buffer,
        state.cn0_estimator,
        state.post_corr_filter,
        output,
        sampling_frequency,
        noise_density,
        noise_density_ready,
        driver_carrier_phase_offset;
        overlay_wiped,
    )
    is_synced = has_bit_or_secondary_code_been_found(bit_buffer)
    sync_change =
        !was_synced && is_synced ? SYNC_FOUND :
        was_synced && !is_synced ? SYNC_LOST : SYNC_UNCHANGED
    new_state = SignalLoopState(
        bit_buffer,
        cn0_estimator,
        post_corr_filter,
        prompt,
        integrated_code_blocks,
        sync_change,
        Base.length(bit_buffer.soft_bits) - num_soft_bits,
        # The FLL cannot compare prompts of different carrier phase polarities.
        carrier_phase_polarity != state.last_carrier_phase_polarity ? complex(0.0, 0.0) :
        state.last_filtered_prompt,
        carrier_phase_polarity,
    )
    new_state, prompt, filtered_correlator, integrated_code_blocks, overshoot
end

"""
    restart_bit_clock(state::SignalLoopState) -> SignalLoopState

The state with a fresh, unsynchronised bit buffer (what a lost record costs),
everything else kept.
"""
restart_bit_clock(state::SignalLoopState) = SignalLoopState(
    state;
    bit_buffer = _fresh_bit_buffer(state.bit_buffer),
    last_sync_change = SYNC_UNCHANGED,
    last_num_new_soft_bits = 0,
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
    SYNC_UNCHANGED,
    0,
    complex(0.0, 0.0),
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
