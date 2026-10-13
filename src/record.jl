# ─────────────────────────────────────────────────────────────────────────────
# The per-record fold of one signal component: normalise the record, filter the
# prompt, advance the C/N₀ estimator and the bit buffer. This is the arithmetic
# `Tracking._apply_correlator_output` runs on a `TrackedSignal`, lifted onto the
# component's bare state so a loop process without Tracking runs the same code
# — and so the record history the receiver mirrors is identical on both paths.
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
    SignalLoopState(signal; num_prompts_for_cn0_estimation = 100, cn0_estimator, post_corr_filter)

The per-record state of one signal component on a satellite: its bit buffer,
C/N₀ estimator, post-correlation filter, the last filtered prompt (which the
FLL discriminator chains from) and the block count of the last record. An
immutable value, rebuilt by [`apply_record`](@ref) per record; the estimators
buffer into vectors they own, so every component needs its own instance.

The host holds one per signal of every satellite — the driver and each
passenger alike — and advances it with [`apply_record`](@ref) on every record.
It also remembers what the last record did to the bit clock, which a
[`LoopRecord`](@ref) built from the state hands to the estimator:
`last_sync_change` (a [`SyncChange`](@ref)) and `last_num_new_soft_bits`, the
soft bits that record appended; `last_previous_prompt`, the prompt the FLL
compares that record's prompt with: the one before it, or `0` where the two
records have another carrier phase polarity (see [`LoopRecord`](@ref)); and
`last_carrier_phase_polarity`, the sign of that record's prompt where the
replica wiped every sign modulation off it, `0` where it is unknown, which the
four-quadrant carrier discriminators read it with.
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
        SYNC_UNCHANGED,
        0,
        complex(0.0, 0.0),
        Int8(0),
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

# Keep the prompt of a record after the sync whose replica has not wiped the
# overlay yet out of a secondary-coded signal's coherent sum, as the sync changed
# the replica under it, but always let it advance the accumulator's block count.
@inline _drops_unwiped_prompt(
    signal::AbstractGNSSSignal,
    synced_before::Bool,
    overlay_wiped::Bool,
) = synced_before && !overlay_wiped && get_secondary_code_length(signal) > 1

"""
    is_dataless_with_secondary_code(signal) -> Bool

Whether `signal` is a dataless signal with a secondary code: GPS L1C-P and L5Q,
Galileo E1C, E5a-Q, E5b-Q and E6-C, BeiDou B1C-P and B2a-Q. Once synced to its
secondary code and correlated with the overlay wiped, such a signal's prompt
keeps the sign the sync read, which the four-quadrant carrier discriminators
read it with (see `fold_record`'s carrier phase polarity). These are the signals
whose pre-sync prompts must be primary-only, so a host that multiplies its
pre-sync records by a secondary chip asks this to pick them.
"""
is_dataless_with_secondary_code(signal::AbstractGNSSSignal) =
    iszero(get_data_frequency(signal)) && get_secondary_code_length(signal) > 1

# The carrier phase polarity of a record: the sign of its prompt, which the
# four-quadrant carrier discriminators read it with (`pll_disc(...; polarity)`,
# `fll_disc(...; polarity)`), or `0` where it is unknown and they stay
# two-quadrant. It is known where the replica wiped every sign modulation off the
# prompt: a dataless signal with a secondary code, its overlay wiped at the synced
# phase (`overlay_wiped`). `bit_buffer` is the one the record was correlated with,
# before the record is folded into it. For these signals pre-sync prompts are
# primary-only, so the sync reads its polarity off the overlay's sign in them,
# which is the sign the prompt keeps once the replica wipes the overlay. For any
# other signal the sync polarity only fixes the sign of the soft bits, which the
# Costas loop leaves ambiguous anyway. If the Costas loop slipped half a cycle
# between the sync and the switch, the four-quadrant PLL pulls the carrier phase
# over by half a cycle, after which the prompt agrees with the sign again.
#
# Data signals give `0`, and so, for now, do the pilots without a secondary code
# (GPS L2 CL, Galileo E5a-QP). That is not a given restriction, only not yet
# implemented: their consecutive prompts share their sign, so their FLL could be
# four-quadrant from the start, and their PLL with the sign the Costas loop holds,
# though no sync resolves that sign.
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
                driver_carrier_phase;
                overlay_wiped = has_bit_or_secondary_code_been_found(bit_buffer))
        -> (bit_buffer, cn0_estimator, post_corr_filter, prompt, filtered_correlator,
            bit_block_count, integrated_code_blocks, overshoot, carrier_phase_polarity)

Apply one completed record to the bare per-record state of a signal component:
normalize the record's raw correlator by its sample count and code amplitude,
update and apply the post-correlation filter, advance the C/N₀ estimator and the
bit buffer. Returns the new state values plus the filtered correlator and block
counts the loop-filter step needs.

The bit accumulator is credited with the blocks *actually* integrated
(`calc_num_code_blocks_for_bit_buffer`), recovered from the record's sample
count. `overshoot` reports a post-sync record that carried the bit
accumulator past the navigation-bit boundary, on which the bit buffer dropped
sync and restarted its search (issue #238). `carrier_phase_polarity` is the
sign of the record's prompt where the replica wiped every sign modulation off it
— a dataless signal with a secondary code, its overlay wiped — and `0` where it
is unknown; a [`LoopRecord`](@ref) carries it to the carrier discriminators.
Everything here is the arithmetic Tracking's per-record advance performs, in the
same order.

For a dataless signal with a secondary code
([`is_dataless_with_secondary_code`](@ref): GPS L1C-P and L5Q, Galileo E1C,
E5a-Q, E5b-Q and E6-C, BeiDou B1C-P and B2a-Q), pre-sync prompts are
primary-only: correlated with a replica that carries no secondary code, so
that the sync's polarity is the sign of the prompt once the overlay is wiped,
which the carrier phase polarity takes. A host whose pre-sync replica carries
a secondary chip on every block multiplies such a signal's pre-sync records by
that chip. For any other signal the sync polarity only fixes the sign of the
soft bits, which the Costas loop leaves ambiguous anyway, so a pre-sync
replica that carries a fixed secondary chip is fine. `overlay_wiped` says
whether the replica under this record wiped the secondary code at the synced
phase; it defaults to the bit buffer having synced before the record. A record
after the sync whose replica did not wipe the overlay yet, e.g. one that
follows a sync detected earlier in the same fold, passes `overlay_wiped =
false`: its prompt is kept out of the coherent bit accumulation of a
secondary-coded signal, though its blocks are credited, it moves the
secondary-code anchor along, and its carrier phase polarity is `0`. For a
signal without a secondary code it makes no difference.
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
    driver_carrier_phase::Real;
    overlay_wiped::Bool = has_bit_or_secondary_code_been_found(bit_buffer),
)
    bit_buffer_before = bit_buffer
    synced_before = has_bit_or_secondary_code_been_found(bit_buffer)
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
        synced_before,
    )
    # Blocks this record actually covered. The floor at 1 covers the
    # fractional-block record right after a sync phase-snap accumulator reset.
    integrated_code_blocks = max(1, bit_block_count)
    # De-rotate the prompt onto the driver's (real) phase frame before both the
    # sync search and the coherent bit accumulation.
    bit_prompt = prompt * _carrier_phase_derotation(driver_carrier_phase, signal)
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
    if synced_before && !overlay_wiped
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
    overshoot,
    _carrier_phase_polarity(signal, bit_buffer_before, overlay_wiped)
end

"""
    apply_record(state::SignalLoopState, signal, prn, output, sampling_frequency,
                 noise_density, noise_density_ready,
                 driver_carrier_phase = get_carrier_phase_offset(signal);
                 overlay_wiped = has_bit_or_secondary_code_been_found(state))
        -> (state, prompt, filtered_correlator, integrated_code_blocks, overshoot)

`fold_record` on a [`SignalLoopState`](@ref): the new state (with the filtered
prompt as its `last_filtered_prompt`), the prompt, the filtered correlator the
discriminators read, the blocks the record covered, and whether the record
overshot the navigation-bit boundary — on which the bit buffer dropped sync and
restarted its search. The new state also records what the record did to the bit
clock (`last_sync_change`) and how many soft bits it appended
(`last_num_new_soft_bits`), the prompt the FLL compares its prompt with
(`last_previous_prompt`) and its prompt's sign (`last_carrier_phase_polarity`),
for a [`LoopRecord`](@ref) built from it.
Nothing here logs; report
`overshoot` the way the caller reports things (Tracking.jl warns once per
satellite).

`driver_carrier_phase` is the carrier-phase offset of the satellite's
estimator-driver signal (`get_carrier_phase_offset`), against which this
component's bit-buffer prompt is de-rotated. It defaults to the signal's own
offset — a no-op, right for the driver itself; a passenger component (the data
half of a pilot/data pair) must be handed the driver's. `overlay_wiped` is as
for `fold_record`, and so is the convention for pre-sync prompts: primary-only
for a dataless signal with a secondary code.
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
        driver_carrier_phase;
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

The state with a fresh, unsynchronised bit buffer — what a lost record costs a
satellite (its bit clock is rebuilt from the signal), everything else kept.
"""
restart_bit_clock(state::SignalLoopState) = SignalLoopState(
    _fresh_bit_buffer(state.bit_buffer),
    state.cn0_estimator,
    state.post_corr_filter,
    state.last_filtered_prompt,
    state.last_num_code_blocks,
    SYNC_UNCHANGED,
    0,
    state.last_previous_prompt,
    state.last_carrier_phase_polarity,
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
