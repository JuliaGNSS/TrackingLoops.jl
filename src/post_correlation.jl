# ─────────────────────────────────────────────────────────────────────────────
# Post-correlation processing of one signal component (normalise, filter, C/N₀, bit synchronizer):
# `Tracking._process_correlator_bank_output`'s arithmetic on bare state, so a loop process
# without Tracking runs the same code and produces the same integration history.
# ─────────────────────────────────────────────────────────────────────────────

"""
    SyncEvent

What one integration did to a signal's bit clock (see [`PostCorrelationState`](@ref) and
[`CoherentIntegration`](@ref)):

  - `SYNC_UNCHANGED`: neither found nor lost the bit or secondary-code sync;
  - `SYNC_FOUND`: found it;
  - `SYNC_LOST`: lost it — a post-sync integration that overshot the navigation-bit
    boundary, on which the bit synchronizer restarted its search.
"""
@enum SyncEvent::Int8 SYNC_UNCHANGED SYNC_FOUND SYNC_LOST

"""
    PostCorrelationState(signal; num_prompts_for_cn0_estimation = 100, cn0_estimator,
                    post_corr_filter)

The per-integration state of one signal component on a satellite: bit synchronizer, C/N₀
estimator, post_corr_filter, the last filtered prompt (the FLL chains from
it) and the last integration's block count. Immutable, rebuilt per integration by
[`process_correlator_bank_output!`](@ref); the estimators own their buffers, so every component needs
its own instance: one per signal of every satellite, driver and passengers alike.

It also keeps what the last integration did, for a [`CoherentIntegration`](@ref) built from it:
`last_sync_change` (a [`SyncEvent`](@ref)), `last_num_new_soft_bits` (the soft bits
it appended), `last_previous_prompt` (the prompt the FLL compares its prompt with:
the one before, or `0` across a change of carrier phase polarity) and
`last_carrier_phase_polarity` (its prompt's sign where known, else `0`; see
[`CoherentIntegration`](@ref)).
"""
struct PostCorrelationState{B<:Unsigned,BF<:AbstractPostCorrFilter,CN0<:AbstractCN0Estimator}
    bit_synchronizer::BitSynchronizer{B}
    cn0_estimator::CN0
    post_corr_filter::BF
    last_filtered_prompt::ComplexF64
    last_num_code_periods::Int
    last_sync_change::SyncEvent
    last_num_new_soft_bits::Int
    last_previous_prompt::ComplexF64
    last_carrier_phase_polarity::Int8
end

function PostCorrelationState(
    signal::AbstractGNSSSignal;
    num_prompts_for_cn0_estimation::Int = 100,
    cn0_estimator::AbstractCN0Estimator = default_cn0_estimator(
        signal,
        num_prompts_for_cn0_estimation,
    ),
    post_corr_filter::AbstractPostCorrFilter = DefaultPostCorrFilter(),
)
    bit_synchronizer = BitSynchronizer{prompt_sign_history_type(signal)}()
    # So the first soft bits after sync do not grow the vector.
    sizehint!(bit_synchronizer.soft_bits, 64)
    # Seed the sync accumulators at the detector's length so no later integration,
    # re-arm or bit-clock restart (they zero in place) allocates.
    if uses_soft_bit_edge_detection(signal)
        _seed_phase_accumulators!(bit_synchronizer.phase_accumulators, get_code_periods_per_bit(signal))
    elseif uses_soft_secondary_code_detection(signal)
        _seed_phase_accumulators!(bit_synchronizer.phase_accumulators, get_secondary_code_length(signal))
    end
    PostCorrelationState(
        bit_synchronizer,
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
    PostCorrelationState(state::PostCorrelationState; bit_synchronizer, cn0_estimator, post_corr_filter,
                    last_filtered_prompt, last_num_code_periods, last_sync_change,
                    last_num_new_soft_bits, last_previous_prompt,
                    last_carrier_phase_polarity)

`state` with the fields passed replaced, every other one kept, so a host that
swaps a part of the state (a C/N₀ estimator, say) need not list the rest.
"""
function PostCorrelationState(
    state::PostCorrelationState;
    bit_synchronizer::Maybe{BitSynchronizer} = nothing,
    cn0_estimator::Maybe{AbstractCN0Estimator} = nothing,
    post_corr_filter::Maybe{AbstractPostCorrFilter} = nothing,
    last_filtered_prompt::Maybe{Complex} = nothing,
    last_num_code_periods::Maybe{Integer} = nothing,
    last_sync_change::Maybe{SyncEvent} = nothing,
    last_num_new_soft_bits::Maybe{Integer} = nothing,
    last_previous_prompt::Maybe{Complex} = nothing,
    last_carrier_phase_polarity::Maybe{Integer} = nothing,
)
    PostCorrelationState(
        isnothing(bit_synchronizer) ? state.bit_synchronizer : bit_synchronizer,
        isnothing(cn0_estimator) ? state.cn0_estimator : cn0_estimator,
        isnothing(post_corr_filter) ? state.post_corr_filter : post_corr_filter,
        isnothing(last_filtered_prompt) ? state.last_filtered_prompt :
        ComplexF64(last_filtered_prompt),
        isnothing(last_num_code_periods) ? state.last_num_code_periods : Int(last_num_code_periods),
        isnothing(last_sync_change) ? state.last_sync_change : last_sync_change,
        isnothing(last_num_new_soft_bits) ? state.last_num_new_soft_bits :
        Int(last_num_new_soft_bits),
        isnothing(last_previous_prompt) ? state.last_previous_prompt :
        ComplexF64(last_previous_prompt),
        isnothing(last_carrier_phase_polarity) ? state.last_carrier_phase_polarity :
        Int8(last_carrier_phase_polarity),
    )
end

is_bit_or_secondary_code_synced(state::PostCorrelationState) =
    is_bit_or_secondary_code_synced(state.bit_synchronizer)
get_soft_bits(state::PostCorrelationState) = get_soft_bits(state.bit_synchronizer)
estimate_cn0(state::PostCorrelationState, coherent_integration_time) =
    estimate_cn0(state.cn0_estimator, coherent_integration_time)

# Add the integration to the C/N₀ estimator, or skip it while a required noise density
# is not measured yet. Both branches return the same type; `requires_noise_density`
# folds away at compile time.
@inline function _update_cn0_estimator(
    estimator::AbstractCN0Estimator,
    prompt,
    signal::AbstractGNSSSignal,
    bit_synchronizer::BitSynchronizer,
    num_code_periods::Integer,
    bit_sync_usable::Bool,
    noise_density,
    noise_density_ready::Bool,
    coherent_integration_time,
)
    if !noise_density_ready && requires_noise_density(estimator)
        return estimator
    end
    # Positional, not keyword: this is the per-integration site.
    update(
        estimator,
        prompt,
        CN0UpdateContext(
            signal,
            bit_synchronizer,
            num_code_periods,
            bit_sync_usable,
            noise_density,
            coherent_integration_time,
        ),
    )
end

# Whether an integration's prompt is kept out of a secondary-coded signal's coherent sum:
# one after the sync whose replica has not wiped the overlay yet (the sync changed
# the replica under it). Its code periods are credited all the same.
@inline _drops_unwiped_prompt(
    signal::AbstractGNSSSignal,
    synced_before::Bool,
    secondary_code_wiped::Bool,
) = synced_before && !secondary_code_wiped && get_secondary_code_length(signal) > 1

"""
    is_dataless_with_secondary_code(signal) -> Bool

Whether `signal` is dataless with a secondary code: GPS L1C-P and L5Q, Galileo E1C,
E5a-Q, E5b-Q and E6-C, BeiDou B1C-P and B2a-Q. Synced and correlated with the overlay
wiped, such a signal's prompt keeps the sign the sync read, so the four-quadrant
carrier discriminators apply. Its pre-sync prompts must be primary-only: a host
whose pre-sync replica carries a secondary chip uses this to pick the integrations to
multiply by that chip.
"""
is_dataless_with_secondary_code(signal::AbstractGNSSSignal) =
    iszero(get_bit_rate(signal)) && get_secondary_code_length(signal) > 1

# An integration's carrier phase polarity (see `_process_correlator_bank_output!`). `bit_synchronizer` is the one the
# integration was correlated with, before processing. The sync read its polarity off the
# overlay's sign in the primary-only pre-sync prompts, the sign the prompt keeps once
# the overlay is wiped. The pilots without a secondary code (GPS L2 CL, Galileo
# E5a-QP) give `0` only for now: their prompts share a sign, so their FLL could be
# four-quadrant from the start and their PLL could take the sign the Costas loop
# holds, though no sync resolves it.
@inline _carrier_phase_polarity(
    signal::AbstractGNSSSignal,
    bit_synchronizer::BitSynchronizer,
    secondary_code_wiped::Bool,
) =
    is_dataless_with_secondary_code(signal) && secondary_code_wiped ? bit_synchronizer.polarity :
    Int8(0)

"""
    _process_correlator_bank_output!(signal, prn, bit_synchronizer, cn0_estimator, post_corr_filter, output,
                sampling_frequency, noise_density, noise_density_ready,
                driver_carrier_phase_offset;
                secondary_code_wiped = is_bit_or_secondary_code_synced(bit_synchronizer))
        -> (bit_synchronizer, cn0_estimator, post_corr_filter, prompt, filtered_correlator_bank,
            bit_block_count, integrated_code_blocks, overshoot, carrier_phase_polarity)

Apply one completed integration to a signal component's bare state: normalize the raw
correlator bank by sample count and code amplitude, update and apply the
post-correlation filter, advance the C/N₀ estimator and the bit synchronizer.

The bit synchronizer is credited with the code periods actually integrated
(`calc_num_bit_synchronizer_code_periods`), recovered from the sample count. Everything
here is Tracking's per-integration arithmetic, in the same order.
`carrier_phase_polarity` is the sign of the integration's prompt where the replica wiped
every sign modulation off it (a dataless signal with a secondary code, overlay
wiped), else `0`; a [`CoherentIntegration`](@ref) carries it to the carrier discriminators.

`overshoot`, `secondary_code_wiped` and the pre-sync contract are as for
[`process_correlator_bank_output!`](@ref).
"""
@inline function _process_correlator_bank_output!(
    signal::AbstractGNSSSignal,
    prn::Integer,
    bit_synchronizer::BitSynchronizer,
    cn0_estimator::AbstractCN0Estimator,
    post_corr_filter::AbstractPostCorrFilter,
    output::CorrelatorBankOutput,
    sampling_frequency,
    noise_density,
    noise_density_ready::Bool,
    driver_carrier_phase_offset::Real;
    secondary_code_wiped::Bool = is_bit_or_secondary_code_synced(bit_synchronizer),
)
    bit_synchronizer_before = bit_synchronizer
    synced_before = is_bit_or_secondary_code_synced(bit_synchronizer)
    normalized_correlator_bank =
        normalize(output.correlator_bank, output.num_integrated_samples, get_code_amplitude(signal))
    post_corr_filter = update(post_corr_filter, get_prompt(normalized_correlator_bank))
    # Used twice: to combine the antennas and to reduce the shared noise covariance
    # to this satellite's density.
    weights = get_weights(post_corr_filter, NumAnts(normalized_correlator_bank))
    filtered_correlator_bank = _filter_correlator_bank(normalized_correlator_bank, weights)
    prompt = get_prompt(filtered_correlator_bank)
    bit_block_count = calc_num_bit_synchronizer_code_periods(
        signal,
        output.num_integrated_samples,
        sampling_frequency,
        synced_before,
    )
    # Floor at 1 for the fractional-block integration right after a sync phase snap.
    integrated_code_blocks = max(1, bit_block_count)
    # Onto the driver's (real) phase frame for sync search and bit accumulation.
    bit_prompt = prompt * _carrier_phase_derotation(driver_carrier_phase_offset, signal)
    drop_prompt = _drops_unwiped_prompt(signal, synced_before, secondary_code_wiped)
    scalar_noise_density = _reduce_noise_density(noise_density, weights)
    cn0_estimator = _update_cn0_estimator(
        cn0_estimator,
        prompt,
        signal,
        bit_synchronizer,
        bit_block_count,
        !drop_prompt,
        scalar_noise_density,
        noise_density_ready,
        output.num_integrated_samples / sampling_frequency,
    )
    bit_synchronizer = buffer(
        signal,
        prn,
        bit_synchronizer,
        bit_block_count,
        drop_prompt ? zero(bit_prompt) : bit_prompt,
    )
    # An unwiped post-sync integration moves the secondary-code anchor: the code-phase
    # snap after this processing aligns the *upcoming* integration to `bit_synchronizer.secondary_code_phase`.
    if synced_before && !secondary_code_wiped
        bit_synchronizer = _advance_secondary_phase(signal, bit_synchronizer, bit_block_count)
    end
    overshoot = is_bit_or_secondary_code_synced(bit_synchronizer_before) &&
        !is_bit_or_secondary_code_synced(bit_synchronizer)
    return bit_synchronizer,
    cn0_estimator,
    post_corr_filter,
    prompt,
    filtered_correlator_bank,
    bit_block_count,
    integrated_code_blocks,
    overshoot,
    _carrier_phase_polarity(signal, bit_synchronizer_before, secondary_code_wiped)
end

"""
    process_correlator_bank_output!(state::PostCorrelationState, signal, prn, output, sampling_frequency,
                 noise_density, noise_density_ready,
                 driver_carrier_phase_offset = get_carrier_phase_offset(signal);
                 secondary_code_wiped = is_bit_or_secondary_code_synced(state))
        -> (state, prompt, filtered_correlator_bank, integrated_code_blocks, overshoot)

`_process_correlator_bank_output!` on a [`PostCorrelationState`](@ref). Returns the new state, the prompt,
the filtered correlator bank the discriminators read, the blocks the integration covered and
`overshoot`: whether a post-sync integration carried the bit accumulator past the bit
boundary, so the buffer dropped sync and restarted its search (`SYNC_LOST`;
JuliaGNSS/Tracking.jl#238). The new state keeps the filtered prompt and what the
integration did, for a [`CoherentIntegration`](@ref) built from it. Nothing here logs; the caller
reports `overshoot` (Tracking.jl warns once per satellite).

`driver_carrier_phase_offset` is the carrier-phase offset (cycles, `get_carrier_phase_offset`) of
the satellite's loop-driving signal; the bit synchronizer's prompt is de-rotated against
it. The default (the signal's own) suits the driver; a passenger component (the data
half of a pilot/data pair) must be handed the driver's.

The pre-sync prompts of an [`is_dataless_with_secondary_code`](@ref) signal must be
primary-only, so that the sync's polarity is the prompt's sign once the overlay is
wiped. For any other signal the sync polarity only fixes the soft bits' sign, which
the Costas loop leaves ambiguous anyway, so a fixed secondary chip in the pre-sync
replica is fine.

`secondary_code_wiped` says whether the integration's replica wiped the secondary code at the
synced phase; it defaults to the bit synchronizer having synced before the integration. Pass
`false` for an integration after a sync it was not correlated for yet (e.g. one found
earlier in the same batch): for a secondary-coded signal its prompt is kept out of
the coherent sum, though its code periods are credited, it moves the secondary-code anchor
and its polarity is `0`.
"""
@inline function process_correlator_bank_output!(
    state::PostCorrelationState,
    signal::AbstractGNSSSignal,
    prn::Integer,
    output::CorrelatorBankOutput,
    sampling_frequency,
    noise_density,
    noise_density_ready::Bool,
    driver_carrier_phase_offset::Real = get_carrier_phase_offset(signal);
    secondary_code_wiped::Bool = is_bit_or_secondary_code_synced(state.bit_synchronizer),
)
    was_synced = is_bit_or_secondary_code_synced(state.bit_synchronizer)
    num_soft_bits = Base.length(state.bit_synchronizer.soft_bits)
    bit_synchronizer,
    cn0_estimator,
    post_corr_filter,
    prompt,
    filtered_correlator_bank,
    _,
    integrated_code_blocks,
    overshoot,
    carrier_phase_polarity = _process_correlator_bank_output!(
        signal,
        prn,
        state.bit_synchronizer,
        state.cn0_estimator,
        state.post_corr_filter,
        output,
        sampling_frequency,
        noise_density,
        noise_density_ready,
        driver_carrier_phase_offset;
        secondary_code_wiped,
    )
    is_synced = is_bit_or_secondary_code_synced(bit_synchronizer)
    sync_change =
        !was_synced && is_synced ? SYNC_FOUND :
        was_synced && !is_synced ? SYNC_LOST : SYNC_UNCHANGED
    new_state = PostCorrelationState(
        bit_synchronizer,
        cn0_estimator,
        post_corr_filter,
        prompt,
        integrated_code_blocks,
        sync_change,
        Base.length(bit_synchronizer.soft_bits) - num_soft_bits,
        # The FLL cannot compare prompts of different carrier phase polarities.
        carrier_phase_polarity != state.last_carrier_phase_polarity ? complex(0.0, 0.0) :
        state.last_filtered_prompt,
        carrier_phase_polarity,
    )
    new_state, prompt, filtered_correlator_bank, integrated_code_blocks, overshoot
end

"""
    reset_sync!(state::PostCorrelationState) -> PostCorrelationState

The state with a fresh, unsynchronised bit synchronizer (what a lost integration costs),
everything else kept.
"""
reset_sync!(state::PostCorrelationState) = PostCorrelationState(
    state;
    bit_synchronizer = _fresh_bit_synchronizer(state.bit_synchronizer),
    last_sync_change = SYNC_UNCHANGED,
    last_num_new_soft_bits = 0,
)

# An unsynchronised bit synchronizer reusing `bit_synchronizer`'s vectors, emptied in place.
function _fresh_bit_synchronizer(bit_synchronizer::BitSynchronizer{B}) where {B}
    empty!(bit_synchronizer.soft_bits)
    BitSynchronizer{B}(
        zero(B),
        0,
        false,
        0,
        Int8(0),
        complex(0.0, 0.0),
        0,
        bit_synchronizer.soft_bits,
        _reset_phase_accumulators!(bit_synchronizer.phase_accumulators),
    )
end

"""
    reset_post_correlation!(state::PostCorrelationState) -> PostCorrelationState

The state a freshly armed channel starts from (no sync, no soft bits, empty C/N₀
estimator, no previous prompt), reusing the old vectors so a re-arm allocates
nothing. The post-correlation filter is kept: it is configuration, not history;
an adaptive beamformer must be reset by its owner.

A custom [`AbstractCN0Estimator`](@ref) with history needs a
`TrackingLoops.reset_cn0_estimator!` method; without one this throws rather than
hand the new satellite the old one's C/N₀.
"""
reset_post_correlation!(state::PostCorrelationState) = PostCorrelationState(
    _fresh_bit_synchronizer(state.bit_synchronizer),
    reset_cn0_estimator!(state.cn0_estimator),
    state.post_corr_filter,
    complex(0.0, 0.0),
    1,
    SYNC_UNCHANGED,
    0,
    complex(0.0, 0.0),
    Int8(0),
)

reset_cn0_estimator!(e::NoiseReferencedCN0Estimator) =
    (fill!(e.buffered_cn0, 0.0); NoiseReferencedCN0Estimator(e.num_integrations, e.buffered_cn0, 0, 0))
reset_cn0_estimator!(e::MomentsCN0Estimator) =
    (fill!(e.prompt_buffer, zero(ComplexF64)); MomentsCN0Estimator(e.prompt_buffer, 0, 0))
function reset_cn0_estimator!(e::NWPRCN0Estimator)
    fill!(e.buffered_narrowband_powers, 0.0)
    fill!(e.buffered_wideband_powers, 0.0)
    _with_window_state(
        e,
        reset_cn0_estimator!(e.fallback);
        ratio_current_index = 0,
        filled_ratio_length = 0,
        num_integrations_per_ratio = 0,
        ratios_are_bit_aligned = false,
    )
end
reset_cn0_estimator!(e::NoCN0Estimator) = e
reset_cn0_estimator!(e::AbstractCN0Estimator) = throw(
    ArgumentError(
        "this C/N₀ estimator has no `TrackingLoops.reset_cn0_estimator!` " *
        "method; add one that empties its history",
    ),
)
