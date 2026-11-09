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
    PostCorrelationState(signal; cn0_estimator, post_corr_filter)

Persistent processing history for one satellite's signal: bit synchronization,
C/N₀, post-correlation filter state, previous prompt and polarity, and the latest integration
period count. Every component owns its buffers. [`process_correlator_bank_output!`](@ref) returns
its updated state together with the completed [`CoherentIntegration`](@ref).
"""
struct PostCorrelationState{B<:Unsigned,BF<:AbstractPostCorrFilter,CN0<:AbstractCN0Estimator}
    bit_synchronizer::BitSynchronizer{B}
    cn0_estimator::CN0
    post_corr_filter::BF
    last_filtered_prompt::ComplexF64
    last_num_code_periods::Int
    last_carrier_phase_polarity::Int8
end

function PostCorrelationState(
    signal::AbstractGNSSSignal;
    cn0_estimator::AbstractCN0Estimator = default_cn0_estimator(signal, 100),
    post_corr_filter::AbstractPostCorrFilter = DefaultPostCorrFilter(),
)
    bit_synchronizer = BitSynchronizer{prompt_sign_history_type(signal)}()
    # So the first soft bits after sync do not grow the vector.
    sizehint!(bit_synchronizer.soft_bits, 64)
    # Seed the sync accumulators at the detector's length so no later integration,
    # re-initialisation or bit-clock restart (they zero in place) allocates.
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
        Int8(0),
    )
end

"""
    PostCorrelationState(state::PostCorrelationState; bit_synchronizer, cn0_estimator, post_corr_filter,
                    last_filtered_prompt, last_num_code_periods, last_carrier_phase_polarity)

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
    last_carrier_phase_polarity::Maybe{Integer} = nothing,
)
    PostCorrelationState(
        isnothing(bit_synchronizer) ? state.bit_synchronizer : bit_synchronizer,
        isnothing(cn0_estimator) ? state.cn0_estimator : cn0_estimator,
        isnothing(post_corr_filter) ? state.post_corr_filter : post_corr_filter,
        isnothing(last_filtered_prompt) ? state.last_filtered_prompt :
        ComplexF64(last_filtered_prompt),
        isnothing(last_num_code_periods) ? state.last_num_code_periods : Int(last_num_code_periods),
        isnothing(last_carrier_phase_polarity) ? state.last_carrier_phase_polarity :
        Int8(last_carrier_phase_polarity),
    )
end

is_bit_or_secondary_code_synced(state::PostCorrelationState) =
    is_bit_or_secondary_code_synced(state.bit_synchronizer)
get_soft_bits(state::PostCorrelationState) = get_soft_bits(state.bit_synchronizer)
drain_soft_bits!(state::PostCorrelationState) = (drain_soft_bits!(state.bit_synchronizer); state)

"""
    get_secondary_code_phase(state::PostCorrelationState) -> Int

The secondary-code chip of the code period after the latest coherent integration
the bit synchronizer advanced over: found at sync, then moved along by every integration
not passed as `secondary_code_wiped` (see [`process_correlator_bank_output!`](@ref)). `0`
before sync and for a signal without a secondary code.
"""
get_secondary_code_phase(state::PostCorrelationState) = state.bit_synchronizer.secondary_code_phase

"""
    get_num_code_periods_into_bit(state::PostCorrelationState) -> Int

The code periods integrated into the current navigation bit since its edge, once
[`is_bit_or_secondary_code_synced`](@ref); `0` before sync. A host that sizes its next
coherent integration so it does not cross a bit edge reads it.
"""
get_num_code_periods_into_bit(state::PostCorrelationState) =
    state.bit_synchronizer.num_code_periods_into_bit

"""
    get_last_num_code_periods(state::PostCorrelationState) -> Int

The code periods the latest coherent integration spanned, which its coherent
integration time and the C/N₀ estimate follow.
"""
get_last_num_code_periods(state::PostCorrelationState) = state.last_num_code_periods

"""
    get_bit_polarity(state::PostCorrelationState) -> Int8

The sign the bit synchronizer found the bits at once
[`is_bit_or_secondary_code_synced`](@ref): `-1` if they come inverted, `+1` if not, `0`
before sync.
"""
get_bit_polarity(state::PostCorrelationState) = state.bit_synchronizer.bit_polarity

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
    update!(
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
    is_dataless_with_secondary_code(signal) && secondary_code_wiped ? bit_synchronizer.bit_polarity :
    Int8(0)

"""
    _process_correlator_bank_output!(state::PostCorrelationState, signal, prn, output,
                sampling_frequency, noise_density, noise_density_ready,
                driver_carrier_phase_offset;
                secondary_code_wiped = is_bit_or_secondary_code_synced(state))
        -> (state, filtered_correlator_bank)

Apply one completed integration to a signal component's state: normalize the raw
correlator bank by sample count and code amplitude, update and apply the
post-correlation filter, advance the C/N₀ estimator and the bit synchronizer. Returns
the new state, which keeps the prompt, the code periods and the carrier phase
polarity, and the filtered correlator bank the discriminators read.

The bit synchronizer is credited with the code periods actually integrated
(`calc_num_bit_synchronizer_code_periods`), recovered from the sample count. Everything
here is Tracking's per-integration arithmetic, in the same order.
`carrier_phase_polarity` is the sign of the integration's prompt where the replica wiped
every sign modulation off it (a dataless signal with a secondary code, overlay
wiped), else `0`; a [`CoherentIntegration`](@ref) carries it to the carrier discriminators.

`secondary_code_wiped` and the pre-sync contract are as for
[`process_correlator_bank_output!`](@ref).
"""
@inline function _process_correlator_bank_output!(
    state::PostCorrelationState,
    signal::AbstractGNSSSignal,
    prn::Integer,
    output::CorrelatorBankOutput,
    sampling_frequency,
    noise_density,
    noise_density_ready::Bool,
    driver_carrier_phase_offset::Real;
    secondary_code_wiped::Bool = is_bit_or_secondary_code_synced(state),
)
    bit_synchronizer_before = state.bit_synchronizer
    synced_before = is_bit_or_secondary_code_synced(bit_synchronizer_before)
    # Normalize, then filter: the antennas combined into one correlator bank.
    normalized_correlator_bank =
        normalize(output.correlator_bank, output.num_integrated_samples, get_code_amplitude(signal))
    post_corr_filter = update!(state.post_corr_filter, get_prompt(normalized_correlator_bank))
    # Used twice: to combine the antennas and to reduce the shared noise covariance
    # to this satellite's density.
    weights = get_weights(post_corr_filter, NumAnts(normalized_correlator_bank))
    filtered_correlator_bank = _filter_correlator_bank(normalized_correlator_bank, weights)
    prompt = get_prompt(filtered_correlator_bank)
    # The code periods integrated, recovered from the sample count.
    num_bit_synchronizer_code_periods = calc_num_bit_synchronizer_code_periods(
        signal,
        output.num_integrated_samples,
        sampling_frequency,
        synced_before,
    )
    # Floor at 1 for the fractional-period integration right after a sync phase snap.
    num_code_periods = max(1, num_bit_synchronizer_code_periods)
    # Onto the driver's (real) phase frame for sync search and bit accumulation.
    bit_prompt = prompt * _carrier_phase_derotation(driver_carrier_phase_offset, signal)
    drop_prompt = _drops_unwiped_prompt(signal, synced_before, secondary_code_wiped)
    # C/N₀ against this satellite's share of the noise density.
    scalar_noise_density = _reduce_noise_density(noise_density, weights)
    cn0_estimator = _update_cn0_estimator(
        state.cn0_estimator,
        prompt,
        signal,
        bit_synchronizer_before,
        num_bit_synchronizer_code_periods,
        !drop_prompt,
        scalar_noise_density,
        noise_density_ready,
        output.num_integrated_samples / sampling_frequency,
    )
    # Bit synchronization: the sync search, or the bit accumulation once synced.
    bit_synchronizer = update_bit_sync!(
        signal,
        prn,
        bit_synchronizer_before,
        num_bit_synchronizer_code_periods,
        drop_prompt ? zero(bit_prompt) : bit_prompt,
    )
    # An unwiped post-sync integration moves the secondary-code anchor: the code-phase
    # snap after this processing aligns the *upcoming* integration to `bit_synchronizer.secondary_code_phase`.
    if synced_before && !secondary_code_wiped
        bit_synchronizer = _advance_secondary_phase(signal, bit_synchronizer, num_bit_synchronizer_code_periods)
    end
    new_state = PostCorrelationState(
        bit_synchronizer, cn0_estimator, post_corr_filter, prompt, num_code_periods,
        _carrier_phase_polarity(signal, bit_synchronizer_before, secondary_code_wiped),
    )
    new_state, filtered_correlator_bank
end

"""
    process_correlator_bank_output!(state::PostCorrelationState, signal, prn, output, sampling_frequency,
                 noise_density, noise_density_ready,
                 driver_carrier_phase_offset = get_carrier_phase_offset(signal);
                 secondary_code_wiped = is_bit_or_secondary_code_synced(state))
        -> (state, integration)

Process one correlator bank output and return the persistent state and completed
[`CoherentIntegration`](@ref). The integration carries synchronization events, integration timing,
C/N₀, and exactly the newly demodulated soft bits. Its `new_soft_bits` borrows the
state's bit vector and remains valid until that vector is drained, reset, or grown;
consume the integration before modifying the soft-bit vector. Processing and resetting reuse
allocated storage. A carrier polarity change sets `integration.previous_prompt` to zero.
`command_issue_sample_index` defaults to the output's sample index;
`sample_index_offset` shifts both indices onto the receiver's shared time grid.

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
    command_issue_sample_index::Integer = output.sample_index,
    sample_index_offset::Integer = 0,
)
    was_synced = is_bit_or_secondary_code_synced(state.bit_synchronizer)
    num_soft_bits = Base.length(state.bit_synchronizer.soft_bits)
    new_state, filtered_correlator_bank = _process_correlator_bank_output!(
        state,
        signal,
        prn,
        output,
        sampling_frequency,
        noise_density,
        noise_density_ready,
        driver_carrier_phase_offset;
        secondary_code_wiped,
    )
    (; bit_synchronizer, cn0_estimator) = new_state
    num_code_periods = new_state.last_num_code_periods
    carrier_phase_polarity = new_state.last_carrier_phase_polarity
    is_synced = is_bit_or_secondary_code_synced(bit_synchronizer)
    sync_change =
        !was_synced && is_synced ? SYNC_FOUND :
        was_synced && !is_synced ? SYNC_LOST : SYNC_UNCHANGED
    # The FLL cannot compare prompts of different carrier phase polarities.
    previous_prompt = carrier_phase_polarity == state.last_carrier_phase_polarity ?
        state.last_filtered_prompt : zero(ComplexF64)
    integration = CoherentIntegration(
        signal, filtered_correlator_bank, previous_prompt,
        output.num_integrated_samples,
        output.sample_index + Int(sample_index_offset),
        Int(command_issue_sample_index) + Int(sample_index_offset),
        num_code_periods, sampling_frequency, Int(prn), output.code_phase,
        is_synced, sync_change,
        is_synced ? bit_synchronizer.num_code_periods_into_bit : 0,
        view(bit_synchronizer.soft_bits, (num_soft_bits + 1):Base.length(bit_synchronizer.soft_bits)),
        cn0_estimator, carrier_phase_polarity,
        is_synced ? bit_synchronizer.bit_polarity : Int8(0),
    )
    new_state, integration
end

"""
    reset_sync!(state::PostCorrelationState) -> PostCorrelationState

The state with a fresh, unsynchronised bit synchronizer (what a lost integration costs),
everything else kept.
"""
reset_sync!(state::PostCorrelationState) = PostCorrelationState(
    state;
    bit_synchronizer = _fresh_bit_synchronizer(state.bit_synchronizer),
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

The state a channel starts tracking from (no sync, no soft bits, empty C/N₀
estimator, no previous prompt), reusing the old vectors so a re-initialisation
allocates nothing. The post-correlation filter is kept: it is configuration, not history;
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
        last_write_index = 0,
        num_filled = 0,
        num_integrations_per_ratio = 0,
        ratios_are_bit_aligned = false,
    )
end
reset_cn0_estimator!(e::NoCN0Estimator) = e
"""Clear estimator history in existing storage and return the reset estimator. Required for custom C/N₀ estimators with history."""
reset_cn0_estimator!(e::AbstractCN0Estimator) = throw(
    ArgumentError(
        "this C/N₀ estimator has no `TrackingLoops.reset_cn0_estimator!` " *
        "method; add one that empties its history",
    ),
)
