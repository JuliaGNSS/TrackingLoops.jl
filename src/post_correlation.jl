# ─────────────────────────────────────────────────────────────────────────────
# Post-correlation processing of one signal component (normalise, filter, C/N₀, bit synchronizer):
# `Tracking._process_correlator_bank_output`'s arithmetic on bare state, so a loop process
# without Tracking runs the same code and produces the same integration history.
# ─────────────────────────────────────────────────────────────────────────────

"""
    SyncEvent

What the latest integration did to a signal's bit clock (see [`PostCorrelationState`](@ref)):

  - `SYNC_UNCHANGED`: neither found nor lost the bit or secondary-code sync;
  - `SYNC_FOUND`: found it;
  - `SYNC_LOST`: lost it — a post-sync integration that overshot the navigation-bit
    boundary, on which the bit synchronizer restarted its search.
"""
@enum SyncEvent::Int8 SYNC_UNCHANGED SYNC_FOUND SYNC_LOST

"""
    PostCorrelationState(signal, prn; driver_carrier_phase_offset = get_carrier_phase_offset(signal),
                         correlator_bank = get_default_correlator_bank(signal), cn0_estimator,
                         post_corr_filter)

The post-correlation processing of one channel, a signal of one satellite, after its
latest coherent integration (Kaplan and Hegarty's predetection integration):
[`process_correlator_bank_output!`](@ref) returns it updated, and a tracking loop steps on
it ([`step_loop!`](@ref)).

It holds the channel's signal and PRN, the carrier-phase offset (cycles,
`get_carrier_phase_offset`) of the satellite's loop-driving signal, the persistent
history (bit synchronizer, C/N₀ estimator, post-correlation filter), and the latest
integration:

  - `last_filtered_correlator_bank`: the integration's correlator bank, normalised and
    with the antennas combined by the post-correlation filter, which the discriminators
    read. Its type is fixed at construction from `correlator_bank`, the bank the host
    correlates with, so the state's type never changes.
  - `previous_filtered_prompt`: the prompt before it, which the FLL compares with;
    `0` for the first integration and across a change of `last_carrier_phase_polarity`,
    since the four-quadrant FLL would read a sign flip as half a cycle of rotation.
  - `last_num_integrated_samples`, `last_num_code_periods`: its span. `last_num_code_periods`
    is at least 1: the carrier-loop staging sizes its phase-lock indicator's averages in
    integrations of that length and divides by it.
  - `last_sample_index`: the device sample index at its end, on the band's
    `sampling_frequency`. For a tracking loop with shared navigation state
    ([`VectorTrackingLoop`](@ref)) `last_sample_index / sampling_frequency` must be the time
    since an origin shared by every satellite of the band; a host whose correlator
    restarts its sample count passes the offset as `sample_index_offset`.
  - `last_command_issue_sample_index`: where the loop issues its command, the
    `last_sample_index` of the last integration of its batch (the integrations a host
    steps before its next command takes effect), from which the command's delay to its
    application sample is measured.
  - `last_code_phase`: the replica's code phase (chips) at its end, from the
    [`CorrelatorBankOutput`](@ref) (`NaN` if not reported).
  - `last_sync_change`: whether it found or lost the bit or secondary-code sync
    ([`SyncEvent`](@ref)).
  - `last_num_new_soft_bits`: how many soft bits it appended.
  - `last_carrier_phase_polarity`: its prompt's sign where the replica wiped every sign
    modulation off it (a dataless signal synced to its secondary code), else `0`.
    Nonzero, the PLL and the FLL read the integration four-quadrant (`pll_disc(...;
    polarity)`, `fll_disc(...; polarity)`); `0`, two-quadrant.

The bit synchronizer's prompt is de-rotated against `driver_carrier_phase_offset`. The
default (the signal's own) suits the driver; a passenger component (the data half of a
pilot/data pair) must be given the driver's. Every component owns its buffers.
"""
struct PostCorrelationState{
    S<:AbstractGNSSSignal,
    C<:AbstractCorrelatorBank,
    B<:Unsigned,
    BF<:AbstractPostCorrFilter,
    CN0<:AbstractCN0Estimator,
}
    signal::S
    prn::Int
    driver_carrier_phase_offset::Float64
    sampling_frequency::typeof(1.0Hz)
    bit_synchronizer::BitSynchronizer{B}
    cn0_estimator::CN0
    post_corr_filter::BF
    last_filtered_correlator_bank::C
    previous_filtered_prompt::ComplexF64
    last_num_integrated_samples::Int
    last_sample_index::Int
    last_command_issue_sample_index::Int
    last_num_code_periods::Int
    last_code_phase::Float64
    last_sync_change::SyncEvent
    last_num_new_soft_bits::Int
    last_carrier_phase_polarity::Int8
end

# `correlator_bank` zeroed and filtered: the latest integration's bank before there is one,
# of the type every later one has.
_zero_filtered_correlator_bank(correlator_bank::AbstractCorrelatorBank, post_corr_filter) =
    _filter_correlator_bank(zero(correlator_bank), get_weights(post_corr_filter, NumAnts(correlator_bank)))

function PostCorrelationState(
    signal::AbstractGNSSSignal,
    prn::Integer;
    driver_carrier_phase_offset::Real = get_carrier_phase_offset(signal),
    correlator_bank::AbstractCorrelatorBank = get_default_correlator_bank(signal),
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
        signal,
        Int(prn),
        Float64(driver_carrier_phase_offset),
        NaN * Hz,
        bit_synchronizer,
        cn0_estimator,
        post_corr_filter,
        _zero_filtered_correlator_bank(correlator_bank, post_corr_filter),
        complex(0.0, 0.0),
        0,
        0,
        0,
        1,
        NaN,
        SYNC_UNCHANGED,
        0,
        Int8(0),
    )
end

"""
    PostCorrelationState(state::PostCorrelationState; prn, sampling_frequency, bit_synchronizer,
                    cn0_estimator, post_corr_filter, last_filtered_correlator_bank,
                    previous_filtered_prompt, last_num_integrated_samples, last_sample_index,
                    last_command_issue_sample_index, last_num_code_periods, last_code_phase,
                    last_sync_change, last_num_new_soft_bits, last_carrier_phase_polarity)

`state` with the fields passed replaced, every other one kept, so a host that
swaps a part of the state (a C/N₀ estimator, say) need not list the rest, and one that
builds a latest integration without [`process_correlator_bank_output!`](@ref) can set it.
"""
function PostCorrelationState(
    state::PostCorrelationState;
    prn::Maybe{Integer} = nothing,
    sampling_frequency = nothing,
    bit_synchronizer::Maybe{BitSynchronizer} = nothing,
    cn0_estimator::Maybe{AbstractCN0Estimator} = nothing,
    post_corr_filter::Maybe{AbstractPostCorrFilter} = nothing,
    last_filtered_correlator_bank::Maybe{AbstractCorrelatorBank} = nothing,
    previous_filtered_prompt::Maybe{Number} = nothing,
    last_num_integrated_samples::Maybe{Integer} = nothing,
    last_sample_index::Maybe{Integer} = nothing,
    last_command_issue_sample_index::Maybe{Integer} = nothing,
    last_num_code_periods::Maybe{Integer} = nothing,
    last_code_phase::Maybe{Real} = nothing,
    last_sync_change::Maybe{SyncEvent} = nothing,
    last_num_new_soft_bits::Maybe{Integer} = nothing,
    last_carrier_phase_polarity::Maybe{Integer} = nothing,
)
    PostCorrelationState(
        state.signal,
        isnothing(prn) ? state.prn : Int(prn),
        state.driver_carrier_phase_offset,
        isnothing(sampling_frequency) ? state.sampling_frequency : _sampling_frequency(sampling_frequency),
        isnothing(bit_synchronizer) ? state.bit_synchronizer : bit_synchronizer,
        isnothing(cn0_estimator) ? state.cn0_estimator : cn0_estimator,
        isnothing(post_corr_filter) ? state.post_corr_filter : post_corr_filter,
        isnothing(last_filtered_correlator_bank) ? state.last_filtered_correlator_bank :
        last_filtered_correlator_bank,
        isnothing(previous_filtered_prompt) ? state.previous_filtered_prompt :
        ComplexF64(previous_filtered_prompt),
        isnothing(last_num_integrated_samples) ? state.last_num_integrated_samples :
        Int(last_num_integrated_samples),
        isnothing(last_sample_index) ? state.last_sample_index : Int(last_sample_index),
        isnothing(last_command_issue_sample_index) ? state.last_command_issue_sample_index :
        Int(last_command_issue_sample_index),
        isnothing(last_num_code_periods) ? state.last_num_code_periods : Int(last_num_code_periods),
        isnothing(last_code_phase) ? state.last_code_phase : Float64(last_code_phase),
        isnothing(last_sync_change) ? state.last_sync_change : last_sync_change,
        isnothing(last_num_new_soft_bits) ? state.last_num_new_soft_bits : Int(last_num_new_soft_bits),
        isnothing(last_carrier_phase_polarity) ? state.last_carrier_phase_polarity :
        Int8(last_carrier_phase_polarity),
    )
end

# The latest integration's coherent integration time, first sample and centre sample, on
# the band's sample count.
_coherent_integration_time(state::PostCorrelationState) = state.last_num_integrated_samples / state.sampling_frequency
_integration_start(state::PostCorrelationState) = state.last_sample_index - state.last_num_integrated_samples
_integration_center(state::PostCorrelationState) = state.last_sample_index - state.last_num_integrated_samples / 2

# A sampling frequency in Hz, as the state keeps it.
_sampling_frequency(sampling_frequency) = Float64(ustrip(Hz, sampling_frequency)) * Hz

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
    is_bit_or_secondary_code_synced(state) ? state.bit_synchronizer.num_code_periods_into_bit : 0

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
get_bit_polarity(state::PostCorrelationState) =
    is_bit_or_secondary_code_synced(state) ? state.bit_synchronizer.bit_polarity : Int8(0)

estimate_cn0(state::PostCorrelationState, coherent_integration_time) =
    estimate_cn0(state.cn0_estimator, coherent_integration_time)

# Add the integration to the state's C/N₀ estimator, or skip it while a required noise density
# is not measured yet. Both branches return the same type; `requires_noise_density`
# folds away at compile time.
@inline function _update_cn0_estimator(
    state::PostCorrelationState,
    prompt,
    num_code_periods::Integer,
    bit_sync_usable::Bool,
    noise_density,
    noise_density_ready::Bool,
    coherent_integration_time,
)
    estimator = state.cn0_estimator
    if !noise_density_ready && requires_noise_density(estimator)
        return estimator
    end
    # Positional, not keyword: this is the per-integration site.
    update!(
        estimator,
        prompt,
        CN0UpdateContext(
            state.signal,
            state.bit_synchronizer,
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

# An integration's carrier phase polarity (see `process_correlator_bank_output!`). `bit_synchronizer` is the one the
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
    process_correlator_bank_output!(state::PostCorrelationState, correlator_bank_output, sampling_frequency,
                 noise_density, noise_density_ready;
                 secondary_code_wiped = is_bit_or_secondary_code_synced(state),
                 command_issue_sample_index = correlator_bank_output.sample_index, sample_index_offset = 0)
        -> PostCorrelationState

Process one correlator bank output, a completed coherent integration, and return the
state with it as its latest integration: normalize the raw correlator bank by sample count
and code amplitude, update and apply the post-correlation filter, advance the C/N₀
estimator against this satellite's share of `noise_density` (skipped while
`noise_density_ready` is `false` and the estimator needs one), and advance the bit
synchronizer. A tracking loop steps on the returned state ([`step_loop!`](@ref)).

The bit synchronizer is credited with the code periods actually integrated
(`calc_num_code_periods_for_bit_sync`), recovered from the sample count. Everything
here is Tracking's per-integration arithmetic, in the same order. Processing and
resetting reuse allocated storage. `command_issue_sample_index` defaults to the output's
sample index; `sample_index_offset` shifts both onto the receiver's shared time grid. The soft
bits it demodulates join the state's soft-bit vector ([`get_soft_bits`](@ref)), which the
host empties with [`drain_soft_bits!`](@ref) once it has read them.

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
    correlator_bank_output::CorrelatorBankOutput,
    sampling_frequency,
    noise_density,
    noise_density_ready::Bool;
    secondary_code_wiped::Bool = is_bit_or_secondary_code_synced(state),
    command_issue_sample_index::Integer = correlator_bank_output.sample_index,
    sample_index_offset::Integer = 0,
)
    (; signal, prn, driver_carrier_phase_offset) = state
    bit_synchronizer_before = state.bit_synchronizer
    synced_before = is_bit_or_secondary_code_synced(bit_synchronizer_before)
    num_soft_bits_before = Base.length(bit_synchronizer_before.soft_bits)
    # 1. Normalize: the raw correlator bank by its sample count and the code amplitude.
    normalized_correlator_bank =
        normalize(correlator_bank_output.correlator_bank, correlator_bank_output.num_integrated_samples, get_code_amplitude(signal))
    # 2. The post-correlation filter, updated on the prompt, combines the antennas into one
    #    correlator bank. Its weights are used twice: to combine the antennas and to reduce
    #    the shared noise covariance to this satellite's density (step 4).
    post_corr_filter = update!(state.post_corr_filter, get_prompt(normalized_correlator_bank))
    weights = get_weights(post_corr_filter, NumAnts(normalized_correlator_bank))
    filtered_correlator_bank = _filter_correlator_bank(normalized_correlator_bank, weights)
    prompt = get_prompt(filtered_correlator_bank)
    # 3. The code periods integrated, recovered from the sample count.
    num_code_periods_for_bit_sync = calc_num_code_periods_for_bit_sync(
        signal,
        correlator_bank_output.num_integrated_samples,
        sampling_frequency,
        synced_before,
    )
    # Floor at 1 for the fractional-period integration right after a sync phase snap.
    num_code_periods = max(1, num_code_periods_for_bit_sync)
    # Onto the driver's (real) phase frame for sync search and bit accumulation.
    bit_prompt = prompt * _carrier_phase_derotation(driver_carrier_phase_offset, signal)
    drop_prompt = _drops_unwiped_prompt(signal, synced_before, secondary_code_wiped)
    # 4. C/N₀ against this satellite's noise density, its share of `noise_density`.
    scalar_noise_density = _reduce_noise_density(noise_density, weights)
    cn0_estimator = _update_cn0_estimator(
        state,
        prompt,
        num_code_periods_for_bit_sync,
        !drop_prompt,
        scalar_noise_density,
        noise_density_ready,
        correlator_bank_output.num_integrated_samples / sampling_frequency,
    )
    # 5. Bit synchronization: the sync search, or the bit accumulation once synced.
    bit_synchronizer = update_bit_sync!(
        signal,
        prn,
        bit_synchronizer_before,
        num_code_periods_for_bit_sync,
        drop_prompt ? zero(bit_prompt) : bit_prompt,
    )
    # An unwiped post-sync integration moves the secondary-code anchor: the code-phase
    # snap after this processing aligns the *upcoming* integration to `bit_synchronizer.secondary_code_phase`.
    if synced_before && !secondary_code_wiped
        bit_synchronizer = _advance_secondary_phase(signal, bit_synchronizer, num_code_periods_for_bit_sync)
    end
    # 6. The latest integration's fields: among them what it did to the sync, and the
    #    prompt the FLL compares with, none across a change of carrier phase polarity.
    is_synced = is_bit_or_secondary_code_synced(bit_synchronizer)
    sync_change =
        !synced_before && is_synced ? SYNC_FOUND :
        synced_before && !is_synced ? SYNC_LOST : SYNC_UNCHANGED
    carrier_phase_polarity = _carrier_phase_polarity(signal, bit_synchronizer_before, secondary_code_wiped)
    previous_filtered_prompt = carrier_phase_polarity == state.last_carrier_phase_polarity ?
        get_prompt(state.last_filtered_correlator_bank) : zero(ComplexF64)
    PostCorrelationState(
        signal,
        prn,
        driver_carrier_phase_offset,
        _sampling_frequency(sampling_frequency),
        bit_synchronizer,
        cn0_estimator,
        post_corr_filter,
        filtered_correlator_bank,
        ComplexF64(previous_filtered_prompt),
        correlator_bank_output.num_integrated_samples,
        correlator_bank_output.sample_index + Int(sample_index_offset),
        Int(command_issue_sample_index) + Int(sample_index_offset),
        num_code_periods,
        correlator_bank_output.code_phase,
        sync_change,
        Base.length(bit_synchronizer.soft_bits) - num_soft_bits_before,
        carrier_phase_polarity,
    )
end

"""
    reset_sync!(state::PostCorrelationState) -> PostCorrelationState

The state with a fresh, unsynchronised bit synchronizer (what a lost integration costs),
everything else kept; the latest integration reports no soft bits and no sync change.
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
    reset_post_correlation!(state::PostCorrelationState; prn = state.prn) -> PostCorrelationState

The state a channel starts tracking from (no sync, no soft bits, empty C/N₀
estimator, no latest integration), reusing the old vectors so a re-initialisation
allocates nothing. Pass `prn` for a channel given another satellite of the same signal. The
post-correlation filter is kept: it is configuration, not history;
an adaptive beamformer must be reset by its owner.

A custom [`AbstractCN0Estimator`](@ref) with history needs a
`TrackingLoops.reset_cn0_estimator!` method; without one this throws rather than
hand the new satellite the old one's C/N₀.
"""
reset_post_correlation!(state::PostCorrelationState; prn::Integer = state.prn) = PostCorrelationState(
    state.signal,
    Int(prn),
    state.driver_carrier_phase_offset,
    NaN * Hz,
    _fresh_bit_synchronizer(state.bit_synchronizer),
    reset_cn0_estimator!(state.cn0_estimator),
    state.post_corr_filter,
    zero(state.last_filtered_correlator_bank),
    complex(0.0, 0.0),
    0,
    0,
    0,
    1,
    NaN,
    SYNC_UNCHANGED,
    0,
    Int8(0),
)
