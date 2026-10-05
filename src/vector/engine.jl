# ─────────────────────────────────────────────────────────────────────────────
# The per-record side of the navigation engine: a satellite joins it on its first
# record, takes up each cycle's decisions on the record after, snapshots itself at
# every navigation epoch and feeds its prompts to its own bit clock, decoder and
# C/N₀ estimator. The record on which the last satellite reaches an epoch runs
# that epoch's cycle.
#
# Times are on the records' grid: `sample_index / sampling_frequency` seconds
# since an origin shared by every satellite. The navigation epochs are the
# multiples of the cycle time on it.
# ─────────────────────────────────────────────────────────────────────────────

# The estimator, built here, after the navigation engine it holds.
function VectorPLLAndDLL(
    signals::AbstractGNSSSignal...;
    inner::AbstractDopplerEstimator = ConventionalAssistedPLLAndDLL(),
    config::Union{VectorTracking,Nothing} = VectorTracking(),
    cycle_time = 100.0ms,
    lock_cn0_threshold = 30.0dBHz,
    max_satellites_per_signal::Integer = 16,
    num_prompts_for_cn0_estimation::Integer = 100,
    approximate_year::Integer = year(now(UTC)),
    enable_ionospheric_correction::Bool = true,
    enable_tropospheric_correction::Bool = true,
)
    _is_fll_assisted(inner) || throw(
        ArgumentError(
            "the vector loop drives the carrier through the FLL branch of an " *
            "FLL-assisted carrier filter; use `ConventionalAssistedPLLAndDLL()` " *
            "or `NCOReferencedPLLAndDLL()` as the inner estimator",
        ),
    )
    isempty(signals) &&
        throw(ArgumentError("vector tracking needs at least one ranging signal"))
    allunique(map(typeof, signals)) ||
        throw(ArgumentError("each ranging signal can be given only once"))
    for signal in signals
        iszero(get_data_frequency(signal)) && throw(
            ArgumentError(
                "vector tracking decodes the navigation data of the signal it steps; " *
                "$(nameof(typeof(signal))) carries none",
            ),
        )
    end
    max_satellites_per_signal >= 1 ||
        throw(ArgumentError("`max_satellites_per_signal` must be at least 1"))
    T = Float64(ustrip(s, cycle_time)) * s
    0.0s < T < Inf * s ||
        throw(ArgumentError("the navigation cycle time must be positive and finite"))
    navigation = VectorNavigation(
        config,
        signals,
        inner;
        cycle_time = T,
        lock_cn0_threshold = Float64(ustrip(lock_cn0_threshold)),
        max_satellites_per_signal = Int(max_satellites_per_signal),
        num_prompts_for_cn0_estimation = Int(num_prompts_for_cn0_estimation),
        approximate_year,
        enable_ionospheric_correction,
        enable_tropospheric_correction,
    )
    VectorPLLAndDLL(inner, navigation)
end

"""
    step_loop(estimator::VectorPLLAndDLL, state, record::LoopRecord, words, landing_sample)
        -> (state, carrier_doppler, code_doppler)

One record through the vector loop (see [`VectorPLLAndDLL`](@ref)): the
satellite joins the navigation engine on its first record, takes up the latest
cycle's admission, release and corrections, snapshots itself when the record
crosses a navigation epoch, runs the loop, and feeds the record's prompt to its
bit clock, decoder and C/N₀ estimator. The record on which the last satellite
reaches an epoch runs that epoch's navigation cycle. Out of the vector loop the
Dopplers are `step_loop(estimator.inner, …)`'s exactly.
"""
@inline step_loop(
    estimator::VectorPLLAndDLL,
    state::SatVectorPLLAndDLL,
    record::LoopRecord,
    words,
    landing_sample::Int64,
) = _step_vector_record(
    estimator,
    estimator.navigation,
    estimator.navigation.groups,
    1,
    state,
    record,
    words,
    landing_sample,
)

# Find the record's signal group — by type, so the search folds at compile time —
# and run the record on it.
@inline _step_vector_record(estimator, nav, ::Tuple{}, g, state, record::LoopRecord, words, landing_sample) =
    throw(
        ArgumentError(
            "this vector-tracking estimator was not built for " *
            "$(nameof(typeof(record.signal))); list it among its signals",
        ),
    )
@inline function _step_vector_record(
    estimator,
    nav,
    groups::Tuple,
    g,
    state,
    record::LoopRecord{S},
    words,
    landing_sample,
) where {S}
    group = first(groups)
    if group.signal isa S
        _step_vector_record(estimator, nav, group, g, state, record, words, landing_sample)
    else
        _step_vector_record(estimator, nav, Base.tail(groups), g + 1, state, record, words, landing_sample)
    end
end

function _step_vector_record(
    estimator::VectorPLLAndDLL,
    nav::VectorNavigation,
    group::VTSlotGroup,
    g::Int,
    state::SatVectorPLLAndDLL,
    record::LoopRecord,
    words,
    landing_sample::Int64,
)
    state, slot = _register!(nav, group, g, state, record)
    landing = landing_sample == NO_LANDING_SAMPLE ? record.sample_index : landing_sample
    state = _take_up_cycle(nav, group, slot, state, record, words, landing)
    if _run_overdue_cycle!(nav, slot, record)
        state = _take_up_cycle(nav, group, slot, state, record, words, landing)
    end
    state = _snapshot_epoch!(nav, group, slot, state, record, words)
    if _run_cycle_if_due!(nav, record)
        state = _take_up_cycle(nav, group, slot, state, record, words, landing)
    end
    state, carrier_doppler, code_doppler = _step_satellite(estimator, state, record, words, landing_sample)
    _advance_slot!(group, slot, record, words)
    state, carrier_doppler, code_doppler
end

_cycle_seconds(nav::VectorNavigation) = ustrip(s, nav.cycle_time)
_sampling_frequency_hz(record::LoopRecord) = Float64(ustrip(Hz, record.sampling_frequency))
_epoch_time(nav::VectorNavigation, epoch::Int) = epoch * _cycle_seconds(nav)

# The first epoch at or after time `t`.
function _epoch_at_or_after(nav::VectorNavigation, t)
    T = _cycle_seconds(nav)
    epoch = ceil(Int, t / T)
    (epoch - 1) * T >= t ? epoch - 1 : epoch
end

# `f(acc, group, args...)` over the slot groups, in order, statically dispatched.
@inline _fold_slot_groups(f::F, acc, ::Tuple{}, args...) where {F} = acc
@inline _fold_slot_groups(f::F, acc, groups::Tuple, args...) where {F} =
    _fold_slot_groups(f, f(acc, first(groups), args...), Base.tail(groups), args...)

# ─────────────────────────────────────────────────────────────────────────────
# Registration

# The slot of the satellite `state` belongs to, registering it on the record's PRN when
# it has none, or the one it had was given away. A satellite seen before — the same PRN,
# re-acquired — gets its old slot back, with the decoded data the decoder keeps; any
# other takes a free slot, and only when none is free does the group grow.
function _register!(nav::VectorNavigation, group::VTSlotGroup, g::Int, state::SatVectorPLLAndDLL, record::LoopRecord)
    slots = group.slots
    index = state.slot
    if 1 <= index <= length(slots)
        slot = slots[index]
        slot.occupied && slot.registration == state.registration && return state, slot
    end
    prn = record.prn
    prn > 0 || throw(
        ArgumentError("vector tracking needs to know the satellite: set the record's `prn`"),
    )
    index = _find_slot(slots, prn)
    if index == 0
        push!(slots, VTSlot(group.signal, prn, group.prototype, group.num_prompts_for_cn0_estimation))
        index = length(slots)
    end
    slot = slots[index]
    nav.registrations += 1
    state = _registered(state, index, nav.registrations)
    _reset_slot!(nav, slot, prn, state, record)
    state, slot
end

# The slot to register `prn` on: the one that holds it, else a free one that held it,
# else any free one; 0 when none is free.
function _find_slot(slots, prn)
    free = 0
    for (index, slot) in enumerate(slots)
        if slot.prn == prn && slot.registration > 0
            return index
        end
        !slot.occupied && free == 0 && (free = index)
    end
    free
end

# Fill `slot` for a satellite registering on it, in place: the bit clock and C/N₀
# estimator start over in the vectors they own, and the decoder restarts its sync —
# keeping the decoded data for the satellite that held the slot before, rebuilt for
# another one.
function _reset_slot!(nav::VectorNavigation, slot::VTSlot, prn::Int, state, record::LoopRecord)
    slot.running_decoder =
        slot.prn == prn && slot.registration > 0 ? reset_decoder_state!(slot.running_decoder) :
        _rebind_decoder(slot.running_decoder, prn)
    slot.prn = prn
    slot.occupied = true
    slot.registration = nav.registrations
    slot.active = false
    slot.bit_buffer = _fresh_bit_buffer(slot.bit_buffer)
    slot.cn0_estimator = _reset_cn0_estimator(slot.cn0_estimator)
    slot.sync_fold_end = typemin(Int)
    fs = _sampling_frequency_hz(record)
    start = record.sample_index - record.integrated_samples
    slot.last_end_sample = start
    slot.last_end_time = start / fs
    slot.last_code_phase_fraction = NaN
    slot.last_integration_time = uconvert(s, record.integrated_samples / record.sampling_frequency)
    slot.chips_since_epoch = 0.0
    if nav.pending_epoch == typemin(Int)
        nav.pending_epoch = _epoch_at_or_after(nav, slot.last_end_time)
    end
    slot.first_epoch = _epoch_at_or_after(nav, slot.last_end_time)
    slot.snapshot_epoch = typemin(Int)
    slot.decoder = slot.running_decoder
    slot.estimator_state = state
    slot.in_lock = false
    slot.pvt_ready = false
    slot.release_reason = VT_NOT_RELEASED
    slot.restart_cycle = -1
    slot.correction_cycle = -1
    slot.member_index = 0
    nothing
end

# A decoder for satellite `prn` on the containers of `decoder`: its sync restarted, the
# vote tally of the old satellite's data emptied, no data decoded.
function _rebind_decoder(decoder::GNSSDecoderState{D}, prn::Int) where {D}
    restarted = reset_decoder_state!(decoder)
    _clear_vote_tally!(restarted.cache)
    GNSSDecoderState(prn, D(), D(), restarted.constants, restarted.cache, nothing, false)
end

@inline function _clear_vote_tally!(cache)
    hasfield(typeof(cache), :old_data) && empty!(getfield(cache, :old_data))
    nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Taking up a cycle

# The satellite's state with the latest cycle's decisions taken up, once: admission,
# a restart, the release (its scalar loop re-seeded from the replica where the command
# lands) and the corrections, sized for that landing.
function _take_up_cycle(
    nav::VectorNavigation,
    group::VTSlotGroup,
    slot::VTSlot,
    state::SatVectorPLLAndDLL,
    record::LoopRecord,
    words,
    landing::Int64,
)
    id = nav.cycle_id
    state.cycle_id == id && return state
    in_loop = slot.estimator_state.vt_on
    if slot.restart_cycle == id && state.vt_on
        state = _enable_vector_tracking(_disable_vector_tracking(state))
    end
    if in_loop && !state.vt_on
        state = _enable_vector_tracking(state)
    elseif !in_loop && state.vt_on
        carrier_doppler, code_doppler = mean_nco_word(words, landing, landing)
        state = _release_from_vector_tracking(state, carrier_doppler * Hz, code_doppler * Hz)
    end
    if in_loop && slot.correction_cycle == id
        code_update, carrier_update, lead =
            _correction_at_landing(nav, group, slot, record, words, landing)
        state = _set_vector_corrections(state, code_update * Hz, carrier_update * Hz, lead * s)
    end
    SatVectorPLLAndDLL(state; cycle_id = id)
end

# The slot's corrections from the latest cycle, for a command landing at sample
# `landing`, and how long after the cycle's epoch that is (s).
function _correction_at_landing(
    nav::VectorNavigation,
    group::VTSlotGroup,
    slot::VTSlot,
    record::LoopRecord,
    words,
    landing::Int64,
)
    fs = _sampling_frequency_hz(record)
    # A command landing on the epoch itself may read a rounding error early.
    lead = landing / fs - _epoch_time(nav, nav.cycle_epoch)
    -1 / fs <= lead <= MAX_LANDING_LEAD_CYCLES * _cycle_seconds(nav) || throw(
        ArgumentError(
            "a command must land between zero and 2.5 navigation cycles after the " *
            "epoch whose corrections it carries",
        ),
    )
    lead = max(lead, 0.0)
    code_frequency = ustrip(Hz, get_code_frequency(group.signal))
    _, code_word = mean_nco_word(words, slot.last_end_sample, landing)
    chips =
        slot.chips_since_epoch + (landing - slot.last_end_sample) / fs * (code_frequency + code_word)
    carrier_doppler, _ = mean_nco_word(words, landing, landing)
    code_update, carrier_update =
        _member_corrections(nav, slot, slot.member_index, lead, chips, carrier_doppler)
    code_update, carrier_update, lead
end

# ─────────────────────────────────────────────────────────────────────────────
# The epoch snapshot

# Snapshot the slot at the pending epoch `E` if this record crosses it: the replica's
# code phase there, relative to the last data-symbol edge, the decoder at the last
# record end, the Dopplers at `E`, the C/N₀ and lock, and the satellite's state with
# the discriminators accumulated up to `E` — which it then starts over. Returns the
# state.
function _snapshot_epoch!(
    nav::VectorNavigation,
    group::VTSlotGroup,
    slot::VTSlot,
    state::SatVectorPLLAndDLL,
    record::LoopRecord,
    words,
)
    fs = _sampling_frequency_hz(record)
    end_time = record.sample_index / fs
    start_time = slot.last_end_time
    epoch_time = _epoch_time(nav, nav.pending_epoch)
    # This satellite is past the pending epoch, and so is every other that could still
    # snapshot it: the records resumed after a gap, so the epochs move on to where they
    # are.
    if nav.num_snapshots == 0 && start_time > epoch_time &&
       !_fold_slot_groups(_can_reach_epoch, false, nav.groups, nav, epoch_time, end_time)
        nav.pending_epoch = _epoch_at_or_after(nav, start_time)
        epoch_time = _epoch_time(nav, nav.pending_epoch)
    end
    slot.snapshot_epoch < nav.pending_epoch && slot.first_epoch <= nav.pending_epoch &&
        start_time <= epoch_time < end_time || return state
    signal = group.signal
    code_frequency = ustrip(Hz, get_code_frequency(signal))
    epoch_sample = epoch_time * fs
    _, code_word = mean_nco_word(words, slot.last_end_sample, epoch_sample)
    chips_to_epoch = (epoch_time - start_time) * (code_frequency + code_word)
    carrier_doppler, code_doppler = mean_nco_word(words, epoch_sample, epoch_sample)
    decoder, code_phase = _decoder_at_code_phase(
        slot.running_decoder,
        _code_phase_from_symbol_edge(signal, slot) + chips_to_epoch,
        code_frequency,
    )
    cn0_dbhz = min(
        Float64(ustrip(estimate_cn0(slot.cn0_estimator, slot.last_integration_time))),
        MAX_CN0_DBHZ,
    )
    in_lock = slot.bit_buffer.found && cn0_dbhz >= nav.lock_cn0_threshold
    slot.decoder = decoder
    slot.estimator_state = state
    slot.code_phase = code_phase
    slot.carrier_phase = 0.0
    slot.carrier_doppler = carrier_doppler * Hz
    slot.code_doppler = code_doppler * Hz
    slot.cn0_dbhz = cn0_dbhz
    slot.coherent_integration_time = slot.last_integration_time
    slot.early_late_spacing = slot.last_early_late_spacing
    slot.in_lock = in_lock
    slot.pvt_ready =
        in_lock && is_decoding_completed_for_positioning(decoder) && is_sat_healthy(decoder)
    slot.chips_since_epoch = -chips_to_epoch
    slot.snapshot_epoch = nav.pending_epoch
    nav.num_snapshots += 1
    _reset_discriminator_accumulators(state)
end

# The C/N₀ the measurements are weighted by is capped at what a receiver can see: the
# moment estimator reads a noise-free signal (a simulation's) as infinite, which would
# give the navigation filter a measurement without noise.
const MAX_CN0_DBHZ = 80.0

# Whether some satellite of the group can still snapshot the pending epoch at
# `epoch_time`: one that is tracked, not stale at `now`, due at that epoch and not yet
# past it.
function _can_reach_epoch(acc, group::VTSlotGroup, nav::VectorNavigation, epoch_time, now)
    acc && return true
    for slot in group.slots
        slot.occupied && !_is_stale(nav, slot, now) && slot.first_epoch <= nav.pending_epoch &&
            slot.last_end_time <= epoch_time && return true
    end
    false
end

# The replica's code phase (chips) at the last record end, counted from the last
# data-symbol edge: the whole code blocks the bit clock has accumulated into the
# current symbol, and the replica's own phase past the block boundary. Before the bit
# sync there is no edge, and nothing reads it.
function _code_phase_from_symbol_edge(signal::AbstractGNSSSignal, slot::VTSlot)
    bit_buffer = slot.bit_buffer
    blocks = bit_buffer.found ? bit_buffer.prompt_accumulator_integrated_code_blocks : 0
    fraction = isnan(slot.last_code_phase_fraction) ? 0.0 : slot.last_code_phase_fraction
    blocks * get_code_length(signal) + fraction
end

# The decoder and the code phase from its last decoded symbol edge, for a code phase
# `code_phase` from the edge of `decoder`'s last symbol: a phase outside the symbol
# (a fraction just short of the edge, or a replica moved past the next edge) moves
# the decoder's symbol count instead, which is all the transmit time reads of it.
function _decoder_at_code_phase(decoder::GNSSDecoderState, code_phase, code_frequency)
    num_bits = decoder.num_bits_after_valid_syncro_sequence
    isnothing(num_bits) && return decoder, code_phase
    chips_per_symbol = code_frequency / ustrip(Hz, get_data_frequency(decoder))
    shift = floor(Int, code_phase / chips_per_symbol)
    shift == 0 && return decoder, code_phase
    GNSSDecoderState(decoder; num_bits_after_valid_syncro_sequence = num_bits + shift),
    code_phase - shift * chips_per_symbol
end

# ─────────────────────────────────────────────────────────────────────────────
# Running the cycle

# Run the pending epoch's cycle once every satellite has snapshotted it — counting
# out those that joined after it and those without a record for two cycles, which
# the cycle drops. Returns whether it ran.
function _run_cycle_if_due!(nav::VectorNavigation, record::LoopRecord)
    nav.num_snapshots == 0 && return false
    now = record.sample_index / _sampling_frequency_hz(record)
    _fold_slot_groups(_all_snapshotted, true, nav.groups, nav, now) || return false
    _navigation_cycle!(nav, now)
    true
end

# Whether the slot is at an epoch past the pending one while that still waits for
# others: its cycle runs now, with the satellites that made it, so the slot can go on.
function _run_overdue_cycle!(nav::VectorNavigation, slot::VTSlot, record::LoopRecord)
    slot.snapshot_epoch == nav.pending_epoch || return false
    now = record.sample_index / _sampling_frequency_hz(record)
    now > _epoch_time(nav, nav.pending_epoch + 1) || return false
    _navigation_cycle!(nav, now)
    true
end

function _all_snapshotted(ready, group::VTSlotGroup, nav::VectorNavigation, now)
    ready || return false
    for slot in group.slots
        slot.occupied && !_is_stale(nav, slot, now) || continue
        slot.snapshot_epoch == nav.pending_epoch || slot.first_epoch > nav.pending_epoch ||
            return false
    end
    true
end

_is_stale(nav::VectorNavigation, slot::VTSlot, now) =
    slot.last_end_time < now - 2 * _cycle_seconds(nav)

# Before a cycle: which slots it reads, every release reason cleared, and the stale
# slots dropped — freed with their storage kept, released if they were members.
# Returns whether a member was dropped.
function _prepare_slots!(released, group::VTSlotGroup, nav::VectorNavigation, now)
    for slot in group.slots
        slot.release_reason = VT_NOT_RELEASED
        slot.active = slot.occupied && slot.snapshot_epoch == nav.pending_epoch
        if slot.occupied && !slot.active && _is_stale(nav, slot, now)
            slot.occupied = false
            if slot.estimator_state.vt_on
                _release!(slot, VT_INELIGIBLE)
                released = true
            end
        end
    end
    released
end

# One navigation cycle at the pending epoch.
#
#   - Vector tracking not running: the scalar PVT over the satellites that are
#     ready for it. A fresh fix seeds the filter from it, puts the fix's satellites
#     into the vector loop and closes their loops a first time at the seeded state.
#     Satellites still in the loop from before are released if no longer eligible,
#     and start over otherwise.
#   - Running: one filter cycle: the members are admitted (decoded, healthy, in lock
#     and a degree above the horizon) and released, their accumulated
#     discriminators measured and fused, and every member left its corrections.
#     Members out of lock stay in the loop, unmeasured. After
#     `insufficient_meas_timeout` of unsolvable epochs, or with no member left, every
#     member is released and the next cycle solves the scalar PVT again.
#   - Built with `config = nothing`: the scalar PVT only.
function _navigation_cycle!(nav::VectorNavigation, now)
    groups = nav.groups
    buffers = nav.buffers
    epoch = nav.pending_epoch
    cycle_time =
        nav.cycle_epoch == typemin(Int) ? nav.cycle_time : (epoch - nav.cycle_epoch) * nav.cycle_time
    nav.cycle_id += 1
    dropped = _fold_slot_groups(_prepare_slots!, false, groups, nav, now)
    enabled = released = fell_back = false
    if nav.enabled && nav.running
        released, fell_back = _run_cycle!(nav, groups, cycle_time)
    else
        # This cycle's solution comes from the scalar solve, so the filter has no
        # per-member report to make.
        empty_keeping_capacity!(nav.member_sats)
        previous_pvt = nav.pvt
        pvt = _solve_scalar_pvt!(nav, groups)
        # `calc_pvt!` returns the very solution it was handed on an epoch it cannot
        # solve, so identity is exactly the freshness test.
        if nav.enabled && pvt !== previous_pvt
            enabled = true
            released = _seed!(nav, groups, cycle_time)
        end
    end
    num_members = _fold_groups(_count_members, 0, groups, buffers.states, 1)
    nav.status = VTStatus(
        nav.running,
        nav.running || fell_back ? position_uncertainty(nav) * m : NaN * m,
        nav.running || fell_back ? clock_uncertainty(nav) * m : NaN * m,
        nav.time_with_insufficient_meas,
        num_members,
        enabled,
        fell_back,
        released || dropped,
    )
    nav.cycle_epoch = epoch
    nav.pending_epoch = epoch + 1
    nav.num_snapshots = 0
    nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# The record's prompt

# After the loop step: the record's prompt into the slot's bit clock, C/N₀ estimator
# and decoder, and the replica moved on to the record's end.
function _advance_slot!(group::VTSlotGroup, slot::VTSlot, record::LoopRecord, words)
    signal = group.signal
    fs = _sampling_frequency_hz(record)
    code_frequency = ustrip(Hz, get_code_frequency(signal))
    _, code_word = mean_nco_word(words, slot.last_end_sample, record.sample_index)
    slot.chips_since_epoch +=
        (record.sample_index - slot.last_end_sample) / fs * (code_frequency + code_word)

    bit_buffer = slot.bit_buffer
    was_synced = bit_buffer.found
    # Records later in the fold the sync was found in were correlated before it.
    correlated_pre_sync = was_synced && record.fold_end == slot.sync_fold_end
    bit_block_count = calc_num_code_blocks_for_bit_buffer(
        signal,
        record.integrated_samples,
        record.sampling_frequency,
        was_synced,
    )
    prompt = get_prompt(record.filtered_correlator)
    bit_prompt = prompt * _carrier_phase_derotation(get_carrier_phase_offset(signal), signal)
    bit_buffer = _advance_bit_buffer(
        signal,
        slot.prn,
        bit_buffer,
        bit_block_count,
        bit_prompt,
        correlated_pre_sync,
    )
    if !was_synced && bit_buffer.found
        slot.sync_fold_end = record.fold_end
    elseif was_synced && !bit_buffer.found
        slot.sync_fold_end = typemin(Int)
    end
    slot.bit_buffer = bit_buffer
    slot.cn0_estimator = update(slot.cn0_estimator, prompt)
    soft_bits = bit_buffer.soft_bits
    if !isempty(soft_bits)
        slot.running_decoder = decode!(slot.running_decoder, soft_bits, length(soft_bits))
        empty!(soft_bits)
    end

    code_length = get_code_length(signal)
    slot.last_end_sample = record.sample_index
    slot.last_end_time = record.sample_index / fs
    slot.last_code_phase_fraction =
        isnan(record.code_phase) ? NaN :
        mod(record.code_phase + code_length / 2, code_length) - code_length / 2
    slot.last_integration_time =
        uconvert(s, record.integrated_samples / record.sampling_frequency)
    slot.last_early_late_spacing =
        get_early_late_sample_spacing(
            record.filtered_correlator,
            record.sampling_frequency,
            get_code_frequency(signal),
        ) * code_frequency / fs
    nothing
end
