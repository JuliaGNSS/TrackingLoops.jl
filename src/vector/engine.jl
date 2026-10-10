# ─────────────────────────────────────────────────────────────────────────────
# The per-record side of the navigation engine: a satellite joins it on its first
# record, takes up each cycle's decisions on the record after and snapshots itself
# at every navigation epoch. Its driver's records close its loops and its decoding
# signal's records feed its decoder — the same records for a plain signal. Its bit
# clock and C/N₀ estimator are the host's, summarised on each record. The record on
# which the last satellite completes an epoch's snapshot runs that epoch's cycle.
#
# Times are on the records' grid: `sample_index / sampling_frequency` seconds
# since an origin shared by every satellite. The navigation epochs are the
# multiples of the cycle time on it.
# ─────────────────────────────────────────────────────────────────────────────

# The ranging signals of the estimator, as `(driver, decoding signal)` pairs: a
# plain signal is both.
_ranging_pair(signal::AbstractGNSSSignal) = (signal, signal)
_ranging_pair(pair::Pair{<:AbstractGNSSSignal,<:AbstractGNSSSignal}) = (pair.first, pair.second)

_is_plain_pair((driver, decoding)) = typeof(driver) == typeof(decoding)

function _check_ranging_pair((driver, decoding))
    if iszero(get_data_frequency(decoding))
        hint =
            _is_plain_pair((driver, decoding)) ?
            "; pair it with its data component, `$(nameof(typeof(driver)))() => data`" : ""
        throw(
            ArgumentError(
                "vector tracking decodes the navigation data of each satellite's decoding " *
                "signal; $(nameof(typeof(decoding))) carries none" * hint,
            ),
        )
    end
    get_code_frequency(decoding) == get_code_frequency(driver) || throw(
        ArgumentError(
            "a decoding signal must have its driver's chip rate: " *
            "$(nameof(typeof(decoding))) has $(get_code_frequency(decoding)), " *
            "$(nameof(typeof(driver))) $(get_code_frequency(driver))",
        ),
    )
    nothing
end

# The estimator, built here, after the navigation engine it holds.
function VectorPLLAndDLL(
    signals::Union{AbstractGNSSSignal,Pair{<:AbstractGNSSSignal,<:AbstractGNSSSignal}}...;
    inner::AbstractDopplerEstimator = ConventionalAssistedPLLAndDLL(),
    config::Union{VectorTracking,Nothing} = VectorTracking(),
    cycle_time = 100.0ms,
    lock_cn0_threshold = 30.0dBHz,
    max_satellites_per_signal::Integer = 16,
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
    pairs = map(_ranging_pair, signals)
    signal_types = DataType[]
    for pair in pairs
        push!(signal_types, typeof(first(pair)))
        _is_plain_pair(pair) || push!(signal_types, typeof(last(pair)))
    end
    allunique(signal_types) ||
        throw(ArgumentError("each signal can be given only once, as a driver or a decoding signal"))
    foreach(_check_ranging_pair, pairs)
    max_satellites_per_signal >= 1 ||
        throw(ArgumentError("`max_satellites_per_signal` must be at least 1"))
    T = Float64(ustrip(s, cycle_time)) * s
    0.0s < T < Inf * s ||
        throw(ArgumentError("the navigation cycle time must be positive and finite"))
    navigation = VectorNavigation(
        config,
        pairs,
        inner;
        cycle_time = T,
        lock_cn0_threshold = Float64(ustrip(lock_cn0_threshold)),
        max_satellites_per_signal = Int(max_satellites_per_signal),
        approximate_year,
        enable_ionospheric_correction,
        enable_tropospheric_correction,
    )
    VectorPLLAndDLL(inner, navigation)
end

"""
    step_loop(estimator::VectorPLLAndDLL, state, record::LoopRecord, words, landing_sample)
        -> (state, carrier_doppler, code_doppler)

One record of a satellite through the vector loop (see
[`VectorPLLAndDLL`](@ref)). The satellite joins the navigation engine on its
first record, of whichever signal. What the record does depends on its signal:

  - the satellite's **driver**: it takes up the latest cycle's admission,
    release and corrections, snapshots the replica, the Dopplers, the
    accumulated discriminators and the C/N₀ when it crosses a navigation epoch,
    and runs the loop;
  - its **decoding signal**: it feeds the soft bits it added to the
    satellite's decoder and notes where the bit clock stands, which the
    snapshot reads the symbol count from. The Dopplers come back as the
    command in force, the state unchanged;
  - for a plain signal, both, on the same record;
  - any other passenger: nothing, as for a scalar loop.

A satellite's snapshot of an epoch is complete once a record of each of the two
has crossed it, in either order, and the record that completes the last
satellite's snapshot runs that epoch's navigation cycle. Out of the vector
loop the driver's Dopplers are `step_loop(estimator.inner, …)`'s exactly.
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

# Find the group the record's signal drives or is decoded on — by type, so the search
# folds at compile time — and run the record on it in that role. A record of a signal
# the estimator was not built for is a passenger it ignores, unless it is the
# satellite's driver.
@inline function _step_vector_record(estimator, nav, ::Tuple{}, g, state, record::LoopRecord, words, landing_sample)
    _is_driver_record(state, record) && _throw_unknown_driver(record)
    _step_passenger(state, record, words, landing_sample)
end
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
        _is_driver_record(state, record) || _throw_foreign_driver(state, record)
        _step_vector_record(estimator, nav, group, g, state, record, words, landing_sample, true, group.decoding_signal isa S)
    elseif group.decoding_signal isa S
        _step_vector_record(estimator, nav, group, g, state, record, words, landing_sample, false, true)
    else
        _step_vector_record(estimator, nav, Base.tail(groups), g + 1, state, record, words, landing_sample)
    end
end

@noinline _throw_unknown_driver(record) = throw(
    ArgumentError(
        "this vector-tracking estimator was not built for " *
        "$(nameof(typeof(record.signal))); list it among its signals",
    ),
)

@noinline _throw_foreign_driver(state, record) = throw(
    ArgumentError(
        "a satellite driven by another signal was stepped with a record of " *
        "$(nameof(typeof(record.signal))), which this estimator ranges on as a driver",
    ),
)

# One record in its roles: `driver` (the record closes the satellite's loops) and
# `decoding` (it feeds the satellite's decoder) — both for a plain signal.
function _step_vector_record(
    estimator::VectorPLLAndDLL,
    nav::VectorNavigation,
    group::VTSlotGroup,
    g::Int,
    state::SatVectorPLLAndDLL,
    record::LoopRecord,
    words,
    landing_sample::Int64,
    driver::Bool,
    decoding::Bool,
)
    state, slot = _register!(nav, group, g, state, record)
    landing = landing_sample == NO_LANDING_SAMPLE ? record.sample_index : landing_sample
    driver && (state = _take_up_cycle(nav, group, slot, state, record, words, landing))
    if _run_overdue_cycle!(nav, slot, record) && driver
        state = _take_up_cycle(nav, group, slot, state, record, words, landing)
    end
    _move_past_gap!(nav, driver ? slot.last_end_time : slot.decoding_last_end_time, record)
    driver && (state = _snapshot_driver!(nav, group, slot, state, record, words))
    decoding && _snapshot_decoding!(nav, group, slot, record, words)
    _complete_snapshot!(nav, group, slot)
    if _run_cycle_if_due!(nav, record) && driver
        state = _take_up_cycle(nav, group, slot, state, record, words, landing)
    end
    if driver
        state, carrier_doppler, code_doppler = _step_satellite(estimator, state, record, words, landing_sample)
        _advance_driver!(group, slot, record, words)
    else
        carrier_doppler, code_doppler = _command_in_force(record, words, landing_sample)
    end
    decoding && _advance_decoding!(group, slot, record)
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
        push!(slots, VTSlot(group.decoding_signal, prn, group.prototype))
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

# Fill `slot` for a satellite registering on it, in place: the decoder restarts its
# sync — keeping the decoded data for the satellite that held the slot before, rebuilt
# for another one — and both signals start from the record's start.
function _reset_slot!(nav::VectorNavigation, slot::VTSlot, prn::Int, state, record::LoopRecord)
    slot.running_decoder =
        slot.prn == prn && slot.registration > 0 ? reset_decoder_state!(slot.running_decoder) :
        _rebind_decoder(slot.running_decoder, prn)
    slot.prn = prn
    slot.occupied = true
    slot.registration = nav.registrations
    slot.active = false
    fs = _sampling_frequency_hz(record)
    start = record.sample_index - record.integrated_samples
    slot.last_end_sample = start
    slot.last_end_time = start / fs
    slot.last_code_phase = NaN
    slot.last_integration_time = uconvert(s, record.integrated_samples / record.sampling_frequency)
    slot.chips_since_epoch = 0.0
    slot.decoding_last_end_sample = start
    slot.decoding_last_end_time = start / fs
    slot.decoding_bit_synced = false
    slot.decoding_blocks_into_symbol = 0
    slot.decoding_code_phase_fraction = NaN
    slot.sync_fold_end = typemin(Int)
    if nav.pending_epoch == typemin(Int)
        nav.pending_epoch = _epoch_at_or_after(nav, slot.last_end_time)
    end
    slot.first_epoch = _epoch_at_or_after(nav, slot.last_end_time)
    slot.driver_epoch = typemin(Int)
    slot.decoding_epoch = typemin(Int)
    slot.snapshot_epoch = typemin(Int)
    slot.decoder = slot.running_decoder
    slot.epoch_decoder = slot.running_decoder
    slot.epoch_bit_synced = false
    # A satellite registers out of the vector loop. One whose state is still in it was
    # released by the cycle that dropped it, or never was the member of this slot: it
    # takes the release up on this record (`_take_up_cycle`), re-seeding its scalar loop.
    slot.estimator_state = _disable_vector_tracking(state)
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

# The records resumed after a gap: a record that starts past the pending epoch, while
# no satellite can still snapshot it, moves the epochs on to where the records are.
function _move_past_gap!(nav::VectorNavigation, start_time, record::LoopRecord)
    epoch_time = _epoch_time(nav, nav.pending_epoch)
    nav.num_snapshots == 0 && start_time > epoch_time || return nothing
    end_time = record.sample_index / _sampling_frequency_hz(record)
    _fold_slot_groups(_can_reach_epoch, false, nav.groups, nav, epoch_time, end_time) &&
        return nothing
    nav.pending_epoch = _epoch_at_or_after(nav, start_time)
    nothing
end

# Whether a record of one of the slot's signals, its last one having ended at
# `last_end_time` and its half of the snapshot taken at `half_epoch`, crosses the
# pending epoch the slot is still to snapshot.
function _crosses_pending_epoch(nav::VectorNavigation, slot::VTSlot, half_epoch, last_end_time, record::LoopRecord)
    epoch = nav.pending_epoch
    epoch_time = _epoch_time(nav, epoch)
    end_time = record.sample_index / _sampling_frequency_hz(record)
    half_epoch < epoch && slot.first_epoch <= epoch && last_end_time <= epoch_time < end_time
end

# The driver's half of the snapshot at the pending epoch `E`, if this record crosses
# it: the replica's code phase at `E`, the Dopplers there, the C/N₀ from the driver's
# C/N₀ estimator, and the satellite's state with the discriminators accumulated up to
# `E` — which it then starts over. Returns the state.
function _snapshot_driver!(
    nav::VectorNavigation,
    group::VTSlotGroup,
    slot::VTSlot,
    state::SatVectorPLLAndDLL,
    record::LoopRecord,
    words,
)
    _crosses_pending_epoch(nav, slot, slot.driver_epoch, slot.last_end_time, record) ||
        return state
    fs = _sampling_frequency_hz(record)
    epoch_time = _epoch_time(nav, nav.pending_epoch)
    code_frequency = ustrip(Hz, get_code_frequency(group.signal))
    epoch_sample = epoch_time * fs
    _, code_word = mean_nco_word(words, slot.last_end_sample, epoch_sample)
    chips_to_epoch = (epoch_time - slot.last_end_time) * (code_frequency + code_word)
    carrier_doppler, code_doppler = mean_nco_word(words, epoch_sample, epoch_sample)
    integration_time = uconvert(s, record.integrated_samples / record.sampling_frequency)
    slot.cn0_dbhz =
        min(Float64(ustrip(estimate_cn0(record.cn0_estimator, integration_time))), MAX_CN0_DBHZ)
    slot.estimator_state = state
    slot.epoch_driver_code_phase = slot.last_code_phase + chips_to_epoch
    slot.carrier_phase = 0.0
    slot.carrier_doppler = carrier_doppler * Hz
    slot.code_doppler = code_doppler * Hz
    slot.coherent_integration_time = slot.last_integration_time
    slot.early_late_spacing = slot.last_early_late_spacing
    slot.chips_since_epoch = -chips_to_epoch
    slot.driver_epoch = nav.pending_epoch
    _reset_discriminator_accumulators(state)
end

# The decoding signal's half of the snapshot at the pending epoch `E`, if this record
# crosses it: the decoder at the decoding signal's last record end, whether its bit
# clock held the sync there, and the code phase at `E` from the edge of the decoder's
# last symbol — the whole code blocks the bit clock had integrated into the current
# symbol, the replica's phase past their boundary and the chips on to `E`, counted in
# the decoding signal's own code.
function _snapshot_decoding!(nav::VectorNavigation, group::VTSlotGroup, slot::VTSlot, record::LoopRecord, words)
    _crosses_pending_epoch(nav, slot, slot.decoding_epoch, slot.decoding_last_end_time, record) ||
        return nothing
    fs = _sampling_frequency_hz(record)
    epoch_time = _epoch_time(nav, nav.pending_epoch)
    signal = group.decoding_signal
    code_frequency = ustrip(Hz, get_code_frequency(signal))
    _, code_word = mean_nco_word(words, slot.decoding_last_end_sample, epoch_time * fs)
    chips_to_epoch = (epoch_time - slot.decoding_last_end_time) * (code_frequency + code_word)
    slot.epoch_decoder = slot.running_decoder
    slot.epoch_bit_synced = slot.decoding_bit_synced
    slot.epoch_symbol_phase = _code_phase_from_symbol_edge(signal, slot) + chips_to_epoch
    slot.decoding_epoch = nav.pending_epoch
    nothing
end

# Complete the slot's snapshot of the pending epoch once both halves are in: the
# decoder and the code phase from its last symbol edge — the symbol count from the
# decoding signal, the phase within the symbol from the driver's replica — and the
# lock: the driver's C/N₀ above the threshold, and the decoding signal's bit sync.
function _complete_snapshot!(nav::VectorNavigation, group::VTSlotGroup, slot::VTSlot)
    epoch = nav.pending_epoch
    slot.snapshot_epoch < epoch && slot.driver_epoch == epoch && slot.decoding_epoch == epoch ||
        return nothing
    code_frequency = ustrip(Hz, get_code_frequency(group.decoding_signal))
    decoder, code_phase = _decoder_at_code_phase(
        slot.epoch_decoder,
        _symbol_phase_on_driver(group, slot),
        code_frequency,
    )
    in_lock = slot.epoch_bit_synced && slot.cn0_dbhz >= nav.lock_cn0_threshold
    slot.decoder = decoder
    slot.code_phase = code_phase
    slot.in_lock = in_lock
    slot.pvt_ready =
        in_lock && is_decoding_completed_for_positioning(decoder) && is_sat_healthy(decoder)
    slot.snapshot_epoch = epoch
    nav.num_snapshots += 1
    nothing
end

# The code phase at the epoch from the decoder's last symbol edge, with its phase within
# the symbol taken from the driver's replica: the decoding signal's count says which
# symbol and roughly where, the driver's replica — the one the discriminators steer —
# where exactly. The two codes are aligned at the satellite, so the driver's phase is
# the symbol phase modulo the shorter of the driver's code period and a data symbol;
# it moves the decoding signal's count to the nearest phase that agrees with it. For a
# plain signal both come from the same replica and agree already. Before the bit sync
# there is no symbol edge, and nothing reads it.
function _symbol_phase_on_driver(group::VTSlotGroup, slot::VTSlot)
    symbol_phase = slot.epoch_symbol_phase
    driver_phase = slot.epoch_driver_code_phase
    slot.epoch_bit_synced && !isnan(driver_phase) || return symbol_phase
    decoding = group.decoding_signal
    chips_per_symbol =
        ustrip(Hz, get_code_frequency(decoding)) / ustrip(Hz, get_data_frequency(decoding))
    period = min(Float64(get_code_length(group.signal)), chips_per_symbol)
    symbol_phase + rem(driver_phase - symbol_phase, period, RoundNearest)
end

# The C/N₀ the measurements are weighted by is capped at what a receiver can see: the
# moment estimator reads a noise-free signal (a simulation's) as infinite, which would
# give the navigation filter a measurement without noise.
const MAX_CN0_DBHZ = 80.0

# Whether some satellite of the group can still complete its snapshot of the pending
# epoch at `epoch_time`: one that is tracked, not stale at `now`, due at that epoch,
# and whose missing halves' signals are not yet past it.
function _can_reach_epoch(acc, group::VTSlotGroup, nav::VectorNavigation, epoch_time, now)
    acc && return true
    epoch = nav.pending_epoch
    for slot in group.slots
        slot.occupied && !_is_stale(nav, slot, now) && slot.first_epoch <= epoch &&
            slot.snapshot_epoch < epoch &&
            (slot.driver_epoch == epoch || slot.last_end_time <= epoch_time) &&
            (slot.decoding_epoch == epoch || slot.decoding_last_end_time <= epoch_time) &&
            return true
    end
    false
end

# The decoding signal's code phase (chips) at its last record end, counted from the
# last data-symbol edge: the whole code blocks the bit clock has integrated into the
# current symbol, and the replica's own phase past the block boundary. Before the bit
# sync there is no edge, and nothing reads it.
function _code_phase_from_symbol_edge(signal::AbstractGNSSSignal, slot::VTSlot)
    blocks = slot.decoding_bit_synced ? slot.decoding_blocks_into_symbol : 0
    fraction = slot.decoding_code_phase_fraction
    blocks * get_code_length(signal) + (isnan(fraction) ? 0.0 : fraction)
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

# Run the pending epoch's cycle once every satellite has completed its snapshot of it —
# counting out those that joined after it and those without a record for two cycles,
# which the cycle drops. Returns whether it ran.
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

# Stale: no record of the driver, or of the decoding signal, for two cycles.
_is_stale(nav::VectorNavigation, slot::VTSlot, now) =
    min(slot.last_end_time, slot.decoding_last_end_time) < now - 2 * _cycle_seconds(nav)

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
# After the record

# After the driver's loop step: the replica moved on to the record's end.
function _advance_driver!(group::VTSlotGroup, slot::VTSlot, record::LoopRecord, words)
    signal = group.signal
    fs = _sampling_frequency_hz(record)
    code_frequency = ustrip(Hz, get_code_frequency(signal))
    _, code_word = mean_nco_word(words, slot.last_end_sample, record.sample_index)
    slot.chips_since_epoch +=
        (record.sample_index - slot.last_end_sample) / fs * (code_frequency + code_word)
    slot.last_end_sample = record.sample_index
    slot.last_end_time = record.sample_index / fs
    slot.last_code_phase = record.code_phase
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

# A decoding-signal record: the soft bits it added into the decoder — exactly those, so
# the host may drain its bit buffer whenever it likes — and where the bit clock stands
# at its end. A record of the fold a sync was found in, correlated before it (it
# follows the record that found it), does not anchor the symbol count: the bit clock
# counts as unsynchronised there until the next fold.
function _advance_decoding!(group::VTSlotGroup, slot::VTSlot, record::LoopRecord)
    soft_bits = record.new_soft_bits
    if !isempty(soft_bits)
        slot.running_decoder = decode!(slot.running_decoder, soft_bits, Base.length(soft_bits))
    end
    if record.sync_change == SYNC_FOUND
        slot.sync_fold_end = record.fold_end
    elseif !record.bit_synced
        slot.sync_fold_end = typemin(Int)
    end
    correlated_pre_sync = record.sync_change != SYNC_FOUND && record.fold_end == slot.sync_fold_end
    code_length = get_code_length(group.decoding_signal)
    fs = _sampling_frequency_hz(record)
    slot.decoding_last_end_sample = record.sample_index
    slot.decoding_last_end_time = record.sample_index / fs
    slot.decoding_bit_synced = record.bit_synced && !correlated_pre_sync
    slot.decoding_blocks_into_symbol = record.blocks_into_symbol
    slot.decoding_code_phase_fraction =
        isnan(record.code_phase) ? NaN :
        mod(record.code_phase + code_length / 2, code_length) - code_length / 2
    nothing
end
