# ─────────────────────────────────────────────────────────────────────────────
# The per-record side of the navigation engine (see `step_loop`).
#
# Times are `sample_index / sampling_frequency` seconds since an origin shared by every
# satellite; the navigation epochs are the multiples of the update interval.
# ─────────────────────────────────────────────────────────────────────────────

# The estimator's ranging signals as `(driver, decoding signal)` pairs; a plain signal
# is both.
_ranging_pair(signal::AbstractGNSSSignal) = (signal, signal)
_ranging_pair(pair::Pair{<:AbstractGNSSSignal,<:AbstractGNSSSignal}) = (pair.first, pair.second)

_is_plain_pair((driver, decoding)) = typeof(driver) == typeof(decoding)

function _check_ranging_pair((driver, decoding))
    if iszero(get_bit_rate(decoding))
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
    get_chip_rate(decoding) == get_chip_rate(driver) || throw(
        ArgumentError(
            "a decoding signal must have its driver's chip rate: " *
            "$(nameof(typeof(decoding))) has $(get_chip_rate(decoding)), " *
            "$(nameof(typeof(driver))) $(get_chip_rate(driver))",
        ),
    )
    nothing
end

# The estimator, built here, after the navigation engine it holds.
function VectorPLLAndDLL(
    signals::Union{AbstractGNSSSignal,Pair{<:AbstractGNSSSignal,<:AbstractGNSSSignal}}...;
    inner::AbstractDopplerEstimator = ConventionalAssistedPLLAndDLL(),
    filter_config::Union{NavigationFilterConfig,Nothing} = NavigationFilterConfig(),
    vt_update_interval = 100.0ms,
    lock_cn0_threshold = 30.0dBHz,
    residual_pll_loop_filter_bandwidth = default_residual_pll_loop_filter_bandwidth(),
    max_satellites_per_signal::Integer = 16,
    approximate_year::Integer = year(now(UTC)),
    enable_ionospheric_correction::Bool = true,
    enable_tropospheric_correction::Bool = true,
)
    _has_fll_path(inner) || throw(
        ArgumentError(
            "a satellite the vector loop releases pulls in again through the FLL " *
            "path of an FLL-assisted carrier filter; use " *
            "`ConventionalAssistedPLLAndDLL()` or `NCOReferencedPLLAndDLL()` as " *
            "the inner estimator",
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
    T = Float64(ustrip(s, vt_update_interval)) * s
    0.0s < T < Inf * s ||
        throw(ArgumentError("`vt_update_interval` must be positive and finite"))
    residual_pll_loop_filter_bandwidth > 0.0Hz ||
        throw(ArgumentError("`residual_pll_loop_filter_bandwidth` must be positive"))
    navigation = VectorNavigation(
        filter_config,
        pairs,
        inner;
        vt_update_interval = T,
        lock_cn0_threshold_dbhz = Float64(ustrip(lock_cn0_threshold)),
        max_satellites_per_signal = Int(max_satellites_per_signal),
        approximate_year,
        enable_ionospheric_correction,
        enable_tropospheric_correction,
    )
    VectorPLLAndDLL(inner, navigation, Float64(ustrip(Hz, residual_pll_loop_filter_bandwidth)) * Hz)
end

"""
    step_loop(estimator::VectorPLLAndDLL, state, record::LoopRecord, words, landing_sample)
        -> (state, carrier_doppler, code_doppler)

One record of a satellite through the vector loop (see [`VectorPLLAndDLL`](@ref)).
The satellite registers with the navigation engine on its first record, of either
signal. A record of its

  - **driver** takes up the latest cycle's admission, release and commands,
    snapshots the replica, Dopplers, accumulated discriminators and C/N₀ when it
    crosses a navigation epoch, and runs the loop;
  - **decoding signal** feeds its new soft bits to the decoder and notes where the bit
    clock stands (the snapshot's symbol count); it returns the command in force and
    the state unchanged;
  - plain signal does both; any other passenger does nothing, as in a scalar loop.

A satellite's snapshot of an epoch is complete once a record of each signal has
crossed it, in either order; the record completing the last satellite's snapshot
runs that epoch's cycle. Out of the vector loop the driver's Dopplers equal
`step_loop(estimator.inner, …)`'s.
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

# Find the group the record's signal drives or is decoded on, by type so the search
# folds at compile time. A record of a signal the estimator was not built for is an
# ignored passenger unless it is the satellite's driver.
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
    if group.driver_signal isa S
        _is_driver_record(state, record) || _throw_foreign_driver(state, record)
        _step_vector_record(estimator, nav, group, g, state, record, words, landing_sample, true, group.decoding_signal isa S)
    elseif group.decoding_signal isa S
        # A satellite another group drives: a passenger of it this group must not take.
        _is_driven_by(state, group.driver_signal) ||
            return _step_passenger(state, record, words, landing_sample)
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

# One record in its roles: `driver` (closes the loops) and `decoding` (feeds the
# decoder); both for a plain signal.
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
    _move_past_gap!(nav, driver ? slot.driver_last_end_time : slot.decoding_last_end_time, record)
    driver && (state = _snapshot_driver!(nav, group, slot, state, record, words))
    decoding && _snapshot_decoding!(nav, group, slot, record, words)
    _complete_snapshot!(nav, group, slot)
    if _run_cycle_if_due!(nav, record) && driver
        state = _take_up_cycle(nav, group, slot, state, record, words, landing)
    end
    if driver
        state, carrier_doppler, code_doppler = _step_satellite(estimator, state, record, words, landing_sample)
        _advance_driver!(group, slot, record, words)
        # In the vector loop the residual PLL stepped on this prompt (`_measure_carrier`);
        # out of it, the history starts afresh when the satellite is added again.
        state.in_vector_loop ? _push_prompt!(slot.prompt_history, state.residual_pll) :
        _clear_prompts!(slot.prompt_history)
    else
        carrier_doppler, code_doppler = _command_in_force(record, words, landing_sample)
    end
    decoding && _advance_decoding!(group, slot, record, state, words, driver)
    state, carrier_doppler, code_doppler
end

_update_interval_seconds(nav::VectorNavigation) = ustrip(s, nav.vt_update_interval)
_sampling_frequency_hz(record::LoopRecord) = Float64(ustrip(Hz, record.sampling_frequency))
_epoch_time(nav::VectorNavigation, epoch::Int) = epoch * _update_interval_seconds(nav)

# The first epoch whose `_epoch_time` is not before `t`, compared exactly as
# `_crosses_pending_epoch` does (`t / T` may round to either side).
function _epoch_at_or_after(nav::VectorNavigation, t)
    T = _update_interval_seconds(nav)
    epoch = ceil(Int, t / T)
    (epoch - 1) * T >= t ? epoch - 1 : epoch * T < t ? epoch + 1 : epoch
end

# `f(acc, group, args...)` over the slot groups, in order, statically dispatched.
@inline _fold_groups(f::F, acc, ::Tuple{}, args...) where {F} = acc
@inline _fold_groups(f::F, acc, groups::Tuple, args...) where {F} =
    _fold_groups(f, f(acc, first(groups), args...), Base.tail(groups), args...)

# ─────────────────────────────────────────────────────────────────────────────
# Registration

# The satellite's slot, registering it on the record's PRN if it has none or lost it.
# A re-acquired PRN gets its old slot back with its decoded data; any other takes a
# free slot, and only when none is free does the group grow.
function _register!(nav::VectorNavigation, group::VTSlotGroup, g::Int, state::SatVectorPLLAndDLL, record::LoopRecord)
    slots = group.slots
    index = state.slot_index
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
        push!(slots, VTSlot(group.decoding_signal, prn, group.prototype, group.prompt_history_length))
        index = length(slots)
    end
    slot = slots[index]
    nav.registrations += 1
    state = _registered(state, index, nav.registrations)
    _reset_slot!(nav, slot, prn, state, record)
    state, slot
end

# The slot that holds or held `prn`, else any free one; 0 when none is free.
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

# Reset `slot` in place for a registering satellite. The decoder restarts its sync,
# keeping the decoded data only for the same PRN; both signals start at the record's
# start.
function _reset_slot!(nav::VectorNavigation, slot::VTSlot, prn::Int, state, record::LoopRecord)
    slot.running_decoder =
        slot.prn == prn && slot.registration > 0 ? reset_decoder_state!(slot.running_decoder) :
        reset_decoder_state!(slot.running_decoder, prn)
    slot.prn = prn
    slot.occupied = true
    slot.registration = nav.registrations
    slot.in_cycle = false
    fs = _sampling_frequency_hz(record)
    start = record.sample_index - record.integrated_samples
    slot.driver_last_end_sample = start
    slot.driver_last_end_time = start / fs
    slot.last_code_phase = NaN
    slot.last_integration_time = uconvert(s, record.integrated_samples / record.sampling_frequency)
    slot.chips_since_epoch = 0.0
    slot.decoding_last_end_sample = start
    slot.decoding_last_end_time = start / fs
    slot.decoding_bit_synced = false
    slot.decoding_blocks_into_bit = 0
    slot.decoding_code_phase_fraction = NaN
    slot.sync_fold_end = typemin(Int)
    slot.symbol_accumulator = zero(ComplexF64)
    slot.symbol_from_edge = false
    _clear_prompts!(slot.prompt_history)
    if nav.pending_epoch == typemin(Int)
        nav.pending_epoch = _epoch_at_or_after(nav, slot.driver_last_end_time)
    end
    # A snapshot the slot held is gone with it.
    slot.snapshot_epoch == nav.pending_epoch && (nav.num_snapshots -= 1)
    # A record starting exactly at an epoch crosses it, so that epoch is the first; if
    # it is not pending yet, the slot is past it by the time it is, and the cycle does
    # not wait for it (`_all_snapshotted`).
    slot.first_epoch = _epoch_at_or_after(nav, slot.driver_last_end_time)
    slot.driver_epoch = typemin(Int)
    slot.decoding_epoch = typemin(Int)
    slot.snapshot_epoch = typemin(Int)
    slot.decoder = slot.running_decoder
    slot.epoch_decoder = slot.running_decoder
    slot.epoch_bit_synced = false
    # Registered out of the vector loop; a state still in it takes up the release in
    # `_take_up_cycle`, re-seeding its scalar loop.
    slot.estimator_state = _release_from_vector_loop(state)
    slot.in_lock = false
    slot.pvt_ready = false
    slot.release_reason = VT_NOT_RELEASED
    slot.command_cycle = -1
    slot.member_index = 0
    nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Taking up a cycle

# Take up the latest cycle's decisions once: admission, release (re-seeding the scalar
# loop at the landing) and commands sized for that landing.
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
    slot_in_loop = slot.estimator_state.in_vector_loop
    if slot_in_loop && !state.in_vector_loop
        state = _add_to_vector_loop(state)
    elseif !slot_in_loop && state.in_vector_loop
        state = _hand_over_to_scalar_loop(state, words, landing, _sampling_frequency_hz(record))
    end
    if slot_in_loop && slot.command_cycle == id
        code_update, ramp, lead =
            _commands_at_landing(nav, group, slot, record, words, landing)
        state = _set_vector_commands(state, code_update * Hz, ramp, lead * s, words, landing)
    end
    SatVectorPLLAndDLL(state; cycle_id = id)
end

# The slot's commands for a landing at sample `landing`: the code correction, the
# carrier Doppler ramp from the predicted Dopplers there and one cycle later (with the
# replica's code moved on at its NCO word), and the lead (s) after the cycle's epoch.
function _commands_at_landing(
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
    -1 / fs <= lead <= MAX_LANDING_LEAD_CYCLES * _update_interval_seconds(nav) || throw(
        ArgumentError(
            "a command must land between zero and 2.5 navigation cycles after the " *
            "epoch whose commands it carries",
        ),
    )
    lead = max(lead, 0.0)
    T = _update_interval_seconds(nav)
    code_frequency = ustrip(Hz, get_chip_rate(group.driver_signal))
    _, code_word = mean_nco_word(words, slot.driver_last_end_sample, landing)
    chips =
        slot.chips_since_epoch + (landing - slot.driver_last_end_sample) / fs * (code_frequency + code_word)
    code_update, doppler = _member_commands(nav, slot, slot.member_index, lead, chips)
    _, next_doppler =
        _member_commands(nav, slot, slot.member_index, lead + T, chips + T * (code_frequency + code_word))
    code_update, CarrierRamp(doppler, (next_doppler - doppler) / T, landing / fs), lead
end

# ─────────────────────────────────────────────────────────────────────────────
# The epoch snapshot

# Records resumed after a gap: a record starting past the pending epoch, while no
# satellite can still snapshot it, moves the epochs on to where the records are.
function _move_past_gap!(nav::VectorNavigation, start_time, record::LoopRecord)
    epoch_time = _epoch_time(nav, nav.pending_epoch)
    nav.num_snapshots == 0 && start_time > epoch_time || return nothing
    end_time = record.sample_index / _sampling_frequency_hz(record)
    _fold_groups(_can_reach_epoch, false, nav.groups, nav, epoch_time, end_time) &&
        return nothing
    nav.pending_epoch = _epoch_at_or_after(nav, start_time)
    nothing
end

# Whether a record of one of the slot's signals (last record ended at
# `driver_last_end_time`, its snapshot half taken at `half_epoch`) crosses the pending
# epoch still to snapshot.
function _crosses_pending_epoch(nav::VectorNavigation, slot::VTSlot, half_epoch, driver_last_end_time, record::LoopRecord)
    epoch = nav.pending_epoch
    epoch_time = _epoch_time(nav, epoch)
    end_time = record.sample_index / _sampling_frequency_hz(record)
    half_epoch < epoch && slot.first_epoch <= epoch && driver_last_end_time <= epoch_time < end_time
end

# The driver's half of the snapshot at the pending epoch if this record crosses it:
# replica code phase, Dopplers, C/N₀ (from the record's estimator) and the state with
# its accumulated discriminators. Returns the state, its accumulators emptied if it
# crossed.
function _snapshot_driver!(
    nav::VectorNavigation,
    group::VTSlotGroup,
    slot::VTSlot,
    state::SatVectorPLLAndDLL,
    record::LoopRecord,
    words,
)
    _crosses_pending_epoch(nav, slot, slot.driver_epoch, slot.driver_last_end_time, record) ||
        return state
    fs = _sampling_frequency_hz(record)
    epoch_time = _epoch_time(nav, nav.pending_epoch)
    code_frequency = ustrip(Hz, get_chip_rate(group.driver_signal))
    epoch_sample = epoch_time * fs
    _, code_word = mean_nco_word(words, slot.driver_last_end_sample, epoch_sample)
    chips_to_epoch = (epoch_time - slot.driver_last_end_time) * (code_frequency + code_word)
    carrier_doppler, code_doppler = mean_nco_word(words, epoch_sample, epoch_sample)
    integration_time = uconvert(s, record.integrated_samples / record.sampling_frequency)
    slot.cn0_dbhz =
        min(Float64(ustrip(estimate_cn0(record.cn0_estimator, integration_time))), MAX_CN0_DBHZ)
    slot.estimator_state = state
    slot.epoch_driver_code_phase = slot.last_code_phase + chips_to_epoch
    slot.carrier_doppler = carrier_doppler * Hz
    slot.code_doppler = code_doppler * Hz
    slot.coherent_integration_time = slot.last_integration_time
    slot.early_late_spacing = slot.last_early_late_spacing
    slot.chips_since_epoch = -chips_to_epoch
    slot.driver_epoch = nav.pending_epoch
    _reset_discriminator_accumulators(state)
end

# The decoding signal's half of the snapshot at the pending epoch if this record
# crosses it: the decoder and bit sync at its last record end, and the code phase at
# the epoch from the last symbol edge, in the decoding signal's own code.
function _snapshot_decoding!(nav::VectorNavigation, group::VTSlotGroup, slot::VTSlot, record::LoopRecord, words)
    _crosses_pending_epoch(nav, slot, slot.decoding_epoch, slot.decoding_last_end_time, record) ||
        return nothing
    fs = _sampling_frequency_hz(record)
    epoch_time = _epoch_time(nav, nav.pending_epoch)
    signal = group.decoding_signal
    code_frequency = ustrip(Hz, get_chip_rate(signal))
    _, code_word = mean_nco_word(words, slot.decoding_last_end_sample, epoch_time * fs)
    chips_to_epoch = (epoch_time - slot.decoding_last_end_time) * (code_frequency + code_word)
    slot.epoch_decoder = slot.running_decoder
    slot.epoch_bit_synced = slot.decoding_bit_synced
    slot.epoch_code_phase = _code_phase_from_symbol_edge(signal, slot) + chips_to_epoch
    slot.decoding_epoch = nav.pending_epoch
    nothing
end

# Complete the slot's snapshot once both halves are in: the decoder and code phase
# from its last symbol edge (see `_code_phase_on_driver`), and the lock (driver's
# C/N₀ at least the threshold and decoding signal's bit sync).
function _complete_snapshot!(nav::VectorNavigation, group::VTSlotGroup, slot::VTSlot)
    epoch = nav.pending_epoch
    slot.snapshot_epoch < epoch && slot.driver_epoch == epoch && slot.decoding_epoch == epoch ||
        return nothing
    code_frequency = ustrip(Hz, get_chip_rate(group.decoding_signal))
    decoder, code_phase = _decoder_at_code_phase(
        slot.epoch_decoder,
        _code_phase_on_driver(group, slot),
        code_frequency,
    )
    in_lock = slot.epoch_bit_synced && slot.cn0_dbhz >= nav.lock_cn0_threshold_dbhz
    slot.decoder = decoder
    slot.code_phase = code_phase
    slot.in_lock = in_lock
    slot.pvt_ready =
        in_lock && is_decoding_completed_for_positioning(decoder) && is_sat_healthy(decoder)
    slot.snapshot_epoch = epoch
    nav.num_snapshots += 1
    nothing
end

# The code phase at the epoch from the last symbol edge: which symbol (and roughly
# where) from the decoding signal's count, exactly where from the driver's replica, the
# one the discriminators steer. The codes are aligned at the satellite, so the driver's
# phase is the code phase modulo the shorter of its code period and a data symbol; the
# count moves to the nearest phase that agrees. A plain signal agrees already; before
# bit sync nothing reads it.
function _code_phase_on_driver(group::VTSlotGroup, slot::VTSlot)
    code_phase = slot.epoch_code_phase
    driver_phase = slot.epoch_driver_code_phase
    slot.epoch_bit_synced && !isnan(driver_phase) || return code_phase
    decoding = group.decoding_signal
    chips_per_bit = get_code_blocks_per_bit(decoding) * get_code_length(decoding)
    period = min(Float64(get_code_length(group.driver_signal)), chips_per_bit)
    code_phase + rem(driver_phase - code_phase, period, RoundNearest)
end

# Cap on the weighting C/N₀: the moment estimator reads a noise-free (simulated) signal
# as infinite, which would give the filter a noiseless measurement.
const MAX_CN0_DBHZ = 80.0

# Whether some satellite of the group can still complete its snapshot of the pending
# epoch: tracked, not stale, due, and with its missing halves' signals not yet past it
# (`_crosses_pending_epoch`).
function _can_reach_epoch(acc, group::VTSlotGroup, nav::VectorNavigation, epoch_time, now)
    acc && return true
    epoch = nav.pending_epoch
    for slot in group.slots
        slot.occupied && !_is_stale(nav, slot, now) && slot.first_epoch <= epoch &&
            slot.snapshot_epoch < epoch &&
            (slot.driver_epoch == epoch || slot.driver_last_end_time <= epoch_time) &&
            (slot.decoding_epoch == epoch || slot.decoding_last_end_time <= epoch_time) &&
            return true
    end
    false
end

# The decoding signal's code phase (chips) at its last record end from the last
# data-symbol edge: the bit clock's whole code blocks plus the phase past the block
# boundary. Meaningless (and unread) before bit sync.
function _code_phase_from_symbol_edge(signal::AbstractGNSSSignal, slot::VTSlot)
    blocks = slot.decoding_bit_synced ? slot.decoding_blocks_into_bit : 0
    fraction = slot.decoding_code_phase_fraction
    blocks * get_code_length(signal) + (isnan(fraction) ? 0.0 : fraction)
end

# Normalise `code_phase` into the current symbol: a phase outside it (just short of the
# edge, or past the next one) shifts the decoder's symbol count instead, which is all
# the transmit time reads.
function _decoder_at_code_phase(decoder::GNSSDecoderState, code_phase, code_frequency)
    num_symbols = decoder.num_bits_since_sync
    isnothing(num_symbols) && return decoder, code_phase
    chips_per_symbol = code_frequency / ustrip(Hz, get_bit_rate(decoder))
    shift = floor(Int, code_phase / chips_per_symbol)
    shift == 0 && return decoder, code_phase
    GNSSDecoderState(decoder; num_bits_since_sync = num_symbols + shift),
    code_phase - shift * chips_per_symbol
end

# ─────────────────────────────────────────────────────────────────────────────
# Running the cycle

# Run the pending cycle once every satellite that still can has snapshotted it,
# ignoring those that registered later, are stale, or are past it without one. Returns
# whether it ran.
#
# Invariant: every live satellite can snapshot the pending epoch unless it got past it
# before it was pending. Such a satellite sits that cycle out, and the cycle moves the
# pending epoch on to one no live satellite is past (`_next_pending_epoch`). The host
# steps the satellites one after another through each chunk, so one stepped early in a
# chunk can cross an epoch that becomes pending only when one stepped later runs the
# cycle before it: after a re-registration on a record starting at the epoch, or after
# an overdue cycle. Neither locks it out for more than that epoch.
function _run_cycle_if_due!(nav::VectorNavigation, record::LoopRecord)
    nav.num_snapshots == 0 && return false
    now = record.sample_index / _sampling_frequency_hz(record)
    _fold_groups(_all_snapshotted, true, nav.groups, nav, now) || return false
    _navigation_cycle!(nav, now)
    true
end

# If the slot has snapshotted the pending epoch and is already past the next one while
# the pending one still waits for others, run that cycle now with the satellites that
# made it. Returns whether it ran. Only a satellite behind by more than a cycle, yet
# not stale, holds an epoch up; the cycle then moves the pending epoch past
# the satellites stepped before (`_next_pending_epoch`), so the next one runs as due.
function _run_overdue_cycle!(nav::VectorNavigation, slot::VTSlot, record::LoopRecord)
    slot.snapshot_epoch == nav.pending_epoch || return false
    now = record.sample_index / _sampling_frequency_hz(record)
    now > _epoch_time(nav, nav.pending_epoch + 1) || return false
    _navigation_cycle!(nav, now)
    true
end

# Whether no satellite of the group can still complete its snapshot of the pending
# epoch.
_all_snapshotted(ready, group::VTSlotGroup, nav::VectorNavigation, now) =
    ready && !_can_reach_epoch(false, group, nav, _epoch_time(nav, nav.pending_epoch), now)

# The epoch to snapshot after the cycle at `epoch`: the next one, or, if a satellite is
# past it already (see `_run_cycle_if_due!`), the first no satellite is past. The cycle
# after skipped epochs is longer, as after a gap. The host steps every satellite through
# each chunk, which is shorter than a cycle, so in step no satellite is past the next
# epoch.
function _next_pending_epoch(nav::VectorNavigation, epoch)
    next = epoch + 1
    latest = _fold_groups(_latest_record_end, -Inf, nav.groups)
    latest > _epoch_time(nav, next) ? _epoch_at_or_after(nav, latest) : next
end

function _latest_record_end(latest, group::VTSlotGroup)
    for slot in group.slots
        slot.occupied || continue
        latest = max(latest, slot.driver_last_end_time, slot.decoding_last_end_time)
    end
    latest
end

# Stale: no record of the driver, or of the decoding signal, for two cycles.
_is_stale(nav::VectorNavigation, slot::VTSlot, now) =
    min(slot.driver_last_end_time, slot.decoding_last_end_time) <
    now - 2 * _update_interval_seconds(nav)

# Before a cycle: mark the slots it reads, clear release reasons, and free stale slots
# (storage kept, members released). Returns whether a member was dropped.
function _prepare_slots!(released, group::VTSlotGroup, nav::VectorNavigation, now)
    for slot in group.slots
        slot.release_reason = VT_NOT_RELEASED
        slot.in_cycle = slot.occupied && slot.snapshot_epoch == nav.pending_epoch
        if slot.occupied && !slot.in_cycle && _is_stale(nav, slot, now)
            slot.occupied = false
            if slot.estimator_state.in_vector_loop
                _mark_release!(slot, VT_INELIGIBLE)
                released = true
            end
        end
    end
    released
end

# One navigation cycle at the pending epoch.
#
#   - Not running: the scalar PVT. A fresh fix seeds the filter, puts its satellites
#     into the vector loop and closes their loops a first time at the seeded state.
#   - Running: one filter cycle (`_run_cycle!`) — admission and release, the members'
#     accumulated discriminators fused, commands for every member. Admitted are
#     satellites decoded, healthy, in lock and `ADMISSION_ELEVATION` above the
#     horizon; members out of lock stay in the loop unmeasured. After
#     `insufficient_measurements_timeout` of unsolvable epochs, or with no member left, every
#     satellite in the loop is released (those that missed the epoch too) and the next
#     cycle solves the scalar PVT.
#   - `filter_config = nothing`: the scalar PVT only.
function _navigation_cycle!(nav::VectorNavigation, now)
    groups = nav.groups
    buffers = nav.buffers
    epoch = nav.pending_epoch
    cycle_time =
        nav.cycle_epoch == typemin(Int) ? nav.vt_update_interval :
        (epoch - nav.cycle_epoch) * nav.vt_update_interval
    nav.cycle_id += 1
    dropped = _fold_groups(_prepare_slots!, false, groups, nav, now)
    seeded = released = fell_back = false
    if nav.enabled && nav.running
        released, fell_back = _run_cycle!(nav, groups, cycle_time)
    else
        # Scalar solve: no per-member report.
        empty_keeping_capacity!(nav.vector_loop_members)
        previous_pvt = nav.pvt
        pvt = _solve_scalar_pvt!(nav, groups)
        # `calc_pvt!` returns its input unchanged when it cannot solve, so identity
        # tests freshness.
        if nav.enabled && pvt !== previous_pvt
            seeded = true
            _seed!(nav, groups, cycle_time)
        end
    end
    num_members = _fold_groups_with_buffers(_count_members, 0, groups, buffers.sat_states, 1)
    nav.status = NavigationStatus(
        nav.running,
        nav.running || fell_back ? position_uncertainty(nav) * m : NaN * m,
        nav.running || fell_back ? clock_uncertainty(nav) * m : NaN * m,
        nav.time_with_insufficient_measurements,
        num_members,
        seeded,
        fell_back,
        released || dropped,
    )
    nav.cycle_epoch = epoch
    nav.pending_epoch = _next_pending_epoch(nav, epoch)
    nav.num_snapshots = 0
    nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# After the record

# After the driver's loop step: move the replica to the record's end.
function _advance_driver!(group::VTSlotGroup, slot::VTSlot, record::LoopRecord, words)
    signal = group.driver_signal
    fs = _sampling_frequency_hz(record)
    code_frequency = ustrip(Hz, get_chip_rate(signal))
    _, code_word = mean_nco_word(words, slot.driver_last_end_sample, record.sample_index)
    slot.chips_since_epoch +=
        (record.sample_index - slot.driver_last_end_sample) / fs * (code_frequency + code_word)
    slot.driver_last_end_sample = record.sample_index
    slot.driver_last_end_time = record.sample_index / fs
    slot.last_code_phase = record.code_phase
    slot.last_integration_time =
        uconvert(s, record.integrated_samples / record.sampling_frequency)
    slot.last_early_late_spacing =
        get_early_late_sample_spacing(
            record.filtered_correlator,
            record.sampling_frequency,
            get_chip_rate(signal),
        ) * code_frequency / fs
    nothing
end

# A decoding-signal record: decode exactly the soft bits it added (so the host may
# drain its bit buffer whenever it likes) and note the bit clock at its end. Records
# later in the fold a sync was found in still have the pre-sync replica (see
# `fold_record`'s `overlay_wiped`): they leave the symbol count unanchored until the
# next fold.
function _advance_decoding!(
    group::VTSlotGroup,
    slot::VTSlot,
    record::LoopRecord,
    state::SatVectorPLLAndDLL,
    words,
    driver::Bool,
)
    if record.sync_change == SYNC_FOUND
        slot.sync_fold_end = record.fold_end
    elseif !record.bit_synced
        slot.sync_fold_end = typemin(Int)
    end
    unwiped_after_sync = record.sync_change != SYNC_FOUND && record.fold_end == slot.sync_fold_end
    _decode_record!(group, slot, record, state, words, driver, unwiped_after_sync)
    code_length = get_code_length(group.decoding_signal)
    fs = _sampling_frequency_hz(record)
    slot.decoding_last_end_sample = record.sample_index
    slot.decoding_last_end_time = record.sample_index / fs
    slot.decoding_bit_synced = record.bit_synced && !unwiped_after_sync
    slot.decoding_blocks_into_bit = record.blocks_into_bit
    slot.decoding_code_phase_fraction =
        isnan(record.code_phase) ? NaN :
        mod(record.code_phase + code_length / 2, code_length) - code_length / 2
    nothing
end

# Decode a decoding-signal record. Out of the vector loop: the host's soft bits. In it
# the carrier NCO runs on the filter's prediction, so the prompts turn at its Doppler
# error: each prompt is rotated onto the incoming carrier by the residual PLL's estimate
# (`_symbol_prompt_phase`) and the decoding signal's carrier-phase offset from the
# driver, summed over the symbol, and the sum's real part decoded (signed as the host's
# bit clock signs its own) on the record where the host's bit clock ends the symbol. A
# symbol not covered from its first block (the one the satellite joins in, or one after
# an `unwiped_after_sync` record), and a record ending more than one, take the host's
# soft bits.
function _decode_record!(
    group::VTSlotGroup,
    slot::VTSlot,
    record::LoopRecord,
    state::SatVectorPLLAndDLL,
    words,
    driver::Bool,
    unwiped_after_sync::Bool,
)
    soft_bits = record.new_soft_bits
    phase =
        state.in_vector_loop && record.bit_synced && record.sync_change == SYNC_UNCHANGED && !unwiped_after_sync ?
        _symbol_prompt_phase(
            state.residual_pll,
            state.ramp,
            state.previous_replica_half,
            slot.prompt_history,
            record,
            words,
            driver,
        ) :
        NaN
    if isnan(phase)
        slot.symbol_accumulator = zero(ComplexF64)
        slot.symbol_from_edge = false
    else
        rotation =
            2π * (get_carrier_phase_offset(group.driver_signal) - get_carrier_phase_offset(group.decoding_signal) -
            phase)
        slot.symbol_accumulator += ComplexF64(get_prompt(record.filtered_correlator)) * cis(rotation)
    end
    isempty(soft_bits) && return nothing
    if !isnan(phase) && slot.symbol_from_edge && Base.length(soft_bits) == 1
        symbol = real(slot.symbol_accumulator)
        slot.soft_bit[1] = Float32(record.bit_polarity < 0 ? -symbol : symbol)
        slot.running_decoder = decode!(slot.running_decoder, slot.soft_bit, 1)
    else
        slot.running_decoder = decode!(slot.running_decoder, soft_bits, Base.length(soft_bits))
    end
    slot.symbol_accumulator = zero(ComplexF64)
    slot.symbol_from_edge = !isnan(phase)
    nothing
end
