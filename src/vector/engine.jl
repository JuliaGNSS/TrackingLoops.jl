# ─────────────────────────────────────────────────────────────────────────────
# The per-record side of the navigation engine: a satellite joins it on its first
# record, takes up each cycle's decisions on the record after, and snapshots itself at
# every navigation epoch. Its driver's records close its loops and carry its C/N₀; its
# decoding signal's records — the driver's own for a plain signal, the data
# component's for a pilot + data pair — carry the soft bits it decodes and the bit
# clock that places the data-symbol edges. The record on which the last satellite
# completes an epoch runs that epoch's cycle.
#
# Times are on the records' grid: `sample_index / sampling_frequency` seconds
# since an origin shared by every satellite. The navigation epochs are the
# multiples of the cycle time on it.
# ─────────────────────────────────────────────────────────────────────────────

# A ranging signal as the constructor takes it: a plain signal, which is its own
# decoding signal, or a `driver => decoding_signal` pair.
_driver_and_decoding(signal::AbstractGNSSSignal) = (signal, signal)
_driver_and_decoding(pair::Pair{<:AbstractGNSSSignal,<:AbstractGNSSSignal}) = (first(pair), last(pair))

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
    pairs = map(_driver_and_decoding, signals)
    allunique(map(pair -> typeof(first(pair)), pairs)) ||
        throw(ArgumentError("each ranging signal can be given only once"))
    for (driver, decoding_signal) in pairs
        iszero(get_data_frequency(decoding_signal)) && throw(
            ArgumentError(
                "vector tracking decodes the navigation data of a satellite's decoding " *
                "signal; $(nameof(typeof(decoding_signal))) carries none" *
                (decoding_signal === driver ?
                 ". Pair a pilot with its data component: `pilot => data`" : ""),
            ),
        )
        get_code_frequency(driver) == get_code_frequency(decoding_signal) || throw(
            ArgumentError(
                "a decoding signal must have the chip rate of its driver; " *
                "$(nameof(typeof(decoding_signal))) and $(nameof(typeof(driver))) do not",
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
[`VectorPLLAndDLL`](@ref)). The satellite is found by the driver its state was
initialised with, and joins the navigation engine on its first record.

  - A record of the driver takes up the latest cycle's admission, release and
    corrections, snapshots the driver's half of a navigation epoch it crosses
    and runs the loop. Out of the vector loop the Dopplers are
    `step_loop(estimator.inner, …)`'s exactly.
  - A record of the decoding signal decodes the soft bits it appended and
    snapshots the decoding half of a navigation epoch it crosses. For a plain
    signal these are the driver's own records.
  - Any other passenger's record is ignored by the engine.

Where the inner loop combines signals (`combine_signals = true`), every
passenger's record also adds its discriminators to the driver's next loop step,
into the PLL only while the satellite is in the vector loop. A record that is
not the driver's returns the command in force. The record on which the last
satellite completes an epoch runs that epoch's navigation cycle.
"""
@inline function step_loop(
    estimator::VectorPLLAndDLL,
    state::SatVectorPLLAndDLL,
    record::LoopRecord,
    words,
    landing_sample::Int64,
)
    _step_vector_record(
        estimator,
        estimator.navigation,
        estimator.navigation.groups,
        1,
        state.inner.driver,
        state,
        record,
        words,
        landing_sample,
    )
end

# Find the group of the satellite's driver, by the key its state recorded — a
# compare per group, each group's call statically dispatched — and run the record
# on it.
@inline _step_vector_record(estimator, nav, ::Tuple{}, g, driver::UInt64, state, record::LoopRecord, words, landing_sample) =
    throw(
        ArgumentError(
            "this vector-tracking estimator was not built for the satellite's driver " *
            "signal; list it among its signals",
        ),
    )
@inline function _step_vector_record(
    estimator,
    nav,
    groups::Tuple,
    g,
    driver::UInt64,
    state,
    record::LoopRecord,
    words,
    landing_sample,
)
    group = first(groups)
    if _signal_key(group.signal) == driver
        _step_vector_record(estimator, nav, group, g, state, record, words, landing_sample)
    else
        _step_vector_record(estimator, nav, Base.tail(groups), g + 1, driver, state, record, words, landing_sample)
    end
end

function _step_vector_record(
    estimator::VectorPLLAndDLL,
    nav::VectorNavigation,
    group::VTSlotGroup{S,DS},
    g::Int,
    state::SatVectorPLLAndDLL,
    record::LoopRecord,
    words,
    landing_sample::Int64,
) where {S,DS}
    is_driver = record.signal isa S
    is_decoding = record.signal isa DS
    # Every passenger's discriminators join the driver's loops where the inner
    # loop combines signals.
    is_driver || (state = _with_passenger_record(estimator, state, record, words))
    is_driver || is_decoding || return (state, _command_in_force(record, words, landing_sample)...)
    state, slot = _register!(nav, group, g, state, record)
    if !is_driver
        _run_overdue_cycle!(nav, slot, record)
        _snapshot_decoding_half!(nav, group, slot, record, words)
        _complete_snapshot!(nav, group, slot)
        _run_cycle_if_due!(nav, record)
        _advance_decoding!(group, slot, record)
        return (state, _command_in_force(record, words, landing_sample)...)
    end
    landing = landing_sample == NO_LANDING_SAMPLE ? record.sample_index : landing_sample
    state = _take_up_cycle(nav, group, slot, state, record, words, landing)
    if _run_overdue_cycle!(nav, slot, record)
        state = _take_up_cycle(nav, group, slot, state, record, words, landing)
    end
    state = _snapshot_driver_half!(nav, group, slot, state, record, words)
    is_decoding && _snapshot_decoding_half!(nav, group, slot, record, words)
    _complete_snapshot!(nav, group, slot)
    if _run_cycle_if_due!(nav, record)
        state = _take_up_cycle(nav, group, slot, state, record, words, landing)
    end
    state, carrier_doppler, code_doppler = _step_satellite(estimator, state, record, words, landing_sample)
    _advance_driver!(group, slot, record, words)
    is_decoding && _advance_decoding!(group, slot, record)
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

# Fill `slot` for a satellite registering on it, in place, on its first record of
# either signal: the decoder restarts its sync — keeping the decoded data for the
# satellite that held the slot before, rebuilt for another one — and both signals' records
# are taken to have ended where this one started.
function _reset_slot!(nav::VectorNavigation, slot::VTSlot, prn::Int, state, record::LoopRecord)
    slot.running_decoder =
        slot.prn == prn && slot.registration > 0 ? reset_decoder_state!(slot.running_decoder) :
        _rebind_decoder(slot.running_decoder, prn)
    slot.prn = prn
    slot.occupied = true
    slot.registration = nav.registrations
    slot.active = false
    slot.bit_synced = false
    slot.blocks_into_symbol = 0
    slot.sync_fold_end = typemin(Int)
    fs = _sampling_frequency_hz(record)
    start = record.sample_index - record.integrated_samples
    slot.last_end_sample = start
    slot.last_end_time = start / fs
    slot.last_code_phase_fraction = NaN
    slot.last_integration_time = uconvert(s, record.integrated_samples / record.sampling_frequency)
    slot.chips_since_epoch = 0.0
    slot.decoding_last_end_sample = start
    slot.decoding_last_end_time = start / fs
    slot.decoding_last_code_phase_fraction = NaN
    if nav.pending_epoch == typemin(Int)
        nav.pending_epoch = _epoch_at_or_after(nav, slot.last_end_time)
    end
    slot.first_epoch = _epoch_at_or_after(nav, slot.last_end_time)
    slot.snapshot_epoch = typemin(Int)
    slot.driver_epoch = typemin(Int)
    slot.decoding_epoch = typemin(Int)
    slot.decoder = slot.running_decoder
    slot.epoch_decoder = slot.running_decoder
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
#
# A satellite's snapshot of the pending epoch `E` has two halves: the driver's, from the
# driver record that crosses `E`, and the decoding signal's, from the decoding-signal
# record that crosses it. It is complete once both are there, in whichever order they
# came; for a plain signal one record takes both.

# The time of the pending epoch for a record of a signal whose previous record ended at
# `start_time`. A satellite past the pending epoch while none can still snapshot it —
# the records resumed after a gap — moves the epochs on to where they are.
function _pending_epoch_time!(nav::VectorNavigation, start_time, end_time)
    epoch_time = _epoch_time(nav, nav.pending_epoch)
    if nav.num_snapshots == 0 && start_time > epoch_time &&
       !_fold_slot_groups(_can_reach_epoch, false, nav.groups, nav, epoch_time, end_time)
        nav.pending_epoch = _epoch_at_or_after(nav, start_time)
        epoch_time = _epoch_time(nav, nav.pending_epoch)
    end
    epoch_time
end

# Whether a record of a signal whose half of the snapshot was last taken at
# `half_epoch` and whose previous record ended at `start_time` crosses the pending
# epoch at `epoch_time`, and that half is still due there.
_crosses_pending_epoch(nav::VectorNavigation, slot::VTSlot, half_epoch, start_time, epoch_time, end_time) =
    half_epoch < nav.pending_epoch && slot.first_epoch <= nav.pending_epoch &&
    start_time <= epoch_time < end_time

# A half of the snapshot was taken: the first of the slot's at this epoch counts it in.
function _count_snapshot!(nav::VectorNavigation, slot::VTSlot)
    slot.driver_epoch == nav.pending_epoch || slot.decoding_epoch == nav.pending_epoch ||
        (nav.num_snapshots += 1)
    nothing
end

# The driver's half at the pending epoch `E`, if this driver record crosses it: the
# replica's code phase there, the Dopplers at `E`, the driver's C/N₀ from the estimator
# the host configured for it, and the satellite's state with the discriminators
# accumulated up to `E` — which it then starts over. Returns the state.
function _snapshot_driver_half!(
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
    epoch_time = _pending_epoch_time!(nav, start_time, end_time)
    _crosses_pending_epoch(nav, slot, slot.driver_epoch, start_time, epoch_time, end_time) ||
        return state
    code_frequency = ustrip(Hz, get_code_frequency(group.signal))
    epoch_sample = epoch_time * fs
    _, code_word = mean_nco_word(words, slot.last_end_sample, epoch_sample)
    chips_to_epoch = (epoch_time - start_time) * (code_frequency + code_word)
    carrier_doppler, code_doppler = mean_nco_word(words, epoch_sample, epoch_sample)
    fraction = isnan(slot.last_code_phase_fraction) ? 0.0 : slot.last_code_phase_fraction
    integration_time = uconvert(s, record.integrated_samples / record.sampling_frequency)
    _count_snapshot!(nav, slot)
    slot.epoch_driver_code_phase = fraction + chips_to_epoch
    slot.estimator_state = state
    slot.carrier_phase = 0.0
    slot.carrier_doppler = carrier_doppler * Hz
    slot.code_doppler = code_doppler * Hz
    slot.cn0_dbhz = min(
        Float64(ustrip(estimate_cn0(record.cn0_estimator, integration_time))),
        MAX_CN0_DBHZ,
    )
    slot.coherent_integration_time = slot.last_integration_time
    slot.early_late_spacing = slot.last_early_late_spacing
    slot.chips_since_epoch = -chips_to_epoch
    slot.driver_epoch = nav.pending_epoch
    _reset_discriminator_accumulators(state)
end

# The decoding signal's half at the pending epoch, if this decoding-signal record
# crosses it: the decoder at its last record end, whether its bit clock was synced
# there, and its replica's code phase at the epoch counted from the last data-symbol
# edge, in the decoding signal's own code blocks.
function _snapshot_decoding_half!(
    nav::VectorNavigation,
    group::VTSlotGroup,
    slot::VTSlot,
    record::LoopRecord,
    words,
)
    fs = _sampling_frequency_hz(record)
    end_time = record.sample_index / fs
    start_time = slot.decoding_last_end_time
    epoch_time = _pending_epoch_time!(nav, start_time, end_time)
    _crosses_pending_epoch(nav, slot, slot.decoding_epoch, start_time, epoch_time, end_time) ||
        return nothing
    signal = group.decoding_signal
    code_frequency = ustrip(Hz, get_code_frequency(signal))
    _, code_word = mean_nco_word(words, slot.decoding_last_end_sample, epoch_time * fs)
    chips_to_epoch = (epoch_time - start_time) * (code_frequency + code_word)
    _count_snapshot!(nav, slot)
    slot.epoch_symbol_code_phase = _code_phase_from_symbol_edge(signal, slot) + chips_to_epoch
    slot.epoch_bit_synced = slot.bit_synced
    slot.epoch_decoder = slot.running_decoder
    slot.decoding_epoch = nav.pending_epoch
    nothing
end

# Once both halves are there: the transmit time's two parts put together — the decoded
# symbol count and the replica's code phase from the last symbol edge — and the lock.
# The code phase is the driver's: the decoding signal places the symbol edge, the
# driver's own replica the phase within its code block.
function _complete_snapshot!(nav::VectorNavigation, group::VTSlotGroup, slot::VTSlot)
    epoch = nav.pending_epoch
    slot.snapshot_epoch < epoch && slot.driver_epoch == epoch && slot.decoding_epoch == epoch ||
        return nothing
    code_frequency = ustrip(Hz, get_code_frequency(group.signal))
    code_phase = slot.epoch_symbol_code_phase
    if _is_paired(group)
        code_phase = _align_to_driver(
            code_phase,
            slot.epoch_driver_code_phase,
            _driver_code_period(group.signal, slot.epoch_decoder),
        )
    end
    decoder, code_phase = _decoder_at_code_phase(slot.epoch_decoder, code_phase, code_frequency)
    in_lock = slot.epoch_bit_synced && slot.cn0_dbhz >= nav.lock_cn0_threshold
    slot.decoder = decoder
    slot.code_phase = code_phase
    slot.in_lock = in_lock
    slot.pvt_ready =
        in_lock && is_decoding_completed_for_positioning(decoder) && is_sat_healthy(decoder)
    slot.snapshot_epoch = epoch
    nothing
end

# The period (chips) in which the driver's code phase repeats against the data-symbol
# edges: its code length, or one symbol when its code spans several (GPS L2CL against
# L2CM's symbols). The two are multiples of one another for every pair.
function _driver_code_period(driver::AbstractGNSSSignal, decoder::GNSSDecoderState)
    chips_per_symbol = ustrip(Hz, get_code_frequency(driver)) / ustrip(Hz, get_data_frequency(decoder))
    min(Float64(get_code_length(driver)), chips_per_symbol)
end

# The code phase from the symbol edge, `symbol_code_phase` as the decoding signal counts
# it, moved onto the driver's code phase `driver_code_phase`, which it equals up to the
# whole driver code periods only the symbol count can tell.
_align_to_driver(symbol_code_phase, driver_code_phase, period) =
    driver_code_phase + period * round((symbol_code_phase - driver_code_phase) / period)

# The C/N₀ the measurements are weighted by is capped at what a receiver can see: the
# moment estimator reads a noise-free signal (a simulation's) as infinite, which would
# give the navigation filter a measurement without noise.
const MAX_CN0_DBHZ = 80.0

# Whether some satellite of the group can still snapshot the pending epoch at
# `epoch_time`: one that is tracked, not stale at `now`, due at that epoch and with a
# half of it still to take that it is not yet past.
function _can_reach_epoch(acc, group::VTSlotGroup, nav::VectorNavigation, epoch_time, now)
    acc && return true
    epoch = nav.pending_epoch
    for slot in group.slots
        slot.occupied && !_is_stale(nav, slot, now) && slot.first_epoch <= epoch || continue
        slot.driver_epoch < epoch && slot.last_end_time <= epoch_time && return true
        slot.decoding_epoch < epoch && slot.decoding_last_end_time <= epoch_time && return true
    end
    false
end

# The decoding signal's replica's code phase (chips) at its last record end, counted
# from the last data-symbol edge: the whole code blocks its bit clock has integrated
# into the current symbol, and the replica's own phase past the block boundary. Before
# the bit sync there is no edge, and nothing reads it.
function _code_phase_from_symbol_edge(signal::AbstractGNSSSignal, slot::VTSlot)
    blocks = slot.bit_synced ? slot.blocks_into_symbol : 0
    fraction =
        isnan(slot.decoding_last_code_phase_fraction) ? 0.0 : slot.decoding_last_code_phase_fraction
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
# others: its cycle runs now, with the satellites that completed it, so the slot can go
# on.
function _run_overdue_cycle!(nav::VectorNavigation, slot::VTSlot, record::LoopRecord)
    slot.driver_epoch == nav.pending_epoch || slot.decoding_epoch == nav.pending_epoch ||
        return false
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

# No record of the driver, or of the decoding signal, for two cycles.
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

# After the loop step: the driver's replica moved on to the record's end.
function _advance_driver!(group::VTSlotGroup, slot::VTSlot, record::LoopRecord, words)
    signal = group.signal
    fs = _sampling_frequency_hz(record)
    code_frequency = ustrip(Hz, get_code_frequency(signal))
    _, code_word = mean_nco_word(words, slot.last_end_sample, record.sample_index)
    slot.chips_since_epoch +=
        (record.sample_index - slot.last_end_sample) / fs * (code_frequency + code_word)
    slot.last_end_sample = record.sample_index
    slot.last_end_time = record.sample_index / fs
    slot.last_code_phase_fraction = _code_phase_fraction(signal, record)
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

# A record of the decoding signal: its soft bits into the decoder — exactly the ones
# this record appended, so the host may drain them whenever it likes — and where its
# bit clock and replica stand at the record's end. The records of the fold the sync was
# found in that come after the sync record were correlated before it: their bit clock
# does not place the symbol edge yet.
function _advance_decoding!(group::VTSlotGroup, slot::VTSlot, record::LoopRecord)
    if record.sync_change == SYNC_FOUND
        slot.sync_fold_end = record.fold_end
    elseif record.sync_change == SYNC_LOST
        slot.sync_fold_end = typemin(Int)
    end
    correlated_pre_sync = record.sync_change != SYNC_FOUND && record.fold_end == slot.sync_fold_end
    slot.bit_synced = record.bit_synced && !correlated_pre_sync
    slot.blocks_into_symbol = record.blocks_into_symbol
    soft_bits = record.new_soft_bits
    if !isempty(soft_bits)
        slot.running_decoder = decode!(slot.running_decoder, soft_bits, length(soft_bits))
    end
    slot.decoding_last_end_sample = record.sample_index
    slot.decoding_last_end_time = record.sample_index / _sampling_frequency_hz(record)
    slot.decoding_last_code_phase_fraction = _code_phase_fraction(group.decoding_signal, record)
    nothing
end

# The record's replica code phase past the nearest code-block boundary (chips), `NaN`
# when the record does not report it.
function _code_phase_fraction(signal::AbstractGNSSSignal, record::LoopRecord)
    code_length = get_code_length(signal)
    isnan(record.code_phase) ? NaN :
    mod(record.code_phase + code_length / 2, code_length) - code_length / 2
end
