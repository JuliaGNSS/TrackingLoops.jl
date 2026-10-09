# ─────────────────────────────────────────────────────────────────────────────
# The per-record side of the navigation engine (see `step_loop`).
#
# Times are `sample_index / sampling_frequency` seconds since an origin shared by every
# satellite; the navigation epochs are the multiples of the cycle time.
# ─────────────────────────────────────────────────────────────────────────────

"""
    VectorTrackingSettings(; combine_signals = false,
                           inner = ConventionalAssistedPLLAndDLL(),
                           config = VectorTracking(), cycle_time = 100ms,
                           lock_cn0_threshold = 30dBHz,
                           max_satellites_per_signal = 16,
                           num_prompts_for_cn0_estimation = 100,
                           approximate_year = year(now(UTC)),
                           enable_ionospheric_correction = true,
                           enable_tropospheric_correction = true)

A [`VectorPLLAndDLL`](@ref)'s settings without its signals, keywords as there, for a
host that keeps its satellites' signal groups itself, such as Tracking.jl's
`TrackState`: [`with_signal_groups`](@ref) builds the estimator from them and the
host's groups, so the groups are listed once. No estimator: nothing to step, no
results; read those from the estimator `with_signal_groups` returned.
"""
struct VectorTrackingSettings{E<:AbstractDopplerEstimator,C<:Union{VectorTracking,Nothing}}
    inner::E
    combine_signals::Bool
    config::C
    cycle_time::typeof(1.0s)
    lock_cn0_threshold::Float64
    max_satellites_per_signal::Int
    num_prompts_for_cn0_estimation::Int
    approximate_year::Int
    enable_ionospheric_correction::Bool
    enable_tropospheric_correction::Bool
end

function VectorTrackingSettings(;
    combine_signals::Bool = false,
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
    combines_signals(inner) && throw(
        ArgumentError(
            "the vector loop combines signals by its own `combine_signals`, out of the " *
            "vector loop too; build `inner` without it",
        ),
    )
    combine_signals &&
        !(inner isa ConventionalPLLAndDLL) &&
        throw(
            ArgumentError(
                "$(nameof(typeof(inner))) steps a phase error predicted to the landing " *
                "sample, which passenger records are not, so it cannot combine signals; " *
                "use `ConventionalAssistedPLLAndDLL()` as the inner estimator",
            ),
        )
    max_satellites_per_signal >= 1 ||
        throw(ArgumentError("`max_satellites_per_signal` must be at least 1"))
    T = Float64(ustrip(s, cycle_time)) * s
    0.0s < T < Inf * s ||
        throw(ArgumentError("the navigation cycle time must be positive and finite"))
    VectorTrackingSettings(
        inner,
        combine_signals,
        config,
        T,
        Float64(ustrip(lock_cn0_threshold)),
        Int(max_satellites_per_signal),
        Int(num_prompts_for_cn0_estimation),
        Int(approximate_year),
        enable_ionospheric_correction,
        enable_tropospheric_correction,
    )
end

# The estimator, built here, after the navigation engine it holds.
function VectorPLLAndDLL(signals::Union{AbstractGNSSSignal,Tuple}...; settings...)
    isempty(signals) && throw(
        ArgumentError(
            "vector tracking needs at least one ranging signal; for a host that binds " *
            "its signal groups later, build `VectorTrackingSettings(; ...)`",
        ),
    )
    with_signal_groups(VectorTrackingSettings(; settings...), signals...)
end

"""
    with_signal_groups(estimator_or_settings, signals...) -> estimator

The estimator for the ranging `signals`, given as for [`VectorPLLAndDLL`](@ref): one
signal, or a satellite's signal group with the driver first. A host that knows its
satellites' signal groups, such as Tracking.jl's `TrackState`, calls it on whatever
it is handed:

  - [`VectorTrackingSettings`](@ref): builds the [`VectorPLLAndDLL`](@ref) for
    `signals`, checked as when built with them;
  - a `VectorPLLAndDLL`: returns it if each of `signals` is one of its groups (any
    subset, in any order, a group's passengers in any order), and throws if a group
    is not, as the estimator could not step it;
  - any other estimator: needs no groups and is returned as it is.
"""
with_signal_groups(estimator::AbstractDopplerEstimator, signals...) = estimator

function with_signal_groups(settings::VectorTrackingSettings, signals...)
    isempty(signals) &&
        throw(ArgumentError("vector tracking needs at least one ranging signal"))
    signal_groups = map(_signal_group, signals)
    foreach(_check_signal_group, signal_groups)
    # A signal may ride in other groups as a passenger; records are routed by driver.
    allunique(map(group -> typeof(first(group)), signal_groups)) ||
        throw(ArgumentError("each ranging signal can drive only one group"))
    navigation = VectorNavigation(
        settings.config,
        signal_groups,
        settings.inner;
        cycle_time = settings.cycle_time,
        lock_cn0_threshold = settings.lock_cn0_threshold,
        max_satellites_per_signal = settings.max_satellites_per_signal,
        num_prompts_for_cn0_estimation = settings.num_prompts_for_cn0_estimation,
        approximate_year = settings.approximate_year,
        enable_ionospheric_correction = settings.enable_ionospheric_correction,
        enable_tropospheric_correction = settings.enable_tropospheric_correction,
    )
    VectorPLLAndDLL(settings.inner, navigation, settings.combine_signals)
end

function with_signal_groups(estimator::VectorPLLAndDLL, signals...)
    bound = map(_group_signal_types, estimator.navigation.groups)
    for group in map(_signal_group, signals)
        types = _group_signal_types(group)
        any(==(types), bound) && continue
        driver = first(group)
        throw(
            ArgumentError(
                any(other -> first(other) == typeof(driver), bound) ?
                "this vector-tracking estimator lists other passengers with " *
                "$(nameof(typeof(driver)))" :
                "this vector-tracking estimator was not built for " *
                "$(nameof(typeof(driver))); build it from `VectorTrackingSettings` " *
                "to have the groups bound",
            ),
        )
    end
    estimator
end

# A group's signal types to compare groups by: the driver, then the set of passengers.
_group_signal_types(group::Tuple{Vararg{AbstractGNSSSignal}}) =
    (typeof(first(group)), Set{DataType}(map(typeof, Base.tail(group))))
_group_signal_types(group::VTSlotGroup) =
    (typeof(group.signal), Set{DataType}(map(typeof, group.passengers)))

# A ranging signal on its own, or a satellite's signals with the driver first.
_signal_group(signal::AbstractGNSSSignal) = (signal,)
function _signal_group(signals::Tuple)
    isempty(signals) ||
        all(signal -> signal isa AbstractGNSSSignal, signals) ||
        throw(ArgumentError("a signal group lists signals, the driver first"))
    isempty(signals) && throw(ArgumentError("a signal group needs at least its driver"))
    signals
end

function _check_signal_group(signals::Tuple)
    driver = first(signals)
    allunique(map(typeof, signals)) ||
        throw(ArgumentError("a signal group lists each signal only once"))
    system = typeof(get_time_system(driver))
    all(signal -> typeof(get_time_system(signal)) == system, signals) || throw(
        ArgumentError(
            "the signals of a group are one satellite's: list signals of one " *
            "constellation",
        ),
    )
    all(signal -> get_code_frequency(signal) == get_code_frequency(driver), signals) &&
    all(signal -> get_center_frequency(signal) == get_center_frequency(driver), signals) ||
        throw(
            ArgumentError(
                "the signals of a group share the driver's code rate and carrier: " *
                "list signals of one band and chip rate, such as a pilot and its " *
                "data component",
            ),
        )
    any(signal -> !iszero(get_data_frequency(signal)), signals) || throw(
        ArgumentError(
            "vector tracking decodes the navigation data of the driver, or of a " *
            "passenger for a dataless driver; $(nameof(typeof(driver))) carries none: " *
            "list its data component with it, e.g. `(GalileoE1C(), GalileoE1B())`",
        ),
    )
    nothing
end

"""
    step_loop(estimator::VectorPLLAndDLL, state, record::LoopRecord, words, landing_sample)
        -> (state, carrier_doppler, code_doppler)

One record through the vector loop (see [`VectorPLLAndDLL`](@ref)): the satellite
joins the navigation engine on its first record, takes up the latest cycle's
admission, release and corrections, snapshots itself when the record crosses a
navigation epoch, runs the loop, and feeds the prompt to its bit clock, decoder and
C/N₀ estimator. The record on which the last satellite reaches an epoch runs that
epoch's cycle. Out of the vector loop the Dopplers equal `step_loop(estimator.inner, …)`'s.

The record's `cn0` (host estimate) weights the measurements and decides lock; with
`NaN` the engine estimates the C/N₀ from the prompts.
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

# Find the record's group by type, so the search folds at compile time.
@inline _step_vector_record(
    estimator,
    nav,
    ::Tuple{},
    g,
    state,
    record::LoopRecord,
    words,
    landing_sample,
) = throw(
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
        _step_vector_record(
            estimator,
            nav,
            Base.tail(groups),
            g + 1,
            state,
            record,
            words,
            landing_sample,
        )
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
    state, carrier_doppler, code_doppler =
        _step_satellite(estimator, state, record, words, landing_sample)
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

# The satellite's slot, registering it on the record's PRN if it has none or lost it.
# A re-acquired PRN gets its old slot back with its decoded data; any other takes a
# free slot, and only when none is free does the group grow.
function _register!(
    nav::VectorNavigation,
    group::VTSlotGroup,
    g::Int,
    state::SatVectorPLLAndDLL,
    record::LoopRecord,
)
    slots = group.slots
    index = state.slot
    if 1 <= index <= length(slots)
        slot = slots[index]
        slot.occupied && slot.registration == state.registration && return state, slot
    end
    prn = record.prn
    prn > 0 || throw(
        ArgumentError(
            "vector tracking needs to know the satellite: set the record's `prn`",
        ),
    )
    index = _find_slot(slots, prn)
    if index == 0
        push!(slots, _new_slot(group, prn, group.prototype))
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

# Reset `slot` in place for a registering satellite, reusing its buffers. The decoder
# restarts its sync, keeping the decoded data only for the same PRN.
function _reset_slot!(
    nav::VectorNavigation,
    slot::VTSlot,
    prn::Int,
    state,
    record::LoopRecord,
)
    slot.running_decoder =
        slot.prn == prn && slot.registration > 0 ?
        reset_decoder_state!(slot.running_decoder) :
        _rebind_decoder(slot.running_decoder, prn)
    slot.prn = prn
    slot.occupied = true
    slot.registration = nav.registrations
    slot.active = false
    slot.bit_buffer = _fresh_bit_buffer(slot.bit_buffer)
    slot.cn0_estimator = _reset_cn0_estimator(slot.cn0_estimator)
    slot.host_cn0_dbhz = NaN
    slot.held_cn0_dbhz = NaN
    slot.sync_fold_end = typemin(Int)
    fs = _sampling_frequency_hz(record)
    start = record.sample_index - record.integrated_samples
    slot.last_end_sample = start
    slot.last_end_time = start / fs
    slot.data_last_end_sample = start
    slot.data_last_end_time = start / fs
    slot.data_last_code_phase_fraction = NaN
    for (i, passenger) in enumerate(slot.passengers)
        slot.passengers[i] = VTPassenger(_reset_cn0_estimator(passenger.cn0_estimator))
    end
    slot.last_integration_time =
        uconvert(s, record.integrated_samples / record.sampling_frequency)
    slot.chips_since_epoch = 0.0
    if nav.pending_epoch == typemin(Int)
        nav.pending_epoch = _epoch_at_or_after(nav, slot.last_end_time)
    end
    slot.first_epoch = _epoch_at_or_after(nav, slot.last_end_time)
    slot.snapshot_epoch = typemin(Int)
    slot.decoder = slot.running_decoder
    # Registered out of the vector loop; a state still in it takes up the release in
    # `_take_up_cycle`, re-seeding its scalar loop.
    slot.estimator_state = _disable_vector_tracking(state)
    slot.in_lock = false
    slot.pvt_ready = false
    slot.release_reason = VT_NOT_RELEASED
    slot.restart_cycle = -1
    slot.correction_cycle = -1
    slot.member_index = 0
    nothing
end

# `decoder`'s containers rebound to `prn`, with sync, vote tally and data cleared.
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

# Take up the latest cycle's decisions once: admission, restart, release (re-seeding
# from the replica at the landing) and corrections sized for that landing.
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
        state =
            _release_from_vector_tracking(state, carrier_doppler * Hz, code_doppler * Hz)
    end
    if in_loop && slot.correction_cycle == id
        code_update, carrier_update, lead =
            _correction_at_landing(nav, group, slot, record, words, landing)
        state =
            _set_vector_corrections(state, code_update * Hz, carrier_update * Hz, lead * s)
    end
    SatVectorPLLAndDLL(state; cycle_id = id)
end

# The slot's corrections for a command landing at sample `landing`, and the lead (s)
# after the cycle's epoch.
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
        slot.chips_since_epoch +
        (landing - slot.last_end_sample) / fs * (code_frequency + code_word)
    carrier_doppler, _ = mean_nco_word(words, landing, landing)
    code_update, carrier_update =
        _member_corrections(nav, slot, slot.member_index, lead, chips, carrier_doppler)
    code_update, carrier_update, lead
end

# ─────────────────────────────────────────────────────────────────────────────
# The epoch snapshot

# Snapshot the slot at the pending epoch if this record crosses it: code phase from the
# last data-symbol edge, decoder, Dopplers, C/N₀, lock and the state with its
# accumulated discriminators. Returns the state with the accumulators emptied.
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
    # This satellite, and every other that could snapshot the pending epoch, is past
    # it (records resumed after a gap): move the epochs on.
    if nav.num_snapshots == 0 &&
       start_time > epoch_time &&
       !_fold_slot_groups(_can_reach_epoch, false, nav.groups, nav, epoch_time, end_time)
        nav.pending_epoch = _epoch_at_or_after(nav, start_time)
        epoch_time = _epoch_time(nav, nav.pending_epoch)
    end
    slot.snapshot_epoch < nav.pending_epoch &&
    slot.first_epoch <= nav.pending_epoch &&
    start_time <= epoch_time < end_time || return state
    code_frequency = ustrip(Hz, get_code_frequency(group.signal))
    epoch_sample = epoch_time * fs
    _, code_word = mean_nco_word(words, slot.last_end_sample, epoch_sample)
    chips_to_epoch = (epoch_time - start_time) * (code_frequency + code_word)
    carrier_doppler, code_doppler = mean_nco_word(words, epoch_sample, epoch_sample)
    # The code phase at the epoch from the data signal's last symbol edge. Its records
    # may already be past the epoch (a data passenger folded ahead of the driver), which
    # moves the decoder back; `minmax` then takes the mean word from the epoch to that end.
    _, data_code_word =
        mean_nco_word(words, minmax(slot.data_last_end_sample, epoch_sample)...)
    data_chips_to_epoch =
        (epoch_time - slot.data_last_end_time) * (code_frequency + data_code_word)
    decoder, code_phase = _decoder_at_code_phase(
        slot.running_decoder,
        _code_phase_from_symbol_edge(group.data_signal, slot) + data_chips_to_epoch,
        code_frequency,
    )
    cn0_dbhz = _cn0_dbhz(
        slot.host_cn0_dbhz,
        slot.held_cn0_dbhz,
        slot.cn0_estimator,
        slot.last_integration_time,
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
    slot.dll_variance_factor = slot.last_dll_variance_factor
    slot.in_lock = in_lock
    slot.pvt_ready =
        in_lock && is_decoding_completed_for_positioning(decoder) && is_sat_healthy(decoder)
    slot.chips_since_epoch = -chips_to_epoch
    slot.snapshot_epoch = nav.pending_epoch
    nav.num_snapshots += 1
    _reset_discriminator_accumulators(state)
end

# Cap on the weighting C/N₀: the moment estimator reads a noise-free (simulated) signal
# as infinite, which would give the filter a noiseless measurement.
const MAX_CN0_DBHZ = 80.0

_capped_cn0_dbhz(cn0_estimator, integration_time) =
    min(Float64(ustrip(estimate_cn0(cn0_estimator, integration_time))), MAX_CN0_DBHZ)

# The host's C/N₀ estimate if given, else the engine's own (`held_cn0_dbhz` while its
# estimator refills, see `_restart_cn0`); both capped.
_cn0_dbhz(host_cn0_dbhz, held_cn0_dbhz, cn0_estimator, integration_time) =
    isnan(host_cn0_dbhz) ? _own_cn0_dbhz(held_cn0_dbhz, cn0_estimator, integration_time) :
    min(host_cn0_dbhz, MAX_CN0_DBHZ)

_own_cn0_dbhz(held_cn0_dbhz, cn0_estimator, integration_time) =
    !isnan(held_cn0_dbhz) && _refilling(cn0_estimator) ? min(held_cn0_dbhz, MAX_CN0_DBHZ) :
    _capped_cn0_dbhz(cn0_estimator, integration_time)

_refilling(cn0_estimator::MomentsCN0Estimator) =
    length(cn0_estimator) < length(get_prompt_buffer(cn0_estimator))

# `(held, estimator)` for a record of `integration_time` after one of
# `previous_integration_time`: unchanged at about the same length, else restarted (the
# moment estimator would mix two noise scales under one integration time) with the old
# estimate held while it refills, as a few prompts read as an arbitrarily high C/N₀.
function _restart_cn0(
    held_cn0_dbhz,
    cn0_estimator,
    previous_integration_time,
    integration_time,
)
    _record_length_changed(previous_integration_time, integration_time) ||
        return held_cn0_dbhz, cn0_estimator
    held =
        iszero(length(cn0_estimator)) ? NaN :
        _own_cn0_dbhz(held_cn0_dbhz, cn0_estimator, previous_integration_time)
    held, _reset_cn0_estimator(cn0_estimator)
end

# Whether some satellite of the group can still snapshot the pending epoch.
function _can_reach_epoch(acc, group::VTSlotGroup, nav::VectorNavigation, epoch_time, now)
    acc && return true
    for slot in group.slots
        slot.occupied &&
            !_is_stale(nav, slot, now) &&
            slot.first_epoch <= nav.pending_epoch &&
            slot.last_end_time <= epoch_time &&
            return true
    end
    false
end

# The replica's code phase (chips) at the last record end from the last data-symbol
# edge: the bit clock's whole code blocks plus the phase past the block boundary.
# Meaningless (and unread) before bit sync.
function _code_phase_from_symbol_edge(signal::AbstractGNSSSignal, slot::VTSlot)
    bit_buffer = slot.bit_buffer
    blocks = bit_buffer.found ? bit_buffer.prompt_accumulator_integrated_code_blocks : 0
    fraction = slot.data_last_code_phase_fraction
    fraction = isnan(fraction) ? 0.0 : fraction
    blocks * get_code_length(signal) + fraction
end

# Normalise `code_phase` into the current symbol: a phase outside it (just short of the
# edge, or past the next one) shifts the decoder's symbol count instead, which is all
# the transmit time reads.
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

# Run the pending cycle once every satellite has snapshotted it, ignoring those that
# joined later or are stale. Returns whether it ran.
function _run_cycle_if_due!(nav::VectorNavigation, record::LoopRecord)
    nav.num_snapshots == 0 && return false
    now = record.sample_index / _sampling_frequency_hz(record)
    _fold_slot_groups(_all_snapshotted, true, nav.groups, nav, now) || return false
    _navigation_cycle!(nav, now)
    true
end

# If the slot is already past the pending epoch, run that cycle now with the
# satellites that made it. Returns whether it ran.
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
        slot.snapshot_epoch == nav.pending_epoch ||
            slot.first_epoch > nav.pending_epoch ||
            return false
    end
    true
end

_is_stale(nav::VectorNavigation, slot::VTSlot, now) =
    slot.last_end_time < now - 2 * _cycle_seconds(nav)

# Before a cycle: mark the slots it reads, clear release reasons, and free stale slots
# (storage kept, members released). Returns whether a member was dropped.
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
#   - Not running: the scalar PVT. A fresh fix seeds the filter and puts its
#     satellites into the vector loop; satellites still in it from before are
#     released if ineligible and restarted otherwise.
#   - Running: one filter cycle (`_run_cycle!`), admitting satellites decoded,
#     healthy, in lock and a degree above the horizon; members out of lock stay in
#     the loop unmeasured. After `insufficient_meas_timeout` of unsolvable epochs, or
#     with no member left, all are released and the next cycle solves the scalar PVT.
#   - `config = nothing`: the scalar PVT only.
function _navigation_cycle!(nav::VectorNavigation, now)
    groups = nav.groups
    buffers = nav.buffers
    epoch = nav.pending_epoch
    cycle_time =
        nav.cycle_epoch == typemin(Int) ? nav.cycle_time :
        (epoch - nav.cycle_epoch) * nav.cycle_time
    nav.cycle_id += 1
    dropped = _fold_slot_groups(_prepare_slots!, false, groups, nav, now)
    enabled = released = fell_back = false
    if nav.enabled && nav.running
        released, fell_back = _run_cycle!(nav, groups, cycle_time)
    else
        # Scalar solve: no per-member report.
        empty_keeping_capacity!(nav.member_sats)
        previous_pvt = nav.pvt
        pvt = _solve_scalar_pvt!(nav, groups)
        # `calc_pvt!` returns its input unchanged when it cannot solve, so identity
        # tests freshness.
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

# After the loop step: feed the driver's prompt to the C/N₀ estimator, and to the bit
# clock and decoder if the driver is the data signal; move the replica to the record's
# end.
function _advance_slot!(group::VTSlotGroup, slot::VTSlot, record::LoopRecord, words)
    signal = group.signal
    fs = _sampling_frequency_hz(record)
    code_frequency = ustrip(Hz, get_code_frequency(signal))
    _, code_word = mean_nco_word(words, slot.last_end_sample, record.sample_index)
    slot.chips_since_epoch +=
        (record.sample_index - slot.last_end_sample) / fs * (code_frequency + code_word)
    _drives_data_signal(group) && _advance_data_signal!(group, slot, record)
    slot.held_cn0_dbhz, slot.cn0_estimator = _restart_cn0(
        slot.held_cn0_dbhz,
        slot.cn0_estimator,
        slot.last_integration_time,
        uconvert(s, record.integrated_samples / record.sampling_frequency),
    )
    slot.cn0_estimator = update(slot.cn0_estimator, get_prompt(record.filtered_correlator))
    slot.host_cn0_dbhz = record.cn0
    slot.last_end_sample = record.sample_index
    slot.last_end_time = record.sample_index / fs
    slot.last_integration_time =
        uconvert(s, record.integrated_samples / record.sampling_frequency)
    slot.last_early_late_spacing = _early_late_spacing_chips(record)
    slot.last_dll_variance_factor = _dll_variance_factor(record.filtered_correlator)
    nothing
end

# Whether the driver is the data signal (signals are unique per group, so type tells).
_drives_data_signal(::VTSlotGroup{S,P,C}) where {S,P,C} = S === C

# The tap spacing of a record's correlator in chips.
_early_late_spacing_chips(record::LoopRecord) =
    get_early_late_sample_spacing(
        record.filtered_correlator,
        record.sampling_frequency,
        get_code_frequency(record.signal),
    ) * ustrip(Hz, get_code_frequency(record.signal)) / _sampling_frequency_hz(record)

# Feed a data-signal record's prompt, in the driver's carrier-phase frame, to the bit
# clock and decoder, and move the data signal to the record's end.
function _advance_data_signal!(group::VTSlotGroup, slot::VTSlot, record::LoopRecord)
    signal = group.data_signal
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
    bit_prompt =
        get_prompt(record.filtered_correlator) *
        _carrier_phase_derotation(get_carrier_phase_offset(group.signal), signal)
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
    soft_bits = bit_buffer.soft_bits
    if !isempty(soft_bits)
        slot.running_decoder = decode!(slot.running_decoder, soft_bits, length(soft_bits))
        empty!(soft_bits)
    end
    code_length = get_code_length(signal)
    slot.data_last_end_sample = record.sample_index
    slot.data_last_end_time = record.sample_index / _sampling_frequency_hz(record)
    slot.data_last_code_phase_fraction =
        isnan(record.code_phase) ? NaN :
        mod(record.code_phase + code_length / 2, code_length) - code_length / 2
    nothing
end
