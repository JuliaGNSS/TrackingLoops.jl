# ─────────────────────────────────────────────────────────────────────────────
# The per-integration side of the navigation engine (see `step_loop!`).
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
function VectorTrackingLoop(
    signals::Union{AbstractGNSSSignal,Pair{<:AbstractGNSSSignal,<:AbstractGNSSSignal}}...;
    inner::AbstractTrackingLoop = ConventionalAssistedPLLAndDLL(),
    filter_config::Union{NavigationFilterConfig,Nothing} = NavigationFilterConfig(),
    navigation_filter_update_interval = 100.0ms,
    min_tracking_cn0 = 30.0dBHz,
    residual_pll_loop_filter_bandwidth = default_residual_pll_loop_filter_bandwidth(),
    satellite_capacity_per_signal::Integer = 16,
    approximate_year::Integer = year(now(UTC)),
    enable_ionospheric_correction::Bool = true,
    enable_tropospheric_correction::Bool = true,
)
    _has_fll_path(inner) || throw(
        ArgumentError(
            "a satellite the vector loop releases pulls in again through the FLL " *
            "path of an FLL-assisted carrier filter; use " *
            "`ConventionalAssistedPLLAndDLL()` or `DelayCompensatedTrackingLoop()` as " *
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
    satellite_capacity_per_signal >= 1 ||
        throw(ArgumentError("`satellite_capacity_per_signal` must be at least 1"))
    T = Float64(ustrip(s, navigation_filter_update_interval)) * s
    0.0s < T < Inf * s ||
        throw(ArgumentError("`navigation_filter_update_interval` must be positive and finite"))
    residual_pll_loop_filter_bandwidth > 0.0Hz ||
        throw(ArgumentError("`residual_pll_loop_filter_bandwidth` must be positive"))
    engine = NavFilterState(
        filter_config,
        pairs,
        inner;
        navigation_filter_update_interval = T,
        min_tracking_cn0_dbhz = Float64(ustrip(min_tracking_cn0)),
        satellite_capacity_per_signal = Int(satellite_capacity_per_signal),
        approximate_year,
        enable_ionospheric_correction,
        enable_tropospheric_correction,
    )
    VectorTrackingLoop(inner, engine, Float64(ustrip(Hz, residual_pll_loop_filter_bandwidth)) * Hz)
end

"""
    step_loop!(estimator::VectorTrackingLoop, state, integration::CoherentIntegration, nco_history, application_sample_index)
        -> (state, carrier_doppler, code_doppler)

One integration of a satellite through the vector loop (see [`VectorTrackingLoop`](@ref)).
The satellite registers with the navigation engine on its first integration, of either
signal. An integration of its

  - **driver** takes up the latest cycle's admission, release and commands,
    snapshots the replica, Dopplers, accumulated discriminators and C/N₀ when it
    crosses a navigation epoch, and runs the loop;
  - **decoding signal** feeds its new soft bits to the decoder and notes where the bit
    clock stands (the snapshot's bit count); it returns the command in force and
    the state unchanged;
  - plain signal does both; any other passenger does nothing, as in a scalar loop.

A satellite's snapshot of an epoch is complete once an integration of each signal has
crossed it, in either order; the integration completing the last satellite's snapshot
runs that epoch's cycle. Out of the vector loop the driver's Dopplers equal
`step_loop!(estimator.inner, …)`'s.
"""
@inline step_loop!(
    estimator::VectorTrackingLoop,
    state::VectorLoopState,
    integration::CoherentIntegration,
    nco_history,
    application_sample_index::Int64,
) = _step_vector_integration(
    estimator,
    estimator.engine,
    estimator.engine.groups,
    1,
    state,
    integration,
    nco_history,
    application_sample_index,
)

# Find the group the integration's signal drives or is decoded on, by type so the search
# folds at compile time. An integration of a signal the estimator was not built for is an
# ignored passenger unless it is the satellite's driver.
@inline function _step_vector_integration(estimator, nav, ::Tuple{}, g, state, integration::CoherentIntegration, nco_history, application_sample_index)
    _is_driver_integration(state, integration) && _throw_unknown_driver(integration)
    _step_passenger(state, integration, nco_history, application_sample_index)
end
@inline function _step_vector_integration(
    estimator,
    nav,
    groups::Tuple,
    g,
    state,
    integration::CoherentIntegration{S},
    nco_history,
    application_sample_index,
) where {S}
    group = first(groups)
    if group.driver_signal isa S
        _is_driver_integration(state, integration) || _throw_foreign_driver(state, integration)
        _step_vector_integration(estimator, nav, group, g, state, integration, nco_history, application_sample_index, true, group.decoding_signal isa S)
    elseif group.decoding_signal isa S
        # A satellite another group drives: a passenger of it this group must not take.
        _is_driven_by(state, group.driver_signal) ||
            return _step_passenger(state, integration, nco_history, application_sample_index)
        _step_vector_integration(estimator, nav, group, g, state, integration, nco_history, application_sample_index, false, true)
    else
        _step_vector_integration(estimator, nav, Base.tail(groups), g + 1, state, integration, nco_history, application_sample_index)
    end
end

@noinline _throw_unknown_driver(integration) = throw(
    ArgumentError(
        "this vector-tracking estimator was not built for " *
        "$(nameof(typeof(integration.signal))); list it among its signals",
    ),
)

@noinline _throw_foreign_driver(state, integration) = throw(
    ArgumentError(
        "a satellite driven by another signal was stepped with an integration of " *
        "$(nameof(typeof(integration.signal))), which this estimator ranges on as a driver",
    ),
)

# One integration in its roles: `driver` (closes the loops) and `decoding` (feeds the
# decoder); both for a plain signal.
function _step_vector_integration(
    estimator::VectorTrackingLoop,
    nav::NavFilterState,
    group::VectorTrackingSlotGroup,
    g::Int,
    state::VectorLoopState,
    integration::CoherentIntegration,
    nco_history,
    application_sample_index::Int64,
    driver::Bool,
    decoding::Bool,
)
    state, slot = _register!(nav, group, g, state, integration)
    application_sample_index = application_sample_index == NO_LANDING_SAMPLE ? integration.sample_index : application_sample_index
    driver && (state = _take_up_cycle(nav, group, slot, state, integration, nco_history, application_sample_index))
    if _run_overdue_cycle!(nav, slot, integration) && driver
        state = _take_up_cycle(nav, group, slot, state, integration, nco_history, application_sample_index)
    end
    _move_past_gap!(nav, driver ? slot.driver_last_end_time : slot.decoding_last_end_time, integration)
    driver && (state = _snapshot_driver!(nav, group, slot, state, integration, nco_history))
    decoding && _snapshot_decoding!(nav, group, slot, integration, nco_history)
    _complete_snapshot!(nav, group, slot)
    if _run_cycle_if_due!(nav, integration) && driver
        state = _take_up_cycle(nav, group, slot, state, integration, nco_history, application_sample_index)
    end
    if driver
        state, carrier_doppler, code_doppler = _step_satellite(estimator, state, integration, nco_history, application_sample_index)
        _advance_driver!(group, slot, integration, nco_history)
        # In the vector loop the residual PLL stepped on this prompt (`_measure_carrier`);
        # out of it, the history starts afresh when the satellite is added again.
        state.vector_controlled ? _push_prompt!(slot.prompt_history, state.residual_pll) :
        _clear_prompts!(slot.prompt_history)
    else
        carrier_doppler, code_doppler = _command_in_force(integration, nco_history, application_sample_index)
    end
    decoding && _advance_decoding!(group, slot, integration, state, nco_history, driver)
    state, carrier_doppler, code_doppler
end

_update_interval_seconds(nav::NavFilterState) = ustrip(s, nav.navigation_filter_update_interval)
_sampling_frequency_hz(integration::CoherentIntegration) = Float64(ustrip(Hz, integration.sampling_frequency))
_epoch_time(nav::NavFilterState, epoch::Int) = epoch * _update_interval_seconds(nav)

# The first epoch whose `_epoch_time` is not before `t`, compared exactly as
# `_crosses_pending_epoch` does (`t / T` may round to either side).
function _epoch_at_or_after(nav::NavFilterState, t)
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

# The satellite's slot, registering it on the integration's PRN if it has none or lost it.
# A re-acquired PRN gets its old slot back with its decoded data; any other takes a
# free slot, and only when none is free does the group grow.
function _register!(nav::NavFilterState, group::VectorTrackingSlotGroup, g::Int, state::VectorLoopState, integration::CoherentIntegration)
    slots = group.slots
    index = state.slot_index
    if 1 <= index <= length(slots)
        slot = slots[index]
        slot.occupied && slot.slot_generation == state.slot_generation && return state, slot
    end
    prn = integration.prn
    prn > 0 || throw(
        ArgumentError("vector tracking needs to know the satellite: set the integration's `prn`"),
    )
    index = _find_slot(slots, prn)
    if index == 0
        push!(slots, VectorTrackingSlot(group.decoding_signal, prn, group.prototype, group.prompt_history_length))
        index = length(slots)
    end
    slot = slots[index]
    nav.next_slot_generation += 1
    state = _registered(state, index, nav.next_slot_generation)
    _reset_slot!(nav, slot, prn, state, integration)
    state, slot
end

# The slot that holds or held `prn`, else any free one; 0 when none is free.
function _find_slot(slots, prn)
    free = 0
    for (index, slot) in enumerate(slots)
        if slot.prn == prn && slot.slot_generation > 0
            return index
        end
        !slot.occupied && free == 0 && (free = index)
    end
    free
end

# Reset `slot` in place for a registering satellite. The decoder restarts its sync,
# keeping the decoded data only for the same PRN; both signals start at the integration's
# start.
function _reset_slot!(nav::NavFilterState, slot::VectorTrackingSlot, prn::Int, state, integration::CoherentIntegration)
    slot.running_decoder =
        slot.prn == prn && slot.slot_generation > 0 ? reset_decoder_state!(slot.running_decoder) :
        reset_decoder_state!(slot.running_decoder, prn)
    slot.prn = prn
    slot.occupied = true
    slot.slot_generation = nav.next_slot_generation
    slot.in_cycle = false
    fs = _sampling_frequency_hz(integration)
    start = integration.sample_index - integration.num_integrated_samples
    slot.driver_last_end_sample_index = start
    slot.driver_last_end_time = start / fs
    slot.last_code_phase = NaN
    slot.last_coherent_integration_time = uconvert(s, integration.num_integrated_samples / integration.sampling_frequency)
    slot.chips_since_epoch = 0.0
    slot.decoding_last_end_sample_index = start
    slot.decoding_last_end_time = start / fs
    slot.decoding_bit_synced = false
    slot.decoding_num_code_periods_into_bit = 0
    slot.decoding_code_phase_fraction = NaN
    slot.sync_batch_end_sample_index = typemin(Int)
    slot.bit_accumulator = zero(ComplexF64)
    slot.bit_from_edge = false
    _clear_prompts!(slot.prompt_history)
    if nav.pending_epoch_index == typemin(Int)
        nav.pending_epoch_index = _epoch_at_or_after(nav, slot.driver_last_end_time)
    end
    # A snapshot the slot held is gone with it.
    slot.snapshot_epoch_index == nav.pending_epoch_index && (nav.num_snapshots -= 1)
    # An integration starting exactly at an epoch crosses it, so that epoch is the first; if
    # it is not pending yet, the slot is past it by the time it is, and the cycle does
    # not wait for it (`_all_snapshotted`).
    slot.first_epoch_index = _epoch_at_or_after(nav, slot.driver_last_end_time)
    slot.driver_epoch_index = typemin(Int)
    slot.decoding_epoch_index = typemin(Int)
    slot.snapshot_epoch_index = typemin(Int)
    slot.decoder = slot.running_decoder
    slot.epoch_decoder = slot.running_decoder
    slot.epoch_bit_synced = false
    # Registered out of the vector loop; a state still in it takes up the release in
    # `_take_up_cycle`, re-seeding its scalar loop.
    slot.loop_state = _release_from_vector_loop(state)
    slot.measurement_ready = false
    slot.positioning_ready = false
    slot.release_reason = VT_NOT_RELEASED
    slot.command_cycle_index = -1
    slot.member_index = 0
    nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Taking up a cycle

# Take up the latest cycle's decisions once: admission, release (re-seeding the scalar
# loop at the application sample) and commands sized for it.
function _take_up_cycle(
    nav::NavFilterState,
    group::VectorTrackingSlotGroup,
    slot::VectorTrackingSlot,
    state::VectorLoopState,
    integration::CoherentIntegration,
    nco_history,
    application_sample_index::Int64,
)
    id = nav.cycle_index
    state.cycle_index == id && return state
    slot_in_loop = slot.loop_state.vector_controlled
    if slot_in_loop && !state.vector_controlled
        state = _add_to_vector_loop(state)
    elseif !slot_in_loop && state.vector_controlled
        state = _hand_over_to_scalar_loop(state, nco_history, application_sample_index, _sampling_frequency_hz(integration))
    end
    if slot_in_loop && slot.command_cycle_index == id
        code_update, ramp, lead =
            _commands_at_application(nav, group, slot, integration, nco_history, application_sample_index)
        state = _set_vector_commands(state, code_update * Hz, ramp, lead * s, nco_history, application_sample_index)
    end
    VectorLoopState(state; cycle_index = id)
end

# The slot's commands applied at sample `application_sample_index`: the code correction, the
# carrier Doppler ramp from the predicted Dopplers there and one cycle later (with the
# replica's code moved on at its NCO command), and the lead (s) after the cycle's epoch.
function _commands_at_application(
    nav::NavFilterState,
    group::VectorTrackingSlotGroup,
    slot::VectorTrackingSlot,
    integration::CoherentIntegration,
    nco_history,
    application_sample_index::Int64,
)
    fs = _sampling_frequency_hz(integration)
    # A command applied on the epoch itself may read a rounding error early.
    lead = application_sample_index / fs - _epoch_time(nav, nav.cycle_epoch_index)
    -1 / fs <= lead <= MAX_APPLICATION_LEAD_CYCLES * _update_interval_seconds(nav) || throw(
        ArgumentError(
            "a command must take effect between zero and 2.5 navigation cycles after the " *
            "epoch whose commands it carries",
        ),
    )
    lead = max(lead, 0.0)
    T = _update_interval_seconds(nav)
    code_frequency = ustrip(Hz, get_chip_rate(group.driver_signal))
    code_command = calc_mean_nco_command(nco_history, slot.driver_last_end_sample_index, application_sample_index).code_doppler
    chips =
        slot.chips_since_epoch + (application_sample_index - slot.driver_last_end_sample_index) / fs * (code_frequency + code_command)
    code_update, doppler = _member_commands(nav, slot, slot.member_index, lead, chips)
    _, next_doppler =
        _member_commands(nav, slot, slot.member_index, lead + T, chips + T * (code_frequency + code_command))
    code_update, CarrierDopplerRamp(doppler, (next_doppler - doppler) / T, application_sample_index / fs), lead
end

# ─────────────────────────────────────────────────────────────────────────────
# The epoch snapshot

# Integrations resumed after a gap: an integration starting past the pending epoch, while no
# satellite can still snapshot it, moves the epochs on to where the integrations are.
function _move_past_gap!(nav::NavFilterState, start_time, integration::CoherentIntegration)
    epoch_time = _epoch_time(nav, nav.pending_epoch_index)
    nav.num_snapshots == 0 && start_time > epoch_time || return nothing
    end_time = integration.sample_index / _sampling_frequency_hz(integration)
    _fold_groups(_can_reach_epoch, false, nav.groups, nav, epoch_time, end_time) &&
        return nothing
    nav.pending_epoch_index = _epoch_at_or_after(nav, start_time)
    nothing
end

# Whether an integration of one of the slot's signals (last integration ended at
# `driver_last_end_time`, its snapshot half taken at `half_epoch_index`) crosses the pending
# epoch still to snapshot.
function _crosses_pending_epoch(nav::NavFilterState, slot::VectorTrackingSlot, half_epoch_index, driver_last_end_time, integration::CoherentIntegration)
    epoch = nav.pending_epoch_index
    epoch_time = _epoch_time(nav, epoch)
    end_time = integration.sample_index / _sampling_frequency_hz(integration)
    half_epoch_index < epoch && slot.first_epoch_index <= epoch && driver_last_end_time <= epoch_time < end_time
end

# The driver's half of the snapshot at the pending epoch if this integration crosses it:
# replica code phase, Dopplers, C/N₀ (from the integration's estimator) and the state with
# its accumulated discriminators. Returns the state, its accumulators emptied if it
# crossed.
function _snapshot_driver!(
    nav::NavFilterState,
    group::VectorTrackingSlotGroup,
    slot::VectorTrackingSlot,
    state::VectorLoopState,
    integration::CoherentIntegration,
    nco_history,
)
    _crosses_pending_epoch(nav, slot, slot.driver_epoch_index, slot.driver_last_end_time, integration) ||
        return state
    fs = _sampling_frequency_hz(integration)
    epoch_time = _epoch_time(nav, nav.pending_epoch_index)
    code_frequency = ustrip(Hz, get_chip_rate(group.driver_signal))
    epoch_sample_index = epoch_time * fs
    code_command = calc_mean_nco_command(nco_history, slot.driver_last_end_sample_index, epoch_sample_index).code_doppler
    chips_to_epoch = (epoch_time - slot.driver_last_end_time) * (code_frequency + code_command)
    (; carrier_doppler, code_doppler) = calc_mean_nco_command(nco_history, epoch_sample_index, epoch_sample_index)
    coherent_integration_time = uconvert(s, integration.num_integrated_samples / integration.sampling_frequency)
    slot.cn0_dbhz =
        min(Float64(ustrip(estimate_cn0(integration.cn0_estimator, coherent_integration_time))), MAX_CN0_DBHZ)
    slot.loop_state = state
    slot.epoch_driver_code_phase = slot.last_code_phase + chips_to_epoch
    slot.carrier_doppler = carrier_doppler * Hz
    slot.code_doppler = code_doppler * Hz
    slot.coherent_integration_time = slot.last_coherent_integration_time
    slot.early_late_spacing = slot.last_early_late_spacing
    slot.chips_since_epoch = -chips_to_epoch
    slot.driver_epoch_index = nav.pending_epoch_index
    _reset_discriminator_accumulators(state)
end

# The decoding signal's half of the snapshot at the pending epoch if this integration
# crosses it: the decoder and bit sync at its last integration end, and the code phase at
# the epoch from the last bit edge, in the decoding signal's own code.
function _snapshot_decoding!(nav::NavFilterState, group::VectorTrackingSlotGroup, slot::VectorTrackingSlot, integration::CoherentIntegration, nco_history)
    _crosses_pending_epoch(nav, slot, slot.decoding_epoch_index, slot.decoding_last_end_time, integration) ||
        return nothing
    fs = _sampling_frequency_hz(integration)
    epoch_time = _epoch_time(nav, nav.pending_epoch_index)
    signal = group.decoding_signal
    code_frequency = ustrip(Hz, get_chip_rate(signal))
    code_command = calc_mean_nco_command(nco_history, slot.decoding_last_end_sample_index, epoch_time * fs).code_doppler
    chips_to_epoch = (epoch_time - slot.decoding_last_end_time) * (code_frequency + code_command)
    slot.epoch_decoder = slot.running_decoder
    slot.epoch_bit_synced = slot.decoding_bit_synced
    slot.epoch_code_phase = _code_phase_from_bit_edge(signal, slot) + chips_to_epoch
    slot.decoding_epoch_index = nav.pending_epoch_index
    nothing
end

# Complete the slot's snapshot once both halves are in: the decoder and code phase
# from its last bit edge (see `_code_phase_on_driver`), and the lock (driver's
# C/N₀ at least the threshold and decoding signal's bit sync).
function _complete_snapshot!(nav::NavFilterState, group::VectorTrackingSlotGroup, slot::VectorTrackingSlot)
    epoch = nav.pending_epoch_index
    slot.snapshot_epoch_index < epoch && slot.driver_epoch_index == epoch && slot.decoding_epoch_index == epoch ||
        return nothing
    code_frequency = ustrip(Hz, get_chip_rate(group.decoding_signal))
    decoder, code_phase = _decoder_at_code_phase(
        slot.epoch_decoder,
        _code_phase_on_driver(group, slot),
        code_frequency,
    )
    measurement_ready = slot.epoch_bit_synced && slot.cn0_dbhz >= nav.min_tracking_cn0_dbhz
    slot.decoder = decoder
    slot.code_phase = code_phase
    slot.measurement_ready = measurement_ready
    slot.positioning_ready =
        measurement_ready && has_positioning_data(decoder) && is_sat_healthy(decoder)
    slot.snapshot_epoch_index = epoch
    nav.num_snapshots += 1
    nothing
end

# The code phase at the epoch from the last bit edge: which bit (and roughly
# where) from the decoding signal's count, exactly where from the driver's replica, the
# one the discriminators steer. The codes are aligned at the satellite, so the driver's
# phase is the code phase modulo the shorter of its code period and a data bit; the
# count moves to the nearest phase that agrees. A plain signal agrees already; before
# bit sync nothing reads it.
function _code_phase_on_driver(group::VectorTrackingSlotGroup, slot::VectorTrackingSlot)
    code_phase = slot.epoch_code_phase
    driver_phase = slot.epoch_driver_code_phase
    slot.epoch_bit_synced && !isnan(driver_phase) || return code_phase
    decoding = group.decoding_signal
    chips_per_bit = get_code_periods_per_bit(decoding) * get_code_length(decoding)
    period = min(Float64(get_code_length(group.driver_signal)), chips_per_bit)
    code_phase + rem(driver_phase - code_phase, period, RoundNearest)
end

# Cap on the weighting C/N₀: the moment estimator reads a noise-free (simulated) signal
# as infinite, which would give the filter a noiseless measurement.
const MAX_CN0_DBHZ = 80.0

# Whether some satellite of the group can still complete its snapshot of the pending
# epoch: tracked, not stale, due, and with its missing halves' signals not yet past it
# (`_crosses_pending_epoch`).
function _can_reach_epoch(acc, group::VectorTrackingSlotGroup, nav::NavFilterState, epoch_time, now)
    acc && return true
    epoch = nav.pending_epoch_index
    for slot in group.slots
        slot.occupied && !_is_stale(nav, slot, now) && slot.first_epoch_index <= epoch &&
            slot.snapshot_epoch_index < epoch &&
            (slot.driver_epoch_index == epoch || slot.driver_last_end_time <= epoch_time) &&
            (slot.decoding_epoch_index == epoch || slot.decoding_last_end_time <= epoch_time) &&
            return true
    end
    false
end

# The decoding signal's code phase (chips) at its last integration end from the last
# data-bit edge: the bit clock's whole code periods plus the phase past the code period
# boundary. Meaningless (and unread) before bit sync.
function _code_phase_from_bit_edge(signal::AbstractGNSSSignal, slot::VectorTrackingSlot)
    code_periods = slot.decoding_bit_synced ? slot.decoding_num_code_periods_into_bit : 0
    fraction = slot.decoding_code_phase_fraction
    code_periods * get_code_length(signal) + (isnan(fraction) ? 0.0 : fraction)
end

# Normalise `code_phase` into the current bit: a phase outside it (just short of the
# edge, or past the next one) shifts the decoder's bit count instead, which is all
# the transmit time reads.
function _decoder_at_code_phase(decoder::GNSSDecoderState, code_phase, code_frequency)
    num_bits = decoder.num_bits_since_time_reference
    isnothing(num_bits) && return decoder, code_phase
    chips_per_bit = code_frequency / ustrip(Hz, get_bit_rate(decoder))
    shift = floor(Int, code_phase / chips_per_bit)
    shift == 0 && return decoder, code_phase
    GNSSDecoderState(decoder; num_bits_since_time_reference = num_bits + shift),
    code_phase - shift * chips_per_bit
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
# cycle before it: after a re-registration on an integration starting at the epoch, or after
# an overdue cycle. Neither locks it out for more than that epoch.
function _run_cycle_if_due!(nav::NavFilterState, integration::CoherentIntegration)
    nav.num_snapshots == 0 && return false
    now = integration.sample_index / _sampling_frequency_hz(integration)
    _fold_groups(_all_snapshotted, true, nav.groups, nav, now) || return false
    _navigation_cycle!(nav, now)
    true
end

# If the slot has snapshotted the pending epoch and is already past the next one while
# the pending one still waits for others, run that cycle now with the satellites that
# made it. Returns whether it ran. Only a satellite behind by more than a cycle, yet
# not stale, holds an epoch up; the cycle then moves the pending epoch past
# the satellites stepped before (`_next_pending_epoch`), so the next one runs as due.
function _run_overdue_cycle!(nav::NavFilterState, slot::VectorTrackingSlot, integration::CoherentIntegration)
    slot.snapshot_epoch_index == nav.pending_epoch_index || return false
    now = integration.sample_index / _sampling_frequency_hz(integration)
    now > _epoch_time(nav, nav.pending_epoch_index + 1) || return false
    _navigation_cycle!(nav, now)
    true
end

# Whether no satellite of the group can still complete its snapshot of the pending
# epoch.
_all_snapshotted(ready, group::VectorTrackingSlotGroup, nav::NavFilterState, now) =
    ready && !_can_reach_epoch(false, group, nav, _epoch_time(nav, nav.pending_epoch_index), now)

# The epoch to snapshot after the cycle at `epoch`: the next one, or, if a satellite is
# past it already (see `_run_cycle_if_due!`), the first no satellite is past. The cycle
# after skipped epochs is longer, as after a gap. The host steps every satellite through
# each chunk, which is shorter than a cycle, so in step no satellite is past the next
# epoch.
function _next_pending_epoch(nav::NavFilterState, epoch)
    next = epoch + 1
    latest = _fold_groups(_latest_integration_end, -Inf, nav.groups)
    latest > _epoch_time(nav, next) ? _epoch_at_or_after(nav, latest) : next
end

function _latest_integration_end(latest, group::VectorTrackingSlotGroup)
    for slot in group.slots
        slot.occupied || continue
        latest = max(latest, slot.driver_last_end_time, slot.decoding_last_end_time)
    end
    latest
end

# Stale: no integration of the driver, or of the decoding signal, for two cycles.
_is_stale(nav::NavFilterState, slot::VectorTrackingSlot, now) =
    min(slot.driver_last_end_time, slot.decoding_last_end_time) <
    now - 2 * _update_interval_seconds(nav)

# Before a cycle: mark the slots it reads, clear release reasons, and free stale slots
# (storage kept, members released). Returns whether a member was dropped.
function _prepare_slots!(released, group::VectorTrackingSlotGroup, nav::NavFilterState, now)
    for slot in group.slots
        slot.release_reason = VT_NOT_RELEASED
        slot.in_cycle = slot.occupied && slot.snapshot_epoch_index == nav.pending_epoch_index
        if slot.occupied && !slot.in_cycle && _is_stale(nav, slot, now)
            slot.occupied = false
            if slot.loop_state.vector_controlled
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
#     `measurement_deficit_timeout` of unsolvable epochs, or with no member left, every
#     satellite in the loop is released (those that missed the epoch too) and the next
#     cycle solves the scalar PVT.
#   - `filter_config = nothing`: the scalar PVT only.
function _navigation_cycle!(nav::NavFilterState, now)
    groups = nav.groups
    buffers = nav.buffers
    epoch = nav.pending_epoch_index
    cycle_time =
        nav.cycle_epoch_index == typemin(Int) ? nav.navigation_filter_update_interval :
        (epoch - nav.cycle_epoch_index) * nav.navigation_filter_update_interval
    nav.cycle_index += 1
    dropped = _fold_groups(_prepare_slots!, false, groups, nav, now)
    seeded = released = fell_back = false
    if nav.enabled && nav.vector_tracking_active
        released, fell_back = _run_cycle!(nav, groups, cycle_time)
    else
        # Scalar solve: no per-member report.
        empty_keeping_capacity!(nav.vector_tracking_members)
        _fold_groups_with_buffers(_collect_pvt_ready!, nothing, groups, buffers.signal_observations, 1)
        result = calc_pvt!(
            nav.pvt,
            nav.workspace,
            buffers.signal_observations,
            nav.pvt;
            approximate_year = nav.approximate_year,
            enable_ionospheric_correction = nav.enable_ionospheric_correction,
            enable_tropospheric_correction = nav.enable_tropospheric_correction,
        )
        if result.status == PositionVelocityTime.solved
            nav.pvt = result.solution
            if nav.enabled
                seeded = true
                _seed!(nav, groups, cycle_time)
            end
        end
    end
    num_members = _fold_groups_with_buffers(_count_members, 0, groups, buffers.signal_observations, 1)
    nav.status = VectorTrackingStatus(
        nav.vector_tracking_active,
        nav.vector_tracking_active || fell_back ? calc_position_std(nav) * m : NaN * m,
        nav.vector_tracking_active || fell_back ? calc_clock_bias_std(nav) * m : NaN * m,
        nav.measurement_deficit_duration,
        num_members,
        seeded,
        fell_back,
        released || dropped,
    )
    nav.cycle_epoch_index = epoch
    nav.pending_epoch_index = _next_pending_epoch(nav, epoch)
    nav.num_snapshots = 0
    nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# After the integration

# After the driver's loop step: move the replica to the integration's end.
function _advance_driver!(group::VectorTrackingSlotGroup, slot::VectorTrackingSlot, integration::CoherentIntegration, nco_history)
    signal = group.driver_signal
    fs = _sampling_frequency_hz(integration)
    code_frequency = ustrip(Hz, get_chip_rate(signal))
    code_command = calc_mean_nco_command(nco_history, slot.driver_last_end_sample_index, integration.sample_index).code_doppler
    slot.chips_since_epoch +=
        (integration.sample_index - slot.driver_last_end_sample_index) / fs * (code_frequency + code_command)
    slot.driver_last_end_sample_index = integration.sample_index
    slot.driver_last_end_time = integration.sample_index / fs
    slot.last_code_phase = integration.code_phase
    slot.last_coherent_integration_time =
        uconvert(s, integration.num_integrated_samples / integration.sampling_frequency)
    slot.last_early_late_spacing =
        calc_early_late_sample_spacing(
            integration.filtered_correlator_bank,
            integration.sampling_frequency,
            get_chip_rate(signal),
        ) * code_frequency / fs
    nothing
end

# A decoding-signal integration: decode exactly the soft bits it added (so the host may
# drain its bit synchronizer whenever it likes) and note the bit clock at its end. Integrations
# later in the batch a sync was found in still have the pre-sync replica (see
# `_process_correlator_bank_output!`'s `secondary_code_wiped`): they leave the bit count unanchored until the
# next batch.
function _advance_decoding!(
    group::VectorTrackingSlotGroup,
    slot::VectorTrackingSlot,
    integration::CoherentIntegration,
    state::VectorLoopState,
    nco_history,
    driver::Bool,
)
    if integration.sync_change == SYNC_FOUND
        slot.sync_batch_end_sample_index = integration.command_issue_sample_index
    elseif !integration.bit_or_secondary_code_synced
        slot.sync_batch_end_sample_index = typemin(Int)
    end
    unwiped_after_sync =
        integration.sync_change != SYNC_FOUND && integration.command_issue_sample_index == slot.sync_batch_end_sample_index
    _decode_integration!(group, slot, integration, state, nco_history, driver, unwiped_after_sync)
    code_length = get_code_length(group.decoding_signal)
    fs = _sampling_frequency_hz(integration)
    slot.decoding_last_end_sample_index = integration.sample_index
    slot.decoding_last_end_time = integration.sample_index / fs
    slot.decoding_bit_synced = integration.bit_or_secondary_code_synced && !unwiped_after_sync
    slot.decoding_num_code_periods_into_bit = integration.num_code_periods_into_bit
    slot.decoding_code_phase_fraction =
        isnan(integration.code_phase) ? NaN :
        mod(integration.code_phase + code_length / 2, code_length) - code_length / 2
    nothing
end

# Decode a decoding-signal integration. Out of the vector loop: the host's soft bits. In it
# the carrier NCO runs on the filter's prediction, so the prompts turn at its Doppler
# error: each prompt is rotated onto the incoming carrier by the residual PLL's estimate
# (`_bit_prompt_phase`) and the decoding signal's carrier-phase offset from the
# driver, summed over the bit, and the sum's real part decoded (signed as the host's
# bit clock signs its own) on the integration where the host's bit clock ends the bit. A
# bit not covered from its first code period (the one the satellite joins in, or one after
# an `unwiped_after_sync` integration), and an integration ending more than one, take the host's
# soft bits.
function _decode_integration!(
    group::VectorTrackingSlotGroup,
    slot::VectorTrackingSlot,
    integration::CoherentIntegration,
    state::VectorLoopState,
    nco_history,
    driver::Bool,
    unwiped_after_sync::Bool,
)
    soft_bits = integration.new_soft_bits
    phase =
        state.vector_controlled && integration.bit_or_secondary_code_synced && integration.sync_change == SYNC_UNCHANGED && !unwiped_after_sync ?
        _bit_prompt_phase(
            state.residual_pll,
            state.ramp,
            state.previous_replica_tail,
            slot.prompt_history,
            integration,
            nco_history,
            driver,
        ) :
        NaN
    if isnan(phase)
        slot.bit_accumulator = zero(ComplexF64)
        slot.bit_from_edge = false
    else
        rotation =
            2π * (get_carrier_phase_offset(group.driver_signal) - get_carrier_phase_offset(group.decoding_signal) -
            phase)
        slot.bit_accumulator += ComplexF64(get_prompt(integration.filtered_correlator_bank)) * cis(rotation)
    end
    isempty(soft_bits) && return nothing
    if !isnan(phase) && slot.bit_from_edge && Base.length(soft_bits) == 1
        bit = real(slot.bit_accumulator)
        slot.soft_bit[1] = Float32(integration.bit_polarity < 0 ? -bit : bit)
        slot.running_decoder = decode!(slot.running_decoder, slot.soft_bit, 1)
    else
        slot.running_decoder = decode!(slot.running_decoder, soft_bits, Base.length(soft_bits))
    end
    slot.bit_accumulator = zero(ComplexF64)
    slot.bit_from_edge = !isnan(phase)
    nothing
end
