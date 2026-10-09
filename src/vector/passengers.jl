# ─────────────────────────────────────────────────────────────────────────────
# Passenger records

# A passenger record. Combining, it goes to the inner loop (only into the PLL in the
# vector loop). A data passenger of a dataless driver advances the bit clock and
# decoder; in the vector loop, combining, its DLL and raw FLL readings go to the
# state (`PassengerReadings`).
function fold_passenger_record(
    estimator::VectorPLLAndDLL,
    state::SatVectorPLLAndDLL,
    record::LoopRecord,
    words;
    driver_signal::AbstractGNSSSignal,
    differential_group_delay_chips::Real = NaN,
)
    takes_passenger_records(estimator) || return state
    combining = combines_signals(estimator)
    if combining
        loops = state.vt_on ? _PLL_ONLY : _scalar_loops_to_combine(state.inner)
        inner = _with_passenger_record(
            state.inner,
            record,
            words,
            loops,
            driver_signal,
            differential_group_delay_chips,
        )
        state = SatVectorPLLAndDLL(state; inner)
    end
    index = _fold_passenger_record!(
        _driver_group(estimator.navigation.groups, driver_signal),
        state,
        record,
    )
    index > 0 && combining && state.vt_on || return state
    dll, fll = _passenger_readings(record, words, differential_group_delay_chips)
    readings = state.passenger_readings
    SatVectorPLLAndDLL(
        state;
        passenger_readings = Base.setindex(
            readings,
            _with_pending(readings[index], dll, fll),
            index,
        ),
    )
end

# The slot group of `driver_signal`, found by type (see `_step_vector_record`).
@inline _driver_group(::Tuple{}, driver_signal) = throw(
    ArgumentError(
        "this vector-tracking estimator was not built for " *
        "$(nameof(typeof(driver_signal))); list it among its signals",
    ),
)
@inline _driver_group(groups::Tuple, driver_signal::S) where {S} =
    first(groups).signal isa S ? first(groups) :
    _driver_group(Base.tail(groups), driver_signal)

# A passenger record into the satellite's slot: the data signal's bit clock and
# decoder, and what the passenger's variances are built from. Returns the
# passenger's index, or 0 before the satellite's first driver record, when it has no
# slot yet.
function _fold_passenger_record!(
    group::VTSlotGroup,
    state::SatVectorPLLAndDLL,
    record::LoopRecord,
)
    index = _passenger_index(group.passengers, record.signal, 1)
    index == 0 && throw(
        ArgumentError(
            "$(nameof(typeof(record.signal))) is not a passenger of " *
            "$(nameof(typeof(group.signal))) in this vector-tracking estimator; " *
            "list it in the driver's signal group",
        ),
    )
    slots = group.slots
    1 <= state.slot <= length(slots) || return 0
    slot = slots[state.slot]
    slot.occupied && slot.registration == state.registration || return 0
    record.signal isa typeof(group.data_signal) &&
        _advance_data_signal!(group, slot, record)
    passenger = slot.passengers[index]
    integration_time = uconvert(s, record.integrated_samples / record.sampling_frequency)
    held, cn0_estimator = _restart_cn0(
        passenger.held_cn0_dbhz,
        passenger.cn0_estimator,
        passenger.integration_time,
        integration_time,
    )
    slot.passengers[index] = VTPassenger(
        update(cn0_estimator, get_prompt(record.filtered_correlator)),
        record.cn0,
        held,
        integration_time,
        _early_late_spacing_chips(record),
        _dll_variance_factor(record.filtered_correlator),
    )
    index
end

# The position of `signal` among the passengers, 0 if it is none of them.
@inline _passenger_index(::Tuple{}, signal, i) = 0
@inline _passenger_index(passengers::Tuple, signal::S, i) where {S} =
    first(passengers) isa S ? i : _passenger_index(Base.tail(passengers), signal, i + 1)

# A passenger's DLL reading (chips, referred to the driver by
# `differential_group_delay_chips`; `nothing` if unknown) and raw FLL reading (Hz;
# `nothing` without a previous prompt; four-quadrant where the record has a
# polarity). Unlike in scalar combining, the form doesn't matter: the vector loop
# keeps the frequency error far inside the two-quadrant range, where both forms
# agree.
@inline function _passenger_readings(
    record::LoopRecord,
    words,
    differential_group_delay_chips,
)
    dll =
        isnan(differential_group_delay_chips) ? nothing :
        _passenger_dll_reading(record, words, differential_group_delay_chips)
    fll =
        iszero(record.previous_prompt) ? nothing :
        _passenger_fll_reading(record, !iszero(record.polarity))
    dll, fll
end
