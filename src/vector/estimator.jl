# The per-record vector loop: a scalar FLL-assisted PLL/DLL that a vector-tracking
# filter can take over (see `VectorPLLAndDLL`).

"""
    VectorPLLAndDLL(signals...; combine_signals = false,
                    inner = ConventionalAssistedPLLAndDLL(),
                    config = VectorTracking(), cycle_time = 100ms,
                    lock_cn0_threshold = 30dBHz, max_satellites_per_signal = 16,
                    num_prompts_for_cn0_estimation = 100,
                    approximate_year = year(now(UTC)),
                    enable_ionospheric_correction = true,
                    enable_tropospheric_correction = true)

Vector-tracking Doppler estimator for the ranging `signals`: `AbstractGNSSSignal`s, each
at most once, or for a satellite tracked on several signals of one band its signal
group, driver first, e.g. `(GalileoE1C(), GalileoE1B())`. A group's signals are one
constellation's and share the driver's code rate and carrier frequency. A host that
keeps its satellites' groups itself builds [`VectorTrackingSettings`](@ref) with these
keywords instead and binds them with [`with_signal_groups`](@ref).

All satellites stepped with this estimator share one navigation engine which, inside
[`step_loop`](@ref), syncs to and decodes the navigation bits, estimates the C/N₀,
solves the scalar PVT, seeds the navigation filter from its first fix and then runs one
filter cycle every `cycle_time`, taking satellites over and handing them back. A host
needs no vector-specific code: it builds the states with
[`init_estimator_state`](@ref) and steps them like any other loop's. Read the results
with [`navigation_solution`](@ref), [`navigation_status`](@ref),
[`release_reason`](@ref), [`member_sats`](@ref), [`position_uncertainty`](@ref) and
[`clock_uncertainty`](@ref).

Records must carry `prn` and `code_phase` (see [`LoopRecord`](@ref)), on a
`sample_index / sampling_frequency` time grid shared by all satellites; a record of a
signal not among `signals`, or without a PRN, throws an `ArgumentError`. Passenger
records go to [`fold_passenger_record`](@ref) (see [`takes_passenger_records`](@ref))
with their own `prn` and `code_phase` and the driver fold's `fold_end`. Step every
satellite at least once per `cycle_time / 2`: a cycle runs once every satellite has
reached its epoch, and one that has not stepped for `2 · cycle_time` is dropped.

Each satellite runs `inner` until the filter takes it over, and again once released.
In the vector loop:

  - the carrier filter runs at the narrow bandwidth, its PLL branch on the satellite's
    own phase discriminator and its FLL branch on the filter's carrier correction
    (integrated as a frequency error, not added to the Doppler);
  - the code filter is frozen and the filter's code correction replaces its output:
    `code_doppler = init_code_doppler + code_freq_update +
    carrier_filter_output · code_center_frequency_ratio`;
  - the DLL output (chips) and raw FLL discriminator (Hz) are accumulated for the
    filter.

The corrections of the latest cycle are taken up on the next record, sized for where
they land: at `landing_sample`, or at the record's end under `NO_LANDING_SAMPLE`. For
the carrier-loop staging and the discriminator forms in either mode see
[Staging and discriminators under vector tracking](@ref).

The engine decodes the bits of the group's first signal that carries navigation data.
With `combine_signals = true` the passengers are combined into `inner` out of the
vector loop and into the PLL in it, and the navigation filter fuses every signal's DLL
and raw FLL readings by inverse variance (see [Several signals of a satellite](@ref)).
The setting covers the scalar fallback too, so `inner` must be built without
`combine_signals`.

`inner` must have an FLL-assisted carrier filter, the carrier correction's only way
into the loop: a `ConventionalPLLAndDLL{ThirdOrderAssistedBilinearLF}` (what
[`ConventionalAssistedPLLAndDLL`](@ref) builds) or an [`NCOReferencedPLLAndDLL`](@ref).
The latter keeps its phase-error prediction to the landing sample in the vector loop
and cannot combine signals, since passenger records are not predicted to it. An
unsupported `inner`, or a group without a signal that carries navigation data (a pilot
such as Galileo E1C on its own), throws an `ArgumentError`. `config = nothing` only
ever solves the scalar PVT.

Storage is allocated for `max_satellites_per_signal` satellites per signal and grows
only past that; a dropped satellite leaves its storage to the next one.
"""
struct VectorPLLAndDLL{E<:AbstractDopplerEstimator,N} <: AbstractDopplerEstimator
    inner::E
    navigation::N
    combine_signals::Bool
end

_is_fll_assisted(::AbstractDopplerEstimator) = false
_is_fll_assisted(::ConventionalPLLAndDLL{<:ThirdOrderAssistedBilinearLF}) = true
_is_fll_assisted(::NCOReferencedPLLAndDLL) = true

"""
    SatVectorPLLAndDLL

Per-satellite state of a [`VectorPLLAndDLL`](@ref): the inner loop's state plus the
interface to the vector-tracking filter.

  - `vt_on`: whether the filter controls this satellite's NCOs; while `false` the inner
    loop runs and nothing is accumulated.
  - `code_discr_acc` / `carrier_discr_acc`: `(count, sum)` of the DLL output (chips)
    and raw FLL discriminator (Hz) since the filter last read them.
  - `code_freq_update` / `carrier_freq_update`: the corrections from the latest cycle.
  - `code_freq_update_history` / `code_update_landing_lead`: the last three code
    corrections, newest first, and how long after the cycle epoch the newest reached
    the replica; the next code measurement uses them to move the mid-cycle mean
    discriminator to the epoch. Kept by [`reset_estimator_state`](@ref) (see there).
  - `slot`, `registration`, `cycle_id`: where the navigation engine keeps this
    satellite (`0` until its first record), and the latest cycle it has taken up.
  - `passenger_readings`: per passenger of the satellite's group, its DLL and raw FLL
    readings for the filter (`TrackingLoops.PassengerReadings`).
"""
struct SatVectorPLLAndDLL{S,P<:Tuple}
    inner::S
    vt_on::Bool
    code_discr_acc::Tuple{Int,Float64}
    carrier_discr_acc::Tuple{Int,typeof(1.0Hz)}
    code_freq_update::typeof(1.0Hz)
    carrier_freq_update::typeof(1.0Hz)
    code_freq_update_history::NTuple{3,typeof(1.0Hz)}
    code_update_landing_lead::typeof(1.0s)
    slot::Int
    registration::Int
    cycle_id::Int
    passenger_readings::P
end

"""
    PassengerReadings

One passenger's DLL (chips, referred to the driver's code phase) and raw FLL (Hz)
readings for the navigation filter as `(count, sum)`, in a satellite's
[`SatVectorPLLAndDLL`](@ref). A passenger record adds its readings to
`pending_code` / `pending_carrier`; the driver's next record moves them to `code` /
`carrier`, its cycle's, which the epoch snapshot hands to the filter with the
driver's. So a reading falls in the cycle of the driver record it ends within, as
the scalar loops combine it into that record, moving by at most one driver record.
"""
struct PassengerReadings
    pending_code::Tuple{Int,Float64}
    pending_carrier::Tuple{Int,typeof(1.0Hz)}
    code::Tuple{Int,Float64}
    carrier::Tuple{Int,typeof(1.0Hz)}
end

PassengerReadings() = PassengerReadings((0, 0.0), (0, 0.0Hz), (0, 0.0), (0, 0.0Hz))

# With a passenger record's readings pending (`nothing`: none).
@inline _with_pending(readings::PassengerReadings, dll, fll) = PassengerReadings(
    _counted(readings.pending_code, dll),
    _counted(readings.pending_carrier, fll),
    readings.code,
    readings.carrier,
)

# The pending readings moved into the cycle's.
@inline _moved_to_cycle(readings::PassengerReadings) = PassengerReadings(
    (0, 0.0),
    (0, 0.0Hz),
    _summed(readings.code, readings.pending_code),
    _summed(readings.carrier, readings.pending_carrier),
)

# The cycle's readings emptied once the snapshot has taken them; the pending stay.
@inline _without_cycle_readings(readings::PassengerReadings) =
    PassengerReadings(readings.pending_code, readings.pending_carrier, (0, 0.0), (0, 0.0Hz))

@inline _summed(a::Tuple, b::Tuple) = (a[1] + b[1], a[2] + b[2])

# Empty readings, one per passenger of `readings`.
@inline _no_passenger_readings(readings::Tuple) = map(_ -> PassengerReadings(), readings)

# A satellite with the vector interface empty, on the slot of `registration`, for the
# passengers of `readings`.
SatVectorPLLAndDLL(
    inner,
    vt_on::Bool,
    readings::Tuple,
    slot::Int = 0,
    registration::Int = 0,
    cycle_id::Int = -1,
) = SatVectorPLLAndDLL(
    inner,
    vt_on,
    (0, 0.0),
    (0, 0.0Hz),
    0.0Hz,
    0.0Hz,
    (0.0Hz, 0.0Hz, 0.0Hz),
    0.0s,
    slot,
    registration,
    cycle_id,
    _no_passenger_readings(readings),
)

function SatVectorPLLAndDLL(
    state::SatVectorPLLAndDLL{S,P};
    inner::Maybe{S} = nothing,
    code_discr_acc::Maybe{Tuple{Int,Float64}} = nothing,
    carrier_discr_acc::Maybe{Tuple{Int,typeof(1.0Hz)}} = nothing,
    cycle_id::Maybe{Int} = nothing,
    passenger_readings::Maybe{P} = nothing,
) where {S,P}
    SatVectorPLLAndDLL{S,P}(
        isnothing(inner) ? state.inner : inner,
        state.vt_on,
        isnothing(code_discr_acc) ? state.code_discr_acc : code_discr_acc,
        isnothing(carrier_discr_acc) ? state.carrier_discr_acc : carrier_discr_acc,
        state.code_freq_update,
        state.carrier_freq_update,
        state.code_freq_update_history,
        state.code_update_landing_lead,
        state.slot,
        state.registration,
        isnothing(cycle_id) ? state.cycle_id : cycle_id,
        isnothing(passenger_readings) ? state.passenger_readings : passenger_readings,
    )
end

# The state on a fresh slot, owing the pick-up of the latest cycle.
_registered(state::SatVectorPLLAndDLL{S,P}, slot::Int, registration::Int) where {S,P} =
    SatVectorPLLAndDLL{S,P}(
        state.inner,
        state.vt_on,
        state.code_discr_acc,
        state.carrier_discr_acc,
        state.code_freq_update,
        state.carrier_freq_update,
        state.code_freq_update_history,
        state.code_update_landing_lead,
        slot,
        registration,
        -1,
        state.passenger_readings,
    )

"""
    init_estimator_state(estimator::VectorPLLAndDLL, driver_signal, carrier_doppler,
                         code_doppler)

The inner loop's state, out of the vector loop with nothing accumulated and no
corrections. The satellite joins the navigation engine on its first record.
"""
init_estimator_state(
    estimator::VectorPLLAndDLL,
    driver_signal::AbstractGNSSSignal,
    carrier_doppler,
    code_doppler,
) = SatVectorPLLAndDLL(
    init_estimator_state(estimator.inner, driver_signal, carrier_doppler, code_doppler),
    false,
    _passengers_of(estimator.navigation.groups, driver_signal),
)

# The passengers of `driver_signal`'s group, found by type; none for a signal that
# drives no group (stepping it throws).
@inline _passengers_of(::Tuple{}, driver_signal) = ()
@inline _passengers_of(groups::Tuple, driver_signal::S) where {S} =
    first(groups).signal isa S ? first(groups).passengers :
    _passengers_of(Base.tail(groups), driver_signal)

"""
    reset_estimator_state(estimator::VectorPLLAndDLL, state, carrier_doppler, code_doppler)

Re-seed the inner loop from the converged Dopplers, zero the accumulators (the
passengers' readings too) and drop the corrections, keeping `vt_on` and the
satellite's place in the navigation engine. The converged Dopplers already contain
the last correction, so keeping it would apply it twice; the replica was still
steered by it, so the correction history and landing lead are kept.
"""
reset_estimator_state(
    estimator::VectorPLLAndDLL,
    state::SatVectorPLLAndDLL{S,P},
    carrier_doppler,
    code_doppler,
) where {S,P} = SatVectorPLLAndDLL{S,P}(
    reset_estimator_state(estimator.inner, state.inner, carrier_doppler, code_doppler),
    state.vt_on,
    (0, 0.0),
    (0, 0.0Hz),
    0.0Hz,
    0.0Hz,
    state.code_freq_update_history,
    state.code_update_landing_lead,
    state.slot,
    state.registration,
    state.cycle_id,
    _no_passenger_readings(state.passenger_readings),
)

# ─────────────────────────────────────────────────────────────────────────────
# What the navigation engine does to a satellite's state; each keeps its place in
# the engine.

# Into the vector loop with empty accumulators and no corrections; a no-op if in it.
_enable_vector_tracking(state::SatVectorPLLAndDLL) =
    state.vt_on ? state :
    SatVectorPLLAndDLL(
        state.inner,
        true,
        state.passenger_readings,
        state.slot,
        state.registration,
        state.cycle_id,
    )

# Back to the inner loop, accumulators and corrections zeroed so a later re-enable
# never runs on stale values.
_disable_vector_tracking(state::SatVectorPLLAndDLL) = SatVectorPLLAndDLL(
    state.inner,
    false,
    state.passenger_readings,
    state.slot,
    state.registration,
    state.cycle_id,
)

# `_disable_vector_tracking` with the inner loop re-seeded from the replica's
# Dopplers, so the scalar loop takes over without a transient.
_release_from_vector_tracking(state::SatVectorPLLAndDLL, carrier_doppler, code_doppler) =
    SatVectorPLLAndDLL(
        _reseed_inner(state.inner, carrier_doppler, code_doppler),
        false,
        state.passenger_readings,
        state.slot,
        state.registration,
        state.cycle_id,
    )

# The inner reset reads only the state, so a default estimator stands in.
_reseed_inner(state::SatConventionalPLLAndDLL, carrier_doppler, code_doppler) =
    reset_estimator_state(ConventionalPLLAndDLL(), state, carrier_doppler, code_doppler)
_reseed_inner(state::SatNCOReferencedPLLAndDLL, carrier_doppler, code_doppler) =
    reset_estimator_state(NCOReferencedPLLAndDLL(), state, carrier_doppler, code_doppler)

# Set the filter's corrections, landing `landing_lead` after the cycle epoch; the code
# correction is pushed onto `code_freq_update_history`.
function _set_vector_corrections(
    state::SatVectorPLLAndDLL{S,P},
    code_freq_update,
    carrier_freq_update,
    landing_lead = 0.0s,
) where {S,P}
    newest, previous, _ = state.code_freq_update_history
    SatVectorPLLAndDLL{S,P}(
        state.inner,
        state.vt_on,
        state.code_discr_acc,
        state.carrier_discr_acc,
        code_freq_update,
        carrier_freq_update,
        (code_freq_update, newest, previous),
        landing_lead,
        state.slot,
        state.registration,
        state.cycle_id,
        state.passenger_readings,
    )
end

# A `(count, sum)` accumulator with one more reading; `nothing` is no reading.
@inline _counted(acc::Tuple, ::Nothing) = acc
@inline _counted(acc::Tuple, reading) = (acc[1] + 1, acc[2] + reading)

# The cycle's accumulators emptied, the driver's and the passengers', once the
# snapshot has taken them; passenger readings still pending stay.
_reset_discriminator_accumulators(state::SatVectorPLLAndDLL) = SatVectorPLLAndDLL(
    state;
    code_discr_acc = (0, 0.0),
    carrier_discr_acc = (0, 0.0Hz),
    passenger_readings = map(_without_cycle_readings, state.passenger_readings),
)

# The accumulated mean DLL output (chips) / raw FLL discriminator (Hz), or `nothing`.
function _mean_code_discriminator(state::SatVectorPLLAndDLL)
    count, discr_sum = state.code_discr_acc
    count == 0 ? nothing : discr_sum / count
end

function _mean_carrier_discriminator(state::SatVectorPLLAndDLL)
    count, discr_sum = state.carrier_discr_acc
    count == 0 ? nothing : discr_sum / count
end

# One record through the inner loop, or in the vector loop through `_step_vector_loop`
# with its discriminators accumulated.
@inline function _step_satellite(
    estimator::VectorPLLAndDLL,
    state::SatVectorPLLAndDLL,
    record::LoopRecord,
    words,
    landing_sample::Int64,
)
    if !state.vt_on
        inner, carrier_doppler, code_doppler =
            step_loop(estimator.inner, state.inner, record, words, landing_sample)
        return SatVectorPLLAndDLL(state; inner), carrier_doppler, code_doppler
    end
    inner, carrier_doppler, code_doppler, dll_discriminator, fll_discriminator =
        _step_vector_loop(
            estimator.inner,
            state.inner,
            record,
            words,
            landing_sample,
            state.code_freq_update,
            state.carrier_freq_update,
        )
    # A record without a previous prompt (after a (re)start, a secondary-code sync or a
    # change of record length) reads 0 Hz on the FLL, which would bias the mean. The
    # passengers' pending readings join this record's cycle.
    SatVectorPLLAndDLL(
        state;
        inner,
        code_discr_acc = _counted(state.code_discr_acc, dll_discriminator),
        carrier_discr_acc = _counted(
            state.carrier_discr_acc,
            iszero(record.previous_prompt) ? nothing : fll_discriminator,
        ),
        passenger_readings = map(_moved_to_cycle, state.passenger_readings),
    ),
    carrier_doppler,
    code_doppler
end

# The inner loop's step in the vector loop (see `VectorPLLAndDLL`). Returns the inner
# state, both Dopplers, and the DLL and raw FLL readings to accumulate; the raw FLL is
# the mean offset from the replica between the two prompts' centres, before any
# re-basing onto a landing word.
@inline function _step_vector_loop(
    estimator,
    state,
    record::LoopRecord,
    words,
    landing_sample::Int64,
    code_freq_update,
    carrier_freq_update,
)
    # The filter owns the code loop and FLL branch, so passengers join the PLL only;
    # their DLL and FLL readings reach the filter via `fold_passenger_record`.
    discriminators = _with_passengers(
        state,
        record,
        _record_discriminators(estimator, state, record, words, landing_sample, true),
        _PLL_ONLY,
    )
    # Not staged: narrow bandwidth, the FLL slot carrying the filter's carrier update;
    # the stage and phase-lock indicator are left as they are.
    carrier_filter_output, carrier_loop_filter = filter_loop(
        state.carrier_loop_filter,
        (discriminators.phase_error, carrier_freq_update),
        discriminators.integration_time,
        _narrow_carrier_bandwidth(state.bandwidths, discriminators.integration_time),
    )
    carrier_doppler, code_doppler = aid_dopplers(
        record.signal,
        state.init_carrier_doppler,
        state.init_code_doppler,
        carrier_filter_output,
        code_freq_update,
    )
    _stepped_state(
        state,
        carrier_loop_filter,
        state.code_loop_filter,
        discriminators.center,
        state.staging,
    ),
    carrier_doppler,
    code_doppler,
    discriminators.code_error,
    discriminators.raw_frequency_error
end

combines_signals(estimator::VectorPLLAndDLL) = estimator.combine_signals

# Passenger records decode a dataless driver's bits and, combining, feed the filter.
takes_passenger_records(estimator::VectorPLLAndDLL) =
    combines_signals(estimator) ||
    any(group -> !_drives_data_signal(group), estimator.navigation.groups)

carrier_loop_stage(state::SatVectorPLLAndDLL) = carrier_loop_stage(state.inner)
phase_lock_indicator(state::SatVectorPLLAndDLL) = phase_lock_indicator(state.inner)
