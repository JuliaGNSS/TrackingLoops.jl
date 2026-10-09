# ─────────────────────────────────────────────────────────────────────────────
# The per-record vector loop: a scalar FLL-assisted PLL/DLL of the user's
# choice, which a vector-tracking filter can take over.
#
# In vector tracking a central navigation filter closes every satellite's loops
# at once (see `VectorTracking`). Per record, the satellite's own loop still
# runs its PLL, but its DLL and FLL outputs are only *accumulated* for the
# filter, and the filter's per-satellite NCO corrections steer the replica:
# the code correction replaces the code filter's output, the carrier
# correction drives the FLL branch of the carrier filter. Before the filter
# takes a satellite over (and after it lets it go) the satellite runs the
# scalar loop it wraps, bit for bit.
# ─────────────────────────────────────────────────────────────────────────────

"""
    VectorPLLAndDLL(signals...; combine_signals = false,
                    inner = ConventionalAssistedPLLAndDLL(),
                    config = VectorTracking(), cycle_time = 100ms,
                    lock_cn0_threshold = 30dBHz, max_satellites_per_signal = 16,
                    num_prompts_for_cn0_estimation = 100,
                    approximate_year = year(now(UTC)),
                    enable_ionospheric_correction = true,
                    enable_tropospheric_correction = true)

Vector-tracking Doppler estimator for the ranging `signals`: one or more
`AbstractGNSSSignal`s, each at most once, or for a satellite tracked on several
signals of one band its signal group, the driver first, as the host steps it,
e.g. `(GalileoE1C(), GalileoE1B())`. A host that keeps its satellites' signal
groups itself, such as Tracking.jl's `TrackState`, builds
[`VectorTrackingSettings`](@ref) with the same keywords instead and binds them
with [`with_signal_groups`](@ref), so the groups are listed once. It does the
whole pipeline inside [`step_loop`](@ref), from what the records carry: every
satellite stepped with this estimator shares one navigation engine, which syncs
to the navigation bits, decodes them, estimates the C/N₀, solves the scalar PVT,
seeds the navigation filter from its first fix and from then on runs one filter
cycle every `cycle_time`, taking satellites over and handing them back. A host
needs no vector-specific code: it builds the satellites' states with
[`init_estimator_state`](@ref) and steps them like any other loop's. Read the
results with [`navigation_solution`](@ref), [`navigation_status`](@ref),
[`release_reason`](@ref), [`member_sats`](@ref), [`position_uncertainty`](@ref)
and [`clock_uncertainty`](@ref).

The records must identify their satellite and replica: `prn` and `code_phase`
on the [`LoopRecord`](@ref), and `sample_index / sampling_frequency` on a time
grid shared by all satellites. Records of a signal not among `signals`, or
without a PRN, throw an `ArgumentError`. The host hands the records of the
passengers, the signals of a group after the driver, to
[`fold_passenger_record`](@ref) ([`takes_passenger_records`](@ref) is `true`
for an estimator with passengers), with their `prn`, `code_phase` and the driver
fold's `fold_end`. The host should step every satellite
at least once per `cycle_time / 2`: a cycle runs once every satellite has
reached its epoch, and a satellite that has not stepped for `2 · cycle_time`
is dropped.

Per satellite, every record runs `inner` — the scalar loop the satellite uses
until the filter takes it over, and again once the filter releases it.
While a satellite is in the vector loop:

  - its carrier filter always runs, with the PLL branch fed by the satellite's
    own phase discriminator and the FLL branch fed by the filter's carrier
    correction (the filter integrates it as a frequency error; it is not added
    to the Doppler directly);
  - its code filter is frozen, and the filter's code correction replaces its
    output: `code_doppler = init_code_doppler + code_freq_update +
    carrier_filter_output · code_center_frequency_ratio`;
  - the DLL output (chips) and the raw FLL discriminator (Hz) are accumulated
    for the filter to read.

The carrier loop's staging and discriminators follow the satellite's mode:

  - Out of the vector loop `inner` stages its carrier loop as it does on its own
    (see [`CarrierLoopStage`](@ref)): FLL-assisted at the wide bandwidth until
    phase lock, then the PLL alone, narrowed once lock has held. In the vector
    loop the FLL branch carries the filter's carrier correction, so it is never
    dropped, the PLL runs at the narrow bandwidth, and the stage and phase-lock
    indicator are left as they are. A satellite the filter releases re-seeds
    `inner` from the replica's Dopplers, which restarts the staging; one it
    takes over keeps `inner`'s state.
  - The discriminators are the record's in either mode: four-quadrant where its
    `polarity` says the prompt is wiped off (a pilot synced to its secondary
    code), two-quadrant otherwise. The PLL reads them in the vector loop too,
    and the raw FLL reading accumulated for the filter is four-quadrant on such
    a record. A record without a
    previous prompt has no FLL reading: the cycle's rate measurement leaves it
    out, and a cycle without any keeps the satellite's code measurement and
    withholds only its rate measurement.

The engine decodes the navigation data of the group's first signal that
carries any, its data signal: the driver, or for a dataless pilot driver its
data passenger, whose records then run the satellite's bit clock and decoder.
The signals of a group share the driver's code rate and carrier frequency.

With `combine_signals = true` the passengers are combined with the driver (see
[Signal combining](@ref)): out of the vector loop by `inner`, in it into the
PLL, and their DLL and raw FLL readings into the navigation filter's code and
rate measurements. A passenger's readings join the cycle of the driver record
they end within, and the filter fuses every signal's cycle mean weighted by its
inverse variance, built from that signal's own C/N₀ estimate, coherent
integration time and tap spacing (taken at its latest record length). A
passenger's DLL readings join only where its group delay relative to the driver
is given. The setting is the vector loop's own, as for
[`ConventionalPLLAndDLL`](@ref) the scalar loop's, and covers its scalar
fallback too: the vector loop folds the passengers into `inner`'s state itself,
so `inner` is built without `combine_signals` (an `inner` with it throws an
`ArgumentError`). An [`NCOReferencedPLLAndDLL`](@ref) `inner` cannot combine: it
steps a phase error predicted to the landing sample, which passenger records are
not.

Each satellite picks up the corrections of the latest cycle on its next record,
sized for where they land: at `landing_sample`, or at the record's end under
`NO_LANDING_SAMPLE`.

`inner` must have an FLL-assisted carrier filter, since that is the only input
path the carrier correction has into the loop: a
`ConventionalPLLAndDLL{ThirdOrderAssistedBilinearLF}` (what
[`ConventionalAssistedPLLAndDLL`](@ref) builds) or an
[`NCOReferencedPLLAndDLL`](@ref). With the latter the phase discriminator keeps
its prediction to the landing sample in the vector loop too. Anything else
throws an `ArgumentError`, as does a group without a signal that carries
navigation data (a pilot such as GPS L1C-P or Galileo E1C on its own).
`config = nothing` only ever solves the scalar PVT.

Storage is allocated at construction for `max_satellites_per_signal`
satellites per signal and only grows past that: a satellite that is dropped
leaves its storage to the next one.
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

Per-satellite state of a [`VectorPLLAndDLL`](@ref): the inner scalar loop's
state and, on top, the interface to the vector-tracking filter.

  - `vt_on`: whether the filter controls this satellite's NCOs. While `false`
    the satellite runs the inner loop and nothing is accumulated.
  - `code_discr_acc` / `carrier_discr_acc`: `(count, sum)` of the DLL output
    (chips) and the raw FLL discriminator (Hz) since the filter last read them.
  - `code_freq_update` / `carrier_freq_update`: the corrections the per-record
    step applies, set from the filter's latest cycle.
  - `code_freq_update_history` and `code_update_landing_lead`: the last three
    code corrections the filter set, newest first, and how long after the
    filter's cycle epoch the newest reached the replica. The filter's next
    code measurement needs them to move the mid-cycle mean discriminator to
    the epoch. They record what the replica was steered by, so a
    [`reset_estimator_state`](@ref), which folds the newest correction into
    the inner loop's Dopplers, keeps them.
  - `slot`, `registration` and `cycle_id`: where the navigation engine keeps
    this satellite (`0` until its first record), and the latest cycle it has
    taken up.
  - `passenger_readings`: per passenger of the satellite's signal group, its DLL
    and raw FLL readings for the filter (`TrackingLoops.PassengerReadings`).
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
readings for the navigation filter, as `(count, sum)`, kept in the
[`SatVectorPLLAndDLL`](@ref) of a satellite in the vector loop. A passenger record
adds its readings to `pending_code` / `pending_carrier`; the driver's next record
moves them to `code` / `carrier`, the readings of that record's cycle, which the
epoch snapshot hands to the filter with the driver's own. So a passenger reading
falls in the cycle of the driver record it ends within, as the scalar loops combine
it into that record, and moves by at most one driver record across an epoch.
"""
struct PassengerReadings
    pending_code::Tuple{Int,Float64}
    pending_carrier::Tuple{Int,typeof(1.0Hz)}
    code::Tuple{Int,Float64}
    carrier::Tuple{Int,typeof(1.0Hz)}
end

PassengerReadings() = PassengerReadings((0, 0.0), (0, 0.0Hz), (0, 0.0), (0, 0.0Hz))

# `readings` with a passenger record's DLL and FLL readings pending (`nothing`: none).
@inline _with_pending(readings::PassengerReadings, dll, fll) = PassengerReadings(
    _counted(readings.pending_code, dll),
    _counted(readings.pending_carrier, fll),
    readings.code,
    readings.carrier,
)

# The pending readings moved into the cycle's, by a driver record in the vector loop.
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

# A satellite with the vector interface empty — out of the loop or just joined —
# on the slot of `registration`, for the passengers of `readings`.
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

# The state on a fresh slot `slot` of `registration`, owing the pick-up of the
# latest cycle.
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

The inner loop's state with the vector interface empty: out of the vector
loop, nothing accumulated, no corrections. The satellite joins the navigation
engine on its first record.
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
passengers' readings too) and stop applying the corrections, keeping `vt_on` and
the satellite's place in the navigation engine. The corrections must leave the
NCO words with the re-seed: the converged Dopplers already contain the last
correction, so keeping it would apply it twice. The replica is still steered by
it, though, so the correction history and its landing lead, which the next code
measurement is moved to the epoch with, are kept.
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
# What the navigation engine does to a satellite's state. Each keeps the
# satellite's place in the engine.

# Put the satellite into the vector loop. A satellite joining starts with empty
# accumulators and no corrections; one already in the loop is returned unchanged.
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

# Hand the satellite back to its inner scalar loop, with the accumulators and
# corrections zeroed so a later re-enable never runs on stale values.
_disable_vector_tracking(state::SatVectorPLLAndDLL) = SatVectorPLLAndDLL(
    state.inner,
    false,
    state.passenger_readings,
    state.slot,
    state.registration,
    state.cycle_id,
)

# `_disable_vector_tracking` with the inner loop re-seeded from the Dopplers the
# replica runs at — what `reset_estimator_state` does — so the scalar loop takes over
# from where the vector loop steered the replica, without a transient.
_release_from_vector_tracking(state::SatVectorPLLAndDLL, carrier_doppler, code_doppler) =
    SatVectorPLLAndDLL(
        _reseed_inner(state.inner, carrier_doppler, code_doppler),
        false,
        state.passenger_readings,
        state.slot,
        state.registration,
        state.cycle_id,
    )

# The inner reset reads the per-satellite state only, never the estimator's
# configuration, so a default-configured estimator stands in for it.
_reseed_inner(state::SatConventionalPLLAndDLL, carrier_doppler, code_doppler) =
    reset_estimator_state(ConventionalPLLAndDLL(), state, carrier_doppler, code_doppler)
_reseed_inner(state::SatNCOReferencedPLLAndDLL, carrier_doppler, code_doppler) =
    reset_estimator_state(NCOReferencedPLLAndDLL(), state, carrier_doppler, code_doppler)

# The filter's NCO corrections for this satellite, taking over at `landing_lead`
# after the filter's cycle epoch. The code correction joins the front of
# `code_freq_update_history`.
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

# Empty the cycle's accumulators, the driver's and the passengers', once the snapshot
# has taken them. Passenger readings still pending stay for the driver's next record.
_reset_discriminator_accumulators(state::SatVectorPLLAndDLL) = SatVectorPLLAndDLL(
    state;
    code_discr_acc = (0, 0.0),
    carrier_discr_acc = (0, 0.0Hz),
    passenger_readings = map(_without_cycle_readings, state.passenger_readings),
)

# The mean DLL output (chips) and raw FLL discriminator (Hz) accumulated since the
# last reset, or `nothing` if nothing has been accumulated.
function _mean_code_discriminator(state::SatVectorPLLAndDLL)
    count, discr_sum = state.code_discr_acc
    count == 0 ? nothing : discr_sum / count
end

function _mean_carrier_discriminator(state::SatVectorPLLAndDLL)
    count, discr_sum = state.carrier_discr_acc
    count == 0 ? nothing : discr_sum / count
end

# One record through the satellite's loop: the inner loop out of the vector loop,
# the vector step with the discriminators accumulated in it.
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
    # A record without a previous prompt (the first after a (re)start, after a
    # pilot's secondary-code sync or a change of record length) has no FLL reading
    # (`fll_disc` reads 0 Hz) and is left out: counted, its 0 Hz would pull the
    # cycle's mean toward zero. The passengers' pending readings join this record's
    # cycle.
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

# The inner loop's record, in the vector loop: the discriminators exactly as the
# inner loop measures them (`_record_discriminators`; the NCO-referenced loop's
# phase error keeps its prediction to the landing sample), the carrier filter
# stepped with the filter's carrier correction in its FLL slot, the code filter
# frozen and the code correction in place of its output. Returns the inner
# state, both Dopplers and the two discriminators to accumulate: the DLL and the
# raw FLL — the mean offset from the replica that ran between the two prompts'
# centres, before any re-basing onto a landing word.
@inline function _step_vector_loop(
    estimator,
    state,
    record::LoopRecord,
    words,
    landing_sample::Int64,
    code_freq_update,
    carrier_freq_update,
)
    # The navigation filter owns the code loop and the FLL branch, so passengers
    # are combined into the PLL only here; their DLL and FLL readings go to the
    # filter from the engine (`fold_passenger_record`), each signal's apart. The
    # driver's FLL is always read, for the filter.
    discriminators = _with_passengers(
        state,
        record,
        _record_discriminators(estimator, state, record, words, landing_sample, true),
        _PLL_ONLY,
    )
    # Not staged: the vector loop runs the carrier filter at the narrow
    # bandwidth, its FLL slot tied to it as before the bandwidths were separated.
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
    # Not staged: the FLL slot carries the navigation filter's carrier update, and
    # the stage and the phase-lock indicator are left as they are.
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

# A vector loop takes the records of its groups' passengers when it combines them, and
# to decode a dataless driver's bits from its data passenger (see
# `fold_passenger_record` in engine.jl); a data driver decodes its own.
takes_passenger_records(estimator::VectorPLLAndDLL) =
    combines_signals(estimator) ||
    any(group -> !_drives_data_signal(group), estimator.navigation.groups)

carrier_loop_stage(state::SatVectorPLLAndDLL) = carrier_loop_stage(state.inner)
phase_lock_indicator(state::SatVectorPLLAndDLL) = phase_lock_indicator(state.inner)
