# ─────────────────────────────────────────────────────────────────────────────
# What every Doppler estimator offers a host, besides its configuration and
# per-satellite state: the per-record `step_loop`, called by Tracking and by a hardware
# correlator's loop:
#
#     step_loop(estimator, state, record, words, landing_sample)
#         -> (state, carrier_doppler, code_doppler)
#
# `words` answers `mean_nco_word(words, a, b)` (a `FixedNCOWord` in software, the
# channel's `NCOTimeline` in hardware); `landing_sample` is the device sample the
# fold's command takes effect at, `NO_LANDING_SAMPLE` for each record's end.
# ─────────────────────────────────────────────────────────────────────────────

# Whether `estimator` combines passenger discriminators into the driver's loops: a
# `ConventionalPLLAndDLL` or `VectorPLLAndDLL` built with `combine_signals = true`.
# Internal: a host asks `takes_passenger_records`, which also covers a vector loop that
# takes passenger records to decode a pilot driver's data without combining.
combines_signals(::AbstractDopplerEstimator) = false

"""
    takes_passenger_records(estimator) -> Bool

Whether the host should hand `estimator` every passenger record with
[`fold_passenger_record`](@ref): if it combines signals (built with
`combine_signals = true`), and for a [`VectorPLLAndDLL`](@ref) with a dataless
driver (it decodes the navigation data from the data passenger). If `false`, a host
may skip the passengers.
"""
takes_passenger_records(estimator::AbstractDopplerEstimator) = combines_signals(estimator)

"""
    fold_passenger_record(estimator, state, record::LoopRecord, words;
                          driver_signal, differential_group_delay_chips = NaN)
        -> state

Fold one completed passenger record into the per-satellite `state`. A scalar loop
that combines signals adds its discriminators, weighted by the signal's ICD power
share and integration time, to the sums the driver's next [`step_loop`](@ref)
closes on (the FLL only in the FLL-assisted stage); for [`VectorPLLAndDLL`](@ref)
see there. Call it for every passenger record in sample order, on the driver's sample
frame, each before the driver record it ends within or at; `words` are the
satellite's replica words.

  - `record`: the passenger's own record, `previous_prompt` per
    [`LoopRecord`](@ref) over the passenger's sequence. The scalar loops read its
    two-quadrant discriminators regardless of `polarity`.
  - `driver_signal`: rotates the passenger's prompt into the driver's carrier
    phase frame by the nominal carrier phase offsets.
  - `differential_group_delay_chips`: passenger minus driver group delay in
    chips, referring its DLL reading to the driver's code phase; `NaN` leaves the
    passenger out of the code loop.

Returns `state` unchanged unless [`takes_passenger_records`](@ref). See
[Signal combining](@ref) and [Host contract](@ref).
"""
@inline fold_passenger_record(
    ::AbstractDopplerEstimator,
    state,
    record::LoopRecord,
    words;
    driver_signal::AbstractGNSSSignal,
    differential_group_delay_chips::Real = NaN,
) = state

"""
    carrier_loop_stage(state) -> CarrierLoopStage

The [`CarrierLoopStage`](@ref) of a scalar estimator's per-satellite state (the
inner loop's for a [`SatVectorPLLAndDLL`](@ref)).
"""
carrier_loop_stage(state::Union{SatConventionalPLLAndDLL,SatNCOReferencedPLLAndDLL}) =
    carrier_loop_stage(state.staging)

"""
    phase_lock_indicator(state) -> Float64

The latest phase-lock indicator of a scalar estimator's per-satellite state (the
inner loop's for a [`SatVectorPLLAndDLL`](@ref), frozen while the satellite is in the
vector loop): `⟨I² − Q²⟩ / A²` ≈ `cos 2φ` over
0.1 s (at least 25 records) of driver prompts, `A²` the moment-estimated signal
power: 1 in lock, 0 for a spinning phase, at any C/N₀; `NaN` until the averages
span that time. Staging compares it to [`phase_lock_indicator_threshold`](@ref);
smooth it over your own horizon before making lock decisions.
"""
phase_lock_indicator(state::Union{SatConventionalPLLAndDLL,SatNCOReferencedPLLAndDLL}) =
    phase_lock_indicator(state.staging)

"The estimator-state type a Doppler estimator produces (for slot typing)."
estimator_state_type(
    estimator::AbstractDopplerEstimator,
    driver_signal::AbstractGNSSSignal,
) = typeof(init_estimator_state(estimator, driver_signal, 0.0Hz, 0.0Hz))

# ── What an estimator knows of the navigation solution ───────────────────────

"""
    navigation_solution(estimator) -> Union{PVTSolution,Nothing}

The latest navigation solution, or `nothing` for the scalar loops.
[`VectorPLLAndDLL`](@ref) returns the scalar PVT's until its filter is seeded,
then the filter's. Its containers are reused by the next cycle: copy what you
keep.
"""
navigation_solution(::AbstractDopplerEstimator) = nothing

"""
    navigation_status(estimator) -> Union{VTStatus,Nothing}

What the latest navigation cycle did ([`VTStatus`](@ref)), or `nothing` for an
estimator without one.
"""
navigation_status(::AbstractDopplerEstimator) = nothing

"""
    navigation_cycle(estimator) -> Union{Int,Nothing}

Number of navigation cycles run, or `nothing` without them. Changes exactly when
[`navigation_solution`](@ref) and [`navigation_status`](@ref) do, so polling it
reads each solution once.
"""
navigation_cycle(::AbstractDopplerEstimator) = nothing

"""
    navigation_epoch(estimator) -> Union{typeof(1.0s),Nothing}

The epoch of the latest navigation solution as `sample_index /
sampling_frequency`; `nothing` before the first cycle or without cycles.
"""
navigation_epoch(::AbstractDopplerEstimator) = nothing

"""
    satellite_report(estimator, signal, prn) -> Union{SatelliteReport,Nothing}

A [`SatelliteReport`](@ref) of satellite `prn` of `signal`, or `nothing` for the
scalar loops or a satellite never seen.
"""
satellite_report(::AbstractDopplerEstimator, ::AbstractGNSSSignal, ::Integer) = nothing
