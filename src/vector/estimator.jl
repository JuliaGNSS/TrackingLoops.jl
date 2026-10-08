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
    VectorPLLAndDLL(signals...; inner = ConventionalAssistedPLLAndDLL(),
                    config = VectorTracking(), cycle_time = 100ms,
                    lock_cn0_threshold = 30dBHz, max_satellites_per_signal = 16,
                    num_prompts_for_cn0_estimation = 100,
                    approximate_year = year(now(UTC)),
                    enable_ionospheric_correction = true,
                    enable_tropospheric_correction = true)

Vector-tracking Doppler estimator for the ranging `signals` (one or more
`AbstractGNSSSignal`s, each at most once). It does the whole pipeline inside
[`step_loop`](@ref), from what the records carry: every satellite stepped with
this estimator shares one navigation engine, which syncs to the navigation
bits, decodes them, estimates the C/N₀, solves the scalar PVT, seeds the
navigation filter from its first fix and from then on runs one filter cycle
every `cycle_time`, taking satellites over and handing them back. A host needs
no vector-specific code: it builds the satellites' states with
[`init_estimator_state`](@ref) and steps them like any other loop's. Read the
results with [`navigation_solution`](@ref), [`navigation_status`](@ref),
[`release_reason`](@ref), [`member_sats`](@ref), [`position_uncertainty`](@ref)
and [`clock_uncertainty`](@ref).

The records must identify their satellite and replica: `prn` and `code_phase`
on the [`LoopRecord`](@ref), and `sample_index / sampling_frequency` on a time
grid shared by all satellites. Records of a signal not among `signals`, or
without a PRN, throw an `ArgumentError`. The host should step every satellite
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

Each satellite picks up the corrections of the latest cycle on its next record,
sized for where they land: at `landing_sample`, or at the record's end under
`NO_LANDING_SAMPLE`.

`inner` must have an FLL-assisted carrier filter, since that is the only input
path the carrier correction has into the loop: a
`ConventionalPLLAndDLL{ThirdOrderAssistedBilinearLF}` (what
[`ConventionalAssistedPLLAndDLL`](@ref) builds) or an
[`NCOReferencedPLLAndDLL`](@ref). With the latter the phase discriminator keeps
its prediction to the landing sample in the vector loop too. Anything else
throws an `ArgumentError`, as does a dataless signal (a pilot such as GPS L1C-P
or Galileo E1C): the estimator decodes the bits of the signal it steps.
`config = nothing` only ever solves the scalar PVT.

Storage is allocated at construction for `max_satellites_per_signal`
satellites per signal and only grows past that: a satellite that is dropped
leaves its storage to the next one.
"""
struct VectorPLLAndDLL{E<:AbstractDopplerEstimator,N} <: AbstractDopplerEstimator
    inner::E
    navigation::N
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
"""
struct SatVectorPLLAndDLL{S}
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
end

# A satellite with the vector interface empty — out of the loop or just joined —
# on the slot of `registration`.
SatVectorPLLAndDLL(inner, vt_on::Bool, slot::Int = 0, registration::Int = 0, cycle_id::Int = -1) =
    SatVectorPLLAndDLL(
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
    )

function SatVectorPLLAndDLL(
    state::SatVectorPLLAndDLL{S};
    inner::Maybe{S} = nothing,
    code_discr_acc::Maybe{Tuple{Int,Float64}} = nothing,
    carrier_discr_acc::Maybe{Tuple{Int,typeof(1.0Hz)}} = nothing,
    cycle_id::Maybe{Int} = nothing,
) where {S}
    SatVectorPLLAndDLL{S}(
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
    )
end

# The state on a fresh slot `slot` of `registration`, owing the pick-up of the
# latest cycle.
_registered(state::SatVectorPLLAndDLL{S}, slot::Int, registration::Int) where {S} =
    SatVectorPLLAndDLL{S}(
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
    )

"""
    init_estimator_state(estimator::VectorPLLAndDLL, driver_signal, carrier_doppler, code_doppler)

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
)

"""
    reset_estimator_state(estimator::VectorPLLAndDLL, state, carrier_doppler, code_doppler)

Re-seed the inner loop from the converged Dopplers, zero both accumulators and
stop applying the corrections, keeping `vt_on` and the satellite's place in the
navigation engine. The corrections must leave the NCO words with the re-seed:
the converged Dopplers already contain the last correction, so keeping it
would apply it twice. The replica is still steered by it, though, so the
correction history and its landing lead, which the next code measurement is
moved to the epoch with, are kept.
"""
reset_estimator_state(
    estimator::VectorPLLAndDLL,
    state::SatVectorPLLAndDLL{S},
    carrier_doppler,
    code_doppler,
) where {S} = SatVectorPLLAndDLL{S}(
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
)

# ─────────────────────────────────────────────────────────────────────────────
# What the navigation engine does to a satellite's state. Each keeps the
# satellite's place in the engine.

# Put the satellite into the vector loop. A satellite joining starts with empty
# accumulators and no corrections; one already in the loop is returned unchanged.
_enable_vector_tracking(state::SatVectorPLLAndDLL) =
    state.vt_on ? state :
    SatVectorPLLAndDLL(state.inner, true, state.slot, state.registration, state.cycle_id)

# Hand the satellite back to its inner scalar loop, with the accumulators and
# corrections zeroed so a later re-enable never runs on stale values.
_disable_vector_tracking(state::SatVectorPLLAndDLL) =
    SatVectorPLLAndDLL(state.inner, false, state.slot, state.registration, state.cycle_id)

# `_disable_vector_tracking` with the inner loop re-seeded from the Dopplers the
# replica runs at — what `reset_estimator_state` does — so the scalar loop takes over
# from where the vector loop steered the replica, without a transient.
_release_from_vector_tracking(state::SatVectorPLLAndDLL, carrier_doppler, code_doppler) =
    SatVectorPLLAndDLL(
        _reseed_inner(state.inner, carrier_doppler, code_doppler),
        false,
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
    state::SatVectorPLLAndDLL{S},
    code_freq_update,
    carrier_freq_update,
    landing_lead = 0.0s,
) where {S}
    newest, previous, _ = state.code_freq_update_history
    SatVectorPLLAndDLL{S}(
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
    )
end

# A record without a previous prompt (the first after a (re)start, after a
# pilot's secondary-code sync or a change of record length) has no FLL reading
# (`fll_disc` reads 0 Hz) and is left out of the accumulator: counted, its 0 Hz
# would pull the cycle's mean toward zero.
@inline _accumulated_fll(acc::Tuple, fll_discriminator, previous_prompt::Complex) =
    iszero(previous_prompt) ? acc : (acc[1] + 1, acc[2] + fll_discriminator)

# Empty both discriminator accumulators, once the filter has read them.
_reset_discriminator_accumulators(state::SatVectorPLLAndDLL) =
    SatVectorPLLAndDLL(state; code_discr_acc = (0, 0.0), carrier_discr_acc = (0, 0.0Hz))

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
    code_count, code_sum = state.code_discr_acc
    carrier_count, carrier_sum = state.carrier_discr_acc
    SatVectorPLLAndDLL(
        state;
        inner,
        code_discr_acc = (code_count + 1, code_sum + dll_discriminator),
        carrier_discr_acc = _accumulated_fll(
            state.carrier_discr_acc,
            fll_discriminator,
            record.previous_prompt,
        ),
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
    # are combined into the PLL only. The DLL and FLL readings accumulated for the
    # filter stay the driver's own; the FLL's is always read, for the filter.
    discriminators = _with_passengers(
        state,
        record,
        _record_discriminators(estimator, state, record, words, landing_sample, true),
        _PLL_ONLY,
    )
    carrier_filter_output, carrier_loop_filter = filter_loop(
        state.carrier_loop_filter,
        (discriminators.phase_error, carrier_freq_update),
        discriminators.integration_time,
        discriminators.carrier_bandwidth,
    )
    carrier_doppler, code_doppler = aid_dopplers(
        record.signal,
        state.init_carrier_doppler,
        state.init_code_doppler,
        carrier_filter_output,
        code_freq_update,
    )
    # Not staged: the FLL slot carries the navigation filter's carrier update, and
    # the frequency lock indicator is left as it is.
    _stepped_state(
        state,
        carrier_loop_filter,
        state.code_loop_filter,
        discriminators.center,
        state.frequency_lock,
    ),
    carrier_doppler,
    code_doppler,
    discriminators.code_error,
    discriminators.raw_frequency_error
end

combines_signals(estimator::VectorPLLAndDLL) = combines_signals(estimator.inner)

# In the vector loop passengers join the PLL only; out of it, the inner loop's.
function combine_passenger_record(
    estimator::VectorPLLAndDLL,
    state::SatVectorPLLAndDLL,
    record::LoopRecord,
    words;
    driver_signal::AbstractGNSSSignal,
    differential_group_delay_chips::Real = NaN,
)
    combines_signals(estimator) || return state
    loops = state.vt_on ? _PLL_ONLY : _scalar_loops_to_combine(state.inner)
    inner = _with_passenger_record(
        state.inner,
        record,
        words,
        loops,
        driver_signal,
        differential_group_delay_chips,
    )
    SatVectorPLLAndDLL(state; inner)
end

drop_pending_combining(estimator::VectorPLLAndDLL, state::SatVectorPLLAndDLL) =
    SatVectorPLLAndDLL(state; inner = drop_pending_combining(estimator.inner, state.inner))
