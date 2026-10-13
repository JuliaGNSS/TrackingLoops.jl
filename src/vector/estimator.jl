# The per-record vector loop: a scalar FLL-assisted PLL/DLL that a vector-tracking
# filter can take over (see `VectorPLLAndDLL`).

"""
    VectorPLLAndDLL(signals...; inner = ConventionalAssistedPLLAndDLL(),
                    config = VectorTracking(), cycle_time = 100ms,
                    lock_cn0_threshold = 30dBHz, max_satellites_per_signal = 16,
                    approximate_year = year(now(UTC)),
                    enable_ionospheric_correction = true,
                    enable_tropospheric_correction = true)

Vector-tracking Doppler estimator for the ranging `signals`: each a signal that carries
navigation data (`GPSL1CA()`) or a pilot + data pair `driver => decoding_signal`
(`GalileoE1C() => GalileoE1B()`; see [Pilot + data pairs](@ref)). A decoding signal
must carry data and have its driver's chip rate. No signal may be given twice.

All satellites stepped with this estimator share one navigation engine which, inside
[`step_loop`](@ref), decodes the navigation data, solves the scalar PVT, seeds the
navigation filter from its first fix and then runs one filter cycle every
`cycle_time`, taking satellites over and handing them back. A host needs no
vector-specific code: it builds a satellite's state with
[`init_estimator_state`](@ref) for its driver and steps it with every record of the
satellite. Read the results, keyed by the driver, with
[`navigation_solution`](@ref), [`navigation_status`](@ref),
[`release_reason`](@ref), [`member_sats`](@ref), [`position_uncertainty`](@ref) and
[`clock_uncertainty`](@ref).

The engine reads the host's bit clock and C/N₀ estimator from the records (see
[What the records must carry](@ref)); the C/N₀, read at each navigation epoch for the
lock and the measurement weights, is capped at 80 dB-Hz. A record without a PRN, or
of a driver not among `signals`, throws an `ArgumentError`; any other passenger is
ignored, also the decoding signal of a pair that does not drive this satellite. Step
every signal of every satellite at least once per `cycle_time / 2` (see
[Stepping the satellites](@ref)).

Every driver record runs `inner` until the filter takes the satellite over, and again
once it is released. In the vector loop:

  - the carrier filter runs at the narrow PLL bandwidth, the FLL branch's tied to it,
    whatever [`CarrierLoopStage`](@ref) the scalar loop had reached
    ([`carrier_loop_stage`](@ref) reports `NARROW_PLL`); its PLL branch is fed by the
    satellite's own phase discriminator, its FLL branch by the filter's carrier
    correction (integrated as a frequency error, not added to the Doppler). A release
    restarts the staging, FLL-assisted, so the scalar loop pulls in afresh;
  - the code filter is frozen and the filter's code correction replaces its output:
    `code_doppler = init_code_doppler + code_freq_update +
    carrier_filter_output · code_center_frequency_ratio`;
  - the DLL output (chips) and raw FLL discriminator (Hz) are accumulated for the
    filter.

The latest cycle's corrections are taken up on the next driver record, sized for
where they land: at `landing_sample`, or at the record's end under
`NO_LANDING_SAMPLE`. A record of any other signal returns the command in force there.

`inner` must have an FLL-assisted carrier filter, the carrier correction's only way
into the loop: a `ConventionalPLLAndDLL{ThirdOrderAssistedBilinearLF}` (what
[`ConventionalAssistedPLLAndDLL`](@ref) builds) or an
[`NCOReferencedPLLAndDLL`](@ref), whose phase discriminator keeps its prediction to
the landing sample in the vector loop too. Anything else throws an `ArgumentError`,
as does a dataless signal given alone (pair a pilot such as GPS L1C-P or Galileo E1C
with its data component). `config = nothing` only ever solves the scalar PVT.

Storage is allocated for `max_satellites_per_signal` satellites per ranging signal
and grows only past that; a dropped satellite leaves its storage to the next one.
"""
struct VectorPLLAndDLL{E<:AbstractDopplerEstimator,N} <: AbstractDopplerEstimator
    inner::E
    navigation::N
end

# Whether an estimator's carrier filter has an FLL path: the FLL-assisted
# conventional loop and the NCO-referenced loop.
_has_fll_path(::AbstractDopplerEstimator) = false
_has_fll_path(::ConventionalPLLAndDLL{CA}) where {CA} = _has_fll_path(CA)
_has_fll_path(::NCOReferencedPLLAndDLL) = true

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

_is_driver_record(state::SatVectorPLLAndDLL, record::LoopRecord) =
    _is_driver_record(state.inner, record)
_is_driven_by(state::SatVectorPLLAndDLL, signal::AbstractGNSSSignal) =
    _is_driven_by(state.inner, signal)

# A satellite with the vector interface empty, on the slot of `registration`.
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
        something(inner, state.inner),
        state.vt_on,
        something(code_discr_acc, state.code_discr_acc),
        something(carrier_discr_acc, state.carrier_discr_acc),
        state.code_freq_update,
        state.carrier_freq_update,
        state.code_freq_update_history,
        state.code_update_landing_lead,
        state.slot,
        state.registration,
        something(cycle_id, state.cycle_id),
    )
end

# The state on a fresh slot, owing the pick-up of the latest cycle.
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

Re-seed the inner loop from the converged Dopplers, zero the accumulators and drop
the corrections, keeping `vt_on` and the satellite's place in the navigation engine.
The converged Dopplers already contain the last correction, so keeping it would apply
it twice; the replica was still steered by it, so the correction history and landing
lead are kept.
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

# `NARROW_PLL` in the vector loop (see `VectorPLLAndDLL`), else the inner loop's stage.
carrier_loop_stage(state::SatVectorPLLAndDLL) =
    state.vt_on ? NARROW_PLL : carrier_loop_stage(state.inner)
phase_lock_indicator(state::SatVectorPLLAndDLL) = phase_lock_indicator(state.inner)

# ─────────────────────────────────────────────────────────────────────────────
# What the navigation engine does to a satellite's state; each keeps its place in
# the engine.

# Into the vector loop with empty accumulators and no corrections; a no-op if in it.
_enable_vector_tracking(state::SatVectorPLLAndDLL) =
    state.vt_on ? state :
    SatVectorPLLAndDLL(state.inner, true, state.slot, state.registration, state.cycle_id)

# Back to the inner loop, accumulators and corrections zeroed so a later re-enable
# never runs on stale values.
_disable_vector_tracking(state::SatVectorPLLAndDLL) =
    SatVectorPLLAndDLL(state.inner, false, state.slot, state.registration, state.cycle_id)

# `_disable_vector_tracking` with the inner loop re-seeded from the replica's Dopplers
# (as `reset_estimator_state`), so the scalar loop takes over without a transient. The
# re-seed restarts the staging: FLL-assisted at the wide bandwidth, which keeps a
# satellite the vector loop let go.
_release_from_vector_tracking(state::SatVectorPLLAndDLL, carrier_doppler, code_doppler) =
    SatVectorPLLAndDLL(
        _reseed_inner(state.inner, carrier_doppler, code_doppler),
        false,
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

# Empty both accumulators once the filter has read them.
_reset_discriminator_accumulators(state::SatVectorPLLAndDLL) =
    SatVectorPLLAndDLL(state; code_discr_acc = (0, 0.0), carrier_discr_acc = (0, 0.0Hz))

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
    code_count, code_sum = state.code_discr_acc
    carrier_count, carrier_sum = state.carrier_discr_acc
    # A record without a previous prompt (see `LoopRecord`) reads 0 Hz on the FLL and
    # still counts, as in Tracking.jl: it pulls the first cycle's mean toward 0 Hz, and
    # the member counts as measured without a real frequency measurement.
    SatVectorPLLAndDLL(
        state;
        inner,
        code_discr_acc = (code_count + 1, code_sum + dll_discriminator),
        carrier_discr_acc = (carrier_count + 1, carrier_sum + fll_discriminator),
    ),
    carrier_doppler,
    code_doppler
end

# The inner loop's step in the vector loop (see `VectorPLLAndDLL`), with the
# discriminators as the inner loop measures them. Returns the inner state, both
# Dopplers, and the DLL and raw FLL readings to accumulate; the raw FLL is the mean
# offset from the replica between the two prompts' centres, before any re-basing
# onto a landing word.
@inline function _step_vector_loop(
    estimator,
    state,
    record::LoopRecord,
    words,
    landing_sample::Int64,
    code_freq_update,
    carrier_freq_update,
)
    discriminators = _record_discriminators(estimator, state, record, words, landing_sample; form_fll = true)
    integration_time = discriminators.integration_time
    # Narrow bandwidth whatever the stage, the FLL slot carrying the carrier correction.
    carrier_filter_output, carrier_loop_filter = filter_loop(
        state.carrier_loop_filter,
        (discriminators.phase_error, carrier_freq_update),
        integration_time,
        effective_carrier_loop_filter_bandwidth(state.bandwidths.narrow_carrier, integration_time),
    )
    carrier_doppler, code_doppler = aid_dopplers(
        record.signal,
        state.init_carrier_doppler,
        state.init_code_doppler,
        carrier_filter_output,
        code_freq_update,
    )
    # The staging advances only to keep `phase_lock_indicator` live.
    _stepped_state(
        state,
        carrier_loop_filter,
        state.code_loop_filter,
        discriminators.center,
        _advance_staging(state.staging, record),
    ),
    carrier_doppler,
    code_doppler,
    discriminators.code_error,
    discriminators.raw_frequency_error
end
