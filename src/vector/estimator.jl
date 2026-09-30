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
    VectorPLLAndDLL(inner = ConventionalAssistedPLLAndDLL())

Vector-tracking Doppler estimator around the scalar loop `inner`, which every
satellite runs until a vector-tracking filter takes it over (and runs again
once the filter releases it). Per-satellite state is a
[`SatVectorPLLAndDLL`](@ref), produced via [`init_estimator_state`](@ref).

While a satellite is in the vector loop (`vt_on`), per record:

  - its carrier filter always runs, with the PLL branch fed by the satellite's
    own phase discriminator and the FLL branch fed by the filter's carrier
    correction (the filter integrates it as a frequency error; it is not added
    to the Doppler directly);
  - its code filter is frozen, and the filter's code correction replaces its
    output: `code_doppler = init_code_doppler + code_freq_update +
    carrier_filter_output · code_center_frequency_ratio`;
  - the DLL output (chips) and the raw FLL discriminator (Hz) are accumulated
    for the filter to read ([`mean_code_discriminator`](@ref),
    [`mean_carrier_discriminator`](@ref)).

`inner` must have an FLL-assisted carrier filter, since that is the only input
path the carrier correction has into the loop: a
`ConventionalPLLAndDLL{ThirdOrderAssistedBilinearLF}` (what
[`ConventionalAssistedPLLAndDLL`](@ref) builds) or an
[`NCOReferencedPLLAndDLL`](@ref). With the latter the phase discriminator keeps
its prediction to the landing sample in the vector loop too. Anything else
throws an `ArgumentError`.
"""
struct VectorPLLAndDLL{E<:AbstractDopplerEstimator} <: AbstractDopplerEstimator
    inner::E
    function VectorPLLAndDLL(inner::E) where {E<:AbstractDopplerEstimator}
        _is_fll_assisted(inner) || throw(
            ArgumentError(
                "the vector loop drives the carrier through the FLL branch of an " *
                "FLL-assisted carrier filter; use `ConventionalAssistedPLLAndDLL()` " *
                "or `NCOReferencedPLLAndDLL()` as the inner estimator",
            ),
        )
        new{E}(inner)
    end
end

VectorPLLAndDLL() = VectorPLLAndDLL(ConventionalAssistedPLLAndDLL())

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
  - `code_freq_update` / `carrier_freq_update`: the corrections in effect,
    written by [`set_vector_corrections`](@ref) only — the per-record step never
    writes them.
  - `previous_code_freq_update` and `code_update_landing_lead`: the code
    correction that was in effect before `code_freq_update`, and how long after
    the filter's cycle epoch `code_freq_update` reached the replica (zero for a
    software correlator). The filter's next code measurement needs both to move
    the mid-cycle mean discriminator to the epoch.
"""
struct SatVectorPLLAndDLL{S}
    inner::S
    vt_on::Bool
    code_discr_acc::Tuple{Int,Float64}
    carrier_discr_acc::Tuple{Int,typeof(1.0Hz)}
    code_freq_update::typeof(1.0Hz)
    carrier_freq_update::typeof(1.0Hz)
    previous_code_freq_update::typeof(1.0Hz)
    code_update_landing_lead::typeof(1.0s)
end

SatVectorPLLAndDLL(inner, vt_on::Bool) =
    SatVectorPLLAndDLL(inner, vt_on, (0, 0.0), (0, 0.0Hz), 0.0Hz, 0.0Hz, 0.0Hz, 0.0s)

function SatVectorPLLAndDLL(
    state::SatVectorPLLAndDLL{S};
    inner::Maybe{S} = nothing,
    code_discr_acc::Maybe{Tuple{Int,Float64}} = nothing,
    carrier_discr_acc::Maybe{Tuple{Int,typeof(1.0Hz)}} = nothing,
) where {S}
    SatVectorPLLAndDLL{S}(
        something(inner, state.inner),
        state.vt_on,
        something(code_discr_acc, state.code_discr_acc),
        something(carrier_discr_acc, state.carrier_discr_acc),
        state.code_freq_update,
        state.carrier_freq_update,
        state.previous_code_freq_update,
        state.code_update_landing_lead,
    )
end

"""
    init_estimator_state(estimator::VectorPLLAndDLL, driver_signal, carrier_doppler, code_doppler)

The inner loop's state with the vector interface empty: out of the vector
loop, nothing accumulated, no corrections.
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

Re-seed the inner loop from the converged Dopplers and zero both accumulators
and all corrections, keeping `vt_on`. The corrections must go with the re-seed:
the converged Dopplers already contain the last correction, so keeping it
would apply it twice.
"""
reset_estimator_state(
    estimator::VectorPLLAndDLL,
    state::SatVectorPLLAndDLL,
    carrier_doppler,
    code_doppler,
) = SatVectorPLLAndDLL(
    reset_estimator_state(estimator.inner, state.inner, carrier_doppler, code_doppler),
    state.vt_on,
)

"""
    enable_vector_tracking(state::SatVectorPLLAndDLL) -> SatVectorPLLAndDLL

Put the satellite into the vector loop. A satellite joining starts with empty
accumulators and no corrections; one already in the loop is returned
unchanged, so the current members can be enabled every cycle.
"""
enable_vector_tracking(state::SatVectorPLLAndDLL) =
    state.vt_on ? state : SatVectorPLLAndDLL(state.inner, true)

"""
    disable_vector_tracking(state::SatVectorPLLAndDLL) -> SatVectorPLLAndDLL

Hand the satellite back to its inner scalar loop, with the accumulators and
corrections zeroed so a later re-enable never runs on stale values. The inner
loop resumes from the filter state it had when the satellite joined, while its
Doppler already carries the vector loop's steering; follow up with
[`reset_estimator_state`](@ref) for a transient-free handover.
"""
disable_vector_tracking(state::SatVectorPLLAndDLL) = SatVectorPLLAndDLL(state.inner, false)

"""
    release_from_vector_tracking(state::SatVectorPLLAndDLL, carrier_doppler, code_doppler)
        -> SatVectorPLLAndDLL

[`disable_vector_tracking`](@ref) with the inner loop re-seeded from the Dopplers
the replica runs at — what [`reset_estimator_state`](@ref) does — so the scalar
loop takes over from where the vector loop steered the replica, without a
transient. This is how the vector-tracking filter hands a satellite back.
"""
release_from_vector_tracking(state::SatVectorPLLAndDLL, carrier_doppler, code_doppler) =
    SatVectorPLLAndDLL(_reseed_inner(state.inner, carrier_doppler, code_doppler), false)

# The inner reset reads the per-satellite state only, never the estimator's
# configuration, so a default-configured estimator stands in for it.
_reseed_inner(state::SatConventionalPLLAndDLL, carrier_doppler, code_doppler) =
    reset_estimator_state(ConventionalPLLAndDLL(), state, carrier_doppler, code_doppler)
_reseed_inner(state::SatNCOReferencedPLLAndDLL, carrier_doppler, code_doppler) =
    reset_estimator_state(NCOReferencedPLLAndDLL(), state, carrier_doppler, code_doppler)

"""
    set_vector_corrections(state, code_freq_update, carrier_freq_update, landing_lead = 0.0s)

The filter's NCO corrections for this satellite, taking over at `landing_lead`
after the filter's cycle epoch (zero for a software correlator). The code
correction in effect so far is kept as `previous_code_freq_update`. Applies to
a satellite out of the vector loop too, where the corrections are simply not
used.
"""
set_vector_corrections(
    state::SatVectorPLLAndDLL{S},
    code_freq_update,
    carrier_freq_update,
    landing_lead = 0.0s,
) where {S} = SatVectorPLLAndDLL{S}(
    state.inner,
    state.vt_on,
    state.code_discr_acc,
    state.carrier_discr_acc,
    code_freq_update,
    carrier_freq_update,
    state.code_freq_update,
    landing_lead,
)

"""
    reset_discriminator_accumulators(state::SatVectorPLLAndDLL) -> SatVectorPLLAndDLL

Empty both discriminator accumulators, once the filter has read them.
"""
reset_discriminator_accumulators(state::SatVectorPLLAndDLL) =
    SatVectorPLLAndDLL(state; code_discr_acc = (0, 0.0), carrier_discr_acc = (0, 0.0Hz))

"""
    mean_code_discriminator(state::SatVectorPLLAndDLL)

The mean DLL output accumulated since the last reset, in chips, or `nothing`
if nothing has been accumulated.
"""
function mean_code_discriminator(state::SatVectorPLLAndDLL)
    count, discr_sum = state.code_discr_acc
    count == 0 ? nothing : discr_sum / count
end

"""
    mean_carrier_discriminator(state::SatVectorPLLAndDLL)

The mean raw FLL discriminator accumulated since the last reset, in Hz, or
`nothing` if nothing has been accumulated.
"""
function mean_carrier_discriminator(state::SatVectorPLLAndDLL)
    count, discr_sum = state.carrier_discr_acc
    count == 0 ? nothing : discr_sum / count
end

"""
    step_loop(estimator::VectorPLLAndDLL, state, record::LoopRecord, words, landing_sample)
        -> (state, carrier_doppler, code_doppler)

One record through the vector loop. Out of the vector loop this is
`step_loop(estimator.inner, …)` exactly. In it, see
[`VectorPLLAndDLL`](@ref).
"""
@inline function step_loop(
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
    # A record without a previous prompt (the first after a (re)start) has an
    # FLL discriminator of 0 Hz and still counts, as in Tracking: the first
    # cycle's mean is then pulled toward 0 Hz, and the member counts as
    # measured although it has no real frequency measurement yet.
    SatVectorPLLAndDLL(
        state;
        inner,
        code_discr_acc = (code_count + 1, code_sum + dll_discriminator),
        carrier_discr_acc = (carrier_count + 1, carrier_sum + fll_discriminator),
    ),
    carrier_doppler,
    code_doppler
end

# The inner loop's record, in the vector loop: the carrier filter stepped with
# the filter's carrier correction in its FLL slot, the code filter frozen and
# the code correction in place of its output. Returns the inner state, both
# Dopplers and the two discriminators to accumulate: the DLL (normalised with
# the code word the record ran on, as in the scalar loop) and the raw FLL — the
# mean offset from the replica that ran between the two prompts' centres,
# before any re-basing onto a landing word.
@inline function _step_vector_loop(
    ::ConventionalPLLAndDLL,
    state::SatConventionalPLLAndDLL,
    record::LoopRecord,
    words,
    landing_sample::Int64,
    code_freq_update,
    carrier_freq_update,
)
    signal = record.signal
    integration_time = record.integrated_samples / record.sampling_frequency
    record_start = record.sample_index - record.integrated_samples
    _, applied_code = mean_nco_word(words, record_start, record.sample_index)
    carrier_bandwidth = state.carrier_loop_filter_bandwidth / record.integrated_code_blocks
    filtered_correlator = record.filtered_correlator
    phase_error = pll_disc(signal, filtered_correlator)
    frequency_error =
        fll_disc(signal, filtered_correlator, record.previous_prompt, integration_time)
    dll_discriminator =
        dll_disc(signal, filtered_correlator, applied_code * Hz, record.sampling_frequency)
    carrier_filter_output, carrier_loop_filter = filter_loop(
        state.carrier_loop_filter,
        (phase_error, carrier_freq_update),
        integration_time,
        carrier_bandwidth,
    )
    carrier_doppler, code_doppler = aid_dopplers(
        signal,
        state.init_carrier_doppler,
        state.init_code_doppler,
        carrier_filter_output,
        code_freq_update,
    )
    SatConventionalPLLAndDLL(state; carrier_loop_filter),
    carrier_doppler,
    code_doppler,
    dll_discriminator,
    frequency_error
end

@inline function _step_vector_loop(
    estimator::NCOReferencedPLLAndDLL,
    state::SatNCOReferencedPLLAndDLL,
    record::LoopRecord,
    words,
    landing_sample::Int64,
    code_freq_update,
    carrier_freq_update,
)
    signal = record.signal
    sampling_frequency = record.sampling_frequency
    shift = landing_sample == NO_LANDING_SAMPLE ? 0 : landing_sample - record.fold_end
    record_end = record.sample_index
    record_samples = record.integrated_samples
    record_start = record_end - record_samples
    center = record_end - record_samples / 2
    integration_time = record_samples / sampling_frequency
    filtered_correlator = record.filtered_correlator
    _, applied_code = mean_nco_word(words, record_start, record_end)
    phase_error = pll_disc(signal, filtered_correlator)
    frequency_error =
        fll_disc(signal, filtered_correlator, record.previous_prompt, integration_time)
    # The PLL keeps its landing prediction; the FLL slot carries the filter's
    # correction, so the frequency re-basing has nothing to act on.
    if estimator.predict_landing && shift > 0
        phase_error = _predict_landing_phase_error(
            phase_error,
            state,
            words,
            center,
            shift,
            integration_time,
            sampling_frequency,
        )
    end
    dll_discriminator =
        dll_disc(signal, filtered_correlator, applied_code * Hz, sampling_frequency)
    carrier_bandwidth = state.carrier_loop_filter_bandwidth / record.integrated_code_blocks
    carrier_filter_output, carrier_loop_filter = filter_loop(
        state.carrier_loop_filter,
        (phase_error, carrier_freq_update),
        integration_time,
        carrier_bandwidth,
    )
    carrier_doppler, code_doppler = aid_dopplers(
        signal,
        state.init_carrier_doppler,
        state.init_code_doppler,
        carrier_filter_output,
        code_freq_update,
    )
    SatNCOReferencedPLLAndDLL(state; carrier_loop_filter, previous_record_center = center),
    carrier_doppler,
    code_doppler,
    dll_discriminator,
    frequency_error
end
