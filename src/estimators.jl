# ─────────────────────────────────────────────────────────────────────────────
# The scalar Doppler estimators: configuration, per-satellite state, and the
# per-record `step_loop`, called by Tracking and by a hardware correlator's loop:
#
#     step_loop(estimator, state, record, words, landing_sample)
#         -> (state, carrier_doppler, code_doppler)
#
# `words` answers `mean_nco_word(words, a, b)` (a `FixedNCOWord` in software, the
# channel's `NCOTimeline` in hardware); `landing_sample` is the device sample the
# fold's command takes effect at, `NO_LANDING_SAMPLE` for each record's end.
# ─────────────────────────────────────────────────────────────────────────────

"""
$(SIGNATURES)

One completed record as the loop-filter step sees it: the signal, the filtered
(antenna-combined, normalised) correlator, the previous filtered prompt, the
record's span and block count, the band's sampling frequency, and `fold_end`, the
end sample of the fold's last record, against which a landing sample is measured.

Fields for an estimator with per-satellite state of its own
([`VectorPLLAndDLL`](@ref)):

  - `prn`: the satellite (`0` if unknown);
  - `code_phase`: the replica's code phase (chips) at `sample_index`, from the
    [`CorrelatorOutput`](@ref) (`NaN` if not reported);
  - `cn0`: the host's C/N₀ estimate in dB-Hz (`NaN` by default), which weights
    the navigation filter's measurements and decides lock; without it the vector
    loop estimates C/N₀ from the prompts. The scalar loops ignore it.

A satellite's driver and passenger records share one sample frame. For such an
estimator `sample_index / sampling_frequency` must be the time since an origin
shared by every satellite of the band; a host whose correlator restarts its
sample count passes the offset as `sample_offset` (added to `sample_index` and
`fold_end`).

`polarity`, from the bit buffer as it was when the record was correlated (before
the fold that may sync it), picks the carrier discriminators: the prompt's sign from
the secondary-code sync ([`get_sync_polarity`](@ref)). Nonzero, the replica wipes
every sign modulation off the prompt, so the PLL and the FLL are four-quadrant; `0`
(default) keeps both two-quadrant (the Costas PLL).

`previous_prompt` is zero, giving no FLL reading, for the first record and for a
record whose length or `polarity` differs from the previous one: the FLL divides
the rotation by this record's integration time, which is the time between the
prompts only for records of one length, and a wipe-off change would read as half a
cycle. The constructor that takes a [`SignalLoopState`](@ref) applies
these rules.
"""
struct LoopRecord{S<:AbstractGNSSSignal,C<:AbstractCorrelator,F}
    signal::S
    filtered_correlator::C
    previous_prompt::ComplexF64
    integrated_samples::Int
    sample_index::Int
    fold_end::Int
    integrated_code_blocks::Int
    sampling_frequency::F
    prn::Int
    code_phase::Float64
    polarity::Int8
    cn0::Float64
end

LoopRecord(
    signal::AbstractGNSSSignal,
    filtered_correlator::AbstractCorrelator,
    previous_prompt,
    integrated_samples::Integer,
    sample_index::Integer,
    fold_end::Integer,
    integrated_code_blocks::Integer,
    sampling_frequency;
    prn::Integer = 0,
    code_phase::Real = NaN,
    polarity::Integer = 0,
    cn0 = NaN,
) = LoopRecord(
    signal,
    filtered_correlator,
    ComplexF64(previous_prompt),
    Int(integrated_samples),
    Int(sample_index),
    Int(fold_end),
    Int(integrated_code_blocks),
    sampling_frequency,
    Int(prn),
    Float64(code_phase),
    Int8(polarity),
    _record_cn0(cn0),
)

LoopRecord(
    signal,
    filtered_correlator,
    previous_prompt,
    output::CorrelatorOutput,
    integrated_code_blocks,
    sampling_frequency;
    fold_end = output.sample_index,
    prn::Integer = 0,
    sample_offset::Integer = 0,
    polarity::Integer = 0,
    cn0 = NaN,
) = LoopRecord(
    signal,
    filtered_correlator,
    ComplexF64(previous_prompt),
    output.integrated_samples,
    output.sample_index + Int(sample_offset),
    Int(fold_end) + Int(sample_offset),
    Int(integrated_code_blocks),
    sampling_frequency,
    Int(prn),
    output.code_phase,
    Int8(polarity),
    _record_cn0(cn0),
)

# C/N₀ in dB-Hz from a number or a dB-Hz quantity (as `estimate_cn0` returns).
_record_cn0(cn0::Real) = Float64(cn0)
_record_cn0(cn0) = Float64(ustrip(cn0))

"""
    LoopRecord(loop::SignalLoopState, signal, filtered_correlator,
               output::CorrelatorOutput, integrated_code_blocks, sampling_frequency;
               prn, fold_end = output.sample_index, sample_offset = 0, cn0 = NaN,
               correlated_pre_sync = false)

The [`LoopRecord`](@ref) of a record `apply_record` folded, built from the
signal's state `loop` *before* that fold, with `polarity` and `previous_prompt`
filled in by `LoopRecord`'s rules (block count compared via
`integrated_code_blocks`). `filtered_correlator` and `integrated_code_blocks` are
what `apply_record` returned; other arguments are as for the `CorrelatorOutput`
constructor. `prn` is required, as a secondary code's polarity can depend on it.

Pass `correlated_pre_sync` as to `apply_record`: a record after a sync found earlier
in its fold was correlated with the pre-sync replica and still carries the secondary
code, so it gets no polarity; read with the sync's, the four-quadrant PLL would take
every secondary chip flip for a half-cycle phase error.
"""
function LoopRecord(
    loop::SignalLoopState,
    signal::AbstractGNSSSignal,
    filtered_correlator,
    output::CorrelatorOutput,
    integrated_code_blocks::Integer,
    sampling_frequency;
    prn::Integer,
    fold_end = output.sample_index,
    sample_offset::Integer = 0,
    cn0 = NaN,
    correlated_pre_sync::Bool = false,
)
    polarity = _correlated_polarity(signal, loop.bit_buffer, prn, correlated_pre_sync)
    chains =
        integrated_code_blocks == loop.last_num_code_blocks &&
        polarity == loop.last_polarity
    LoopRecord(
        signal,
        filtered_correlator,
        chains ? loop.last_filtered_prompt : complex(0.0, 0.0),
        output,
        integrated_code_blocks,
        sampling_frequency;
        fold_end,
        prn,
        sample_offset,
        polarity,
        cn0,
    )
end

# ── The conventional PLL/DLL ─────────────────────────────────────────────────

"""
Per-satellite state of a [`ConventionalPLLAndDLL`](@ref): initial Dopplers, loop
filters, their [`LoopBandwidths`](@ref TrackingLoops.LoopBandwidths), the
[`CarrierLoopStaging`](@ref TrackingLoops.CarrierLoopStaging) (the
[`CarrierLoopStage`](@ref) and phase-lock indicator), and the passengers' pending
[`SignalCombiningSums`](@ref).
"""
@kwdef struct SatConventionalPLLAndDLL{CA<:AbstractLoopFilter,CO<:AbstractLoopFilter}
    init_carrier_doppler::typeof(1.0Hz)
    init_code_doppler::typeof(1.0Hz)
    carrier_loop_filter::CA = ThirdOrderBilinearLF()
    code_loop_filter::CO = SecondOrderBilinearLF()
    bandwidths::LoopBandwidths = LoopBandwidths()
    staging::CarrierLoopStaging = CarrierLoopStaging(carrier_loop_filter)
    signal_combining_sums::SignalCombiningSums = SignalCombiningSums()
end

function SatConventionalPLLAndDLL(
    sat_conventional_pll_and_dll::SatConventionalPLLAndDLL{CA,CO};
    carrier_loop_filter::Maybe{CA} = nothing,
    code_loop_filter::Maybe{CO} = nothing,
    bandwidths::Maybe{LoopBandwidths} = nothing,
    staging::Maybe{CarrierLoopStaging} = nothing,
    signal_combining_sums::Maybe{SignalCombiningSums} = nothing,
) where {CA<:AbstractLoopFilter,CO<:AbstractLoopFilter}
    state = sat_conventional_pll_and_dll
    SatConventionalPLLAndDLL{CA,CO}(
        state.init_carrier_doppler,
        state.init_code_doppler,
        something(carrier_loop_filter, state.carrier_loop_filter),
        something(code_loop_filter, state.code_loop_filter),
        something(bandwidths, state.bandwidths),
        something(staging, state.staging),
        something(signal_combining_sums, state.signal_combining_sums),
    )
end

"""
$(SIGNATURES)

Conventional PLL and DLL Doppler estimator. Configuration only; per-satellite
state is a [`SatConventionalPLLAndDLL`](@ref) from [`init_estimator_state`](@ref).

`CA` and `CO` are the carrier and code loop filter types. The bandwidths (wide,
code, narrow and FLL path; see [`CarrierLoopStage`](@ref)) default to `nothing`,
**auto**: sized per satellite from its driver signal via
[`default_wide_carrier_loop_filter_bandwidth`](@ref),
[`default_code_loop_filter_bandwidth`](@ref),
[`default_narrow_carrier_loop_filter_bandwidth`](@ref) and
[`default_fll_assist_loop_filter_bandwidth`](@ref). Each is capped against the
record's integration time at filter time (see [Loop-filter bandwidths](@ref)), so
longer integration needs no re-tuning.

`combine_signals = true` combines the discriminators of a satellite's other
signals (passengers, folded with [`fold_passenger_record`](@ref)) into the loops
of the driver signal [`step_loop`](@ref) runs on; see [Signal combining](@ref).
Passengers are assumed to integrate no longer than the driver: a longer one
enters only the driver record it ends in, dominating it (weight ∝ its integration
time, its cube for the FLL) with an FLL range of only ±1/(4·T_passenger). Make the
longest-integrating signal (typically the pilot) the driver.
"""
struct ConventionalPLLAndDLL{CA<:AbstractLoopFilter,CO<:AbstractLoopFilter} <:
       AbstractDopplerEstimator
    wide_carrier_loop_filter_bandwidth::Maybe{typeof(1.0Hz)}
    code_loop_filter_bandwidth::Maybe{typeof(1.0Hz)}
    narrow_carrier_loop_filter_bandwidth::Maybe{typeof(1.0Hz)}
    fll_assist_loop_filter_bandwidth::Maybe{typeof(1.0Hz)}
    combine_signals::Bool
end

function ConventionalPLLAndDLL(
    ::Type{CA} = ThirdOrderBilinearLF,
    ::Type{CO} = SecondOrderBilinearLF;
    wide_carrier_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    code_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    narrow_carrier_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    fll_assist_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    combine_signals::Bool = false,
) where {CA<:AbstractLoopFilter,CO<:AbstractLoopFilter}
    ConventionalPLLAndDLL{CA,CO}(
        wide_carrier_loop_filter_bandwidth,
        code_loop_filter_bandwidth,
        narrow_carrier_loop_filter_bandwidth,
        fll_assist_loop_filter_bandwidth,
        combine_signals,
    )
end

"""
$(SIGNATURES)

A [`ConventionalPLLAndDLL`](@ref) with the FLL-assisted carrier filter
`ThirdOrderAssistedBilinearLF` (see [`CarrierLoopStage`](@ref)). Keywords as
there.
"""
function ConventionalAssistedPLLAndDLL(
    ::Type{CO} = SecondOrderBilinearLF;
    wide_carrier_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    code_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    narrow_carrier_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    fll_assist_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    combine_signals::Bool = false,
) where {CO<:AbstractLoopFilter}
    ConventionalPLLAndDLL(
        ThirdOrderAssistedBilinearLF,
        CO;
        wide_carrier_loop_filter_bandwidth,
        code_loop_filter_bandwidth,
        narrow_carrier_loop_filter_bandwidth,
        fll_assist_loop_filter_bandwidth,
        combine_signals,
    )
end

# Copy with the given (non-`nothing`) fields replaced.
function ConventionalPLLAndDLL(
    pll_and_dll::ConventionalPLLAndDLL{CA,CO};
    wide_carrier_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    code_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    narrow_carrier_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    fll_assist_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    combine_signals::Maybe{Bool} = nothing,
) where {CA<:AbstractLoopFilter,CO<:AbstractLoopFilter}
    ConventionalPLLAndDLL{CA,CO}(
        isnothing(wide_carrier_loop_filter_bandwidth) ?
        pll_and_dll.wide_carrier_loop_filter_bandwidth : wide_carrier_loop_filter_bandwidth,
        isnothing(code_loop_filter_bandwidth) ? pll_and_dll.code_loop_filter_bandwidth :
        code_loop_filter_bandwidth,
        isnothing(narrow_carrier_loop_filter_bandwidth) ?
        pll_and_dll.narrow_carrier_loop_filter_bandwidth :
        narrow_carrier_loop_filter_bandwidth,
        isnothing(fll_assist_loop_filter_bandwidth) ?
        pll_and_dll.fll_assist_loop_filter_bandwidth : fll_assist_loop_filter_bandwidth,
        isnothing(combine_signals) ? pll_and_dll.combine_signals : combine_signals,
    )
end

"""
    init_estimator_state(estimator, driver_signal, carrier_doppler, code_doppler)

Per-satellite estimator state for a satellite driven by `driver_signal`, starting
at the given Dopplers; auto bandwidths are resolved here from `driver_signal`.
Must be **pure**: Tracking.jl also calls it for template states and re-seeding.
"""
function init_estimator_state(
    estimator::ConventionalPLLAndDLL{CA,CO},
    driver_signal::AbstractGNSSSignal,
    carrier_doppler,
    code_doppler,
) where {CA<:AbstractLoopFilter,CO<:AbstractLoopFilter}
    carrier_loop_filter = _constructorof(CA)()
    SatConventionalPLLAndDLL(
        carrier_doppler,
        code_doppler,
        carrier_loop_filter,
        _constructorof(CO)(),
        _resolve_bandwidths(estimator, driver_signal),
        CarrierLoopStaging(carrier_loop_filter),
        SignalCombiningSums(),
    )
end

# `Accessors.constructorof` without the dependency.
_constructorof(::Type{T}) where {T} = Base.typename(T).wrapper

"""
    reset_estimator_state(estimator, state, carrier_doppler, code_doppler)

Re-seed `state` at the given Dopplers: zeroed loop filters, restarted
[`CarrierLoopStage`](@ref) and phase-lock indicator, no pending passengers; the
bandwidths are kept.
"""
function reset_estimator_state(
    ::ConventionalPLLAndDLL,
    state::SatConventionalPLLAndDLL,
    carrier_doppler,
    code_doppler,
)
    carrier_loop_filter = _constructorof(typeof(state.carrier_loop_filter))()
    SatConventionalPLLAndDLL(
        carrier_doppler,
        code_doppler,
        carrier_loop_filter,
        _constructorof(typeof(state.code_loop_filter))(),
        state.bandwidths,
        CarrierLoopStaging(carrier_loop_filter),
        SignalCombiningSums(),
    )
end

"""
    step_loop(estimator::ConventionalPLLAndDLL, state, record::LoopRecord, words,
              landing_sample)
        -> (state, carrier_doppler, code_doppler)

One record through the conventional loop at the bandwidths of its
[`CarrierLoopStage`](@ref), capped against the record's integration time; the DLL
is normalised with the code word the record ran on. `landing_sample` is ignored:
the command is assumed to act before the next record.
"""
@inline step_loop(
    estimator::ConventionalPLLAndDLL,
    state::SatConventionalPLLAndDLL,
    record::LoopRecord,
    words,
    landing_sample::Int64,
) = _step_scalar_loop(estimator, state, record, words, landing_sample)

# One record's discriminators plus integration time, capped bandwidths and centre
# sample; shared with `_step_vector_loop` so the two cannot drift apart.
# `phase_error` (cycles) and `frequency_error` (Hz) feed the carrier filter;
# `raw_frequency_error` is the FLL reading before re-basing onto a landing word.
# The FLL is read only if `fll`, else both frequency errors are zero.
@inline function _record_discriminators(
    ::ConventionalPLLAndDLL,
    state::SatConventionalPLLAndDLL,
    record::LoopRecord,
    words,
    landing_sample::Int64,
    fll::Bool,
)
    signal = record.signal
    integration_time = record.integrated_samples / record.sampling_frequency
    record_start = record.sample_index - record.integrated_samples
    _, applied_code = mean_nco_word(words, record_start, record.sample_index)
    filtered_correlator = record.filtered_correlator
    frequency_error =
        fll ?
        fll_disc(
            signal,
            filtered_correlator,
            record.previous_prompt,
            integration_time;
            four_quadrant = !iszero(record.polarity),
        ) : 0.0Hz
    (;
        integration_time,
        carrier_bandwidth = _carrier_bandwidth(
            state.bandwidths,
            state.staging.stage,
            integration_time,
        ),
        fll_assist_bandwidth = _fll_assist_bandwidth(state.bandwidths, integration_time),
        phase_error = pll_disc(signal, filtered_correlator; record.polarity),
        frequency_error,
        raw_frequency_error = frequency_error,
        code_error = dll_disc(
            signal,
            filtered_correlator,
            applied_code * Hz,
            record.sampling_frequency,
        ),
        center = NaN,
    )
end

# The carrier filter's input: the FLL-assisted filter takes both discriminators,
# any other the phase discriminator alone.
@inline _carrier_filter_input(
    ::ThirdOrderAssistedBilinearLF,
    phase_error,
    frequency_error,
) = (phase_error, frequency_error)
@inline _carrier_filter_input(::AbstractLoopFilter, phase_error, frequency_error) =
    phase_error

@inline _carrier_filter_bandwidth(::ThirdOrderAssistedBilinearLF, discriminators) =
    (discriminators.carrier_bandwidth, discriminators.fll_assist_bandwidth)
@inline _carrier_filter_bandwidth(::AbstractLoopFilter, discriminators) =
    discriminators.carrier_bandwidth

# The state after one record; `staging` is from `_staged_carrier_loop`.
@inline _stepped_state(
    state::SatConventionalPLLAndDLL,
    carrier_loop_filter,
    code_loop_filter,
    center,
    staging::CarrierLoopStaging,
) = SatConventionalPLLAndDLL(
    state;
    carrier_loop_filter,
    code_loop_filter,
    staging,
    signal_combining_sums = SignalCombiningSums(),
)

# Steps the staging (see `CarrierLoopStage`): advances the phase-lock indicator
# and moves the stage on for the next record. Returns the FLL input to feed (zero
# once the FLL is dropped) and the stepped staging.
@inline function _staged_carrier_loop(
    staging::CarrierLoopStaging,
    carrier_loop_filter,
    record::LoopRecord,
    discriminators,
)
    integration_time = discriminators.integration_time
    phase_lock = _update_phase_lock(
        staging.phase_lock,
        get_prompt(record.filtered_correlator),
        integration_time,
        record.signal,
    )
    stage = staging.stage
    frequency_error =
        _fll_in_use(carrier_loop_filter, staging) ? discriminators.frequency_error :
        zero(discriminators.frequency_error)
    if stage == FLL_ASSISTED_PLL && (
        !_uses_fll(carrier_loop_filter) ||
        _held(phase_lock.hold, _FLL_DROP_TIME_CONSTANTS, integration_time)
    )
        stage = WIDE_PLL
        phase_lock = _restart_hold(phase_lock)
    elseif stage == WIDE_PLL &&
           _held(phase_lock.hold, _NARROWING_TIME_CONSTANTS, integration_time)
        stage = NARROW_PLL
    end
    frequency_error, CarrierLoopStaging(stage, phase_lock)
end

# The driver's discriminators combined with pending passengers' ones, if any.
@inline _with_passengers(state, record::LoopRecord, discriminators, loops) = discriminators
@inline _with_passengers(
    state::SatConventionalPLLAndDLL,
    record::LoopRecord,
    discriminators,
    loops,
) = _combine_discriminators(discriminators, state.signal_combining_sums, record, loops)

@inline _uses_fll(::ThirdOrderAssistedBilinearLF) = true
@inline _uses_fll(::AbstractLoopFilter) = false

@inline _fll_in_use(carrier_loop_filter, staging::CarrierLoopStaging) =
    _uses_fll(carrier_loop_filter) && staging.stage == FLL_ASSISTED_PLL
@inline _fll_in_use(state) = _fll_in_use(state.carrier_loop_filter, state.staging)

# One record through a scalar loop (see `step_loop`).
@inline function _step_scalar_loop(
    estimator,
    state,
    record::LoopRecord,
    words,
    landing_sample::Int64,
)
    discriminators = _with_passengers(
        state,
        record,
        _record_discriminators(
            estimator,
            state,
            record,
            words,
            landing_sample,
            _fll_in_use(state),
        ),
        _ALL_LOOPS,
    )
    integration_time = discriminators.integration_time
    code_bandwidth =
        effective_code_loop_filter_bandwidth(state.bandwidths.code, integration_time)
    frequency_error, staging = _staged_carrier_loop(
        state.staging,
        state.carrier_loop_filter,
        record,
        discriminators,
    )
    carrier_freq_update, carrier_loop_filter = filter_loop(
        state.carrier_loop_filter,
        _carrier_filter_input(
            state.carrier_loop_filter,
            discriminators.phase_error,
            frequency_error,
        ),
        integration_time,
        _carrier_filter_bandwidth(state.carrier_loop_filter, discriminators),
    )
    code_freq_update, code_loop_filter = filter_loop(
        state.code_loop_filter,
        discriminators.code_error,
        integration_time,
        code_bandwidth,
    )
    carrier_doppler, code_doppler = aid_dopplers(
        record.signal,
        state.init_carrier_doppler,
        state.init_code_doppler,
        carrier_freq_update,
        code_freq_update,
    )
    _stepped_state(
        state,
        carrier_loop_filter,
        code_loop_filter,
        discriminators.center,
        staging,
    ),
    carrier_doppler,
    code_doppler
end

# Whether `estimator` combines passenger discriminators into the driver's loops: a
# `ConventionalPLLAndDLL` or `VectorPLLAndDLL` built with `combine_signals = true`.
# Internal: a host asks `takes_passenger_records`, which also covers a vector loop that
# takes passenger records to decode a pilot driver's data without combining.
combines_signals(::AbstractDopplerEstimator) = false
combines_signals(estimator::ConventionalPLLAndDLL) = estimator.combine_signals

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

@inline function fold_passenger_record(
    estimator::ConventionalPLLAndDLL,
    state::SatConventionalPLLAndDLL,
    record::LoopRecord,
    words;
    driver_signal::AbstractGNSSSignal,
    differential_group_delay_chips::Real = NaN,
)
    combines_signals(estimator) || return state
    _with_passenger_record(
        state,
        record,
        words,
        _scalar_loops_to_combine(state),
        driver_signal,
        differential_group_delay_chips,
    )
end

@inline _scalar_loops_to_combine(state::SatConventionalPLLAndDLL) =
    (pll = true, fll = _fll_in_use(state), dll = true)

@inline _with_passenger_record(
    state::SatConventionalPLLAndDLL,
    record::LoopRecord,
    words,
    loops,
    driver_signal::AbstractGNSSSignal,
    differential_group_delay_chips::Real,
) = SatConventionalPLLAndDLL(
    state;
    signal_combining_sums = _add_passenger_discriminators(
        state.signal_combining_sums,
        record,
        words,
        loops,
        driver_signal,
        differential_group_delay_chips,
    ),
)

# ── The NCO-referenced (delay-aware) PLL/DLL ─────────────────────────────────

"""
    NCOReferencedPLLAndDLL(; wide_carrier_loop_filter_bandwidth = nothing,
                             code_loop_filter_bandwidth = nothing,
                             narrow_carrier_loop_filter_bandwidth = nothing,
                             fll_assist_loop_filter_bandwidth = nothing,
                             predict_landing = true)

FLL-assisted PLL and DLL Doppler estimator for a replica whose NCO words are
applied with a known delay; the hardware-correlator receiver's default.

It is the [`ConventionalAssistedPLLAndDLL`](@ref) (same filters, gains and
[`CarrierLoopStage`](@ref) staging) with its internals referenced to the device
NCO. The one different default is the wide carrier bandwidth: the narrow one
([`default_narrow_carrier_loop_filter_bandwidth`](@ref), 18 Hz) instead of 50 Hz,
since delay tolerance shrinks with bandwidth (seven records of delay at 50 Hz,
thirteen at 18 Hz); a method of [`default_wide_carrier_loop_filter_bandwidth`](@ref)
does not reach it. It still narrows once phase lock has held, to the same 18 Hz
but under the tighter narrow cap (0.04/T against 0.09/T): from 18 to 10 Hz on
4 ms records, from 9 to 4 Hz on 10 ms ones.

 1. **Every record is attributed to the word that ran under it.** Both
    discriminators are measured against the applied replica, so `applied word +
    FLL discriminator` is an *absolute* Doppler measurement; the DLL is
    normalised with the applied code word. The conventional loop does this too.
 2. **The correction is sized for the moment it lands.** The filter is fed the
    discriminators *predicted at the new command's landing sample*: the phase
    error (cycles) advanced by `∫ (f̂ − w(τ)) dτ` over the words already
    scheduled, and the frequency measurement re-based onto the word running
    there.

With zero delay step 2 is the identity and the estimator *is* the conventional
loop at the same bandwidths (the defaults differ in the wide one), inheriting its
noise performance. `predict_landing = false` drops
step 2, giving the conventional loop: the **negative control**, which fails like
it at a few epochs of delay.
"""
struct NCOReferencedPLLAndDLL{CO<:AbstractLoopFilter} <: AbstractDopplerEstimator
    wide_carrier_loop_filter_bandwidth::Maybe{typeof(1.0Hz)}
    code_loop_filter_bandwidth::Maybe{typeof(1.0Hz)}
    narrow_carrier_loop_filter_bandwidth::Maybe{typeof(1.0Hz)}
    fll_assist_loop_filter_bandwidth::Maybe{typeof(1.0Hz)}
    predict_landing::Bool
end

function NCOReferencedPLLAndDLL(
    ::Type{CO} = SecondOrderBilinearLF;
    wide_carrier_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    code_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    narrow_carrier_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    fll_assist_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    predict_landing::Bool = true,
) where {CO<:AbstractLoopFilter}
    NCOReferencedPLLAndDLL{CO}(
        wide_carrier_loop_filter_bandwidth,
        code_loop_filter_bandwidth,
        narrow_carrier_loop_filter_bandwidth,
        fll_assist_loop_filter_bandwidth,
        predict_landing,
    )
end

"""
    SatNCOReferencedPLLAndDLL

Per-satellite state of an [`NCOReferencedPLLAndDLL`](@ref): handover Dopplers
(the filters output offsets from them), both filters, their
[`LoopBandwidths`](@ref TrackingLoops.LoopBandwidths), the centre sample of the
last record (`NaN` before the first; the FLL's replica word is averaged from there),
and the [`CarrierLoopStaging`](@ref TrackingLoops.CarrierLoopStaging).
"""
struct SatNCOReferencedPLLAndDLL{CA<:ThirdOrderAssistedBilinearLF,CO<:AbstractLoopFilter}
    init_carrier_doppler::typeof(1.0Hz)
    init_code_doppler::typeof(1.0Hz)
    carrier_loop_filter::CA
    code_loop_filter::CO
    bandwidths::LoopBandwidths
    previous_record_center::Float64
    staging::CarrierLoopStaging
end

function SatNCOReferencedPLLAndDLL(
    state::SatNCOReferencedPLLAndDLL{CA,CO};
    carrier_loop_filter::Maybe{CA} = nothing,
    code_loop_filter::Maybe{CO} = nothing,
    previous_record_center::Maybe{Float64} = nothing,
    staging::Maybe{CarrierLoopStaging} = nothing,
) where {CA,CO}
    SatNCOReferencedPLLAndDLL{CA,CO}(
        state.init_carrier_doppler,
        state.init_code_doppler,
        something(carrier_loop_filter, state.carrier_loop_filter),
        something(code_loop_filter, state.code_loop_filter),
        state.bandwidths,
        something(previous_record_center, state.previous_record_center),
        something(staging, state.staging),
    )
end

function init_estimator_state(
    estimator::NCOReferencedPLLAndDLL{CO},
    driver_signal::AbstractGNSSSignal,
    carrier_doppler,
    code_doppler,
) where {CO}
    SatNCOReferencedPLLAndDLL(
        carrier_doppler,
        code_doppler,
        ThirdOrderAssistedBilinearLF(),
        _constructorof(CO)(),
        # The narrow default for the wide bandwidth too: see `NCOReferencedPLLAndDLL`.
        _resolve_bandwidths(
            estimator,
            driver_signal;
            wide_default = default_narrow_carrier_loop_filter_bandwidth(driver_signal),
        ),
        NaN,
        CarrierLoopStaging(ThirdOrderAssistedBilinearLF()),
    )
end

function reset_estimator_state(
    ::NCOReferencedPLLAndDLL,
    state::SatNCOReferencedPLLAndDLL,
    carrier_doppler,
    code_doppler,
)
    SatNCOReferencedPLLAndDLL(
        carrier_doppler,
        code_doppler,
        _constructorof(typeof(state.carrier_loop_filter))(),
        _constructorof(typeof(state.code_loop_filter))(),
        state.bandwidths,
        NaN,
        CarrierLoopStaging(ThirdOrderAssistedBilinearLF()),
    )
end

"""
    wrap_half_cycle(phase)

Fold a carrier-phase error in cycles into `[−1/4, 1/4]`, the range of the Costas
[`pll_disc`](@ref) (a bit flip turns the prompt by half a cycle). Exact inside it.
"""
wrap_half_cycle(phase) = rem(phase, 0.5, RoundNearest)

# The phase error (cycles) the record would show `shift` samples later under the
# scheduled words, integrated from the record's centre with the filter's Doppler
# estimate before this record's update. Folded to ±1/4 cycle for the Costas PLL
# (`polarity = 0`), ±1/2 for the four-quadrant one.
@inline function _predict_landing_phase_error(
    phase_error,
    state::SatNCOReferencedPLLAndDLL,
    words,
    center,
    shift,
    integration_time,
    sampling_frequency,
    polarity = 0,
)
    sampling_freq_hz = Float64(ustrip(Hz, uconvert(Hz, sampling_frequency)))
    carrier_loop_filter = state.carrier_loop_filter
    f_hat = ustrip(
        Hz,
        uconvert(
            Hz,
            state.init_carrier_doppler +
            carrier_loop_filter.x1 +
            integration_time / 2 * carrier_loop_filter.x2,
        ),
    )
    ramp_word = first(mean_nco_word(words, center, center + shift))
    predicted = phase_error + shift * (f_hat - ramp_word) / sampling_freq_hz
    iszero(polarity) ? wrap_half_cycle(predicted) : rem(predicted, 1.0, RoundNearest)
end

"""
    step_loop(estimator::NCOReferencedPLLAndDLL, state, record::LoopRecord, words,
              landing_sample)
        -> (state, carrier_doppler, code_doppler)

One record through the NCO-referenced loop. `words` are the replica words the
record ran on; `landing_sample` is where this fold's command lands
(`NO_LANDING_SAMPLE`: at the record's end). See [`NCOReferencedPLLAndDLL`](@ref).
"""
@inline step_loop(
    estimator::NCOReferencedPLLAndDLL,
    state::SatNCOReferencedPLLAndDLL,
    record::LoopRecord,
    words,
    landing_sample::Int64,
) = _step_scalar_loop(estimator, state, record, words, landing_sample)

@inline function _record_discriminators(
    estimator::NCOReferencedPLLAndDLL,
    state::SatNCOReferencedPLLAndDLL,
    record::LoopRecord,
    words,
    landing_sample::Int64,
    fll::Bool,
)
    signal = record.signal
    sampling_frequency = record.sampling_frequency
    previous_center = state.previous_record_center
    # Every record of the fold maps onto the delay-free loop's record this far ahead.
    shift = landing_sample == NO_LANDING_SAMPLE ? 0 : landing_sample - record.fold_end
    record_end = record.sample_index
    record_samples = record.integrated_samples
    record_start = record_end - record_samples
    center = record_end - record_samples / 2
    integration_time = record_samples / sampling_frequency
    filtered_correlator = record.filtered_correlator

    applied_carrier, applied_code = mean_nco_word(words, record_start, record_end)
    phase_error = pll_disc(signal, filtered_correlator; record.polarity)
    raw_frequency_error =
        fll ?
        fll_disc(
            signal,
            filtered_correlator,
            record.previous_prompt,
            integration_time;
            four_quadrant = !iszero(record.polarity),
        ) : 0.0Hz
    frequency_error = raw_frequency_error

    if estimator.predict_landing && shift > 0
        phase_error = _predict_landing_phase_error(
            phase_error,
            state,
            words,
            center,
            shift,
            integration_time,
            sampling_frequency,
            record.polarity,
        )
        # Re-base the FLL reading from the word between the prompts' centres onto
        # the word running `shift` samples ahead.
        if fll
            fll_word =
                isnan(previous_center) ? applied_carrier :
                first(mean_nco_word(words, previous_center, center))
            landing_word =
                first(mean_nco_word(words, record_start + shift, record_end + shift))
            frequency_error += (fll_word - landing_word) * Hz
        end
    end

    (;
        integration_time,
        carrier_bandwidth = _carrier_bandwidth(
            state.bandwidths,
            state.staging.stage,
            integration_time,
        ),
        fll_assist_bandwidth = _fll_assist_bandwidth(state.bandwidths, integration_time),
        phase_error,
        frequency_error,
        raw_frequency_error,
        code_error = dll_disc(
            signal,
            filtered_correlator,
            applied_code * Hz,
            sampling_frequency,
        ),
        center,
    )
end

@inline _stepped_state(
    state::SatNCOReferencedPLLAndDLL,
    carrier_loop_filter,
    code_loop_filter,
    center,
    staging::CarrierLoopStaging,
) = SatNCOReferencedPLLAndDLL(
    state;
    carrier_loop_filter,
    code_loop_filter,
    previous_record_center = center,
    staging,
)

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
