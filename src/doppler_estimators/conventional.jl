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

# The driver's discriminators combined with pending passengers' ones, if any.
@inline _with_passengers(state, record::LoopRecord, discriminators, loops) = discriminators
@inline _with_passengers(
    state::SatConventionalPLLAndDLL,
    record::LoopRecord,
    discriminators,
    loops,
) = _combine_discriminators(discriminators, state.signal_combining_sums, record, loops)

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

combines_signals(estimator::ConventionalPLLAndDLL) = estimator.combine_signals

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
