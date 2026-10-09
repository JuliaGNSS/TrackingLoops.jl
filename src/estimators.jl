# ─────────────────────────────────────────────────────────────────────────────
# The Doppler estimators: their configuration, their per-satellite state, and
# the one per-record `step_loop` that both the software receiver (through Tracking)
# and a hardware correlator's loop process call.
#
# The conventional FLL-assisted PLL/DLL and the delay-aware NCO-referenced loop
# share one signature:
#
#     step_loop(estimator, state, record, words, landing_sample) -> (state, carrier_doppler, code_doppler)
#
# `words` answers `mean_nco_word(words, a, b)` for the replica word over a span
# — a `FixedNCOWord` for a software correlator, the channel's `NCOTimeline` for
# hardware — and `landing_sample` is the device sample the command computed
# from this fold takes effect at, `NO_LANDING_SAMPLE` for "at each record's
# end". With a fixed word and no landing sample the NCO-referenced estimator
# *is* the conventional loop to the bit.
# ─────────────────────────────────────────────────────────────────────────────

"""
$(SIGNATURES)

One completed record as the loop-filter step sees it: the signal it belongs
to, the filtered (antenna-combined, normalised) correlator, the previous
record's filtered prompt the FLL chains from, the record's span and the blocks
it covered, the band's sampling frequency, and `fold_end` — the end sample of
the last record of the fold this record belongs to, which is what a landing
sample is measured against (every record of a fold maps onto the delay-free
loop's record the same distance ahead).

Two fields identify the record to an estimator that keeps per-satellite state
of its own, as [`VectorPLLAndDLL`](@ref) does:

  - `prn`: the satellite (`0` when the host does not say);
  - `code_phase`: the replica's code phase (chips) at `sample_index`, from the
    [`CorrelatorOutput`](@ref) (`NaN` when the producer does not report it);
  - `cn0`: the host's C/N₀ estimate of the record's signal in dB-Hz (`NaN`, the
    default, when it gives none), which the vector loop weights the navigation
    filter's measurements and decides lock by; without it the vector loop
    estimates the C/N₀ from the prompts itself. The scalar loops do not read it.

A satellite's driver and passenger records share one sample frame, which signal
combining compares their ends on. For such an estimator `sample_index /
sampling_frequency` must also be the time since one origin shared by every
satellite of the band. A host whose
correlator restarts its sample count passes that origin's offset as
`sample_offset` to the constructor that takes a `CorrelatorOutput`; it is added
to `sample_index` and `fold_end`.

`polarity` picks the carrier discriminators, from the signal's bit buffer as it was
when the record was correlated (before the fold that may sync it): the prompt's
sign from the secondary-code sync ([`get_sync_polarity`](@ref)). Nonzero, the
replica wipes every sign modulation off the prompt, so the PLL and the FLL are
four-quadrant; `0` (the default) keeps both two-quadrant (the Costas PLL).

`previous_prompt` is the previous record's filtered prompt, or zero where the
FLL has nothing to compare with: the first record, and a record whose length or
whose `polarity` differs from the previous record's. The FLL divides the
rotation between the two prompts by this record's integration time, which is
the time between them only for records of one length, and a sign flip between a
prompt with and one without the wipe-off would read as half a cycle. A record
with a zero previous prompt gives no FLL reading. The constructor that takes a
[`SignalLoopState`](@ref) fills in `previous_prompt` and `polarity` by these
rules.
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

# A record's C/N₀ in dB-Hz, from a number in dB-Hz or a dB-Hz quantity, such as
# `estimate_cn0` returns.
_record_cn0(cn0::Real) = Float64(cn0)
_record_cn0(cn0) = Float64(ustrip(cn0))

"""
    LoopRecord(loop::SignalLoopState, signal, filtered_correlator,
               output::CorrelatorOutput, integrated_code_blocks, sampling_frequency;
               prn, fold_end = output.sample_index, sample_offset = 0, cn0 = NaN,
               correlated_pre_sync = false)

The [`LoopRecord`](@ref) of a record `apply_record` folded, built from the
signal's state `loop` *before* that fold, with the fields the record contract
asks of a host filled in:

  - `polarity` ([`get_sync_polarity`](@ref)) from the bit buffer as the record
    was correlated;
  - `previous_prompt`: the last filtered prompt, or zero where the FLL must not
    compare with it — the first record, and a record whose block count
    (`integrated_code_blocks`, as `apply_record` returned it) or polarity differs
    from the previous record's.

The arguments are those of the constructor that takes a `CorrelatorOutput`,
with `loop` in place of the previous prompt: `filtered_correlator` and
`integrated_code_blocks` are what `apply_record` returned for the record, and
`fold_end`, `sample_offset` and `cn0` are as there. `prn` is required here, as
the polarity of a secondary code can depend on it.

Pass the same `correlated_pre_sync` as to `apply_record`: a record that follows a
sync found earlier in the same fold was correlated with the pre-sync replica, so its
prompt still carries the secondary code. It gets no polarity: read with the sync's,
the four-quadrant PLL would take every secondary chip flip for a half-cycle phase
error.
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
        integrated_code_blocks == loop.last_num_code_blocks && polarity == loop.last_polarity
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
Per-satellite state for the conventional PLL and DLL Doppler estimator.
Holds initial Doppler values, loop filter states, their
[`LoopBandwidths`](@ref TrackingLoops.LoopBandwidths), the carrier loop's
[`CarrierLoopStaging`](@ref TrackingLoops.CarrierLoopStaging) (its
[`CarrierLoopStage`](@ref) and phase-lock indicator) and the passengers' pending
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

Conventional Phase-Locked Loop (PLL) and Delay-Locked Loop (DLL) Doppler
estimator. Configuration-only — per-satellite state is a
[`SatConventionalPLLAndDLL`](@ref), produced via [`init_estimator_state`](@ref).

Type parameters `CA` and `CO` select the carrier and code loop filter types;
the bandwidth fields configure the loop bandwidths used when seeding new
satellites: the wide carrier bandwidth, the code bandwidth, the tracking
carrier bandwidth the loop narrows to once phase lock has held and the FLL path's
bandwidth (see [`CarrierLoopStage`](@ref)). Each bandwidth field is
`Maybe{typeof(1.0Hz)}`: a `nothing` field (the default) means **auto** — the
bandwidth is sized per satellite from its estimator-driver signal via
[`default_wide_carrier_loop_filter_bandwidth`](@ref),
[`default_code_loop_filter_bandwidth`](@ref),
[`default_narrow_carrier_loop_filter_bandwidth`](@ref) and
[`default_fll_assist_loop_filter_bandwidth`](@ref). At filter time each is
capped against the record's integration time (see [Loop-filter bandwidths](@ref);
the code bandwidth by [`effective_code_loop_filter_bandwidth`](@ref)), so a
longer coherent integration needs no re-tuning.

`combine_signals = true` combines the discriminators of a satellite's other
signals, the passengers, into the loops of the signal whose records
[`step_loop`](@ref) closes them on, the driver; the host folds each passenger
record with [`fold_passenger_record`](@ref). See
[Signal combining](@ref). Each passenger is assumed to integrate no longer than
the driver. A longer passenger record is combined only into the driver record it
ends in, with a weight proportional to its integration time (its cube for the
FLL), so it dominates that one loop update with a reading averaged over its own,
longer record; its FLL reading also has the narrower range ±1/(4·T_passenger).
Make the longest-integrating signal (typically the pilot) the driver.
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

Create a ConventionalPLLAndDLL with FLL-assisted carrier tracking: a
`ThirdOrderAssistedBilinearLF` carrier loop filter combining the PLL and FLL
discriminators, with the FLL path at its own bandwidth until the carrier
Doppler has converged (see [`CarrierLoopStage`](@ref)). Bandwidths default to
`nothing` (auto) and signal combining to off, see
[`ConventionalPLLAndDLL`](@ref).
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

# Kwarg-update constructor for tweaking the configuration in place.
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
        pll_and_dll.wide_carrier_loop_filter_bandwidth :
        wide_carrier_loop_filter_bandwidth,
        isnothing(code_loop_filter_bandwidth) ?
        pll_and_dll.code_loop_filter_bandwidth :
        code_loop_filter_bandwidth,
        isnothing(narrow_carrier_loop_filter_bandwidth) ?
        pll_and_dll.narrow_carrier_loop_filter_bandwidth :
        narrow_carrier_loop_filter_bandwidth,
        isnothing(fll_assist_loop_filter_bandwidth) ?
        pll_and_dll.fll_assist_loop_filter_bandwidth :
        fll_assist_loop_filter_bandwidth,
        isnothing(combine_signals) ? pll_and_dll.combine_signals : combine_signals,
    )
end

"""
    init_estimator_state(estimator, driver_signal, carrier_doppler, code_doppler)

Build the per-satellite estimator state for a satellite whose loop is driven
by `driver_signal` and starts at the given Dopplers. Auto bandwidths (`nothing`
on the estimator) are resolved here from the driver signal.

This function must be **pure**: Tracking.jl also calls it to build template
states and to re-seed satellites.
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

# `Accessors.constructorof` without the dependency: the loop filters are plain
# parametric structs whose zero-argument constructor is their type name.
_constructorof(::Type{T}) where {T} = Base.typename(T).wrapper

"""
    reset_estimator_state(estimator, state, carrier_doppler, code_doppler)

Zero the loop-filter integrators, restart the carrier loop's staging (see
[`CarrierLoopStage`](@ref)) with a fresh phase-lock indicator, drop
pending passenger discriminators and re-seed the state from the converged
Dopplers, keeping the per-satellite bandwidths.
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
    step_loop(estimator::ConventionalPLLAndDLL, state, record::LoopRecord, words, landing_sample)
        -> (state, carrier_doppler, code_doppler)

One record through the conventional loop: PLL (and FLL, for the assisted
filter, until the carrier Doppler has converged) discriminators against the
filtered prompt, the DLL normalised with the code word the record ran on, the
bandwidths of the record's [`CarrierLoopStage`](@ref) capped by their stability
products against the record's integration time, the phase-lock indicator
advanced and the Dopplers aided. `landing_sample` is ignored: the conventional loop assumes
its command acts before the next record.
"""
@inline step_loop(
    estimator::ConventionalPLLAndDLL,
    state::SatConventionalPLLAndDLL,
    record::LoopRecord,
    words,
    landing_sample::Int64,
) = _step_scalar_loop(estimator, state, record, words, landing_sample)

# The discriminators of one record, as the loop filters are fed them, and what
# the step needs around them: the per-record integration time, the capped
# carrier bandwidth, and the record's centre sample. Shared by the scalar step
# and the vector step (`_step_vector_loop`), so the two cannot drift apart.
# `phase_error` (cycles) and `frequency_error` (Hz) are what the scalar loop's
# carrier filter is fed; `raw_frequency_error` is the FLL
# discriminator as measured, against the replica that ran between the two
# prompts' centres, before any re-basing onto a landing word. The DLL output
# `code_error` is normalised with the code word the record ran on. The FLL
# discriminator is read only where it is used (`fll`): both frequency errors are
# zero otherwise.
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
@inline _carrier_filter_input(::ThirdOrderAssistedBilinearLF, phase_error, frequency_error) =
    (phase_error, frequency_error)
@inline _carrier_filter_input(::AbstractLoopFilter, phase_error, frequency_error) =
    phase_error

# The carrier filter's bandwidth: the FLL-assisted filter takes the PLL and the
# FLL path's bandwidth as a pair, any other the PLL's alone.
@inline _carrier_filter_bandwidth(::ThirdOrderAssistedBilinearLF, discriminators) =
    (discriminators.carrier_bandwidth, discriminators.fll_assist_bandwidth)
@inline _carrier_filter_bandwidth(::AbstractLoopFilter, discriminators) =
    discriminators.carrier_bandwidth

# The state after one record, with both loop filters and the carrier loop's
# `staging` stepped.
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

# The carrier loop's staging (see `CarrierLoopStage`). The FLL-assisted filter is
# fed the FLL reading while FLL-assisted and zero after, which is exactly the
# third-order PLL (same state, same coefficients), so dropping the FLL is free
# (Kaplan & Hegarty §5.5; Ward, ION GPS 1998). The record's driver prompt advances
# the phase-lock indicator, and how long it has read lock moves the stage on for
# the next record: phase lock ends the FLL-assisted stage, as the carrier
# Doppler has converged once the phase is locked, and phase lock held on, counted
# afresh from there, narrows the loop. Returns the FLL input to feed and the
# stepped staging.
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

# The driver's discriminators with the passengers' pending ones, for a state that
# holds them; the step's `_stepped_state` starts the sums afresh.
@inline _with_passengers(state, record::LoopRecord, discriminators, loops) = discriminators
@inline _with_passengers(
    state::SatConventionalPLLAndDLL,
    record::LoopRecord,
    discriminators,
    loops,
) = _combine_discriminators(discriminators, state.signal_combining_sums, record, loops)

@inline _uses_fll(::ThirdOrderAssistedBilinearLF) = true
@inline _uses_fll(::AbstractLoopFilter) = false

# Whether the scalar loop reads the FLL discriminator: an FLL-assisted filter in
# its FLL-assisted stage.
@inline _fll_in_use(carrier_loop_filter, staging::CarrierLoopStaging) =
    _uses_fll(carrier_loop_filter) && staging.stage == FLL_ASSISTED_PLL
@inline _fll_in_use(state) = _fll_in_use(state.carrier_loop_filter, state.staging)

# One record through a scalar loop: the carrier filter fed its discriminators
# (the FLL's while FLL-assisted) at its stage's bandwidths, the code filter the
# DLL with its bandwidth capped by its stability product against the record's
# integration time, the staging stepped and the Dopplers aided.
@inline function _step_scalar_loop(estimator, state, record::LoopRecord, words, landing_sample::Int64)
    discriminators = _with_passengers(
        state,
        record,
        _record_discriminators(estimator, state, record, words, landing_sample, _fll_in_use(state)),
        _ALL_LOOPS,
    )
    integration_time = discriminators.integration_time
    code_bandwidth =
        effective_code_loop_filter_bandwidth(state.bandwidths.code, integration_time)
    frequency_error, staging =
        _staged_carrier_loop(state.staging, state.carrier_loop_filter, record, discriminators)
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
[`fold_passenger_record`](@ref): where it combines signals (built with
`combine_signals = true`), and for a [`VectorPLLAndDLL`](@ref) with a dataless
driver, which decodes the navigation data from its data passenger whether it
combines or not. Where it is `false`,
`fold_passenger_record` returns the state unchanged, so a host may skip the
passenger walk.
"""
takes_passenger_records(estimator::AbstractDopplerEstimator) = combines_signals(estimator)

"""
    fold_passenger_record(estimator, state, record::LoopRecord, words;
                          driver_signal, differential_group_delay_chips = NaN)
        -> state

Fold one completed passenger record into the per-satellite `state`. In a
scalar loop that combines signals its discriminators, weighted by its signal's
ICD power share and integration time, join the sums the driver's next
[`step_loop`](@ref) closes its loops on; for what a [`VectorPLLAndDLL`](@ref)
does with it, see there. Call it for every passenger record in sample order,
each before the driver record it ends within (or ends at); records left pending
after the driver's last carry over to its next, unless they ended before it
starts (see [`SignalCombiningSums`](@ref)). The passenger's records share the
driver's sample frame (`sample_index`). `words` are the replica words the
satellite ran on.

  - `record` is the passenger's own record, its `previous_prompt` following
    [`LoopRecord`](@ref)'s contract for the passenger's own record sequence.
    The scalar loops read its two-quadrant discriminators whatever its
    `polarity`, and combine a four-quadrant driver reading
    with them only within the two-quadrant range.
  - `driver_signal` rotates the passenger's prompt onto the driver's carrier
    phase frame by the nominal carrier phase offsets.
  - `differential_group_delay_chips`, the passenger's group delay minus the
    driver's in chips, refers its DLL discriminator to the driver's code phase;
    `NaN` (unknown) leaves the passenger out of the code loop.

The FLL is combined only while it is formed, in the FLL-assisted stage. An
estimator that takes no passenger records ([`takes_passenger_records`](@ref))
returns `state` unchanged. See [Signal combining](@ref) and
[Host contract](@ref).
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

# The scalar loops passengers are combined into: all, but the FLL only while it
# is formed.
@inline _scalar_loops_to_combine(state::SatConventionalPLLAndDLL) = (
    pll = true,
    fll = _fll_in_use(state),
    dll = true,
)

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
applied with a known delay — the hardware-correlator receiver's default.

It is the [`ConventionalAssistedPLLAndDLL`](@ref) — the same third-order
assisted bilinear carrier filter, the same second-order code filter, the same
gains and the same staging ([`CarrierLoopStage`](@ref)) — with its loop
internals referenced to the device NCO instead of to the word the filter last
computed. Its one different default is the wide carrier bandwidth: the
narrow bandwidth ([`default_narrow_carrier_loop_filter_bandwidth`](@ref),
18 Hz) instead of the conventional loop's 50 Hz
([`default_wide_carrier_loop_filter_bandwidth`](@ref), which it does not read, so
a method of it for a signal type does not widen this loop), because a loop's tolerance of
command delay shrinks as its bandwidth grows (at 50 Hz it holds through seven
records of delay, at 18 Hz through thirteen). Its staging still drops
the FLL at phase lock and narrows once lock has held: to the same 18 Hz,
but under the tighter narrow cap (0.04/T against 0.09/T), so from 18 to
10 Hz on 4 ms records and from 9 to 4 Hz on 10 ms ones.

 1. **Every record is attributed to the word that ran under it.** The phase
    discriminator is measured against the applied replica by construction; the
    frequency discriminator is re-based onto it too, so `applied word + FLL
    discriminator` is an *absolute* measurement of the signal's Doppler. The
    DLL is normalised with the applied code word.
 2. **The correction is sized for the moment it lands.** The filter is stepped
    with the discriminators *predicted at the landing sample of the new
    command*: the measured phase error (cycles) advanced by `∫ (f̂ − w(τ)) dτ`
    over the words already scheduled at the NCO, and the frequency measurement
    taken relative to the word that will be running there.

With zero delay both steps are the identity and the estimator *is* the
conventional loop at the same bandwidths (the defaults differ in the wide
one), so the software receiver's noise performance is inherited rather than
re-tuned.

Step 1 is not specific to this estimator: the conventional
[`step_loop`](@ref) reads the applied code word from `words` too, and both
discriminators are measured against the replica that ran, so neither loop
needs a correction for it. What sets this estimator apart is step 2, including
the re-basing of the frequency measurement onto the word that will be running
at landing. `predict_landing = false` drops step 2 and is then arithmetically
the [`ConventionalAssistedPLLAndDLL`](@ref): the documented **negative
control**, which fails exactly like the conventional loop at a few epochs of
delay.
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

Per-satellite state of an [`NCOReferencedPLLAndDLL`](@ref): the handover
Dopplers the loop filters' outputs are offsets from, both filters, their
[`LoopBandwidths`](@ref TrackingLoops.LoopBandwidths), the centre sample of the
last record folded (the FLL measures the mean frequency offset between two prompts'
centres, so that is the span its replica word is averaged over), and the carrier
loop's [`CarrierLoopStaging`](@ref TrackingLoops.CarrierLoopStaging).
"""
struct SatNCOReferencedPLLAndDLL{CA<:ThirdOrderAssistedBilinearLF,CO<:AbstractLoopFilter}
    init_carrier_doppler::typeof(1.0Hz)
    init_code_doppler::typeof(1.0Hz)
    carrier_loop_filter::CA
    code_loop_filter::CO
    bandwidths::LoopBandwidths
    # Device sample at the centre of the last record folded; `NaN` before the
    # first.
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

Fold a carrier-phase error in cycles into `[−1/4, 1/4]`, the range a BPSK
prompt — and therefore the Costas [`pll_disc`](@ref) — can tell the phase in,
since a data bit flip turns the prompt by half a cycle. Exact for any phase
already inside it.
"""
wrap_half_cycle(phase) = rem(phase, 0.5, RoundNearest)

# The phase error a record would show `shift` samples later, under the words
# the NCO will run in between: the mean phase sits at the record's centre, so
# the ramp is integrated from there. The signal's Doppler is the filter's own
# estimate, before this record's innovation. In cycles, as `pll_disc` reads
# it, and folded into the range it reads: ±1/4 cycle for the Costas PLL
# (`polarity = 0`), ±1/2 cycle for the four-quadrant one.
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
    step_loop(estimator::NCOReferencedPLLAndDLL, state, record::LoopRecord, words, landing_sample)
        -> (state, carrier_doppler, code_doppler)

One record through the NCO-referenced loop. `words` gives the replica words
the record really ran on; `landing_sample` is where the command computed from
this record's fold lands (`NO_LANDING_SAMPLE` for a software correlator, where
it acts at the record's end). See [`NCOReferencedPLLAndDLL`](@ref).
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
    # The command this fold produces is computed after its last record and
    # lands at `landing_sample`; every record of the fold maps onto the
    # delay-free loop's record that far ahead of it.
    shift = landing_sample == NO_LANDING_SAMPLE ? 0 : landing_sample - record.fold_end
    record_end = record.sample_index
    record_samples = record.integrated_samples
    record_start = record_end - record_samples
    center = record_end - record_samples / 2
    # Per-record integration time — the block time, not the chunk time.
    integration_time = record_samples / sampling_frequency
    filtered_correlator = record.filtered_correlator

    # The words this record really ran on.
    applied_carrier, applied_code = mean_nco_word(words, record_start, record_end)
    # Discriminators against the applied replica. The phase error is that by
    # construction; the frequency error is the mean offset from the replica
    # between the two prompts' centres.
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
        # The absolute frequency measurement, relative to the word that will be
        # running under the record `shift` samples ahead. `fll_disc` measured
        # against the word that ran between the two prompts' centres.
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
        # Bandwidths exactly as the conventional loop.
        carrier_bandwidth = _carrier_bandwidth(
            state.bandwidths,
            state.staging.stage,
            integration_time,
        ),
        fll_assist_bandwidth = _fll_assist_bandwidth(state.bandwidths, integration_time),
        phase_error,
        frequency_error,
        raw_frequency_error,
        # The DLL normalises with the code word the replica actually ran on.
        code_error = dll_disc(signal, filtered_correlator, applied_code * Hz, sampling_frequency),
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
inner loop's for a [`SatVectorPLLAndDLL`](@ref)): `⟨I² − Q²⟩ / A²`, an estimate
of `cos 2φ` from exponential averages over 0.1 s (at least 25 records) of the
driver's prompt, normalised by the signal power `A²` estimated from the
prompt's moments: 1 in phase lock, 0 for a uniformly spinning phase, the same
at any C/N₀. `NaN` until the averages span that time. The carrier-loop staging
reads it against [`phase_lock_indicator_threshold`](@ref) (see
[`CarrierLoopStage`](@ref)); for a receiver's own lock decisions, smooth it over
the receiver's own horizon rather than act on single readings.
"""
phase_lock_indicator(state::Union{SatConventionalPLLAndDLL,SatNCOReferencedPLLAndDLL}) =
    phase_lock_indicator(state.staging)

"The estimator-state type a Doppler estimator produces (for slot typing)."
estimator_state_type(estimator::AbstractDopplerEstimator, driver_signal::AbstractGNSSSignal) =
    typeof(init_estimator_state(estimator, driver_signal, 0.0Hz, 0.0Hz))

# ── What an estimator knows of the navigation solution ───────────────────────

"""
    navigation_solution(estimator) -> Union{PVTSolution,Nothing}

The latest navigation solution an estimator computed, or `nothing` for an
estimator that computes none (the scalar loops). [`VectorPLLAndDLL`](@ref)
returns the scalar PVT's until its filter is seeded and the filter's after
that: position, velocity, time, clock bias and drift, the DOP, the satellites
that determined it with their residuals, and the inter-system and
inter-frequency biases. Its containers are reused by the next cycle, so copy
out what is needed later.
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

How many navigation cycles the estimator has run, or `nothing` for an estimator
without them. The count changes exactly when [`navigation_solution`](@ref) and
[`navigation_status`](@ref) do, so a consumer polling it after every step reads
each solution once.
"""
navigation_cycle(::AbstractDopplerEstimator) = nothing

"""
    navigation_epoch(estimator) -> Union{typeof(1.0s),Nothing}

The epoch the latest navigation solution refers to, on the records' time grid:
`sample_index / sampling_frequency` of the moment it describes. `nothing`
before the first cycle, and for an estimator without navigation cycles.
"""
navigation_epoch(::AbstractDopplerEstimator) = nothing

"""
    satellite_report(estimator, signal, prn) -> Union{SatelliteReport,Nothing}

What the estimator knows of satellite `prn` of `signal` (a
[`SatelliteReport`](@ref)), or `nothing` when it keeps no per-satellite
navigation state (the scalar loops) or has never seen the satellite.
"""
satellite_report(::AbstractDopplerEstimator, ::AbstractGNSSSignal, ::Integer) = nothing
