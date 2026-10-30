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

# The soft bits a record appended: a contiguous view of the bit buffer's vector.
const SoftBitsView = SubArray{Float32,1,Vector{Float32},Tuple{UnitRange{Int}},true}

# The soft bits of a record built without a signal state: none. Never written to.
const _NO_SOFT_BITS = Float32[]

@inline _no_soft_bits() = view(_NO_SOFT_BITS, 1:0)

"""
$(SIGNATURES)

One completed record as the loop-filter step sees it: the signal, the filtered
(antenna-combined, normalised) correlator, the previous filtered prompt the FLL
compares with, the record's span and block count, the band's sampling frequency,
and `fold_end_sample_index`, the `sample_index` of the fold's last record, against
which a landing sample is measured.

`integrated_code_blocks` must be at least 1: the carrier-loop staging sizes its
phase-lock indicator's averages in records of that length and divides by it. A
record built from a [`SignalLoopState`](@ref) always is ([`apply_record`](@ref)
counts at least one block); a host building records otherwise must ensure it.

The four-quadrant FLL reads a prompt sign flip as half a cycle of rotation, so a
record whose `carrier_phase_polarity` differs from the previous record's comes with
`previous_prompt = 0`, giving no FLL reading, as for the first record. The
constructor that takes a `SignalLoopState` applies this; a host building records
otherwise passes `0` itself.

Fields for an estimator with per-satellite state of its own
([`VectorPLLAndDLL`](@ref)):

  - `prn`: the satellite (`0` if unknown);
  - `code_phase`: the replica's code phase (chips) at `sample_index`, from the
    [`CorrelatorOutput`](@ref) (`NaN` if not reported).

For such an estimator `sample_index / sampling_frequency` must also be the time
since an origin shared by every satellite of the band; a host whose correlator
restarts its sample count passes the offset as `sample_offset` (added to
`sample_index` and `fold_end_sample_index`).

The rest summarises the signal's [`SignalLoopState`](@ref) after
[`apply_record`](@ref) folded this record, for an estimator that decodes or weights
by C/N₀ (the vector loop; the scalar loops ignore it):

  - `bit_synced`: whether the bit clock holds bit or secondary-code sync;
  - `sync_change`: whether this record found or lost it ([`SyncChange`](@ref));
  - `blocks_into_bit`: code blocks integrated into the current navigation bit
    (`0` before sync);
  - `new_soft_bits`: exactly the soft bits *this record* appended, a view of the
    bit buffer's vector valid for the call, so it does not matter when the host
    drains the buffer;
  - `cn0_estimator`: the signal's C/N₀ estimator, for [`estimate_cn0`](@ref);
  - `bit_polarity`: the sign the bit clock's sync found its soft bits at (`-1` if
    it negates the accumulated prompts' real part, `+1` if it keeps it, `0` before
    sync).

`carrier_phase_polarity` is the prompt's sign where the replica wiped every sign
modulation off it (a dataless signal synced to its secondary code), else `0`.
Nonzero, the PLL and the FLL read the record four-quadrant (`pll_disc(...;
polarity)`, `fll_disc(...; polarity)`); `0`, two-quadrant.

The constructor taking a `SignalLoopState` fills all of this in. The others leave
the bit clock unsynchronised, add no soft bits, carry a [`NoCN0Estimator`](@ref)
and take `carrier_phase_polarity` as a keyword (`0` by default): a host with its
own per-signal state passes the one `fold_record` returned for the record.
"""
struct LoopRecord{S<:AbstractGNSSSignal,C<:AbstractCorrelator,F,CN0<:AbstractCN0Estimator}
    signal::S
    filtered_correlator::C
    previous_prompt::ComplexF64
    num_integrated_samples::Int
    sample_index::Int
    fold_end_sample_index::Int
    integrated_code_blocks::Int
    sampling_frequency::F
    prn::Int
    code_phase::Float64
    bit_synced::Bool
    sync_change::SyncChange
    blocks_into_bit::Int
    new_soft_bits::SoftBitsView
    cn0_estimator::CN0
    carrier_phase_polarity::Int8
    bit_polarity::Int8
end

LoopRecord(
    signal::AbstractGNSSSignal,
    filtered_correlator::AbstractCorrelator,
    previous_prompt,
    num_integrated_samples::Integer,
    sample_index::Integer,
    fold_end_sample_index::Integer,
    integrated_code_blocks::Integer,
    sampling_frequency;
    prn::Integer = 0,
    code_phase::Real = NaN,
    carrier_phase_polarity::Integer = 0,
) = LoopRecord(
    signal,
    filtered_correlator,
    ComplexF64(previous_prompt),
    Int(num_integrated_samples),
    Int(sample_index),
    Int(fold_end_sample_index),
    Int(integrated_code_blocks),
    sampling_frequency,
    Int(prn),
    Float64(code_phase),
    false,
    SYNC_UNCHANGED,
    0,
    _no_soft_bits(),
    NoCN0Estimator(),
    Int8(carrier_phase_polarity),
    Int8(0),
)

LoopRecord(
    signal,
    filtered_correlator,
    previous_prompt,
    output::CorrelatorOutput,
    integrated_code_blocks::Integer,
    sampling_frequency;
    fold_end_sample_index = output.sample_index,
    prn::Integer = 0,
    sample_offset::Integer = 0,
    carrier_phase_polarity::Integer = 0,
) = LoopRecord(
    signal,
    filtered_correlator,
    ComplexF64(previous_prompt),
    output.num_integrated_samples,
    output.sample_index + Int(sample_offset),
    Int(fold_end_sample_index) + Int(sample_offset),
    Int(integrated_code_blocks),
    sampling_frequency,
    Int(prn),
    output.code_phase,
    false,
    SYNC_UNCHANGED,
    0,
    _no_soft_bits(),
    NoCN0Estimator(),
    Int8(carrier_phase_polarity),
    Int8(0),
)

"""
    LoopRecord(signal, filtered_correlator, output::CorrelatorOutput,
               state::SignalLoopState, sampling_frequency;
               fold_end_sample_index = output.sample_index, prn = 0, sample_offset = 0)

The record of `output` with its block count, previous prompt, bit-clock summary,
C/N₀ estimator and carrier phase polarity taken from `state`, the
[`SignalLoopState`](@ref) that [`apply_record`](@ref) just folded `output` into
(see `LoopRecord` for the fields).
"""
function LoopRecord(
    signal,
    filtered_correlator,
    output::CorrelatorOutput,
    state::SignalLoopState,
    sampling_frequency;
    fold_end_sample_index = output.sample_index,
    prn::Integer = 0,
    sample_offset::Integer = 0,
)
    bit_buffer = state.bit_buffer
    synced = has_bit_or_secondary_code_been_found(bit_buffer)
    soft_bits = bit_buffer.soft_bits
    num_soft_bits = Base.length(soft_bits)
    first_new = max(1, num_soft_bits - state.last_num_new_soft_bits + 1)
    LoopRecord(
        signal,
        filtered_correlator,
        state.last_previous_prompt,
        output.num_integrated_samples,
        output.sample_index + Int(sample_offset),
        Int(fold_end_sample_index) + Int(sample_offset),
        state.last_num_code_blocks,
        sampling_frequency,
        Int(prn),
        output.code_phase,
        synced,
        state.last_sync_change,
        synced ? bit_buffer.prompt_accumulator_integrated_code_blocks : 0,
        view(soft_bits, first_new:num_soft_bits),
        state.cn0_estimator,
        state.last_carrier_phase_polarity,
        synced ? bit_buffer.polarity : Int8(0),
    )
end

# ── The conventional PLL/DLL ─────────────────────────────────────────────────

"""
Per-satellite state of a [`ConventionalPLLAndDLL`](@ref): initial Dopplers, loop
filters, their [`LoopBandwidths`](@ref), the key of the driver signal whose records
close the loops (`0`: every record's; see [`step_loop`](@ref)), and the
[`CarrierLoopStaging`](@ref).
"""
@kwdef struct SatConventionalPLLAndDLL{CA<:AbstractLoopFilter,CO<:AbstractLoopFilter}
    init_carrier_doppler::typeof(1.0Hz)
    init_code_doppler::typeof(1.0Hz)
    carrier_loop_filter::CA = ThirdOrderBilinearLF()
    code_loop_filter::CO = SecondOrderBilinearLF()
    bandwidths::LoopBandwidths = LoopBandwidths()
    driver::UInt64 = UInt64(0)
    staging::CarrierLoopStaging = CarrierLoopStaging(typeof(carrier_loop_filter))
end

function SatConventionalPLLAndDLL(
    sat_conventional_pll_and_dll::SatConventionalPLLAndDLL{CA,CO};
    carrier_loop_filter::Maybe{CA} = nothing,
    code_loop_filter::Maybe{CO} = nothing,
    bandwidths::Maybe{LoopBandwidths} = nothing,
    staging::Maybe{CarrierLoopStaging} = nothing,
) where {CA<:AbstractLoopFilter,CO<:AbstractLoopFilter}
    SatConventionalPLLAndDLL{CA,CO}(
        sat_conventional_pll_and_dll.init_carrier_doppler,
        sat_conventional_pll_and_dll.init_code_doppler,
        isnothing(carrier_loop_filter) ? sat_conventional_pll_and_dll.carrier_loop_filter :
        carrier_loop_filter,
        isnothing(code_loop_filter) ? sat_conventional_pll_and_dll.code_loop_filter :
        code_loop_filter,
        isnothing(bandwidths) ? sat_conventional_pll_and_dll.bandwidths : bandwidths,
        sat_conventional_pll_and_dll.driver,
        isnothing(staging) ? sat_conventional_pll_and_dll.staging : staging,
    )
end

# The key a scalar loop's state records its driver by: the signal's id as an
# integer, so the state stays a plain bits value of one type whatever the driver,
# and a host can keep the states of satellites with different drivers in one
# vector.
@inline _signal_key(signal::AbstractGNSSSignal) = UInt64(objectid(get_signal_id(signal)))

"""
$(SIGNATURES)

Conventional PLL and DLL Doppler estimator. Configuration only; per-satellite
state is a [`SatConventionalPLLAndDLL`](@ref) from [`init_estimator_state`](@ref).

`CA` and `CO` are the carrier and code loop filter types. The bandwidths (the PLL's
wide and narrow, the FLL path's and the DLL's; see [`CarrierLoopStage`](@ref))
default to `nothing`, **auto**: sized per satellite from its driver signal via
[`default_wide_carrier_loop_filter_bandwidth`](@ref),
[`default_narrow_carrier_loop_filter_bandwidth`](@ref),
[`default_fll_assist_loop_filter_bandwidth`](@ref) and
[`default_code_loop_filter_bandwidth`](@ref). Each is capped against the record's
integration time at filter time (see [`LoopBandwidths`](@ref)), so longer
integration needs no re-tuning.
"""
struct ConventionalPLLAndDLL{CA<:AbstractLoopFilter,CO<:AbstractLoopFilter} <:
       AbstractDopplerEstimator
    wide_carrier_loop_filter_bandwidth::Maybe{typeof(1.0Hz)}
    narrow_carrier_loop_filter_bandwidth::Maybe{typeof(1.0Hz)}
    fll_assist_loop_filter_bandwidth::Maybe{typeof(1.0Hz)}
    code_loop_filter_bandwidth::Maybe{typeof(1.0Hz)}
end

function ConventionalPLLAndDLL(
    ::Type{CA} = ThirdOrderBilinearLF,
    ::Type{CO} = SecondOrderBilinearLF;
    wide_carrier_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    narrow_carrier_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    fll_assist_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    code_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
) where {CA<:AbstractLoopFilter,CO<:AbstractLoopFilter}
    ConventionalPLLAndDLL{CA,CO}(
        wide_carrier_loop_filter_bandwidth,
        narrow_carrier_loop_filter_bandwidth,
        fll_assist_loop_filter_bandwidth,
        code_loop_filter_bandwidth,
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
    narrow_carrier_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    fll_assist_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    code_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
) where {CO<:AbstractLoopFilter}
    ConventionalPLLAndDLL(
        ThirdOrderAssistedBilinearLF,
        CO;
        wide_carrier_loop_filter_bandwidth,
        narrow_carrier_loop_filter_bandwidth,
        fll_assist_loop_filter_bandwidth,
        code_loop_filter_bandwidth,
    )
end

# Copy with the given (non-`nothing`) fields replaced.
function ConventionalPLLAndDLL(
    pll_and_dll::ConventionalPLLAndDLL{CA,CO};
    wide_carrier_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    narrow_carrier_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    fll_assist_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    code_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
) where {CA<:AbstractLoopFilter,CO<:AbstractLoopFilter}
    ConventionalPLLAndDLL{CA,CO}(
        isnothing(wide_carrier_loop_filter_bandwidth) ?
        pll_and_dll.wide_carrier_loop_filter_bandwidth : wide_carrier_loop_filter_bandwidth,
        isnothing(narrow_carrier_loop_filter_bandwidth) ?
        pll_and_dll.narrow_carrier_loop_filter_bandwidth :
        narrow_carrier_loop_filter_bandwidth,
        isnothing(fll_assist_loop_filter_bandwidth) ?
        pll_and_dll.fll_assist_loop_filter_bandwidth : fll_assist_loop_filter_bandwidth,
        isnothing(code_loop_filter_bandwidth) ? pll_and_dll.code_loop_filter_bandwidth :
        code_loop_filter_bandwidth,
    )
end

# The estimator's bandwidths, each `nothing` taken from `defaults` (the driver
# signal's).
@inline _resolve_bandwidths(estimator, defaults::LoopBandwidths) = LoopBandwidths(
    isnothing(estimator.wide_carrier_loop_filter_bandwidth) ? defaults.wide_carrier :
    estimator.wide_carrier_loop_filter_bandwidth,
    isnothing(estimator.narrow_carrier_loop_filter_bandwidth) ? defaults.narrow_carrier :
    estimator.narrow_carrier_loop_filter_bandwidth,
    isnothing(estimator.fll_assist_loop_filter_bandwidth) ? defaults.fll_assist :
    estimator.fll_assist_loop_filter_bandwidth,
    isnothing(estimator.code_loop_filter_bandwidth) ? defaults.code :
    estimator.code_loop_filter_bandwidth,
)

"""
    init_estimator_state(estimator, driver_signal, carrier_doppler, code_doppler)

Per-satellite estimator state for a satellite driven by `driver_signal`, starting
at the given Dopplers; auto bandwidths are resolved here from `driver_signal`. The
state records the driver, so [`step_loop`](@ref) tells its records from the
satellite's other signals', and starts the [`CarrierLoopStage`](@ref) staging.
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
        _resolve_bandwidths(estimator, LoopBandwidths(driver_signal)),
        _signal_key(driver_signal),
        CarrierLoopStaging(typeof(carrier_loop_filter)),
    )
end

# `Accessors.constructorof` without the dependency: the loop filters are plain
# parametric structs, constructed by their type name.
_constructorof(::Type{T}) where {T} = Base.typename(T).wrapper

"""
    reset_estimator_state(estimator, state, carrier_doppler, code_doppler)

Re-seed `state` at the given Dopplers with zeroed loop filters, keeping the
bandwidths and the carrier loop's staging (see [`CarrierLoopStage`](@ref)): a reset
is a handoff, not a re-acquisition, so after a change of integration length, say,
the loop holds its stage, the phase-lock indicator reads on, and the bandwidths are
capped for the new length.
"""
function reset_estimator_state(
    ::ConventionalPLLAndDLL,
    state::SatConventionalPLLAndDLL,
    carrier_doppler,
    code_doppler,
)
    SatConventionalPLLAndDLL(
        carrier_doppler,
        code_doppler,
        _constructorof(typeof(state.carrier_loop_filter))(),
        _constructorof(typeof(state.code_loop_filter))(),
        state.bandwidths,
        state.driver,
        state.staging,
    )
end

"""
    step_loop(estimator::ConventionalPLLAndDLL, state, record::LoopRecord, words,
              landing_sample)
        -> (state, carrier_doppler, code_doppler)

One record through the conventional loop at the bandwidths of its
[`CarrierLoopStage`](@ref), capped against the record's integration time; the DLL
is normalised with the code word the record ran on, and the Dopplers are aided
([`aid_dopplers`](@ref)). `landing_sample` is ignored:
the command is assumed to act before the next record.

A host hands every record of a satellite to `step_loop`, the driver's and its
passengers' alike; only the driver's step the loop. A passenger record leaves the
state as it is and returns the command in force at the landing sample (the
record's end under `NO_LANDING_SAMPLE`), so applying it changes nothing.
"""
@inline step_loop(
    estimator::ConventionalPLLAndDLL,
    state::SatConventionalPLLAndDLL,
    record::LoopRecord,
    words,
    landing_sample::Int64,
) =
    _is_driver_record(state, record) ?
    _step_scalar_loop(estimator, state, record, words, landing_sample) :
    _step_passenger(state, record, words, landing_sample)

# Whether `record` belongs to the signal the scalar state was initialised with.
@inline _is_driver_record(state, record::LoopRecord) = _is_driven_by(state, record.signal)

# Whether the scalar state's satellite is driven by `signal` (any signal for a state
# without a recorded driver).
@inline _is_driven_by(state, signal::AbstractGNSSSignal) =
    state.driver == 0 || _signal_key(signal) == state.driver

# A passenger record through a loop it does not close: the state unchanged and the
# command in force where this record's command would land.
@inline function _step_passenger(state, record::LoopRecord, words, landing_sample::Int64)
    carrier_doppler, code_doppler = _command_in_force(record, words, landing_sample)
    state, carrier_doppler, code_doppler
end

@inline function _command_in_force(record::LoopRecord, words, landing_sample::Int64)
    landing = landing_sample == NO_LANDING_SAMPLE ? record.sample_index : landing_sample
    carrier_doppler, code_doppler = mean_nco_word(words, landing, landing)
    carrier_doppler * Hz, code_doppler * Hz
end

# One record's discriminators plus its integration time and centre sample.
# `phase_error` (cycles) and `frequency_error` (Hz) feed the scalar carrier filter;
# the frequency error is formed only if `form_fll` (while FLL-assisted), else zero.
# The DLL's `code_error` is normalised with the code word the record ran on.
@inline function _record_discriminators(
    ::ConventionalPLLAndDLL,
    state::SatConventionalPLLAndDLL,
    record::LoopRecord,
    words,
    landing_sample::Int64;
    form_fll::Bool,
)
    signal = record.signal
    integration_time = record.num_integrated_samples / record.sampling_frequency
    record_start_sample_index = record.sample_index - record.num_integrated_samples
    _, applied_code = mean_nco_word(words, record_start_sample_index, record.sample_index)
    filtered_correlator = record.filtered_correlator
    frequency_error =
        form_fll ?
        fll_disc(
            signal,
            filtered_correlator,
            record.previous_prompt,
            integration_time;
            polarity = record.carrier_phase_polarity,
        ) : 0.0Hz
    (;
        integration_time,
        phase_error = pll_disc(signal, filtered_correlator; polarity = record.carrier_phase_polarity),
        frequency_error,
        code_error = dll_disc(
            signal,
            filtered_correlator,
            applied_code * Hz,
            record.sampling_frequency,
        ),
        center = NaN,
    )
end

# The carrier filter's input: the FLL-assisted filter takes both
# discriminators, any other the phase discriminator alone.
@inline _carrier_filter_input(::ThirdOrderAssistedBilinearLF, phase_error, frequency_error) =
    (phase_error, frequency_error)
@inline _carrier_filter_input(::AbstractLoopFilter, phase_error, frequency_error) =
    phase_error

# The scalar carrier filter's bandwidth for a record of `integration_time`: the
# stage's PLL bandwidth, paired with the FLL path's for an FLL-assisted filter.
# Capped against the time the record actually integrated, not the intended length:
# records folded after a mid-fold sync, and the truncated first post-sync
# integration, are short.
@inline function _carrier_filter_bandwidth(state, integration_time)
    pll = effective_carrier_loop_filter_bandwidth(
        state.staging.stage == NARROW_PLL ? state.bandwidths.narrow_carrier :
        state.bandwidths.wide_carrier,
        integration_time,
    )
    _has_fll_path(typeof(state.carrier_loop_filter)) || return pll
    pll, effective_fll_assist_loop_filter_bandwidth(state.bandwidths.fll_assist, integration_time)
end

# The state after one record, with both loop filters stepped and the staging
# advanced.
@inline _stepped_state(
    state::SatConventionalPLLAndDLL,
    carrier_loop_filter,
    code_loop_filter,
    center,
    staging,
) = SatConventionalPLLAndDLL(state; carrier_loop_filter, code_loop_filter, staging)

# One record through a scalar loop (see `step_loop`).
@inline function _step_scalar_loop(estimator, state, record::LoopRecord, words, landing_sample::Int64)
    discriminators = _record_discriminators(
        estimator,
        state,
        record,
        words,
        landing_sample;
        form_fll = state.staging.stage == FLL_ASSISTED_PLL,
    )
    integration_time = discriminators.integration_time
    code_bandwidth =
        effective_code_loop_filter_bandwidth(state.bandwidths.code, integration_time)
    carrier_freq_update, carrier_loop_filter = filter_loop(
        state.carrier_loop_filter,
        _carrier_filter_input(
            state.carrier_loop_filter,
            discriminators.phase_error,
            discriminators.frequency_error,
        ),
        integration_time,
        _carrier_filter_bandwidth(state, integration_time),
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
        _advance_staging(state.staging, record),
    ),
    carrier_doppler,
    code_doppler
end

# ── The NCO-referenced (delay-aware) PLL/DLL ─────────────────────────────────

"""
    NCOReferencedPLLAndDLL(; wide_carrier_loop_filter_bandwidth = nothing,
                             narrow_carrier_loop_filter_bandwidth = nothing,
                             fll_assist_loop_filter_bandwidth = nothing,
                             code_loop_filter_bandwidth = nothing,
                             predict_landing = true)

FLL-assisted PLL and DLL Doppler estimator for a replica whose NCO words are
applied with a known delay; the hardware-correlator receiver's default.

It is the [`ConventionalAssistedPLLAndDLL`](@ref) (same filters, gains,
[`CarrierLoopStage`](@ref) staging and bandwidth defaults) with its internals
referenced to the device NCO. The one different default is the wide PLL bandwidth:
the narrow one ([`default_narrow_carrier_loop_filter_bandwidth`](@ref)), as a wider
loop tolerates less command delay.

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
loop at the same bandwidths, inheriting its noise performance.
`predict_landing = false` drops step 2, making it arithmetically the
[`ConventionalAssistedPLLAndDLL`](@ref): the documented **negative control**,
which fails like it at a few epochs of delay.
"""
struct NCOReferencedPLLAndDLL{CO<:AbstractLoopFilter} <: AbstractDopplerEstimator
    wide_carrier_loop_filter_bandwidth::Maybe{typeof(1.0Hz)}
    narrow_carrier_loop_filter_bandwidth::Maybe{typeof(1.0Hz)}
    fll_assist_loop_filter_bandwidth::Maybe{typeof(1.0Hz)}
    code_loop_filter_bandwidth::Maybe{typeof(1.0Hz)}
    predict_landing::Bool
end

function NCOReferencedPLLAndDLL(
    ::Type{CO} = SecondOrderBilinearLF;
    wide_carrier_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    narrow_carrier_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    fll_assist_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    code_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    predict_landing::Bool = true,
) where {CO<:AbstractLoopFilter}
    NCOReferencedPLLAndDLL{CO}(
        wide_carrier_loop_filter_bandwidth,
        narrow_carrier_loop_filter_bandwidth,
        fll_assist_loop_filter_bandwidth,
        code_loop_filter_bandwidth,
        predict_landing,
    )
end

"""
    SatNCOReferencedPLLAndDLL

Per-satellite state of an [`NCOReferencedPLLAndDLL`](@ref): handover Dopplers
(the filters output offsets from them), both filters, their bandwidths, the centre
sample of the last record (`NaN` before the first; the FLL measures between two
prompts' centres, so its replica word is averaged over that span), and the driver key
and staging as in [`SatConventionalPLLAndDLL`](@ref).
"""
struct SatNCOReferencedPLLAndDLL{CA<:ThirdOrderAssistedBilinearLF,CO<:AbstractLoopFilter}
    init_carrier_doppler::typeof(1.0Hz)
    init_code_doppler::typeof(1.0Hz)
    carrier_loop_filter::CA
    code_loop_filter::CO
    bandwidths::LoopBandwidths
    previous_record_center::Float64
    driver::UInt64
    staging::CarrierLoopStaging
end

function SatNCOReferencedPLLAndDLL(
    state::SatNCOReferencedPLLAndDLL{CA,CO};
    carrier_loop_filter::Maybe{CA} = nothing,
    code_loop_filter::Maybe{CO} = nothing,
    previous_record_center::Maybe{Float64} = nothing,
    staging::Maybe{CarrierLoopStaging} = nothing,
    bandwidths::Maybe{LoopBandwidths} = nothing,
) where {CA,CO}
    SatNCOReferencedPLLAndDLL{CA,CO}(
        state.init_carrier_doppler,
        state.init_code_doppler,
        something(carrier_loop_filter, state.carrier_loop_filter),
        something(code_loop_filter, state.code_loop_filter),
        something(bandwidths, state.bandwidths),
        something(previous_record_center, state.previous_record_center),
        state.driver,
        something(staging, state.staging),
    )
end

# The NCO-referenced loop pulls in at the narrow bandwidth by default: a wide
# loop tolerates less command delay.
function init_estimator_state(
    estimator::NCOReferencedPLLAndDLL{CO},
    driver_signal::AbstractGNSSSignal,
    carrier_doppler,
    code_doppler,
) where {CO}
    carrier_loop_filter = ThirdOrderAssistedBilinearLF()
    SatNCOReferencedPLLAndDLL(
        carrier_doppler,
        code_doppler,
        carrier_loop_filter,
        _constructorof(CO)(),
        _resolve_bandwidths(
            estimator,
            LoopBandwidths(
                driver_signal;
                wide_carrier = default_narrow_carrier_loop_filter_bandwidth(driver_signal),
            ),
        ),
        NaN,
        _signal_key(driver_signal),
        CarrierLoopStaging(typeof(carrier_loop_filter)),
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
        state.driver,
        state.staging,
    )
end

"""
    wrap_phase_error(phase_error; polarity = 0)

Fold a carrier-phase error in cycles into the range of [`pll_disc`](@ref) with the
same `polarity`: ±1/4 cycle for the Costas PLL (`polarity = 0`; a bit flip turns
the prompt by half a cycle), ±1/2 for the four-quadrant one. Exact inside it.
"""
wrap_phase_error(phase_error; polarity::Real = 0) =
    rem(phase_error, iszero(polarity) ? 0.5 : 1.0, RoundNearest)

# The phase error (cycles) the record would show `shift` samples later under the
# scheduled words, integrated from the record's centre with the filter's Doppler
# estimate before this record's update, folded as by `wrap_phase_error`.
@inline function _predict_landing_phase_error(
    phase_error,
    polarity,
    state::SatNCOReferencedPLLAndDLL,
    words,
    center,
    shift,
    integration_time,
    sampling_frequency,
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
    wrap_phase_error(phase_error + shift * (f_hat - ramp_word) / sampling_freq_hz; polarity)
end

"""
    step_loop(estimator::NCOReferencedPLLAndDLL, state, record::LoopRecord, words,
              landing_sample)
        -> (state, carrier_doppler, code_doppler)

One record through the NCO-referenced loop. `words` are the replica words the
record ran on; `landing_sample` is where this fold's command lands
(`NO_LANDING_SAMPLE`: at the record's end). See [`NCOReferencedPLLAndDLL`](@ref).
Passenger records are handled as by the conventional loop.
"""
@inline step_loop(
    estimator::NCOReferencedPLLAndDLL,
    state::SatNCOReferencedPLLAndDLL,
    record::LoopRecord,
    words,
    landing_sample::Int64,
) =
    _is_driver_record(state, record) ?
    _step_scalar_loop(estimator, state, record, words, landing_sample) :
    _step_passenger(state, record, words, landing_sample)

@inline function _record_discriminators(
    estimator::NCOReferencedPLLAndDLL,
    state::SatNCOReferencedPLLAndDLL,
    record::LoopRecord,
    words,
    landing_sample::Int64;
    form_fll::Bool,
)
    signal = record.signal
    sampling_frequency = record.sampling_frequency
    previous_center = state.previous_record_center
    # Every record of the fold maps onto the delay-free loop's record this far ahead.
    shift = landing_sample == NO_LANDING_SAMPLE ? 0 : landing_sample - record.fold_end_sample_index
    record_end_sample_index = record.sample_index
    record_samples = record.num_integrated_samples
    record_start_sample_index = record_end_sample_index - record_samples
    center = record_end_sample_index - record_samples / 2
    integration_time = record_samples / sampling_frequency
    filtered_correlator = record.filtered_correlator

    applied_carrier, applied_code = mean_nco_word(words, record_start_sample_index, record_end_sample_index)
    # Discriminators against the applied replica; the FLL's only if `form_fll`.
    phase_error = pll_disc(signal, filtered_correlator; polarity = record.carrier_phase_polarity)
    frequency_error =
        form_fll ?
        fll_disc(
            signal,
            filtered_correlator,
            record.previous_prompt,
            integration_time;
            polarity = record.carrier_phase_polarity,
        ) : 0.0Hz

    if estimator.predict_landing && shift > 0
        phase_error = _predict_landing_phase_error(
            phase_error,
            record.carrier_phase_polarity,
            state,
            words,
            center,
            shift,
            integration_time,
            sampling_frequency,
        )
        # Re-base the FLL reading from the word between the prompts' centres onto
        # the word running `shift` samples ahead.
        fll_word =
            isnan(previous_center) ? applied_carrier :
            first(mean_nco_word(words, previous_center, center))
        landing_word =
            first(mean_nco_word(words, record_start_sample_index + shift, record_end_sample_index + shift))
        form_fll && (frequency_error += (fll_word - landing_word) * Hz)
    end

    (;
        integration_time,
        phase_error,
        frequency_error,
        code_error = dll_disc(signal, filtered_correlator, applied_code * Hz, sampling_frequency),
        center,
    )
end

@inline _stepped_state(
    state::SatNCOReferencedPLLAndDLL,
    carrier_loop_filter,
    code_loop_filter,
    center,
    staging,
) = SatNCOReferencedPLLAndDLL(
    state;
    carrier_loop_filter,
    code_loop_filter,
    previous_record_center = center,
    staging,
)

"""
    carrier_loop_stage(state) -> CarrierLoopStage

The [`CarrierLoopStage`](@ref) of a scalar estimator's per-satellite state, or
of a [`VectorPLLAndDLL`](@ref)'s: `VECTOR_LOOP` while the satellite is in
the vector loop, the inner loop's stage out of it.
"""
carrier_loop_stage(state::Union{SatConventionalPLLAndDLL,SatNCOReferencedPLLAndDLL}) =
    state.staging.stage

"""
    phase_lock_indicator(state) -> Float64

The latest phase-lock indicator of a scalar estimator's per-satellite state, or of
a [`VectorPLLAndDLL`](@ref)'s (in the vector loop its residual PLL's, on the prompts
it rotated; out of it the inner loop's), which the carrier-loop staging reads (see
[`CarrierLoopStage`](@ref)): `⟨I² − Q²⟩ / A²` of the driver's prompts, an estimate
of `cos 2φ` for the carrier phase error `φ` (1 in lock, 0 for a spinning phase),
averaged over 0.1 s or 25 records, whichever is longer. The signal power
`A² = √(2 M₂² − M₄)`, from the prompts' moments `Mₖ = ⟨|P|ᵏ⟩`, is free of noise, so
unlike `⟨I² − Q²⟩ / ⟨I² + Q²⟩` a locked loop's reading gets noisier at low C/N₀,
not lower. `NaN` until the averages span that time.
"""
phase_lock_indicator(state::Union{SatConventionalPLLAndDLL,SatNCOReferencedPLLAndDLL}) =
    state.staging.lock_detector.indicator

"""
    phase_locked(state) -> Bool

Whether the carrier loop of a scalar estimator's per-satellite state, or of a
[`VectorPLLAndDLL`](@ref)'s, is phase-locked, as the loop itself decides it: a
scalar loop while its [`phase_lock_indicator`](@ref) reads at least 0.5, the
staging's threshold (`false` while `NaN`). In the vector loop it is the residual
PLL's lock, which also asks that the C/N₀ the rotated prompts' moments read be at
most 6 dB below the estimator's `lock_cn0_threshold`, as on bare noise the indicator
alone reads lock now and then. Out of the vector loop, the inner loop's.
"""
phase_locked(state::Union{SatConventionalPLLAndDLL,SatNCOReferencedPLLAndDLL}) =
    phase_lock_indicator(state) >= _PHASE_LOCK_THRESHOLD

"""
    carrier_phase_correction(state, words, sample_index, sampling_frequency) -> Float64

The carrier phase (cycles) the loop estimates the incoming carrier to lead the
replica by at `sample_index` (of the records' grid at `sampling_frequency`), with
`words` the words the replica ran on up to there, an [`NCOTimeline`](@ref) or
a [`FixedNCOWord`](@ref). The replica's carrier phase plus this is the loop's
estimate of the incoming carrier phase, the carrier-phase measurement a host
reports (RINEX `L`).

A scalar loop steers its replica onto the incoming carrier, so the replica's
phase is its estimate and this is `0.0`; so is a [`VectorPLLAndDLL`](@ref)'s
out of the vector loop. In it the navigation filter sets the carrier NCO and
the residual PLL follows the carrier the NCO leaves without steering it, so
the replica's phase alone drifts off the carrier, by whole cycles over time:
this is the residual PLL's prediction at `sample_index`, from its latest prompt on,
counting every cycle, never folded (`0.0` before its first prompt). Meant for
a sample index at or after the end of the driver's latest record, as a host's
replica phase after its last record is.

Like a scalar PLL's replica phase, it is ambiguous by half a cycle under the
Costas discriminator, and [`phase_locked`](@ref) says whether it can be
trusted. It steps where the loop does not follow the carrier continuously,
which a RINEX host flags in the loss-of-lock indicator:

  - by half a cycle when the record's `carrier_phase_polarity` turns
    four-quadrant (a synced pilot) and the residual PLL finds itself half a
    cycle off the polarity's frame;
  - on a release from the vector loop, when [`carrier_loop_stage`](@ref)
    leaves `VECTOR_LOOP`: the correction drops to zero while the scalar loop
    pulls its replica in afresh from where the NCO left it, so the phase steps
    by the residual and may slip.

A satellite added to the vector loop keeps its phase: the residual PLL starts
at zero on the phase-locked scalar loop's replica.
"""
carrier_phase_correction(
    ::Union{SatConventionalPLLAndDLL,SatNCOReferencedPLLAndDLL},
    words,
    sample_index,
    sampling_frequency,
) = 0.0

"The estimator-state type a Doppler estimator produces (for slot typing)."
estimator_state_type(estimator::AbstractDopplerEstimator, driver_signal::AbstractGNSSSignal) =
    typeof(init_estimator_state(estimator, driver_signal, 0.0Hz, 0.0Hz))

# ── What an estimator knows of the navigation solution ───────────────────────

"""
    navigation_solution(estimator) -> Union{PVTSolution,Nothing}

The latest navigation solution (position, velocity, time, clock bias and drift, DOP,
the satellites used with their residuals, inter-system and inter-frequency biases),
or `nothing` for the scalar loops. For [`VectorPLLAndDLL`](@ref) see its method.
"""
navigation_solution(::AbstractDopplerEstimator) = nothing

"""
    navigation_status(estimator) -> Union{NavigationStatus,Nothing}

What the latest navigation cycle did ([`NavigationStatus`](@ref)), or `nothing` for an
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
