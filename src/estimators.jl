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

# The soft bits a record appended: a contiguous view of the bit buffer's vector.
const SoftBitsView = SubArray{Float32,1,Vector{Float32},Tuple{UnitRange{Int}},true}

# The soft bits of a record built without a signal state: none. Never written to.
const _NO_SOFT_BITS = Float32[]

@inline _no_soft_bits() = view(_NO_SOFT_BITS, 1:0)

"""
$(SIGNATURES)

One completed record as the loop-filter step sees it: the signal it belongs
to, the filtered (antenna-combined, normalised) correlator, the previous
record's filtered prompt the FLL chains from, the record's span and the blocks
it covered, the band's sampling frequency, and `fold_end` — the end sample of
the last record of the fold this record belongs to, which is what a landing
sample is measured against (every record of a fold maps onto the delay-free
loop's record the same distance ahead).

`integrated_code_blocks`, the primary code blocks the record covered, must be at
least 1: the carrier-loop staging sizes its phase-lock indicator's averages in
records of that length and divides by it. A record built from a
[`SignalLoopState`](@ref) always is, as [`apply_record`](@ref) counts at least
one block; a host that builds records otherwise passes at least 1.

The four-quadrant FLL reads a sign flip between the previous and this record's
prompt as half a cycle of rotation. A record whose `carrier_phase_polarity`
differs from the previous record's therefore comes without a previous prompt
(`0`), and the FLL gives no reading for it, as for the first record. The
constructor that takes a [`SignalLoopState`](@ref) takes the previous prompt
from the state (its `last_previous_prompt`), which applies this; a host that
builds records otherwise passes `0` itself.

Two fields identify the record to an estimator that keeps per-satellite state
of its own, as [`VectorPLLAndDLL`](@ref) does:

  - `prn`: the satellite (`0` when the host does not say);
  - `code_phase`: the replica's code phase (chips) at `sample_index`, from the
    [`CorrelatorOutput`](@ref) (`NaN` when the producer does not report it).

For such an estimator `sample_index / sampling_frequency` must also be the
time since one origin shared by every satellite of the band. A host whose
correlator restarts its sample count passes that origin's offset as
`sample_offset` to the constructor that takes a `CorrelatorOutput`; it is added
to `sample_index` and `fold_end`.

The rest summarises the signal's [`SignalLoopState`](@ref) after
[`apply_record`](@ref) folded this record into it, for an estimator that
decodes or weights by C/N₀ (the vector loop; the scalar loops ignore it):

  - `bit_synced`: whether the bit clock holds the bit or secondary-code sync
    after this record;
  - `sync_change`: whether this record found or lost it ([`SyncChange`](@ref));
  - `blocks_into_symbol`: the code blocks integrated into the current data
    symbol (`0` before the sync);
  - `new_soft_bits`: exactly the soft bits *this record* appended — a view of
    the bit buffer's vector, valid for the duration of the call, so it does not
    matter when the host drains the buffer;
  - `cn0_estimator`: the signal's C/N₀ estimator, for [`estimate_cn0`](@ref);
  - `bit_polarity`: the sign the bit clock's sync found its soft bits at (`-1`
    where it negates the accumulated prompts' real part, `+1` where it keeps
    it, `0` before the sync).

`carrier_phase_polarity` says how the carrier discriminators read the record:
the sign of its prompt where the replica wiped every sign modulation off it — a
dataless signal synced to its secondary code — and `0` where it is unknown.
Nonzero, the PLL and the FLL read it four-quadrant (`pll_disc(...; polarity)`,
`fll_disc(...; polarity)`); `0`, both stay two-quadrant.

Build such a record with the constructor that takes the state in place of the
block count and the previous prompt: `LoopRecord(signal, filtered_correlator,
output, state, sampling_frequency)`. The other constructors leave the bit clock
unsynchronised, add no soft bits, carry a [`NoCN0Estimator`](@ref) and take
`carrier_phase_polarity` as a keyword, `0` by default: a host with its own
per-signal state passes the one `fold_record` returned for the record.
"""
struct LoopRecord{S<:AbstractGNSSSignal,C<:AbstractCorrelator,F,CN0<:AbstractCN0Estimator}
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
    bit_synced::Bool
    sync_change::SyncChange
    blocks_into_symbol::Int
    new_soft_bits::SoftBitsView
    cn0_estimator::CN0
    carrier_phase_polarity::Int8
    bit_polarity::Int8
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
    carrier_phase_polarity::Integer = 0,
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
    fold_end = output.sample_index,
    prn::Integer = 0,
    sample_offset::Integer = 0,
    carrier_phase_polarity::Integer = 0,
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
               fold_end = output.sample_index, prn = 0, sample_offset = 0)

The record of `output`, with the summary of the signal's bit clock, its C/N₀
estimator and its carrier phase polarity taken from `state` — the
[`SignalLoopState`](@ref) that [`apply_record`](@ref) just folded `output` into
— along with the blocks the record covered, and the previous prompt its FLL
compares with: the state's `last_previous_prompt`, `0` where the record does not
follow on from the one before it (see `LoopRecord`).
"""
function LoopRecord(
    signal,
    filtered_correlator,
    output::CorrelatorOutput,
    state::SignalLoopState,
    sampling_frequency;
    fold_end = output.sample_index,
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
        output.integrated_samples,
        output.sample_index + Int(sample_offset),
        Int(fold_end) + Int(sample_offset),
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
Per-satellite state for the conventional PLL and DLL Doppler estimator.
Holds initial Doppler values, the loop filter states, the loop bandwidths
([`LoopBandwidths`](@ref)), the key of the satellite's driver signal, whose
records close the loops (see [`step_loop`](@ref)), which
[`init_estimator_state`](@ref) records, and the carrier loop's staging (see
[`CarrierLoopStage`](@ref)). `driver = 0` takes every record to be the driver's.
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

Conventional Phase-Locked Loop (PLL) and Delay-Locked Loop (DLL) Doppler
estimator. Configuration-only — per-satellite state is a
[`SatConventionalPLLAndDLL`](@ref), produced via [`init_estimator_state`](@ref).

Type parameters `CA` and `CO` select the carrier and code loop filter types;
the bandwidth fields configure the loop bandwidths used when seeding new
satellites: the PLL's wide bandwidth, which it pulls in at, and its narrow one,
which it narrows to once phase lock has held, the FLL-assist bandwidth of an
FLL-assisted carrier filter's FLL path, and the code bandwidth (see
[`CarrierLoopStage`](@ref)). Each bandwidth field is `Maybe{typeof(1.0Hz)}`: a
`nothing` field (the default) means **auto** — the bandwidth is sized per
satellite from its estimator-driver signal via
[`default_wide_carrier_loop_filter_bandwidth`](@ref),
[`default_narrow_carrier_loop_filter_bandwidth`](@ref),
[`default_fll_assist_loop_filter_bandwidth`](@ref) and
[`default_code_loop_filter_bandwidth`](@ref). At filter time each is capped
against the record's integration time
([`effective_carrier_loop_filter_bandwidth`](@ref),
[`effective_fll_assist_loop_filter_bandwidth`](@ref),
[`effective_code_loop_filter_bandwidth`](@ref)), so a longer coherent
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

Create a ConventionalPLLAndDLL with FLL-assisted carrier tracking: a
`ThirdOrderAssistedBilinearLF` carrier loop filter combining the PLL and FLL
discriminators until phase lock (see [`CarrierLoopStage`](@ref)). Bandwidths
default to `nothing` (auto).
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

# Kwarg-update constructor for tweaking bandwidths in place.
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

# The bandwidths of a scalar estimator for a satellite: each the estimator's,
# each `nothing` there the one of `defaults`, the driver signal's.
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

Build the per-satellite estimator state for a satellite whose loop is driven
by `driver_signal` and starts at the given Dopplers. Auto bandwidths (`nothing`
on the estimator) are resolved here from the driver signal. The state records
the driver, so [`step_loop`](@ref) tells the driver's records from those of
the satellite's other signals, and starts the carrier loop's staging (see
[`CarrierLoopStage`](@ref)).

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
        _resolve_bandwidths(estimator, LoopBandwidths(driver_signal)),
        _signal_key(driver_signal),
        CarrierLoopStaging(typeof(carrier_loop_filter)),
    )
end

# `Accessors.constructorof` without the dependency: the loop filters are plain
# parametric structs whose zero-argument constructor is their type name.
_constructorof(::Type{T}) where {T} = Base.typename(T).wrapper

"""
    reset_estimator_state(estimator, state, carrier_doppler, code_doppler)

Zero the loop-filter integrators and re-seed the state from the converged
Dopplers, keeping the per-satellite bandwidths and the carrier loop's staging
(see [`CarrierLoopStage`](@ref)). A reset is a handoff, not a re-acquisition:
after a change of the integration length, say, the re-seeded loop holds its
lock at the stage it has reached, its bandwidths capped for the new length, and
the phase-lock indicator reads on. [`init_estimator_state`](@ref) starts the
staging over.
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
    step_loop(estimator::ConventionalPLLAndDLL, state, record::LoopRecord, words, landing_sample)
        -> (state, carrier_doppler, code_doppler)

One record through the conventional loop: the PLL discriminator against the
filtered prompt, and the FLL discriminator while the FLL-assisted filter is in
its FLL-assisted stage, at the bandwidths of the carrier loop's stage (see
[`CarrierLoopStage`](@ref)); the DLL normalised with the code word the record
ran on; every bandwidth capped by its stability product against the record's
integration time, and the Dopplers aided. The record's prompt then advances the
phase-lock indicator, which moves the stage on for the next record.
`landing_sample` is ignored: the conventional loop assumes its command acts
before the next record.

A host hands every record of a satellite to `step_loop`, the driver's and its
passengers' alike. Only the driver's — the records of the signal the state was
initialised with — step the loop. A passenger record leaves the state as it is
and returns the command in force: the words at the landing sample (at the
record's end under `NO_LANDING_SAMPLE`), so applying them changes nothing.
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

# Whether the scalar state's satellite is driven by `signal` (any signal, for a state
# without a recorded driver).
@inline _is_driven_by(state, signal::AbstractGNSSSignal) =
    state.driver == 0 || _signal_key(signal) == state.driver

# A passenger record through a loop it does not close: the state unchanged and the
# command already in force where this record's command would land.
@inline function _step_passenger(state, record::LoopRecord, words, landing_sample::Int64)
    carrier_doppler, code_doppler = _command_in_force(record, words, landing_sample)
    state, carrier_doppler, code_doppler
end

@inline function _command_in_force(record::LoopRecord, words, landing_sample::Int64)
    landing = landing_sample == NO_LANDING_SAMPLE ? record.sample_index : landing_sample
    carrier_doppler, code_doppler = mean_nco_word(words, landing, landing)
    carrier_doppler * Hz, code_doppler * Hz
end

# The discriminators of one record, as the loop filters are fed them, and what the
# step needs around them: the per-record integration time and the record's centre
# sample. `phase_error` (cycles) and `frequency_error` (Hz) are what the scalar
# loop's carrier filter is fed; `raw_frequency_error` is the FLL discriminator as
# measured, against the replica that ran between the two prompts' centres, before
# any re-basing onto a landing word. Both FLL errors are formed only where
# `form_fll` asks for them — while the loop uses the FLL — and are zero otherwise.
# The DLL output `code_error` is normalised with the code word the record ran on.
@inline function _record_discriminators(
    ::ConventionalPLLAndDLL,
    state::SatConventionalPLLAndDLL,
    record::LoopRecord,
    words,
    landing_sample::Int64;
    form_fll::Bool,
)
    signal = record.signal
    integration_time = record.integrated_samples / record.sampling_frequency
    record_start = record.sample_index - record.integrated_samples
    _, applied_code = mean_nco_word(words, record_start, record.sample_index)
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

# The carrier filter's input: the FLL-assisted filter takes both
# discriminators, any other the phase discriminator alone.
@inline _carrier_filter_input(::ThirdOrderAssistedBilinearLF, phase_error, frequency_error) =
    (phase_error, frequency_error)
@inline _carrier_filter_input(::AbstractLoopFilter, phase_error, frequency_error) =
    phase_error

# The bandwidth the scalar loop's carrier filter is stepped at for a record that
# integrated for `integration_time`: the PLL's of the carrier loop's stage,
# narrow once it has narrowed and wide before, paired with the FLL path's for an
# FLL-assisted filter. Each is capped against the time the record actually
# integrated rather than the intended integration length: records folded after
# a mid-fold sync, or the truncated first post-sync integration, are still
# short.
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

# One record through a scalar loop: the carrier filter fed its discriminators at
# the bandwidths of its stage, the FLL's only while it uses it, the code filter
# the DLL with its bandwidth capped by its stability product against the
# record's integration time, the Dopplers aided and the staging advanced.
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
applied with a known delay — the hardware-correlator receiver's default.

It is the [`ConventionalAssistedPLLAndDLL`](@ref) — the same third-order
assisted bilinear carrier filter, the same second-order code filter, the same
gains, the same staging (see [`CarrierLoopStage`](@ref)) and the same bandwidth
defaults, but for the wide PLL bandwidth, which defaults to the narrow one
([`default_narrow_carrier_loop_filter_bandwidth`](@ref)), as a wider loop
tolerates less command delay — with its loop internals referenced to the device
NCO instead of to the word the filter last computed:

 1. **Every record is attributed to the word that ran under it.** The phase
    discriminator is measured against the applied replica by construction; the
    frequency discriminator is re-based onto it too, so `applied word + FLL
    discriminator` is an *absolute* measurement of the signal's Doppler. The
    DLL is normalised with the applied code word.
 2. **The correction is sized for the moment it lands.** The filter is stepped
    with the discriminators *predicted at the landing sample of the new
    command*: the measured phase error advanced by `∫ (f̂ − w(τ)) dτ` cycles
    over the words already scheduled at the NCO, and the frequency measurement
    taken relative to the word that will be running there.

With zero delay both steps are the identity and the estimator *is* the
conventional loop, so the software receiver's noise performance is inherited
rather than re-tuned.

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

Per-satellite state of an [`NCOReferencedPLLAndDLL`](@ref): the handover
Dopplers the loop filters' outputs are offsets from, both filters, their
bandwidths, the centre sample of the last record folded (the FLL measures
the mean frequency offset between two prompts' centres, so that is the span its
replica word is averaged over), the key of the satellite's driver signal,
whose records close the loops (`0` takes every record to be the driver's), and
the carrier loop's staging (see [`CarrierLoopStage`](@ref)).
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

Fold a carrier-phase error (in cycles) into the range [`pll_disc`](@ref) reads
it in with the same `polarity`: ±1/4 cycle for the Costas PLL (`polarity = 0`),
whose prompt a data bit flip turns by half a cycle, and ±1/2 cycle for the
four-quadrant PLL (a known `polarity`). Exact for any phase error already
inside it.
"""
wrap_phase_error(phase_error; polarity::Real = 0) =
    rem(phase_error, iszero(polarity) ? 0.5 : 1.0, RoundNearest)

# The phase error a record would show `shift` samples later, under the words
# the NCO will run in between, folded into the range of the record's PLL
# discriminator: the mean phase sits at the record's centre, so
# the ramp is integrated from there. The signal's Doppler is the filter's own
# estimate, before this record's innovation.
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
    step_loop(estimator::NCOReferencedPLLAndDLL, state, record::LoopRecord, words, landing_sample)
        -> (state, carrier_doppler, code_doppler)

One record through the NCO-referenced loop. `words` gives the replica words
the record really ran on; `landing_sample` is where the command computed from
this record's fold lands (`NO_LANDING_SAMPLE` for a software correlator, where
it acts at the record's end). See [`NCOReferencedPLLAndDLL`](@ref). A
passenger record — of a signal other than the one the state was initialised
with — leaves the state as it is and returns the command in force, as for
the conventional loop.
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
    # between the two prompts' centres; it is formed only where `form_fll` asks for
    # it, as for the conventional loop.
    phase_error = pll_disc(signal, filtered_correlator; polarity = record.carrier_phase_polarity)
    raw_frequency_error =
        form_fll ?
        fll_disc(
            signal,
            filtered_correlator,
            record.previous_prompt,
            integration_time;
            polarity = record.carrier_phase_polarity,
        ) : 0.0Hz
    frequency_error = raw_frequency_error

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
        # The absolute frequency measurement, relative to the word that will be
        # running under the record `shift` samples ahead. `fll_disc` measured
        # against the word that ran between the two prompts' centres.
        fll_word =
            isnan(previous_center) ? applied_carrier :
            first(mean_nco_word(words, previous_center, center))
        landing_word = first(mean_nco_word(words, record_start + shift, record_end + shift))
        form_fll && (frequency_error += (fll_word - landing_word) * Hz)
    end

    (;
        integration_time,
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

The phase-lock indicator the carrier-loop staging reads (see
[`CarrierLoopStage`](@ref)), of a scalar estimator's per-satellite state, or of
a [`VectorPLLAndDLL`](@ref)'s: in the vector loop its residual PLL's, on the
prompts the residual PLL rotated, out of it the inner loop's. `⟨I² − Q²⟩ / A²` of the
driver's prompts, an estimate of `cos 2φ` for the carrier phase error `φ` — 1
in phase lock, 0 for a spinning phase — averaged over 0.1 s or 25 records,
whichever is longer, and normalised by the signal power `A²` estimated from the
prompts' moments, so it does not read low at a low C/N₀ while the loop is
locked; it only gets noisier. `NaN` until the averages span that time.
"""
phase_lock_indicator(state::Union{SatConventionalPLLAndDLL,SatNCOReferencedPLLAndDLL}) =
    state.staging.lock_detector.indicator

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
