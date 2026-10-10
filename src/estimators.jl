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
  - `cn0_estimator`: the signal's C/N₀ estimator, for [`estimate_cn0`](@ref).

Build such a record with the constructor that takes the state in place of the
block count: `LoopRecord(signal, filtered_correlator, previous_prompt, output,
state, sampling_frequency)`. The other constructors leave the bit clock
unsynchronised, add no soft bits and carry a [`NoCN0Estimator`](@ref).
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
)

"""
    LoopRecord(signal, filtered_correlator, previous_prompt, output::CorrelatorOutput,
               state::SignalLoopState, sampling_frequency;
               fold_end = output.sample_index, prn = 0, sample_offset = 0)

The record of `output`, with the summary of the signal's bit clock and its C/N₀
estimator taken from `state` — the [`SignalLoopState`](@ref) that
[`apply_record`](@ref) just folded `output` into — along with the blocks the
record covered. Read `previous_prompt` off the state before `apply_record`
replaces it.
"""
function LoopRecord(
    signal,
    filtered_correlator,
    previous_prompt,
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
        ComplexF64(previous_prompt),
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
    )
end

# ── The conventional PLL/DLL ─────────────────────────────────────────────────

"""
Per-satellite state for the conventional PLL and DLL Doppler estimator.
Holds initial Doppler values and loop filter states, and the key of the
satellite's driver signal, whose records close the loops (see
[`step_loop`](@ref)), which [`init_estimator_state`](@ref) records. `driver =
0` takes every record to be the driver's.
"""
@kwdef struct SatConventionalPLLAndDLL{CA<:AbstractLoopFilter,CO<:AbstractLoopFilter}
    init_carrier_doppler::typeof(1.0Hz)
    init_code_doppler::typeof(1.0Hz)
    carrier_loop_filter::CA = ThirdOrderBilinearLF()
    code_loop_filter::CO = SecondOrderBilinearLF()
    carrier_loop_filter_bandwidth::typeof(1.0Hz) = 18.0Hz
    code_loop_filter_bandwidth::typeof(1.0Hz) = 1.0Hz
    driver::UInt64 = UInt64(0)
end

function SatConventionalPLLAndDLL(
    sat_conventional_pll_and_dll::SatConventionalPLLAndDLL{CA,CO};
    carrier_loop_filter::Maybe{CA} = nothing,
    code_loop_filter::Maybe{CO} = nothing,
    carrier_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    code_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
) where {CA<:AbstractLoopFilter,CO<:AbstractLoopFilter}
    SatConventionalPLLAndDLL{CA,CO}(
        sat_conventional_pll_and_dll.init_carrier_doppler,
        sat_conventional_pll_and_dll.init_code_doppler,
        isnothing(carrier_loop_filter) ? sat_conventional_pll_and_dll.carrier_loop_filter :
        carrier_loop_filter,
        isnothing(code_loop_filter) ? sat_conventional_pll_and_dll.code_loop_filter :
        code_loop_filter,
        isnothing(carrier_loop_filter_bandwidth) ?
        sat_conventional_pll_and_dll.carrier_loop_filter_bandwidth :
        carrier_loop_filter_bandwidth,
        isnothing(code_loop_filter_bandwidth) ?
        sat_conventional_pll_and_dll.code_loop_filter_bandwidth :
        code_loop_filter_bandwidth,
        sat_conventional_pll_and_dll.driver,
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
satellites. Each bandwidth field is `Maybe{typeof(1.0Hz)}`: a `nothing`
field (the default) means **auto** — the bandwidth is sized per satellite from
its estimator-driver signal via [`default_carrier_loop_filter_bandwidth`](@ref)
/ [`default_code_loop_filter_bandwidth`](@ref). At filter time both are capped
against the record's integration time
([`effective_carrier_loop_filter_bandwidth`](@ref),
[`effective_code_loop_filter_bandwidth`](@ref)), so a longer coherent
integration needs no re-tuning.
"""
struct ConventionalPLLAndDLL{CA<:AbstractLoopFilter,CO<:AbstractLoopFilter} <:
       AbstractDopplerEstimator
    carrier_loop_filter_bandwidth::Maybe{typeof(1.0Hz)}
    code_loop_filter_bandwidth::Maybe{typeof(1.0Hz)}
end

function ConventionalPLLAndDLL(
    ::Type{CA} = ThirdOrderBilinearLF,
    ::Type{CO} = SecondOrderBilinearLF;
    carrier_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    code_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
) where {CA<:AbstractLoopFilter,CO<:AbstractLoopFilter}
    ConventionalPLLAndDLL{CA,CO}(carrier_loop_filter_bandwidth, code_loop_filter_bandwidth)
end

"""
$(SIGNATURES)

Create a ConventionalPLLAndDLL with FLL-assisted carrier tracking: a
`ThirdOrderAssistedBilinearLF` carrier loop filter combining the PLL and FLL
discriminators. Bandwidths default to `nothing` (auto).
"""
function ConventionalAssistedPLLAndDLL(
    ::Type{CO} = SecondOrderBilinearLF;
    carrier_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    code_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
) where {CO<:AbstractLoopFilter}
    ConventionalPLLAndDLL(
        ThirdOrderAssistedBilinearLF,
        CO;
        carrier_loop_filter_bandwidth,
        code_loop_filter_bandwidth,
    )
end

# Kwarg-update constructor for tweaking bandwidths in place.
function ConventionalPLLAndDLL(
    pll_and_dll::ConventionalPLLAndDLL{CA,CO};
    carrier_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    code_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
) where {CA<:AbstractLoopFilter,CO<:AbstractLoopFilter}
    ConventionalPLLAndDLL{CA,CO}(
        isnothing(carrier_loop_filter_bandwidth) ?
        pll_and_dll.carrier_loop_filter_bandwidth : carrier_loop_filter_bandwidth,
        isnothing(code_loop_filter_bandwidth) ? pll_and_dll.code_loop_filter_bandwidth :
        code_loop_filter_bandwidth,
    )
end

"""
    init_estimator_state(estimator, driver_signal, carrier_doppler, code_doppler)

Build the per-satellite estimator state for a satellite whose loop is driven
by `driver_signal` and starts at the given Dopplers. Auto bandwidths (`nothing`
on the estimator) are resolved here from the driver signal. The state records
the driver, so [`step_loop`](@ref) tells the driver's records from those of
the satellite's other signals.

This function must be **pure**: Tracking.jl also calls it to build template
states and to re-seed satellites.
"""
function init_estimator_state(
    estimator::ConventionalPLLAndDLL{CA,CO},
    driver_signal::AbstractGNSSSignal,
    carrier_doppler,
    code_doppler,
) where {CA<:AbstractLoopFilter,CO<:AbstractLoopFilter}
    SatConventionalPLLAndDLL(
        carrier_doppler,
        code_doppler,
        _constructorof(CA)(),
        _constructorof(CO)(),
        isnothing(estimator.carrier_loop_filter_bandwidth) ?
        default_carrier_loop_filter_bandwidth(driver_signal) :
        estimator.carrier_loop_filter_bandwidth,
        isnothing(estimator.code_loop_filter_bandwidth) ?
        default_code_loop_filter_bandwidth(driver_signal) :
        estimator.code_loop_filter_bandwidth,
        _signal_key(driver_signal),
    )
end

# `Accessors.constructorof` without the dependency: the loop filters are plain
# parametric structs whose zero-argument constructor is their type name.
_constructorof(::Type{T}) where {T} = Base.typename(T).wrapper

"""
    reset_estimator_state(estimator, state, carrier_doppler, code_doppler)

Zero the loop-filter integrators and re-seed the state from the converged
Dopplers, keeping the per-satellite bandwidths.
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
        state.carrier_loop_filter_bandwidth,
        state.code_loop_filter_bandwidth,
        state.driver,
    )
end

"""
    step_loop(estimator::ConventionalPLLAndDLL, state, record::LoopRecord, words, landing_sample)
        -> (state, carrier_doppler, code_doppler)

One record through the conventional loop: PLL (and FLL, for the assisted
filter) discriminators against the filtered prompt, the DLL normalised with the
code word the record ran on, both bandwidths capped by their stability products
against the record's integration time, and the Dopplers aided. `landing_sample` is ignored: the conventional loop assumes its
command acts before the next record.

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
@inline _is_driver_record(state, record::LoopRecord) =
    state.driver == 0 || _signal_key(record.signal) == state.driver

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

# The discriminators of one record, as the loop filters are fed them, and what
# the step needs around them: the per-record integration time, the capped
# carrier bandwidth, and the record's centre sample. Shared by the scalar step
# and the vector step (`_step_vector_loop`), so the two cannot drift apart.
# `phase_error` (cycles) and `frequency_error` (Hz) are what the scalar loop's
# carrier filter is fed; `raw_frequency_error` is the FLL
# discriminator as measured, against the replica that ran between the two
# prompts' centres, before any re-basing onto a landing word. The DLL output
# `code_error` is normalised with the code word the record ran on.
@inline function _record_discriminators(
    ::ConventionalPLLAndDLL,
    state::SatConventionalPLLAndDLL,
    record::LoopRecord,
    words,
    landing_sample::Int64,
)
    signal = record.signal
    integration_time = record.integrated_samples / record.sampling_frequency
    record_start = record.sample_index - record.integrated_samples
    _, applied_code = mean_nco_word(words, record_start, record.sample_index)
    filtered_correlator = record.filtered_correlator
    frequency_error =
        fll_disc(signal, filtered_correlator, record.previous_prompt, integration_time)
    (;
        integration_time,
        carrier_bandwidth = _carrier_bandwidth(state, integration_time),
        phase_error = _phase_error_in_cycles(pll_disc(signal, filtered_correlator)),
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

# The carrier bandwidth for a record that integrated for `integration_time`,
# capped against the time the record actually integrated rather than the
# intended integration length: records folded after a mid-fold sync, or the
# truncated first post-sync integration, are still short.
@inline _carrier_bandwidth(state, integration_time) =
    effective_carrier_loop_filter_bandwidth(state.carrier_loop_filter_bandwidth, integration_time)

# The carrier filter's input: the FLL-assisted filter takes both discriminators,
# any other the phase discriminator alone.
@inline _carrier_filter_input(::ThirdOrderAssistedBilinearLF, phase_error, frequency_error) =
    (phase_error, frequency_error)
@inline _carrier_filter_input(::AbstractLoopFilter, phase_error, frequency_error) =
    phase_error

# The state after one record, with both loop filters stepped.
@inline _stepped_state(state::SatConventionalPLLAndDLL, carrier_loop_filter, code_loop_filter, center) =
    SatConventionalPLLAndDLL(state; carrier_loop_filter, code_loop_filter)

# One record through a scalar loop: the carrier filter fed its discriminators,
# the code filter the DLL with its bandwidth capped by its stability product
# against the record's integration time, and the Dopplers aided.
@inline function _step_scalar_loop(estimator, state, record::LoopRecord, words, landing_sample::Int64)
    discriminators = _record_discriminators(estimator, state, record, words, landing_sample)
    integration_time = discriminators.integration_time
    code_bandwidth =
        effective_code_loop_filter_bandwidth(state.code_loop_filter_bandwidth, integration_time)
    carrier_freq_update, carrier_loop_filter = filter_loop(
        state.carrier_loop_filter,
        _carrier_filter_input(
            state.carrier_loop_filter,
            discriminators.phase_error,
            discriminators.frequency_error,
        ),
        integration_time,
        discriminators.carrier_bandwidth,
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
    _stepped_state(state, carrier_loop_filter, code_loop_filter, discriminators.center),
    carrier_doppler,
    code_doppler
end

# ── The NCO-referenced (delay-aware) PLL/DLL ─────────────────────────────────

"""
    NCOReferencedPLLAndDLL(; carrier_loop_filter_bandwidth = nothing,
                             code_loop_filter_bandwidth = nothing,
                             predict_landing = true)

FLL-assisted PLL and DLL Doppler estimator for a replica whose NCO words are
applied with a known delay — the hardware-correlator receiver's default.

It is the [`ConventionalAssistedPLLAndDLL`](@ref) — the same third-order
assisted bilinear carrier filter, the same second-order code filter, the same
gains and the same bandwidth defaults — with its loop internals referenced to
the device NCO instead of to the word the filter last computed:

 1. **Every record is attributed to the word that ran under it.** The phase
    discriminator is measured against the applied replica by construction; the
    frequency discriminator is re-based onto it too, so `applied word + FLL
    discriminator` is an *absolute* measurement of the signal's Doppler. The
    DLL is normalised with the applied code word.
 2. **The correction is sized for the moment it lands.** The filter is stepped
    with the discriminators *predicted at the landing sample of the new
    command*: the measured phase error advanced by `2π ∫ (f̂ − w(τ)) dτ` over the
    words already scheduled at the NCO, and the frequency measurement taken
    relative to the word that will be running there.

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
    carrier_loop_filter_bandwidth::Maybe{typeof(1.0Hz)}
    code_loop_filter_bandwidth::Maybe{typeof(1.0Hz)}
    predict_landing::Bool
end

function NCOReferencedPLLAndDLL(
    ::Type{CO} = SecondOrderBilinearLF;
    carrier_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    code_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    predict_landing::Bool = true,
) where {CO<:AbstractLoopFilter}
    NCOReferencedPLLAndDLL{CO}(
        carrier_loop_filter_bandwidth,
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
replica word is averaged over), and the key of the satellite's driver signal,
whose records close the loops (`0` takes every record to be the driver's).
"""
struct SatNCOReferencedPLLAndDLL{CA<:ThirdOrderAssistedBilinearLF,CO<:AbstractLoopFilter}
    init_carrier_doppler::typeof(1.0Hz)
    init_code_doppler::typeof(1.0Hz)
    carrier_loop_filter::CA
    code_loop_filter::CO
    carrier_loop_filter_bandwidth::typeof(1.0Hz)
    code_loop_filter_bandwidth::typeof(1.0Hz)
    # Device sample at the centre of the last record folded; `NaN` before the
    # first.
    previous_record_center::Float64
    driver::UInt64
end

function SatNCOReferencedPLLAndDLL(
    state::SatNCOReferencedPLLAndDLL{CA,CO};
    carrier_loop_filter::Maybe{CA} = nothing,
    code_loop_filter::Maybe{CO} = nothing,
    previous_record_center::Maybe{Float64} = nothing,
) where {CA,CO}
    SatNCOReferencedPLLAndDLL{CA,CO}(
        state.init_carrier_doppler,
        state.init_code_doppler,
        something(carrier_loop_filter, state.carrier_loop_filter),
        something(code_loop_filter, state.code_loop_filter),
        state.carrier_loop_filter_bandwidth,
        state.code_loop_filter_bandwidth,
        something(previous_record_center, state.previous_record_center),
        state.driver,
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
        something(
            estimator.carrier_loop_filter_bandwidth,
            default_carrier_loop_filter_bandwidth(driver_signal),
        ),
        something(
            estimator.code_loop_filter_bandwidth,
            default_code_loop_filter_bandwidth(driver_signal),
        ),
        NaN,
        _signal_key(driver_signal),
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
        state.carrier_loop_filter_bandwidth,
        state.code_loop_filter_bandwidth,
        NaN,
        state.driver,
    )
end

"""
    wrap_half_cycle(phase)

Fold a carrier-phase error (in radians) into `[−π/2, π/2]`, the range a
BPSK prompt — and therefore [`pll_disc`](@ref) — can tell the phase in, since
a data bit flip turns the prompt by π. Exact for any phase already inside it.
"""
wrap_half_cycle(phase) = rem(phase, π, RoundNearest)

# The phase error a record would show `shift` samples later, under the words
# the NCO will run in between: the mean phase sits at the record's centre, so
# the ramp is integrated from there. The signal's Doppler is the filter's own
# estimate, before this record's innovation.
@inline function _predict_landing_phase_error(
    phase_error,
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
    wrap_half_cycle(phase_error + 2π * shift * (f_hat - ramp_word) / sampling_freq_hz)
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
    landing_sample::Int64,
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
    phase_error = pll_disc(signal, filtered_correlator)
    raw_frequency_error =
        fll_disc(signal, filtered_correlator, record.previous_prompt, integration_time)
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
        )
        # The absolute frequency measurement, relative to the word that will be
        # running under the record `shift` samples ahead. `fll_disc` measured
        # against the word that ran between the two prompts' centres.
        fll_word =
            isnan(previous_center) ? applied_carrier :
            first(mean_nco_word(words, previous_center, center))
        landing_word = first(mean_nco_word(words, record_start + shift, record_end + shift))
        frequency_error += (fll_word - landing_word) * Hz
    end

    (;
        integration_time,
        # Bandwidths exactly as the conventional loop.
        carrier_bandwidth = _carrier_bandwidth(state, integration_time),
        # Predicted in radians, where `wrap_half_cycle` works; the filter takes cycles.
        phase_error = _phase_error_in_cycles(phase_error),
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
) = SatNCOReferencedPLLAndDLL(
    state;
    carrier_loop_filter,
    code_loop_filter,
    previous_record_center = center,
)

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
