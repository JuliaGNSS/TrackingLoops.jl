abstract type AbstractCorrelator{M} end
abstract type AbstractEarlyPromptLateCorrelator{M} <: AbstractCorrelator{M} end

"""
$(SIGNATURES)

A single completed correlator output produced within one processing chunk.

`track!` processes each measurement in fixed-size time chunks. Every time a
signal's coherent integration completes inside a chunk, the (raw,
un-normalized) accumulator is snapshotted into a [`CorrelatorOutput`] and
appended to that signal's `correlator_outputs` array; the Doppler estimator
then folds over those records after the chunk. Storing the raw correlator plus
`integrated_samples` lets the estimator normalize it (dividing by the sample
count) and matches `last_fully_integrated_correlator`.

An **external correlator producer** (e.g. an FPGA) can build these itself and
feed them to [`apply_record`](@ref) and [`step_loop`](@ref) directly, or to
Tracking.jl's `append_correlator_output!`.

Fields:

  - `correlator`: the raw accumulated correlator at completion (not normalized).
    Fill it with the accumulator scaling `normalize` expects — the raw
    sum-of-products over `integrated_samples` — which an external producer gets
    for free by reusing [`EarlyPromptLateCorrelator`](@ref) / `update_accumulator`.
  - `integrated_samples`: samples integrated into this output (for `normalize`,
    the loop-filter `integration_time`, and the bit-buffer block count). For an
    external producer this is the true sample count of that integration.
  - `sample_index`: where this integration ended on the sample grid. As
    [`mean_nco_word`](@ref) measures spans, the record covers
    `[sample_index - integrated_samples, sample_index)`, so a word landing at
    `sample_index` belongs to the next record. (Tracking.jl's software
    correlator stores the 1-based index of the record's last sample, which is
    the same number.)
    The Doppler estimators read it to look up the replica words the record
    ran on ([`mean_nco_word`](@ref)) and, for
    [`NCOReferencedPLLAndDLL`](@ref), to place the record relative to the
    landing sample. It must therefore be on the same time grid as the `words`
    handed to [`step_loop`](@ref): device samples for an [`NCOTimeline`](@ref).
    For a [`FixedNCOWord`](@ref) the words are the same everywhere, so any
    consistent origin works — Tracking.jl's software correlator writes it
    relative to the current `track!` measurement, and an external producer
    feeding Tracking.jl must map its global counter onto that per-chunk origin
    too, so every satellite stays on one time grid for vector tracking.
"""
struct CorrelatorOutput{C<:AbstractCorrelator}
    correlator::C
    integrated_samples::Int
    sample_index::Int
end

type_for_num_ants(num_ants::NumAnts{1}) = ComplexF64
type_for_num_ants(num_ants::NumAnts{N}) where {N} = SVector{N,ComplexF64}

function get_initial_accumulator(
    num_ants::NumAnts,
    num_accumulators::NumAccumulators{M},
) where {M}
    zero(SVector{M,type_for_num_ants(num_ants)})
end

function get_initial_accumulator(num_ants::NumAnts, num_accumulators::Integer)
    [zero(type_for_num_ants(num_ants)) for i = 1:num_accumulators]
end

"""
$(SIGNATURES)

Get number of antennas from correlator
"""
get_num_ants(correlator::AbstractCorrelator{M}) where {M} = M

# The same count as a **type-stable** `NumAnts{M}`, taken off the correlator's own
# type parameter. `NumAnts(get_num_ants(c))` would route the count through a
# runtime `Int` and lose inference, which costs an allocation per record on every
# path that dispatches on it.
@inline _num_ants_val(::AbstractCorrelator{M}) where {M} = NumAnts{M}()

"""
$(SIGNATURES)

Get number of accumulators
"""
get_num_accumulators(correlator::AbstractCorrelator) = size(correlator.accumulators, 1)

"""
$(SIGNATURES)

Get all correlator accumulators
"""
get_accumulators(correlator::AbstractCorrelator) = correlator.accumulators

"""
$(SIGNATURES)

Get prompt correlator index
"""
function get_prompt_index(correlator::AbstractCorrelator)
    accumulators = get_accumulators(correlator)
    div(length(accumulators) - 1, 2) + 1
end

"""
$(SIGNATURES)

Get prompt correlator
"""
function get_prompt(correlator::AbstractCorrelator)
    get_accumulators(correlator)[get_prompt_index(correlator)]
end

function get_late_accumulator_index(correlator::AbstractEarlyPromptLateCorrelator)
    max(1, get_prompt_index(correlator) - 1)
end

function get_early_accumulator_index(correlator::AbstractEarlyPromptLateCorrelator)
    min(length(get_accumulators(correlator)), get_prompt_index(correlator) + 1)
end

"""
$(SIGNATURES)

Get early correlator
"""
function get_early(correlator::AbstractEarlyPromptLateCorrelator)
    get_accumulators(correlator)[get_early_accumulator_index(correlator)]
end

"""
$(SIGNATURES)

Get late correlator
"""
function get_late(correlator::AbstractEarlyPromptLateCorrelator)
    get_accumulators(correlator)[get_late_accumulator_index(correlator)]
end

"""
$(SIGNATURES)

Calculate the total spacing between early and late correlator in samples.
"""
function get_early_late_sample_spacing(
    correlator::AbstractEarlyPromptLateCorrelator,
    sampling_frequency,
    code_frequency,
)
    sample_shifts =
        get_correlator_sample_shifts(correlator, sampling_frequency, code_frequency)
    sample_shifts[get_early_accumulator_index(correlator)] -
    sample_shifts[get_late_accumulator_index(correlator)]
end

"""
$(SIGNATURES)

Zero the correlator
"""
function zero(correlator::AbstractCorrelator)
    update_accumulator(correlator, zero(correlator.accumulators))
end

"""
$(SIGNATURES)

Is zero correlator
"""
function is_zero(correlator::AbstractCorrelator)
    get_prompt(correlator)[1] == 0
end

"""
$(SIGNATURES)

Filter the correlator by the function `post_corr_filter`
"""
# Specialised on the filter's type (`F`): Julia does not specialise a
# bare `Function` argument on its own, and the unspecialised `map` is a dynamic
# call a trimmed binary cannot resolve.
function apply(post_corr_filter::F, correlator::AbstractCorrelator) where {F}
    update_accumulator(correlator, map(post_corr_filter, get_accumulators(correlator)))
end

"""
$(SIGNATURES)

Normalize the correlator by the number of `integrated_samples` and, optionally,
the `code_amplitude` (the RMS amplitude of the sampled code replica; see
`get_code_amplitude`). For a ±1 code `code_amplitude` is `1` and this is
just the per-sample average; for a multi-level code (CBOC) dividing by
`code_amplitude` additionally undoes the code's integer scale so the normalized
prompt is independent of modulation.
"""
function normalize(correlator::AbstractCorrelator, integrated_samples, code_amplitude = 1)
    apply(x -> x / (integrated_samples * code_amplitude), correlator)
end

function calc_preferred_code_shift_to_sample_shift(
    preferred_code_shift,
    sampling_frequency,
    code_frequency,
)
    sample_shift = round(Int, preferred_code_shift * sampling_frequency / code_frequency)
    max(1, sample_shift)
end
