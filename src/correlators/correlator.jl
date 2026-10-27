"""
    AbstractCorrelator{M}

Abstract supertype for the accumulators of one integration, for `M` antennas.
A subtype holds one complex accumulator per tap (and antenna) and answers
[`get_accumulators`](@ref), [`get_prompt`](@ref), [`get_num_ants`](@ref),
[`get_correlator_sample_shifts`](@ref) and the other accessors.
"""
abstract type AbstractCorrelator{M} end

"""
    AbstractEarlyPromptLateCorrelator{M} <: AbstractCorrelator{M}

Abstract supertype for correlators with early, prompt and late taps
([`get_early`](@ref), [`get_prompt`](@ref), [`get_late`](@ref)), e.g.
[`EarlyPromptLateCorrelator`](@ref) and [`VeryEarlyPromptLateCorrelator`](@ref).
"""
abstract type AbstractEarlyPromptLateCorrelator{M} <: AbstractCorrelator{M} end

"""
$(SIGNATURES)

One completed coherent integration (a *record*): the raw accumulator snapshotted
when the integration completes. Tracking.jl's `track!` produces these; an external
producer (e.g. an FPGA) can build them and feed [`apply_record`](@ref) and
[`step_loop`](@ref), or Tracking.jl's `append_correlator_output!`.

Fields:

  - `correlator`: the raw (un-normalized) sum of products, as `normalize` expects;
    an external producer gets it by reusing `update_accumulator` on an
    [`EarlyPromptLateCorrelator`](@ref).
  - `integrated_samples`: samples integrated (for `normalize`, the integration time
    and the bit-buffer block count).
  - `sample_index`: where the integration ended; the record covers
    `[sample_index - integrated_samples, sample_index)` (numerically Tracking.jl's
    1-based index of the record's last sample). Used to look up the words the
    replica ran on ([`mean_nco_word`](@ref)) and, for
    [`NCOReferencedPLLAndDLL`](@ref), to place the record against the landing
    sample, so it must share the time grid of the `words` passed to
    [`step_loop`](@ref): device samples for an [`NCOTimeline`](@ref). With a
    [`FixedNCOWord`](@ref) any origin works as long as all satellites share it
    (vector tracking); Tracking.jl uses the current `track!` measurement, so an
    external producer feeding it must map its counter onto that origin.
  - `code_phase`: the replica's code phase (chips) at `sample_index`, read by
    [`VectorPLLAndDLL`](@ref) since the end sample pins it only to one sample. Only
    the part past the nearest code-block boundary is used, so any wrap convention
    works. `NaN` when not reported (three-argument constructor); only vector
    tracking needs it.
"""
struct CorrelatorOutput{C<:AbstractCorrelator}
    correlator::C
    integrated_samples::Int
    sample_index::Int
    code_phase::Float64
end

CorrelatorOutput(correlator::AbstractCorrelator, integrated_samples::Integer, sample_index::Integer) =
    CorrelatorOutput(correlator, Int(integrated_samples), Int(sample_index), NaN)

type_for_num_ants(num_ants::NumAnts{1}) = ComplexF64
type_for_num_ants(num_ants::NumAnts{N}) where {N} = SVector{N,ComplexF64}

"""
    get_initial_accumulator(num_ants::NumAnts,
                            num_accumulators::NumAccumulators{M}) -> SVector{M}
    get_initial_accumulator(num_ants::NumAnts, num_accumulators::Integer) -> Vector

The zeroed taps a correlator starts from: `num_accumulators` of them, each a
`ComplexF64` for one antenna or an `SVector{N,ComplexF64}` for `N` antennas. With
[`NumAccumulators`](@ref) the tap count is a type parameter and the taps an `SVector`,
as a custom correlator's keyword constructor builds them; with an integer they are a
`Vector`.
"""
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

# Type-stable `NumAnts{M}`; `NumAnts(get_num_ants(c))` would allocate per record.
@inline NumAnts(::AbstractCorrelator{M}) where {M} = NumAnts{M}()

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

# `where {F}` forces specialisation; an unspecialised `map` is a dynamic call a
# trimmed binary cannot resolve.
"""
$(SIGNATURES)

Apply `post_corr_filter` to every accumulator of `correlator`, returning a correlator of
the same type: the post-correlation filter's step, or any other per-tap map such as
negating the taps.
"""
function apply(post_corr_filter::F, correlator::AbstractCorrelator) where {F}
    update_accumulator(correlator, map(post_corr_filter, get_accumulators(correlator)))
end

"""
$(SIGNATURES)

Normalize the correlator by `integrated_samples` times `code_amplitude` (the RMS
amplitude of the sampled code replica, `get_code_amplitude`; `1` for a ±1 code), so
the normalized prompt is independent of modulation (e.g. CBOC).
"""
function normalize(correlator::AbstractCorrelator, integrated_samples, code_amplitude = 1)
    apply(x -> x / (integrated_samples * code_amplitude), correlator)
end

"""
    calc_preferred_code_shift_to_sample_shift(preferred_code_shift, sampling_frequency,
                                              code_frequency) -> Int

The whole number of samples closest to `preferred_code_shift` chips at
`sampling_frequency` and `code_frequency`, and at least one, so that two taps never
coincide. A correlator's [`get_correlator_sample_shifts`](@ref) places its taps with it.
"""
function calc_preferred_code_shift_to_sample_shift(
    preferred_code_shift,
    sampling_frequency,
    code_frequency,
)
    sample_shift = round(Int, preferred_code_shift * sampling_frequency / code_frequency)
    max(1, sample_shift)
end
