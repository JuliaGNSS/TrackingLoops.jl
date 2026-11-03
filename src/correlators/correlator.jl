"""
    AbstractCorrelatorBank{M}

Abstract supertype for the correlator banks of one integration, for `M` antennas.
A subtype holds one complex correlator output per correlator (and antenna) and answers
[`get_correlator_outputs`](@ref), [`get_prompt`](@ref), [`get_num_ants`](@ref),
[`calc_correlator_sample_offsets`](@ref) and the other accessors.
"""
abstract type AbstractCorrelatorBank{M} end

"""
    AbstractEarlyPromptLateCorrelatorBank{M} <: AbstractCorrelatorBank{M}

Abstract supertype for correlator banks with early, prompt and late correlators
([`get_early`](@ref), [`get_prompt`](@ref), [`get_late`](@ref)), e.g.
[`EarlyPromptLateCorrelatorBank`](@ref) and [`VeryEarlyPromptLateCorrelatorBank`](@ref).
"""
abstract type AbstractEarlyPromptLateCorrelatorBank{M} <: AbstractCorrelatorBank{M} end

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
    replica ran on ([`calc_mean_nco_command`](@ref)) and, for
    [`NCOReferencedPLLAndDLL`](@ref), to place the record against the landing
    sample, so it must share the time grid of the `words` passed to
    [`step_loop`](@ref): device samples for an [`NCOTimeline`](@ref). With a
    [`ConstantNCOCommand`](@ref) any origin works as long as all satellites share it
    (vector tracking); Tracking.jl uses the current `track!` measurement, so an
    external producer feeding it must map its counter onto that origin.
  - `code_phase`: the replica's code phase (chips) at `sample_index`, read by
    [`VectorPLLAndDLL`](@ref) since the end sample pins it only to one sample. Only
    the part past the nearest code-block boundary is used, so any wrap convention
    works. `NaN` when not reported (three-argument constructor); only vector
    tracking needs it.
"""
struct CorrelatorOutput{C<:AbstractCorrelatorBank}
    correlator_bank::C
    num_integrated_samples::Int
    sample_index::Int
    code_phase::Float64
end

CorrelatorOutput(correlator_bank::AbstractCorrelatorBank, num_integrated_samples::Integer, sample_index::Integer) =
    CorrelatorOutput(correlator_bank, Int(num_integrated_samples), Int(sample_index), NaN)

correlator_output_type(num_ants::NumAnts{1}) = ComplexF64
correlator_output_type(num_ants::NumAnts{M}) where {M} = SVector{M,ComplexF64}

"""
    zero_correlator_outputs(num_ants::NumAnts,
                            num_correlators::NumCorrelators{K}) -> SVector{K}
    zero_correlator_outputs(num_ants::NumAnts, num_correlators::Integer) -> Vector

The zeroed correlator outputs a correlator bank starts from: `num_correlators` of them, each a
`ComplexF64` for one antenna or an `SVector{M,ComplexF64}` for `M` antennas. With
[`NumCorrelators`](@ref) the correlator count is a type parameter and the correlator outputs an `SVector`,
as a custom correlator bank's keyword constructor builds them; with an integer they are a
`Vector`.
"""
function zero_correlator_outputs(
    num_ants::NumAnts,
    num_correlators::NumCorrelators{K},
) where {K}
    zero(SVector{K,correlator_output_type(num_ants)})
end

function zero_correlator_outputs(num_ants::NumAnts, num_correlators::Integer)
    [zero(correlator_output_type(num_ants)) for i = 1:num_correlators]
end

"""
$(SIGNATURES)

Get number of antennas from correlator bank
"""
get_num_ants(correlator_bank::AbstractCorrelatorBank{M}) where {M} = M

# Type-stable `NumAnts{M}`; `NumAnts(get_num_ants(c))` would allocate per record.
@inline NumAnts(::AbstractCorrelatorBank{M}) where {M} = NumAnts{M}()

"""
$(SIGNATURES)

Get number of correlators
"""
get_num_correlators(correlator_bank::AbstractCorrelatorBank) = size(correlator_bank.correlator_outputs, 1)

"""
$(SIGNATURES)

Get all correlator outputs
"""
get_correlator_outputs(correlator_bank::AbstractCorrelatorBank) = correlator_bank.correlator_outputs

"""
$(SIGNATURES)

Get prompt correlator index
"""
function get_prompt_index(correlator_bank::AbstractCorrelatorBank)
    correlator_outputs = get_correlator_outputs(correlator_bank)
    div(length(correlator_outputs) - 1, 2) + 1
end

"""
$(SIGNATURES)

Get prompt correlator output
"""
function get_prompt(correlator_bank::AbstractCorrelatorBank)
    get_correlator_outputs(correlator_bank)[get_prompt_index(correlator_bank)]
end

function get_late_index(correlator_bank::AbstractEarlyPromptLateCorrelatorBank)
    max(1, get_prompt_index(correlator_bank) - 1)
end

function get_early_index(correlator_bank::AbstractEarlyPromptLateCorrelatorBank)
    min(length(get_correlator_outputs(correlator_bank)), get_prompt_index(correlator_bank) + 1)
end

"""
$(SIGNATURES)

Get early correlator output
"""
function get_early(correlator_bank::AbstractEarlyPromptLateCorrelatorBank)
    get_correlator_outputs(correlator_bank)[get_early_index(correlator_bank)]
end

"""
$(SIGNATURES)

Get late correlator output
"""
function get_late(correlator_bank::AbstractEarlyPromptLateCorrelatorBank)
    get_correlator_outputs(correlator_bank)[get_late_index(correlator_bank)]
end

"""
$(SIGNATURES)

Calculate the total spacing between early and late correlator in samples.
"""
function calc_early_late_sample_spacing(
    correlator_bank::AbstractEarlyPromptLateCorrelatorBank,
    sampling_frequency,
    code_frequency,
)
    sample_shifts =
        calc_correlator_sample_offsets(correlator_bank, sampling_frequency, code_frequency)
    sample_shifts[get_early_index(correlator_bank)] -
    sample_shifts[get_late_index(correlator_bank)]
end

"""
$(SIGNATURES)

Zero the correlator bank
"""
function zero(correlator_bank::AbstractCorrelatorBank)
    set_correlator_outputs(correlator_bank, zero(correlator_bank.correlator_outputs))
end

# `where {F}` forces specialisation; an unspecialised `map` is a dynamic call a
# trimmed binary cannot resolve.
"""
$(SIGNATURES)

Apply `post_corr_filter` to every correlator output of `correlator_bank`, returning a correlator bank of
the same type: the post-correlation filter's step, or any other per-correlator-output map such as
negating the correlator outputs.
"""
function apply(post_corr_filter::F, correlator_bank::AbstractCorrelatorBank) where {F}
    set_correlator_outputs(correlator_bank, map(post_corr_filter, get_correlator_outputs(correlator_bank)))
end

"""
$(SIGNATURES)

Normalize the correlator bank by `num_integrated_samples` times `code_amplitude` (the RMS
amplitude of the sampled code replica, `get_code_amplitude`; `1` for a ±1 code), so
the normalized prompt is independent of modulation (e.g. CBOC).
"""
function normalize(correlator_bank::AbstractCorrelatorBank, num_integrated_samples, code_amplitude = 1)
    apply(x -> x / (num_integrated_samples * code_amplitude), correlator_bank)
end

"""
    calc_correlator_sample_offset(preferred_code_shift, sampling_frequency,
                                              code_frequency) -> Int

The whole number of samples closest to `preferred_code_shift` chips at
`sampling_frequency` and `code_frequency`, and at least one, so that two correlators never
coincide. A correlator bank's [`calc_correlator_sample_offsets`](@ref) places its correlators with it.
"""
function calc_correlator_sample_offset(
    preferred_code_shift,
    sampling_frequency,
    code_frequency,
)
    sample_shift = round(Int, preferred_code_shift * sampling_frequency / code_frequency)
    max(1, sample_shift)
end
