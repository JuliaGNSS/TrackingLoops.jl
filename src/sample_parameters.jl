"""
$(SIGNATURES)

Longest coherent integration, in primary code blocks, that `signal`'s own
structure allows — the ceiling `calc_num_code_blocks_to_integrate` clamps a
satellite's preferred integration length against.

One full symbol: the data-bit period for data-bearing signals, or the
secondary-code period for pilots (`data_frequency == 0`, e.g. GPS L1C-P).
Integrating past it would straddle a symbol boundary and average two opposite
signs away.

The default degenerates to a single block for a signal that has neither a
multi-block data bit nor an overlay; Galileo E5a-QP overrides it to a whole
31-block code cycle (see `galileo/e5a_qp.jl`). GPS L2CL is the same shape and
deliberately keeps the default: its 1.5 s primary period is already a whole
coherent integration.
"""
@inline function max_num_code_blocks_to_integrate(signal::AbstractGNSSSignal)
    data_freq = get_data_frequency(signal)
    iszero(data_freq) ? get_secondary_code_length(signal) :
    Int(get_code_frequency(signal) / (get_code_length(signal) * data_freq))
end

"""
$(SIGNATURES)

Coherent integration length, in primary code blocks, a freshly tracked signal
starts at. One block for every signal but Galileo E5a-QP, whose 64.5 µs block is
too short to run a loop on and which starts at a whole 31-block code cycle.
"""
@inline default_num_code_blocks_to_integrate(::AbstractGNSSSignal) = 1

"""
$(SIGNATURES)

Returns the appropriate number of code blocks to integrate. It will be just a
single code block as long as the secondary code or bit hasn't been found. Once
found, the coherent integration is capped by
[`max_num_code_blocks_to_integrate`](@ref) and clamped to the largest divisor
of that ceiling not exceeding the preferred value, so an integration never
straddles a symbol boundary.
"""
function calc_num_code_blocks_to_integrate(
    signal::AbstractGNSSSignal,
    preferred_num_code_blocks::Int,
    secondary_code_or_bit_found::Bool,
)
    secondary_code_or_bit_found || return 1
    num_code_blocks_that_form_a_symbol = max_num_code_blocks_to_integrate(signal)
    num_code_blocks =
        clamp(preferred_num_code_blocks, 1, num_code_blocks_that_form_a_symbol)
    while num_code_blocks_that_form_a_symbol % num_code_blocks != 0
        num_code_blocks -= 1
    end
    num_code_blocks
end

"""
$(SIGNATURES)

Number of primary code blocks to credit the bit-buffer accumulator with for a
just-completed integration: always 1 before bit/secondary sync (the detectors
shift exactly one prompt sign per call), and afterwards the whole blocks the
record actually covered, recovered from its sample count.
"""
@inline function calc_num_code_blocks_for_bit_buffer(
    signal::AbstractGNSSSignal,
    integrated_samples::Integer,
    sampling_frequency,
    secondary_code_or_bit_found::Bool,
)
    secondary_code_or_bit_found || return 1
    round(
        Int,
        integrated_samples * get_code_frequency(signal) /
        (get_code_length(signal) * sampling_frequency),
    )
end
