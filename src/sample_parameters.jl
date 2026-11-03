"""
$(SIGNATURES)

Longest coherent integration, in primary code periods, that `signal`'s own
structure allows — the ceiling `calc_num_code_periods_to_integrate` clamps a
satellite's preferred integration length against.

One full bit: the data-bit period for data-bearing signals, or the
secondary-code period for pilots (a bit rate of `0 Hz`, e.g. GPS L1C-P).
Integrating past it would straddle a bit boundary and average two opposite
signs away.

The default degenerates to a single code period for a signal that has neither a
multi-period data bit nor an overlay; Galileo E5a-QP overrides it to a whole
31-period code cycle (see `galileo/e5a_qp.jl`). GPS L2CL is the same shape and
deliberately keeps the default: its 1.5 s primary period is already a whole
coherent integration.
"""
@inline max_num_code_periods_to_integrate(signal::AbstractGNSSSignal) =
    max(get_secondary_code_length(signal), get_code_periods_per_bit(signal))

"""
$(SIGNATURES)

Coherent integration length, in primary code periods, a freshly tracked signal
starts at. One code period for every signal but Galileo E5a-QP, whose 64.5 µs code period is
too short to run a loop on and which starts at a whole 31-period code cycle.
"""
@inline default_num_code_periods_to_integrate(::AbstractGNSSSignal) = 1

"""
$(SIGNATURES)

Returns the appropriate number of code periods to integrate. It will be just a
single code period as long as the secondary code or bit hasn't been found. Once
found, the coherent integration is capped by
[`max_num_code_periods_to_integrate`](@ref) and clamped to the largest divisor
of that ceiling not exceeding the preferred value, so an integration never
straddles a bit boundary.
"""
function calc_num_code_periods_to_integrate(
    signal::AbstractGNSSSignal,
    preferred_num_code_periods::Int,
    bit_or_secondary_code_synced::Bool,
)
    bit_or_secondary_code_synced || return 1
    num_code_periods_that_form_a_bit = max_num_code_periods_to_integrate(signal)
    num_code_periods =
        clamp(preferred_num_code_periods, 1, num_code_periods_that_form_a_bit)
    while num_code_periods_that_form_a_bit % num_code_periods != 0
        num_code_periods -= 1
    end
    num_code_periods
end

"""
$(SIGNATURES)

Number of primary code periods to credit the bit synchronizer's bit sum with for a
just-completed integration: always 1 before bit/secondary sync (the detectors
shift exactly one prompt sign per call), and afterwards the whole code periods the
integration actually covered, recovered from its sample count.
"""
@inline function calc_num_bit_synchronizer_code_periods(
    signal::AbstractGNSSSignal,
    num_integrated_samples::Integer,
    sampling_frequency,
    bit_or_secondary_code_synced::Bool,
)
    bit_or_secondary_code_synced || return 1
    round(
        Int,
        num_integrated_samples * get_chip_rate(signal) /
        (get_code_length(signal) * sampling_frequency),
    )
end
