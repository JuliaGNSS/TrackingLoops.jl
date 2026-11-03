"""
$(SIGNATURES)

Hard-path secondary-code sync detector for Galileo E5a-I over the shared CS20
code (Galileo OS SIS ICD Table 19) — see `_detect_secondary_code_sync`. One CS20
period is exactly one 50 bit/s F/NAV bit. The default detector is the soft one
([`uses_soft_secondary_code_detection`](@ref)).
"""
@inline function detect_sync(
    signal::GalileoE5aI,
    prn::Integer,        # ignored by the reference; CS20 is shared across PRNs
    code_period_bits::Unsigned,
    num_code_periods::Integer,
)
    _detect_secondary_code_sync(signal, prn, code_period_bits, num_code_periods)
end

"""
$(SIGNATURES)

Hard-path secondary-code sync detector for the Galileo E5a-Q pilot over the
per-PRN CS100 code (Galileo OS SIS ICD Table 20; 100 ms cycle) — see
`_detect_secondary_code_sync`. The default detector is the soft one
([`uses_soft_secondary_code_detection`](@ref)).
"""
@inline function detect_sync(
    signal::GalileoE5aQ,
    prn::Integer,        # selects the PRN's CS100 column in the reference
    code_period_bits::Unsigned,
    num_code_periods::Integer,
)
    _detect_secondary_code_sync(signal, prn, code_period_bits, num_code_periods)
end

# BPSK: EarlyPromptLate (see `get_default_correlator_bank`).
function get_default_correlator_bank(
    ::Union{GalileoE5aI,GalileoE5aQ},
    num_ants::NumAnts = NumAnts(1),
)
    EarlyPromptLateCorrelatorBank(; num_ants)
end

# Hold one CS20 / CS100 period (see `prompt_sign_history_type`).
@inline prompt_sign_history_type(::GalileoE5aI) = UInt32
@inline prompt_sign_history_type(::GalileoE5aQ) = UInt128
