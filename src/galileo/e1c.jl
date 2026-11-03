
"""
$(SIGNATURES)

Hard-path secondary-code sync detector for the Galileo E1C pilot
over the shared CS25 code (Galileo OS SIS ICD Table 4; 100 ms cycle) — see
`_detect_secondary_code_sync`. The default detector is the soft one
([`uses_soft_secondary_code_detection`](@ref)).
"""
@inline function detect_sync(
    signal::GalileoE1C,
    prn::Integer,        # ignored by the reference; CS25 is shared across PRNs
    code_period_bits::Unsigned,
    num_code_periods::Integer,
)
    _detect_secondary_code_sync(signal, prn, code_period_bits, num_code_periods)
end

# BOC(1,1)-class (CBOC): VeryEarlyPromptLate (see `get_default_correlator_bank`).
function get_default_correlator_bank(::GalileoE1C, num_antennas::NumAntennas = NumAntennas(1))
    VeryEarlyPromptLateCorrelatorBank(; num_antennas)
end

# Holds one CS25 period (see `prompt_sign_history_type`).
@inline prompt_sign_history_type(::GalileoE1C) = UInt32
