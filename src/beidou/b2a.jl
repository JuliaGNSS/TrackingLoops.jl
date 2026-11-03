# BeiDou B2a is a data/pilot pair on the 1176.45 MHz carrier — the `L5` band it
# shares with GPS L5 and Galileo E5a. Both components are BPSK(10) 10230-chip
# primary codes at 10.23 Mcps (1 ms period), split 50/50 in power: the in-phase
# B2a data channel carries B-CNAV2 at 200 bit/s under a 5-chip shared secondary
# code, the quadrature B2a pilot is dataless under a per-PRN 100-chip secondary
# code (BDS-SIS-ICD-B2a-1.0 §5.2). They are tracked as two independent signals,
# the same split as GPS L5 and Galileo E5a/E5b.

"""
$(SIGNATURES)

Hard-path secondary-code sync detector for the BeiDou B2a data component over
the shared 5-chip code (`00010`; BDS-SIS-ICD-B2a-1.0 §5.2.1) — see
`_detect_secondary_code_sync`. One period is exactly one 200 bit/s
B-CNAV2 bit. The default detector is the soft one
([`uses_soft_secondary_code_detection`](@ref)).
"""
@inline function detect_sync(
    signal::BeiDouB2aI,
    prn::Integer,        # ignored by the reference; the 5-chip code is shared
    code_period_bits::Unsigned,
    num_code_periods::Integer,
)
    _detect_secondary_code_sync(signal, prn, code_period_bits, num_code_periods)
end

"""
$(SIGNATURES)

Hard-path secondary-code sync detector for the BeiDou B2a pilot over the
per-PRN 100-chip code (BDS-SIS-ICD-B2a-1.0 §5.2.1 Table 5-4; 100 ms cycle) —
see `_detect_secondary_code_sync`. The default detector is the soft one
([`uses_soft_secondary_code_detection`](@ref)).
"""
@inline function detect_sync(
    signal::BeiDouB2aQ,
    prn::Integer,        # selects the PRN's secondary-code column
    code_period_bits::Unsigned,
    num_code_periods::Integer,
)
    _detect_secondary_code_sync(signal, prn, code_period_bits, num_code_periods)
end

# BPSK: EarlyPromptLate (see `get_default_correlator_bank`).
function get_default_correlator_bank(
    ::Union{BeiDouB2aI,BeiDouB2aQ},
    num_ants::NumAnts = NumAnts(1),
)
    EarlyPromptLateCorrelatorBank(; num_ants)
end

# Hold one 5-chip / 100-chip period (see `prompt_sign_history_type`).
@inline prompt_sign_history_type(::BeiDouB2aI) = UInt32
@inline prompt_sign_history_type(::BeiDouB2aQ) = UInt128
