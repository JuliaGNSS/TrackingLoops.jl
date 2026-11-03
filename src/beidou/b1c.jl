# BeiDou B1C is the modern data/pilot pair on the L1 carrier (1575.42 MHz),
# alongside GPS L1 C/A, GPS L1C and Galileo E1. Both components ride a
# 10230-chip primary code at 1.023 Mcps (10 ms period): B1C data carries B-CNAV1
# at 100 bit/s with no secondary code, B1C pilot is dataless under a per-PRN
# 1800-chip overlay (18 s cycle). The power split is 1:3 data:pilot
# (BDS-SIS-ICD-B1C-1.0 §4.2.2), which is what `get_relative_power` reports.
#
# Both components are generated as BOC(1,1). The pilot's QMBOC(6,1,4/33) puts
# its BOC(6,1) arm in quadrature, so a correlator bank on the pilot's carrier phase
# sees only the 29/33-power BOC(1,1) arm.

"""
$(SIGNATURES)

Bit-sync detector for the BeiDou B1C data component: one B-CNAV1 bit per
10 ms primary code period (100 bit/s; BDS-SIS-ICD-B1C-1.0 Table 5-1), so it
locks immediately — see `_detect_bit_is_code_period_sync`.
"""
@inline detect_sync(
    signal::BeiDouB1C_D,
    prn::Integer,
    code_period_bits::Unsigned,
    num_code_periods::Integer,
) = _detect_bit_is_code_period_sync(signal, prn, code_period_bits, num_code_periods)

"""
$(SIGNATURES)

Secondary-code sync detector for the BeiDou B1C pilot: the hard
`_detect_secondary_code_sync` over the per-PRN 1800-chip overlay
(BDS-SIS-ICD-B1C-1.0 §5.2.2) on the 10 ms primary code, an 18 s cycle — the
same shape as GPS L1C-P. Needs 1800 buffered code periods; accepts up to 45 errors
(2.5 %).
"""
function detect_sync(
    signal::BeiDouB1C_P,
    prn::Integer,        # selects the PRN's 1800-chip overlay column
    code_period_bits::UInt1800,
    num_code_periods::Integer,
)
    _detect_secondary_code_sync(signal, prn, code_period_bits, num_code_periods)
end

# BOC(1,1)-class (see header): VeryEarlyPromptLate (see `get_default_correlator_bank`).
function get_default_correlator_bank(
    ::Union{BeiDouB1C_D,BeiDouB1C_P},
    num_ants::NumAnts = NumAnts(1),
)
    VeryEarlyPromptLateCorrelatorBank(; num_ants)
end

# One bit per code period: nothing to search (see `prompt_sign_history_type`).
@inline prompt_sign_history_type(::BeiDouB1C_D) = UInt8

# Exact-width container for the 1800-chip overlay, shared with GPS L1C-P.
@inline prompt_sign_history_type(::BeiDouB1C_P) = UInt1800

# Already the default; explicit so the pilot stays on the hard sweep even if
# the 100-chip cap of `uses_soft_secondary_code_detection` is widened.
@inline uses_soft_secondary_code_detection(::BeiDouB1C_P) = false
