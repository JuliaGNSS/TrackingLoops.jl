
"""
$(SIGNATURES)

Bit-sync detector for Galileo E1B: one I/NAV bit per 4 ms
primary code period (250 bit/s; Galileo OS SIS ICD Tables 11, 15), so it locks
immediately — see `_detect_bit_is_code_period_sync`.
"""
@inline detect_sync(
    signal::GalileoE1B,
    prn::Integer,
    code_period_bits::Unsigned,
    num_code_periods::Integer,
) = _detect_bit_is_code_period_sync(signal, prn, code_period_bits, num_code_periods)

# BOC(1,1)-class (CBOC): VeryEarlyPromptLate (see `get_default_correlator_bank`).
function get_default_correlator_bank(::GalileoE1B, num_antennas::NumAntennas = NumAntennas(1))
    VeryEarlyPromptLateCorrelatorBank(; num_antennas)
end

# One bit per code period: nothing to search (see `prompt_sign_history_type`).
@inline prompt_sign_history_type(::GalileoE1B) = UInt8
