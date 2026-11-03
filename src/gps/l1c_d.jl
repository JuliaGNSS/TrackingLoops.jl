"""
$(SIGNATURES)

Bit-sync detector for GPS L1C-D: one CNAV-2 bit per 10 ms primary code
period (100 bit/s; IS-GPS-800G §3.2.3), so it locks immediately — see
`_detect_bit_is_code_period_sync`.
"""
@inline detect_sync(
    signal::GPSL1C_D,
    prn::Integer,
    code_period_bits::Unsigned,
    num_code_periods::Integer,
) = _detect_bit_is_code_period_sync(signal, prn, code_period_bits, num_code_periods)

# BOC(1,1)-class (L1C-D BOC(1,1), L1C-P TMBOC(6,1,4/33)): VeryEarlyPromptLate
# (see `get_default_correlator_bank`).
function get_default_correlator_bank(::Union{GPSL1C_D,GPSL1C_P}, num_antennas::NumAntennas = NumAntennas(1))
    VeryEarlyPromptLateCorrelatorBank(; num_antennas)
end

# One bit per code period: nothing to search (see `prompt_sign_history_type`).
@inline prompt_sign_history_type(::GPSL1C_D) = UInt8
