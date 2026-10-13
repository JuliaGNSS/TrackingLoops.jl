"""
$(SIGNATURES)

Symbol-sync detector for GPS L1C-D: one CNAV-2 symbol per 10 ms primary code
period (100 sps; IS-GPS-800G §3.2.3), so it locks immediately — see
`_detect_symbol_is_code_block_sync`.
"""
@inline detect_bit_or_secondary_code_sync(
    signal::GPSL1C_D,
    prn::Integer,
    code_block_bits::Unsigned,
    num_code_blocks::Integer,
) = _detect_symbol_is_code_block_sync(signal, prn, code_block_bits, num_code_blocks)

# BOC(1,1)-class (L1C-D BOC(1,1), L1C-P TMBOC(6,1,4/33)): VeryEarlyPromptLate
# (see `get_default_correlator`).
function get_default_correlator(::Union{GPSL1C_D,GPSL1C_P}, num_ants::NumAnts = NumAnts(1))
    VeryEarlyPromptLateCorrelator(; num_ants)
end

# One symbol per code period: nothing to search (see `get_code_block_buffer_type`).
@inline get_code_block_buffer_type(::GPSL1C_D) = UInt8
