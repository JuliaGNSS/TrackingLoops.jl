
"""
$(SIGNATURES)

Bit-sync detector for Galileo E1B: one I/NAV bit per 4 ms
primary code period (250 bit/s; Galileo OS SIS ICD Tables 11, 15), so it locks
immediately — see `_detect_bit_is_code_period_sync`.
"""
@inline detect_bit_or_secondary_code_sync(
    signal::GalileoE1B,
    prn::Integer,
    code_block_bits::Unsigned,
    num_code_blocks::Integer,
) = _detect_bit_is_code_period_sync(signal, prn, code_block_bits, num_code_blocks)

# BOC(1,1)-class (CBOC): VeryEarlyPromptLate (see `get_default_correlator`).
function get_default_correlator(::GalileoE1B, num_ants::NumAnts = NumAnts(1))
    VeryEarlyPromptLateCorrelator(; num_ants)
end

# One bit per code period: nothing to search (see `get_code_block_buffer_type`).
@inline get_code_block_buffer_type(::GalileoE1B) = UInt8
