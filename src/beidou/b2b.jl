# BeiDou B2b_I is the open-service data signal on the 1207.14 MHz carrier — the
# `E5b` band, which Galileo E5b shares exactly. A single BPSK(10) 10230-chip
# primary code at 10.23 Mcps (1 ms period), no secondary code, and the B-CNAV3
# message at 1000 bit/s, so the primary period already *is* the bit period
# (BDS-SIS-ICD-B2b-1.0 §5). No open-service quadrature counterpart, hence
# `get_relative_power` 1.0.
#
# Note the ICD defines ranging codes for PRN 6-58 only; GNSSSignals returns an
# all-zero code for any other PRN index, which correlates to nothing. That is an
# upstream property of the code table, not something the tracker screens for.

"""
$(SIGNATURES)

Bit-sync detector for BeiDou B2b_I: one B-CNAV3 bit per 1 ms primary code
period (1000 bit/s; BDS-SIS-ICD-B2b-1.0 §5), so it locks immediately — see
`_detect_bit_is_code_block_sync`.
"""
@inline detect_bit_or_secondary_code_sync(
    signal::BeiDouB2bI,
    prn::Integer,
    code_block_bits::Unsigned,
    num_code_blocks::Integer,
) = _detect_bit_is_code_block_sync(signal, prn, code_block_bits, num_code_blocks)

# BPSK: EarlyPromptLate (see `get_default_correlator`).
function get_default_correlator(::BeiDouB2bI, num_ants::NumAnts = NumAnts(1))
    EarlyPromptLateCorrelator(; num_ants)
end

# One bit per code period: nothing to search (see `get_code_block_buffer_type`).
@inline get_code_block_buffer_type(::BeiDouB2bI) = UInt8
