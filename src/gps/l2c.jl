# GPS L2C is a time-multiplexed pair of codes sharing the L2 carrier
# (1227.60 MHz): the moderate-length L2CM code carries the CNAV data
# stream, the very long L2CL code is a dataless pilot. They are tracked as
# two independent signals (`GPSL2CM`, `GPSL2CL`), same as the L1C data/pilot
# split.

"""
$(SIGNATURES)

Bit-sync detector for GPS L2CM: one CNAV bit per 20 ms code period
(50 bit/s; IS-GPS-200N §3.2.2), so it locks immediately — see
`_detect_bit_is_code_block_sync`.
"""
@inline detect_bit_or_secondary_code_sync(
    signal::GPSL2CM,
    prn::Integer,
    code_block_bits::Unsigned,
    num_code_blocks::Integer,
) = _detect_bit_is_code_block_sync(signal, prn, code_block_bits, num_code_blocks)

"""
$(SIGNATURES)

"Sync" detector for GPS L2CL — a no-op that never reports `found`. L2CL is a
dataless pilot without overlay whose 767250-chip (1.5 s) code period already is
the coherent integration, so there is nothing to lock.
"""
@inline detect_bit_or_secondary_code_sync(::GPSL2CL, ::Integer, ::Unsigned, ::Integer) =
    SyncResult(false, 0, Int8(0))

# BPSK: EarlyPromptLate (see `get_default_correlator`).
function get_default_correlator(::Union{GPSL2CM,GPSL2CL}, num_ants::NumAnts = NumAnts(1))
    EarlyPromptLateCorrelator(; num_ants)
end

# One bit per code period: nothing to search (see `get_code_block_buffer_type`).
@inline get_code_block_buffer_type(::GPSL2CM) = UInt8
# L2CL: no sync feature at all.
@inline get_code_block_buffer_type(::GPSL2CL) = UInt8
