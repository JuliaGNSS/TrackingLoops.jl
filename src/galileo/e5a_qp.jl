# Galileo E5a-QP is the E5a quasi-pilot acquisition aid (OS SIS ICD v2.2
# §2.3.1.4): dataless BPSK(5) at 5.115 Mcps on `L5`, a 330-chip (64.5 µs)
# primary code with no overlay; it integrates a whole 31-block code cycle by
# default (see below; JuliaGNSS/Tracking.jl#236). It runs at half the E5a-I/Q
# chip rate, so it cannot share their Tracking.jl `SignalGroup`
# (JuliaGNSS/Tracking.jl#151), and the ICD gives no carrier phase relation to
# E5a-I (`get_carrier_phase_offset` is the 0.0 default). As an aid, a 330-chip
# search is 31× cheaper than E5a-I's; the Doppler transfers and the code phase pins
# E5a-I modulo 64.5 µs. The ICD defines codes for PRN 1-40 only, and only part of
# the constellation transmits it: upstream properties, not screened for here.

"""
$(SIGNATURES)

"Sync" detector for Galileo E5a-QP — reports `found = true` from the very first
integration via `_detect_symbol_is_code_block_sync`.

With no data and no overlay every block boundary is equivalent; the lock only
gates the switch from single-block pre-sync integration to the whole-cycle one
([`max_num_code_blocks_to_integrate`](@ref)).
"""
@inline detect_bit_or_secondary_code_sync(
    signal::GalileoE5aQP,
    prn::Integer,
    code_block_bits::Unsigned,
    num_code_blocks::Integer,
) = _detect_symbol_is_code_block_sync(signal, prn, code_block_bits, num_code_blocks)

# BPSK: EarlyPromptLate (see `get_default_correlator`).
function get_default_correlator(::GalileoE5aQP, num_ants::NumAnts = NumAnts(1))
    EarlyPromptLateCorrelator(; num_ants)
end

# Nothing to search (see `get_code_block_buffer_type`).
@inline get_code_block_buffer_type(::GalileoE5aQP) = UInt8

# One whole code cycle: 31 × 330 = 10230 chips = 2 ms (OS SIS ICD v2.2
# Table 15). The generic ceiling (data bit or secondary-code period) would be a
# single block, since E5a-QP has neither.
@inline max_num_code_blocks_to_integrate(::GalileoE5aQP) = 31
# Also the default: a single 64.5 µs block (15.5 kHz loop rate) is unusable.
@inline default_num_code_blocks_to_integrate(::GalileoE5aQP) = 31
