
"""
$(SIGNATURES)

Hard-path secondary-code sync detector for the Galileo E1C pilot
over the shared CS25 code (Galileo OS SIS ICD Table 4; 100 ms cycle) — see
`_detect_secondary_code_sync`. The default detector is the soft one
([`uses_soft_secondary_code_detection`](@ref)).
"""
@inline function detect_bit_or_secondary_code_sync(
    signal::GalileoE1C,
    prn::Integer,        # ignored by the reference; CS25 is shared across PRNs
    code_block_bits::Unsigned,
    num_code_blocks::Integer,
)
    _detect_secondary_code_sync(signal, prn, code_block_bits, num_code_blocks)
end

# BOC(1,1)-class (CBOC): VeryEarlyPromptLate (see `get_default_correlator_bank`).
function get_default_correlator_bank(::GalileoE1C, num_ants::NumAnts = NumAnts(1))
    VeryEarlyPromptLateCorrelatorBank(; num_ants)
end

# Holds one CS25 period (see `get_code_block_buffer_type`).
@inline get_code_block_buffer_type(::GalileoE1C) = UInt32
