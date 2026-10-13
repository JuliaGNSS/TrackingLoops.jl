# GPS L1 C/A (20 blocks per bit, no secondary code) uses the soft bit-edge
# detector `_detect_bit_edge_cfar` via the `uses_soft_bit_edge_detection`
# default, so it has no `detect_bit_or_secondary_code_sync` method.

"""
$(SIGNATURES)

Default correlator for a signal: [`EarlyPromptLateCorrelator`](@ref) for BPSK
signals such as GPS L1 C/A, [`VeryEarlyPromptLateCorrelator`](@ref) for the
BOC(1,1)-class ones (GPS L1C, Galileo E1, BeiDou B1C). Their autocorrelation has
±0.5-chip side peaks that can capture a plain early-late discriminator; the
very-early/very-late taps feed the VEML [`dll_disc`](@ref) that mitigates them.
"""
function get_default_correlator(gpsl1::GPSL1CA, num_ants::NumAnts = NumAnts(1))
    EarlyPromptLateCorrelator(; num_ants)
end

# The soft detector does not search it; at sync it yields up to 3 pre-sync bits.
@inline get_code_block_buffer_type(::GPSL1CA) = UInt64
