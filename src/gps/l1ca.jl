# GPS L1 C/A (20 code periods per bit, no secondary code) uses the soft bit-edge
# detector `_detect_bit_edge_cfar` via the `uses_soft_bit_edge_detection`
# default, so it has no `detect_sync` method.

"""
$(SIGNATURES)

Default correlator bank for a signal: [`EarlyPromptLateCorrelatorBank`](@ref) for BPSK
signals such as GPS L1 C/A, [`VeryEarlyPromptLateCorrelatorBank`](@ref) for the
BOC(1,1)-class ones (GPS L1C, Galileo E1, BeiDou B1C). Their autocorrelation has
±0.5-chip side peaks that can capture a plain early-late discriminator; the
very-early/very-late correlators feed the VEML [`dll_disc`](@ref) that mitigates them.
"""
function get_default_correlator_bank(gpsl1::GPSL1CA, num_ants::NumAnts = NumAnts(1))
    EarlyPromptLateCorrelatorBank(; num_ants)
end

# The soft detector does not search it; at sync it yields up to 3 pre-sync bits.
@inline prompt_sign_history_type(::GPSL1CA) = UInt64
