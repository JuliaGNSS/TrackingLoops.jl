# BeiDou B3I is the legacy open-service signal on the 1268.52 MHz carrier
# (`B3`): a single BPSK 10230-chip primary code at 10.23 Mcps (1 ms period),
# again with no open-service quadrature counterpart (`get_relative_power` 1.0).
#
# Structurally B1I at five times the chipping rate, with the same NH20 overlay
# on MEO/IGSO and none on GEO (BDS-SIS-ICD-B3I-1.0 §5.2.1); the `b1i.jl` header
# applies verbatim.

"""
$(SIGNATURES)

Hard-path secondary-code sync detector for BeiDou B3I over the per-PRN NH20
overlay — see `_detect_secondary_code_sync`; behaves as
`BeiDouB1I`'s. The default detector is the soft one
([`uses_soft_secondary_code_detection`](@ref)).
"""
@inline function detect_sync(
    signal::BeiDouB3I,
    prn::Integer,        # selects the PRN's NH20 column (all-ones on GEO)
    code_period_bits::Unsigned,
    num_code_periods::Integer,
)
    _detect_secondary_code_sync(signal, prn, code_period_bits, num_code_periods)
end

# BPSK: EarlyPromptLate (see `get_default_correlator_bank`).
function get_default_correlator_bank(::BeiDouB3I, num_antennas::NumAntennas = NumAntennas(1))
    EarlyPromptLateCorrelatorBank(; num_antennas)
end

# Holds one NH20 period (see `prompt_sign_history_type`).
@inline prompt_sign_history_type(::BeiDouB3I) = UInt32
