"""
$(SIGNATURES)

Secondary-code sync detector for GPS L1C-P: the hard
`_detect_secondary_code_sync` over the per-PRN 1800-chip overlay
(IS-GPS-800G §3.2.2.1.2) on the 10 ms primary code, an 18 s cycle. Needs 1800
buffered code periods; accepts up to 45 errors (2.5 %).
"""
function detect_sync(
    signal::GPSL1C_P,
    prn::Integer,
    code_period_bits::UInt1800,
    num_code_periods::Integer,
)
    _detect_secondary_code_sync(signal, prn, code_period_bits, num_code_periods)
end

# The generic `_packed_secondary_code` rebuilds the reference on every call
# until lock; not cached, since the 1800-phase sweep dominates its cost.

# `get_default_correlator_bank(::GPSL1C_P)` is defined with GPS L1C-D in `l1c_d.jl`.

# Exact-width container for the 1800-chip overlay search
# (`BitIntegers.@define_integers 1800` in `src/TrackingLoops.jl`).
@inline prompt_sign_history_type(::GPSL1C_P) = UInt1800

# Already the default (see `uses_soft_secondary_code_detection`); explicit so
# L1C-P stays on the hard sweep even if that default's 100-chip cap is widened.
@inline uses_soft_secondary_code_detection(::GPSL1C_P) = false
