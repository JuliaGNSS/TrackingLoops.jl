# Per-signal sync and correlator defaults, derived from the GNSSSignals type traits
# (`get_secondary_code_length`, `get_code_periods_per_bit`, `get_modulation`). They
# read the type-level traits, not the instance's code tables, so each folds to a
# constant per signal type. Only Galileo E5a-QP needs its own methods.

"""
$(SIGNATURES)

Hard sync detector of `signal`, called with the prompt-sign history and the number of
code periods in it when no soft detector applies ([`uses_soft_bit_edge_detection`](@ref),
[`uses_soft_secondary_code_detection`](@ref)):

  - a secondary code (`N > 1`): the rotation sweep `_detect_secondary_code_sync`
    (GPS L1C-P, BeiDou B1C-P; the soft-path signals if their soft trait is overridden);
  - one bit per code period (GPS L1C-D, L2CM, Galileo E1B, E6-B, BeiDou B2b-I, B1C-D):
    nothing to search, `found` from the first integration;
  - otherwise (GPS L2CL, a dataless pilot without overlay whose 1.5 s code period
    already is the coherent integration): never `found`.

Galileo E5a-QP (no data, no overlay) reports `found` from the first integration: the
lock only gates the switch to whole-code-cycle integration
([`max_num_code_periods_to_integrate`](@ref)).
"""
@inline function detect_sync(
    signal::AbstractGNSSSignal,
    prn::Integer,
    prompt_sign_history::Unsigned,
    num_code_periods::Integer,
)
    S = typeof(signal)
    if get_secondary_code_length(S) > 1
        _detect_secondary_code_sync(signal, prn, prompt_sign_history, num_code_periods)
    elseif get_code_periods_per_bit(S) == 1
        _detect_bit_is_code_period_sync(signal, prn, prompt_sign_history, num_code_periods)
    else
        SyncResult(false, 0, Int8(0))
    end
end

@inline detect_sync(
    signal::GalileoE5aQP,
    prn::Integer,
    prompt_sign_history::Unsigned,
    num_code_periods::Integer,
) = _detect_bit_is_code_period_sync(signal, prn, prompt_sign_history, num_code_periods)

"""
$(SIGNATURES)

Default correlator bank for a signal: [`EarlyPromptLateCorrelatorBank`](@ref) for BPSK
signals such as GPS L1 C/A, [`VeryEarlyPromptLateCorrelatorBank`](@ref) for the
BOC-modulated ones (GPS L1C, Galileo E1, BeiDou B1C). Their autocorrelation has
±0.5-chip side peaks that can capture a plain early-late discriminator; the
very-early/very-late correlators feed the VEML [`dll_disc`](@ref) that mitigates them.
"""
@inline function get_default_correlator_bank(
    signal::AbstractGNSSSignal,
    num_ants::NumAnts = NumAnts(1),
)
    if get_modulation(typeof(signal)) isa GNSSSignals.BOC
        VeryEarlyPromptLateCorrelatorBank(; num_ants)
    else
        EarlyPromptLateCorrelatorBank(; num_ants)
    end
end

"""
$(SIGNATURES)

Concrete `Unsigned` width `B` of `BitSynchronizer.prompt_sign_history` for `signal`, a
type parameter so construction is type-stable: the smallest of `UInt8` … `UInt128`,
`UInt1800` with at least `max(N, code periods per bit)` bits, so the hard detectors see a
whole secondary-code period (the 1800-chip overlay pilots get the exact-width
`UInt1800`). GPS L1 C/A, on the soft bit-edge detector, gets `UInt64`: it holds the
up to 3 pre-sync bits recovered at sync. Per-signal values: [Prompt-sign history
widths](@ref).
"""
@inline function prompt_sign_history_type(signal::AbstractGNSSSignal)
    S = typeof(signal)
    N = get_secondary_code_length(S)
    periods_per_bit = get_code_periods_per_bit(S)
    periods_per_bit > 1 && N == 1 && return UInt64
    _unsigned_with_bits(max(N, periods_per_bit))
end

@inline _unsigned_with_bits(n::Int) =
    n <= 8 ? UInt8 :
    n <= 16 ? UInt16 :
    n <= 32 ? UInt32 :
    n <= 64 ? UInt64 :
    n <= 128 ? UInt128 :
    n <= 1800 ? UInt1800 : throw(ArgumentError("no prompt-sign history type for $n bits"))

# Galileo E5a-QP: one whole 31 × 330-chip code cycle (OS SIS ICD v2.2 Table 15), not the
# single 64.5 µs code period of the generic ceiling, since it has no bit and no overlay.
@inline max_num_code_periods_to_integrate(::GalileoE5aQP) = 31
@inline default_num_code_periods_to_integrate(::GalileoE5aQP) = 31
