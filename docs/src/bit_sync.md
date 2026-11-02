```@meta
CurrentModule = TrackingLoops
```

# Bit and secondary-code synchronisation

Until the navigation-bit boundary (for a data signal) or the secondary-code
phase (for a pilot) is known, a signal can only be integrated over one primary
code period. The [`BitSynchronizer`](@ref TrackingLoops.BitSynchronizer) collects the prompts of those short
integrations, and a per-signal detector
([`detect_sync`](@ref TrackingLoops.detect_sync)) searches them for the boundary.
Once [`is_bit_or_secondary_code_synced`](@ref) returns `true`, longer
coherent integrations become available (see
[`calc_num_code_periods_to_integrate`](@ref TrackingLoops.calc_num_code_periods_to_integrate)) and the bit synchronizer accumulates the
demodulated soft bits, which [`get_soft_bits`](@ref) returns. For a pilot, the
secondary-code sync also gives the prompt's sign, which an integration's
`carrier_phase_polarity` hands to the four-quadrant carrier discriminators (see
[`CoherentIntegration`](@ref)).

Detectors exist for every GPS, Galileo and BeiDou signal that GNSSSignals.jl
models. How a signal is synchronised is a method of
[`detect_sync`](@ref TrackingLoops.detect_sync) and of the small trait functions
below ([`uses_soft_bit_edge_detection`](@ref TrackingLoops.uses_soft_bit_edge_detection),
[`get_secondary_code_error_tolerance`](@ref TrackingLoops.get_secondary_code_error_tolerance), …), so a new signal type
can be supported by adding methods for it.

```@docs
TrackingLoops.BitSynchronizer
TrackingLoops.SyncHypothesisAccumulators
TrackingLoops.SyncResult
TrackingLoops.get_sync_detection_confidence(::AbstractGNSSSignal)
TrackingLoops.get_secondary_code_error_tolerance(::AbstractGNSSSignal)
TrackingLoops.prompt_sign_history_type(::AbstractGNSSSignal)
get_soft_bits(::BitSynchronizer)
is_bit_or_secondary_code_synced(::BitSynchronizer)
drain_soft_bits!
TrackingLoops.uses_soft_bit_edge_detection(::AbstractGNSSSignal)
TrackingLoops.uses_soft_secondary_code_detection(::AbstractGNSSSignal)
get_default_correlator_bank(::GPSL1CA, ::NumAnts = NumAnts(1))
TrackingLoops.detect_sync(::GPSL1C_D, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_sync(::GPSL1C_P, ::Integer, ::UInt1800, ::Integer)
TrackingLoops.detect_sync(::GPSL2CL, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_sync(::GPSL2CM, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_sync(::GPSL5I, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_sync(::GPSL5Q, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_sync(::GalileoE1B, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_sync(::GalileoE1C, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_sync(::GalileoE5aI, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_sync(::GalileoE5aQ, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_sync(::GalileoE5aQP, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_sync(::GalileoE5bI, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_sync(::GalileoE5bQ, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_sync(::GalileoE6B, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_sync(::GalileoE6C, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_sync(::BeiDouB1I, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_sync(::BeiDouB3I, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_sync(::BeiDouB2aI, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_sync(::BeiDouB2aQ, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_sync(::BeiDouB2bI, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_sync(::BeiDouB1C_D, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_sync(::BeiDouB1C_P, ::Integer, ::UInt1800, ::Integer)
```

## Prompt-sign history widths

[`prompt_sign_history_type`](@ref TrackingLoops.prompt_sign_history_type) per signal, and what the prompt-sign history of that
width is for:

| Signal        | Returns    | Detector / history role                             |
|:------------- |:---------- |:--------------------------------------------------- |
| GPS L1 C/A    | `UInt64`   | soft bit-edge CFAR — holds up to 3 pre-sync bits    |
| Galileo E1B   | `UInt8`    | bit = primary period, history unused                |
| GPS L5I       | `UInt32`   | soft secondary CFAR — sign history vestigial        |
| GPS L5Q       | `UInt32`   | soft secondary CFAR — sign history vestigial        |
| GPS L1C-D     | `UInt8`    | bit = primary period, history unused                |
| GPS L1C-P     | `UInt1800` | **hard** rotation sweep — 1800-chip overlay horizon |
| GPS L2CM      | `UInt8`    | bit = primary period, history unused                |
| GPS L2CL      | `UInt8`    | dataless pilot, no sync, history unused             |
| Galileo E1C   | `UInt32`   | soft secondary CFAR — sign history vestigial        |
| Galileo E5a-I | `UInt32`   | soft secondary CFAR — sign history vestigial        |
| Galileo E5a-Q | `UInt128`  | soft secondary CFAR — sign history vestigial        |
| Galileo E5a-QP| `UInt8`    | no data, no overlay, history unused                 |
| Galileo E5b-I | `UInt32`   | soft secondary CFAR — sign history vestigial        |
| Galileo E5b-Q | `UInt128`  | soft secondary CFAR — sign history vestigial        |
| Galileo E6-B  | `UInt8`    | bit = primary period, history unused                |
| Galileo E6-C  | `UInt128`  | soft secondary CFAR — sign history vestigial        |
| BeiDou B1I    | `UInt32`   | soft secondary CFAR — sign history vestigial        |
| BeiDou B3I    | `UInt32`   | soft secondary CFAR — sign history vestigial        |
| BeiDou B2a-I  | `UInt32`   | soft secondary CFAR — sign history vestigial        |
| BeiDou B2a-Q  | `UInt128`  | soft secondary CFAR — sign history vestigial        |
| BeiDou B2b-I  | `UInt8`    | bit = primary period, history unused                |
| BeiDou B1C-D  | `UInt8`    | bit = primary period, history unused                |
| BeiDou B1C-P  | `UInt1800` | **hard** rotation sweep — 1800-chip overlay horizon |

The soft detectors read the incremental [`SyncHypothesisAccumulators`](@ref TrackingLoops.SyncHypothesisAccumulators) instead of the
prompt-sign history. GPS L1 C/A still recovers the pre-sync bits it holds once sync is
found; the secondary-coded signals do not, as their pre-sync signs carry the
overlay, not data.

## BeiDou GEO satellites

BDS-SIS-ICD-B1I-3.0 §5.2.1 applies the 20-bit Neuman-Hoffman overlay only on the
MEO/IGSO satellites (PRN 6-58), which broadcast the D1 message at 50 bit/s: one
NH20 period is one data bit. The GEO satellites (PRN 1-5 and 59-63) broadcast
D2 at 500 bit/s and carry no overlay.

TrackingLoops uses the signal-wide secondary-code length and bit grid for B1I
and B3I. The stored secondary code has an all-ones column for GEO PRNs, which is
rotation-invariant. With two-period D2 bits, the 20-period search bins average
several bits and no phase hypothesis consistently beats its competitors. The
secondary-code detector therefore stays pre-sync: GEO satellites remain in
single-period tracking and their navigation bits are not decoded.
