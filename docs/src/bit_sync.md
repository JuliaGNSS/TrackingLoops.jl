```@meta
CurrentModule = TrackingLoops
```

# Bit and secondary-code synchronisation

Until the navigation-bit boundary (for a data signal) or the secondary-code
phase (for a pilot) is known, a signal can only be integrated over one primary
code period. The [`BitBuffer`](@ref TrackingLoops.BitBuffer) collects the prompts of those short
integrations, and a per-signal detector
([`detect_bit_or_secondary_code_sync`](@ref TrackingLoops.detect_bit_or_secondary_code_sync)) searches them for the boundary.
Once [`has_bit_or_secondary_code_been_found`](@ref) returns `true`, longer
coherent integrations become available (see
[`calc_num_code_blocks_to_integrate`](@ref TrackingLoops.calc_num_code_blocks_to_integrate)) and the buffer accumulates the
decoded soft bits, which [`get_soft_bits`](@ref) returns. For a pilot, the
secondary-code sync also gives the prompt's sign, which a record's
`carrier_phase_polarity` hands to the four-quadrant carrier discriminators (see
[`LoopRecord`](@ref)).

Detectors exist for every GPS, Galileo and BeiDou signal that GNSSSignals.jl
models. How a signal is synchronised is a method of
[`detect_bit_or_secondary_code_sync`](@ref TrackingLoops.detect_bit_or_secondary_code_sync) and of the small trait functions
below ([`uses_soft_bit_edge_detection`](@ref TrackingLoops.uses_soft_bit_edge_detection),
[`get_bit_edge_or_secondary_code_tolerance`](@ref TrackingLoops.get_bit_edge_or_secondary_code_tolerance), …), so a new signal type
can be supported by adding methods for it.

```@docs
TrackingLoops.BitBuffer
TrackingLoops.PhaseAccumulators
TrackingLoops.SyncResult
TrackingLoops.get_bit_edge_detection_confidence(::AbstractGNSSSignal)
TrackingLoops.get_bit_edge_or_secondary_code_tolerance(::AbstractGNSSSignal)
TrackingLoops.get_code_block_buffer_type(::AbstractGNSSSignal)
get_soft_bits(::BitBuffer)
has_bit_or_secondary_code_been_found(::BitBuffer)
TrackingLoops.reset(::BitBuffer{B}) where {B<:Unsigned}
TrackingLoops.uses_soft_bit_edge_detection(::AbstractGNSSSignal)
TrackingLoops.uses_soft_secondary_code_detection(::AbstractGNSSSignal)
get_default_correlator(::GPSL1CA, ::NumAnts = NumAnts(1))
TrackingLoops.detect_bit_or_secondary_code_sync(::GPSL1C_D, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_bit_or_secondary_code_sync(::GPSL1C_P, ::Integer, ::UInt1800, ::Integer)
TrackingLoops.detect_bit_or_secondary_code_sync(::GPSL2CL, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_bit_or_secondary_code_sync(::GPSL2CM, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_bit_or_secondary_code_sync(::GPSL5I, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_bit_or_secondary_code_sync(::GPSL5Q, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_bit_or_secondary_code_sync(::Union{GalileoE1B, GalileoE1B_BOC11}, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_bit_or_secondary_code_sync(::Union{GalileoE1C, GalileoE1C_BOC11}, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_bit_or_secondary_code_sync(::GalileoE5aI, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_bit_or_secondary_code_sync(::GalileoE5aQ, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_bit_or_secondary_code_sync(::GalileoE5aQP, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_bit_or_secondary_code_sync(::GalileoE5bI, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_bit_or_secondary_code_sync(::GalileoE5bQ, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_bit_or_secondary_code_sync(::GalileoE6B, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_bit_or_secondary_code_sync(::GalileoE6C, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_bit_or_secondary_code_sync(::BeiDouB1I, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_bit_or_secondary_code_sync(::BeiDouB3I, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_bit_or_secondary_code_sync(::BeiDouB2aI, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_bit_or_secondary_code_sync(::BeiDouB2aQ, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_bit_or_secondary_code_sync(::BeiDouB2bI, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_bit_or_secondary_code_sync(::BeiDouB1C_D, ::Integer, ::Unsigned, ::Integer)
TrackingLoops.detect_bit_or_secondary_code_sync(::BeiDouB1C_P, ::Integer, ::UInt1800, ::Integer)
```

## Code-block buffer widths

[`get_code_block_buffer_type`](@ref TrackingLoops.get_code_block_buffer_type) per signal, and what the packed buffer of that
width is for:

| Signal        | Returns    | Detector / buffer role                              |
|:------------- |:---------- |:--------------------------------------------------- |
| GPS L1 C/A    | `UInt64`   | soft bit-edge CFAR — holds up to 3 pre-sync bits    |
| Galileo E1B   | `UInt8`    | symbol = primary period, buffer unused              |
| GPS L5I       | `UInt32`   | soft secondary CFAR — packed buffer vestigial       |
| GPS L5Q       | `UInt32`   | soft secondary CFAR — packed buffer vestigial       |
| GPS L1C-D     | `UInt8`    | symbol = primary period, buffer unused              |
| GPS L1C-P     | `UInt1800` | **hard** rotation sweep — 1800-chip overlay horizon |
| GPS L2CM      | `UInt8`    | symbol = primary period, buffer unused              |
| GPS L2CL      | `UInt8`    | dataless pilot, no sync, buffer unused              |
| Galileo E1C   | `UInt32`   | soft secondary CFAR — packed buffer vestigial       |
| Galileo E5a-I | `UInt32`   | soft secondary CFAR — packed buffer vestigial       |
| Galileo E5a-Q | `UInt128`  | soft secondary CFAR — packed buffer vestigial       |
| Galileo E5a-QP| `UInt8`    | no data, no overlay, buffer unused                  |
| Galileo E5b-I | `UInt32`   | soft secondary CFAR — packed buffer vestigial       |
| Galileo E5b-Q | `UInt128`  | soft secondary CFAR — packed buffer vestigial       |
| Galileo E6-B  | `UInt8`    | symbol = primary period, buffer unused              |
| Galileo E6-C  | `UInt128`  | soft secondary CFAR — packed buffer vestigial       |
| BeiDou B1I    | `UInt32`   | soft secondary CFAR — packed buffer vestigial       |
| BeiDou B3I    | `UInt32`   | soft secondary CFAR — packed buffer vestigial       |
| BeiDou B2a-I  | `UInt32`   | soft secondary CFAR — packed buffer vestigial       |
| BeiDou B2a-Q  | `UInt128`  | soft secondary CFAR — packed buffer vestigial       |
| BeiDou B2b-I  | `UInt8`    | symbol = primary period, buffer unused              |
| BeiDou B1C-D  | `UInt8`    | symbol = primary period, buffer unused              |
| BeiDou B1C-P  | `UInt1800` | **hard** rotation sweep — 1800-chip overlay horizon |

The soft detectors read the incremental [`PhaseAccumulators`](@ref TrackingLoops.PhaseAccumulators) instead of the
packed buffer. GPS L1 C/A still recovers the pre-sync bits it holds once sync is
found; the secondary-coded signals do not, as their pre-sync signs carry the
overlay, not data.

## BeiDou GEO satellites

BDS-SIS-ICD-B1I-3.0 §5.2.1 applies the 20-bit Neuman-Hoffman overlay only on the
MEO/IGSO satellites (PRN 6-58), which broadcast the D1 message at 50 sym/s: one
NH20 period is one data symbol. The GEO satellites (PRN 1-5 and 59-63) broadcast
D2 at 500 sym/s and carry no overlay; GNSSSignals models that with all-ones
columns of a per-PRN secondary code, so B1I and B3I report a 20-chip secondary code
for every PRN and use the secondary-code detector, not the bit-edge one.

On a GEO PRN that search never completes: an all-ones reference is
rotation-invariant, so the 20 rotation hypotheses differ only in how their 20-block
bins straddle the data, and with 2-block D2 symbols every bin averages about ten
random symbols and none stands out. The soft CFAR detector needs a peak that beats
its runner-up, so it declines to lock. A GEO satellite therefore tracks and can be
ranged on, but stays in the one-block pre-sync integration and decodes no bits. It
takes both the missing overlay and the D2 rate: without an overlay at the D1 rate the
bins would reduce to GPS L1 C/A's bit-edge search and lock on the symbol boundary
(`test/signals.jl` tests both cases). Decoding a GEO satellite needs a per-PRN data
rate upstream: `get_data_frequency` is per signal type and reports the D1 50 Hz for
every PRN.
