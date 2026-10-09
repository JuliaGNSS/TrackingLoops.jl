# Bit and secondary-code synchronisation

Until the navigation-bit boundary (for a data signal) or the secondary-code
phase (for a pilot) is known, a signal can only be integrated over one primary
code period. The [`BitBuffer`](@ref) collects the prompts of those short
integrations, and a per-signal detector
([`detect_bit_or_secondary_code_sync`](@ref)) searches them for the boundary.
Once [`has_bit_or_secondary_code_been_found`](@ref) returns `true`, longer
coherent integrations become available (see
[`calc_num_code_blocks_to_integrate`](@ref)) and the buffer accumulates the
decoded soft bits, which [`get_soft_bits`](@ref) returns.

Detectors exist for every GPS, Galileo and BeiDou signal that GNSSSignals.jl
models. How a signal is synchronised is a method of
[`detect_bit_or_secondary_code_sync`](@ref) and of the small trait functions
below ([`uses_soft_bit_edge_detection`](@ref),
[`get_bit_edge_or_secondary_code_tolerance`](@ref), …), so a new signal type
can be supported by adding methods for it.

```@autodocs
Modules = [TrackingLoops]
Pages = [
    "bit_buffer.jl",
    "gps/l1ca.jl", "gps/l1c_d.jl", "gps/l1c_p.jl", "gps/l2c.jl", "gps/l5.jl",
    "galileo/e1b.jl", "galileo/e1c.jl", "galileo/e5a.jl", "galileo/e5a_qp.jl",
    "galileo/e5b.jl", "galileo/e6.jl",
    "beidou/b1i.jl", "beidou/b3i.jl", "beidou/b2a.jl", "beidou/b2b.jl", "beidou/b1c.jl",
]
Private = false
```

## Code-block buffer widths

[`get_code_block_buffer_type`](@ref) per signal, and what the packed buffer of that
width is for:

| Signal        | Returns    | Detector / buffer role                              |
|:------------- |:---------- |:--------------------------------------------------- |
| GPS L1 C/A    | `UInt64`   | soft bit-edge CFAR — packed buffer vestigial        |
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

The soft detectors read the incremental [`PhaseAccumulators`](@ref) instead of the
packed buffer, so for them the buffer is built but not consulted; its width is kept
at the detector's horizon so the hard path stays available. Any other signal gets
the `UInt64` default.

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
ranged on, but stays in the one-block pre-sync integration and decodes no bits. Both
facts rule the lock out together; with no overlay at the D1 rate, the bins would
reduce to the bit-edge search GPS L1 C/A uses and lock on the symbol boundary
(`test/beidou_b1i.jl` pins both). Decoding a GEO satellite needs a per-PRN data
rate upstream: `get_data_frequency` is per signal type and reports the D1 50 Hz for
every PRN.

