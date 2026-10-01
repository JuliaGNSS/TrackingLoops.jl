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
    "galileo/e1b.jl", "galileo/e1c.jl", "galileo/e5a.jl", "galileo/e5a_qp.jl", "galileo/e5b.jl", "galileo/e6.jl",
    "beidou/b1i.jl", "beidou/b3i.jl", "beidou/b2a.jl", "beidou/b2b.jl", "beidou/b1c.jl",
]
Private = false
```
