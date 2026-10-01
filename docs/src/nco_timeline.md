# NCO timeline

A hardware correlator holds each NCO word until the next one lands, often
several records after the record that motivated it. An [`NCOTimeline`](@ref)
keeps track of the word in effect and the words scheduled at named device
samples, so that a record can be attributed to the word it was really produced
under ([`mean_nco_word`](@ref)) rather than the one that was last requested.
Pass it as the `words` argument of [`step_loop`](@ref). A software correlator,
whose replica follows every command at once, passes a [`FixedNCOWord`](@ref)
instead.

```@autodocs
Modules = [TrackingLoops]
Pages = ["nco_timeline.jl"]
Private = false
```
