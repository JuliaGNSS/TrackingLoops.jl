```@meta
CurrentModule = TrackingLoops
```

# NCO timeline

A hardware correlator holds each NCO word until the next one lands, often
several records after the record that motivated it. An [`NCOTimeline`](@ref)
keeps track of the word in effect and the words scheduled at named device
samples, so that a record can be attributed to the word it was really produced
under ([`mean_nco_word`](@ref)) rather than the one that was last requested.
Pass it as the `words` argument of [`step_loop`](@ref). A software correlator,
whose replica follows every command at once, passes a [`FixedNCOWord`](@ref)
instead.

```@docs
NO_LANDING_SAMPLE
FixedNCOWord
NCOTimeline
mean_nco_word(::NCOTimeline, ::Real, ::Real)
nco_word_at(::NCOTimeline, ::Any)
promote_words!(::NCOTimeline, ::Any)
reschedule_word!(::NCOTimeline, ::Any, ::Any)
reset_timeline!(::NCOTimeline, ::Any, ::Any)
schedule_word!(::NCOTimeline, ::Any, ::Any, ::Any)
TrackingLoops.scheduled_words(::NCOTimeline)
TrackingLoops.word_changes_within(::NCOTimeline, ::Any, ::Any)
```
