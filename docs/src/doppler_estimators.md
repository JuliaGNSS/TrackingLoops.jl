# Doppler estimators and loop filters

A Doppler estimator closes the carrier and code loops. Its configuration — loop
filter types and bandwidths — is a plain value, a subtype of
[`AbstractDopplerEstimator`](@ref); the per-satellite state it advances is
created with [`init_estimator_state`](@ref). Every estimator is stepped with the
same call,

```julia
state, carrier_doppler, code_doppler = step_loop(estimator, state, record, words, landing_sample)
```

where `record` is a [`LoopRecord`](@ref), `words` is the replica frequency the
record was produced under — a [`FixedNCOWord`](@ref) for a software correlator,
the channel's [`NCOTimeline`](@ref) for a hardware one — and `landing_sample`
is the sample the new command takes effect at, or [`NO_LANDING_SAMPLE`](@ref)
for "at the end of each record".

- [`ConventionalPLLAndDLL`](@ref) — a PLL and a carrier-aided DLL.
- [`ConventionalAssistedPLLAndDLL`](@ref) — the same, with the PLL assisted by
  an FLL, which pulls in larger initial frequency errors. This is the default
  in Tracking.jl.
- [`NCOReferencedPLLAndDLL`](@ref) — stays stable when the correction it
  computes only takes effect several records later, as it does with a hardware
  NCO. With a fixed word and no landing sample it is the conventional loop.

The bandwidth rules ([`default_carrier_loop_filter_bandwidth`](@ref),
[`effective_code_loop_filter_bandwidth`](@ref), …) keep the loops stable for a
given integration time, and the integration-length rules
([`calc_num_code_blocks_to_integrate`](@ref), …) say how long a signal may be
integrated coherently once its bit or secondary code has been found.

## The per-record fold

[`apply_record`](@ref) advances one signal component's
[`SignalLoopState`](@ref) — its prompt filter, C/N₀ estimator and bit buffer —
by one record, so every caller does it the same way.

```@autodocs
Modules = [TrackingLoops]
Pages = ["record.jl"]
Private = false
```

## Estimators

```@autodocs
Modules = [TrackingLoops]
Pages = ["estimators.jl"]
Private = false
```

## Loop-filter bandwidths

```@autodocs
Modules = [TrackingLoops]
Pages = ["loop_filters.jl"]
Private = false
```

## Integration length

```@autodocs
Modules = [TrackingLoops]
Pages = ["sample_parameters.jl"]
Private = false
```
