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
[`effective_carrier_loop_filter_bandwidth`](@ref), …) keep the loops stable for a
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

The carrier bandwidth defaults to a flat 18 Hz and the code bandwidth to a flat
1 Hz for every signal. Both are one-sided noise bandwidths `BL` in the sense of
Kaplan & Hegarty, so they plug into the usual PLL jitter and dynamic-stress
formulas: the carrier filter is fed the phase error in cycles and the FLL error
in Hz. 18 Hz is the third-order PLL bandwidth of the literature (Kaplan &
Hegarty Table 5.6, Pany Table 3.3; Borre et al. quote about 20 Hz), none of
which scales it with the primary code period: thermal jitter and dynamic
stress, which set the bandwidth, do not depend on it.

### Stability cap

Stability does depend on the loop update interval `Δt`, so at filter time each
bandwidth is capped against the record's actual integration time
([`effective_carrier_loop_filter_bandwidth`](@ref),
[`effective_code_loop_filter_bandwidth`](@ref)). An explicit bandwidth is
capped the same way; the cap only ever narrows a loop.

```julia
BL_carrier = min(BL_carrier_configured, 0.09  / Δt)
BL_code    = min(BL_code_configured,    0.018 / Δt)
```

The default FLL-assisted third-order carrier filter diverges at `BL · Δt ≈ 0.4`
(the plain third-order one at ≈ 0.43). Below that, the loop's actual noise
bandwidth runs wider than configured:

| `BL · Δt`                      | 0.018 | 0.036 | 0.072 | 0.09  | 0.18  | 0.36 |
|--------------------------------|------:|------:|------:|------:|------:|-----:|
| `ThirdOrderAssistedBilinearLF` | 1.05× | 1.09× | 1.19× | 1.25× | 1.65× | 5.2× |
| `ThirdOrderBilinearLF`         | 0.97× | 1.00× | 1.07× | 1.12× | 1.38× | 2.8× |

The carrier cap at 0.09 keeps the loop within 25 % of its configured bandwidth
with about 4× stability margin. It is the product of Kaplan & Hegarty's
third-order design example (18 Hz at 5 ms) and matches GNSS-SDR's narrow
post-sync bandwidths (5 Hz at 20 ms). The resulting defaults:

| Integration | Signals                                                   | Carrier BL |  Code BL |
|-------------|-----------------------------------------------------------|-----------:|---------:|
| 1 ms        | GPS L1 C/A, GPS L5, Galileo E5a, …                        |      18 Hz |     1 Hz |
| 2 ms        | Galileo E5a-QP                                            |      18 Hz |     1 Hz |
| 4 ms        | Galileo E1B / E1C                                         |      18 Hz |     1 Hz |
| 10 ms       | GPS L1C-D / L1C-P, BeiDou B1C; GPS L5I synced at 10 ms    |       9 Hz |     1 Hz |
| 20 ms       | GPS L2 CM; GPS L1 C/A, L5Q, Galileo E5a-I synced at 20 ms |     4.5 Hz |   0.9 Hz |
| 1.5 s       | GPS L2 CL                                                 |    0.06 Hz | 0.012 Hz |

The code cap of 0.018 is conservative for the second-order code filter, which
destabilizes only around `BL · Δt ≈ 0.4` (S. A. Stephens and J. B. Thomas,
"Controlled-Root Formulation for Digital Phase-Locked Loops", IEEE Trans.
Aerospace and Electronic Systems 31(1), 1995). Carrier-aided, the DLL has
almost no dynamics to track, so its 1 Hz is a thermal-noise-versus-pull-in
choice, applied in full up to 18 ms.

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
