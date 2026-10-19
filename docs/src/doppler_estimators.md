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

A host steps every record of a satellite, of every signal it tracks. The
estimator state remembers the driver it was initialised with: the driver's
records close the loops, and the scalar loops leave any other record — a
passenger, such as the data component of a pilot + data pair — to the host,
returning the state unchanged and the command in force.

- [`ConventionalPLLAndDLL`](@ref) — a PLL and a carrier-aided DLL.
- [`ConventionalAssistedPLLAndDLL`](@ref) — the same, with the PLL assisted by
  an FLL, which pulls in larger initial frequency errors, until phase lock (see
  [Carrier-loop staging](@ref)). This is the default in Tracking.jl.
- [`NCOReferencedPLLAndDLL`](@ref) — stays stable when the correction it
  computes only takes effect several records later, as it does with a hardware
  NCO. With a fixed word and no landing sample it is the conventional loop.

The bandwidth rules ([`default_wide_carrier_loop_filter_bandwidth`](@ref),
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

## Two- and four-quadrant discriminators

The carrier discriminators are two-quadrant (Costas) by default: they read a
prompt and its negation alike, so a data bit or secondary-code chip flipping
the prompt's sign does not disturb them, over ±1/4 cycle for the PLL and
±1/(4T) for the FLL. Where the replica wipes every sign modulation off the
prompt — a dataless signal synced to its secondary code — the prompt's sign is
known, and the estimators switch to the four-quadrant discriminators, over ±1/2
cycle and ±1/(2T), worth up to 6 dB of tracking threshold. The record says so
with its `carrier_phase_polarity`, the prompt's sign, which
[`apply_record`](@ref) computes into the [`SignalLoopState`](@ref) and a
[`LoopRecord`](@ref) built from that state carries, for the scalar loops and the
vector loop alike. A record whose carrier phase polarity differs from the
previous record's comes without a previous prompt, so the four-quadrant FLL
never compares prompts across the switch.

GPS L1C-P and L5Q, Galileo E1C, E5a-Q, E5b-Q and E6-C, and BeiDou B1C-P and
B2a-Q switch once their secondary code is synced. Data signals never do, nor,
for now, do the pilots without a secondary code (GPS L2 CL, Galileo E5a-QP),
whose sync reads no sign.
The NCO-referenced loop folds its predicted phase error into whichever range
the record's PLL reads.

The sign rests on a contract with whoever correlates the records: until the
sync, the records of these signals
([`is_dataless_with_secondary_code`](@ref)) are primary-only, correlated with
a replica that carries no secondary code. A replica that carries the same
secondary chip on every block is fine once the host multiplies the records by
that chip (Tracking.jl's carries chip 0, and multiplies by it). An external
producer, a hardware correlator say, must keep to this as well. If its
pre-sync replica carries a negative chip it does not multiply out, the sync
reads the wrong sign, and on the switch the four-quadrant PLL pulls the
carrier phase over by half a cycle.

## Estimators

```@autodocs
Modules = [TrackingLoops]
Pages = ["estimators.jl"]
Private = false
```

## Carrier-loop staging

The scalar loops stage their carrier loop on phase lock
([`CarrierLoopStage`](@ref)):

  - **FLL-assisted PLL** (`FLL_ASSISTED_PLL`) at the wide PLL bandwidth, its FLL
    path at its own FLL-assist bandwidth, until the phase-lock indicator has read
    lock for one time constant of its averages;
  - **Wide PLL** (`WIDE_PLL`): the FLL dropped, until the indicator has read
    lock for another four time constants;
  - **Narrow PLL** (`NARROW_PLL`) at the narrow bandwidth.

A satellite in the vector loop reports **vector loop** (`VECTOR_LOOP`): its
carrier NCO is the navigation filter's, and the vector loop's residual PLL
follows the carrier (see [Vector tracking](vector_tracking.md)).

The phase-lock indicator ([`phase_lock_indicator`](@ref)) is `⟨I² − Q²⟩ / A²`,
an estimate of `cos 2φ`, from exponential averages of the driver's prompts over
0.1 s or 25 records, whichever is longer. The signal power `A²` comes from the
prompts' moments, `√(2 M₂² − M₄)`, which the noise cancels from, so a locked loop
does not read low at a low C/N₀. It reads lock at 0.5, an RMS phase error of
about 30°. Dropping the FLL feeds the FLL-assisted filter a zero FLL input, which
is exactly the third-order PLL, so the switch is free, and the FLL discriminator
is no longer formed.

The stages only advance: a loop that loses lock keeps the narrow PLL. A
satellite's release from the vector loop starts them over, so the scalar loop
that takes over pulls in afresh. [`reset_estimator_state`](@ref) keeps them: it
is the handoff to a new integration length, after which the re-seeded loop
holds its lock at the stage it has reached, its bandwidths capped for the new
length. While the vector loop holds a satellite the navigation filter sets its
carrier and a residual PLL follows it (see [The carrier in the vector
loop](@ref vector-carrier)); its scalar loop takes over again on release.

A wider loop tolerates less command delay: at 50 Hz the conventional loop holds
through three records of delay. The [`NCOReferencedPLLAndDLL`](@ref), meant for
hardware correlators whose commands land records later, therefore pulls in at
the narrow 18 Hz by default.

```@autodocs
Modules = [TrackingLoops]
Pages = ["carrier_loop_staging.jl"]
Private = false
```

## Loop-filter bandwidths

The PLL pulls in at a wide 50 Hz and narrows to 18 Hz once phase lock has held;
the FLL path of the FLL-assisted filter runs at 5 Hz while it is in use, and the
code bandwidth is a flat 1 Hz, all for every signal. All are one-sided noise
bandwidths `BL` in the sense of Kaplan & Hegarty, so they plug into the usual
jitter and dynamic-stress formulas: the carrier filter is fed the phase error in
cycles and the FLL error in Hz. 18 Hz is the third-order PLL bandwidth of the
literature (Kaplan & Hegarty Table 5.6, Pany Table 3.3; Borre et al. quote about
20 Hz), none of which scales it with the primary code period: thermal jitter
and dynamic stress, which set the bandwidth, do not depend on it.

### Stability cap

Stability does depend on the loop update interval `Δt`, so at filter time each
bandwidth is capped against the record's actual integration time
([`effective_carrier_loop_filter_bandwidth`](@ref),
[`effective_fll_assist_loop_filter_bandwidth`](@ref),
[`effective_code_loop_filter_bandwidth`](@ref)). An explicit bandwidth is
capped the same way; the cap only ever narrows a loop.

```julia
BL_pll  = min(BL_pll_configured,  0.09  / Δt)   # wide and narrow alike
BL_fll  = min(BL_fll_configured,  0.02  / Δt)
BL_code = min(BL_code_configured, 0.018 / Δt)
```

The default FLL-assisted third-order carrier filter diverges at `BL · Δt ≈ 0.4`
(the plain third-order one at ≈ 0.43). Below that, the loop's actual noise
bandwidth runs wider than configured:

| `BL · Δt`                      | 0.018 | 0.036 | 0.072 | 0.09  | 0.18  | 0.36 |
|--------------------------------|------:|------:|------:|------:|------:|-----:|
| `ThirdOrderAssistedBilinearLF` | 1.05× | 1.09× | 1.19× | 1.25× | 1.65× | 5.2× |
| `ThirdOrderBilinearLF`         | 0.97× | 1.00× | 1.07× | 1.12× | 1.38× | 2.8× |

The PLL cap at 0.09 keeps the loop within 25 % of its configured bandwidth
with about 4× stability margin, at the wide and the narrow bandwidth alike. It
is the product of Kaplan & Hegarty's third-order design example (18 Hz at 5 ms)
and matches GNSS-SDR's narrow post-sync bandwidths (5 Hz at 20 ms). A tighter
cap on the narrow bandwidth would buy little: at 20 ms and 30 dB-Hz the thermal
jitter is about 4° at 4.5 Hz and 2.6° at 2 Hz, while a TCXO's jitter (Allan
deviation 1e-10, GPS L1) grows from about 6° to 13° and the jerk the loop
tolerates falls with `BL³`. From 10 ms on, the cap sets wide and narrow alike.
The resulting defaults:

| Integration | Signals                                                   |  Wide BL | Narrow BL | FLL-assist BL |  Code BL |
|-------------|-----------------------------------------------------------|---------:|----------:|--------------:|---------:|
| 1 ms        | GPS L1 C/A, GPS L5, Galileo E5a, …                        |    50 Hz |     18 Hz |          5 Hz |     1 Hz |
| 2 ms        | Galileo E5a-QP                                            |    45 Hz |     18 Hz |          5 Hz |     1 Hz |
| 4 ms        | Galileo E1B / E1C                                         |  22.5 Hz |     18 Hz |          5 Hz |     1 Hz |
| 10 ms       | GPS L1C-D / L1C-P, BeiDou B1C; GPS L5I synced at 10 ms    |     9 Hz |      9 Hz |          2 Hz |     1 Hz |
| 20 ms       | GPS L2 CM; GPS L1 C/A, L5Q, Galileo E5a-I synced at 20 ms |   4.5 Hz |    4.5 Hz |          1 Hz |   0.9 Hz |
| 1.5 s       | GPS L2 CL                                                 |  0.06 Hz |   0.06 Hz |      0.013 Hz | 0.012 Hz |

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
