# Doppler estimators and loop filters

A Doppler estimator closes the carrier and code loops. Its configuration (loop
filter types and bandwidths) is a plain value, a subtype of
[`AbstractDopplerEstimator`](@ref); the per-satellite state it advances is
created with [`init_estimator_state`](@ref). Every estimator is stepped with the
same call,

```julia
state, carrier_doppler, code_doppler =
    step_loop(estimator, state, record, words, landing_sample)
```

where `record` is a [`LoopRecord`](@ref), `words` is the replica frequency the
record was produced under (a [`FixedNCOWord`](@ref) for a software correlator,
the channel's [`NCOTimeline`](@ref) for a hardware one) and `landing_sample` is
the sample the new command takes effect at, or [`NO_LANDING_SAMPLE`](@ref) for
"at the end of each record".

- [`ConventionalPLLAndDLL`](@ref): a PLL and a carrier-aided DLL.
- [`ConventionalAssistedPLLAndDLL`](@ref): the same with an FLL-assisted PLL,
  which pulls in larger initial frequency errors. The default in Tracking.jl.
- [`NCOReferencedPLLAndDLL`](@ref): stays stable when its correction takes
  effect only several records later, as with a hardware NCO. With a fixed word
  and no landing sample it is the conventional loop at the same bandwidths.

The bandwidth rules ([`default_wide_carrier_loop_filter_bandwidth`](@ref),
[`effective_code_loop_filter_bandwidth`](@ref), …) keep the loops stable for a
given integration time; the integration-length rules
([`calc_num_code_blocks_to_integrate`](@ref), …) say how long a signal may be
integrated coherently once its bit or secondary code has been found.

## The per-record fold

[`apply_record`](@ref) advances one signal component's
[`SignalLoopState`](@ref) (its prompt filter, C/N₀ estimator and bit buffer) by
one record, so every caller does it the same way.

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

The carrier loop has one bandwidth per stage (see [Carrier loop staging](@ref)):
50 Hz while it pulls in, 18 Hz once phase lock has held, and 5 Hz for the FLL
path of the FLL-assisted filter. The code loop defaults to 1 Hz. All are one-sided noise
bandwidths `BL` (Kaplan & Hegarty), so they plug into the usual PLL jitter and
dynamic-stress formulas: the carrier filter is fed the phase error in cycles and
the FLL error in Hz, and takes the PLL and FLL bandwidths as a pair
(`ω₀f = BL_FLL / 0.53`).

- 18 Hz is the literature's third-order PLL bandwidth (Kaplan & Hegarty
  Table 5.6, Pany Table 3.3; Borre et al. quote about 20 Hz).
- 50 Hz reaches phase lock about as fast as Tracking v8's loop, whose PLL ran
  about 5.6× wider than configured.
- 5 Hz pulls a handover error of up to a quarter of the two-quadrant FLL range
  in without becoming the 1 ms loop's noise source (sample-level simulation,
  single signals at 25–45 dB-Hz).

Before JuliaGNSS/Tracking.jl#244 the carrier filter was fed radians, a loop gain 2π too
high (a configured 18 Hz behaved like about 100 Hz). Bandwidths tuned then, such
as the old default `BL = 0.018 / T`, are about 5–6× narrower than the loops they
were tuned on.

### Stability cap

Stability depends on the loop update interval `Δt`, so at filter time every
bandwidth, default or explicit, is capped against the record's integration time
(for the code loop by [`effective_code_loop_filter_bandwidth`](@ref)). The cap
only ever narrows a loop.

```julia
BL_carrier  = min(BL_carrier_configured,  0.09  / Δt)   # pull-in
BL_tracking = min(BL_tracking_configured, 0.04  / Δt)   # after phase lock
BL_FLL      = min(BL_FLL_configured,      0.02  / Δt)
BL_code     = min(BL_code_configured,     0.018 / Δt)
```

The default FLL-assisted third-order carrier filter diverges at `BL · Δt ≈ 0.4`
(the plain third-order one at ≈ 0.43). Below that, the loop's actual noise
bandwidth runs wider than configured:

| `BL · Δt`                      | 0.018 | 0.036 | 0.072 | 0.09  | 0.18  | 0.36 |
|--------------------------------|------:|------:|------:|------:|------:|-----:|
| `ThirdOrderAssistedBilinearLF` | 1.05× | 1.09× | 1.19× | 1.25× | 1.65× | 5.2× |
| `ThirdOrderBilinearLF`         | 0.97× | 1.00× | 1.07× | 1.12× | 1.38× | 2.8× |

The carrier cap of 0.09 keeps the loop within 25 % of its configured bandwidth
with about 4× stability margin; it is the product of Kaplan & Hegarty's
third-order design example (18 Hz at 5 ms) and matches GNSS-SDR's narrow
post-sync bandwidths (5 Hz at 20 ms). The code cap of 0.018 is conservative (see
`MAX_CODE_LOOP_BANDWIDTH_TIME_PRODUCT`); at 1 Hz it binds only past 18 ms. The
resulting defaults, `BL` in Hz:

| Records | Signals                                  |  Wide | Narrow | FLL-assist |  Code |
|---------|------------------------------------------|------:|-------:|-----------:|------:|
| 1 ms    | GPS L1 C/A, GPS L5, Galileo E5a, …       |    50 |     18 |          5 |     1 |
| 2 ms    | Galileo E5a-QP                           |    45 |     18 |          5 |     1 |
| 4 ms    | Galileo E1B / E1C                        |  22.5 |     10 |          5 |     1 |
| 10 ms   | GPS L1C, BeiDou B1C; L5I synced at 10 ms |     9 |      4 |          2 |     1 |
| 20 ms   | GPS L2 CM; L1 C/A, L5Q, E5a-I synced     |   4.5 |      2 |          1 |   0.9 |
| 1.5 s   | GPS L2 CL                                |  0.06 |  0.027 |      0.013 | 0.012 |

```@autodocs
Modules = [TrackingLoops]
Pages = ["loop_filters.jl"]
Private = false
```

## Carrier loop staging

The FLL-assisted scalar loops ([`ConventionalAssistedPLLAndDLL`](@ref),
[`NCOReferencedPLLAndDLL`](@ref), and the inner loop of
[`VectorPLLAndDLL`](@ref) out of the vector loop) follow Kaplan & Hegarty's
closure sequence (§5.3, §5.5): "apply the error inputs from both discriminators
as an FLL-assisted PLL until phase lock is achieved, then convert to pure PLL".
The stages ([`CarrierLoopStage`](@ref)) run FLL-assisted at the wide bandwidth
until phase lock, then as a wide pure PLL until lock has held a while longer,
then as a narrow pure PLL. They only advance, until
[`reset_estimator_state`](@ref). With a carrier filter that has no FLL path the
loop starts at the wide pure PLL.

The pure PLL is the FLL-assisted filter with a zero FLL input, which is exactly
the third-order PLL `ThirdOrderBilinearLF` (same state, same coefficients), so
dropping the FLL leaves no transient.

### Phase-lock indicator

The staging reads [`phase_lock_indicator`](@ref), `⟨I² − Q²⟩ / A²`, an estimate
of `cos 2φ` from exponential averages of the driver's prompt over 0.1 s (at
least 25 records). Normalising by the signal power `A²`, estimated from the
prompt's moments as `√(2 M₂² − M₄)`, makes it read the same at any C/N₀; the
usual `⟨I² − Q²⟩ / ⟨I² + Q²⟩` reads low at low C/N₀ even when locked. Its
threshold,
[`phase_lock_indicator_threshold`](@ref TrackingLoops.phase_lock_indicator_threshold)
(0.5, about 30° RMS phase error), is overridable per signal type.

Dropping the FLL on phase lock was chosen over a separate frequency-lock
indicator, `Re ⟨(Pₖ P̄ₖ₋ₗ)²⟩ / A⁴` over prompts 10 ms apart. In a sample-level
simulation (GPS L1 C/A, L5Q, Galileo E1C, GPS L1C-P at 27.5–45 dB-Hz) phase lock
gave the same lock rate and jitter, far less residual Doppler when the FLL
dropped at low C/N₀ (p90 1–8 Hz against 6–23 Hz) and no aliases, which the
frequency indicator reads every 50 Hz within the 1 ms FLL range. Only on 10 ms
records did the FLL drop about 0.3 s later.

### Four-quadrant discriminators

Where the replica wipes every sign modulation off the prompt and the sign is known
(a pilot synced to its secondary code), the discriminators can be four-quadrant. The
record's `polarity` says so, from the bit buffer as it was when the record was
correlated (see [`LoopRecord`](@ref)). Nonzero, it makes the FLL four-quadrant
(twice the pull-in range) and the PLL (linear over ±180°, worth up to 6 dB of
threshold), which reads the prompt with the sign the secondary-code sync found. A
short overlay (GPS L5Q's 20 ms) can sync while the loop still pulls in; a Costas slip
after that makes the switch a half-cycle jump: the start of the resolved phase, not
part of a continuous one.

Data signals stay on the two-quadrant (Costas) discriminators, and so, by
choice, do the pilots without a secondary code (GPS L2 CL, Galileo E5a-QP). Their
FLL could be four-quadrant from the start, as consecutive prompts share their sign,
and their PLL could switch with the sign the Costas loop holds; but that sign
leaves the phase unresolved and would be decided off a single noisy prompt, and the
FLL alone was not worth a second per-record flag next to `polarity`.

### Bandwidth and NCO delay

A wider loop tolerates less NCO delay. At 50 Hz the conventional loop holds
through three records of delay and the NCO-referenced loop through seven; at
18 Hz the latter holds through thirteen, so [`NCOReferencedPLLAndDLL`](@ref)
defaults its wide bandwidth to the narrow 18 Hz.

## Signal combining

A satellite tracked on several signals of one band (e.g. Galileo E1C and E1B)
closes its loops on one of them, the driver, whose records go through
[`step_loop`](@ref). With `combine_signals = true` on the estimator the
discriminators of the other signals, the passengers, are averaged with the
driver's before the loop filters read them; a mean rather than a sum, so the
loop gain does not depend on the number of signals. The host folds each
passenger record with [`fold_passenger_record`](@ref) (see
[Host contract](@ref)).

  - **Weights** are each signal's ICD power share
    (`GNSSSignals.get_relative_power`) times its integration time, cubed for the
    FLL, whose noise variance falls with the cube. All signals of a satellite
    share one antenna and path, so the power split fixes their C/N₀ ratio and no
    C/N₀ estimate is needed. The discriminators are calibrated (cycles, Hz,
    chips), so the mean is unbiased whatever the weights.
  - **Time alignment:** the passenger records folded before a driver record are
    combined into it; those folded after the driver's last stay pending
    ([`SignalCombiningSums`](@ref)) for its next, which drops them if they are
    stale (the host dropped the driver's in-flight integration). Without
    pending passengers the driver's loops close on its own discriminators, bit
    for bit. A passenger integrating longer than the driver would dominate the
    one update it joins, so make the longest-integrating signal (typically the
    pilot) the driver.
  - **PLL and FLL:** passengers read the two-quadrant (Costas) discriminators,
    blind to a sign flip of a whole record, so data passengers and pre-sync
    records combine like any other. The FLL is combined only while it is in
    use. Into a four-quadrant driver discriminator passengers are combined only
    while the driver's reading lies within the two-quadrant range (±1/4 cycle
    for the PLL, ±1/(4T) for the FLL). Passengers stay two-quadrant even when
    their own prompt is wiped off, because the gate reads only the driver: a
    Costas driver locked half a cycle off still reads within that range, while a
    four-quadrant passenger would not. In simulation (Galileo E1C driving, E1B
    combined) this lowers the carrier-phase jitter after the sync by about 30 %
    from 20 to 40 dB-Hz, less at 18 dB-Hz, without adding cycle slips. A
    passenger's prompt is rotated onto the driver's phase frame by the nominal
    carrier phase offsets (`get_carrier_phase_offset`); a residual phase bias
    between the components is not modelled and shifts the combined lock point.
  - **DLL:** a passenger joins the code loop only where its group delay relative
    to the driver's is known (`differential_group_delay_chips`; zero is not
    assumed), which refers its reading to the driver's code phase.
  - **Vector tracking:** [`VectorPLLAndDLL`](@ref) has its own
    `combine_signals`, which also covers its scalar fallback. In the vector loop
    passengers join the PLL, while their DLL and FLL readings go to the
    navigation filter, weighted by each signal's C/N₀ (see
    [Vector tracking](@ref)).
  - [`NCOReferencedPLLAndDLL`](@ref) does not combine: it steps the driver's
    phase error predicted to the landing sample, which the passengers' records
    are not.

```@autodocs
Modules = [TrackingLoops]
Pages = ["signal_combining.jl"]
Private = false
```

## Host contract

Everything a host owes the estimators, scalar and vector, in one place.

  - **Records.** One [`LoopRecord`](@ref) per completed record. Built with
    `LoopRecord(loop, signal, filtered, output, blocks, fs; prn)` from the
    signal's [`SignalLoopState`](@ref) *before* [`apply_record`](@ref) folded
    it, with `correlated_pre_sync` as there, a record meets the contract by
    construction (`previous_prompt`, `polarity`); a host building records itself
    follows the rules in [`LoopRecord`](@ref).
  - **The driver.** Every driver record, in order, goes to
    [`step_loop`](@ref)`(estimator, state, record, words, landing_sample)`, with
    the replica words the record ran on and the sample where the resulting
    command lands (`NO_LANDING_SAMPLE`: at the record's end).
  - **Passengers.** Where [`takes_passenger_records`](@ref) holds, every
    passenger record goes to [`fold_passenger_record`](@ref), in sample order,
    each before the driver record it ends within (or at), with
    `differential_group_delay_chips` where known, on the driver's sample frame
    (`sample_index`).
  - **C/N₀.** A record may carry the host's C/N₀ estimate (`cn0`); only the
    vector loop reads it, and estimates the C/N₀ itself without it.
  - **Vector tracking** also needs each record's `prn`, the replica's code phase
    at its end (`code_phase`, from the [`CorrelatorOutput`](@ref)), and a time
    grid shared by all satellites. The end sample alone pins the code phase only
    to within one sample (some 75 m at 4 MHz); only its part past the nearest
    code-block boundary is read, so any wrap convention works, and without it
    (`NaN`) the record is taken to end on a block boundary. A host whose
    correlator restarts its sample count restores the common grid with
    `sample_offset`.

The passengers' state is kept out of the host's sight: the carrier loop's
pending sums live in the satellite's loop state (in a [`VectorPLLAndDLL`](@ref),
the inner loop's, so they travel with the scalar fallback), and the readings
the navigation filter fuses live per satellite in the navigation engine.

## Integration length

```@autodocs
Modules = [TrackingLoops]
Pages = ["sample_parameters.jl"]
Private = false
```
