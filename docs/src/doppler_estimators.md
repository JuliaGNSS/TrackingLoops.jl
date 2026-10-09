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
  NCO. With a fixed word and no landing sample it is the conventional loop at
  the same bandwidths.

The bandwidth rules ([`default_wide_carrier_loop_filter_bandwidth`](@ref),
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

The carrier loop has three bandwidths, one per role in its staging (see
[Carrier loop staging](@ref)): a flat 50 Hz pull-in PLL bandwidth
([`default_wide_carrier_loop_filter_bandwidth`](@ref)), a flat 18 Hz tracking PLL
bandwidth the loop narrows to once phase lock has held
([`default_narrow_carrier_loop_filter_bandwidth`](@ref)) and a flat 5 Hz FLL
bandwidth for the FLL path of the FLL-assisted filter
([`default_fll_assist_loop_filter_bandwidth`](@ref)). The code bandwidth defaults
to a flat 1 Hz. All are one-sided noise bandwidths `BL` in the sense of Kaplan &
Hegarty, so they plug into the usual PLL jitter and dynamic-stress formulas: the
carrier filter is fed the phase error in cycles and the FLL error in Hz, and it
takes the PLL and the FLL bandwidth as a pair (`ω₀f = BL_FLL / 0.53`), so the
FLL path is no longer tied to the PLL's. 18 Hz is the third-order PLL bandwidth
of the literature (Kaplan & Hegarty Table 5.6, Pany Table 3.3; Borre et al.
quote about 20 Hz). The 50 Hz wide bandwidth reaches phase lock about as fast
as Tracking v8's loop, whose PLL ran about 5.6× wider than configured, and 5 Hz
is the FLL bandwidth that pulls a handover error of up to a quarter of the
two-quadrant FLL range in without being the 1 ms loop's noise source (a
sample-level simulation study, single signals at 25–45 dB-Hz).

Before Tracking.jl#244 the carrier filter was fed the phase error in radians,
which made the loop gain 2π too high: a configured 18 Hz behaved like a loop of
about 100 Hz, and the old default `BL = 0.018 / T` (with `1/N` for `N`
integrated blocks) was tuned against that gain. Bandwidths carried over from
then are about 5–6× narrower than the loops they were tuned on.

### Stability cap

Stability does depend on the loop update interval `Δt`, so at filter time each
bandwidth is capped against the record's actual integration time (for the code
loop by [`effective_code_loop_filter_bandwidth`](@ref)). An explicit bandwidth
is capped the same way; the cap only ever narrows a loop.

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

The carrier cap at 0.09 keeps the loop within 25 % of its configured bandwidth
with about 4× stability margin. It is the product of Kaplan & Hegarty's
third-order design example (18 Hz at 5 ms) and matches GNSS-SDR's narrow
post-sync bandwidths (5 Hz at 20 ms). The resulting defaults:

| Integration | Signals                                                   |    Wide BL |   Narrow BL | FLL-assist BL |  Code BL |
|-------------|-----------------------------------------------------------|-----------:|------------:|--------------:|---------:|
| 1 ms        | GPS L1 C/A, GPS L5, Galileo E5a, …                        |      50 Hz |       18 Hz |     5 Hz |     1 Hz |
| 2 ms        | Galileo E5a-QP                                            |      45 Hz |       18 Hz |     5 Hz |     1 Hz |
| 4 ms        | Galileo E1B / E1C                                         |    22.5 Hz |       10 Hz |     5 Hz |     1 Hz |
| 10 ms       | GPS L1C-D / L1C-P, BeiDou B1C; GPS L5I synced at 10 ms    |       9 Hz |        4 Hz |     2 Hz |     1 Hz |
| 20 ms       | GPS L2 CM; GPS L1 C/A, L5Q, Galileo E5a-I synced at 20 ms |     4.5 Hz |        2 Hz |     1 Hz |   0.9 Hz |
| 1.5 s       | GPS L2 CL                                                 |    0.06 Hz |    0.027 Hz |  0.013 Hz | 0.012 Hz |

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

## Carrier loop staging

The FLL-assisted scalar loops ([`ConventionalAssistedPLLAndDLL`](@ref),
[`NCOReferencedPLLAndDLL`](@ref), and the inner loop of
[`VectorPLLAndDLL`](@ref) out of the vector loop) run through three stages, a
[`CarrierLoopStage`](@ref) in the per-satellite state, along Kaplan & Hegarty's
closure sequence (§5.3, §5.5): "apply the error inputs from both discriminators
as an FLL-assisted PLL until phase lock is achieved, then convert to pure PLL".
There is no down-staging: the stages only advance until
[`reset_estimator_state`](@ref) restarts them.

  - **FLL-assisted PLL** (`FLL_ASSISTED_PLL`) at the wide bandwidth, with the
    FLL path at its own bandwidth, until the phase-lock indicator has read lock
    for one time constant of its averages without a break: the carrier is in
    phase lock, so its Doppler has converged.
  - **Wide pure PLL** (`WIDE_PLL`) at the wide bandwidth, until the phase-lock
    indicator has read lock for four more time constants without a break,
    counted from the end of the FLL-assisted stage. Dropping the FLL and
    narrowing are separate steps: narrowing while the loop still settles from
    the FLL's last correction loses lock.
  - **Narrow pure PLL** (`NARROW_PLL`) at the narrow bandwidth.

The **phase-lock indicator** ([`phase_lock_indicator`](@ref)), part of the
per-satellite state and advanced by every driver record, is `⟨I² − Q²⟩ / A²`,
an estimate of `cos 2φ`, from exponential averages of the prompt, updated every
record, over a time constant of 0.1 s and at least 25 records (0.25 s for 10 ms
records). It is normalised by the signal power `A²` averaged alike and estimated
from the prompt's moments as `√(2 M₂² − M₄)`, so it reads the same at any C/N₀:
unlike `⟨I² − Q²⟩ / ⟨I² + Q²⟩` it does not read low at low C/N₀ when the loop is
locked. Its threshold, [`phase_lock_indicator_threshold`](@ref TrackingLoops.phase_lock_indicator_threshold)
(0.5, an RMS phase error of about 30°), is overridable per signal type; a
receiver may read the indicator for its own lock decisions.

A separate frequency-lock indicator, `Re ⟨(Pₖ P̄ₖ₋ₗ)²⟩ / A⁴` over prompts 10 ms
apart, was tried to drop the FLL as soon as the Doppler converged. In a
sample-level simulation (GPS L1 C/A, L5Q, Galileo E1C, GPS L1C-P at 27.5–45
dB-Hz) dropping on phase lock staged as well or better: the same lock rate and
jitter, the FLL dropped with far less residual Doppler at low C/N₀ (p90 1–8 Hz
against 6–23 Hz), and no aliases, which the frequency indicator reads every
50 Hz within the 1 ms FLL range. Only on 10 ms records did the FLL drop about
0.3 s later.

  - **The pure PLL** is the FLL-assisted filter with a zero FLL input, which is
    exactly the third-order PLL `ThirdOrderBilinearLF` (same state, same
    coefficients), so dropping the FLL costs nothing and leaves no transient.
  - **Four-quadrant discriminators** apply where the replica wipes every sign
    modulation off the prompt and the sign is known: a pilot synced to its
    secondary code. The host passes that sign per record as `polarity`
    ([`get_sync_polarity`](@ref)), from the bit buffer as it was when the record
    was correlated. Nonzero, it makes both discriminators four-quadrant: the FLL
    (twice the pull-in range) and the PLL (linear over ±180°, worth up to 6 dB
    of threshold), which reads the prompt with the sign the secondary-code sync
    found. A short overlay (GPS L5Q's 20 ms) can sync while the loop is still
    pulling in, and a Costas slip after it makes the switch a half-cycle jump of
    the carrier phase: the start of the resolved phase, not part of a continuous
    one.

    The pilots without a secondary code (GPS L2 CL, Galileo E5a-QP) stay
    two-quadrant. Their prompt keeps its sign, so the FLL could be four-quadrant
    from the start, needing only that consecutive prompts share the sign, and
    the PLL with the sign the Costas loop holds. But without a sync that sign is
    arbitrary, so the switch would leave the carrier phase unresolved and be
    taken off a single noisy prompt; and the FLL alone was not worth a second
    per-record flag next to `polarity`.

    Data signals stay on the two-quadrant (Costas) discriminators. A record
    whose `polarity` differs from the previous one's must come without a
    previous prompt (see [`LoopRecord`](@ref)), so the four-quadrant FLL never
    compares across the sync.

With a carrier filter other than the FLL-assisted one the loop starts at the
wide pure PLL. In the vector loop the FLL branch carries the navigation filter's
carrier correction, so it is not staged: it runs at the narrow bandwidth with
the FLL slot tied to it, and its discriminators follow `polarity` as in the
scalar loop.

A wider loop is less tolerant of NCO delay. At the default 50 Hz wide
bandwidth the conventional loop holds through three records of delay; at 50 Hz
the NCO-referenced loop would hold through seven. The NCO-referenced loop, meant
for hardware correlators whose commands land records later, therefore defaults
its wide bandwidth to the 18 Hz narrow bandwidth, where it holds through
thirteen.

## Signal combining

A satellite tracked on several signals of one band (e.g. Galileo E1C and E1B)
closes its loops on one of them, the driver, whose records go through
[`step_loop`](@ref). With `combine_signals = true` on the estimator, e.g.
`ConventionalAssistedPLLAndDLL(; combine_signals = true)`, the discriminators of
the other signals, the passengers, are combined with the driver's into a
weighted mean before the loop filters read it. A mean rather than a sum, so the
loop gain does not change with the number of signals. The host folds each
passenger record with [`fold_passenger_record`](@ref), in sample order and
before the driver record it ends within, and asks
[`takes_passenger_records`](@ref) whether to.

  - **Weights** are each signal's ICD power share
    (`GNSSSignals.get_relative_power`) times the record's integration time, and
    for the FLL times the integration time squared on top, as its noise variance
    falls with its cube. All signals of a satellite share one antenna and one
    path, so the power split fixes their C/N₀ ratio; no C/N₀ estimate is read.
    The discriminators are calibrated (PLL in cycles, FLL in Hz, DLL in chips),
    so the mean is unbiased whatever the weights; they only decide how much
    noise is removed.
  - **Time alignment:** every passenger record folded before a driver record is
    combined into it, and the driver's step starts the sums afresh. Records
    folded after the driver's last one stay pending in the satellite's state
    ([`SignalCombiningSums`](@ref)) for its next. A pending record that ended at
    or before the start of the driver record that steps it belongs to a driver
    record that never came (the host dropped the driver's in-flight integration,
    e.g. at a code-phase snap), so the step drops the pending sums. This needs a
    satellite's driver and passenger records on one sample frame (their
    `sample_index`).
    Where no passenger record is pending, the driver's loops close on its own
    discriminators, bit for bit. Each passenger is assumed to integrate no
    longer than the driver, as a longer record would dominate the one driver
    update it is combined into: make the longest-integrating signal (typically
    the pilot) the driver.
  - **PLL and FLL:** passengers read the two-quadrant (Costas) discriminators,
    which are blind to a sign flip of a whole record, so data passengers and
    records correlated before a passenger's own sync are combined like any
    other. They are combined into a carrier loop while it is formed, the FLL
    in the FLL-assisted stage. Into a four-quadrant driver discriminator they are
    combined only while its reading lies within their own two-quadrant range,
    ±1/4 cycle for the PLL and ±1/(4T) for the FLL, where both read the error
    alike. Passengers stay two-quadrant even where their own prompt is wiped
    off, as the gate reads only the driver: a Costas driver locked half a cycle
    off, or a two-quadrant driver FLL folding an error beyond its range, still
    reads within that range, while a four-quadrant passenger would not. In
    simulation (Galileo E1C driving, E1B combined) this lowers the
    carrier-phase jitter after the sync by about 30 % from 20 to 40 dB-Hz, and
    less at 18 dB-Hz, without adding cycle slips. A passenger's prompt is
    rotated onto the driver's phase frame by the signals' nominal carrier phase
    offsets (`get_carrier_phase_offset`); a residual phase bias between the
    components is not modelled and shifts the combined lock point.
  - **DLL:** a passenger is combined into the code loop only where its group
    delay relative to the driver's is known, passed as
    `differential_group_delay_chips`; zero is not assumed. Its discriminator is
    referred to the driver's code phase by it.
  - **Vector tracking:** one switch here too, the vector loop's own,
    `VectorPLLAndDLL(signals...; combine_signals = true)`; it covers the scalar
    fallback, so the inner loop is built without it. Out of the vector loop it
    combines as the scalar loop does. In it,
    passengers are combined into the PLL, while the code loop and the FLL branch
    belong to the navigation filter: it reads every signal's DLL and FLL
    readings, each in the cycle of the driver record it ends within, and fuses
    them weighted by
    each signal's own C/N₀ (see [Vector tracking](@ref)).
  - The [`NCOReferencedPLLAndDLL`](@ref) does not combine: it steps the
    driver's phase error predicted to the landing sample, which the passengers'
    records are not.

```@autodocs
Modules = [TrackingLoops]
Pages = ["signal_combining.jl"]
Private = false
```

## Host contract

Everything a host owes the estimators, scalar and vector, in one place.

  - **Records.** One [`LoopRecord`](@ref) per completed record. Built with
    `LoopRecord(loop, signal, filtered, output, blocks, fs; prn)` from the
    signal's [`SignalLoopState`](@ref) *before* [`apply_record`](@ref) folded the
    record, it follows the contract by construction: `previous_prompt` is zero
    on the first record and wherever the block count or the polarity changes
    (a sync that wipes a pilot's prompt off, a change of integration length),
    and `polarity` is the bit buffer's as the record was correlated (pass
    `correlated_pre_sync` as to `apply_record`). A host that builds records
    itself applies the same rules.
  - **The driver.** Every driver record, in order, goes to
    [`step_loop`](@ref)`(estimator, state, record, words, landing_sample)`:
    `words` are the replica words the record ran on, `landing_sample` is where
    the command computed from the record's fold lands (`NO_LANDING_SAMPLE`: from
    the record's end; a hardware host passes the landing sample, and the
    estimator sizes its corrections for that moment).
  - **Passengers.** Where [`takes_passenger_records`](@ref) holds, every
    passenger record goes to [`fold_passenger_record`](@ref), in sample order,
    each before the driver record it ends within (or ends at). Each passenger's
    records follow the record contract for its own sequence;
    `differential_group_delay_chips`, the passenger's group delay minus the
    driver's (`NaN`: unknown), refers its code reading to the driver's. A
    satellite's passenger and driver records share one sample frame
    (`sample_index`): passenger sums still pending from before a dropped driver
    integration (the code-phase snap at a sync) are discarded by their end.
  - **C/N₀.** A record may carry the host's C/N₀ estimate of its signal
    (`cn0`, `NaN` when unknown); only the vector loop reads it, and estimates
    the C/N₀ from the prompts itself without it.
  - **Vector tracking** needs, of every record handed to it, the satellite
    (`prn`); the replica's code phase in chips at the record's end
    (`code_phase`, from the [`CorrelatorOutput`](@ref): the end sample alone pins
    it only to within one sample, some 75 m at 4 MHz; only its part past the
    nearest code-block boundary is read, so any wrap convention works, and
    without it, `NaN`, the record is taken to end on a block boundary); and a
    common time grid: `sample_index / sampling_frequency` must be the time since
    one origin shared by all satellites, which a host whose correlator restarts
    its sample count restores with `sample_offset`.

Where the passengers' state lives follows who owns it, and a host sees none of
it. The sums a carrier loop combines are part of the satellite's loop state
([`SignalCombiningSums`](@ref)): in a [`VectorPLLAndDLL`](@ref) the inner loop's,
as they are its scalar fallback's own and travel with it into and out of the
vector loop. The readings the navigation filter fuses — each passenger's DLL
and FLL readings counted per cycle, and its C/N₀ — belong to the navigation
engine, which owns the cycles, and are kept per satellite there.

## Integration length

```@autodocs
Modules = [TrackingLoops]
Pages = ["sample_parameters.jl"]
Private = false
```
