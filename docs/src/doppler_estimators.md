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

## Carrier loop staging

The FLL-assisted scalar loops ([`ConventionalAssistedPLLAndDLL`](@ref),
[`NCOReferencedPLLAndDLL`](@ref), and the inner loop of
[`VectorPLLAndDLL`](@ref) out of the vector loop) start as an FLL-assisted PLL
and drop the FLL once the frequency has converged, along Kaplan & Hegarty's
closure sequence (§5.3, §5.5): "apply the error inputs from both discriminators
as an FLL-assisted PLL until phase lock is achieved, then convert to pure PLL".
There is no down-staging: the pure PLL stays until
[`reset_estimator_state`](@ref) restarts the staging.

  - **Frequency lock** is declared once the mean FLL discriminator, the
    residual frequency error, stays below [`frequency_lock_threshold`](@ref)
    (3 Hz, at most 1/(16T) on long records, a quarter of the two-quadrant FLL's
    range) over a [`frequency_lock_window`](@ref) (0.5 s, at least four
    records). The window mean is the phase advance across the window over its
    length, so its noise falls with the window length rather than with each
    record's SNR. Both are overridable per signal type. Kaplan & Hegarty's
    phase lock indicator (§5.11.2), specified unnormalised for 20 ms updates,
    never declares lock at 1 ms and 30 dB-Hz, which would leave the noisy FLL
    branch in. The indicator, a [`FrequencyLockIndicator`](@ref), is part of
    the per-satellite state.
  - **The pure PLL** is the FLL-assisted filter with a zero FLL input, which is
    exactly the third-order PLL `ThirdOrderBilinearLF` (same state, same
    coefficients), so the switch costs nothing and leaves no transient.
  - **Four-quadrant discriminators** apply where the replica wipes every sign
    modulation off the prompt: a dataless signal, synced to its secondary code
    where it has one. The host says so per record, from the bit buffer as it
    was when the record was correlated:
      * `wiped_off` ([`is_wiped_off`](@ref)) makes the FLL four-quadrant (twice
        the pull-in range). It needs no sign, only that consecutive prompts
        share it, so it applies from the start to the pilots without a
        secondary code (GPS L2 CL, Galileo E5a-QP).
      * `polarity` ([`sync_polarity`](@ref)) makes the PLL four-quadrant
        (linear over ±180°, worth up to 6 dB of threshold), reading the prompt
        with the sign the secondary-code sync found. A short overlay (GPS L5Q's
        20 ms) can sync while the loop is still pulling in, and a Costas slip
        after it makes the switch a half-cycle jump of the carrier phase: the
        start of the resolved phase, not part of a continuous one.
      * The pilots without a secondary code (GPS L2 CL, Galileo E5a-QP) keep
        the Costas PLL. That is a choice, not a necessity: their prompt keeps
        its sign too, so the PLL could turn four-quadrant with the sign the
        Costas loop holds. But without a sync that sign is arbitrary, so the
        switch would leave the carrier phase unresolved, and it would have to
        be taken off a single noisy prompt; the wider range alone was not
        considered worth that.

    Data signals stay on the two-quadrant (Costas) discriminators. A record
    whose `wiped_off` differs from the previous one's must come without a
    previous prompt (see [`LoopRecord`](@ref)), so the four-quadrant FLL never
    compares across the sync.

With a carrier filter other than the FLL-assisted one the loop is a PLL from the
start and runs no frequency lock indicator. In the vector loop the FLL branch
carries the navigation filter's carrier correction, so it is not staged; its
discriminators follow `wiped_off` and `polarity` as in the scalar loop.

Dropping the FLL makes a loop less tolerant of NCO delay: at 85 Hz the
NCO-referenced loop holds through five records of delay instead of six. At the
default 18 Hz both the conventional and the NCO-referenced loop hold through
about twenty.

## Signal combining

A satellite tracked on several signals of one band (e.g. Galileo E1C and E1B)
closes its loops on one of them, the driver, whose records go through
[`step_loop`](@ref). With `combine_signals = true` on the estimator, e.g.
`ConventionalAssistedPLLAndDLL(; combine_signals = true)`, the discriminators of
the other signals, the passengers, are combined with the driver's into a
weighted mean before the loop filters read it. A mean rather than a sum, so the
loop gain does not change with the number of signals. The host folds each
passenger record with [`combine_passenger_record`](@ref), in sample order and
before the driver record it ends within, and asks [`combines_signals`](@ref)
whether to.

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
    ([`SignalCombiningSums`](@ref)) for its next; a host that drops the driver's
    in-flight integration drops them with [`drop_pending_passengers`](@ref).
    Where no passenger record is pending, the driver's loops close on its own
    discriminators, bit for bit. Each passenger is assumed to integrate no
    longer than the driver, as a longer record would dominate the one driver
    update it is combined into: make the longest-integrating signal (typically
    the pilot) the driver.
  - **PLL and FLL:** passengers read the two-quadrant (Costas) discriminators,
    which are blind to a sign flip of a whole record, so data passengers and
    records correlated before a passenger's own sync are combined like any
    other. They are combined into a carrier loop while it is formed, the FLL
    until frequency lock. Into a four-quadrant driver discriminator they are
    combined only while its reading lies within their own two-quadrant range,
    ±1/4 cycle for the PLL and ±1/(4T) for the FLL, where both read the error
    alike. In simulation (Galileo E1C driving, E1B combined) this lowers the
    carrier-phase jitter after the sync by about 30 % from 20 to 40 dB-Hz, and
    less at 18 dB-Hz, without adding cycle slips. A passenger's prompt is
    rotated onto the driver's phase frame by the signals' nominal carrier phase
    offsets (`get_carrier_phase_offset`); a residual phase bias between the
    components is not modelled and shifts the combined lock point.
  - **DLL:** a passenger is combined into the code loop only where its group
    delay relative to the driver's is known, passed as
    `differential_group_delay_chips`; zero is not assumed. Its discriminator is
    referred to the driver's code phase by it.
  - **Vector tracking:** out of the vector loop, [`VectorPLLAndDLL`](@ref)
    combines as its inner loop does. In the vector loop passengers are combined
    into the PLL only: the code loop and the FLL branch belong to the
    navigation filter, which reads the driver's own discriminators.
  - The [`NCOReferencedPLLAndDLL`](@ref) does not combine: it steps the
    driver's phase error predicted to the landing sample, which the passengers'
    records are not.

```@autodocs
Modules = [TrackingLoops]
Pages = ["signal_combining.jl"]
Private = false
```

## Integration length

```@autodocs
Modules = [TrackingLoops]
Pages = ["sample_parameters.jl"]
Private = false
```
