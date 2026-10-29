```@meta
CurrentModule = TrackingLoops
```

# Doppler estimators and loop filters

A Doppler estimator closes the carrier and code loops. Its configuration (loop
filter types and bandwidths) is a plain value, a subtype of
[`AbstractDopplerEstimator`](@ref); the per-satellite state it advances is
created with [`init_estimator_state`](@ref). Every estimator is stepped with the
same call,

```julia
state, carrier_doppler, code_doppler = step_loop(estimator, state, record, words, landing_sample)
```

where `record` is a [`LoopRecord`](@ref), `words` is the replica frequency the
record was produced under (a [`FixedNCOWord`](@ref) for a software correlator,
the channel's [`NCOTimeline`](@ref) for a hardware one) and `landing_sample` is
the sample the new command takes effect at, or [`NO_LANDING_SAMPLE`](@ref) for
"at the end of each record".

A host steps every record of a satellite, of every signal it tracks. The state
remembers the driver it was initialised with: the driver's records close the
loops, and the scalar loops leave any other record (a passenger, such as the
data component of a pilot + data pair) to the host, returning the state
unchanged and the command in force.

- [`ConventionalPLLAndDLL`](@ref): a PLL and a carrier-aided DLL.
- [`ConventionalAssistedPLLAndDLL`](@ref): the same with an FLL assisting the
  PLL until phase lock (see [Carrier-loop staging](@ref)), to pull in larger
  initial frequency errors. The default in Tracking.jl.
- [`NCOReferencedPLLAndDLL`](@ref): stays stable when its correction takes
  effect only several records later, as with a hardware NCO. With a fixed word
  and no landing sample it is the FLL-assisted conventional loop at the same
  bandwidths.

The bandwidth rules ([`default_wide_carrier_loop_filter_bandwidth`](@ref),
[`effective_carrier_loop_filter_bandwidth`](@ref), …) keep the loops stable for a
given integration time; the integration-length rules
([`calc_num_code_blocks_to_integrate`](@ref TrackingLoops.calc_num_code_blocks_to_integrate), …) say how long a signal may be
integrated coherently once its bit or secondary code has been found.

## The per-record fold

[`apply_record`](@ref) advances one signal component's
[`SignalLoopState`](@ref) (its prompt filter, C/N₀ estimator and bit buffer) by
one record, so every caller does it the same way.

```@docs
SignalLoopState
SignalLoopState(::SignalLoopState)
TrackingLoops.SyncChange
apply_record(::SignalLoopState, signal::AbstractGNSSSignal, ::Integer, ::CorrelatorOutput, ::Any, ::Any, ::Bool, ::Real = get_carrier_phase_offset(signal))
is_dataless_with_secondary_code(::AbstractGNSSSignal)
reset_signal_state(::SignalLoopState)
restart_bit_clock(::SignalLoopState)
```

## Two- and four-quadrant discriminators

The carrier discriminators are two-quadrant (Costas) by default: they read a
prompt and its negation alike, so a data bit or secondary-code chip flipping the
prompt's sign does not disturb them, over ±1/4 cycle for the PLL and ±1/(4T) for
the FLL. Where the replica wipes every sign modulation off the prompt (a
dataless signal synced to its secondary code) the prompt's sign is known, and
the estimators switch to the four-quadrant discriminators, over ±1/2 cycle and
±1/(2T), worth up to 6 dB of tracking threshold. The record's
`carrier_phase_polarity`, the prompt's sign, says so; [`apply_record`](@ref)
computes it into the [`SignalLoopState`](@ref), and a [`LoopRecord`](@ref) built
from that state carries it to the scalar and vector loops alike. A record whose
polarity differs from the previous record's comes without a previous prompt, so
the four-quadrant FLL never compares prompts across the switch.

GPS L1C-P and L5Q, Galileo E1C, E5a-Q, E5b-Q and E6-C, and BeiDou B1C-P and
B2a-Q switch once their secondary code is synced; if the Costas loop slipped
half a cycle between the sync and the switch, the four-quadrant PLL pulls the
carrier phase over by half a cycle, after which the prompt agrees with the sign
again. Data signals never switch, nor, for now, do the pilots without a
secondary code (GPS L2 CL, Galileo E5a-QP), whose sync reads no sign.
The NCO-referenced loop folds its predicted phase error into whichever range the
record's PLL reads.

The sign rests on a contract with whoever correlates the records, a hardware
correlator as much as Tracking.jl: until the sync, these signals' records
([`is_dataless_with_secondary_code`](@ref)) are primary-only. A replica carrying
the same secondary chip on every block is fine once the host multiplies the records
by that chip (Tracking.jl's carries chip 0 and multiplies by it). A negative chip
left in makes the sync read the wrong sign, and on the switch the four-quadrant PLL
pulls the carrier phase over by half a cycle.

## Estimators

```@docs
ConventionalPLLAndDLL
LoopRecord
LoopRecord(::Any, ::Any, ::CorrelatorOutput, ::SignalLoopState, ::Any)
NCOReferencedPLLAndDLL
TrackingLoops.SatConventionalPLLAndDLL
TrackingLoops.SatNCOReferencedPLLAndDLL
ConventionalAssistedPLLAndDLL(::Type{CO} = SecondOrderBilinearLF) where {CO<:AbstractLoopFilter}
carrier_loop_stage(::Union{SatConventionalPLLAndDLL, SatNCOReferencedPLLAndDLL})
TrackingLoops.carrier_phase_correction(::Union{SatConventionalPLLAndDLL, SatNCOReferencedPLLAndDLL}, ::Any, ::Any, ::Any)
TrackingLoops.estimator_state_type(::AbstractDopplerEstimator, ::AbstractGNSSSignal)
init_estimator_state(::ConventionalPLLAndDLL{CA, CO}, ::AbstractGNSSSignal, ::Any, ::Any) where {CA<:AbstractLoopFilter, CO<:AbstractLoopFilter}
navigation_cycle(::AbstractDopplerEstimator)
navigation_epoch(::AbstractDopplerEstimator)
navigation_solution(::AbstractDopplerEstimator)
navigation_status(::AbstractDopplerEstimator)
TrackingLoops.phase_lock_indicator(::Union{SatConventionalPLLAndDLL, SatNCOReferencedPLLAndDLL})
TrackingLoops.phase_locked(::Union{SatConventionalPLLAndDLL, SatNCOReferencedPLLAndDLL})
reset_estimator_state(::ConventionalPLLAndDLL, ::SatConventionalPLLAndDLL, ::Any, ::Any)
satellite_report(::AbstractDopplerEstimator, ::AbstractGNSSSignal, ::Integer)
step_loop(::ConventionalPLLAndDLL, ::SatConventionalPLLAndDLL, ::LoopRecord, ::Any, ::Int)
step_loop(::NCOReferencedPLLAndDLL, ::SatNCOReferencedPLLAndDLL, ::LoopRecord, ::Any, ::Int)
```

## Carrier-loop staging

The scalar loops stage their carrier loop on phase lock: FLL-assisted at the
wide PLL bandwidth, then a pure PLL at the wide bandwidth, then narrow, each
stage moving on once the phase-lock indicator ([`phase_lock_indicator`](@ref TrackingLoops.phase_lock_indicator))
has read lock ([`phase_locked`](@ref TrackingLoops.phase_locked): at least 0.5, an RMS phase error of about
30°) long enough. The stages and when they advance are given under
[`CarrierLoopStage`](@ref TrackingLoops.CarrierLoopStage); in the vector loop, see [The carrier in the vector
loop](@ref vector-carrier).

A wider loop tolerates less command delay: at 50 Hz the conventional loop holds
through three records of delay. The [`NCOReferencedPLLAndDLL`](@ref), meant for
hardware correlators whose commands land records later, therefore pulls in at
the narrow 18 Hz by default.

```@docs
TrackingLoops.CarrierLoopStage
```

## Loop-filter bandwidths

The PLL pulls in at a wide 50 Hz and narrows to 18 Hz once phase lock has held;
the FLL path of the FLL-assisted filter runs at 5 Hz while it is in use, and the
code loop at 1 Hz, all for every signal. All are one-sided noise bandwidths `BL`
(Kaplan & Hegarty), so they plug into the usual jitter and dynamic-stress
formulas: the carrier filter is fed the phase error in cycles and the FLL error
in Hz. 18 Hz is the third-order PLL bandwidth of the literature (Kaplan & Hegarty
Table 5.6, Pany Table 3.3; Borre et al. quote about 20 Hz), none of which scales
it with the primary code period: thermal jitter and dynamic stress, which set
the bandwidth, do not depend on it.

### Stability cap

Stability does depend on the loop update interval `Δt`, so at filter time every
bandwidth, default or explicit, is capped against the record's integration time
([`effective_carrier_loop_filter_bandwidth`](@ref),
[`effective_fll_assist_loop_filter_bandwidth`](@ref TrackingLoops.effective_fll_assist_loop_filter_bandwidth),
[`effective_code_loop_filter_bandwidth`](@ref TrackingLoops.effective_code_loop_filter_bandwidth)). The cap only ever narrows a
loop.

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

The PLL cap of 0.09 keeps the loop within 25 % of its configured bandwidth with
about 4× stability margin, at the wide and the narrow bandwidth alike. It is the
product of Kaplan & Hegarty's third-order design example (18 Hz at 5 ms) and
matches GNSS-SDR's narrow post-sync bandwidths (5 Hz at 20 ms). A tighter cap on
the narrow bandwidth would buy little: at 20 ms and 30 dB-Hz the thermal jitter
is about 4° at 4.5 Hz and 2.6° at 2 Hz, while a TCXO's jitter (Allan deviation
1e-10, GPS L1) grows from about 6° to 13° and the jerk the loop tolerates falls
with `BL³`. From 10 ms on, the cap sets wide and narrow alike. The resulting
defaults, `BL` in Hz:

| Records | Signals                                  |  Wide | Narrow | FLL-assist |  Code |
|---------|------------------------------------------|------:|-------:|-----------:|------:|
| 1 ms    | GPS L1 C/A, GPS L5, Galileo E5a, …       |    50 |     18 |          5 |     1 |
| 2 ms    | Galileo E5a-QP                           |    45 |     18 |          5 |     1 |
| 4 ms    | Galileo E1B / E1C                        |  22.5 |     18 |          5 |     1 |
| 10 ms   | GPS L1C, BeiDou B1C; L5I synced at 10 ms |     9 |      9 |          2 |     1 |
| 20 ms   | GPS L2 CM; L1 C/A, L5Q, E5a-I synced     |   4.5 |    4.5 |          1 |   0.9 |
| 1.5 s   | GPS L2 CL                                |  0.06 |   0.06 |      0.013 | 0.012 |

The code cap of 0.018 is conservative (see
[`MAX_CODE_LOOP_BANDWIDTH_TIME_PRODUCT`](@ref TrackingLoops.MAX_CODE_LOOP_BANDWIDTH_TIME_PRODUCT)); the default 1 Hz applies in full
up to 18 ms.

```@docs
TrackingLoops.MAX_CARRIER_LOOP_BANDWIDTH_TIME_PRODUCT
TrackingLoops.MAX_CODE_LOOP_BANDWIDTH_TIME_PRODUCT
TrackingLoops.MAX_FLL_ASSIST_LOOP_BANDWIDTH_TIME_PRODUCT
TrackingLoops.LoopBandwidths
TrackingLoops.aid_dopplers(::AbstractGNSSSignal, ::Any, ::Any, ::Any, ::Any)
TrackingLoops.calc_code_frequency_update(::AbstractGNSSSignal, ::AbstractLoopFilter, ::AbstractCorrelator, ::Any, ::Any, ::Any, ::Any)
TrackingLoops.default_code_loop_filter_bandwidth(::AbstractGNSSSignal)
TrackingLoops.default_fll_assist_loop_filter_bandwidth(::AbstractGNSSSignal)
TrackingLoops.default_narrow_carrier_loop_filter_bandwidth(::AbstractGNSSSignal)
default_wide_carrier_loop_filter_bandwidth(::AbstractGNSSSignal)
effective_carrier_loop_filter_bandwidth(::Any, ::Any)
TrackingLoops.effective_code_loop_filter_bandwidth(::Any, ::Any)
TrackingLoops.effective_fll_assist_loop_filter_bandwidth(::Any, ::Any)
```

## Integration length

```@docs
TrackingLoops.calc_num_code_blocks_for_bit_buffer(::AbstractGNSSSignal, ::Integer, ::Any, ::Bool)
TrackingLoops.calc_num_code_blocks_to_integrate(::AbstractGNSSSignal, ::Int, ::Bool)
TrackingLoops.default_num_code_blocks_to_integrate(::AbstractGNSSSignal)
TrackingLoops.max_num_code_blocks_to_integrate(::AbstractGNSSSignal)
```
