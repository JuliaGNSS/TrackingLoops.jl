```@meta
CurrentModule = TrackingLoops
```

# Tracking loops and loop filters

A tracking loop steps once per coherent integration of one signal, ahead of the
discriminators (one hardware dump or a sum of dumps): the latest integration of the
signal's [`PostCorrelationState`](@ref), with the bookkeeping the loop needs.

A tracking loop closes the carrier and code loops. Scalar loop configuration
(loop-filter types and bandwidths) is a plain value. A [`VectorTrackingLoop`](@ref)
implements the same interface and owns the shared navigation processor. Scalar loops
keep per-satellite state with the satellite. The scalar state
`previous_prompt_sample_index` records the last driver prompt center and its
`driver_key` selects the integrations that close the loops. The state is created with
[`init_loop_state`](@ref). Every loop is stepped with the
same call,

```julia
loop_state, carrier_doppler, code_doppler =
    step_loop!(tracking_loop, loop_state, post_correlation_state, nco_history, application_sample_index)
```

where `post_correlation_state` is the signal's [`PostCorrelationState`](@ref) after its latest
integration, `nco_history` describes the replica frequencies the
integration was produced under (a [`ConstantNCOCommand`](@ref) for a software correlator,
the channel's [`NCOTimeline`](@ref) for a hardware one) and `application_sample_index` is
the sample the new command takes effect at, or `nothing` for
"at the end of each integration".

A host steps every integration of a satellite, of every signal it tracks. The state
remembers the driver it was initialised with: the driver's integrations close the
loops, and the scalar loops leave any other integration (a passenger, such as the
data component of a pilot + data pair) to the host, returning the state
unchanged and the command in force.

- [`ScalarTrackingLoop`](@ref): a PLL and a carrier-aided DLL.
- `ScalarTrackingLoop(ThirdOrderAssistedBilinearLF)`: an FLL assists the
  PLL until phase lock (see [Carrier-loop staging](@ref)), to pull in larger
  initial frequency errors. The default in Tracking.jl.
- [`DelayCompensatedTrackingLoop`](@ref): stays stable when its correction takes
  effect only several integrations later, as with a hardware NCO. With a fixed command
  and no application sample it is the FLL-assisted conventional loop at the same
  bandwidths.

The bandwidth rules ([`default_wide_carrier_loop_filter_bandwidth`](@ref),
[`effective_carrier_loop_filter_bandwidth`](@ref), …) keep the loops stable for a
given integration time; the integration-length rules
([`calc_num_code_periods_to_integrate`](@ref TrackingLoops.calc_num_code_periods_to_integrate), …) say how long a signal may be
integrated coherently once its bit or secondary code has been found.

## Post-correlation processing

[`process_correlator_bank_output!`](@ref) advances one signal component's
[`PostCorrelationState`](@ref) (its post-correlation filter, C/N₀ estimator and bit synchronizer) by
one integration and returns the state, with that integration as its latest. Pass it directly
to `step_loop!`. The soft bits it demodulates join the bit synchronizer's soft-bit vector
([`get_soft_bits`](@ref)), which the host empties with [`drain_soft_bits!`](@ref).

```@docs
PostCorrelationState
PostCorrelationState(::PostCorrelationState)
TrackingLoops.SyncEvent
process_correlator_bank_output!(::PostCorrelationState, ::CorrelatorBankOutput, ::Any, ::Any, ::Bool)
is_dataless_with_secondary_code(::AbstractGNSSSignal)
reset_post_correlation!(::PostCorrelationState)
reset_sync!(::PostCorrelationState)
get_secondary_code_phase(::PostCorrelationState)
get_num_code_periods_into_bit(::PostCorrelationState)
get_last_num_code_periods(::PostCorrelationState)
get_bit_polarity(::PostCorrelationState)
```

## Two- and four-quadrant discriminators

The carrier discriminators are two-quadrant (Costas) by default: they read a
prompt and its negation alike, so a data bit or secondary-code chip flipping the
prompt's sign does not disturb them, over ±1/4 cycle for the PLL and ±1/(4T) for
the FLL. Where the replica wipes every sign modulation off the prompt (a
dataless signal synced to its secondary code) the prompt's sign is known, and
the loops switch to the four-quadrant discriminators, over ±1/2 cycle and
±1/(2T), worth up to 6 dB of tracking threshold. The latest integration's
`last_carrier_phase_polarity`, the prompt's sign, says so; [`process_correlator_bank_output!`](@ref)
computes it into the [`PostCorrelationState`](@ref), which carries it to the scalar and
vector loops alike. An integration whose polarity differs from the previous
integration's comes without a previous prompt, so the four-quadrant FLL never compares
prompts across the switch.

GPS L1C-P and L5Q, Galileo E1C, E5a-Q, E5b-Q and E6-C, and BeiDou B1C-P and
B2a-Q switch once their secondary code is synced; if the Costas loop slipped
half a cycle between the sync and the switch, the four-quadrant PLL pulls the
carrier phase over by half a cycle, after which the prompt agrees with the sign
again. Data signals never switch, nor, for now, do the pilots without a
secondary code (GPS L2 CL, Galileo E5a-QP), whose sync reads no sign.
The NCO-referenced loop folds its predicted phase error into whichever range the
integration's PLL reads.

The sign rests on a contract with whoever correlates the integrations, a hardware
correlator as much as Tracking.jl: until the sync, these signals' integrations
([`is_dataless_with_secondary_code`](@ref)) are primary-only. A replica carrying
the same secondary chip on every code period is fine once the host multiplies the integrations
by that chip (Tracking.jl's carries chip 0 and multiplies by it). A negative chip
left in makes the sync read the wrong sign, and on the switch the four-quadrant PLL
pulls the carrier phase over by half a cycle.

## Tracking loops

```@docs
ScalarTrackingLoop
DelayCompensatedTrackingLoop
TrackingLoops.ScalarLoopState
get_carrier_loop_stage(::ScalarLoopState)
TrackingLoops.calc_carrier_phase_correction(::ScalarLoopState, ::Any, ::Any, ::Any)
TrackingLoops.loop_state_type(::AbstractTrackingLoop, ::AbstractGNSSSignal)
init_loop_state(::ScalarTrackingLoop{CA, CO}, ::AbstractGNSSSignal, ::Any, ::Any) where {CA<:AbstractLoopFilter, CO<:AbstractLoopFilter}
get_navigation_cycle_index(::AbstractTrackingLoop)
get_navigation_elapsed_time(::AbstractTrackingLoop)
get_navigation_solution(::AbstractTrackingLoop)
get_navigation_status(::AbstractTrackingLoop)
TrackingLoops.get_phase_lock_indicator(::ScalarLoopState)
TrackingLoops.is_phase_locked(::ScalarLoopState)
reinit_loop_state(::AbstractTrackingLoop, ::ScalarLoopState, ::Any, ::Any)
get_satellite_report(::AbstractTrackingLoop, ::AbstractGNSSSignal, ::Integer)
step_loop!(::ScalarTrackingLoop, ::ScalarLoopState, ::PostCorrelationState, ::Any, ::Union{Nothing,Integer})
step_loop!(::DelayCompensatedTrackingLoop, ::ScalarLoopState, ::PostCorrelationState, ::Any, ::Union{Nothing,Integer})
```

## Carrier-loop staging

The scalar loops stage their carrier loop on phase lock: FLL-assisted at the
wide PLL bandwidth, then a pure PLL at the wide bandwidth, then narrow, each
stage moving on once the phase-lock indicator ([`get_phase_lock_indicator`](@ref TrackingLoops.get_phase_lock_indicator))
has read lock ([`is_phase_locked`](@ref TrackingLoops.is_phase_locked): at least 0.5, an RMS phase error of about
30°) long enough. The stages and when they advance are given under
[`CarrierLoopStage`](@ref TrackingLoops.CarrierLoopStage); in the vector loop, see [The carrier in the vector
loop](@ref vector-carrier).

A wider loop tolerates less command delay: at 50 Hz the conventional loop holds
through three integrations of delay. The [`DelayCompensatedTrackingLoop`](@ref), meant for
hardware correlators whose commands take effect integrations later, therefore pulls in at
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
bandwidth, default or explicit, is capped against the coherent integration time
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

| Integration time | Signals                                  |  Wide | Narrow | FLL-assist |  Code |
|------------------|------------------------------------------|------:|-------:|-----------:|------:|
| 1 ms             | GPS L1 C/A, GPS L5, Galileo E5a, …       |    50 |     18 |          5 |     1 |
| 2 ms             | Galileo E5a-QP                           |    45 |     18 |          5 |     1 |
| 4 ms             | Galileo E1B / E1C                        |  22.5 |     18 |          5 |     1 |
| 10 ms            | GPS L1C, BeiDou B1C; L5I synced at 10 ms |     9 |      9 |          2 |     1 |
| 20 ms            | GPS L2 CM; L1 C/A, L5Q, E5a-I synced     |   4.5 |    4.5 |          1 |   0.9 |
| 1.5 s            | GPS L2 CL                                |  0.06 |   0.06 |      0.013 | 0.012 |

The code cap of 0.018 is conservative (see
[`MAX_CODE_LOOP_BANDWIDTH_TIME_PRODUCT`](@ref TrackingLoops.MAX_CODE_LOOP_BANDWIDTH_TIME_PRODUCT)); the default 1 Hz applies in full
up to 18 ms.

```@docs
TrackingLoops.MAX_CARRIER_LOOP_BANDWIDTH_TIME_PRODUCT
TrackingLoops.MAX_CODE_LOOP_BANDWIDTH_TIME_PRODUCT
TrackingLoops.MAX_FLL_ASSIST_LOOP_BANDWIDTH_TIME_PRODUCT
TrackingLoops.LoopBandwidths
TrackingLoops.calc_carrier_aided_dopplers(::AbstractGNSSSignal, ::Any, ::Any, ::Any, ::Any)
TrackingLoops.step_code_loop(::AbstractGNSSSignal, ::AbstractLoopFilter, ::AbstractCorrelatorBank, ::Any, ::Any, ::Any, ::Any)
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
TrackingLoops.calc_num_bit_synchronizer_code_periods(::AbstractGNSSSignal, ::Integer, ::Any, ::Bool)
TrackingLoops.calc_num_code_periods_to_integrate(::AbstractGNSSSignal, ::Int, ::Bool)
TrackingLoops.default_num_code_periods_to_integrate(::AbstractGNSSSignal)
TrackingLoops.max_num_code_periods_to_integrate(::AbstractGNSSSignal)
```
