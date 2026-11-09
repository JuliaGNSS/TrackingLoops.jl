```@meta
CurrentModule = TrackingLoops
```

# Glossary

The package's terms and the textbook terms they stand for (Kaplan & Hegarty,
*Understanding GPS/GNSS*; Misra & Enge, *Global Positioning System*; Spilker et al.,
*Global Positioning System: Theory and Applications*).

- **Integration** (coherent integration): Kaplan and Hegarty's predetection
  integration, the correlator outputs of one signal summed over one or more primary
  code periods ahead of the discriminators; one hardware dump or a sum of dumps.
  A [`PostCorrelationState`](@ref) holds the latest one of a signal.
- **Batch, chunk**: what a host processes at once. A chunk is a span of samples a
  software receiver correlates in one call (Tracking.jl's `track!`); a batch is the
  integrations of one signal a host hands over together, which share a command issue
  sample index. Neither changes what an integration measures.
- **Bit vs symbol**: outside GNSSDecoder a *bit* is a navigation bit as tracked,
  i.e. a channel symbol before error-correction decoding where the signal has FEC
  (GPS L5, L2C, Galileo E1B, BeiDou B-CNAV), as in GNSSSignals.
- **Driver, passenger**: of a satellite's signals on one carrier, the *driver* is
  the one whose integrations close the carrier and code loops; a *passenger* rides on
  the driver's replica and is only demodulated or used for C/N₀. For GPS L1 C/A,
  a signal without a pilot, the signal is its own driver. For a pilot + data pair
  (GPS L5Q + L5I, Galileo E1C + E1B) the pilot drives and the data component is the
  passenger (the *decoding signal* in vector tracking).
- **Slot**: the navigation processor's per-satellite record in vector tracking (a
  receiver channel in the textbook sense, holding both the driver and the decoding
  signal); a host's hardware channel is per signal.
- **Member**: a satellite in the vector loop, on one ranging signal.
- **Used measurements**: the members whose measurements a navigation filter update
  takes: measurement-ready, with accumulated discriminators, above the horizon. No
  measurement gate selects among them yet; a planned one would take these as its
  candidates.
- **Snapshot**: what a satellite latches at a measurement epoch for the next
  update: replica code phase, Dopplers, C/N₀, decoder and bit clock, and its loop state.
- **Measurement epoch**: the time point a navigation filter update estimates the state
  at, a multiple of the update interval `Δt_nav` on the integrations' sample grid.
- **Update**: one navigation filter update (time update and measurement update) at a
  measurement epoch; `Δt_nav` is the interval between two. *Cycles* are carrier-phase
  cycles only.
- **Feedback delay**: how long after the measurement epoch an update's NCO commands
  take effect; a hardware NCO applies a command at an *application sample index*,
  later than the *command issue sample index* where the loop computed it.
- **Residual PLL**: in vector tracking, the PLL that tracks the residual carrier phase
  of a member against the navigation filter's carrier ramp; it never steers the NCO.
- **Ramp**: the carrier Doppler ramp the navigation filter commands a member's NCO
  with, the predicted Doppler and its rate over the next update interval.
- **Staging**: the scalar carrier loop's stages (FLL-assisted PLL, wide PLL, narrow
  PLL) and the phase-lock indicator that moves it between them
  ([`CarrierLoopStage`](@ref)).
- **Loop bandwidth**: the closed-loop noise bandwidth `B_n` (Kaplan & Hegarty's
  `B_n`, one-sided), as every `*_loop_bandwidth` keyword sets it.

## Sign conventions

- **DLL discriminator** ([`dll_disc`](@ref TrackingLoops.dll_disc)): positive when the
  early correlator outweighs the late one, i.e. the replica code lags the incoming
  code; the code NCO then runs faster. In chips.
- **Carrier Doppler**: positive for a closing satellite (the received frequency above
  the nominal one).
- **Pseudorange rate**: positive while the satellite recedes, as PositionVelocityTime
  reports it; a carrier Doppler `f_D` corresponds to a pseudorange rate `-λ f_D`.
