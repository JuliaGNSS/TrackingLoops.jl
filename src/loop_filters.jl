# Target `BL · Δt` for a loop update interval of `Δt` — ~10× margin from the
# `BL · Δt < 0.18` practical stability edge of the bilinear third-order carrier
# filter. Sizes the carrier default against the primary code period, and caps
# the code loop against each record's actual integration time (see
# `effective_code_loop_filter_bandwidth`). For the code loop the shared product
# is a conservative reuse: its filter is the *second*-order bilinear one, which
# destabilizes only around the classic `BL · Δt ≈ 0.4` of transform-designed
# digital loops (Stephens & Thomas 1995, "Controlled-Root Formulation for
# Digital Phase-Locked Loops", IEEE Trans. AES 31(1) — this implementation's
# linearized edge lands there numerically too), so the cap only adds margin.
const MAX_LOOP_BANDWIDTH_TIME_PRODUCT = 0.018

"""
$(SIGNATURES)

Recommended carrier-loop-filter bandwidth for `signal`'s primary integration
period. Sized so that the PLL time-bandwidth product `BL * T` lands at
about 0.018 (≈10× margin from the 0.18 stability edge of the bilinear
third-order filter). Used by [`Tracking.TrackState`](@ref) when the
user doesn't pass an explicit `doppler_estimator`.

Override by defining a method for your signal type, or by constructing
[`ConventionalAssistedPLLAndDLL`](@ref) yourself with explicit
`carrier_loop_filter_bandwidth =` / `code_loop_filter_bandwidth =` kwargs.

```julia
T = get_code_length(signal) / get_code_frequency(signal)   # primary period
BL = 0.018 / T                                              # this default
```

`T` here is the **primary**-code period, not the chosen coherent
integration length. For GPS L1 C/A (T = 1 ms) and GPS L5I (T = 1 ms, a
10230-chip code at 10.23 MHz) this returns 18 Hz — matching the historical
hand-picked default. For L1C-D / L1C-P (T = 10 ms) it returns 1.8 Hz, and
for Galileo E1B (T = 4 ms) 4.5 Hz — the well-inside-stability values the
multi-signal flagship use case needs.

This value is the **reference** bandwidth for a one-primary-code-period
integration; it is not the bandwidth that ends up in the loop when you
integrate longer. Coherently integrating `N` primary blocks grows the loop
update interval to `N·T`, which would push `BL·N·T` toward the ~0.18
stability edge of the bilinear filter. To avoid that, the conventional
estimator **automatically scales the effective loop bandwidth by
`1/N`** at filter time (see [`ConventionalPLLAndDLL`](@ref)), holding the
`BL·Δt` stability product fixed at its single-period value. So you set this
reference bandwidth once and the loop stays stable at any integration length
— no manual `1/N` adjustment is needed.
"""
function default_carrier_loop_filter_bandwidth(signal::AbstractGNSSSignal)
    # T = the primary code period — one code block, not the chosen coherent
    # integration length. The estimator's bandwidth fields are typed
    # `typeof(1.0Hz)`, so explicitly land on Hz (otherwise `1/s` propagates and
    # trips the typed field assignment).
    primary_period = get_code_length(signal) / get_code_frequency(signal)
    uconvert(Hz, MAX_LOOP_BANDWIDTH_TIME_PRODUCT / primary_period)
end

"""
$(SIGNATURES)

Recommended code-loop-filter (DLL) bandwidth for `signal`: a flat 1 Hz for every
signal.

A *carrier-aided* DLL has almost no dynamic stress to track — the code Doppler
is handed to it by the PLL (see `aid_dopplers`) — so its bandwidth is a
thermal-noise-versus-pull-in trade that scales with neither the symbol rate
(the old 18:1 carrier:code ratio starved the long-primary signals' pull-in) nor
the coherent integration length. 1 Hz sits inside the 0.25–2 Hz the reference
software receivers (GNSS-SDR, SoftGNSS, PocketSDR) use across signals.

Unlike the carrier bandwidth this is an **absolute** value, not a
per-primary-code-period reference. Only the loop's own `BL · Δt` stability
product caps it, at filter time, against each record's actual integration time
— see [`effective_code_loop_filter_bandwidth`](@ref); the cap binds only past
18 ms (0.9 Hz for a 20 ms L2 CM integration, 0.012 Hz for a 1.5 s L2 CL one).

Override by defining a method for your signal type.
"""
function default_code_loop_filter_bandwidth(signal::AbstractGNSSSignal)
    1.0Hz
end

"""
$(SIGNATURES)

Effective code-loop bandwidth for a record that integrated for
`integration_time`: the configured bandwidth, capped so the code loop's
`BL · Δt` product stays inside `MAX_LOOP_BANDWIDTH_TIME_PRODUCT`.

The carrier loop takes a `1/N` scaling instead, because its configured
bandwidth is a per-primary-code-period *reference* — see
[`ConventionalPLLAndDLL`](@ref). The DLL's is an absolute value: carrier-aided,
it has no dynamic stress that grows with the integration length, and neither its
pull-in time nor its thermal-noise floor does either, so integrating longer must
not narrow it. Only stability may, and stability depends on the update interval
the record actually had — hence the cap against `integration_time` rather than a
scaling by the block count. A `1/N` here would take a 20 ms L1 C/A integration
down to 0.05 Hz where stability allows 0.9 Hz, re-introducing through the
integration length exactly the pull-in sag that sizing the DLL off the carrier
default used to cause by signal.

For a single-block integration of any signal at or below the 18 ms period where
the cap starts to bind, this returns the configured bandwidth unchanged.
"""
@inline function effective_code_loop_filter_bandwidth(bandwidth, integration_time)
    min(bandwidth, uconvert(Hz, MAX_LOOP_BANDWIDTH_TIME_PRODUCT / integration_time))
end

"""
$(SIGNATURES)

Aid dopplers. That is velocity aiding for the carrier doppler and carrier aiding
for the code doppler.
"""
function aid_dopplers(
    signal::AbstractGNSSSignal,
    init_carrier_doppler,
    init_code_doppler,
    carrier_freq_update,
    code_freq_update,
)
    carrier_doppler = carrier_freq_update
    code_doppler =
        code_freq_update + carrier_doppler * get_code_center_frequency_ratio(signal)
    init_carrier_doppler + carrier_doppler, init_code_doppler + code_doppler
end


function calculate_carrier_frequency_update(
    signal::AbstractGNSSSignal,
    carrier_loop_filter::ThirdOrderAssistedBilinearLF,
    correlator::AbstractCorrelator,
    previous_prompt::Complex,
    integration_time,
    loop_bandwidth,
)
    pll_discriminator = pll_disc(signal, correlator)
    fll_discriminator = fll_disc(signal, correlator, previous_prompt, integration_time)
    filter_loop(
        carrier_loop_filter,
        (pll_discriminator, fll_discriminator),
        integration_time,
        loop_bandwidth,
    )
end

function calculate_carrier_frequency_update(
    signal::AbstractGNSSSignal,
    carrier_loop_filter::AbstractLoopFilter,
    correlator::AbstractCorrelator,
    previous_prompt::Complex,
    integration_time,
    loop_bandwidth,
)
    pll_discriminator = pll_disc(signal, correlator)
    filter_loop(carrier_loop_filter, pll_discriminator, integration_time, loop_bandwidth)
end

function calculate_code_frequency_update(
    signal::AbstractGNSSSignal,
    code_loop_filter::AbstractLoopFilter,
    correlator::AbstractCorrelator,
    code_doppler,
    sampling_frequency,
    integration_time,
    loop_bandwidth,
)
    dll_discriminator = dll_disc(signal, correlator, code_doppler, sampling_frequency)
    filter_loop(code_loop_filter, dll_discriminator, integration_time, loop_bandwidth)
end

# De-rotation applied to a component's bit-buffer prompt so its own energy is
# real again, given the loops lock the driver onto the real axis. The rotation
# is `cis(driver_carrier_phase − get_carrier_phase_offset(signal))`. For an
# in-phase component the difference is 0 and `cis(0) === 1 + 0im`, a
# bit-identical no-op; a quadrature component (GPS L5 / Galileo E5a I-vs-Q)
# rotates by `±90°` onto the real axis.
@inline _carrier_phase_derotation(driver_carrier_phase::Real, signal) =
    cis(driver_carrier_phase - get_carrier_phase_offset(signal))
