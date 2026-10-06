"""
    MAX_CARRIER_LOOP_BANDWIDTH_TIME_PRODUCT

The largest carrier-loop bandwidth–update interval product `BL · Δt`: `0.09`.
[`effective_carrier_loop_filter_bandwidth`](@ref) caps the carrier bandwidth
at `0.09 / Δt`. The default FLL-assisted third-order filter diverges at
`BL · Δt ≈ 0.4`, and at `0.09` its noise bandwidth runs 25 % wider than
configured.
"""
const MAX_CARRIER_LOOP_BANDWIDTH_TIME_PRODUCT = 0.09

"""
    MAX_CODE_LOOP_BANDWIDTH_TIME_PRODUCT

The largest code-loop bandwidth–update interval product `BL · Δt`: `0.018`.
[`effective_code_loop_filter_bandwidth`](@ref) caps the code bandwidth at
`0.018 / Δt`. Conservative for the second-order code filter, which
destabilizes only around `BL · Δt ≈ 0.4` (Stephens & Thomas 1995,
"Controlled-Root Formulation for Digital Phase-Locked Loops", IEEE Trans. AES
31(1)).
"""
const MAX_CODE_LOOP_BANDWIDTH_TIME_PRODUCT = 0.018

"""
$(SIGNATURES)

Recommended carrier-loop-filter bandwidth for `signal`: a flat 18 Hz, the
third-order PLL bandwidth of the literature (Kaplan & Hegarty, Pany). It is the
loop's one-sided noise bandwidth `BL`, seeded for every satellite whose
estimator leaves `carrier_loop_filter_bandwidth` as `nothing`, and capped at
filter time by [`effective_carrier_loop_filter_bandwidth`](@ref).

Override by defining a method for your signal type, or pass
`carrier_loop_filter_bandwidth =` to the estimator.
"""
function default_carrier_loop_filter_bandwidth(signal::AbstractGNSSSignal)
    18.0Hz
end

"""
$(SIGNATURES)

Recommended code-loop-filter (DLL) bandwidth for `signal`: a flat 1 Hz, inside
the 0.25–2 Hz of the reference software receivers (GNSS-SDR, SoftGNSS,
PocketSDR). Carrier-aided (see [`aid_dopplers`](@ref)), the DLL has almost no
dynamics to track, so the bandwidth is a thermal-noise-versus-pull-in trade
independent of the signal. Capped at filter time by
[`effective_code_loop_filter_bandwidth`](@ref).

Override by defining a method for your signal type.
"""
function default_code_loop_filter_bandwidth(signal::AbstractGNSSSignal)
    1.0Hz
end

"""
$(SIGNATURES)

Effective carrier-loop bandwidth for a record that integrated for
`integration_time`: the configured bandwidth, capped at
`MAX_CARRIER_LOOP_BANDWIDTH_TIME_PRODUCT / integration_time` to keep the loop
stable on long integrations. The default 18 Hz runs unchanged up to 5 ms of
integration, at 9 Hz for 10 ms and 4.5 Hz for 20 ms. The cap only ever narrows
a loop, an explicit bandwidth as much as the default.
"""
@inline function effective_carrier_loop_filter_bandwidth(bandwidth, integration_time)
    min(
        bandwidth,
        uconvert(Hz, MAX_CARRIER_LOOP_BANDWIDTH_TIME_PRODUCT / integration_time),
    )
end

"""
$(SIGNATURES)

Effective code-loop bandwidth for a record that integrated for
`integration_time`: the configured bandwidth, capped at
`MAX_CODE_LOOP_BANDWIDTH_TIME_PRODUCT / integration_time` for stability.
Longer integration must not otherwise narrow the DLL: neither its dynamics nor
its noise floor depend on it. At the default 1 Hz the cap binds only past
18 ms (0.9 Hz for a 20 ms integration, 0.012 Hz for a 1.5 s one).
"""
@inline function effective_code_loop_filter_bandwidth(bandwidth, integration_time)
    min(bandwidth, uconvert(Hz, MAX_CODE_LOOP_BANDWIDTH_TIME_PRODUCT / integration_time))
end

"""
$(SIGNATURES)

Window of the frequency lock indicator for `signal` at the record length
`integration_time`: the FLL is dropped once its mean reading over this window
stays below [`frequency_lock_threshold`](@ref). By default 0.5 s, and at least
four records. Override by defining a method for your signal type.
"""
frequency_lock_window(signal::AbstractGNSSSignal, integration_time) =
    max(0.5s, uconvert(s, 4 * integration_time))

"""
$(SIGNATURES)

Threshold of the frequency lock indicator for `signal` at the record length
`integration_time`, see [`frequency_lock_window`](@ref). By default 3 Hz, and at
most 1/(16T), a quarter of the two-quadrant FLL's range and an eighth of the
four-quadrant one's (0.04 Hz at GPS L2 CL's 1.5 s). Override by defining a method
for your signal type.
"""
frequency_lock_threshold(signal::AbstractGNSSSignal, integration_time) =
    min(3.0Hz, uconvert(Hz, 1 / (16 * integration_time)))

"""
    FrequencyLockIndicator()

The frequency lock indicator of a satellite's carrier loop, held in the scalar
estimators' per-satellite state. The carrier loop starts as an FLL-assisted PLL
and drops the FLL once the mean FLL reading over a
[`frequency_lock_window`](@ref) stays below [`frequency_lock_threshold`](@ref).
The lock latches; [`reset_estimator_state`](@ref) restarts the staging.

Fields:

  - `integrated_frequency_error`: FLL readings integrated over the current window;
  - `window_time`: length of the current window;
  - `locked`: whether frequency lock has been declared (latched).
"""
struct FrequencyLockIndicator
    integrated_frequency_error::typeof(1.0Hz * 1.0s)
    window_time::typeof(1.0s)
    locked::Bool
end

FrequencyLockIndicator() = FrequencyLockIndicator(0.0Hz * 0.0s, 0.0s, false)

# Advance the frequency lock indicator by one record's FLL reading, in windows
# of `frequency_lock_window`. The window mean is the phase advance across the
# window over its length, so its noise falls with the window length rather than
# with each record's SNR. A record without a previous prompt has no FLL reading
# and is left out; a latched lock is kept as is.
@inline function _update_frequency_lock(
    indicator::FrequencyLockIndicator,
    signal::AbstractGNSSSignal,
    fll_discriminator,
    previous_prompt::Complex,
    integration_time,
)
    (indicator.locked || iszero(previous_prompt)) && return indicator
    dt = uconvert(s, integration_time)
    integrated_error = indicator.integrated_frequency_error + fll_discriminator * dt
    window_time = indicator.window_time + dt
    window_time < frequency_lock_window(signal, integration_time) &&
        return FrequencyLockIndicator(integrated_error, window_time, false)
    locked =
        abs(integrated_error / window_time) <
        frequency_lock_threshold(signal, integration_time)
    FrequencyLockIndicator(0.0Hz * 0.0s, 0.0s, locked)
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

# The carrier filter's coefficients assume consistent units, so it is fed the
# phase error in cycles and the FLL error in Hz (cycles per second) to output a
# Doppler in Hz. Fed `pll_disc`'s radians, the loop gain was 2π too high, and
# the loop's noise bandwidth about 5–6× the configured one.
@inline _phase_error_in_cycles(phase_error_in_radians) = phase_error_in_radians / 2π

"""
    calculate_carrier_frequency_update(signal, carrier_loop_filter, correlator, previous_prompt, integration_time, loop_bandwidth)
        -> (carrier_freq_update, carrier_loop_filter)

One carrier-loop step: the PLL discriminator ([`pll_disc`](@ref)) of
`correlator`, converted to cycles, filtered by `carrier_loop_filter` at
`loop_bandwidth`. An FLL-assisted filter (`ThirdOrderAssistedBilinearLF`) is
additionally fed the FLL discriminator ([`fll_disc`](@ref)) between
`previous_prompt` and this record's prompt. Returns the carrier-frequency
correction in Hz and the advanced filter.
"""
function calculate_carrier_frequency_update(
    signal::AbstractGNSSSignal,
    carrier_loop_filter::ThirdOrderAssistedBilinearLF,
    correlator::AbstractCorrelator,
    previous_prompt::Complex,
    integration_time,
    loop_bandwidth,
)
    pll_discriminator = _phase_error_in_cycles(pll_disc(signal, correlator))
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
    pll_discriminator = _phase_error_in_cycles(pll_disc(signal, correlator))
    filter_loop(carrier_loop_filter, pll_discriminator, integration_time, loop_bandwidth)
end

"""
    calculate_code_frequency_update(signal, code_loop_filter, correlator, code_doppler, sampling_frequency, integration_time, loop_bandwidth)
        -> (code_freq_update, code_loop_filter)

One code-loop step: the DLL discriminator ([`dll_disc`](@ref)) of
`correlator`, filtered by `code_loop_filter` at `loop_bandwidth`. `code_doppler`
is the code Doppler the replica ran with, which the discriminator needs to
convert the tap spacing from samples to chips. Returns the code-frequency
correction (before carrier aiding, see [`aid_dopplers`](@ref)) and the advanced
filter.
"""
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
