"""
    MAX_CARRIER_LOOP_BANDWIDTH_TIME_PRODUCT

The largest carrier-loop bandwidth–update interval product `BL · Δt`, `0.09`, that
[`effective_carrier_loop_filter_bandwidth`](@ref) caps at. The default FLL-assisted
third-order filter diverges at `BL · Δt ≈ 0.4`, and at `0.09` its noise bandwidth
already runs 25 % wider than configured.
"""
const MAX_CARRIER_LOOP_BANDWIDTH_TIME_PRODUCT = 0.09

"""
    MAX_CODE_LOOP_BANDWIDTH_TIME_PRODUCT

The largest code-loop bandwidth–update interval product `BL · Δt`, `0.018`, that
[`effective_code_loop_filter_bandwidth`](@ref) caps at. Conservative: the
second-order code filter destabilizes only around `BL · Δt ≈ 0.4` (Stephens &
Thomas 1995, "Controlled-Root Formulation for Digital Phase-Locked Loops", IEEE
Trans. AES 31(1)).
"""
const MAX_CODE_LOOP_BANDWIDTH_TIME_PRODUCT = 0.018

"""
    MAX_FLL_ASSIST_LOOP_BANDWIDTH_TIME_PRODUCT

The largest bandwidth–update interval product `BL · Δt` of an FLL-assisted carrier
filter's FLL path, `0.02`, that [`effective_fll_assist_loop_filter_bandwidth`](@ref)
caps at: 5 Hz up to 4 ms records, 2 Hz at 10 ms.
"""
const MAX_FLL_ASSIST_LOOP_BANDWIDTH_TIME_PRODUCT = 0.02

# Returned by the generic `default_*_loop_filter_bandwidth` methods; also the
# defaults of a `LoopBandwidths` built without a signal.
const _DEFAULT_WIDE_CARRIER_LOOP_FILTER_BANDWIDTH = 50.0Hz
const _DEFAULT_NARROW_CARRIER_LOOP_FILTER_BANDWIDTH = 18.0Hz
const _DEFAULT_FLL_ASSIST_LOOP_FILTER_BANDWIDTH = 5.0Hz
const _DEFAULT_CODE_LOOP_FILTER_BANDWIDTH = 1.0Hz

"""
$(SIGNATURES)

Recommended wide (pull-in) carrier-loop-filter bandwidth for `signal`: 50 Hz, the
third-order PLL's noise bandwidth until phase lock has held (see
[`CarrierLoopStage`](@ref)). Capped at filter time by
[`effective_carrier_loop_filter_bandwidth`](@ref): 50 Hz at 1 ms, 22.5 Hz at 4 ms,
9 Hz at 10 ms. [`NCOReferencedPLLAndDLL`](@ref) does not read it (its wide default
is the narrow one).

Override by defining a method for your signal type, or pass
`wide_carrier_loop_filter_bandwidth =` to the estimator.
"""
function default_wide_carrier_loop_filter_bandwidth(signal::AbstractGNSSSignal)
    _DEFAULT_WIDE_CARRIER_LOOP_FILTER_BANDWIDTH
end

"""
$(SIGNATURES)

Recommended narrow (tracking) carrier-loop-filter bandwidth for `signal`: 18 Hz,
the third-order PLL bandwidth of the literature (Kaplan & Hegarty, Pany), which the
loop narrows to once phase lock has held (see [`CarrierLoopStage`](@ref)). Capped
at filter time by [`effective_carrier_loop_filter_bandwidth`](@ref): 18 Hz up to
5 ms, 9 Hz at 10 ms, 4.5 Hz at 20 ms.

Override by defining a method for your signal type, or pass
`narrow_carrier_loop_filter_bandwidth =` to the estimator.
"""
function default_narrow_carrier_loop_filter_bandwidth(signal::AbstractGNSSSignal)
    _DEFAULT_NARROW_CARRIER_LOOP_FILTER_BANDWIDTH
end

"""
$(SIGNATURES)

Recommended FLL-assist bandwidth for `signal`: 5 Hz, the noise bandwidth of the
second-order FLL path of an FLL-assisted carrier filter while it is FLL-assisted
(the `fll` half of its `(pll, fll)` bandwidth pair). Capped at filter time by
[`effective_fll_assist_loop_filter_bandwidth`](@ref).

Override by defining a method for your signal type, or pass
`fll_assist_loop_filter_bandwidth =` to the estimator.
"""
function default_fll_assist_loop_filter_bandwidth(signal::AbstractGNSSSignal)
    _DEFAULT_FLL_ASSIST_LOOP_FILTER_BANDWIDTH
end

"""
$(SIGNATURES)

Recommended code-loop-filter (DLL) bandwidth for `signal`: 1 Hz, within the
0.25–2 Hz of GNSS-SDR, SoftGNSS and PocketSDR. Carrier-aided (see
[`aid_dopplers`](@ref)), the DLL has almost no dynamics left, so this is a
noise-versus-pull-in trade independent of the signal. Capped at filter time by
[`effective_code_loop_filter_bandwidth`](@ref).

Override by defining a method for your signal type.
"""
function default_code_loop_filter_bandwidth(signal::AbstractGNSSSignal)
    _DEFAULT_CODE_LOOP_FILTER_BANDWIDTH
end

"""
    LoopBandwidths(; wide_carrier = 50Hz, narrow_carrier = 18Hz, fll_assist = 5Hz,
                     code = 1Hz)
    LoopBandwidths(signal; wide_carrier, narrow_carrier, fll_assist, code)

A scalar estimator's per-satellite bandwidths: the PLL's before and after it
narrows, the FLL path's while FLL-assisted (see [`CarrierLoopStage`](@ref)), and
the DLL's. Given a `signal`, each one not given is the signal's
`default_*_loop_filter_bandwidth`, so methods of those for a signal type apply.
[`init_estimator_state`](@ref) resolves them from the estimator's, each `nothing`
there from the driver signal's default. Each is capped at filter time
([`effective_carrier_loop_filter_bandwidth`](@ref),
[`effective_fll_assist_loop_filter_bandwidth`](@ref),
[`effective_code_loop_filter_bandwidth`](@ref)).
"""
@kwdef struct LoopBandwidths
    wide_carrier::typeof(1.0Hz) = _DEFAULT_WIDE_CARRIER_LOOP_FILTER_BANDWIDTH
    narrow_carrier::typeof(1.0Hz) = _DEFAULT_NARROW_CARRIER_LOOP_FILTER_BANDWIDTH
    fll_assist::typeof(1.0Hz) = _DEFAULT_FLL_ASSIST_LOOP_FILTER_BANDWIDTH
    code::typeof(1.0Hz) = _DEFAULT_CODE_LOOP_FILTER_BANDWIDTH
end

LoopBandwidths(
    signal::AbstractGNSSSignal;
    wide_carrier = default_wide_carrier_loop_filter_bandwidth(signal),
    narrow_carrier = default_narrow_carrier_loop_filter_bandwidth(signal),
    fll_assist = default_fll_assist_loop_filter_bandwidth(signal),
    code = default_code_loop_filter_bandwidth(signal),
) = LoopBandwidths(wide_carrier, narrow_carrier, fll_assist, code)

"""
$(SIGNATURES)

PLL bandwidth for a record of `integration_time`, wide or narrow alike: `bandwidth`
capped at `MAX_CARRIER_LOOP_BANDWIDTH_TIME_PRODUCT / integration_time` for
stability on long integrations. Only ever narrows, an explicit bandwidth as much as
a default.
"""
@inline function effective_carrier_loop_filter_bandwidth(bandwidth, integration_time)
    min(
        bandwidth,
        uconvert(Hz, MAX_CARRIER_LOOP_BANDWIDTH_TIME_PRODUCT / integration_time),
    )
end

"""
$(SIGNATURES)

Code-loop bandwidth for a record of `integration_time`: `bandwidth` capped at
`MAX_CODE_LOOP_BANDWIDTH_TIME_PRODUCT / integration_time`. Integration time does
not otherwise narrow the DLL, as neither its dynamics nor its noise floor depend on
it. The default 1 Hz is capped only past 18 ms (0.9 Hz at 20 ms).
"""
@inline function effective_code_loop_filter_bandwidth(bandwidth, integration_time)
    min(bandwidth, uconvert(Hz, MAX_CODE_LOOP_BANDWIDTH_TIME_PRODUCT / integration_time))
end

"""
$(SIGNATURES)

FLL-assist bandwidth for a record of `integration_time`: `bandwidth` capped at
`MAX_FLL_ASSIST_LOOP_BANDWIDTH_TIME_PRODUCT / integration_time`.
"""
@inline function effective_fll_assist_loop_filter_bandwidth(bandwidth, integration_time)
    min(
        bandwidth,
        uconvert(Hz, MAX_FLL_ASSIST_LOOP_BANDWIDTH_TIME_PRODUCT / integration_time),
    )
end

"""
$(SIGNATURES)

Add the loop-filter updates to the initial Dopplers, carrier-aiding the code
Doppler: the carrier update times the code-to-carrier frequency ratio is added to it.
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
        code_freq_update + carrier_doppler * get_chip_rate_to_center_frequency_ratio(signal)
    init_carrier_doppler + carrier_doppler, init_code_doppler + code_doppler
end

"""
    calc_code_frequency_update(signal, code_loop_filter, correlator, code_doppler,
                               sampling_frequency, integration_time, loop_bandwidth)
        -> (code_freq_update, code_loop_filter)

One code-loop step: the DLL discriminator ([`dll_disc`](@ref)) of `correlator`
filtered by `code_loop_filter` at `loop_bandwidth`. `code_doppler`, the replica's,
converts the tap spacing from samples to chips. Returns the code-frequency
correction (before [`aid_dopplers`](@ref)) and the advanced filter.
"""
function calc_code_frequency_update(
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

# Rotates a component's bit-buffer prompt onto the real axis, given the loops lock
# the driver there: a bit-identical no-op (`cispi(0)`) for an in-phase component,
# `±90°` for a quadrature one (GPS L5 / Galileo E5a I vs Q). The offsets are in cycles.
@inline _carrier_phase_derotation(driver_carrier_phase_offset::Real, signal) =
    cispi(2 * (driver_carrier_phase_offset - get_carrier_phase_offset(signal)))
