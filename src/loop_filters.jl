"""
    MAX_CARRIER_LOOP_BANDWIDTH_TIME_PRODUCT

The largest product `B_n T` of the carrier loop's noise bandwidth and the coherent
integration time, `0.09`, that [`calc_effective_carrier_loop_bandwidth`](@ref) caps at.
The third-order filter, FLL-assisted or not, diverges at `B_n T ≈ 0.65`, and at `0.09` its
noise bandwidth already runs 16 % wider than configured.
"""
const MAX_CARRIER_LOOP_BANDWIDTH_TIME_PRODUCT = 0.09

"""
    MAX_CODE_LOOP_BANDWIDTH_TIME_PRODUCT

The largest product `B_n T` of the code loop's noise bandwidth and the coherent
integration time, `0.018`, that [`calc_effective_code_loop_bandwidth`](@ref) caps at.
Conservative: the second-order code filter destabilizes only around `B_n T ≈ 0.4` (Stephens &
Thomas 1995, "Controlled-Root Formulation for Digital Phase-Locked Loops", IEEE
Trans. AES 31(1)).
"""
const MAX_CODE_LOOP_BANDWIDTH_TIME_PRODUCT = 0.018

"""
    MAX_FLL_ASSIST_LOOP_BANDWIDTH_TIME_PRODUCT

The largest product `B_n T` of an FLL-assisted carrier filter's FLL-path noise bandwidth
and the coherent integration time, `0.02`, that
[`calc_effective_fll_assist_loop_bandwidth`](@ref) caps at: 5 Hz up to 4 ms integrations,
2 Hz at 10 ms.
"""
const MAX_FLL_ASSIST_LOOP_BANDWIDTH_TIME_PRODUCT = 0.02

# Returned by the generic `default_*_loop_bandwidth` methods; also the
# defaults of a `LoopBandwidths` built without a signal.
const _DEFAULT_WIDE_CARRIER_LOOP_BANDWIDTH = 50.0Hz
const _DEFAULT_NARROW_CARRIER_LOOP_BANDWIDTH = 18.0Hz
const _DEFAULT_FLL_ASSIST_LOOP_BANDWIDTH = 5.0Hz
const _DEFAULT_CODE_LOOP_BANDWIDTH = 1.0Hz

"""
$(SIGNATURES)

Recommended wide (pull-in) carrier-loop bandwidth for `signal`: 50 Hz, the
third-order PLL's noise bandwidth until phase lock has held (see
[`CarrierLoopStage`](@ref)). Capped at filter time by
[`calc_effective_carrier_loop_bandwidth`](@ref): 50 Hz at 1 ms, 22.5 Hz at 4 ms,
9 Hz at 10 ms. [`DelayCompensatedTrackingLoop`](@ref) does not read it (its wide default
is the narrow one).

Override by defining a method for your signal type, or pass
`wide_carrier_loop_bandwidth =` to the tracking loop.
"""
function default_wide_carrier_loop_bandwidth(signal::AbstractGNSSSignal)
    _DEFAULT_WIDE_CARRIER_LOOP_BANDWIDTH
end

"""
$(SIGNATURES)

Recommended narrow (tracking) carrier-loop bandwidth for `signal`: 18 Hz, which the
loop narrows to once phase lock has held (see [`CarrierLoopStage`](@ref)). It is not a
typical bandwidth but Kaplan & Hegarty's stability limit of a third-order PLL at a
predetection integration time `T` of 5 ms (Table 5.6); `18 Hz × 5 ms = 0.09` is
`MAX_CARRIER_LOOP_BANDWIDTH_TIME_PRODUCT`. Capped at filter time by
[`calc_effective_carrier_loop_bandwidth`](@ref): 18 Hz up to 5 ms, 9 Hz at 10 ms,
4.5 Hz at 20 ms.

Override by defining a method for your signal type, or pass
`narrow_carrier_loop_bandwidth =` to the tracking loop.
"""
function default_narrow_carrier_loop_bandwidth(signal::AbstractGNSSSignal)
    _DEFAULT_NARROW_CARRIER_LOOP_BANDWIDTH
end

"""
$(SIGNATURES)

Recommended FLL-assist bandwidth for `signal`: 5 Hz, the noise bandwidth of the
second-order FLL path of an FLL-assisted carrier filter while it is FLL-assisted
(the `fll` half of its `(pll, fll)` bandwidth pair). Capped at filter time by
[`calc_effective_fll_assist_loop_bandwidth`](@ref).

Override by defining a method for your signal type, or pass
`fll_assist_loop_bandwidth =` to the tracking loop.
"""
function default_fll_assist_loop_bandwidth(signal::AbstractGNSSSignal)
    _DEFAULT_FLL_ASSIST_LOOP_BANDWIDTH
end

"""
$(SIGNATURES)

Recommended code-loop-filter (DLL) bandwidth for `signal`: 1 Hz, within the
0.25–2 Hz of GNSS-SDR, SoftGNSS and PocketSDR. Carrier-aided (see
[`calc_carrier_aided_dopplers`](@ref)), the DLL has almost no dynamics left, so this is a
noise-versus-pull-in trade independent of the signal. Capped at filter time by
[`calc_effective_code_loop_bandwidth`](@ref).

Override by defining a method for your signal type.
"""
function default_code_loop_bandwidth(signal::AbstractGNSSSignal)
    _DEFAULT_CODE_LOOP_BANDWIDTH
end

"""
    LoopBandwidths(; wide_carrier = 50Hz, narrow_carrier = 18Hz, fll_assist = 5Hz,
                     code = 1Hz)
    LoopBandwidths(signal; wide_carrier, narrow_carrier, fll_assist, code)

A scalar loop's per-satellite bandwidths: the PLL's before and after it
narrows, the FLL path's while FLL-assisted (see [`CarrierLoopStage`](@ref)), and
the DLL's. Given a `signal`, each one not given is the signal's
`default_*_loop_bandwidth`, so methods of those for a signal type apply.
[`init_loop_state`](@ref) resolves them from the tracking loop's, each `nothing`
there from the driver signal's default. Each is capped at filter time
([`calc_effective_carrier_loop_bandwidth`](@ref),
[`calc_effective_fll_assist_loop_bandwidth`](@ref),
[`calc_effective_code_loop_bandwidth`](@ref)).
"""
@kwdef struct LoopBandwidths
    wide_carrier::typeof(1.0Hz) = _DEFAULT_WIDE_CARRIER_LOOP_BANDWIDTH
    narrow_carrier::typeof(1.0Hz) = _DEFAULT_NARROW_CARRIER_LOOP_BANDWIDTH
    fll_assist::typeof(1.0Hz) = _DEFAULT_FLL_ASSIST_LOOP_BANDWIDTH
    code::typeof(1.0Hz) = _DEFAULT_CODE_LOOP_BANDWIDTH
end

LoopBandwidths(
    signal::AbstractGNSSSignal;
    wide_carrier = default_wide_carrier_loop_bandwidth(signal),
    narrow_carrier = default_narrow_carrier_loop_bandwidth(signal),
    fll_assist = default_fll_assist_loop_bandwidth(signal),
    code = default_code_loop_bandwidth(signal),
) = LoopBandwidths(wide_carrier, narrow_carrier, fll_assist, code)

# Both scalar strategies keep the same optional bandwidth configuration. Resolve
# auto values once for each satellite's driver signal, into concrete LoopBandwidths.
@kwdef struct _LoopBandwidthConfiguration
    wide_carrier_loop_bandwidth::Maybe{typeof(1.0Hz)} = nothing
    narrow_carrier_loop_bandwidth::Maybe{typeof(1.0Hz)} = nothing
    fll_assist_loop_bandwidth::Maybe{typeof(1.0Hz)} = nothing
    code_loop_bandwidth::Maybe{typeof(1.0Hz)} = nothing
end

function _LoopBandwidthConfiguration(
    configured::_LoopBandwidthConfiguration;
    wide_carrier_loop_bandwidth = nothing,
    narrow_carrier_loop_bandwidth = nothing,
    fll_assist_loop_bandwidth = nothing,
    code_loop_bandwidth = nothing,
)
    _LoopBandwidthConfiguration(
        isnothing(wide_carrier_loop_bandwidth) ? configured.wide_carrier_loop_bandwidth : wide_carrier_loop_bandwidth,
        isnothing(narrow_carrier_loop_bandwidth) ? configured.narrow_carrier_loop_bandwidth : narrow_carrier_loop_bandwidth,
        isnothing(fll_assist_loop_bandwidth) ? configured.fll_assist_loop_bandwidth : fll_assist_loop_bandwidth,
        isnothing(code_loop_bandwidth) ? configured.code_loop_bandwidth : code_loop_bandwidth,
    )
end

@inline _resolve_bandwidths(configured::_LoopBandwidthConfiguration, defaults::LoopBandwidths) =
    LoopBandwidths(
        something(configured.wide_carrier_loop_bandwidth, defaults.wide_carrier),
        something(configured.narrow_carrier_loop_bandwidth, defaults.narrow_carrier),
        something(configured.fll_assist_loop_bandwidth, defaults.fll_assist),
        something(configured.code_loop_bandwidth, defaults.code),
    )

"""
$(SIGNATURES)

PLL bandwidth for an integration of `coherent_integration_time`, wide or narrow alike: `bandwidth`
capped at `MAX_CARRIER_LOOP_BANDWIDTH_TIME_PRODUCT / coherent_integration_time` for
stability on long integrations. Only ever narrows, an explicit bandwidth as much as
a default.
"""
@inline function calc_effective_carrier_loop_bandwidth(bandwidth, coherent_integration_time)
    min(
        bandwidth,
        uconvert(Hz, MAX_CARRIER_LOOP_BANDWIDTH_TIME_PRODUCT / coherent_integration_time),
    )
end

"""
$(SIGNATURES)

Code-loop bandwidth for an integration of `coherent_integration_time`: `bandwidth` capped at
`MAX_CODE_LOOP_BANDWIDTH_TIME_PRODUCT / coherent_integration_time`. Integration time does
not otherwise narrow the DLL, as neither its dynamics nor its noise floor depend on
it. The default 1 Hz is capped only past 18 ms (0.9 Hz at 20 ms).
"""
@inline function calc_effective_code_loop_bandwidth(bandwidth, coherent_integration_time)
    min(bandwidth, uconvert(Hz, MAX_CODE_LOOP_BANDWIDTH_TIME_PRODUCT / coherent_integration_time))
end

"""
$(SIGNATURES)

FLL-assist bandwidth for an integration of `coherent_integration_time`: `bandwidth` capped at
`MAX_FLL_ASSIST_LOOP_BANDWIDTH_TIME_PRODUCT / coherent_integration_time`.
"""
@inline function calc_effective_fll_assist_loop_bandwidth(bandwidth, coherent_integration_time)
    min(
        bandwidth,
        uconvert(Hz, MAX_FLL_ASSIST_LOOP_BANDWIDTH_TIME_PRODUCT / coherent_integration_time),
    )
end

"""
$(SIGNATURES)

Add the loop-filter updates to the initial Dopplers, carrier-aiding the code
Doppler: the carrier update times the code-to-carrier frequency ratio is added to it.
"""
function calc_carrier_aided_dopplers(
    signal::AbstractGNSSSignal,
    initial_carrier_doppler,
    initial_code_doppler,
    carrier_frequency_correction,
    code_frequency_correction,
)
    carrier_doppler = carrier_frequency_correction
    code_doppler =
        code_frequency_correction + carrier_doppler * get_chip_rate_to_center_frequency_ratio(signal)
    initial_carrier_doppler + carrier_doppler, initial_code_doppler + code_doppler
end
