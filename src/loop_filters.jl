"""
    MAX_CODE_LOOP_BANDWIDTH_TIME_PRODUCT

The largest code-loop bandwidth–update interval product `BL · Δt`, `0.018`, that
[`effective_code_loop_filter_bandwidth`](@ref) caps at. Conservative: the
second-order code filter destabilizes only around `BL · Δt ≈ 0.4` (Stephens &
Thomas 1995, "Controlled-Root Formulation for Digital Phase-Locked Loops", IEEE
Trans. AES 31(1)).
"""
const MAX_CODE_LOOP_BANDWIDTH_TIME_PRODUCT = 0.018

# The carrier loop's largest products `BL · Δt` (see `_capped_bandwidth`): wide
# 0.09 (the FLL-assisted third-order filter diverges at ≈ 0.4, and at 0.09 its noise
# bandwidth already runs 25 % wide), narrow 0.04, FLL path 0.02.
const _MAX_WIDE_CARRIER_LOOP_BANDWIDTH_TIME_PRODUCT = 0.09
const _MAX_NARROW_CARRIER_LOOP_BANDWIDTH_TIME_PRODUCT = 0.04
const _MAX_FLL_ASSIST_LOOP_BANDWIDTH_TIME_PRODUCT = 0.02

# Returned by the `default_*_loop_filter_bandwidth` methods; also the defaults of a
# per-satellite state built without a signal.
const _DEFAULT_WIDE_CARRIER_LOOP_FILTER_BANDWIDTH = 50.0Hz
const _DEFAULT_NARROW_CARRIER_LOOP_FILTER_BANDWIDTH = 18.0Hz
const _DEFAULT_FLL_ASSIST_LOOP_FILTER_BANDWIDTH = 5.0Hz
const _DEFAULT_CODE_LOOP_FILTER_BANDWIDTH = 1.0Hz

"""
$(SIGNATURES)

Recommended wide (pull-in) carrier-loop-filter bandwidth for `signal`: 50 Hz, the
PLL's noise bandwidth until phase lock (see [`CarrierLoopStage`](@ref)). Capped at
filter time at `0.09 / T` for integration time `T`: 50 Hz at 1 ms, 22.5 Hz at
4 ms, 9 Hz at 10 ms. Wider locks faster but loses lock near threshold (at 90 Hz,
1 ms records lose lock at 32.5 dB-Hz).

Override by defining a method for your signal type, or pass
`wide_carrier_loop_filter_bandwidth =` to the estimator.
[`NCOReferencedPLLAndDLL`](@ref) starts from the narrow default instead.
"""
function default_wide_carrier_loop_filter_bandwidth(signal::AbstractGNSSSignal)
    _DEFAULT_WIDE_CARRIER_LOOP_FILTER_BANDWIDTH
end

"""
$(SIGNATURES)

Recommended narrow (tracking) carrier-loop-filter bandwidth for `signal`: 18 Hz,
the PLL's noise bandwidth once phase lock is confirmed (see
[`CarrierLoopStage`](@ref)). Capped at filter time at `0.04 / T`: 18 Hz at 1 ms,
10 Hz at 4 ms, 4 Hz at 10 ms.

Override by defining a method for your signal type, or pass
`narrow_carrier_loop_filter_bandwidth =` to the estimator.
"""
function default_narrow_carrier_loop_filter_bandwidth(signal::AbstractGNSSSignal)
    _DEFAULT_NARROW_CARRIER_LOOP_FILTER_BANDWIDTH
end

"""
$(SIGNATURES)

Recommended FLL-assist bandwidth for `signal`: 5 Hz, the noise bandwidth of the
second-order FLL path of an FLL-assisted carrier filter (the `fll` half of its
`(pll, fll)` bandwidth pair; Kaplan & Hegarty). Capped at filter time at
`0.02 / T`: 5 Hz up to 4 ms, 2 Hz at 10 ms. Narrower pulls large handover errors
in too slowly; wider is the dominant noise source at 1 ms.

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

# `bandwidth` capped at `max_product / integration_time` for stability on long
# integrations. Only ever narrows, an explicit bandwidth as much as a default.
@inline _capped_bandwidth(bandwidth, integration_time, max_product) =
    min(bandwidth, uconvert(Hz, max_product / integration_time))

"""
$(SIGNATURES)

Code-loop bandwidth for a record of `integration_time`: `bandwidth` capped at
`MAX_CODE_LOOP_BANDWIDTH_TIME_PRODUCT / integration_time`. Integration time does
not otherwise narrow the DLL, as neither its dynamics nor its noise floor depend on
it. The default 1 Hz is capped only past 18 ms (0.9 Hz at 20 ms).
"""
@inline effective_code_loop_filter_bandwidth(bandwidth, integration_time) =
    _capped_bandwidth(bandwidth, integration_time, MAX_CODE_LOOP_BANDWIDTH_TIME_PRODUCT)

"""
    LoopBandwidths(; wide_carrier = 50Hz, narrow_carrier = 18Hz, fll_assist = 5Hz,
                     code = 1Hz)

A scalar estimator's per-satellite bandwidths (see [`CarrierLoopStage`](@ref)): the
PLL's until and after it narrows, the FLL path's and the DLL's, resolved by
[`init_estimator_state`](@ref) from the estimator's or the driver signal's defaults
and capped at filter time (see [Loop-filter bandwidths](@ref)).
"""
@kwdef struct LoopBandwidths
    wide_carrier::typeof(1.0Hz) = _DEFAULT_WIDE_CARRIER_LOOP_FILTER_BANDWIDTH
    narrow_carrier::typeof(1.0Hz) = _DEFAULT_NARROW_CARRIER_LOOP_FILTER_BANDWIDTH
    fll_assist::typeof(1.0Hz) = _DEFAULT_FLL_ASSIST_LOOP_FILTER_BANDWIDTH
    code::typeof(1.0Hz) = _DEFAULT_CODE_LOOP_FILTER_BANDWIDTH
end

# The estimator's `*_loop_filter_bandwidth` fields, each `nothing` replaced by
# `driver_signal`'s default (the wide one's by `wide_default`).
@inline _resolve_bandwidths(
    estimator,
    driver_signal::AbstractGNSSSignal;
    wide_default = default_wide_carrier_loop_filter_bandwidth(driver_signal),
) = LoopBandwidths(
    something(estimator.wide_carrier_loop_filter_bandwidth, wide_default),
    something(
        estimator.narrow_carrier_loop_filter_bandwidth,
        default_narrow_carrier_loop_filter_bandwidth(driver_signal),
    ),
    something(
        estimator.fll_assist_loop_filter_bandwidth,
        default_fll_assist_loop_filter_bandwidth(driver_signal),
    ),
    something(
        estimator.code_loop_filter_bandwidth,
        default_code_loop_filter_bandwidth(driver_signal),
    ),
)

# The stage's PLL bandwidth, capped against the time the record actually integrated,
# not the intended length: records around a mid-fold sync are short.
@inline _carrier_bandwidth(bandwidths::LoopBandwidths, stage, integration_time) =
    stage == NARROW_PLL ? _narrow_carrier_bandwidth(bandwidths, integration_time) :
    _capped_bandwidth(
        bandwidths.wide_carrier,
        integration_time,
        _MAX_WIDE_CARRIER_LOOP_BANDWIDTH_TIME_PRODUCT,
    )

@inline _narrow_carrier_bandwidth(bandwidths::LoopBandwidths, integration_time) =
    _capped_bandwidth(
        bandwidths.narrow_carrier,
        integration_time,
        _MAX_NARROW_CARRIER_LOOP_BANDWIDTH_TIME_PRODUCT,
    )

@inline _fll_assist_bandwidth(bandwidths::LoopBandwidths, integration_time) =
    _capped_bandwidth(
        bandwidths.fll_assist,
        integration_time,
        _MAX_FLL_ASSIST_LOOP_BANDWIDTH_TIME_PRODUCT,
    )

"""
$(SIGNATURES)

Add the loop-filter updates to the initial Dopplers, carrier-aiding the code
Doppler by the code-to-carrier frequency ratio.
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

"""
    calculate_carrier_frequency_update(signal, carrier_loop_filter, correlator,
                                       previous_prompt, integration_time, loop_bandwidth;
                                       fll_assist_loop_bandwidth)
        -> (carrier_freq_update, carrier_loop_filter)

One carrier-loop step: the PLL discriminator ([`pll_disc`](@ref), cycles) of
`correlator` filtered by `carrier_loop_filter` at `loop_bandwidth`; an FLL-assisted
filter (`ThirdOrderAssistedBilinearLF`) also gets the FLL discriminator
([`fll_disc`](@ref), Hz) between `previous_prompt` and this prompt, its FLL path at
`fll_assist_loop_bandwidth` (by default the signal's
[`default_fll_assist_loop_filter_bandwidth`](@ref), capped as [`step_loop`](@ref)
caps it). Returns the carrier-frequency correction in Hz and the advanced filter.

Unstaged: the FLL is never dropped and both discriminators are two-quadrant. The
estimators' [`step_loop`](@ref) stages the loop (see
[Carrier loop staging](@ref)) and uses four-quadrant discriminators where it can.
"""
function calculate_carrier_frequency_update(
    signal::AbstractGNSSSignal,
    carrier_loop_filter::ThirdOrderAssistedBilinearLF,
    correlator::AbstractCorrelator,
    previous_prompt::Complex,
    integration_time,
    loop_bandwidth;
    fll_assist_loop_bandwidth = _capped_bandwidth(
        default_fll_assist_loop_filter_bandwidth(signal),
        integration_time,
        _MAX_FLL_ASSIST_LOOP_BANDWIDTH_TIME_PRODUCT,
    ),
)
    pll_discriminator = pll_disc(signal, correlator)
    fll_discriminator = fll_disc(signal, correlator, previous_prompt, integration_time)
    filter_loop(
        carrier_loop_filter,
        (pll_discriminator, fll_discriminator),
        integration_time,
        (loop_bandwidth, fll_assist_loop_bandwidth),
    )
end

function calculate_carrier_frequency_update(
    signal::AbstractGNSSSignal,
    carrier_loop_filter::AbstractLoopFilter,
    correlator::AbstractCorrelator,
    previous_prompt::Complex,
    integration_time,
    loop_bandwidth;
    fll_assist_loop_bandwidth = nothing,
)
    pll_discriminator = pll_disc(signal, correlator)
    filter_loop(carrier_loop_filter, pll_discriminator, integration_time, loop_bandwidth)
end

"""
    calculate_code_frequency_update(signal, code_loop_filter, correlator, code_doppler,
                                    sampling_frequency, integration_time, loop_bandwidth)
        -> (code_freq_update, code_loop_filter)

One code-loop step: the DLL discriminator ([`dll_disc`](@ref)) of `correlator`
filtered by `code_loop_filter` at `loop_bandwidth`. `code_doppler`, the replica's,
converts the tap spacing from samples to chips. Returns the code-frequency
correction (before [`aid_dopplers`](@ref)) and the advanced filter.
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
