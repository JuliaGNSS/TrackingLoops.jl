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
    MAX_FLL_ASSIST_LOOP_BANDWIDTH_TIME_PRODUCT

The largest bandwidth–update interval product `BL · Δt` of the FLL path of an
FLL-assisted carrier filter: `0.02`.
[`effective_fll_assist_loop_filter_bandwidth`](@ref) caps the FLL-assist
bandwidth at `0.02 / Δt`: 5 Hz up to 4 ms records, 2 Hz at 10 ms.
"""
const MAX_FLL_ASSIST_LOOP_BANDWIDTH_TIME_PRODUCT = 0.02

# The signal-independent default loop-filter bandwidths: what the generic
# `default_*_loop_filter_bandwidth` methods return, and what a `LoopBandwidths`
# built without a signal holds.
const _DEFAULT_WIDE_CARRIER_LOOP_FILTER_BANDWIDTH = 50.0Hz
const _DEFAULT_NARROW_CARRIER_LOOP_FILTER_BANDWIDTH = 18.0Hz
const _DEFAULT_FLL_ASSIST_LOOP_FILTER_BANDWIDTH = 5.0Hz
const _DEFAULT_CODE_LOOP_FILTER_BANDWIDTH = 1.0Hz

"""
$(SIGNATURES)

Recommended wide carrier-loop-filter bandwidth for `signal`: 50 Hz, the
third-order PLL's noise bandwidth `BL` while the loop pulls in, FLL-assisted,
and until phase lock has held (see [`CarrierLoopStage`](@ref)). Seeded for every
satellite whose estimator leaves `wide_carrier_loop_filter_bandwidth` as
`nothing`, and capped at filter time by
[`effective_carrier_loop_filter_bandwidth`](@ref): 50 Hz for 1 ms records,
22.5 Hz for 4 ms and 9 Hz for 10 ms. The [`NCOReferencedPLLAndDLL`](@ref) does
not read it: its wide default is the narrow one,
[`default_narrow_carrier_loop_filter_bandwidth`](@ref), for its tolerance of
command delay.

Override by defining a method for your signal type, or pass
`wide_carrier_loop_filter_bandwidth =` to the estimator.
"""
function default_wide_carrier_loop_filter_bandwidth(signal::AbstractGNSSSignal)
    _DEFAULT_WIDE_CARRIER_LOOP_FILTER_BANDWIDTH
end

"""
$(SIGNATURES)

Recommended narrow carrier-loop-filter bandwidth for `signal`: 18 Hz, the
third-order PLL bandwidth of the literature (Kaplan & Hegarty, Pany), which the
loop narrows to once phase lock has held (see [`CarrierLoopStage`](@ref)) and
the vector loop runs at. Capped at filter time by
[`effective_carrier_loop_filter_bandwidth`](@ref): 18 Hz up to 5 ms records,
9 Hz for 10 ms and 4.5 Hz for 20 ms.

Override by defining a method for your signal type, or pass
`narrow_carrier_loop_filter_bandwidth =` to the estimator.
"""
function default_narrow_carrier_loop_filter_bandwidth(signal::AbstractGNSSSignal)
    _DEFAULT_NARROW_CARRIER_LOOP_FILTER_BANDWIDTH
end

"""
$(SIGNATURES)

Recommended FLL-assist bandwidth for `signal`: 5 Hz, the noise bandwidth of the
second-order FLL path of an FLL-assisted carrier filter while it is
FLL-assisted, set independently of the PLL's (the filter's `(pll, fll)`
bandwidth pair). Capped at filter time by
[`effective_fll_assist_loop_filter_bandwidth`](@ref): 5 Hz up to 4 ms records,
2 Hz for 10 ms.

Override by defining a method for your signal type, or pass
`fll_assist_loop_filter_bandwidth =` to the estimator.
"""
function default_fll_assist_loop_filter_bandwidth(signal::AbstractGNSSSignal)
    _DEFAULT_FLL_ASSIST_LOOP_FILTER_BANDWIDTH
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
    _DEFAULT_CODE_LOOP_FILTER_BANDWIDTH
end

"""
    LoopBandwidths(; wide_carrier = 50Hz, narrow_carrier = 18Hz, fll_assist = 5Hz,
                     code = 1Hz)
    LoopBandwidths(signal; wide_carrier, narrow_carrier, fll_assist, code)

The loop-filter bandwidths of a scalar estimator's per-satellite state:
`wide_carrier` for the PLL until it narrows and `narrow_carrier` after (see
[`CarrierLoopStage`](@ref)), `fll_assist` for the FLL path while the loop is
FLL-assisted, and `code` for the DLL. Given a `signal`, each bandwidth not given
defaults to the signal's ([`default_wide_carrier_loop_filter_bandwidth`](@ref),
[`default_narrow_carrier_loop_filter_bandwidth`](@ref),
[`default_fll_assist_loop_filter_bandwidth`](@ref),
[`default_code_loop_filter_bandwidth`](@ref)), so methods of those for a signal
type apply; without one, to the generic defaults. [`init_estimator_state`](@ref)
resolves them from the estimator's, each `nothing` there from the driver
signal's default. Each is capped against the record's integration time at
filter time (see [`effective_carrier_loop_filter_bandwidth`](@ref),
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

Effective PLL bandwidth for a record that integrated for `integration_time`,
wide or narrow alike: the configured bandwidth, capped at
`MAX_CARRIER_LOOP_BANDWIDTH_TIME_PRODUCT / integration_time` to keep the loop
stable on long integrations. The default narrow 18 Hz runs unchanged up to
5 ms of integration, at 9 Hz for 10 ms and 4.5 Hz for 20 ms. The cap only ever
narrows a loop, an explicit bandwidth as much as the default.
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

Effective FLL-assist bandwidth for a record that integrated for
`integration_time`: the configured bandwidth, capped at
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

"""
    calculate_carrier_frequency_update(signal, carrier_loop_filter, correlator, previous_prompt, integration_time, loop_bandwidth)
        -> (carrier_freq_update, carrier_loop_filter)

One carrier-loop step: the PLL discriminator ([`pll_disc`](@ref)) of
`correlator`, in cycles, filtered by `carrier_loop_filter` at
`loop_bandwidth`. An FLL-assisted filter (`ThirdOrderAssistedBilinearLF`) is
additionally fed the FLL discriminator ([`fll_disc`](@ref)) between
`previous_prompt` and this record's prompt, its FLL path at the bandwidth tied
to `loop_bandwidth`. Returns the carrier-frequency correction in Hz and the
advanced filter. It does not stage the carrier loop: the estimators'
[`step_loop`](@ref) does (see [`CarrierLoopStage`](@ref)).
"""
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
