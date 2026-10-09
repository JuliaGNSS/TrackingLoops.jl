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
    CarrierLoopStage

The stage of a satellite's carrier loop in the scalar estimators' per-satellite
state (see [`carrier_loop_stage`](@ref)):

  - `FLL_ASSISTED_PLL`: FLL-assisted PLL at the wide bandwidth
    ([`default_wide_carrier_loop_filter_bandwidth`](@ref)) and the FLL
    bandwidth ([`default_fll_assist_loop_filter_bandwidth`](@ref)), until the
    phase-lock indicator ([`phase_lock_indicator`](@ref)) has read lock for one
    time constant of its average: phase lock implies a converged Doppler;
  - `WIDE_PLL`: pure PLL at the wide bandwidth, until lock has held for another
    four time constants;
  - `NARROW_PLL`: pure PLL at the narrow bandwidth
    ([`default_narrow_carrier_loop_filter_bandwidth`](@ref)).

A filter without an FLL path starts at `WIDE_PLL`. Stages only advance;
[`reset_estimator_state`](@ref) restarts them. Dropping the FLL and narrowing are
separate steps because narrowing while the loop still settles from the FLL's last
correction loses lock. Dropping the FLL means feeding the assisted filter a zero
frequency error, which is exactly the third-order PLL (same state, same
coefficients), so the switch is free (Kaplan & Hegarty §5.5; Ward, ION GPS 1998).
"""
@enum CarrierLoopStage FLL_ASSISTED_PLL WIDE_PLL NARROW_PLL

# Unbroken lock, in indicator time constants, that ends the FLL-assisted stage (1;
# about two of prompts with the average's own memory) and then, counted afresh,
# narrows the loop (4; shorter narrowed too early on 4 and 10 ms records).
const _FLL_DROP_TIME_CONSTANTS = 1
const _NARROWING_TIME_CONSTANTS = 4

# Time constant of the phase-lock indicator's averages: 0.1 s but at least 25
# records, so its signal-power normaliser is averaged over enough records (0.25 s
# at 10 ms, 0.5 s at 20 ms).
@inline _phase_lock_time_constant(integration_time) = max(
    _PHASE_LOCK_TIME_CONSTANT,
    uconvert(s, _MIN_PHASE_LOCK_RECORDS * float(integration_time)),
)

const _PHASE_LOCK_TIME_CONSTANT = 0.1s
const _MIN_PHASE_LOCK_RECORDS = 25

"""
$(SIGNATURES)

Phase-lock indicator ([`phase_lock_indicator`](@ref)) threshold for the
carrier-loop staging: `0.5`, an RMS phase error of about 30°.
Override by defining a method for your signal type.
"""
phase_lock_indicator_threshold(signal::AbstractGNSSSignal, integration_time) = 0.5

# Phase-lock indicator `⟨I² − Q²⟩ / A²` ≈ `cos 2φ` of the driver's prompts (1 in
# lock, 0 for a spinning phase), from exponential averages over
# `_phase_lock_time_constant`. `A² = √(2 M₂² − M₄)`, `Mₖ` the average of `|P|ᵏ`, so
# unlike `⟨I² − Q²⟩ / ⟨I² + Q²⟩` it does not read low at low C/N₀. `phase_lock` is
# `NaN` until the averages span one time constant, and `hold` is how long it has read
# lock (at or above `phase_lock_indicator_threshold`) without a break. A record of
# another length restarts the averages, which would otherwise mix two power scales,
# and the hold.
struct PhaseLockIndicator
    coherent_power::Float64
    power::Float64
    squared_power::Float64
    num_records::Int
    integration_time::typeof(1.0s)
    phase_lock::Float64
    hold::typeof(1.0s)
end

PhaseLockIndicator() = PhaseLockIndicator(0.0, 0.0, 0.0, 0, 0.0s, NaN, 0.0s)

phase_lock_indicator(indicator::PhaseLockIndicator) = indicator.phase_lock

# Relative record-length change that restarts the indicator and the vector engine's
# C/N₀ estimates: well above sample rounding jitter, well below the factor between two
# lengths (1 → 20 ms at bit sync).
const _RECORD_LENGTH_CHANGE = 0.25

@inline _record_length_changed(previous_integration_time, integration_time) =
    abs(integration_time - previous_integration_time) >
    _RECORD_LENGTH_CHANGE * integration_time

# Exponential average with weight `max(1/n, α)`: the plain mean of the first `1/α`
# samples, so it starts unbiased.
@inline _average(mean, x, n, α) = mean + max(1 / n, α) * (x - mean)

@inline function _update_phase_lock(
    indicator::PhaseLockIndicator,
    prompt::Complex,
    integration_time,
    signal::AbstractGNSSSignal,
)
    prompt = ComplexF64(prompt)
    integration_time = uconvert(s, float(integration_time))
    restart = _record_length_changed(indicator.integration_time, integration_time)
    coherent_power, power, squared_power, num_records =
        restart ? (0.0, 0.0, 0.0, 0) :
        (
            indicator.coherent_power,
            indicator.power,
            indicator.squared_power,
            indicator.num_records,
        )
    α = Float64(integration_time / _phase_lock_time_constant(integration_time))
    num_records += 1
    prompt_power = abs2(prompt)
    coherent_power =
        _average(coherent_power, real(prompt)^2 - imag(prompt)^2, num_records, α)
    power = _average(power, prompt_power, num_records, α)
    squared_power = _average(squared_power, prompt_power^2, num_records, α)
    signal_power = sqrt(max(2 * power^2 - squared_power, 0.0))
    phase_lock =
        num_records < ceil(Int, 1 / α - 1e-9) ? NaN :
        signal_power > 0 ? coherent_power / signal_power : 0.0
    hold =
        phase_lock >= phase_lock_indicator_threshold(signal, integration_time) ?
        indicator.hold + integration_time : 0.0s
    PhaseLockIndicator(
        coherent_power,
        power,
        squared_power,
        num_records,
        integration_time,
        phase_lock,
        hold,
    )
end

# The indicator with its hold restarted.
@inline _restart_hold(indicator::PhaseLockIndicator) = PhaseLockIndicator(
    indicator.coherent_power,
    indicator.power,
    indicator.squared_power,
    indicator.num_records,
    indicator.integration_time,
    indicator.phase_lock,
    0.0s,
)

@inline _held(hold, time_constants, integration_time) =
    hold >= time_constants * _phase_lock_time_constant(integration_time) - 1e-9s

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
    CarrierLoopStaging(stage, phase_lock)

A scalar estimator's [`CarrierLoopStage`](@ref) and the phase-lock indicator that
moves it on; read with [`carrier_loop_stage`](@ref) and
[`phase_lock_indicator`](@ref).
"""
struct CarrierLoopStaging
    stage::CarrierLoopStage
    phase_lock::PhaseLockIndicator
end

carrier_loop_stage(staging::CarrierLoopStaging) = staging.stage
phase_lock_indicator(staging::CarrierLoopStaging) = phase_lock_indicator(staging.phase_lock)

# The initial staging: FLL-assisted if the filter has an FLL path, else the wide PLL.
CarrierLoopStaging(carrier_loop_filter::AbstractLoopFilter) = CarrierLoopStaging(
    _uses_fll(carrier_loop_filter) ? FLL_ASSISTED_PLL : WIDE_PLL,
    PhaseLockIndicator(),
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

# Rotates a component's bit-buffer prompt onto the real axis, given the loops lock
# the driver there: a bit-identical no-op (`cis(0)`) for an in-phase component,
# `±90°` for a quadrature one (GPS L5 / Galileo E5a I vs Q).
@inline _carrier_phase_derotation(driver_carrier_phase_offset::Real, signal) =
    cis(driver_carrier_phase_offset - get_carrier_phase_offset(signal))
