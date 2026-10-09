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

# The largest bandwidth–update interval products `BL · Δt` of the carrier loop,
# against which each record's integration time caps its bandwidths
# (`_capped_bandwidth`):
#
#   - wide, while it pulls in: 0.09. The default FLL-assisted third-order filter
#     diverges at `BL · Δt ≈ 0.4`, and at 0.09 its noise bandwidth runs 25 % wider
#     than configured;
#   - narrow, once it tracks: 0.04;
#   - the FLL path of an FLL-assisted filter: 0.02.
const _MAX_WIDE_CARRIER_LOOP_BANDWIDTH_TIME_PRODUCT = 0.09
const _MAX_NARROW_CARRIER_LOOP_BANDWIDTH_TIME_PRODUCT = 0.04
const _MAX_FLL_ASSIST_LOOP_BANDWIDTH_TIME_PRODUCT = 0.02

# The signal-independent default bandwidths the `default_*_loop_filter_bandwidth`
# methods return, and a per-satellite state built without a signal starts with.
const _DEFAULT_WIDE_CARRIER_LOOP_FILTER_BANDWIDTH = 50.0Hz
const _DEFAULT_NARROW_CARRIER_LOOP_FILTER_BANDWIDTH = 18.0Hz
const _DEFAULT_FLL_ASSIST_LOOP_FILTER_BANDWIDTH = 5.0Hz
const _DEFAULT_CODE_LOOP_FILTER_BANDWIDTH = 1.0Hz

"""
$(SIGNATURES)

Recommended wide carrier-loop-filter bandwidth for `signal`: 50 Hz, the
third-order PLL's noise bandwidth `BL` while the loop pulls the acquisition's
Doppler error in, FLL-assisted, and until phase lock (see
[`CarrierLoopStage`](@ref)). Seeded for every satellite whose estimator leaves
`wide_carrier_loop_filter_bandwidth` as `nothing`, and capped at filter time at
`0.09 / T` for a record's integration time `T`: 50 Hz for 1 ms records, 22.5 Hz
for 4 ms and 9 Hz for 10 ms. A wider pull-in loop locks faster but loses lock
near threshold: at 90 Hz, 1 ms records lose lock at 32.5 dB-Hz.

Override by defining a method for your signal type, or pass
`wide_carrier_loop_filter_bandwidth =` to the estimator. The
[`NCOReferencedPLLAndDLL`](@ref) does not read it: its wide default is
[`default_narrow_carrier_loop_filter_bandwidth`](@ref), for its tolerance of command
delay, so pass it `wide_carrier_loop_filter_bandwidth =` to widen it.
"""
function default_wide_carrier_loop_filter_bandwidth(signal::AbstractGNSSSignal)
    _DEFAULT_WIDE_CARRIER_LOOP_FILTER_BANDWIDTH
end

"""
$(SIGNATURES)

Recommended narrow carrier-loop-filter bandwidth for `signal`: 18 Hz, the
third-order PLL's noise bandwidth once the phase-lock indicator has confirmed
lock (see [`CarrierLoopStage`](@ref)). Capped at filter time at `0.04 / T`:
18 Hz for 1 ms records, 10 Hz for 4 ms and 4 Hz for 10 ms.

Override by defining a method for your signal type, or pass
`narrow_carrier_loop_filter_bandwidth =` to the estimator.
"""
function default_narrow_carrier_loop_filter_bandwidth(signal::AbstractGNSSSignal)
    _DEFAULT_NARROW_CARRIER_LOOP_FILTER_BANDWIDTH
end

"""
$(SIGNATURES)

Recommended FLL-assist bandwidth for `signal`: 5 Hz, the noise bandwidth of the
second-order FLL path of an FLL-assisted carrier filter, set independently of
the PLL's (Kaplan & Hegarty's FLL-assisted PLL, the filter's `(pll, fll)`
bandwidth pair). Capped at filter time at `0.02 / T`: 5 Hz up to 4 ms records,
2 Hz for 10 ms. A narrower FLL path pulls large handover errors in too slowly;
a wider one is the noise source at 1 ms.

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

# A bandwidth for a record that integrated for `integration_time`, capped at
# `max_product / integration_time` to keep the loop stable on long integrations.
# The cap only ever narrows a loop, an explicit bandwidth as much as the default.
@inline _capped_bandwidth(bandwidth, integration_time, max_product) =
    min(bandwidth, uconvert(Hz, max_product / integration_time))

"""
$(SIGNATURES)

Effective code-loop bandwidth for a record that integrated for
`integration_time`: the configured bandwidth, capped at
`MAX_CODE_LOOP_BANDWIDTH_TIME_PRODUCT / integration_time` for stability.
Longer integration must not otherwise narrow the DLL: neither its dynamics nor
its noise floor depend on it. At the default 1 Hz the cap binds only past
18 ms (0.9 Hz for a 20 ms integration, 0.012 Hz for a 1.5 s one).
"""
@inline effective_code_loop_filter_bandwidth(bandwidth, integration_time) =
    _capped_bandwidth(bandwidth, integration_time, MAX_CODE_LOOP_BANDWIDTH_TIME_PRODUCT)

"""
    CarrierLoopStage

The stage of a satellite's carrier loop, held in the scalar estimators'
per-satellite state (see [`carrier_loop_stage`](@ref)):

  - `FLL_ASSISTED_PLL`: the FLL-assisted PLL at the wide bandwidth
    ([`default_wide_carrier_loop_filter_bandwidth`](@ref)) with its own FLL
    bandwidth ([`default_fll_assist_loop_filter_bandwidth`](@ref)), until the
    phase-lock indicator has read lock for one time constant of its average:
    the carrier is in phase lock, so its Doppler has converged;
  - `WIDE_PLL`: the pure PLL at the wide bandwidth, until the phase-lock
    indicator has read lock for another four time constants;
  - `NARROW_PLL`: the pure PLL at the narrow bandwidth
    ([`default_narrow_carrier_loop_filter_bandwidth`](@ref)).

A carrier filter without an FLL path starts at `WIDE_PLL`. The stages only
advance; [`reset_estimator_state`](@ref) restarts them. Dropping the FLL and
narrowing are separate steps: narrowing while the loop still settles from the
FLL's last correction loses lock.
"""
@enum CarrierLoopStage FLL_ASSISTED_PLL WIDE_PLL NARROW_PLL

# How long, in time constants of the phase-lock indicator's average, it must read
# lock without a break to end the FLL-assisted stage (1; with the average's own
# memory that is about two time constants of prompts), and then, counted afresh,
# to narrow the loop (4; shorter narrowed too early on 4 and 10 ms records).
const _FLL_DROP_TIME_CONSTANTS = 1
const _NARROWING_TIME_CONSTANTS = 4

# The time constant of the phase-lock indicator's averages at the record length
# `integration_time`: 0.1 s, and at least 25 records, so that the signal power the
# indicator is normalised with is averaged over enough records (0.1 s for 1 and 4 ms
# records, 0.25 s for 10 ms, 0.5 s for 20 ms).
@inline _phase_lock_time_constant(integration_time) = max(
    _PHASE_LOCK_TIME_CONSTANT,
    uconvert(s, _MIN_PHASE_LOCK_RECORDS * float(integration_time)),
)

const _PHASE_LOCK_TIME_CONSTANT = 0.1s
const _MIN_PHASE_LOCK_RECORDS = 25

"""
$(SIGNATURES)

Threshold of the phase-lock indicator ([`phase_lock_indicator`](@ref)) the
carrier-loop staging reads: `0.5`, i.e. an RMS phase error of about 30°.
Override by defining a method for your signal type.
"""
phase_lock_indicator_threshold(signal::AbstractGNSSSignal, integration_time) = 0.5

# The phase-lock indicator of a satellite's carrier loop, held in the scalar
# estimators' per-satellite state and advanced by every record of the
# estimator-driver signal: `⟨I² − Q²⟩ / A²`, an estimate of `cos 2φ` — 1 in phase
# lock, 0 for a uniformly spinning phase. The averages are exponential, updated
# every record, over the time constant `_phase_lock_time_constant`, and the signal
# power `A²` is estimated from the prompt's moments averaged alike as `√(2 M₂² − M₄)`,
# `Mₖ` the average of `|P|ᵏ`, so it reads the same at any C/N₀: unlike
# `⟨I² − Q²⟩ / ⟨I² + Q²⟩` it does not read low at low C/N₀ when the loop is
# locked. `phase_lock` is the latest reading, `NaN` until the averages span one
# time constant, and `hold` how long it has read lock (at or above
# `phase_lock_indicator_threshold`) without a break, which the carrier-loop staging
# moves on by. A record of another length restarts the averages, whose prompts
# would otherwise mix two power scales, and so the hold.
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

# The relative change of the record length that restarts the phase-lock
# indicator's averages, and the vector engine's C/N₀ estimates: well beyond the
# sample-rounding jitter of one record length, well within the factor between two
# (1 → 20 ms at a bit sync, say).
const _RECORD_LENGTH_CHANGE = 0.25

@inline _record_length_changed(previous_integration_time, integration_time) =
    abs(integration_time - previous_integration_time) >
    _RECORD_LENGTH_CHANGE * integration_time

# One step of an exponential average with the weight `max(1/n, α)`: the plain
# mean of the first `1/α` samples, so the average starts unbiased, then
# exponential with the time constant `T / α`.
@inline _average(mean, x, n, α) = mean + max(1 / n, α) * (x - mean)

# Advance the phase-lock indicator of a loop on `signal` by one record's prompt.
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
    # A reading once the averages span a time constant.
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

# The indicator with its hold counted afresh from now.
@inline _restart_hold(indicator::PhaseLockIndicator) = PhaseLockIndicator(
    indicator.coherent_power,
    indicator.power,
    indicator.squared_power,
    indicator.num_records,
    indicator.integration_time,
    indicator.phase_lock,
    0.0s,
)

# Whether `hold` spans `time_constants` of the phase-lock indicator's averages at
# the record length `integration_time`.
@inline _held(hold, time_constants, integration_time) =
    hold >= time_constants * _phase_lock_time_constant(integration_time) - 1e-9s

"""
    LoopBandwidths(; wide_carrier = 50Hz, narrow_carrier = 18Hz, fll_assist = 5Hz,
                     code = 1Hz)

The loop-filter bandwidths of a scalar estimator's per-satellite state:
`wide_carrier` for the PLL until it narrows (see [`CarrierLoopStage`](@ref)),
`narrow_carrier` once phase lock has held, `fll_assist` for the FLL path while
FLL-assisted, and `code` for the DLL. [`init_estimator_state`](@ref) resolves them
from the estimator's, each `nothing` there from the driver signal's default. Each is
capped against the record's integration time at filter time (see
[Loop-filter bandwidths](@ref)).
"""
@kwdef struct LoopBandwidths
    wide_carrier::typeof(1.0Hz) = _DEFAULT_WIDE_CARRIER_LOOP_FILTER_BANDWIDTH
    narrow_carrier::typeof(1.0Hz) = _DEFAULT_NARROW_CARRIER_LOOP_FILTER_BANDWIDTH
    fll_assist::typeof(1.0Hz) = _DEFAULT_FLL_ASSIST_LOOP_FILTER_BANDWIDTH
    code::typeof(1.0Hz) = _DEFAULT_CODE_LOOP_FILTER_BANDWIDTH
end

# The bandwidths of a scalar estimator (its `*_loop_filter_bandwidth` fields) for a
# satellite driven by `driver_signal`: each given one as it is, each `nothing` the
# signal's default, the wide one's `wide_default`.
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

# The PLL bandwidth of `stage` for a record that integrated for `integration_time`:
# the wide one until the loop narrows, the narrow one after. Capped against the time
# the record actually integrated, not the intended length: records folded after a
# mid-fold sync, or the truncated first post-sync integration, are short.
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

A scalar estimator's carrier-loop staging, in its per-satellite state: the
[`CarrierLoopStage`](@ref) and the phase-lock indicator that moves it on, with how
long that has read lock. Read them with [`carrier_loop_stage`](@ref) and
[`phase_lock_indicator`](@ref).
"""
struct CarrierLoopStaging
    stage::CarrierLoopStage
    phase_lock::PhaseLockIndicator
end

carrier_loop_stage(staging::CarrierLoopStaging) = staging.stage
phase_lock_indicator(staging::CarrierLoopStaging) = phase_lock_indicator(staging.phase_lock)

# The staging a carrier loop starts with: FLL-assisted where the filter has an FLL
# path, the wide pure PLL where it has none, and a fresh indicator.
CarrierLoopStaging(carrier_loop_filter::AbstractLoopFilter) = CarrierLoopStaging(
    _uses_fll(carrier_loop_filter) ? FLL_ASSISTED_PLL : WIDE_PLL,
    PhaseLockIndicator(),
)

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
    calculate_carrier_frequency_update(signal, carrier_loop_filter, correlator,
                                       previous_prompt, integration_time, loop_bandwidth;
                                       fll_assist_loop_bandwidth)
        -> (carrier_freq_update, carrier_loop_filter)

One carrier-loop step: the PLL discriminator ([`pll_disc`](@ref)) of
`correlator`, in cycles, filtered by `carrier_loop_filter` at `loop_bandwidth`.
An FLL-assisted filter (`ThirdOrderAssistedBilinearLF`) is additionally fed the
FLL discriminator ([`fll_disc`](@ref)) between `previous_prompt` and this
record's prompt, its FLL path at `fll_assist_loop_bandwidth`: by default the
signal's [`default_fll_assist_loop_filter_bandwidth`](@ref) capped against
`integration_time`, as [`step_loop`](@ref) runs it. Other filters ignore it.
Returns the carrier-frequency correction in Hz and the advanced filter. The
filter's coefficients assume consistent units: fed the phase error in cycles and
the FLL error in Hz (cycles per second), it outputs a Doppler in Hz.

A standalone step: it does not stage the loop. An FLL-assisted filter is fed
the two-quadrant FLL discriminator on every call, never dropped at phase
lock, and the PLL is the two-quadrant Costas one; the estimators'
[`step_loop`](@ref) stages the carrier loop (see [Carrier loop staging](@ref))
and picks four-quadrant discriminators where the record allows.
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
# is `cis(driver_carrier_phase_offset − get_carrier_phase_offset(signal))`. For an
# in-phase component the difference is 0 and `cis(0) === 1 + 0im`, a
# bit-identical no-op; a quadrature component (GPS L5 / Galileo E5a I-vs-Q)
# rotates by `±90°` onto the real axis.
@inline _carrier_phase_derotation(driver_carrier_phase_offset::Real, signal) =
    cis(driver_carrier_phase_offset - get_carrier_phase_offset(signal))
