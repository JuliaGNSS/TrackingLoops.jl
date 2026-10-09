# ─────────────────────────────────────────────────────────────────────────────
# Carrier-loop staging of the scalar estimators: the stage, the phase-lock indicator
# that moves it on, and the per-record staging step (see `CarrierLoopStage`).
# ─────────────────────────────────────────────────────────────────────────────

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

@inline _uses_fll(::ThirdOrderAssistedBilinearLF) = true
@inline _uses_fll(::AbstractLoopFilter) = false

@inline _fll_in_use(carrier_loop_filter, staging::CarrierLoopStaging) =
    _uses_fll(carrier_loop_filter) && staging.stage == FLL_ASSISTED_PLL
@inline _fll_in_use(state) = _fll_in_use(state.carrier_loop_filter, state.staging)

# Steps the staging (see `CarrierLoopStage`): advances the phase-lock indicator
# and moves the stage on for the next record. Returns the FLL input to feed (zero
# once the FLL is dropped) and the stepped staging.
@inline function _staged_carrier_loop(
    staging::CarrierLoopStaging,
    carrier_loop_filter,
    record::LoopRecord,
    discriminators,
)
    integration_time = discriminators.integration_time
    phase_lock = _update_phase_lock(
        staging.phase_lock,
        get_prompt(record.filtered_correlator),
        integration_time,
        record.signal,
    )
    stage = staging.stage
    frequency_error =
        _fll_in_use(carrier_loop_filter, staging) ? discriminators.frequency_error :
        zero(discriminators.frequency_error)
    if stage == FLL_ASSISTED_PLL && (
        !_uses_fll(carrier_loop_filter) ||
        _held(phase_lock.hold, _FLL_DROP_TIME_CONSTANTS, integration_time)
    )
        stage = WIDE_PLL
        phase_lock = _restart_hold(phase_lock)
    elseif stage == WIDE_PLL &&
           _held(phase_lock.hold, _NARROWING_TIME_CONSTANTS, integration_time)
        stage = NARROW_PLL
    end
    frequency_error, CarrierLoopStaging(stage, phase_lock)
end
