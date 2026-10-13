# The scalar loops' carrier-loop staging; see `CarrierLoopStage`.

"""
    CarrierLoopStage

The stage of a satellite's carrier loop in the scalar estimators' per-satellite
state (see [`carrier_loop_stage`](@ref)):

  - `FLL_ASSISTED_PLL`: FLL-assisted PLL at the wide bandwidth
    ([`default_wide_carrier_loop_filter_bandwidth`](@ref)), its FLL path at
    [`default_fll_assist_loop_filter_bandwidth`](@ref), until the phase-lock
    indicator ([`phase_lock_indicator`](@ref)) has read lock for one time constant
    of its averages: phase lock implies a converged Doppler;
  - `WIDE_PLL`: PLL alone at the wide bandwidth, until lock has held for another
    four time constants;
  - `NARROW_PLL`: PLL alone at the narrow bandwidth
    ([`default_narrow_carrier_loop_filter_bandwidth`](@ref)).

A filter without an FLL path starts at `WIDE_PLL`. Dropping the FLL feeds the
assisted filter a zero FLL input, which is exactly the third-order PLL (same state,
same coefficients), so the switch is free (Kaplan & Hegarty §5.5); the FLL
discriminator is then no longer formed. Dropping the FLL and narrowing are separate
steps: narrowing while the loop still settles from the FLL's last correction risks
the lock.

Stages only advance; [`reset_estimator_state`](@ref) restarts them, as does a
satellite's release from the vector loop. In the vector loop the carrier loop runs
at the narrow bandwidth and [`carrier_loop_stage`](@ref) reports `NARROW_PLL`.
"""
@enum CarrierLoopStage FLL_ASSISTED_PLL WIDE_PLL NARROW_PLL

# Unbroken lock, in time constants of the indicator's averages, that drops the FLL
# and then, counted afresh, narrows the loop.
const _FLL_DROP_TIME_CONSTANTS = 1
const _NARROWING_TIME_CONSTANTS = 4

# Time constant of the phase-lock indicator's averages, in records of
# `num_code_blocks` primary code blocks: 0.1 s but at least 25 records, so its
# signal-power normaliser is averaged over enough records (100 of 1 ms, 25 of 4 ms
# or longer: 0.25 s at 10 ms, 0.5 s at 20 ms). Counted in chips from the nominal
# record length, so it does not move with a record's sample jitter.
const _PHASE_LOCK_TIME_CONSTANT = 0.1s
const _MIN_PHASE_LOCK_RECORDS = 25

@inline function _phase_lock_time_constant(signal::AbstractGNSSSignal, num_code_blocks)
    chips = round(Int, get_code_frequency(signal) * _PHASE_LOCK_TIME_CONSTANT)
    max(cld(chips, num_code_blocks * get_code_length(signal)), _MIN_PHASE_LOCK_RECORDS)
end

# The phase-lock indicator's lock threshold: an RMS phase error of about 30°.
const _PHASE_LOCK_THRESHOLD = 0.5

"""
    CarrierLoopStaging(carrier_loop_filter)

A scalar estimator's [`CarrierLoopStage`](@ref) and the phase-lock indicator that
moves it on, read with [`carrier_loop_stage`](@ref) and
[`phase_lock_indicator`](@ref). `CarrierLoopStaging(carrier_loop_filter)` starts it
(FLL-assisted if the filter has an FLL path, else at the wide PLL); it then advances
with every driver record.
"""
struct CarrierLoopStaging
    stage::CarrierLoopStage
    # Exponential averages over `_phase_lock_time_constant` driver records of the
    # indicator `⟨I² − Q²⟩ / A²` (see `phase_lock_indicator`): `power_difference`
    # of `I² − Q²` (Van Dierendonck's narrowband difference, NBD), `power` and
    # `squared_power` of `|P|²` and `|P|⁴` (M₂, M₄). `A² = √(2 M₂² − M₄)`, which
    # the noise cancels from, so unlike `⟨I² − Q²⟩ / ⟨I² + Q²⟩` the indicator does
    # not read low at low C/N₀ while locked.
    power_difference::Float64
    power::Float64
    squared_power::Float64
    num_records::Int
    # The latest reading (`NaN` until the averages span one time constant) and the
    # consecutive records it has read lock, restarted at each stage and no longer
    # counted once narrowed.
    phase_lock_indicator::Float64
    num_locked_records::Int
end

CarrierLoopStaging(carrier_loop_filter::AbstractLoopFilter) = CarrierLoopStaging(
    _has_fll_path(typeof(carrier_loop_filter)) ? FLL_ASSISTED_PLL : WIDE_PLL,
    0.0,
    0.0,
    0.0,
    0,
    NaN,
    0,
)

# Whether a carrier loop filter type has an FLL path (more methods in
# `vector/estimator.jl`).
@inline _has_fll_path(::Type{<:ThirdOrderAssistedBilinearLF}) = true
@inline _has_fll_path(::Type{<:AbstractLoopFilter}) = false

# Exponential mean with weight `max(1/n, 1/time_constant)`: the plain mean of the
# first `time_constant` samples, so it starts unbiased.
@inline _exponential_mean(mean, x, n, time_constant) =
    mean + (x - mean) / min(n, time_constant)

# The staging after one driver `LoopRecord`: its prompt advances the indicator,
# whose lock count moves the stage on for the next record.
@inline function _advance_staging(staging::CarrierLoopStaging, record)
    time_constant = _phase_lock_time_constant(record.signal, record.integrated_code_blocks)
    prompt = get_prompt(record.filtered_correlator)
    num_records = staging.num_records + 1
    prompt_power = Float64(abs2(prompt))
    power_difference = _exponential_mean(
        staging.power_difference,
        Float64(real(prompt)^2 - imag(prompt)^2),
        num_records,
        time_constant,
    )
    power = _exponential_mean(staging.power, prompt_power, num_records, time_constant)
    squared_power =
        _exponential_mean(staging.squared_power, prompt_power^2, num_records, time_constant)
    signal_power = sqrt(max(2 * power^2 - squared_power, 0.0))
    phase_lock =
        num_records < time_constant ? NaN :
        signal_power > 0 ? power_difference / signal_power : 0.0
    stage = staging.stage
    num_locked_records =
        stage != NARROW_PLL && phase_lock >= _PHASE_LOCK_THRESHOLD ?
        staging.num_locked_records + 1 : 0
    if stage == FLL_ASSISTED_PLL &&
       num_locked_records >= _FLL_DROP_TIME_CONSTANTS * time_constant
        stage = WIDE_PLL
        num_locked_records = 0
    elseif stage == WIDE_PLL &&
           num_locked_records >= _NARROWING_TIME_CONSTANTS * time_constant
        stage = NARROW_PLL
        num_locked_records = 0
    end
    CarrierLoopStaging(
        stage,
        power_difference,
        power,
        squared_power,
        num_records,
        phase_lock,
        num_locked_records,
    )
end
