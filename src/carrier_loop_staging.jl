# The scalar loops' carrier-loop staging; see `CarrierLoopStage`.

"""
    CarrierLoopStage

The stage of a satellite's carrier loop in the scalar estimators' per-satellite
state (see [`carrier_loop_stage`](@ref)):

  - `FLL_ASSISTED_PLL`: FLL-assisted PLL at the wide bandwidth
    ([`default_wide_carrier_loop_filter_bandwidth`](@ref)), its FLL path at
    [`default_fll_assist_loop_filter_bandwidth`](@ref), until the phase-lock
    indicator ([`phase_lock_indicator`](@ref)) has read lock
    ([`phase_locked`](@ref)) without a break for one time constant of its averages:
    phase lock implies a converged Doppler;
  - `WIDE_PLL`: PLL alone at the wide bandwidth, the FLL discriminator no longer
    formed, until lock has held unbroken for another four time constants;
  - `NARROW_PLL`: PLL alone at the narrow bandwidth
    ([`default_narrow_carrier_loop_filter_bandwidth`](@ref));
  - `VECTOR_LOOP`: no loop filter on the carrier NCO, which the navigation filter
    sets while the satellite is in the vector loop; its residual PLL follows the
    carrier (see [`VectorPLLAndDLL`](@ref)'s `residual_pll_loop_filter_bandwidth`).
    The scalar staging never reaches it; only a satellite in the vector loop reports it.

A filter without an FLL path starts at `WIDE_PLL`. Dropping the FLL feeds the
assisted filter a zero FLL input, which is exactly the third-order PLL (same state,
same coefficients), so the switch is free (Kaplan & Hegarty §5.5). Dropping the FLL
and narrowing are separate steps: narrowing while the loop still settles from the
FLL's last correction risks the lock.

Each driver record's prompt advances the indicator, and the stage it reaches applies
from the next record. Stages only advance: a loop that loses lock keeps the narrow
PLL. [`init_estimator_state`](@ref) starts them over, as does a satellite's release
from the vector loop; [`reset_estimator_state`](@ref) keeps them (a handoff to a
new integration length, say, keeps the lock). In the vector loop the staging stands
still.
"""
@enum CarrierLoopStage FLL_ASSISTED_PLL WIDE_PLL NARROW_PLL VECTOR_LOOP

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
    chips = round(Int, get_chip_rate(signal) * _PHASE_LOCK_TIME_CONSTANT)
    max(cld(chips, num_code_blocks * get_code_length(signal)), _MIN_PHASE_LOCK_RECORDS)
end

# The phase-lock indicator's lock threshold: an RMS phase error of about 30°.
const _PHASE_LOCK_THRESHOLD = 0.5

# The phase-lock indicator (see `phase_lock_indicator`) and its exponential averages
# over `_phase_lock_time_constant` records: `power_difference` of `I² − Q²` (Van
# Dierendonck's narrowband difference, NBD), `power` and `squared_power` of `|P|²` and
# `|P|⁴` (M₂, M₄). `indicator` is the latest reading, `NaN` until the averages span one
# time constant. The scalar staging keeps one on the driver's prompts, the vector
# loop's residual PLL one on the prompts it rotated.
struct PhaseLockDetector
    power_difference::Float64
    power::Float64
    squared_power::Float64
    num_records::Int
    indicator::Float64
end

PhaseLockDetector() = PhaseLockDetector(0.0, 0.0, 0.0, 0, NaN)

# `A²` of the detector's averages.
_signal_power(detector::PhaseLockDetector) =
    sqrt(max(2 * detector.power^2 - detector.squared_power, 0.0))

"""
    CarrierLoopStaging(carrier_loop_filter_type)

A scalar estimator's [`CarrierLoopStage`](@ref) and the phase-lock indicator that
moves it on, read with [`carrier_loop_stage`](@ref) and
[`phase_lock_indicator`](@ref). `CarrierLoopStaging(typeof(carrier_loop_filter))`
starts it (FLL-assisted if the filter has an FLL path, else at the wide PLL); it
then advances with every driver record. Not exported: the estimator states' keyword
constructors start it; only code building a state positionally names it, as
`TrackingLoops.CarrierLoopStaging`.
"""
struct CarrierLoopStaging
    stage::CarrierLoopStage
    lock_detector::PhaseLockDetector
    # The consecutive records the detector has read lock, restarted at each stage and
    # no longer counted once narrowed.
    num_locked_records::Int
end

CarrierLoopStaging(::Type{F}) where {F<:AbstractLoopFilter} = CarrierLoopStaging(
    _has_fll_path(F) ? FLL_ASSISTED_PLL : WIDE_PLL,
    PhaseLockDetector(),
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

# The phase-lock detector after one more prompt.
@inline function _advance_lock_detector(detector::PhaseLockDetector, prompt, time_constant)
    num_records = detector.num_records + 1
    prompt_power = Float64(abs2(prompt))
    updated = PhaseLockDetector(
        _exponential_mean(
            detector.power_difference,
            Float64(real(prompt)^2 - imag(prompt)^2),
            num_records,
            time_constant,
        ),
        _exponential_mean(detector.power, prompt_power, num_records, time_constant),
        _exponential_mean(detector.squared_power, prompt_power^2, num_records, time_constant),
        num_records,
        NaN,
    )
    signal_power = _signal_power(updated)
    indicator =
        num_records < time_constant ? NaN :
        signal_power > 0 ? updated.power_difference / signal_power : 0.0
    PhaseLockDetector(
        updated.power_difference,
        updated.power,
        updated.squared_power,
        num_records,
        indicator,
    )
end

# The staging after one driver `LoopRecord`: its prompt advances the indicator,
# whose lock count moves the stage on for the next record.
@inline function _advance_staging(staging::CarrierLoopStaging, record)
    time_constant = _phase_lock_time_constant(record.signal, record.integrated_code_blocks)
    lock_detector = _advance_lock_detector(
        staging.lock_detector,
        get_prompt(record.filtered_correlator),
        time_constant,
    )
    stage = staging.stage
    num_locked_records =
        stage != NARROW_PLL && lock_detector.indicator >= _PHASE_LOCK_THRESHOLD ?
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
    CarrierLoopStaging(stage, lock_detector, num_locked_records)
end
