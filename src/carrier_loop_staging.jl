# ─────────────────────────────────────────────────────────────────────────────
# The carrier loop's staging: the scalar loops start as an FLL-assisted PLL at a
# wide bandwidth, drop the FLL once a phase-lock indicator has confirmed lock,
# and narrow the PLL once lock has held on. The stages only advance; a fresh
# estimator state and a release from the vector loop restart them, a reset of the
# estimator state keeps them. The vector loop runs at the narrow PLL whatever the
# stage.
# ─────────────────────────────────────────────────────────────────────────────

"""
    CarrierLoopStage

The stage of a satellite's carrier loop, held in the scalar estimators'
per-satellite state (see [`carrier_loop_stage`](@ref)):

  - `FLL_ASSISTED_PLL`: the FLL-assisted PLL at the wide bandwidth
    ([`default_wide_carrier_loop_filter_bandwidth`](@ref)), its FLL path at the
    FLL-assist bandwidth ([`default_fll_assist_loop_filter_bandwidth`](@ref)),
    until the phase-lock indicator ([`phase_lock_indicator`](@ref)) has read lock
    for one time constant of its averages: the carrier phase is locked, so its
    Doppler has converged;
  - `WIDE_PLL`: the PLL alone at the wide bandwidth, until the indicator has read
    lock for another four time constants;
  - `NARROW_PLL`: the PLL alone at the narrow bandwidth
    ([`default_narrow_carrier_loop_filter_bandwidth`](@ref)).

A carrier filter without an FLL path starts at `WIDE_PLL`. Dropping the FLL
feeds the FLL-assisted filter a zero FLL input, which is exactly the
third-order PLL (same state, same coefficients), so the switch is free (Kaplan
& Hegarty §5.5). Dropping the FLL and narrowing are separate steps: narrowing
while the loop still settles from the FLL's last correction risks the lock.

The stages only advance. [`init_estimator_state`](@ref) starts them over, as
does a satellite's release from the vector loop. [`reset_estimator_state`](@ref)
keeps them, since a handoff to a new integration length, say, keeps the lock.
While a satellite is in the vector loop its carrier loop runs at the narrow
bandwidth, and [`carrier_loop_stage`](@ref) reports `NARROW_PLL`.
"""
@enum CarrierLoopStage FLL_ASSISTED_PLL WIDE_PLL NARROW_PLL

# How long the phase-lock indicator must read lock without a break, in time
# constants of its averages, to drop the FLL, and then, counted afresh, to narrow
# the loop.
const _FLL_DROP_TIME_CONSTANTS = 1
const _NARROWING_TIME_CONSTANTS = 4

# The time constant of the phase-lock indicator's averages, in records of
# `num_code_blocks` primary code blocks of `signal`: 0.1 s, and at least 25
# records, so that the signal power it is normalised with is averaged over enough
# of them (100 records of 1 ms, 25 of 4 ms or longer, so 0.25 s for 10 ms and
# 0.5 s for 20 ms). Taken from the nominal record length, counted in chips, so it
# is exact and does not move with a record's sample jitter.
const _PHASE_LOCK_TIME_CONSTANT = 0.1s
const _MIN_PHASE_LOCK_RECORDS = 25

@inline function _phase_lock_time_constant(signal::AbstractGNSSSignal, num_code_blocks)
    chips = round(Int, get_code_frequency(signal) * _PHASE_LOCK_TIME_CONSTANT)
    max(cld(chips, num_code_blocks * get_code_length(signal)), _MIN_PHASE_LOCK_RECORDS)
end

# The phase-lock indicator's threshold of lock: an RMS phase error of about 30°.
const _PHASE_LOCK_THRESHOLD = 0.5

"""
    CarrierLoopStaging(carrier_loop_filter_type)

A scalar estimator's carrier-loop staging on phase lock, in its per-satellite
state: the [`CarrierLoopStage`](@ref) and the phase-lock indicator that moves it
on. Read them off the state with [`carrier_loop_stage`](@ref) and
[`phase_lock_indicator`](@ref). `CarrierLoopStaging(typeof(carrier_loop_filter))`
starts it, FLL-assisted where the carrier loop filter has an FLL path, at the
wide PLL where it has none; the staging then advances with every record of the
loop's driver. It is not exported: the estimator states' keyword constructors
start it, and only code that builds a state positionally names it, as
`TrackingLoops.CarrierLoopStaging`.
"""
struct CarrierLoopStaging
    stage::CarrierLoopStage
    # The phase-lock indicator `⟨I² − Q²⟩ / A²`, an estimate of `cos 2φ`: 1 in
    # phase lock, 0 for a spinning phase. Its averages are exponential, updated
    # every record of the loop's driver over `_phase_lock_time_constant` records,
    # and the signal power `A²` comes from the prompt's moments averaged alike,
    # `√(2 M₂² − M₄)` with `Mₖ` the average of `|P|ᵏ`, which the noise cancels
    # from: unlike `⟨I² − Q²⟩ / ⟨I² + Q²⟩` the indicator does not read low at low
    # C/N₀ while the loop is locked. `power_difference` averages `I² − Q²` (Van
    # Dierendonck's narrowband difference, NBD), `power` and `squared_power`
    # average `|P|²` and `|P|⁴` (M₂ and M₄).
    power_difference::Float64
    power::Float64
    squared_power::Float64
    num_records::Int
    # The latest reading, `NaN` until the averages span one time constant, and
    # for how many consecutive records it has read lock, counted afresh at each
    # stage and no longer once the loop has narrowed, the last stage.
    phase_lock_indicator::Float64
    num_locked_records::Int
end

# The staging a carrier loop starts with: FLL-assisted where its filter has an
# FLL path, the wide PLL where it has none.
CarrierLoopStaging(::Type{F}) where {F<:AbstractLoopFilter} = CarrierLoopStaging(
    _has_fll_path(F) ? FLL_ASSISTED_PLL : WIDE_PLL,
    0.0,
    0.0,
    0.0,
    0,
    NaN,
    0,
)

# Whether a carrier loop filter type has an FLL path (see the methods for the
# estimators in `vector/estimator.jl`).
@inline _has_fll_path(::Type{<:ThirdOrderAssistedBilinearLF}) = true
@inline _has_fll_path(::Type{<:AbstractLoopFilter}) = false

# One step of an exponential mean over `time_constant` samples with the weight
# `max(1/n, 1/time_constant)`: the plain mean of the first `time_constant`
# samples, so it starts unbiased, then exponential.
@inline _exponential_mean(mean, x, n, time_constant) =
    mean + (x - mean) / min(n, time_constant)

# The staging after one record of the driver (a `LoopRecord`): its prompt
# advances the phase-lock indicator, and for how many records that has read lock
# moves the stage on for the next record.
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
