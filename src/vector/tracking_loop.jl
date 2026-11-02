# The per-integration vector loop: a scalar FLL-assisted PLL/DLL that a vector-tracking
# filter can take over (see `VectorTrackingLoop`).

"""
    VectorTrackingLoop(signals...; scalar_loop = ScalarTrackingLoop(ThirdOrderAssistedBilinearLF),
                    filter_config = NavigationFilterConfig(), navigation_filter_update_interval = 100ms,
                    min_tracking_cn0 = 30dBHz,
                    residual_pll_loop_filter_bandwidth = default_residual_pll_loop_filter_bandwidth(),
                    satellite_capacity_per_signal = 16,
                    approximate_year = year(now(UTC)),
                    enable_ionospheric_correction = true,
                    enable_tropospheric_correction = true)

Vector-tracking Doppler loop for the ranging `signals`: each a signal that carries
navigation data (`GPSL1CA()`) or a pilot + data pair `driver => decoding_signal`
(`GalileoE1C() => GalileoE1B()`; see [Pilot + data pairs](@ref)). A decoding signal
must carry data and have its driver's chip rate. No signal may be given twice.

All satellites stepped with this loop share one navigation processor which, inside
[`step_loop!`](@ref), decodes the navigation data, solves the scalar PVT, seeds the
navigation filter from its first fix and then runs one filter cycle every
`navigation_filter_update_interval`, taking satellites over and handing them back. A host needs no
vector-specific code: it builds a satellite's state with
[`init_loop_state`](@ref) for its driver and steps it with every integration of the
satellite. Read the results, keyed by the driver, with
[`get_navigation_solution`](@ref), [`get_navigation_status`](@ref),
[`get_satellite_report`](@ref), [`get_vector_tracking_members`](@ref), [`calc_position_std`](@ref)
and [`calc_clock_bias_std`](@ref).

The navigation processor reads the host's bit clock and C/N₀ estimator from the integrations (see
[What the integrations must carry](@ref)); the C/N₀, read at each navigation epoch for the
lock and the measurement weights, is capped at 80 dB-Hz. A satellite is in lock when
the driver's C/N₀ is at least `min_tracking_cn0` and the decoding signal holds its
bit sync. An integration without a PRN, or of a driver not among `signals`, throws an
`ArgumentError`; any other passenger is ignored, also the decoding signal of a pair
that does not drive this satellite. Step every signal of every satellite at least
once per `navigation_filter_update_interval / 2` (see [Stepping the satellites](@ref)).

Every driver integration runs `scalar_loop` until the filter takes the satellite over, and again
once it is released. In the vector loop (see [The carrier in the vector
loop](@ref vector-carrier)):

  - the navigation filter alone sets the carrier NCO, to its predicted Doppler at the
    middle of the integration the command takes effect on;
  - a residual PLL (`TrackingLoops.ResidualPLL`) follows the carrier on the prompts the
    NCO leaves, without steering it; the scalar loop, its staging included, stands
    still. [`get_phase_lock_indicator`](@ref) and [`is_phase_locked`](@ref) are the residual
    PLL's, and [`get_carrier_loop_stage`](@ref) reports `VECTOR_LOOP`;
  - the code filter is frozen and the filter's code correction replaces its output:
    `code_doppler = reference_code_doppler + code_frequency_correction +
    (carrier_doppler - reference_carrier_doppler) · chip_rate_to_center_frequency_ratio`;
  - the DLL output (chips) and the FLL discriminator (Hz) are accumulated for the
    filter;
  - the navigation processor decodes the decoding signal's prompts rotated by the residual PLL's
    phase, not the host's soft bits. The host's C/N₀ estimator should read power only:
    a coherent one such as [`NWPRCN0Estimator`](@ref) reads the turning prompts as a
    small loss.

A satellite is added at its scalar loop's Doppler and released at its residual PLL's
estimate of the incoming Doppler and its rate (out of lock, at the filter's), into a
restarted staging, FLL-assisted again.

`residual_pll_loop_filter_bandwidth` is the residual PLL's noise bandwidth, capped like
the scalar PLL's ([`effective_carrier_loop_filter_bandwidth`](@ref)). Under the
default `motion_model_order = 2` the prediction does not carry the user's acceleration, so
the residual PLL follows a manoeuvre by itself: a third-order loop's error to a jerk
`j` is `j / (λ ω₀³)` cycles (`ω₀ = 1.2 B_L`), 5.4° at 10 Hz for the 2.5 m/s³ of a
0.5 g manoeuvre built up over two seconds, 43° at 5 Hz. Narrower suits a gentler
platform or `motion_model_order = 3`.

The latest cycle's commands are taken up on the next driver integration, sized for where
they take effect: at `application_sample_index`, or at the integration's end when `application_sample_index = nothing`. An
integration of any other signal returns the command in force there.

`scalar_loop` must have an FLL-assisted carrier filter, which pulls a released satellite in
again: a `ScalarTrackingLoop(ThirdOrderAssistedBilinearLF)` or a
[`DelayCompensatedTrackingLoop`](@ref). Anything else throws an `ArgumentError`, as does a
dataless signal given alone (pair a pilot such as GPS L1C-P or Galileo E1C with its
data component). `filter_config = nothing` only ever solves the scalar PVT.

Storage is allocated for `satellite_capacity_per_signal` satellites per ranging signal
(a `pilot => data` pair is one) and grows only past that; a dropped satellite leaves
its storage to the next one.
"""
struct VectorTrackingLoop{E<:AbstractTrackingLoop,N} <: AbstractTrackingLoop
    scalar_loop::E
    navigation_processor::N
    # Taken on by each satellite's state at `init_loop_state`.
    residual_pll_loop_filter_bandwidth::typeof(1.0Hz)
end

# Whether a loop's carrier filter has an FLL path: an FLL-assisted scalar loop and
# the delay-compensated loop.
_has_fll_path(::AbstractTrackingLoop) = false
_has_fll_path(::ScalarTrackingLoop{CA}) where {CA} = _has_fll_path(CA)
_has_fll_path(::DelayCompensatedTrackingLoop) = true

"""
    VectorLoopState

Per-satellite state of a [`VectorTrackingLoop`](@ref), built by
[`init_loop_state`](@ref) and advanced by [`step_loop!`](@ref). Its fields are
internal and may change in any release. Inspect it with
[`get_phase_lock_indicator`](@ref), [`is_phase_locked`](@ref) and [`get_carrier_loop_stage`](@ref);
[`get_satellite_report`](@ref) has what the navigation processor knows of the satellite.
"""
struct VectorLoopState{S}
    # The scalar loop's state plus the interface to the vector-tracking filter:
    #
    #   - `vector_controlled`: whether the filter controls the NCOs; while `false` the scalar
    #     loop runs and nothing is accumulated.
    #   - `code_disc_sum` / `fll_disc_sum`: `(count, sum)` of the DLL output (chips) and
    #     the FLL discriminator (Hz, integrations with a previous prompt only) since the
    #     filter last read them.
    #   - `previous_replica_tail`: `(cycles, duration)` of the replica's carrier over the
    #     second half of the driver's last integration (`NaN` before the first), which the
    #     residual PLL propagates its phase by, across a hand-over too.
    #   - `code_frequency_correction`: the code correction from the latest cycle.
    #   - `code_frequency_correction_history` / `code_command_delay`: the last three code
    #     corrections, newest first, and how long after the cycle epoch the newest
    #     reached the replica; the next code measurement uses them to move the mid-cycle
    #     mean discriminator to the epoch. Kept by `reinit_loop_state` (see there).
    #   - `ramp` / `residual_pll`: the filter's carrier Doppler ramp and the residual
    #     PLL, set by the first command (which adds the satellite to the vector loop)
    #     and cleared out of it (`_has_carrier_ramp`).
    #   - `slot_index`, `slot_generation`, `cycle_index`: where the navigation processor keeps
    #     this satellite (`0` until its first integration), and the latest cycle taken up.
    scalar_state::S
    vector_controlled::Bool
    code_disc_sum::Tuple{Int,Float64}
    fll_disc_sum::Tuple{Int,typeof(1.0Hz)}
    previous_replica_tail::Tuple{Float64,typeof(1.0s)}
    code_frequency_correction::typeof(1.0Hz)
    code_frequency_correction_history::NTuple{3,typeof(1.0Hz)}
    code_command_delay::typeof(1.0s)
    ramp::CarrierDopplerRamp
    residual_pll::ResidualPLL
    slot_index::Int
    slot_generation::Int
    cycle_index::Int
end

_is_driver_integration(loop_state::VectorLoopState, post_correlation_state::PostCorrelationState) =
    _is_driver_integration(loop_state.scalar_state, post_correlation_state)
_is_driven_by(loop_state::VectorLoopState, signal::AbstractGNSSSignal) =
    _is_driven_by(loop_state.scalar_state, signal)

# A satellite with the vector interface empty, on the slot of `slot_generation`. The
# replica's last half integration and the residual PLL's bandwidth are not part of it and
# are kept.
VectorLoopState(
    scalar_state,
    vector_controlled::Bool,
    slot_index::Int = 0,
    slot_generation::Int = 0,
    cycle_index::Int = -1;
    previous_replica_tail::Tuple{Float64,typeof(1.0s)} = (NaN, NaN * s),
    residual_pll_loop_filter_bandwidth::typeof(1.0Hz) = default_residual_pll_loop_filter_bandwidth(),
) =
    VectorLoopState(
        scalar_state,
        vector_controlled,
        (0, 0.0),
        (0, 0.0Hz),
        previous_replica_tail,
        0.0Hz,
        (0.0Hz, 0.0Hz, 0.0Hz),
        0.0s,
        CarrierDopplerRamp(),
        ResidualPLL(residual_pll_loop_filter_bandwidth),
        slot_index,
        slot_generation,
        cycle_index,
    )

function VectorLoopState(
    loop_state::VectorLoopState{S};
    scalar_state::Maybe{S} = nothing,
    vector_controlled::Maybe{Bool} = nothing,
    code_disc_sum::Maybe{Tuple{Int,Float64}} = nothing,
    fll_disc_sum::Maybe{Tuple{Int,typeof(1.0Hz)}} = nothing,
    previous_replica_tail::Maybe{Tuple{Float64,typeof(1.0s)}} = nothing,
    code_frequency_correction::Maybe{typeof(1.0Hz)} = nothing,
    code_frequency_correction_history::Maybe{NTuple{3,typeof(1.0Hz)}} = nothing,
    code_command_delay::Maybe{typeof(1.0s)} = nothing,
    ramp::Maybe{CarrierDopplerRamp} = nothing,
    residual_pll::Maybe{ResidualPLL} = nothing,
    slot_index::Maybe{Int} = nothing,
    slot_generation::Maybe{Int} = nothing,
    cycle_index::Maybe{Int} = nothing,
) where {S}
    VectorLoopState{S}(
        isnothing(scalar_state) ? loop_state.scalar_state : scalar_state,
        isnothing(vector_controlled) ? loop_state.vector_controlled : vector_controlled,
        isnothing(code_disc_sum) ? loop_state.code_disc_sum : code_disc_sum,
        isnothing(fll_disc_sum) ? loop_state.fll_disc_sum : fll_disc_sum,
        isnothing(previous_replica_tail) ? loop_state.previous_replica_tail :
        previous_replica_tail,
        isnothing(code_frequency_correction) ? loop_state.code_frequency_correction : code_frequency_correction,
        isnothing(code_frequency_correction_history) ? loop_state.code_frequency_correction_history :
        code_frequency_correction_history,
        isnothing(code_command_delay) ? loop_state.code_command_delay :
        code_command_delay,
        isnothing(ramp) ? loop_state.ramp : ramp,
        isnothing(residual_pll) ? loop_state.residual_pll : residual_pll,
        isnothing(slot_index) ? loop_state.slot_index : slot_index,
        isnothing(slot_generation) ? loop_state.slot_generation : slot_generation,
        isnothing(cycle_index) ? loop_state.cycle_index : cycle_index,
    )
end

# The state on a fresh slot, owing the pick-up of the latest cycle.
_registered(loop_state::VectorLoopState, slot_index::Int, slot_generation::Int) =
    VectorLoopState(loop_state; slot_index, slot_generation, cycle_index = -1)

"""
    init_loop_state(tracking_loop::VectorTrackingLoop, driver_signal, carrier_doppler, code_doppler)

The scalar loop's state with the vector interface empty (out of the vector loop,
nothing accumulated, no commands) and the residual PLL at the loop's
`residual_pll_loop_filter_bandwidth`. The satellite registers with the navigation
processor on its first integration.
"""
init_loop_state(
    tracking_loop::VectorTrackingLoop,
    driver_signal::AbstractGNSSSignal,
    carrier_doppler,
    code_doppler,
) = VectorLoopState(
    init_loop_state(tracking_loop.scalar_loop, driver_signal, carrier_doppler, code_doppler),
    false;
    residual_pll_loop_filter_bandwidth = tracking_loop.residual_pll_loop_filter_bandwidth,
)

"""
    reinit_loop_state(tracking_loop::VectorTrackingLoop, loop_state, carrier_doppler, code_doppler)

Re-seed the scalar loop from the converged Dopplers, zero the accumulators and drop
the code correction, keeping `vector_controlled`, the carrier ramp, the residual PLL and
the satellite's place in the navigation processor. The converged Dopplers already contain
the code correction, so keeping it would apply it twice; the replica was still steered
by it, so the correction history and application_sample_index lead are kept. The carrier ramp replaces
the scalar loop's carrier command rather than adding to it, so it stays.
"""
reinit_loop_state(
    tracking_loop::VectorTrackingLoop,
    loop_state::VectorLoopState,
    carrier_doppler,
    code_doppler,
) = VectorLoopState(
    loop_state;
    scalar_state = reinit_loop_state(tracking_loop.scalar_loop, loop_state.scalar_state, carrier_doppler, code_doppler),
    code_disc_sum = (0, 0.0),
    fll_disc_sum = (0, 0.0Hz),
    code_frequency_correction = 0.0Hz,
)

get_carrier_loop_stage(loop_state::VectorLoopState) =
    loop_state.vector_controlled ? VECTOR_LOOP : get_carrier_loop_stage(loop_state.scalar_state)
get_phase_lock_indicator(loop_state::VectorLoopState) =
    loop_state.vector_controlled ? loop_state.residual_pll.lock.lock_detector.indicator :
    get_phase_lock_indicator(loop_state.scalar_state)
is_phase_locked(loop_state::VectorLoopState) =
    loop_state.vector_controlled ? loop_state.residual_pll.lock.locked : is_phase_locked(loop_state.scalar_state)
function calc_carrier_phase_correction(loop_state::VectorLoopState, nco_history, sample_index, sampling_frequency)
    residual_pll = loop_state.residual_pll
    loop_state.vector_controlled || return 0.0
    # Both are set by the first driver integration.
    isnan(residual_pll.last_prompt_sample_index) && return 0.0
    isnan(first(loop_state.previous_replica_tail)) && return 0.0
    fs = Float64(ustrip(Hz, sampling_frequency))
    _residual_phase_after(residual_pll, loop_state.ramp, loop_state.previous_replica_tail, sample_index, fs, nco_history)
end


# ─────────────────────────────────────────────────────────────────────────────
# What the navigation processor does to a satellite's state; each keeps its place in
# the navigation processor.

# Into the vector loop with the vector interface cleared, so it never runs on stale
# values; a no-op if already in it.
_add_to_vector_loop(loop_state::VectorLoopState) =
    loop_state.vector_controlled ? loop_state :
    VectorLoopState(
        loop_state.scalar_state,
        true,
        loop_state.slot_index,
        loop_state.slot_generation,
        loop_state.cycle_index;
        loop_state.previous_replica_tail,
        residual_pll_loop_filter_bandwidth = loop_state.residual_pll.bandwidth,
    )

# Out of the vector loop with the vector interface cleared and the scalar loop as it
# stands, not re-seeded: it marks the release on the slot's copy of the state
# (`_mark_release!`; the satellite's own state hands over in `_take_up_cycle`) and
# registers a satellite out of the loop.
_release_from_vector_loop(loop_state::VectorLoopState) =
    VectorLoopState(
        loop_state.scalar_state,
        false,
        loop_state.slot_index,
        loop_state.slot_generation,
        loop_state.cycle_index;
        loop_state.previous_replica_tail,
        residual_pll_loop_filter_bandwidth = loop_state.residual_pll.bandwidth,
    )

# `_release_from_vector_loop` with the scalar loop re-seeded at the sample `application_sample_index`
# (integrations' grid at `fs`, Hz) where the release takes effect: at the residual PLL's estimate of
# the incoming Doppler (ramp plus residual) and its rate, which a third-order carrier
# filter starts from. A residual PLL out of lock (or never stepped) knows no better than
# the filter, so then the replica's carrier command there and the ramp's rate. The code
# starts from the replica's code command. Unlike a reset, it restarts the staging, so the
# scalar loop pulls in again FLL-assisted at the wide bandwidth.
function _hand_over_to_scalar_loop(loop_state::VectorLoopState, nco_history, application_sample_index, fs)
    nco_command = calc_mean_nco_command(nco_history, application_sample_index, application_sample_index)
    carrier_command, code_command = nco_command.carrier_doppler, nco_command.code_doppler
    carrier_doppler = carrier_command * Hz
    code_doppler = code_command * Hz
    time = application_sample_index / fs
    ramp, residual_pll = loop_state.ramp, loop_state.residual_pll
    rate = ramp.doppler_rate
    if residual_pll.lock.locked && !isnan(residual_pll.last_prompt_sample_index)
        dt = (application_sample_index - residual_pll.last_prompt_sample_index) / fs
        carrier_doppler =
            (_doppler_at(ramp, time) + _residual_doppler(residual_pll) + _residual_doppler_rate(residual_pll) * dt) * Hz
        rate += _residual_doppler_rate(residual_pll)
    end
    VectorLoopState(
        _release_from_vector_loop(loop_state);
        scalar_state = _reinit_scalar_state(loop_state.scalar_state, carrier_doppler, code_doppler, rate),
    )
end

# The scalar reset reads only the state, never the loop's configuration, so a
# default loop stands in. The carrier filter starts at the Doppler rate `rate`
# (Hz/s).
_reinit_scalar_state(loop_state::ScalarLoopState, carrier_doppler, code_doppler, rate) =
    ScalarLoopState(
        reinit_loop_state(ScalarTrackingLoop(), loop_state, carrier_doppler, code_doppler);
        carrier_loop_filter = _carrier_filter_at_rate(loop_state.carrier_loop_filter, rate),
        staging = CarrierLoopStaging(typeof(loop_state.carrier_loop_filter)),
    )
# A fresh carrier filter of `filter`'s type at the Doppler rate `rate` (Hz/s), kept in
# `x2` by a third-order one; one of lower order starts at zero.
_carrier_filter_at_rate(filter, rate) =
    hasfield(typeof(filter), :x2) ? typeof(filter)(zero(filter.x1), rate * Hz^2) :
    _reset_loop_filter(filter)

# Set the latest cycle's command, applied `application_lead` after the cycle epoch: the
# code correction (pushed onto `code_frequency_correction_history`) and the carrier `ramp`. A
# satellite just added starts its residual PLL from the scalar loop's carrier command at
# `application_sample_index`; a member keeps its residual PLL's estimate of the incoming Doppler.
# `fs` (Hz) is the integrations' sampling frequency.
function _set_vector_commands(
    loop_state::VectorLoopState,
    code_frequency_correction,
    ramp::CarrierDopplerRamp,
    application_lead,
    nco_history,
    application_sample_index,
    fs,
)
    newest, previous, _ = loop_state.code_frequency_correction_history
    residual_pll = if _has_carrier_ramp(loop_state)
        _residual_pll_on_new_ramp(loop_state.residual_pll, loop_state.ramp, ramp, fs)
    else
        scalar_doppler = calc_mean_nco_command(nco_history, application_sample_index, application_sample_index).carrier_doppler
        _initialized_residual_pll(loop_state.residual_pll, ramp, scalar_doppler, ramp.reference_time, loop_state.scalar_state.staging)
    end
    VectorLoopState(
        loop_state;
        code_frequency_correction,
        code_frequency_correction_history = (code_frequency_correction, newest, previous),
        code_command_delay = application_lead,
        ramp,
        residual_pll,
    )
end

# Whether the satellite has a carrier ramp: from the filter's first command until it
# leaves the vector loop. A member always has one; `_set_vector_commands` reads this to
# tell the first command from the later ones.
_has_carrier_ramp(loop_state::VectorLoopState) = !isnan(loop_state.ramp.reference_time)

# Empty both accumulators once the filter has read them.
_reset_discriminator_accumulators(loop_state::VectorLoopState) =
    VectorLoopState(loop_state; code_disc_sum = (0, 0.0), fll_disc_sum = (0, 0.0Hz))

# The mean of a discriminator sum `(count, sum)`, or `nothing` if empty.
_disc_mean((count, disc_sum)) = count == 0 ? nothing : disc_sum / count

# `(cycles, duration)` the replica's carrier advanced by over each half of `integration`,
# from the commands it ran on (Hz).
@inline function _replica_halves(post_correlation_state::PostCorrelationState, nco_history)
    fs = post_correlation_state.sampling_frequency
    integration_end_sample_index = post_correlation_state.last_sample_index
    half_samples = post_correlation_state.last_num_integrated_samples / 2
    center = integration_end_sample_index - half_samples
    first_command = calc_mean_nco_command(nco_history, integration_end_sample_index - post_correlation_state.last_num_integrated_samples, center).carrier_doppler
    second_command = calc_mean_nco_command(nco_history, center, integration_end_sample_index).carrier_doppler
    half_duration = convert(typeof(1.0s), half_samples / fs)
    first_cycles = Float64(uconvert(NoUnits, first_command * Hz * half_duration))
    second_cycles = Float64(uconvert(NoUnits, second_command * Hz * half_duration))
    (first = (first_cycles, half_duration), second = (second_cycles, half_duration))
end

# The FLL accumulator with `integration`'s discriminator added. An integration without a previous
# prompt (see `PostCorrelationState`) has no frequency measurement and is left out, not taken as
# 0 Hz.
@inline _accumulate_fll((count, disc_sum), fll_discriminator, post_correlation_state::PostCorrelationState) =
    iszero(post_correlation_state.last_previous_filtered_prompt) ? (count, disc_sum) :
    (count + 1, disc_sum + fll_discriminator)

# One integration through the scalar loop, or in the vector loop through `_step_vector_loop`.
@inline function _step_satellite(
    tracking_loop::VectorTrackingLoop,
    loop_state::VectorLoopState,
    post_correlation_state::PostCorrelationState,
    nco_history,
    application_sample_index::Union{Nothing,Int64},
)
    replica_halves = _replica_halves(post_correlation_state, nco_history)
    if !loop_state.vector_controlled
        scalar_state, carrier_doppler, code_doppler =
            step_loop!(tracking_loop.scalar_loop, loop_state.scalar_state, post_correlation_state, nco_history, application_sample_index)
        return VectorLoopState(loop_state; scalar_state, previous_replica_tail = replica_halves.second),
        carrier_doppler,
        code_doppler
    end
    _step_vector_loop(loop_state, post_correlation_state, nco_history, application_sample_index, replica_halves, tracking_loop.navigation_processor.min_tracking_cn0_dbhz)
end

# An integration's DLL output (chips), normalised by the code command the replica ran on, as
# the scalar loops measure it.
@inline function _code_discriminator(post_correlation_state::PostCorrelationState, nco_history)
    integration_start_sample_index = post_correlation_state.last_sample_index - post_correlation_state.last_num_integrated_samples
    applied_code = calc_mean_nco_command(nco_history, integration_start_sample_index, post_correlation_state.last_sample_index).code_doppler
    dll_disc(post_correlation_state.signal, post_correlation_state.last_filtered_correlator_bank, applied_code * Hz, post_correlation_state.sampling_frequency)
end

# One integration of a member: the DLL accumulated, the carrier measured
# (`_measure_carrier`), and the next command: the ramp at the middle of the integration it
# takes effect on and, carrier-aided, the code correction. The scalar loop and its staging
# stand still; a release re-seeds them.
@inline function _step_vector_loop(
    loop_state::VectorLoopState,
    post_correlation_state::PostCorrelationState,
    nco_history,
    application_sample_index::Union{Nothing,Int64},
    replica_halves,
    min_tracking_cn0_dbhz::Float64,
)
    scalar_state = loop_state.scalar_state
    application_sample_index = isnothing(application_sample_index) ? post_correlation_state.last_sample_index : application_sample_index
    residual_pll, fll_disc_sum = _measure_carrier(loop_state, post_correlation_state, nco_history, replica_halves, min_tracking_cn0_dbhz)
    # The command takes effect on an integration taken to be this one's length.
    carrier =
        _doppler_at(loop_state.ramp, (application_sample_index + post_correlation_state.last_num_integrated_samples / 2) / _sampling_frequency_hz(post_correlation_state))
    carrier_doppler, code_doppler = calc_carrier_aided_dopplers(
        post_correlation_state.signal,
        scalar_state.reference_carrier_doppler,
        scalar_state.reference_code_doppler,
        carrier * Hz - scalar_state.reference_carrier_doppler,
        loop_state.code_frequency_correction,
    )
    VectorLoopState(
        loop_state;
        code_disc_sum = loop_state.code_disc_sum .+ (1, _code_discriminator(post_correlation_state, nco_history)),
        fll_disc_sum,
        previous_replica_tail = replica_halves.second,
        residual_pll,
    ),
    carrier_doppler,
    code_doppler
end

# A member's carrier: the FLL accumulated, the residual PLL stepped on the prompt and
# its lock tested (`_with_lock_test`). Returns the residual PLL and the FLL accumulator.
# Both span the previous prompt's centre to this one's, including any gap a host left
# between the integrations (the replica ran on through it on the commands in force).
@inline function _measure_carrier(
    loop_state::VectorLoopState,
    post_correlation_state::PostCorrelationState,
    nco_history,
    replica_halves,
    min_tracking_cn0_dbhz::Float64,
)
    fs = _sampling_frequency_hz(post_correlation_state)
    coherent_integration_time = post_correlation_state.last_num_integrated_samples / fs
    previous_cycles, previous_duration = loop_state.previous_replica_tail
    first_cycles, first_duration = replica_halves.first
    integration_start_sample_index = post_correlation_state.last_sample_index - post_correlation_state.last_num_integrated_samples
    # The previous integration's end, known once the residual PLL has stepped on its prompt.
    previous_end_sample_index = loop_state.residual_pll.last_prompt_sample_index + ustrip(s, previous_duration) * fs
    gap = isnan(previous_end_sample_index) ? 0.0 : (integration_start_sample_index - previous_end_sample_index) / fs
    gap_cycles = 0.0
    if gap > 0.5 / fs
        gap_command = calc_mean_nco_command(nco_history, previous_end_sample_index, integration_start_sample_index).carrier_doppler
        gap_cycles = gap_command * gap
    else
        gap = 0.0
    end
    span = ustrip(s, previous_duration) + gap + ustrip(s, first_duration)
    # The incoming carrier against the replica between the prompts' centres; the replica
    # runs on the filter's command alone, so only the signal moves this.
    fll_discriminator = fll_disc(
        post_correlation_state.signal,
        post_correlation_state.last_filtered_correlator_bank,
        post_correlation_state.last_previous_filtered_prompt,
        (isnan(span) ? coherent_integration_time : span) * s;
        polarity = post_correlation_state.last_carrier_phase_polarity,
    )
    residual_pll, wiped_prompt =
        _step_residual_pll(loop_state.residual_pll, loop_state.ramp, post_correlation_state, previous_cycles + gap_cycles + first_cycles)
    host_cn0_dbhz = Float64(ustrip(estimate_cn0(post_correlation_state.cn0_estimator, coherent_integration_time * s)))
    residual_pll = _with_lock_test(
        residual_pll,
        wiped_prompt,
        coherent_integration_time,
        _phase_lock_time_constant(post_correlation_state.signal, post_correlation_state.last_num_code_periods),
        min_tracking_cn0_dbhz,
        host_cn0_dbhz >= min_tracking_cn0_dbhz,
    )
    residual_pll, _accumulate_fll(loop_state.fll_disc_sum, fll_discriminator, post_correlation_state)
end

