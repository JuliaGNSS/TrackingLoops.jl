# The per-integration vector loop: a scalar FLL-assisted PLL/DLL that a vector-tracking
# filter can take over (see `VectorTrackingLoop`).

"""
    VectorTrackingLoop(signals...; scalar_loop = ScalarTrackingLoop(ThirdOrderAssistedBilinearLF),
                    nav_filter_config = NavFilterConfig(), navigation_update_interval = 100ms,
                    min_tracking_cn0 = 30dBHz,
                    residual_pll_loop_bandwidth = default_residual_pll_loop_bandwidth(),
                    satellite_capacity_per_signal = 16,
                    approximate_year = year(now(UTC)),
                    enable_ionospheric_correction = true,
                    enable_tropospheric_correction = true)

Vector-tracking loop for the ranging `signals`: each a signal that carries
navigation data (`GPSL1CA()`) or a pilot + data pair `driver => decoding_signal`
(`GalileoE1C() => GalileoE1B()`; see [Pilot + data pairs](@ref)). A decoding signal
must carry data and have its driver's chip rate. No signal may be given twice.

All satellites stepped with this loop share one navigation processor which, inside
[`step_loop!`](@ref), decodes the navigation data, solves the scalar PVT, seeds the
navigation filter from its first fix and then runs one filter update every
`navigation_update_interval`, taking satellites over and handing them back. A host needs no
vector-specific code: it builds a satellite's state with
[`init_loop_state`](@ref) for its driver and steps it with every integration of the
satellite. Read the results, keyed by the driver, with
[`get_navigation_solution`](@ref), [`get_navigation_status`](@ref) (with the filter's
`position_std` and `clock_bias_std`), [`get_satellite_report`](@ref) and
[`get_vector_tracking_members`](@ref).

The navigation processor reads the host's bit clock and C/N₀ estimator from the integrations (see
[What the integrations must carry](@ref)); the C/N₀, read at each navigation epoch for the
lock and the measurement weights, is capped at 80 dB-Hz. A satellite is in lock when
the driver's C/N₀ is at least `min_tracking_cn0` and the decoding signal holds its
bit sync. An integration without a PRN, or of a driver not among `signals`, throws an
`ArgumentError`; any other passenger is ignored, also the decoding signal of a pair
that does not drive this satellite. Step every signal of every satellite at least
once per `navigation_update_interval / 2` (see [Stepping the satellites](@ref)).

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
    `code_doppler = initial_code_doppler + code_frequency_correction +
    (carrier_doppler - initial_carrier_doppler) · chip_rate_to_center_frequency_ratio`;
  - the DLL output (chips) and the FLL discriminator (Hz) are accumulated for the
    filter;
  - the navigation processor decodes the decoding signal's prompts rotated by the residual PLL's
    phase, not the host's soft bits. The host's C/N₀ estimator should read power only:
    a coherent one such as [`NWPRCN0Estimator`](@ref) reads the turning prompts as a
    small loss.

A satellite is added at its scalar loop's Doppler and released at its residual PLL's
estimate of the incoming Doppler and its rate (out of lock, at the filter's), into a
restarted staging, FLL-assisted again.

`residual_pll_loop_bandwidth` is the residual PLL's noise bandwidth, capped like
the scalar PLL's ([`calc_effective_carrier_loop_bandwidth`](@ref)). Under the
default `motion_model_order = 2` the prediction does not carry the user's acceleration, so
the residual PLL follows a manoeuvre by itself: a third-order loop's error to a jerk
`j` is `j / (λ ω₀³)` cycles (`ω₀ = 1.2 B_L`), 5.4° at 10 Hz for the 2.5 m/s³ of a
0.5 g manoeuvre built up over two seconds, 43° at 5 Hz. Narrower suits a gentler
platform or `motion_model_order = 3`.

The latest update's commands are applied on the next driver integration, sized for where
they take effect: at `application_sample_index`, or at the integration's end when `application_sample_index = nothing`. An
integration of any other signal returns the command in force there.

`scalar_loop` is a [`ScalarTrackingLoop`](@ref) or a [`DelayCompensatedTrackingLoop`](@ref),
whose per-satellite state the vector loop holds and hands over. A dataless signal given
alone throws an `ArgumentError` (pair a pilot such as GPS L1C-P or Galileo E1C with its
data component).

Storage is allocated for `satellite_capacity_per_signal` satellites per ranging signal
(a `pilot => data` pair is one) and grows only past that; a dropped satellite leaves
its storage to the next one.
"""
struct VectorTrackingLoop{E<:Union{ScalarTrackingLoop,DelayCompensatedTrackingLoop},N} <: AbstractTrackingLoop
    scalar_loop::E
    navigation_processor::N
    # Taken on by each satellite's state at `init_loop_state`.
    residual_pll_loop_bandwidth::typeof(1.0Hz)
end

# The loop's ranging signals as `(driver, decoding signal)` pairs; a plain signal
# is both.
_ranging_pair(signal::AbstractGNSSSignal) = (signal, signal)
_ranging_pair(pair::Pair{<:AbstractGNSSSignal,<:AbstractGNSSSignal}) = (pair.first, pair.second)

_is_plain_pair((driver, decoding)) = typeof(driver) == typeof(decoding)

function _check_ranging_pair((driver, decoding))
    if iszero(get_bit_rate(decoding))
        hint =
            _is_plain_pair((driver, decoding)) ?
            "; pair it with its data component, `$(nameof(typeof(driver)))() => data`" : ""
        throw(
            ArgumentError(
                "vector tracking decodes the navigation data of each satellite's decoding " *
                "signal; $(nameof(typeof(decoding))) carries none" * hint,
            ),
        )
    end
    get_chip_rate(decoding) == get_chip_rate(driver) || throw(
        ArgumentError(
            "a decoding signal must have its driver's chip rate: " *
            "$(nameof(typeof(decoding))) has $(get_chip_rate(decoding)), " *
            "$(nameof(typeof(driver))) $(get_chip_rate(driver))",
        ),
    )
    nothing
end

function VectorTrackingLoop(
    signals::Union{AbstractGNSSSignal,Pair{<:AbstractGNSSSignal,<:AbstractGNSSSignal}}...;
    scalar_loop::Union{ScalarTrackingLoop,DelayCompensatedTrackingLoop} = ScalarTrackingLoop(ThirdOrderAssistedBilinearLF),
    nav_filter_config::NavFilterConfig = NavFilterConfig(),
    navigation_update_interval = 100.0ms,
    min_tracking_cn0 = 30.0dBHz,
    residual_pll_loop_bandwidth = default_residual_pll_loop_bandwidth(),
    satellite_capacity_per_signal::Integer = 16,
    approximate_year::Integer = year(now(UTC)),
    enable_ionospheric_correction::Bool = true,
    enable_tropospheric_correction::Bool = true,
)
    isempty(signals) &&
        throw(ArgumentError("vector tracking needs at least one ranging signal"))
    pairs = map(_ranging_pair, signals)
    signal_types = DataType[]
    for pair in pairs
        push!(signal_types, typeof(first(pair)))
        _is_plain_pair(pair) || push!(signal_types, typeof(last(pair)))
    end
    allunique(signal_types) ||
        throw(ArgumentError("each signal can be given only once, as a driver or a decoding signal"))
    foreach(_check_ranging_pair, pairs)
    satellite_capacity_per_signal >= 1 ||
        throw(ArgumentError("`satellite_capacity_per_signal` must be at least 1"))
    Δt_nav = Float64(ustrip(s, navigation_update_interval)) * s
    0.0s < Δt_nav < Inf * s ||
        throw(ArgumentError("`navigation_update_interval` must be positive and finite"))
    residual_pll_loop_bandwidth > 0.0Hz ||
        throw(ArgumentError("`residual_pll_loop_bandwidth` must be positive"))
    navigation_processor = NavigationProcessor(
        nav_filter_config,
        pairs;
        navigation_update_interval = Δt_nav,
        min_tracking_cn0 = convert(typeof(1.0dBHz), min_tracking_cn0),
        satellite_capacity_per_signal = Int(satellite_capacity_per_signal),
        approximate_year,
        enable_ionospheric_correction,
        enable_tropospheric_correction,
    )
    VectorTrackingLoop(scalar_loop, navigation_processor, Float64(ustrip(Hz, residual_pll_loop_bandwidth)) * Hz)
end

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
    #   - `discriminators`: what the filter measures the satellite by, accumulated since it
    #     last read them (`DiscriminatorAccumulator`); its correction history and feedback
    #     delay are kept by `reinit_loop_state` (see there).
    #   - `last_replica_tail`: the cycles of the replica's carrier over the second half
    #     of the driver's last integration (`NaN` before the first), which the residual PLL
    #     propagates its phase by, across a hand-over too.
    #   - `code_frequency_correction`: the code correction from the latest update.
    #   - `ramp` / `residual_pll`: the filter's carrier Doppler ramp and the residual
    #     PLL, set by the first command (which adds the satellite to the vector loop)
    #     and cleared out of it (`_has_carrier_ramp`).
    #   - `slot_index`, `slot_generation`, `applied_update_index`: where the navigation processor keeps
    #     this satellite (`0` until its first integration), and the latest update applied.
    scalar_state::S
    vector_controlled::Bool
    discriminators::DiscriminatorAccumulator
    last_replica_tail::Float64
    code_frequency_correction::typeof(1.0Hz)
    ramp::CarrierDopplerRamp
    residual_pll::ResidualPLL
    slot_index::Int
    slot_generation::Int
    applied_update_index::Int
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
    applied_update_index::Int = -1;
    last_replica_tail::Float64 = NaN,
    residual_pll_loop_bandwidth::typeof(1.0Hz) = default_residual_pll_loop_bandwidth(),
) =
    VectorLoopState(
        scalar_state,
        vector_controlled,
        DiscriminatorAccumulator(),
        last_replica_tail,
        0.0Hz,
        CarrierDopplerRamp(),
        ResidualPLL(residual_pll_loop_bandwidth),
        slot_index,
        slot_generation,
        applied_update_index,
    )

function VectorLoopState(
    loop_state::VectorLoopState{S};
    scalar_state::Maybe{S} = nothing,
    vector_controlled::Maybe{Bool} = nothing,
    discriminators::Maybe{DiscriminatorAccumulator} = nothing,
    last_replica_tail::Maybe{Float64} = nothing,
    code_frequency_correction::Maybe{typeof(1.0Hz)} = nothing,
    ramp::Maybe{CarrierDopplerRamp} = nothing,
    residual_pll::Maybe{ResidualPLL} = nothing,
    slot_index::Maybe{Int} = nothing,
    slot_generation::Maybe{Int} = nothing,
    applied_update_index::Maybe{Int} = nothing,
) where {S}
    VectorLoopState{S}(
        isnothing(scalar_state) ? loop_state.scalar_state : scalar_state,
        isnothing(vector_controlled) ? loop_state.vector_controlled : vector_controlled,
        isnothing(discriminators) ? loop_state.discriminators : discriminators,
        isnothing(last_replica_tail) ? loop_state.last_replica_tail :
        last_replica_tail,
        isnothing(code_frequency_correction) ? loop_state.code_frequency_correction : code_frequency_correction,
        isnothing(ramp) ? loop_state.ramp : ramp,
        isnothing(residual_pll) ? loop_state.residual_pll : residual_pll,
        isnothing(slot_index) ? loop_state.slot_index : slot_index,
        isnothing(slot_generation) ? loop_state.slot_generation : slot_generation,
        isnothing(applied_update_index) ? loop_state.applied_update_index : applied_update_index,
    )
end

# The state on a fresh slot, owing the latest update's feedback.
_registered(loop_state::VectorLoopState, slot_index::Int, slot_generation::Int) =
    VectorLoopState(loop_state; slot_index, slot_generation, applied_update_index = -1)

"""
    init_loop_state(tracking_loop::VectorTrackingLoop, driver_signal, carrier_doppler, code_doppler)

The scalar loop's state with the vector interface empty (out of the vector loop,
nothing accumulated, no commands) and the residual PLL at the loop's
`residual_pll_loop_bandwidth`. The satellite registers with the navigation
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
    residual_pll_loop_bandwidth = tracking_loop.residual_pll_loop_bandwidth,
)

"""
    reinit_loop_state(tracking_loop::VectorTrackingLoop, loop_state, carrier_doppler, code_doppler)

Re-initialise the scalar loop at the converged Dopplers, zero the accumulators and drop
the code correction, keeping `vector_controlled`, the carrier ramp, the residual PLL and
the satellite's place in the navigation processor. The converged Dopplers already contain
the code correction, so keeping it would apply it twice; the replica was still steered
by it, so the correction history and the feedback delay are kept. The carrier ramp replaces
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
    discriminators = _with_sums_emptied(loop_state.discriminators),
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
    _has_span(residual_pll.last_prompt_span) || return 0.0
    isnan(loop_state.last_replica_tail) && return 0.0
    fs = Float64(ustrip(Hz, sampling_frequency))
    _residual_phase_after(loop_state, sample_index - residual_pll.last_prompt_span.end_sample_index, fs, nco_history)
end

# ─────────────────────────────────────────────────────────────────────────────
# What the navigation processor does to a satellite's state; each keeps its place in
# the navigation processor.

# Into (`true`) or out of (`false`) the vector loop, with the vector interface cleared so
# it never runs on stale values and the scalar loop as it stands, not re-initialised (a
# satellite leaving the vector loop hands over in `_hand_over_to_scalar_loop`). The
# replica's last half integration and the residual PLL's bandwidth are kept.
_set_vector_controlled(loop_state::VectorLoopState, vector_controlled::Bool) =
    VectorLoopState(
        loop_state.scalar_state,
        vector_controlled,
        loop_state.slot_index,
        loop_state.slot_generation,
        loop_state.applied_update_index;
        loop_state.last_replica_tail,
        residual_pll_loop_bandwidth = loop_state.residual_pll.bandwidth,
    )

# Out of the vector loop (`_set_vector_controlled`) with `scalar_loop`'s state re-initialised at the sample `application_sample_index`
# (integrations' grid at `fs`, Hz) where the release takes effect: at the residual PLL's estimate of
# the incoming Doppler (ramp plus residual) and its rate, which a third-order carrier
# filter starts from. A residual PLL out of lock (or never stepped) knows no better than
# the filter, so then the replica's carrier command there and the ramp's rate. The code
# starts from the replica's code command. Unlike a reset, it restarts the staging, so the
# scalar loop pulls in again FLL-assisted at the wide bandwidth.
function _hand_over_to_scalar_loop(scalar_loop, loop_state::VectorLoopState, nco_history, application_sample_index, fs)
    (; carrier_doppler, code_doppler) = get_nco_command(nco_history, application_sample_index)
    ramp, residual_pll = loop_state.ramp, loop_state.residual_pll
    rate = ustrip(Hz / s, ramp.doppler_rate)
    if residual_pll.lock.locked && _has_span(residual_pll.last_prompt_span)
        dt = _samples_since_center(residual_pll.last_prompt_span, application_sample_index) / fs
        carrier_doppler =
            (_doppler_at(ramp, application_sample_index, fs) + _residual_doppler(residual_pll) + _residual_doppler_rate(residual_pll) * dt) * Hz
        rate += _residual_doppler_rate(residual_pll)
    end
    scalar_state = loop_state.scalar_state
    VectorLoopState(
        _set_vector_controlled(loop_state, false);
        scalar_state = ScalarLoopState(
            reinit_loop_state(scalar_loop, scalar_state, carrier_doppler, code_doppler);
            carrier_loop_filter = _carrier_filter_at_rate(scalar_state.carrier_loop_filter, rate),
            staging = CarrierLoopStaging(typeof(scalar_state.carrier_loop_filter)),
        ),
    )
end

# Set the latest update's command, applied `feedback_delay` after the measurement epoch: the
# code correction (pushed onto the correction history) and the carrier `ramp`. A
# satellite just added starts its residual PLL from the scalar loop's carrier command at
# `application_sample_index`; a member keeps its residual PLL's estimate of the incoming Doppler.
# `fs` (Hz) is the integrations' sampling frequency.
function _set_vector_commands(
    loop_state::VectorLoopState,
    code_frequency_correction,
    ramp::CarrierDopplerRamp,
    feedback_delay,
    nco_history,
    application_sample_index,
    fs,
)
    discriminators = loop_state.discriminators
    newest, previous, _ = discriminators.code_frequency_correction_history
    residual_pll = if _has_carrier_ramp(loop_state)
        _residual_pll_on_new_ramp(loop_state.residual_pll, loop_state.ramp, ramp, fs)
    else
        scalar_doppler = ustrip(Hz, get_nco_command(nco_history, application_sample_index).carrier_doppler)
        _initialized_residual_pll(loop_state.residual_pll, ramp, scalar_doppler, loop_state.scalar_state.staging)
    end
    VectorLoopState(
        loop_state;
        code_frequency_correction,
        discriminators = DiscriminatorAccumulator(
            discriminators.code_disc_sum,
            discriminators.fll_disc_sum,
            (code_frequency_correction, newest, previous),
            feedback_delay,
        ),
        ramp,
        residual_pll,
    )
end

# Whether the satellite has a carrier ramp: from the filter's first command until it
# leaves the vector loop. A member always has one; `_set_vector_commands` reads this to
# tell the first command from the later ones.
_has_carrier_ramp(loop_state::VectorLoopState) = loop_state.ramp.start_sample_index != typemin(Int)

# Empty both accumulators once the filter has read them.
_reset_discriminator_accumulators(loop_state::VectorLoopState) =
    VectorLoopState(loop_state; discriminators = _with_sums_emptied(loop_state.discriminators))

# The mean of a discriminator sum `(count, sum)`, or `nothing` if empty.
_disc_mean((count, disc_sum)) = count == 0 ? nothing : disc_sum / count

# Whether a member accumulated both a code and a carrier measurement this update (none
# without a full coherent integration, e.g. admitted at the update's end or starved of samples). The zero
# `calc_mean_code_disc_at_epoch` and `calc_mean_fll_disc` substitute would be a confidently zero
# residual at full `R` weight, pulling the state toward the NCO, so such members are
# withheld from the measurements (still commanded, like a member out of code lock). The
# carrier sum counts only integrations with a previous prompt, so it can be empty alone (a
# update whose only integration is the first after a (re)start); that member is withheld too.
has_disc_sums(discriminators::DiscriminatorAccumulator) =
    first(discriminators.code_disc_sum) > 0 && first(discriminators.fll_disc_sum) > 0

# The cycles the replica's carrier advanced by over each half of `integration`, from the
# commands it ran on, and the mean command over the whole integration (`command`).
@inline function _replica_halves(post_correlation_state::PostCorrelationState, nco_history)
    fs = post_correlation_state.sampling_frequency
    integration_end_sample_index = post_correlation_state.last_sample_index
    half_samples = post_correlation_state.last_num_integrated_samples / 2
    center = _integration_center(post_correlation_state)
    first_command = calc_mean_nco_command(nco_history, _integration_start(post_correlation_state), center)
    second_command = calc_mean_nco_command(nco_history, center, integration_end_sample_index)
    half_duration = convert(typeof(1.0s), half_samples / fs)
    first_cycles = ustrip(NoUnits, first_command.carrier_doppler * half_duration)
    second_cycles = ustrip(NoUnits, second_command.carrier_doppler * half_duration)
    command = NCOCommand(
        (first_command.carrier_doppler + second_command.carrier_doppler) / 2,
        (first_command.code_doppler + second_command.code_doppler) / 2,
    )
    (first = first_cycles, second = second_cycles, command)
end

# The FLL accumulator with `integration`'s discriminator added. An integration without a previous
# prompt (see `PostCorrelationState`) has no frequency measurement and is left out, not taken as
# 0 Hz.
@inline _accumulate_fll((count, disc_sum), fll_discriminator, post_correlation_state::PostCorrelationState) =
    iszero(post_correlation_state.previous_filtered_prompt) ? (count, disc_sum) :
    (count + 1, disc_sum + fll_discriminator)

# One integration through the scalar loop, or in the vector loop through `_step_vector_loop`.
@inline function _step_satellite(
    tracking_loop::VectorTrackingLoop,
    loop_state::VectorLoopState,
    post_correlation_state::PostCorrelationState,
    nco_history,
    application_sample_index::Int64,
)
    replica_halves = _replica_halves(post_correlation_state, nco_history)
    if !loop_state.vector_controlled
        scalar_state, carrier_doppler, code_doppler =
            step_loop!(tracking_loop.scalar_loop, loop_state.scalar_state, post_correlation_state, nco_history, application_sample_index)
        return VectorLoopState(loop_state; scalar_state, last_replica_tail = replica_halves.second),
        carrier_doppler,
        code_doppler
    end
    _step_vector_loop(loop_state, post_correlation_state, nco_history, application_sample_index, replica_halves, tracking_loop.navigation_processor.min_tracking_cn0)
end

# One integration of a member: its discriminators measured as the scalar loops measure
# them, the FLL since the residual PLL's previous prompt; the DLL accumulated, the carrier
# measured (`_measure_carrier`), and the next command: the ramp at the middle of the integration it
# takes effect on and, carrier-aided, the code correction. The scalar loop and its staging
# stand still; a release re-initialises them.
@inline function _step_vector_loop(
    loop_state::VectorLoopState,
    post_correlation_state::PostCorrelationState,
    nco_history,
    application_sample_index::Int64,
    replica_halves,
    min_tracking_cn0::typeof(1.0dBHz),
)
    scalar_state = loop_state.scalar_state
    measured = _measure_discriminators(post_correlation_state, replica_halves.command, loop_state.residual_pll.last_prompt_span; form_fll = true)
    residual_pll, fll_disc_sum =
        _measure_carrier(loop_state, post_correlation_state, nco_history, replica_halves, measured.carrier_frequency_error, min_tracking_cn0)
    # The command takes effect on an integration taken to be this one's length.
    carrier =
        _doppler_at(
            loop_state.ramp,
            application_sample_index + post_correlation_state.last_num_integrated_samples / 2,
            _sampling_frequency_hz(post_correlation_state),
        )
    carrier_doppler, code_doppler = calc_carrier_aided_dopplers(
        post_correlation_state.signal,
        scalar_state.initial_carrier_doppler,
        scalar_state.initial_code_doppler,
        carrier * Hz - scalar_state.initial_carrier_doppler,
        loop_state.code_frequency_correction,
    )
    discriminators = loop_state.discriminators
    VectorLoopState(
        loop_state;
        discriminators = DiscriminatorAccumulator(
            discriminators.code_disc_sum .+ (1, measured.code_phase_error),
            fll_disc_sum,
            discriminators.code_frequency_correction_history,
            discriminators.feedback_delay,
        ),
        last_replica_tail = replica_halves.second,
        residual_pll,
    ),
    carrier_doppler,
    code_doppler
end

# A member's carrier: the FLL discriminator `fll_discriminator` accumulated, the residual PLL
# stepped on the prompt and its lock tested (`_with_lock_test`). Returns the residual PLL and
# the FLL accumulator. Both span the previous prompt's centre to this one's, including any
# gap a host left between the integrations (the replica ran on through it on the commands
# in force); the replica runs on the filter's command alone, so only the signal moves the FLL.
@inline function _measure_carrier(
    loop_state::VectorLoopState,
    post_correlation_state::PostCorrelationState,
    nco_history,
    replica_halves,
    fll_discriminator,
    min_tracking_cn0::typeof(1.0dBHz),
)
    fs = _sampling_frequency_hz(post_correlation_state)
    coherent_integration_time = post_correlation_state.last_num_integrated_samples / fs
    # The gap since the previous integration's end, known once the residual PLL has stepped
    # on its prompt.
    previous_span = loop_state.residual_pll.last_prompt_span
    num_gap_samples =
        _has_span(previous_span) ? max(0, _integration_start(post_correlation_state) - previous_span.end_sample_index) : 0
    residual_pll, wiped_prompt = _step_residual_pll(
        loop_state,
        post_correlation_state,
        _replica_cycles_after(loop_state, num_gap_samples, fs, nco_history) + replica_halves.first,
    )
    host_cn0 = estimate_cn0(post_correlation_state.cn0_estimator, coherent_integration_time * s)
    residual_pll = _with_lock_test(
        residual_pll,
        wiped_prompt,
        coherent_integration_time,
        _phase_lock_time_constant(post_correlation_state.signal, post_correlation_state.last_num_code_periods),
        min_tracking_cn0,
        host_cn0 >= min_tracking_cn0,
    )
    residual_pll, _accumulate_fll(loop_state.discriminators.fll_disc_sum, fll_discriminator, post_correlation_state)
end

