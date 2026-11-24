# The carrier in the vector loop (see the manual's "The carrier in the vector loop"):
# the filter's carrier ramp and each member's residual PLL. The residual PLL propagates
# over the NCO's exact commands (`calc_mean_nco_command`), so what the NCO is commanded never
# changes its estimates while the residual stays inside the discriminator's range. Its
# phase rotates the decoding signal's prompts for the navigation processor's soft bits
# (`_advance_decoding!`).

"""
    CarrierDopplerRamp

The carrier Doppler the navigation filter commands a member with: `doppler` (Hz) from
`start_sample_index` on, the sample of the integrations' grid where the command takes
effect, and its `doppler_rate` (Hz/s). Each integration is commanded the ramp at the
middle of the integration it takes effect on.
"""
struct CarrierDopplerRamp
    doppler::typeof(1.0Hz)
    doppler_rate::typeof(1.0Hz / s)
    start_sample_index::Int
end

CarrierDopplerRamp() = CarrierDopplerRamp(NaN * Hz, 0.0Hz / s, typemin(Int))

# The ramp's Doppler (Hz) at the sample `sample_index` of the integrations' grid sampled at
# `fs` (Hz), unit-free as the residual PLL's states it is added to. The time since the
# start is a difference of sample indices, exact however long a session runs.
@inline _doppler_at(ramp::CarrierDopplerRamp, sample_index, fs) =
    ustrip(Hz, ramp.doppler) + ustrip(Hz / s, ramp.doppler_rate) * ((sample_index - ramp.start_sample_index) / fs)

"""
$(SIGNATURES)

Default noise bandwidth of the residual PLL that follows each satellite's carrier in
the vector loop (10 Hz, [`VectorTrackingLoop`](@ref)'s
`residual_pll_loop_bandwidth`), capped like the scalar PLL's by
[`calc_effective_carrier_loop_bandwidth`](@ref).
"""
default_residual_pll_loop_bandwidth() = 10.0Hz

# The residual PLL's loop filter: the scalar PLL's third-order filter without its FLL
# assist.
const ResidualPLLLoopFilter = ThirdOrderBilinearLF{typeof(1.0Hz),typeof(1.0Hz^2),typeof(1.0s)}

# The residual PLL's lock (`_with_lock_test`): a `PhaseLockDetector` on the prompts it
# rotated, whether it is in lock, and for how many integrations in a row the test has held.
struct ResidualPLLLock
    lock_detector::PhaseLockDetector
    locked::Bool
    num_locked_integrations::Int
end

# Out of lock with a fresh detector; or taking over the scalar loop's detector, locked.
ResidualPLLLock() = ResidualPLLLock(PhaseLockDetector(), false, 0)
ResidualPLLLock(staging::CarrierLoopStaging) = ResidualPLLLock(staging.lock_detector, true, 0)

"""
    ResidualPLL

The residual PLL of a satellite in the vector loop (see [`VectorTrackingLoop`](@ref)).
Its fields are internal.

A PLL in software: it wipes each prompt with its own prediction of the incoming
carrier phase against the NCO's, and the scalar PLL's third-order loop filter without
FLL assist (`ThirdOrderBilinearLF`) closes it at `bandwidth`, capped like the scalar
PLL's. It keeps

  - `predicted_prompt_phase`: the residual phase (cycles, unwrapped) it predicted at the latest
    prompt's centre and wiped that prompt with;
  - `carrier_frequency_correction`: the loop filter's output (Hz), which the prediction runs at
    against the ramp until the next prompt;
  - `loop_filter`: which estimates the incoming Doppler against the ramp
    (`predict_frequency`, Hz) and its rate against the ramp's (`get_frequency_rate`,
    Hz/s). A new ramp leaves these estimates where they are (`_residual_pll_on_new_ramp`).

Its `lock` is tested on the prompts it rotated (`_with_lock_test`), which also holds
it at the ramp while the host's C/N₀ says there is no signal.
"""
struct ResidualPLL
    bandwidth::typeof(1.0Hz)
    carrier_frequency_correction::typeof(1.0Hz)
    loop_filter::ResidualPLLLoopFilter
    # The latest prompt's integration on the integrations' grid (none until the first) and
    # the carrier phase polarity its discriminator ran with.
    last_prompt_span::IntegrationSpan
    carrier_phase_polarity::Int8
    # The residual phase (cycles) the latest prompt was wiped with: the prediction.
    predicted_prompt_phase::Float64
    lock::ResidualPLLLock
end

ResidualPLL(bandwidth) = ResidualPLL(
    bandwidth,
    0.0Hz,
    reinit(ResidualPLLLoopFilter),
    IntegrationSpan(),
    Int8(0),
    NaN,
    ResidualPLLLock(),
)

ResidualPLL(
    residual_pll::ResidualPLL;
    carrier_frequency_correction::Maybe{typeof(1.0Hz)} = nothing,
    loop_filter::Maybe{ResidualPLLLoopFilter} = nothing,
    last_prompt_span::Maybe{IntegrationSpan} = nothing,
    carrier_phase_polarity::Maybe{Integer} = nothing,
    predicted_prompt_phase::Maybe{Float64} = nothing,
    lock::Maybe{ResidualPLLLock} = nothing,
) = ResidualPLL(
    residual_pll.bandwidth,
    isnothing(carrier_frequency_correction) ? residual_pll.carrier_frequency_correction : carrier_frequency_correction,
    isnothing(loop_filter) ? residual_pll.loop_filter : loop_filter,
    isnothing(last_prompt_span) ? residual_pll.last_prompt_span : last_prompt_span,
    isnothing(carrier_phase_polarity) ? residual_pll.carrier_phase_polarity :
    carrier_phase_polarity,
    isnothing(predicted_prompt_phase) ? residual_pll.predicted_prompt_phase : predicted_prompt_phase,
    isnothing(lock) ? residual_pll.lock : lock,
)

# A residual PLL taking over from the phase-locked scalar loop as the satellite is added:
# at its command `scalar_doppler` (Hz) against the ramp where it starts, in lock, with the
# averages of the scalar loop's `staging`.
function _initialized_residual_pll(
    residual_pll::ResidualPLL,
    ramp::CarrierDopplerRamp,
    scalar_doppler,
    staging::CarrierLoopStaging,
)
    frequency = (scalar_doppler - ustrip(Hz, ramp.doppler)) * Hz
    ResidualPLL(
        residual_pll.bandwidth,
        frequency,
        reinit(ResidualPLLLoopFilter; frequency),
        IntegrationSpan(),
        Int8(0),
        NaN,
        ResidualPLLLock(staging),
    )
end

# The residual PLL against the ramp `new` in place of `old`, its estimates of the
# incoming Doppler and of its rate unmoved; `fs` (Hz) dates the prompts. The loop filter
# restarts at its estimate at the end of its latest prompt's integration (before the
# first: at the start of its first, where the old ramp took over), so it goes on
# integrating from there; the filter's output is moved by the step at that prompt's centre.
function _residual_pll_on_new_ramp(residual_pll::ResidualPLL, old::CarrierDopplerRamp, new::CarrierDopplerRamp, fs)
    span = residual_pll.last_prompt_span
    step_at(sample_index) = _doppler_at(new, sample_index, fs) - _doppler_at(old, sample_index, fs)
    filter_sample_index = _has_span(span) ? span.end_sample_index : old.start_sample_index
    prompt_sample_index = _has_span(span) ? _center(span) : old.start_sample_index
    loop_filter = residual_pll.loop_filter
    ResidualPLL(
        residual_pll;
        carrier_frequency_correction = residual_pll.carrier_frequency_correction - step_at(prompt_sample_index) * Hz,
        loop_filter = reinit(
            loop_filter;
            frequency = predict_frequency(loop_filter, 0.0s) - step_at(filter_sample_index) * Hz,
            frequency_rate = get_frequency_rate(loop_filter) - (new.doppler_rate - old.doppler_rate),
        ),
    )
end

# The residual PLL with its lock tested on the prompt it wiped to `wiped`, an integration of
# `coherent_integration_time` (s), averaged over `time_constant` integrations: the phase-lock
# indicator at least the scalar loop's threshold and the C/N₀ of the same moments at most
# `RESIDUAL_PLL_CN0_MARGIN` (6 dB) below `min_tracking_cn0` (on noise alone the indicator's
# signal power comes out near zero now and then, and the ratio reads lock). It drops
# lock on the first failing integration and regains it after `time_constant` (0.1 s, at
# least 25 integrations) consecutive passing ones.
#
# Without `signal_present` (the host's C/N₀ below the threshold) its frequency and rate
# are held at the ramp. The host's estimate decides that because on bare noise these
# moments read up to the threshold, half the time within the margin, where the default
# noise-referenced C/N₀ estimator stays more than 5 dB below (pipeline simulation).
const RESIDUAL_PLL_CN0_MARGIN = 10^(-6 / 10) # 6 dB, a factor on the linear C/N₀
function _with_lock_test(
    residual_pll::ResidualPLL,
    wiped,
    coherent_integration_time,
    time_constant,
    min_tracking_cn0,
    signal_present::Bool,
)
    lock = residual_pll.lock
    lock_detector = _advance_lock_detector(lock.lock_detector, wiped, time_constant)
    signal_power = _signal_power(lock_detector)
    noise_power = max(lock_detector.power - signal_power, eps(lock_detector.power))
    cn0 = signal_power / (noise_power * coherent_integration_time) # linear, Hz
    good = lock_detector.indicator >= _PHASE_LOCK_THRESHOLD &&
        cn0 >= ustrip(Hz, min_tracking_cn0) * RESIDUAL_PLL_CN0_MARGIN
    num_locked_integrations = good ? lock.num_locked_integrations + 1 : 0
    locked = lock.locked ? good : num_locked_integrations >= time_constant
    lock = ResidualPLLLock(lock_detector, locked, num_locked_integrations)
    signal_present ? ResidualPLL(residual_pll; lock) : _reset_to_ramp(residual_pll, lock)
end

# With `lock`, and the residual Doppler and rate held at the ramp.
_reset_to_ramp(residual_pll::ResidualPLL, lock::ResidualPLLLock) =
    ResidualPLL(residual_pll; carrier_frequency_correction = 0.0Hz, loop_filter = reinit(residual_pll.loop_filter), lock)

# The residual phase (cycles) predicted `samples_since_prompt` samples (at `fs`, Hz) after the
# latest prompt's centre: the incoming phase advances at the ramp plus the loop filter's
# output, the replica by `replica_cycles` from that prompt's centre.
@inline function _predicted_phase(residual_pll::ResidualPLL, ramp::CarrierDopplerRamp, samples_since_prompt, fs, replica_cycles)
    dt = samples_since_prompt / fs
    # The ramp's mean Doppler since that prompt is its value halfway there.
    mean_doppler = _doppler_at(ramp, _center(residual_pll.last_prompt_span) + samples_since_prompt / 2, fs)
    residual_pll.predicted_prompt_phase + (mean_doppler + ustrip(Hz, residual_pll.carrier_frequency_correction)) * dt - replica_cycles
end

# The residual phase (cycles) of a `VectorLoopState`'s residual PLL predicted
# `samples_since_end` samples (at `fs`, Hz) after the end of the latest prompt's integration:
# the replica advancing by the second half of that integration (`last_replica_tail`),
# then by the commands it ran on from its end. `NaN` while that half is unknown. Exact from
# the integration's end on; within the half it takes the later commands to have run there.
@inline function _residual_phase_after(loop_state, samples_since_end, fs, nco_history)
    (; residual_pll, ramp) = loop_state
    isnan(loop_state.last_replica_tail) && return NaN
    _predicted_phase(
        residual_pll,
        ramp,
        samples_since_end + residual_pll.last_prompt_span.num_samples / 2,
        fs,
        _replica_cycles_after(loop_state, samples_since_end, fs, nco_history),
    )
end

# The cycles of the replica's carrier from the middle of the latest prompt's integration to
# `samples_since_end` samples (at `fs`, Hz) after its end: its second half
# (`last_replica_tail`), then the commands it ran on from the end, through any gap.
@inline function _replica_cycles_after(loop_state, samples_since_end, fs, nco_history)
    samples_since_end == 0 && return loop_state.last_replica_tail
    end_sample_index = loop_state.residual_pll.last_prompt_span.end_sample_index
    command = calc_mean_nco_command(nco_history, end_sample_index, end_sample_index + samples_since_end).carrier_doppler
    loop_state.last_replica_tail + ustrip(Hz, command) * (samples_since_end / fs)
end

# One integration through a `VectorLoopState`'s residual PLL: predict the phase to the
# prompt, wipe it, and run the loop filter on the discriminator. `replica_cycles` is the
# replica's advance from the previous prompt's centre to this one's. Returns the residual
# PLL and the wiped prompt.
function _step_residual_pll(loop_state, post_correlation_state::PostCorrelationState, replica_cycles)
    (; residual_pll, ramp) = loop_state
    fs = _sampling_frequency_hz(post_correlation_state)
    coherent_integration_time = get_last_coherent_integration_time(post_correlation_state)
    span = _integration_span(post_correlation_state)
    polarity = post_correlation_state.last_carrier_phase_polarity

    # The scalar loop hands over phase-locked: the first prompt is read as is.
    first_integration = !_has_span(residual_pll.last_prompt_span) || isnan(replica_cycles)
    predicted_phase = first_integration ? 0.0 :
        _predicted_phase(residual_pll, ramp, _samples_since_center(residual_pll.last_prompt_span, span), fs, replica_cycles)

    wiped = _rotated(post_correlation_state.last_filtered_correlator_bank, cis(-2π * predicted_phase))
    phase_error = pll_disc(wiped; polarity)
    polarity_changed = polarity != residual_pll.carrier_phase_polarity
    if polarity_changed && !iszero(polarity) && abs(phase_error) > 0.25
        # A four-quadrant discriminator taking over from a two-quadrant one may find the
        # residual PLL half a cycle off the polarity's frame.
        predicted_phase += 0.5
        wiped = _rotated(wiped, -1)
        phase_error = pll_disc(wiped; polarity)
    end

    bandwidth = calc_effective_carrier_loop_bandwidth(residual_pll.bandwidth, coherent_integration_time)
    carrier_frequency_correction, loop_filter = filter_loop(residual_pll.loop_filter, phase_error, coherent_integration_time, bandwidth)
    ResidualPLL(
        residual_pll;
        carrier_frequency_correction = uconvert(Hz, carrier_frequency_correction),
        loop_filter,
        last_prompt_span = span,
        carrier_phase_polarity = polarity,
        predicted_prompt_phase = predicted_phase,
    ),
    ComplexF64(get_prompt(wiped))
end

# `correlator_bank` with every correlator output rotated by `phasor`.
_rotated(correlator_bank, phasor) = set_correlator_outputs(correlator_bank, get_correlator_outputs(correlator_bank) .* phasor)

# A slot's ring of the residual phases (cycles) predicted at the driver's latest prompts
# and their centres (sample indices), newest at `newest`, `count` valid. A pair's decoding signal reads
# its integrations' phase from it (`_bit_prompt_phase`), since a host may step the data's
# integrations after the pilot's of the same span. Allocated once at `_prompt_history_length`;
# cleared while the filter does not command the satellite, so it never mixes two
# residual PLLs' prompts.
mutable struct PromptHistory
    const spans::Vector{IntegrationSpan}
    const phases::Vector{Float64}
    newest::Int
    count::Int
end

PromptHistory(length::Int) = PromptHistory(fill(IntegrationSpan(), length), fill(NaN, length), 0, 0)

# A slot's history length for a group ranging on `driver` and decoding `decoding`: half
# a `navigation_update_interval` of the driver's shortest (one-code-period) integrations, as the host
# steps every signal at least once per half update interval, plus one for the straddling integration,
# counted in chips as `_phase_lock_time_constant` counts. None for a plain signal.
function _prompt_history_length(driver::AbstractGNSSSignal, decoding::AbstractGNSSSignal, navigation_update_interval)
    typeof(driver) == typeof(decoding) && return 0
    chips = round(Int, get_chip_rate(driver) * navigation_update_interval / 2)
    cld(chips, get_code_length(driver)) + 1
end

# Record the residual PLL's latest prompt, and forget them all.
function _push_prompt!(history::PromptHistory, residual_pll::ResidualPLL)
    n = Base.length(history.spans)
    n == 0 && return history
    history.newest = mod1(history.newest + 1, n)
    history.spans[history.newest] = residual_pll.last_prompt_span
    history.phases[history.newest] = residual_pll.predicted_prompt_phase
    history.count = min(history.count + 1, n)
    history
end
_clear_prompts!(history::PromptHistory) = (history.count = 0; history)

# The residual PLL's phase (cycles) at a decoding signal's prompt. A plain signal's
# (driver) integration: the phase it was wiped with, a prediction that prompt did not correct.
# A passenger integration at or before the latest prompt: the nearest prompt's phase in
# `history`, moved on at the loop filter's output. One after it: the prediction, the
# replica advancing by `last_replica_tail` and the commands from there. `NaN` before
# the first prompt, and for an integration more than a driver integration older than the oldest
# prompt held, rather than a phase moved back over integrations that output did not hold for.
function _bit_prompt_phase(
    loop_state,
    history::PromptHistory,
    post_correlation_state::PostCorrelationState,
    nco_history,
    is_driver::Bool,
)
    residual_pll = loop_state.residual_pll
    latest = residual_pll.last_prompt_span
    _has_span(latest) || return NaN
    is_driver && return residual_pll.predicted_prompt_phase
    fs = _sampling_frequency_hz(post_correlation_state)
    span = _integration_span(post_correlation_state)
    _samples_since_center(latest, span) > 0.5 &&
        return _residual_phase_after(loop_state, _samples_since_center(latest, span) - latest.num_samples / 2, fs, nco_history)
    history.count == 0 && return NaN
    n = Base.length(history.spans)
    nearest = history.newest
    for k in 1:history.count
        i = mod1(history.newest - k + 1, n)
        abs(_samples_since_center(history.spans[i], span)) < abs(_samples_since_center(history.spans[nearest], span)) && (nearest = i)
    end
    oldest = history.spans[mod1(history.newest - history.count + 1, n)]
    -_samples_since_center(oldest, span) > max(latest.num_samples, span.num_samples) && return NaN
    history.phases[nearest] + ustrip(Hz, residual_pll.carrier_frequency_correction) * _samples_since_center(history.spans[nearest], span) / fs
end
