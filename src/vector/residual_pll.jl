# ─────────────────────────────────────────────────────────────────────────────
# The carrier in the vector loop: the navigation filter sets each member's carrier NCO
# to its predicted Doppler, and a residual PLL per satellite — a third-order PLL in
# software, on the scalar PLL's filter — follows the carrier on the prompts the NCO
# leaves. The residual PLL estimates; it never steers the NCO. Every quantity it
# propagates over is the NCO's exact word (`mean_nco_word`), so what the NCO is
# commanded never changes what the residual PLL estimates, as long as the residual
# stays inside the discriminator's range.
#
# The residual PLL gives the phase the navigation engine rotates the decoding signal's
# prompts by before it accumulates them into the soft bits it decodes
# (`_advance_decoding!`). The host's bit buffer reads the prompts as the NCO leaves
# them: its bit clock only counts blocks once synced, and its soft bits are not decoded
# while the vector loop holds the satellite.
# ─────────────────────────────────────────────────────────────────────────────

"""
    CarrierRamp

The carrier Doppler the navigation filter commands a member with: the
predicted Doppler `doppler` (Hz) at `time` (s, on the records' grid) and its
rate `rate` (Hz/s), from the filter's state propagated with its velocity, its
acceleration where it models one, and the clock drift. Each record is
commanded the ramp at the middle of the record it lands on.
"""
struct CarrierRamp
    doppler::Float64
    rate::Float64
    time::Float64
end

CarrierRamp() = CarrierRamp(NaN, 0.0, NaN)

@inline _doppler_at(ramp::CarrierRamp, t) = ramp.doppler + ramp.rate * (t - ramp.time)

# The ramp's mean Doppler over `[a, b]` (s): its value at the midpoint, exactly.
@inline _mean_doppler(ramp::CarrierRamp, a, b) = _doppler_at(ramp, (a + b) / 2)

"""
$(SIGNATURES)

Recommended noise bandwidth of the residual PLL that follows each satellite's
carrier in the vector loop: 10 Hz, the default of
[`VectorPLLAndDLL`](@ref)'s `residual_pll_loop_filter_bandwidth`. Capped at
filter time like the scalar PLL's, by
[`effective_carrier_loop_filter_bandwidth`](@ref).
"""
default_residual_pll_loop_filter_bandwidth() = 10.0Hz

# The residual PLL's loop filter: the scalar PLL's third-order filter without its FLL
# assist.
const ResidualPLLLoopFilter = ThirdOrderBilinearLF{typeof(1.0Hz),typeof(1.0Hz^2)}

# The residual PLL's lock (`_with_lock_test`): the phase-lock detector on the prompts the
# residual PLL rotated (`PhaseLockDetector`, as the scalar loop's staging keeps one on its
# own prompts), whether the residual PLL is in lock and for how many records in a row its
# lock test has held.
struct ResidualPLLLock
    lock_detector::PhaseLockDetector
    locked::Bool
    num_locked_records::Int
end

# A residual PLL out of lock with a fresh detector, and one taking over the scalar loop's
# detector and its lock.
ResidualPLLLock() = ResidualPLLLock(PhaseLockDetector(), false, 0)
ResidualPLLLock(staging::CarrierLoopStaging) = ResidualPLLLock(staging.lock_detector, true, 0)

"""
    ResidualPLL

The residual PLL of a satellite in the vector loop, whose carrier NCO
the navigation filter sets (see [`VectorPLLAndDLL`](@ref)). Its fields are
internal and may change in any release.

It is a PLL in software: the replica it runs is its own prediction of the
incoming carrier phase against the NCO's, which it wipes each prompt with, and
the scalar PLL's third-order loop filter without its FLL assist
(`ThirdOrderBilinearLF`) closes it at
`bandwidth`, capped like the scalar PLL's against the record's integration
time. It keeps

  - `prompt_phase`: the residual phase (cycles) it predicted at the latest
    prompt's centre and wiped that prompt with (unwrapped: it moves
    continuously, so it is not folded);
  - `carrier_freq_update`: the loop filter's output (Hz), which the
    prediction runs at against the filter's commanded ramp until the next
    prompt, as the scalar loop's carrier word runs at its own;
  - `loop_filter`: the loop filter, whose states are its estimates of the
    incoming Doppler against the ramp (`x1`, Hz) and of its rate against the
    ramp's (`x2`, Hz/s).

A new ramp from the navigation filter does not move the residual PLL's estimates of
the incoming Doppler and its rate: `carrier_freq_update` and `x1` take the step
between the old and the new ramp, and `x2` the change in its rate.

Its lock, `lock` (`_with_lock_test`), is read on the prompts it rotated, with the scalar
loop's phase-lock indicator and the C/N₀ the same prompts' moments read. While
the host's C/N₀ is below the lock threshold — no signal — the frequency and
its rate are held at the filter's ramp (zero), so the estimate does not
wander off on the noise and the phase pulls in from the ramp once the signal
is back. With the signal there, the loop runs on whether its phase is locked
or not, so it pulls in on the signal by itself.
"""
struct ResidualPLL
    bandwidth::typeof(1.0Hz)
    carrier_freq_update::Float64
    loop_filter::ResidualPLLLoopFilter
    # The latest prompt's centre (s, on the records' grid; `NaN` until the first record)
    # and the carrier phase polarity its discriminator ran with.
    prompt_center::Float64
    carrier_phase_polarity::Int8
    # The residual phase (cycles) the latest prompt was wiped with: the prediction.
    prompt_phase::Float64
    lock::ResidualPLLLock
end

# The residual PLL's loop filter at the Doppler `frequency` (Hz) and its rate `rate`
# (Hz/s) against the ramp.
_residual_pll_loop_filter(frequency, rate) = ThirdOrderBilinearLF(frequency * Hz, rate * Hz^2)

# The residual PLL's estimate of the incoming Doppler against the ramp (Hz) and of its rate
# against the ramp's (Hz/s): the loop filter's states. Its output, `carrier_freq_update`,
# adds the latest phase error's correction, which the prediction runs on but which is no
# estimate.
_residual_doppler(residual_pll) = ustrip(Hz, residual_pll.loop_filter.x1)
_residual_doppler_rate(residual_pll) = ustrip(Hz^2, residual_pll.loop_filter.x2)

ResidualPLL(bandwidth) = ResidualPLL(
    bandwidth,
    0.0,
    _residual_pll_loop_filter(0.0, 0.0),
    NaN,
    Int8(0),
    NaN,
    ResidualPLLLock(),
)

ResidualPLL(
    residual_pll::ResidualPLL;
    carrier_freq_update::Maybe{Float64} = nothing,
    loop_filter::Maybe{ResidualPLLLoopFilter} = nothing,
    prompt_center::Maybe{Float64} = nothing,
    carrier_phase_polarity::Maybe{Integer} = nothing,
    prompt_phase::Maybe{Float64} = nothing,
    lock::Maybe{ResidualPLLLock} = nothing,
) = ResidualPLL(
    residual_pll.bandwidth,
    isnothing(carrier_freq_update) ? residual_pll.carrier_freq_update : carrier_freq_update,
    isnothing(loop_filter) ? residual_pll.loop_filter : loop_filter,
    isnothing(prompt_center) ? residual_pll.prompt_center : prompt_center,
    isnothing(carrier_phase_polarity) ? residual_pll.carrier_phase_polarity :
    carrier_phase_polarity,
    isnothing(prompt_phase) ? residual_pll.prompt_phase : prompt_phase,
    isnothing(lock) ? residual_pll.lock : lock,
)

# A residual PLL taking over from the scalar loop when the satellite is added to the vector
# loop: the incoming Doppler is the scalar loop's word `scalar_doppler` (Hz), which it
# was phase-locked at, against the ramp at `time`; in lock, as the scalar loop was, its
# averages carried on from the scalar loop's `staging`.
_initialized_residual_pll(
    residual_pll::ResidualPLL,
    ramp::CarrierRamp,
    scalar_doppler,
    time,
    staging::CarrierLoopStaging,
) =
    ResidualPLL(
        residual_pll.bandwidth,
        scalar_doppler - _doppler_at(ramp, time),
        _residual_pll_loop_filter(scalar_doppler - _doppler_at(ramp, time), 0.0),
        NaN,
        Int8(0),
        NaN,
        ResidualPLLLock(staging),
    )

# The residual PLL with the filter's ramp replaced by `new`: its estimates of the
# incoming Doppler at its latest prompt (or, before its first, where the old ramp
# took over) and of its rate stay where they are.
function _residual_pll_on_new_ramp(residual_pll::ResidualPLL, old::CarrierRamp, new::CarrierRamp)
    time = isnan(residual_pll.prompt_center) ? old.time : residual_pll.prompt_center
    step = _doppler_at(new, time) - _doppler_at(old, time)
    ResidualPLL(
        residual_pll;
        carrier_freq_update = residual_pll.carrier_freq_update - step,
        loop_filter = _residual_pll_loop_filter(
            _residual_doppler(residual_pll) - step,
            _residual_doppler_rate(residual_pll) - (new.rate - old.rate),
        ),
    )
end

# The residual PLL with its lock tested after a record of `integration_time` (s), whose
# prompt the residual PLL wiped to `wiped`, over `time_constant` records: the phase-lock
# indicator its averages read is at least the scalar loop's threshold, and the C/N₀ the same
# prompt moments read is at most `RESIDUAL_PLL_CN0_MARGIN` below `lock_cn0_threshold` — on
# noise alone the indicator's signal power, which the noise cancels from only on average,
# comes out near zero now and then, and the ratio with it reads lock. A residual PLL in lock
# leaves it on the first record that fails the test; one out of lock is in lock again once
# the test has held for one time constant of the averages in a row (0.1 s).
#
# Without `signal_present` — the host's C/N₀ below the lock threshold — its frequency and
# rate are held at the ramp. The host's estimate decides that, not the residual PLL's: on
# bare noise the moments of 0.1 s of 1 ms prompts read up to the threshold, half the time
# within `RESIDUAL_PLL_CN0_MARGIN` of it, where the default noise-referenced estimator stays
# more than 5 dB below it (in the pipeline simulation).
const RESIDUAL_PLL_CN0_MARGIN = 6.0 # dB
function _with_lock_test(
    residual_pll::ResidualPLL,
    wiped,
    integration_time,
    time_constant,
    lock_cn0_threshold_dbhz,
    signal_present::Bool,
)
    lock = residual_pll.lock
    lock_detector = _advance_lock_detector(lock.lock_detector, wiped, time_constant)
    signal_power = _signal_power(lock_detector)
    noise_power = max(lock_detector.power - signal_power, eps(lock_detector.power))
    cn0_dbhz = 10 * log10(max(signal_power / (noise_power * integration_time), 1e-3))
    good = lock_detector.indicator >= _PHASE_LOCK_THRESHOLD && cn0_dbhz >= lock_cn0_threshold_dbhz - RESIDUAL_PLL_CN0_MARGIN
    num_locked_records = good ? lock.num_locked_records + 1 : 0
    locked = lock.locked ? good : num_locked_records >= time_constant
    lock = ResidualPLLLock(lock_detector, locked, num_locked_records)
    signal_present ? ResidualPLL(residual_pll; lock) : _reset_to_ramp(residual_pll, lock)
end

# The residual PLL with its lock replaced by `lock` and its residual Doppler and rate held
# at the filter's ramp.
_reset_to_ramp(residual_pll::ResidualPLL, lock::ResidualPLLLock) =
    ResidualPLL(residual_pll; carrier_freq_update = 0.0, loop_filter = _residual_pll_loop_filter(0.0, 0.0), lock)

# The residual phase (cycles) the residual PLL predicts at `center` (s), after its
# latest prompt, where the replica advanced by `replica_cycles` from that prompt's
# centre: the incoming phase advances by the ramp plus the loop filter's output.
@inline function _predicted_phase(residual_pll::ResidualPLL, ramp::CarrierRamp, center, replica_cycles)
    dt = center - residual_pll.prompt_center
    residual_pll.prompt_phase +
    (_mean_doppler(ramp, residual_pll.prompt_center, center) + residual_pll.carrier_freq_update) * dt -
    replica_cycles
end

# One record through the residual PLL: predict the residual phase to the record's prompt,
# wipe the prompt with it, read the discriminator and run the loop filter on it, whose
# output the prediction runs at until the next prompt, as an NCO runs at the scalar
# loop's. Returns the residual PLL and the wiped prompt.
#
# `replica_cycles` is the phase (cycles) the replica advanced by from the previous
# prompt's centre to this one's, from the words it ran on.
function _step_residual_pll(residual_pll::ResidualPLL, ramp::CarrierRamp, record::LoopRecord, replica_cycles)
    fs = _sampling_frequency_hz(record)
    num_samples = record.integrated_samples
    integration_time = num_samples / fs
    center = (record.sample_index - num_samples / 2) / fs
    polarity = record.carrier_phase_polarity

    # The scalar loop hands over phase-locked: the first prompt's phase is its
    # residual, read as is.
    first_record = isnan(residual_pll.prompt_center) || isnan(replica_cycles)
    predicted_phase = first_record ? 0.0 : _predicted_phase(residual_pll, ramp, center, replica_cycles)

    wiped = _rotated(record.filtered_correlator, cis(-2π * predicted_phase))
    phase_error = pll_disc(record.signal, wiped; polarity)
    polarity_changed = polarity != residual_pll.carrier_phase_polarity
    if polarity_changed && !iszero(polarity) && abs(phase_error) > 0.25
        # A four-quadrant discriminator taking over from a two-quadrant one: the
        # residual PLL may sit half a cycle off the polarity's frame. Move it there.
        predicted_phase += 0.5
        wiped = _rotated(wiped, -1)
        phase_error = pll_disc(record.signal, wiped; polarity)
    end

    bandwidth = effective_carrier_loop_filter_bandwidth(residual_pll.bandwidth, integration_time * s)
    carrier_freq_update, loop_filter = filter_loop(residual_pll.loop_filter, phase_error, integration_time * s, bandwidth)
    ResidualPLL(
        residual_pll;
        carrier_freq_update = ustrip(Hz, carrier_freq_update),
        loop_filter,
        prompt_center = center,
        carrier_phase_polarity = polarity,
        prompt_phase = predicted_phase,
    ),
    ComplexF64(get_prompt(wiped))
end

# `correlator` with every tap rotated by `phasor`.
_rotated(correlator, phasor) = update_accumulator(correlator, get_accumulators(correlator) .* phasor)

# The residual phase (cycles) the residual PLL predicted at the driver's latest prompts,
# and their centres (s): a ring with the newest at `newest` and `count` of them valid,
# kept by the satellite's slot. A pilot + data pair's decoding signal reads the phase at
# its own records from it (`_symbol_prompt_phase`), as a host may step a satellite's
# signals one after the other, the data's records after the pilot's of the same span.
# It holds `_prompt_history_length` prompts, allocated once; the engine records every
# prompt the residual PLL steps on and clears it while the filter does not command the
# satellite, so it never mixes the prompts of two residual PLLs.
mutable struct PromptHistory
    const centers::Vector{Float64}
    const phases::Vector{Float64}
    newest::Int
    count::Int
end

PromptHistory(length::Int) = PromptHistory(fill(NaN, length), fill(NaN, length), 0, 0)

# How many prompts a slot's history holds for a group ranging on `driver` and decoding
# `decoding`: half a navigation cycle (`vt_update_interval`) of the driver's shortest
# records, one code period each — a host steps every signal at least once per half cycle,
# so no signal's records trail another's by more — and one more for the record that
# straddles it, counted in chips as `_phase_lock_time_constant` counts. A plain signal
# decodes its own records and keeps none.
function _prompt_history_length(driver::AbstractGNSSSignal, decoding::AbstractGNSSSignal, vt_update_interval)
    typeof(driver) == typeof(decoding) && return 0
    chips = round(Int, get_code_frequency(driver) * vt_update_interval / 2)
    cld(chips, get_code_length(driver)) + 1
end

# Record the residual PLL's latest prompt, and forget them all.
function _push_prompt!(history::PromptHistory, residual_pll::ResidualPLL)
    n = Base.length(history.centers)
    n == 0 && return history
    history.newest = mod1(history.newest + 1, n)
    history.centers[history.newest] = residual_pll.prompt_center
    history.phases[history.newest] = residual_pll.prompt_phase
    history.count = min(history.count + 1, n)
    history
end
_clear_prompts!(history::PromptHistory) = (history.count = 0; history)

# The residual phase (cycles) of the prompt of a decoding signal's record, as the
# residual PLL estimates it. For a plain signal, whose driver record it is, the phase the
# residual PLL wiped it with: its prediction, which that prompt did not correct. For a
# passenger record at or before the latest prompt the residual PLL stepped on — a host
# may step the driver's records first — the phase it predicted at the nearest prompt of
# `history`, moved on at its loop filter's output. For one after it, its prediction: the
# replica advancing by the second half of the driver's last record
# (`previous_replica_half`) and by the words it ran on from there. `NaN` before the
# residual PLL's first prompt, and for a record more than a driver record older than
# the oldest prompt `history` holds: rather than a phase moved back over records the
# loop filter's output did not hold for.
function _symbol_prompt_phase(
    residual_pll::ResidualPLL,
    ramp::CarrierRamp,
    previous_replica_half,
    history::PromptHistory,
    record::LoopRecord,
    words,
    driver::Bool,
)
    isnan(residual_pll.prompt_center) && return NaN
    driver && return residual_pll.prompt_phase
    fs = _sampling_frequency_hz(record)
    center = (record.sample_index - record.integrated_samples / 2) / fs
    if center > residual_pll.prompt_center + 0.5 / fs
        previous_cycles, previous_duration = previous_replica_half
        isnan(previous_cycles) && return NaN
        previous_end = residual_pll.prompt_center + Float64(ustrip(s, previous_duration))
        word, _ = mean_nco_word(words, previous_end * fs, center * fs)
        return _predicted_phase(residual_pll, ramp, center, previous_cycles + word * (center - previous_end))
    end
    history.count == 0 && return NaN
    n = Base.length(history.centers)
    nearest = history.newest
    for k in 1:history.count
        i = mod1(history.newest - k + 1, n)
        abs(history.centers[i] - center) < abs(history.centers[nearest] - center) && (nearest = i)
    end
    oldest = history.centers[mod1(history.newest - history.count + 1, n)]
    driver_duration = 2 * Float64(ustrip(s, last(previous_replica_half)))
    oldest - center > max(driver_duration, record.integrated_samples / fs) && return NaN
    history.phases[nearest] + residual_pll.carrier_freq_update * (center - history.centers[nearest])
end
