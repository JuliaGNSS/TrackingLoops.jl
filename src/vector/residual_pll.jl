# The carrier in the vector loop (see the manual's "The carrier in the vector loop"):
# the filter's carrier ramp and each member's residual PLL. The residual PLL propagates
# over the NCO's exact words (`mean_nco_word`), so what the NCO is commanded never
# changes its estimates while the residual stays inside the discriminator's range. Its
# phase rotates the decoding signal's prompts for the engine's soft bits
# (`_advance_decoding!`).

"""
    CarrierRamp

The carrier Doppler the navigation filter commands a member with: `doppler` (Hz) at
`time` (s, records' grid) and its `rate` (Hz/s). Each record is commanded the ramp at
the middle of the record it lands on.
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

Default noise bandwidth of the residual PLL that follows each satellite's carrier in
the vector loop (10 Hz, [`VectorPLLAndDLL`](@ref)'s
`residual_pll_loop_filter_bandwidth`), capped like the scalar PLL's by
[`effective_carrier_loop_filter_bandwidth`](@ref).
"""
default_residual_pll_loop_filter_bandwidth() = 10.0Hz

# The residual PLL's loop filter: the scalar PLL's third-order filter without its FLL
# assist.
const ResidualPLLLoopFilter = ThirdOrderBilinearLF{typeof(1.0Hz),typeof(1.0Hz^2)}

# The residual PLL's lock (`_with_lock_test`): a `PhaseLockDetector` on the prompts it
# rotated, whether it is in lock, and for how many records in a row the test has held.
struct ResidualPLLLock
    lock_detector::PhaseLockDetector
    locked::Bool
    num_locked_records::Int
end

# Out of lock with a fresh detector; or taking over the scalar loop's detector, locked.
ResidualPLLLock() = ResidualPLLLock(PhaseLockDetector(), false, 0)
ResidualPLLLock(staging::CarrierLoopStaging) = ResidualPLLLock(staging.lock_detector, true, 0)

"""
    ResidualPLL

The residual PLL of a satellite in the vector loop (see [`VectorPLLAndDLL`](@ref)).
Its fields are internal.

A PLL in software: it wipes each prompt with its own prediction of the incoming
carrier phase against the NCO's, and the scalar PLL's third-order loop filter without
FLL assist (`ThirdOrderBilinearLF`) closes it at `bandwidth`, capped like the scalar
PLL's. It keeps

  - `prompt_phase`: the residual phase (cycles, unwrapped) it predicted at the latest
    prompt's centre and wiped that prompt with;
  - `carrier_freq_update`: the loop filter's output (Hz), which the prediction runs at
    against the ramp until the next prompt;
  - `loop_filter`: whose states estimate the incoming Doppler against the ramp (`x1`,
    Hz) and its rate against the ramp's (`x2`, Hz/s). A new ramp leaves these
    estimates where they are (`_residual_pll_on_new_ramp`).

Its `lock` is tested on the prompts it rotated (`_with_lock_test`), which also holds
it at the ramp while the host's C/N₀ says there is no signal.
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

# The residual PLL's estimates of the incoming Doppler (Hz) and its rate (Hz/s) against
# the ramp: the loop filter's states, not `carrier_freq_update`, which adds the latest
# phase error's correction.
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

# A residual PLL taking over from the phase-locked scalar loop as the satellite is added:
# at its word `scalar_doppler` (Hz) against the ramp at `time`, in lock, with the
# averages of the scalar loop's `staging`.
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

# The residual PLL against the ramp `new` in place of `old`, its estimates of the
# incoming Doppler at its latest prompt (before the first: where the old ramp took
# over) and of its rate unmoved.
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

# The residual PLL with its lock tested on the prompt it wiped to `wiped`, a record of
# `integration_time` (s), averaged over `time_constant` records: the phase-lock
# indicator at least the scalar loop's threshold and the C/N₀ of the same moments at most
# `RESIDUAL_PLL_CN0_MARGIN` below `lock_cn0_threshold` (on noise alone the indicator's
# signal power comes out near zero now and then, and the ratio reads lock). It drops
# lock on the first failing record and regains it after `time_constant` (0.1 s, at
# least 25 records) consecutive passing ones.
#
# Without `signal_present` (the host's C/N₀ below the threshold) its frequency and rate
# are held at the ramp. The host's estimate decides that because on bare noise these
# moments read up to the threshold, half the time within the margin, where the default
# noise-referenced estimator stays more than 5 dB below (pipeline simulation).
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

# With `lock`, and the residual Doppler and rate held at the ramp.
_reset_to_ramp(residual_pll::ResidualPLL, lock::ResidualPLLLock) =
    ResidualPLL(residual_pll; carrier_freq_update = 0.0, loop_filter = _residual_pll_loop_filter(0.0, 0.0), lock)

# The residual phase (cycles) predicted at `center` (s), after the latest prompt: the
# incoming phase advances at the ramp plus the loop filter's output, the replica by
# `replica_cycles` from that prompt's centre.
@inline function _predicted_phase(residual_pll::ResidualPLL, ramp::CarrierRamp, center, replica_cycles)
    dt = center - residual_pll.prompt_center
    residual_pll.prompt_phase +
    (_mean_doppler(ramp, residual_pll.prompt_center, center) + residual_pll.carrier_freq_update) * dt -
    replica_cycles
end

# One record through the residual PLL: predict the phase to the prompt, wipe it, and run
# the loop filter on the discriminator. `replica_cycles` is the replica's advance from the
# previous prompt's centre to this one's. Returns the residual PLL and the wiped prompt.
function _step_residual_pll(residual_pll::ResidualPLL, ramp::CarrierRamp, record::LoopRecord, replica_cycles)
    fs = _sampling_frequency_hz(record)
    num_samples = record.integrated_samples
    integration_time = num_samples / fs
    center = (record.sample_index - num_samples / 2) / fs
    polarity = record.carrier_phase_polarity

    # The scalar loop hands over phase-locked: the first prompt is read as is.
    first_record = isnan(residual_pll.prompt_center) || isnan(replica_cycles)
    predicted_phase = first_record ? 0.0 : _predicted_phase(residual_pll, ramp, center, replica_cycles)

    wiped = _rotated(record.filtered_correlator, cis(-2π * predicted_phase))
    phase_error = pll_disc(record.signal, wiped; polarity)
    polarity_changed = polarity != residual_pll.carrier_phase_polarity
    if polarity_changed && !iszero(polarity) && abs(phase_error) > 0.25
        # A four-quadrant discriminator taking over from a two-quadrant one may find the
        # residual PLL half a cycle off the polarity's frame.
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

# A slot's ring of the residual phases (cycles) predicted at the driver's latest prompts
# and their centres (s), newest at `newest`, `count` valid. A pair's decoding signal reads
# its records' phase from it (`_symbol_prompt_phase`), since a host may step the data's
# records after the pilot's of the same span. Allocated once at `_prompt_history_length`;
# cleared while the filter does not command the satellite, so it never mixes two
# residual PLLs' prompts.
mutable struct PromptHistory
    const centers::Vector{Float64}
    const phases::Vector{Float64}
    newest::Int
    count::Int
end

PromptHistory(length::Int) = PromptHistory(fill(NaN, length), fill(NaN, length), 0, 0)

# A slot's history length for a group ranging on `driver` and decoding `decoding`: half
# a `vt_update_interval` of the driver's shortest (one-code-period) records, as the host
# steps every signal at least once per half cycle, plus one for the straddling record,
# counted in chips as `_phase_lock_time_constant` counts. None for a plain signal.
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

# The residual PLL's phase (cycles) at a decoding signal's prompt. A plain signal's
# (driver) record: the phase it was wiped with, a prediction that prompt did not correct.
# A passenger record at or before the latest prompt: the nearest prompt's phase in
# `history`, moved on at the loop filter's output. One after it: the prediction, the
# replica advancing by `previous_replica_half` and the words from there. `NaN` before
# the first prompt, and for a record more than a driver record older than the oldest
# prompt held, rather than a phase moved back over records that output did not hold for.
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
