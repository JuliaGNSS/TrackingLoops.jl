# ─────────────────────────────────────────────────────────────────────────────
# Signal combining: passengers (every signal of a satellite but the driver) mix their
# discriminators into the driver's before its loop filters read them. The host folds
# each passenger record with `fold_passenger_record` in sample order; the driver's
# next `step_loop` closes on the weighted means and resets the sums. See "Signal
# combining" in the manual.
# ─────────────────────────────────────────────────────────────────────────────

# One loop's weighted sum of passenger discriminators and its summed weights
# (weights: see `_discriminator_weight`).
struct WeightedSum{S,W}
    sum::S
    weight::W
end

@inline _accumulated(ws::WeightedSum, reading, weight) =
    WeightedSum(ws.sum + weight * reading, ws.weight + weight)

"""
    SignalCombiningSums()

The passengers' weighted discriminators pending for the driver's next record, one
weighted sum per loop, held per satellite by an estimator that combines signals:

  - `pll`: PLL discriminators in cycles, weighted in s;
  - `fll`: FLL discriminators in Hz, weighted in s³;
  - `dll`: DLL discriminators referred to the driver's code phase, in chips,
    weighted in s;
  - `first_end_sample`: the end sample of the earliest pending record
    (`typemax(Int)` with none). A passenger enters only the driver record it ends
    in, so if that is at or before the start of the driver record that steps the
    sums, they belong to one that never came (the host dropped the driver's
    in-flight integration, e.g. at a code-phase snap) and are dropped.
"""
struct SignalCombiningSums
    pll::WeightedSum{typeof(1.0s),typeof(1.0s)}
    fll::WeightedSum{typeof(1.0Hz * 1.0s^3),typeof(1.0s^3)}
    dll::WeightedSum{typeof(1.0s),typeof(1.0s)}
    first_end_sample::Int
end

SignalCombiningSums() = SignalCombiningSums(
    WeightedSum(0.0s, 0.0s),
    WeightedSum(0.0Hz * 0.0s^3, 0.0s^3),
    WeightedSum(0.0s, 0.0s),
    typemax(Int),
)

# `pending`, or none if stale for the driver `record` (see `SignalCombiningSums`).
@inline _pending_for(pending::SignalCombiningSums, record) =
    pending.first_end_sample <= record.sample_index - record.integrated_samples ?
    SignalCombiningSums() : pending

# The loops passengers are combined into.
const _ALL_LOOPS = (pll = true, fll = true, dll = true)
const _PLL_ONLY = (pll = true, fll = false, dll = false)

# Weighted mean of the driver's discriminator and the passengers' sum; exactly the
# driver's own with no passenger weight (`(w · d) / w` is not `d` bit for bit).
@inline _weighted_mean(own, own_weight, pending::WeightedSum) =
    iszero(pending.weight) ? own :
    (own_weight * own + pending.sum) / (own_weight + pending.weight)

# The weighted mean only while the driver's reading lies within the two-quadrant
# `range`: there a four-quadrant driver reads the error as a two-quadrant passenger
# does; beyond it the passenger would fold it by half a cycle.
@inline _gated_mean(own, own_weight, pending::WeightedSum, range) =
    abs(own) < range ? _weighted_mean(own, own_weight, pending) : own

# Two-quadrant ranges: ±1/4 cycle for the PLL's `atan(Q / I)`, ±1/(4T) for the
# FLL's `atan(cross / dot)`.
const _TWO_QUADRANT_PLL_RANGE = 0.25
@inline _two_quadrant_fll_range(integration_time) = uconvert(Hz, 1 / (4 * integration_time))

# A record's discriminator weight: its signal's ICD power share times its
# integration time, cubed for the FLL, whose noise variance falls with the cube.
@inline _discriminator_weight(signal::AbstractGNSSSignal, integration_time) =
    get_relative_power(signal) * uconvert(s, integration_time)
@inline _fll_discriminator_weight(signal::AbstractGNSSSignal, integration_time) =
    get_relative_power(signal) * uconvert(s, integration_time)^3

# A passenger's DLL reading (chips), normalised with the code word the replica ran
# on and referred to the driver's code phase.
@inline function _passenger_dll_reading(record, words, differential_group_delay_chips)
    record_start = record.sample_index - record.integrated_samples
    _, applied_code = mean_nco_word(words, record_start, record.sample_index)
    dll_disc(
        record.signal,
        record.filtered_correlator,
        applied_code * Hz,
        record.sampling_frequency,
    ) + differential_group_delay_chips
end

# A passenger's FLL reading (Hz). The record must have a previous prompt.
@inline _passenger_fll_reading(record, four_quadrant::Bool) = fll_disc(
    record.signal,
    record.filtered_correlator,
    record.previous_prompt,
    record.integrated_samples / record.sampling_frequency;
    four_quadrant,
)

# One passenger record's weighted discriminators added to `sums`, for the loops in
# `loops` only. The PLL is read on the driver's carrier-phase frame. PLL and FLL are
# always two-quadrant, blind to a whole-record sign flip, so data passengers and
# pre-sync records count like any other; the driver side is gated (`_gated_mean`).
# A four-quadrant passenger reading could not be gated: a Costas driver locked half
# a cycle off, or a two-quadrant FLL folding an error, still reads within range.
# The DLL counts only where `differential_group_delay_chips` is known (not `NaN`).
@inline function _add_passenger_discriminators(
    sums::SignalCombiningSums,
    record,
    words,
    loops,
    driver_signal::AbstractGNSSSignal,
    differential_group_delay_chips::Real,
)
    signal = record.signal
    correlator = record.filtered_correlator
    integration_time = record.integrated_samples / record.sampling_frequency
    weight = _discriminator_weight(signal, integration_time)
    pll_weight = loops.pll ? weight : zero(weight)
    derotation = _carrier_phase_derotation(get_carrier_phase_offset(driver_signal), signal)
    pll =
        iszero(pll_weight) ? 0.0 :
        pll_disc(
            signal,
            update_accumulator(correlator, get_accumulators(correlator) .* derotation),
        )
    fll_weight =
        loops.fll && !iszero(record.previous_prompt) ?
        _fll_discriminator_weight(signal, integration_time) : 0.0s^3
    fll = iszero(fll_weight) ? 0.0Hz : _passenger_fll_reading(record, false)
    dll_weight = loops.dll && !isnan(differential_group_delay_chips) ? weight : zero(weight)
    dll =
        iszero(dll_weight) ? 0.0 :
        _passenger_dll_reading(record, words, differential_group_delay_chips)
    SignalCombiningSums(
        _accumulated(sums.pll, pll, pll_weight),
        _accumulated(sums.fll, fll, fll_weight),
        _accumulated(sums.dll, dll, dll_weight),
        min(sums.first_end_sample, record.sample_index),
    )
end

# The driver's discriminators combined with the pending sums in `loops`; the
# identity when nothing is pending. The FLL goes into `frequency_error` and
# `raw_frequency_error` alike: only the conventional loop combines, where they are
# one reading.
@inline function _combine_discriminators(
    discriminators,
    pending::SignalCombiningSums,
    record,
    loops,
)
    pending = _pending_for(pending, record)
    integration_time = discriminators.integration_time
    weight = _discriminator_weight(record.signal, integration_time)
    phase_error =
        loops.pll ?
        _gated_mean(
            discriminators.phase_error,
            weight,
            pending.pll,
            _TWO_QUADRANT_PLL_RANGE,
        ) : discriminators.phase_error
    fll_weight =
        iszero(record.previous_prompt) ? 0.0s^3 :
        _fll_discriminator_weight(record.signal, integration_time)
    frequency_error =
        loops.fll ?
        _gated_mean(
            discriminators.frequency_error,
            fll_weight,
            pending.fll,
            _two_quadrant_fll_range(integration_time),
        ) : discriminators.frequency_error
    code_error =
        loops.dll ? _weighted_mean(discriminators.code_error, weight, pending.dll) :
        discriminators.code_error
    raw_frequency_error = loops.fll ? frequency_error : discriminators.raw_frequency_error
    merge(discriminators, (; phase_error, frequency_error, raw_frequency_error, code_error))
end

# Rotates a component's bit-buffer prompt onto the real axis, given the loops lock
# the driver there: a bit-identical no-op (`cis(0)`) for an in-phase component,
# `±90°` for a quadrature one (GPS L5 / Galileo E5a I vs Q).
@inline _carrier_phase_derotation(driver_carrier_phase_offset::Real, signal) =
    cis(driver_carrier_phase_offset - get_carrier_phase_offset(signal))
