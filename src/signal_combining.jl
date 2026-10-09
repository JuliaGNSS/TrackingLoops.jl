# ─────────────────────────────────────────────────────────────────────────────
# Signal combining: the passengers (every signal of a satellite but the driver
# whose record `step_loop` closes the loops on) mix their discriminators into the
# driver's before its loop filters read them. The host steps each passenger
# record with `step_loop` as it completes, as it does the driver's; the passenger
# step adds it to the sums, and the driver's next step closes on the weighted
# means and starts the sums afresh. See "Signal combining" in the manual.
# ─────────────────────────────────────────────────────────────────────────────

"""
    WeightedSum(sum, weight)

One loop's weighted sum of passenger discriminators and its summed weights.
The weight of a record is its signal's ICD power share times its integration
time, cubed for the FLL, whose noise variance falls with the cube.
"""
struct WeightedSum{S,W}
    sum::S
    weight::W
end

# One more discriminator `reading` with weight `weight` added.
@inline _accumulated(ws::WeightedSum, reading, weight) =
    WeightedSum(ws.sum + weight * reading, ws.weight + weight)

"""
    SignalCombiningSums()

The passengers' weighted discriminators pending for the driver's next record,
one [`WeightedSum`](@ref) per loop, held in the per-satellite state of an
estimator that combines signals:

  - `pll`: PLL discriminators in cycles, weighted in s;
  - `fll`: FLL discriminators in Hz, weighted in s³;
  - `dll`: DLL discriminators referred to the driver's code phase, in chips,
    weighted in s.
"""
struct SignalCombiningSums
    pll::WeightedSum{typeof(1.0s),typeof(1.0s)}
    fll::WeightedSum{typeof(1.0Hz * 1.0s^3),typeof(1.0s^3)}
    dll::WeightedSum{typeof(1.0s),typeof(1.0s)}
end

SignalCombiningSums() = SignalCombiningSums(
    WeightedSum(0.0s, 0.0s),
    WeightedSum(0.0Hz * 0.0s^3, 0.0s^3),
    WeightedSum(0.0s, 0.0s),
)

# The loops passengers are combined into.
const _ALL_LOOPS = (pll = true, fll = true, dll = true)
const _PLL_ONLY = (pll = true, fll = false, dll = false)

# A weighted mean of the driver's own discriminator and the passengers' sum. With
# no passenger weight it is the driver's own as is: `(w · d) / w` is not `d` bit
# for bit.
@inline _weighted_mean(own, own_weight, pending::WeightedSum) =
    iszero(pending.weight) ? own :
    (own_weight * own + pending.sum) / (own_weight + pending.weight)

# The weighted mean, but only while the driver's own reading lies within `range`,
# the passengers' two-quadrant range: there a four-quadrant driver reads the error
# as they do, beyond it they would fold it by half a cycle. A two-quadrant
# driver's reading never leaves the range.
@inline _gated_mean(own, own_weight, pending::WeightedSum, range) =
    abs(own) < range ? _weighted_mean(own, own_weight, pending) : own

# The two-quadrant discriminators' ranges, which the passengers read and which
# `_gated_mean` keeps a four-quadrant driver within: ±1/4 cycle for the PLL's
# `atan(Q / I)`, ±1/(4T) for the FLL's `atan(cross / dot)`.
const _TWO_QUADRANT_PLL_RANGE = 0.25
@inline _two_quadrant_fll_range(integration_time) = uconvert(Hz, 1 / (4 * integration_time))

# The weight of one record's discriminator: its signal's ICD power share times its
# integration time. The FLL's is the integration time cubed, as its noise
# variance falls with its cube.
@inline _discriminator_weight(signal::AbstractGNSSSignal, integration_time) =
    get_relative_power(signal) * uconvert(s, integration_time)
@inline _fll_discriminator_weight(signal::AbstractGNSSSignal, integration_time) =
    get_relative_power(signal) * uconvert(s, integration_time)^3

# One passenger record's weighted discriminators added to `sums`, formed only for
# the loops it is combined into. Its PLL is read on the driver's carrier phase
# frame. Its two-quadrant carrier discriminators are blind to a sign flip of a
# whole record, so data passengers and records correlated before the passenger's
# own sync count like any other. Its DLL is normalised with the code word the
# satellite's replica ran on and referred to the driver's code phase by the
# record's `differential_group_delay_chips` (`NaN`: unknown, not combined).
# `driver_carrier_phase` is the driver's carrier phase offset (rad).
@inline function _add_passenger_discriminators(
    sums::SignalCombiningSums,
    record,
    words,
    loops,
    driver_carrier_phase::Real,
)
    differential_group_delay_chips = record.differential_group_delay_chips
    signal = record.signal
    correlator = record.filtered_correlator
    integration_time = record.integrated_samples / record.sampling_frequency
    weight = _discriminator_weight(signal, integration_time)
    pll_weight = loops.pll ? weight : zero(weight)
    derotation = _carrier_phase_derotation(driver_carrier_phase, signal)
    pll =
        iszero(pll_weight) ? 0.0 :
        _phase_error_in_cycles(
            pll_disc(
                signal,
                update_accumulator(correlator, get_accumulators(correlator) .* derotation),
            ),
        )
    fll_weight =
        loops.fll && !iszero(record.previous_prompt) ?
        _fll_discriminator_weight(signal, integration_time) : 0.0s^3
    fll =
        iszero(fll_weight) ? 0.0Hz :
        fll_disc(signal, correlator, record.previous_prompt, integration_time)
    dll_weight =
        loops.dll && !isnan(differential_group_delay_chips) ? weight : zero(weight)
    dll = if iszero(dll_weight)
        0.0
    else
        record_start = record.sample_index - record.integrated_samples
        _, applied_code = mean_nco_word(words, record_start, record.sample_index)
        dll_disc(signal, correlator, applied_code * Hz, record.sampling_frequency) +
        differential_group_delay_chips
    end
    SignalCombiningSums(
        _accumulated(sums.pll, pll, pll_weight),
        _accumulated(sums.fll, fll, fll_weight),
        _accumulated(sums.dll, dll, dll_weight),
    )
end

# The driver's discriminators of one record combined with the passengers'
# pending sums in the loops `loops`, which only `pending` already restricts, so
# this is the identity where no passenger record is pending. A four-quadrant
# driver reading is combined only within the passengers' two-quadrant range. The
# FLL is combined into `frequency_error` and `raw_frequency_error` alike: only
# the conventional loop combines, where the two are one reading.
@inline function _combine_discriminators(
    discriminators,
    pending::SignalCombiningSums,
    record,
    loops,
)
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
