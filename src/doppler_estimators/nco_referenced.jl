# ── The NCO-referenced (delay-aware) PLL/DLL ─────────────────────────────────

"""
    NCOReferencedPLLAndDLL(; wide_carrier_loop_filter_bandwidth = nothing,
                             code_loop_filter_bandwidth = nothing,
                             narrow_carrier_loop_filter_bandwidth = nothing,
                             fll_assist_loop_filter_bandwidth = nothing,
                             predict_landing = true)

FLL-assisted PLL and DLL Doppler estimator for a replica whose NCO words are
applied with a known delay; the hardware-correlator receiver's default.

It is the [`ConventionalAssistedPLLAndDLL`](@ref) (same filters, gains and
[`CarrierLoopStage`](@ref) staging) with its internals referenced to the device
NCO. The one different default is the wide carrier bandwidth: the narrow one
([`default_narrow_carrier_loop_filter_bandwidth`](@ref), 18 Hz) instead of 50 Hz,
since delay tolerance shrinks with bandwidth (seven records of delay at 50 Hz,
thirteen at 18 Hz); a method of [`default_wide_carrier_loop_filter_bandwidth`](@ref)
does not reach it. It still narrows once phase lock has held, to the same 18 Hz
but under the tighter narrow cap (0.04/T against 0.09/T): from 18 to 10 Hz on
4 ms records, from 9 to 4 Hz on 10 ms ones.

 1. **Every record is attributed to the word that ran under it.** Both
    discriminators are measured against the applied replica, so `applied word +
    FLL discriminator` is an *absolute* Doppler measurement; the DLL is
    normalised with the applied code word. The conventional loop does this too.
 2. **The correction is sized for the moment it lands.** The filter is fed the
    discriminators *predicted at the new command's landing sample*: the phase
    error (cycles) advanced by `∫ (f̂ − w(τ)) dτ` over the words already
    scheduled, and the frequency measurement re-based onto the word running
    there.

With zero delay step 2 is the identity and the estimator *is* the conventional
loop at the same bandwidths (the defaults differ in the wide one), inheriting its
noise performance. `predict_landing = false` drops
step 2, giving the conventional loop: the **negative control**, which fails like
it at a few epochs of delay.
"""
struct NCOReferencedPLLAndDLL{CO<:AbstractLoopFilter} <: AbstractDopplerEstimator
    wide_carrier_loop_filter_bandwidth::Maybe{typeof(1.0Hz)}
    code_loop_filter_bandwidth::Maybe{typeof(1.0Hz)}
    narrow_carrier_loop_filter_bandwidth::Maybe{typeof(1.0Hz)}
    fll_assist_loop_filter_bandwidth::Maybe{typeof(1.0Hz)}
    predict_landing::Bool
end

function NCOReferencedPLLAndDLL(
    ::Type{CO} = SecondOrderBilinearLF;
    wide_carrier_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    code_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    narrow_carrier_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    fll_assist_loop_filter_bandwidth::Maybe{typeof(1.0Hz)} = nothing,
    predict_landing::Bool = true,
) where {CO<:AbstractLoopFilter}
    NCOReferencedPLLAndDLL{CO}(
        wide_carrier_loop_filter_bandwidth,
        code_loop_filter_bandwidth,
        narrow_carrier_loop_filter_bandwidth,
        fll_assist_loop_filter_bandwidth,
        predict_landing,
    )
end

"""
    SatNCOReferencedPLLAndDLL

Per-satellite state of an [`NCOReferencedPLLAndDLL`](@ref): handover Dopplers
(the filters output offsets from them), both filters, their
[`LoopBandwidths`](@ref TrackingLoops.LoopBandwidths), the centre sample of the
last record (`NaN` before the first; the FLL's replica word is averaged from there),
and the [`CarrierLoopStaging`](@ref TrackingLoops.CarrierLoopStaging).
"""
struct SatNCOReferencedPLLAndDLL{CA<:ThirdOrderAssistedBilinearLF,CO<:AbstractLoopFilter}
    init_carrier_doppler::typeof(1.0Hz)
    init_code_doppler::typeof(1.0Hz)
    carrier_loop_filter::CA
    code_loop_filter::CO
    bandwidths::LoopBandwidths
    previous_record_center::Float64
    staging::CarrierLoopStaging
end

function SatNCOReferencedPLLAndDLL(
    state::SatNCOReferencedPLLAndDLL{CA,CO};
    carrier_loop_filter::Maybe{CA} = nothing,
    code_loop_filter::Maybe{CO} = nothing,
    previous_record_center::Maybe{Float64} = nothing,
    staging::Maybe{CarrierLoopStaging} = nothing,
) where {CA,CO}
    SatNCOReferencedPLLAndDLL{CA,CO}(
        state.init_carrier_doppler,
        state.init_code_doppler,
        something(carrier_loop_filter, state.carrier_loop_filter),
        something(code_loop_filter, state.code_loop_filter),
        state.bandwidths,
        something(previous_record_center, state.previous_record_center),
        something(staging, state.staging),
    )
end

function init_estimator_state(
    estimator::NCOReferencedPLLAndDLL{CO},
    driver_signal::AbstractGNSSSignal,
    carrier_doppler,
    code_doppler,
) where {CO}
    SatNCOReferencedPLLAndDLL(
        carrier_doppler,
        code_doppler,
        ThirdOrderAssistedBilinearLF(),
        _constructorof(CO)(),
        # The narrow default for the wide bandwidth too: see `NCOReferencedPLLAndDLL`.
        _resolve_bandwidths(
            estimator,
            driver_signal;
            wide_default = default_narrow_carrier_loop_filter_bandwidth(driver_signal),
        ),
        NaN,
        CarrierLoopStaging(ThirdOrderAssistedBilinearLF()),
    )
end

function reset_estimator_state(
    ::NCOReferencedPLLAndDLL,
    state::SatNCOReferencedPLLAndDLL,
    carrier_doppler,
    code_doppler,
)
    SatNCOReferencedPLLAndDLL(
        carrier_doppler,
        code_doppler,
        _constructorof(typeof(state.carrier_loop_filter))(),
        _constructorof(typeof(state.code_loop_filter))(),
        state.bandwidths,
        NaN,
        CarrierLoopStaging(ThirdOrderAssistedBilinearLF()),
    )
end

"""
    wrap_half_cycle(phase)

Fold a carrier-phase error in cycles into `[−1/4, 1/4]`, the range of the Costas
[`pll_disc`](@ref) (a bit flip turns the prompt by half a cycle). Exact inside it.
"""
wrap_half_cycle(phase) = rem(phase, 0.5, RoundNearest)

# The phase error (cycles) the record would show `shift` samples later under the
# scheduled words, integrated from the record's centre with the filter's Doppler
# estimate before this record's update. Folded to ±1/4 cycle for the Costas PLL
# (`polarity = 0`), ±1/2 for the four-quadrant one.
@inline function _predict_landing_phase_error(
    phase_error,
    state::SatNCOReferencedPLLAndDLL,
    words,
    center,
    shift,
    integration_time,
    sampling_frequency,
    polarity = 0,
)
    sampling_freq_hz = Float64(ustrip(Hz, uconvert(Hz, sampling_frequency)))
    carrier_loop_filter = state.carrier_loop_filter
    f_hat = ustrip(
        Hz,
        uconvert(
            Hz,
            state.init_carrier_doppler +
            carrier_loop_filter.x1 +
            integration_time / 2 * carrier_loop_filter.x2,
        ),
    )
    ramp_word = first(mean_nco_word(words, center, center + shift))
    predicted = phase_error + shift * (f_hat - ramp_word) / sampling_freq_hz
    iszero(polarity) ? wrap_half_cycle(predicted) : rem(predicted, 1.0, RoundNearest)
end

"""
    step_loop(estimator::NCOReferencedPLLAndDLL, state, record::LoopRecord, words,
              landing_sample)
        -> (state, carrier_doppler, code_doppler)

One record through the NCO-referenced loop. `words` are the replica words the
record ran on; `landing_sample` is where this fold's command lands
(`NO_LANDING_SAMPLE`: at the record's end). See [`NCOReferencedPLLAndDLL`](@ref).
"""
@inline step_loop(
    estimator::NCOReferencedPLLAndDLL,
    state::SatNCOReferencedPLLAndDLL,
    record::LoopRecord,
    words,
    landing_sample::Int64,
) = _step_scalar_loop(estimator, state, record, words, landing_sample)

@inline function _record_discriminators(
    estimator::NCOReferencedPLLAndDLL,
    state::SatNCOReferencedPLLAndDLL,
    record::LoopRecord,
    words,
    landing_sample::Int64,
    fll::Bool,
)
    signal = record.signal
    sampling_frequency = record.sampling_frequency
    previous_center = state.previous_record_center
    # Every record of the fold maps onto the delay-free loop's record this far ahead.
    shift = landing_sample == NO_LANDING_SAMPLE ? 0 : landing_sample - record.fold_end
    record_end = record.sample_index
    record_samples = record.integrated_samples
    record_start = record_end - record_samples
    center = record_end - record_samples / 2
    integration_time = record_samples / sampling_frequency
    filtered_correlator = record.filtered_correlator

    applied_carrier, applied_code = mean_nco_word(words, record_start, record_end)
    phase_error = pll_disc(signal, filtered_correlator; record.polarity)
    raw_frequency_error =
        fll ?
        fll_disc(
            signal,
            filtered_correlator,
            record.previous_prompt,
            integration_time;
            four_quadrant = !iszero(record.polarity),
        ) : 0.0Hz
    frequency_error = raw_frequency_error

    if estimator.predict_landing && shift > 0
        phase_error = _predict_landing_phase_error(
            phase_error,
            state,
            words,
            center,
            shift,
            integration_time,
            sampling_frequency,
            record.polarity,
        )
        # Re-base the FLL reading from the word between the prompts' centres onto
        # the word running `shift` samples ahead.
        if fll
            fll_word =
                isnan(previous_center) ? applied_carrier :
                first(mean_nco_word(words, previous_center, center))
            landing_word =
                first(mean_nco_word(words, record_start + shift, record_end + shift))
            frequency_error += (fll_word - landing_word) * Hz
        end
    end

    (;
        integration_time,
        carrier_bandwidth = _carrier_bandwidth(
            state.bandwidths,
            state.staging.stage,
            integration_time,
        ),
        fll_assist_bandwidth = _fll_assist_bandwidth(state.bandwidths, integration_time),
        phase_error,
        frequency_error,
        raw_frequency_error,
        code_error = dll_disc(
            signal,
            filtered_correlator,
            applied_code * Hz,
            sampling_frequency,
        ),
        center,
    )
end

@inline _stepped_state(
    state::SatNCOReferencedPLLAndDLL,
    carrier_loop_filter,
    code_loop_filter,
    center,
    staging::CarrierLoopStaging,
) = SatNCOReferencedPLLAndDLL(
    state;
    carrier_loop_filter,
    code_loop_filter,
    previous_record_center = center,
    staging,
)
