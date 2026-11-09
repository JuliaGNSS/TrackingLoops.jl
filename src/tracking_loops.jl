# ─────────────────────────────────────────────────────────────────────────────
# The scalar tracking loops: configuration, per-satellite state, and the
# per-integration `step_loop!`, called by Tracking and by a hardware correlator's loop:
#
#     step_loop!(tracking_loop, state, integration, nco_history, application_sample_index)
#         -> (state, carrier_doppler, code_doppler)
#
# `nco_history` answers `calc_mean_nco_command(nco_history, a, b)` (a `ConstantNCOCommand` in software, the
# channel's `NCOTimeline` in hardware); `application_sample_index` is the device sample the
# batch's command takes effect at, `nothing` for each integration's end.
# ─────────────────────────────────────────────────────────────────────────────

# The soft bits an integration appended: a contiguous view of the bit synchronizer's vector.
const SoftBitsView = SubArray{Float32,1,Vector{Float32},Tuple{UnitRange{Int}},true}

# The soft bits of an integration built without a signal state: none. Never written to.
const _NO_SOFT_BITS = Float32[]

@inline _no_soft_bits() = view(_NO_SOFT_BITS, 1:0)

"""
$(SIGNATURES)

A coherent integration is one integration of one signal ahead of the discriminators, one
integrate-and-dump interval (Kaplan and Hegarty's predetection integration): the
correlator bank output, from one hardware dump or a sum of dumps, with the bookkeeping
the loop needs.

One completed integration as the loop-filter step sees it: the signal, the filtered
(antenna-combined, normalised) correlator bank, the previous filtered prompt the FLL
compares with, the integration's span and period count, the band's sampling frequency,
and `command_issue_sample_index`, where the loop issues its command: the `sample_index` of
the last integration of its batch (the integrations a host steps before its next command
takes effect), from which the command's delay to its application sample is measured.

`num_code_periods` must be at least 1: the carrier-loop staging sizes its
phase-lock indicator's averages in integrations of that length and divides by it. An
integration built from a [`PostCorrelationState`](@ref) always is ([`process_correlator_bank_output!`](@ref)
counts at least one code period); a host building integrations otherwise must ensure it.

The four-quadrant FLL reads a prompt sign flip as half a cycle of rotation, so an
integration whose `carrier_phase_polarity` differs from the previous integration's comes with
`previous_prompt = 0`, giving no FLL reading, as for the first integration. [`process_correlator_bank_output!`](@ref) applies this; a host building integrations otherwise passes `0` itself.

Fields used by a tracking loop with shared navigation state
([`VectorTrackingLoop`](@ref)):

  - `prn`: the satellite (`0` if unknown);
  - `code_phase`: the replica's code phase (chips) at `sample_index`, from the
    [`CorrelatorBankOutput`](@ref) (`NaN` if not reported).

For such a tracking loop `sample_index / sampling_frequency` must also be the time
since an origin shared by every satellite of the band; a host whose correlator
restarts its sample count passes the offset as `sample_index_offset` (added to
`sample_index` and `command_issue_sample_index`).

The rest summarises the signal's [`PostCorrelationState`](@ref) after
[`process_correlator_bank_output!`](@ref) processed this integration, for a tracking loop that decodes or weights
by C/N₀ (the vector loop; the scalar loops ignore it):

  - `bit_or_secondary_code_synced`: whether the bit clock holds bit or secondary-code sync;
  - `sync_change`: whether this integration found or lost it ([`SyncEvent`](@ref));
  - `num_code_periods_into_bit`: code periods integrated into the current navigation bit
    (`0` before sync);
  - `new_soft_bits`: exactly the soft bits *this integration* appended, a view of the
    bit synchronizer's vector valid until the host drains, resets, or grows that vector;
  - `cn0_estimator`: the signal's C/N₀ estimator, for [`estimate_cn0`](@ref);
  - `bit_polarity`: the sign the bit clock's sync found its soft bits at (`-1` if
    it negates the accumulated prompts' real part, `+1` if it keeps it, `0` before
    sync).

`carrier_phase_polarity` is the prompt's sign where the replica wiped every sign
modulation off it (a dataless signal synced to its secondary code), else `0`.
Nonzero, the PLL and the FLL read the integration four-quadrant (`pll_disc(...;
polarity)`, `fll_disc(...; polarity)`); `0`, two-quadrant.

[`process_correlator_bank_output!`](@ref) fills all of this in. The constructors below leave
the bit clock unsynchronised, add no soft bits, carry a [`NoCN0Estimator`](@ref)
and take `carrier_phase_polarity` as a keyword (`0` by default): a host with its
own per-signal state passes the one `_process_correlator_bank_output!` returned for the integration.
"""
struct CoherentIntegration{S<:AbstractGNSSSignal,C<:AbstractCorrelatorBank,F,CN0<:AbstractCN0Estimator}
    signal::S
    filtered_correlator_bank::C
    previous_prompt::ComplexF64
    num_integrated_samples::Int
    sample_index::Int
    command_issue_sample_index::Int
    num_code_periods::Int
    sampling_frequency::F
    prn::Int
    code_phase::Float64
    bit_or_secondary_code_synced::Bool
    sync_change::SyncEvent
    num_code_periods_into_bit::Int
    new_soft_bits::SoftBitsView
    cn0_estimator::CN0
    carrier_phase_polarity::Int8
    bit_polarity::Int8
end

CoherentIntegration(
    signal::AbstractGNSSSignal,
    filtered_correlator_bank::AbstractCorrelatorBank,
    previous_prompt,
    num_integrated_samples::Integer,
    sample_index::Integer,
    command_issue_sample_index::Integer,
    num_code_periods::Integer,
    sampling_frequency;
    prn::Integer = 0,
    code_phase::Real = NaN,
    carrier_phase_polarity::Integer = 0,
) = CoherentIntegration(
    signal,
    filtered_correlator_bank,
    ComplexF64(previous_prompt),
    Int(num_integrated_samples),
    Int(sample_index),
    Int(command_issue_sample_index),
    Int(num_code_periods),
    sampling_frequency,
    Int(prn),
    Float64(code_phase),
    false,
    SYNC_UNCHANGED,
    0,
    _no_soft_bits(),
    NoCN0Estimator(),
    Int8(carrier_phase_polarity),
    Int8(0),
)

CoherentIntegration(
    signal,
    filtered_correlator_bank,
    previous_prompt,
    output::CorrelatorBankOutput,
    num_code_periods::Integer,
    sampling_frequency;
    command_issue_sample_index = output.sample_index,
    prn::Integer = 0,
    sample_index_offset::Integer = 0,
    carrier_phase_polarity::Integer = 0,
) = CoherentIntegration(
    signal,
    filtered_correlator_bank,
    ComplexF64(previous_prompt),
    output.num_integrated_samples,
    output.sample_index + Int(sample_index_offset),
    Int(command_issue_sample_index) + Int(sample_index_offset),
    Int(num_code_periods),
    sampling_frequency,
    Int(prn),
    output.code_phase,
    false,
    SYNC_UNCHANGED,
    0,
    _no_soft_bits(),
    NoCN0Estimator(),
    Int8(carrier_phase_polarity),
    Int8(0),
)

# ── The conventional PLL/DLL ─────────────────────────────────────────────────

"""
Shared per-satellite state of [`ScalarTrackingLoop`](@ref) and
[`DelayCompensatedTrackingLoop`](@ref): reference Dopplers, loop
filters, their [`LoopBandwidths`](@ref), the key of the driver signal whose integrations
close the loops (`0`: every integration's; see [`step_loop!`](@ref)), and the
[`CarrierLoopStaging`](@ref). `previous_prompt_sample_index` is the last driver
prompt's center sample (`NaN` initially), used to attribute FLL measurements
between successive prompts.
"""
@kwdef struct ScalarLoopState{CA<:AbstractLoopFilter,CO<:AbstractLoopFilter}
    reference_carrier_doppler::typeof(1.0Hz)
    reference_code_doppler::typeof(1.0Hz)
    carrier_loop_filter::CA = ThirdOrderBilinearLF()
    code_loop_filter::CO = SecondOrderBilinearLF()
    bandwidths::LoopBandwidths = LoopBandwidths()
    previous_prompt_sample_index::Float64 = NaN
    driver_key::UInt64 = UInt64(0)
    staging::CarrierLoopStaging = CarrierLoopStaging(typeof(carrier_loop_filter))
end

function ScalarLoopState(
    state::ScalarLoopState{CA,CO};
    carrier_loop_filter::Maybe{CA} = nothing,
    code_loop_filter::Maybe{CO} = nothing,
    bandwidths::Maybe{LoopBandwidths} = nothing,
    staging::Maybe{CarrierLoopStaging} = nothing,
    previous_prompt_sample_index::Maybe{Float64} = nothing,
) where {CA<:AbstractLoopFilter,CO<:AbstractLoopFilter}
    ScalarLoopState{CA,CO}(
        state.reference_carrier_doppler,
        state.reference_code_doppler,
        isnothing(carrier_loop_filter) ? state.carrier_loop_filter :
        carrier_loop_filter,
        isnothing(code_loop_filter) ? state.code_loop_filter :
        code_loop_filter,
        isnothing(bandwidths) ? state.bandwidths : bandwidths,
        something(previous_prompt_sample_index, state.previous_prompt_sample_index),
        state.driver_key,
        isnothing(staging) ? state.staging : staging,
    )
end

# The key a scalar loop's state records its driver by: the signal's id as an
# integer, so the state stays a plain bits value of one type whatever the driver,
# and a host can keep the states of satellites with different drivers in one
# vector.
@inline _signal_key(signal::AbstractGNSSSignal) = UInt64(objectid(get_signal_id(signal)))

"""
$(SIGNATURES)

Conventional PLL and DLL Doppler loop. Configuration only; per-satellite
state is a [`ScalarLoopState`](@ref) from [`init_loop_state`](@ref).

`CA` and `CO` are the carrier and code loop filter types. The bandwidths (the PLL's
wide and narrow, the FLL path's and the DLL's; see [`CarrierLoopStage`](@ref))
default to `nothing`, **auto**: sized per satellite from its driver signal via
[`default_wide_carrier_loop_filter_bandwidth`](@ref),
[`default_narrow_carrier_loop_filter_bandwidth`](@ref),
[`default_fll_assist_loop_filter_bandwidth`](@ref) and
[`default_code_loop_filter_bandwidth`](@ref). Each is capped against the integration's
integration time at filter time (see [`LoopBandwidths`](@ref)), so longer
integration needs no re-tuning.

Use `ScalarTrackingLoop(ThirdOrderAssistedBilinearLF)` for FLL assistance during
pull-in, followed by the same carrier staging and PLL tracking.
"""
struct ScalarTrackingLoop{CA<:AbstractLoopFilter,CO<:AbstractLoopFilter} <:
       AbstractTrackingLoop
    bandwidths::_LoopBandwidthConfiguration
end

function ScalarTrackingLoop(
    ::Type{CA} = ThirdOrderBilinearLF,
    ::Type{CO} = SecondOrderBilinearLF;
    kwargs...,
) where {CA<:AbstractLoopFilter,CO<:AbstractLoopFilter}
    ScalarTrackingLoop{CA,CO}(_LoopBandwidthConfiguration(; kwargs...))
end

# Copy with the given (non-`nothing`) bandwidths replaced.
ScalarTrackingLoop(loop::ScalarTrackingLoop{CA,CO}; kwargs...) where {CA,CO} =
    ScalarTrackingLoop{CA,CO}(_LoopBandwidthConfiguration(loop.bandwidths; kwargs...))

"""
    init_loop_state(loop, driver_signal, carrier_doppler, code_doppler)

Per-satellite loop state for a satellite driven by `driver_signal`, starting
at the given Dopplers; auto bandwidths are resolved here from `driver_signal`. The
state records the driver, so [`step_loop!`](@ref) tells its integrations from the
satellite's other signals', and starts the [`CarrierLoopStage`](@ref) staging.
Must be **pure**: Tracking.jl also calls it for template states and re-initialisation.
"""
function init_loop_state(
    loop::ScalarTrackingLoop{CA,CO},
    driver_signal::AbstractGNSSSignal,
    carrier_doppler,
    code_doppler,
) where {CA<:AbstractLoopFilter,CO<:AbstractLoopFilter}
    _init_scalar_loop_state(loop, driver_signal, carrier_doppler, code_doppler,
        _init_loop_filter(CA), _init_loop_filter(CO), LoopBandwidths(driver_signal))
end

"""
    reinit_loop_state(loop, state, carrier_doppler, code_doppler)

Re-initialise `state` at the given Dopplers with zeroed loop filters, keeping the
bandwidths and the carrier loop's staging (see [`CarrierLoopStage`](@ref)): a
re-initialisation is a handover, not a re-acquisition, so after a change of
integration length, say, the loop holds its stage, the phase-lock indicator reads on,
and the bandwidths are capped for the new length.
"""
function reinit_loop_state(
    ::AbstractTrackingLoop,
    state::ScalarLoopState,
    carrier_doppler,
    code_doppler,
)
    ScalarLoopState(
        carrier_doppler,
        code_doppler,
        _reset_loop_filter(state.carrier_loop_filter),
        _reset_loop_filter(state.code_loop_filter),
        state.bandwidths,
        NaN,
        state.driver_key,
        state.staging,
    )
end

"""
    step_loop!(loop::ScalarTrackingLoop, state, integration::CoherentIntegration, nco_history,
              application_sample_index)
        -> (state, carrier_doppler, code_doppler)

One integration through the conventional loop at the bandwidths of its
[`CarrierLoopStage`](@ref), capped against the coherent integration time; the DLL
is normalised with the code command the integration ran on, and the Dopplers are aided
([`calc_carrier_aided_dopplers`](@ref)). `application_sample_index` is ignored:
the command is assumed to act before the next integration.

A host hands every integration of a satellite to `step_loop!`, the driver's and its
passengers' alike; only the driver's step the loop. A passenger integration leaves the
state as it is and returns the command in force at the application sample (the
integration's end under `nothing`), so applying it changes nothing.
"""
@inline step_loop!(
    loop::ScalarTrackingLoop,
    state::ScalarLoopState,
    integration::CoherentIntegration,
    nco_history,
    application_sample_index::Union{Nothing,Integer},
) =
    _is_driver_integration(state, integration) ?
    _step_scalar_loop(loop, state, integration, nco_history, application_sample_index) :
    _step_passenger(state, integration, nco_history, application_sample_index)

# Whether `integration` belongs to the signal the scalar state was initialised with.
@inline _is_driver_integration(state, integration::CoherentIntegration) = _is_driven_by(state, integration.signal)

# Whether the scalar state's satellite is driven by `signal` (any signal for a state
# without a recorded driver).
@inline _is_driven_by(state, signal::AbstractGNSSSignal) =
    state.driver_key == 0 || _signal_key(signal) == state.driver_key

# A passenger integration through a loop it does not close: the state unchanged and the
# command in force where this integration's command would take effect.
@inline function _step_passenger(state, integration::CoherentIntegration, nco_history, application_sample_index::Union{Nothing,Integer})
    carrier_doppler, code_doppler = _command_in_force(integration, nco_history, application_sample_index)
    state, carrier_doppler, code_doppler
end

@inline function _command_in_force(integration::CoherentIntegration, nco_history, application_sample_index::Union{Nothing,Integer})
    application_sample_index = isnothing(application_sample_index) ? integration.sample_index : application_sample_index
    (; carrier_doppler, code_doppler) = calc_mean_nco_command(nco_history, application_sample_index, application_sample_index)
    carrier_doppler * Hz, code_doppler * Hz
end

# One integration's discriminators plus its integration time and centre sample.
# `phase_error` (cycles) and `frequency_error` (Hz) feed the scalar carrier filter;
# the frequency error is formed only if `form_fll` (while FLL-assisted), else zero.
# The DLL's `code_error` is normalised with the code command the integration ran on.
@inline function _discriminators(
    ::ScalarTrackingLoop,
    state::ScalarLoopState,
    integration::CoherentIntegration,
    nco_history,
    application_sample_index::Union{Nothing,Integer};
    form_fll::Bool,
)
    _measure_scalar_integration(integration, nco_history; form_fll)
end

@inline function _measure_scalar_integration(integration::CoherentIntegration, nco_history; form_fll::Bool)
    signal = integration.signal
    coherent_integration_time = integration.num_integrated_samples / integration.sampling_frequency
    integration_start_sample_index = integration.sample_index - integration.num_integrated_samples
    nco_command = calc_mean_nco_command(nco_history, integration_start_sample_index, integration.sample_index)
    applied_carrier, applied_code = nco_command.carrier_doppler, nco_command.code_doppler
    filtered_correlator_bank = integration.filtered_correlator_bank
    frequency_error =
        form_fll ?
        fll_disc(
            signal,
            filtered_correlator_bank,
            integration.previous_prompt,
            coherent_integration_time;
            polarity = integration.carrier_phase_polarity,
        ) : 0.0Hz
    (;
        coherent_integration_time,
        applied_carrier,
        phase_error = pll_disc(signal, filtered_correlator_bank; polarity = integration.carrier_phase_polarity),
        frequency_error,
        code_error = dll_disc(
            signal,
            filtered_correlator_bank,
            applied_code * Hz,
            integration.sampling_frequency,
        ),
        center = integration.sample_index - integration.num_integrated_samples / 2,
    )
end

# The carrier filter's input: the FLL-assisted filter takes both
# discriminators, any other the phase discriminator alone.
@inline _carrier_filter_input(::ThirdOrderAssistedBilinearLF, phase_error, frequency_error) =
    (phase_error, frequency_error)
@inline _carrier_filter_input(::AbstractLoopFilter, phase_error, frequency_error) =
    phase_error

# The scalar carrier filter's bandwidth for an integration of `coherent_integration_time`: the
# stage's PLL bandwidth, paired with the FLL path's for an FLL-assisted filter.
# Capped against the time the integration actually integrated, not the intended length:
# integrations stepped after a sync found mid-batch, and the truncated first post-sync
# integration, are short.
@inline function _carrier_filter_bandwidth(state, coherent_integration_time)
    pll = effective_carrier_loop_filter_bandwidth(
        state.staging.stage == NARROW_PLL ? state.bandwidths.narrow_carrier :
        state.bandwidths.wide_carrier,
        coherent_integration_time,
    )
    _has_fll_path(typeof(state.carrier_loop_filter)) || return pll
    pll, effective_fll_assist_loop_filter_bandwidth(state.bandwidths.fll_assist, coherent_integration_time)
end

# The state after one integration, with both loop filters stepped and the staging
# advanced.
@inline _stepped_state(
    state::ScalarLoopState,
    carrier_loop_filter,
    code_loop_filter,
    center,
    staging,
) = ScalarLoopState(state; carrier_loop_filter, code_loop_filter, staging, previous_prompt_sample_index = center)

# One integration through a scalar loop (see `step_loop!`).
@inline function _step_scalar_loop(loop, state, integration::CoherentIntegration, nco_history, application_sample_index::Union{Nothing,Integer})
    discriminators = _discriminators(
        loop,
        state,
        integration,
        nco_history,
        application_sample_index;
        form_fll = state.staging.stage == FLL_ASSISTED_PLL,
    )
    coherent_integration_time = discriminators.coherent_integration_time
    code_loop_filter_bandwidth =
        effective_code_loop_filter_bandwidth(state.bandwidths.code, coherent_integration_time)
    carrier_frequency_correction, carrier_loop_filter = filter_loop(
        state.carrier_loop_filter,
        _carrier_filter_input(
            state.carrier_loop_filter,
            discriminators.phase_error,
            discriminators.frequency_error,
        ),
        coherent_integration_time,
        _carrier_filter_bandwidth(state, coherent_integration_time),
    )
    code_frequency_correction, code_loop_filter = filter_loop(
        state.code_loop_filter,
        discriminators.code_error,
        coherent_integration_time,
        code_loop_filter_bandwidth,
    )
    carrier_doppler, code_doppler = calc_carrier_aided_dopplers(
        integration.signal,
        state.reference_carrier_doppler,
        state.reference_code_doppler,
        carrier_frequency_correction,
        code_frequency_correction,
    )
    _stepped_state(
        state,
        carrier_loop_filter,
        code_loop_filter,
        discriminators.center,
        _advance_staging(state.staging, integration),
    ),
    carrier_doppler,
    code_doppler
end

# ── The NCO-referenced (delay-aware) PLL/DLL ─────────────────────────────────

"""
    DelayCompensatedTrackingLoop(; wide_carrier_loop_filter_bandwidth = nothing,
                             narrow_carrier_loop_filter_bandwidth = nothing,
                             fll_assist_loop_filter_bandwidth = nothing,
                             code_loop_filter_bandwidth = nothing,
                             compensate_delay = true)

FLL-assisted PLL and DLL Doppler loop for a replica whose NCO commands are
applied with a known delay; the hardware-correlator receiver's default.

It is the `ScalarTrackingLoop(ThirdOrderAssistedBilinearLF)` (same filters, gains,
[`CarrierLoopStage`](@ref) staging and bandwidth defaults) with its internals
referenced to the device NCO. The one different default is the wide PLL bandwidth:
the narrow one ([`default_narrow_carrier_loop_filter_bandwidth`](@ref)), as a wider
loop tolerates less command delay.

 1. **Every integration is attributed to the command that ran under it.** Both
    discriminators are measured against the applied replica, so `applied command +
    FLL discriminator` is an *absolute* Doppler measurement; the DLL is
    normalised with the applied code command. The conventional loop does this too.
 2. **The correction is sized for the moment it takes effect.** The filter is fed the
    discriminators *predicted at the new command's application sample*: the phase
    error (cycles) advanced by `∫ (f̂ − w(τ)) dτ` over the commands already
    scheduled, and the frequency measurement re-based onto the command running
    there.

With zero delay step 2 is the identity and the loop *is* the conventional
loop at the same bandwidths, inheriting its noise performance.
`compensate_delay = false` drops step 2, making it arithmetically the
`ScalarTrackingLoop(ThirdOrderAssistedBilinearLF)`: the documented **negative control**,
which fails like it at a few epochs of delay.
"""
struct DelayCompensatedTrackingLoop{CO<:AbstractLoopFilter} <: AbstractTrackingLoop
    bandwidths::_LoopBandwidthConfiguration
    compensate_delay::Bool
end

function DelayCompensatedTrackingLoop(
    ::Type{CO} = SecondOrderBilinearLF;
    compensate_delay::Bool = true,
    kwargs...,
) where {CO<:AbstractLoopFilter}
    DelayCompensatedTrackingLoop{CO}(_LoopBandwidthConfiguration(; kwargs...), compensate_delay)
end

function _init_scalar_loop_state(loop, driver_signal, carrier_doppler, code_doppler,
    carrier_loop_filter, code_loop_filter, defaults::LoopBandwidths)
    ScalarLoopState(
        carrier_doppler,
        code_doppler,
        carrier_loop_filter,
        code_loop_filter,
        _resolve_bandwidths(loop.bandwidths, defaults),
        NaN,
        _signal_key(driver_signal),
        CarrierLoopStaging(typeof(carrier_loop_filter)),
    )
end

# Delay compensation pulls in at the narrow bandwidth: a wider loop tolerates
# less command delay. All other initialization is shared with the scalar loop.
function init_loop_state(
    loop::DelayCompensatedTrackingLoop{CO},
    driver_signal::AbstractGNSSSignal,
    carrier_doppler,
    code_doppler,
) where {CO}
    defaults = LoopBandwidths(driver_signal;
        wide_carrier = default_narrow_carrier_loop_filter_bandwidth(driver_signal))
    _init_scalar_loop_state(loop, driver_signal, carrier_doppler, code_doppler,
        ThirdOrderAssistedBilinearLF(), _init_loop_filter(CO), defaults)
end

"""
    wrap_phase_error(phase_error; polarity = 0)

Fold a carrier-phase error in cycles into the range of [`pll_disc`](@ref) with the
same `polarity`: ±1/4 cycle for the Costas PLL (`polarity = 0`; a bit flip turns
the prompt by half a cycle), ±1/2 for the four-quadrant one. Exact inside it.
"""
wrap_phase_error(phase_error; polarity::Real = 0) =
    rem(phase_error, iszero(polarity) ? 0.5 : 1.0, RoundNearest)

# The phase error (cycles) the integration would show `shift` samples later under the
# scheduled commands, integrated from the integration's centre with the filter's Doppler
# estimate before this integration's update, folded as by `wrap_phase_error`.
@inline function _predict_application_phase_error(
    phase_error,
    polarity,
    state::ScalarLoopState,
    nco_history,
    center,
    shift,
    coherent_integration_time,
    sampling_frequency,
)
    sampling_freq_hz = Float64(ustrip(Hz, uconvert(Hz, sampling_frequency)))
    carrier_loop_filter = state.carrier_loop_filter
    f_hat = ustrip(
        Hz,
        uconvert(
            Hz,
            state.reference_carrier_doppler +
            carrier_loop_filter.x1 +
            coherent_integration_time / 2 * carrier_loop_filter.x2,
        ),
    )
    ramp_command = calc_mean_nco_command(nco_history, center, center + shift).carrier_doppler
    wrap_phase_error(phase_error + shift * (f_hat - ramp_command) / sampling_freq_hz; polarity)
end

"""
    step_loop!(loop::DelayCompensatedTrackingLoop, state, integration::CoherentIntegration, nco_history,
              application_sample_index)
        -> (state, carrier_doppler, code_doppler)

One integration through the delay-compensated loop. `nco_history` describes the replica commands the
integration ran on; `application_sample_index` is where this batch's command takes effect
(`nothing`: at the integration's end). See [`DelayCompensatedTrackingLoop`](@ref).
Passenger integrations are handled as by the conventional loop.
"""
@inline step_loop!(
    loop::DelayCompensatedTrackingLoop,
    state::ScalarLoopState,
    integration::CoherentIntegration,
    nco_history,
    application_sample_index::Union{Nothing,Integer},
) =
    _is_driver_integration(state, integration) ?
    _step_scalar_loop(loop, state, integration, nco_history, application_sample_index) :
    _step_passenger(state, integration, nco_history, application_sample_index)

@inline function _discriminators(
    loop::DelayCompensatedTrackingLoop,
    state::ScalarLoopState,
    integration::CoherentIntegration,
    nco_history,
    application_sample_index::Union{Nothing,Integer};
    form_fll::Bool,
)
    sampling_frequency = integration.sampling_frequency
    previous_center = state.previous_prompt_sample_index
    # Every integration of the batch maps onto the delay-free loop's integration this far ahead.
    shift = isnothing(application_sample_index) ? 0 : application_sample_index - integration.command_issue_sample_index
    integration_end_sample_index = integration.sample_index
    integration_samples = integration.num_integrated_samples
    integration_start_sample_index = integration_end_sample_index - integration_samples
    center = integration_end_sample_index - integration_samples / 2
    coherent_integration_time = integration_samples / sampling_frequency
    measured = _measure_scalar_integration(integration, nco_history; form_fll)
    phase_error = measured.phase_error
    frequency_error = measured.frequency_error

    if loop.compensate_delay && shift > 0
        phase_error = _predict_application_phase_error(
            phase_error,
            integration.carrier_phase_polarity,
            state,
            nco_history,
            center,
            shift,
            coherent_integration_time,
            sampling_frequency,
        )
        # Re-base the FLL reading from the command between the prompts' centres onto
        # the command running `shift` samples ahead.
        fll_command =
            isnan(previous_center) ? measured.applied_carrier :
            calc_mean_nco_command(nco_history, previous_center, center).carrier_doppler
        application_command =
            calc_mean_nco_command(nco_history, integration_start_sample_index + shift, integration_end_sample_index + shift).carrier_doppler
        form_fll && (frequency_error += (fll_command - application_command) * Hz)
    end

    (;
        coherent_integration_time,
        phase_error,
        frequency_error,
        code_error = measured.code_error,
        center,
    )
end

"""
    get_carrier_loop_stage(state) -> CarrierLoopStage

The [`CarrierLoopStage`](@ref) of a scalar loop's per-satellite state, or
of a [`VectorTrackingLoop`](@ref)'s: `VECTOR_LOOP` while the satellite is in
the vector loop, the scalar loop's stage out of it.
"""
get_carrier_loop_stage(state::ScalarLoopState) =
    state.staging.stage

"""
    get_phase_lock_indicator(state) -> Float64

The latest phase-lock indicator of a scalar loop's per-satellite state, or of
a [`VectorTrackingLoop`](@ref)'s (in the vector loop its residual PLL's, on the prompts
it rotated; out of it the inner loop's), which the carrier-loop staging reads (see
[`CarrierLoopStage`](@ref)): `⟨I² − Q²⟩ / A²` of the driver's prompts, an estimate
of `cos 2φ` for the carrier phase error `φ` (1 in lock, 0 for a spinning phase),
averaged over 0.1 s or 25 integrations, whichever is longer. The signal power
`A² = √(2 M₂² − M₄)`, from the prompts' moments `Mₖ = ⟨|P|ᵏ⟩`, is free of noise, so
unlike `⟨I² − Q²⟩ / ⟨I² + Q²⟩` a locked loop's reading gets noisier at low C/N₀,
not lower. `NaN` until the averages span that time.
"""
get_phase_lock_indicator(state::ScalarLoopState) =
    state.staging.lock_detector.indicator

"""
    is_phase_locked(state) -> Bool

Whether the carrier loop of a scalar loop's per-satellite state, or of a
[`VectorTrackingLoop`](@ref)'s, is phase-locked, as the loop itself decides it: a
scalar loop while its [`get_phase_lock_indicator`](@ref) reads at least 0.5, the
staging's threshold (`false` while `NaN`). In the vector loop it is the residual
PLL's lock, which also asks that the C/N₀ the rotated prompts' moments read be at
most 6 dB below the loop's `min_tracking_cn0`, as on bare noise the indicator
alone reads lock now and then. Out of the vector loop, the inner loop's.
"""
is_phase_locked(state::ScalarLoopState) =
    get_phase_lock_indicator(state) >= _PHASE_LOCK_THRESHOLD

"""
    calc_carrier_phase_correction(state, nco_history, sample_index, sampling_frequency) -> Float64

The carrier phase (cycles) the loop estimates the incoming carrier to lead the
replica by at `sample_index` (of the integrations' grid at `sampling_frequency`), with
`nco_history` describing the commands the replica ran on up to there, an [`NCOTimeline`](@ref) or
a [`ConstantNCOCommand`](@ref). The replica's carrier phase plus this is the loop's
estimate of the incoming carrier phase, the carrier-phase measurement a host
reports (RINEX `L`).

A scalar loop steers its replica onto the incoming carrier, so the replica's
phase is its estimate and this is `0.0`; so is a [`VectorTrackingLoop`](@ref)'s
out of the vector loop. In it the navigation filter sets the carrier NCO and
the residual PLL follows the carrier the NCO leaves without steering it, so
the replica's phase alone drifts off the carrier, by whole cycles over time:
this is the residual PLL's prediction at `sample_index`, from its latest prompt on,
counting every cycle, never folded (`0.0` before its first prompt). Meant for
a sample index at or after the end of the driver's latest integration, as a host's
replica phase after its last integration is.

Like a scalar PLL's replica phase, it is ambiguous by half a cycle under the
Costas discriminator, and [`is_phase_locked`](@ref) says whether it can be
trusted. It steps where the loop does not follow the carrier continuously,
which a RINEX host flags in the loss-of-lock indicator:

  - by half a cycle when the integration's `carrier_phase_polarity` turns
    four-quadrant (a synced pilot) and the residual PLL finds itself half a
    cycle off the polarity's frame;
  - on a release from the vector loop, when [`get_carrier_loop_stage`](@ref)
    leaves `VECTOR_LOOP`: the correction drops to zero while the scalar loop
    pulls its replica in afresh from where the NCO left it, so the phase steps
    by the residual and may slip.

A satellite added to the vector loop keeps its phase: the residual PLL starts
at zero on the phase-locked scalar loop's replica.
"""
calc_carrier_phase_correction(
    ::ScalarLoopState,
    nco_history,
    sample_index,
    sampling_frequency,
) = 0.0

"The loop-state type a Doppler loop produces (for slot typing)."
loop_state_type(loop::AbstractTrackingLoop, driver_signal::AbstractGNSSSignal) =
    typeof(init_loop_state(loop, driver_signal, 0.0Hz, 0.0Hz))

# ── What a loop knows of the navigation solution ───────────────────────

"""
    get_navigation_solution(loop) -> Union{PVTSolution,Nothing}

The latest navigation solution (position, velocity, time, clock bias and drift, DOP,
the satellites used with their residuals, inter-system and inter-frequency biases),
or `nothing` for the scalar loops. For [`VectorTrackingLoop`](@ref) see its method.
"""
get_navigation_solution(::AbstractTrackingLoop) = nothing

"""
    get_navigation_status(loop) -> Union{VectorTrackingStatus,Nothing}

What the latest navigation cycle did ([`VectorTrackingStatus`](@ref)), or `nothing` for a
loop without one.
"""
get_navigation_status(::AbstractTrackingLoop) = nothing

"""
    get_navigation_cycle_index(loop) -> Union{Int,Nothing}

Number of navigation cycles run, or `nothing` without them. Advances with each
attempted cycle and its [`get_navigation_status`](@ref).
An unsuccessful positioning attempt retains the previous [`get_navigation_solution`](@ref),
so a new cycle index does not by itself indicate a new fix.
"""
get_navigation_cycle_index(::AbstractTrackingLoop) = nothing

"""
    get_navigation_elapsed_time(loop) -> Union{typeof(1.0s),Nothing}

Elapsed time of the latest navigation cycle, in seconds since the shared origin
of the integrations' sample grid. Returns `nothing` before the first cycle or for a loop
without navigation cycles. This advances even when a positioning attempt fails and
the previous solution is retained. Read `get_navigation_solution(loop).time` for
the solution's absolute timestamp.
"""
get_navigation_elapsed_time(::AbstractTrackingLoop) = nothing

"""
    get_satellite_report(loop, signal, prn) -> Union{SatelliteReport,Nothing}

A [`SatelliteReport`](@ref) of satellite `prn` of `signal`, or `nothing` for the
scalar loops or a satellite never seen.
"""
get_satellite_report(::AbstractTrackingLoop, ::AbstractGNSSSignal, ::Integer) = nothing

get_decoder(::AbstractTrackingLoop, ::AbstractGNSSSignal, ::Integer) = nothing
