# A closed-loop simulation of vector tracking on synthetic satellites.
#
# The ephemerides are PositionVelocityTime's decoded fixtures; everything else is
# synthesised from a chosen true trajectory, so the truth is exact: each satellite's
# true transmit time is solved from PositionVelocityTime's own range model at the
# receiver's true position and clock, and its replica — the code phase and the carrier
# Doppler the tracking_loop commands — is propagated integration by integration. Every integration's
# correlator bank is the triangle autocorrelation at the true-minus-replica code error and
# the carrier phase error, so the per-integration `VectorTrackingLoop` and the filter above it
# run on exactly what a real correlator would see, without the noise. Optionally the
# true ranges carry the atmospheric delays of PositionVelocityTime's models and a
# per-signal hardware delay, an inter-frequency bias.
#
# Two harnesses drive the tracking_loop. The first fills its navigation processor by hand at
# every epoch from the fixture decoders, for the update itself on every constellation;
# the second (the pipeline, at the end) hands it nothing but integrations, and GPS L1 C/A
# satellites broadcasting real navigation bits.
using TrackingLoopFilters: ThirdOrderAssistedBilinearLF
using PositionVelocityTime:
    _precompile_states,
    _PRECOMPILE_GPS_L1CA_STATES,
    _PRECOMPILE_GALILEO_E1B_STATES,
    _precompile_cnav,
    predict_atmospheric_delays,
    prepare_observation,
    prepare_observations,
    calc_transmit_time,
    IonosphericModel,
    correct_clock,
    calc_satellite_position_and_velocity,
    calc_satellite_clock_drift,
    predict_pseudoranges!,
    BiasColumns,
    SignalObservation,
    calc_pvt
using GNSSDecoder:
    GNSSDecoder,
    GNSSDecoderState,
    get_broadcast_ephemeris,
    get_broadcast_time_of_week,
    get_group_delay
import PositionVelocityTime
using Geodesy: ECEF
using LinearAlgebra: norm
using Random: Xoshiro, randn

include("lnav_encoder.jl")

const SIM_FS = 4e6Hz

# The fixture decoders of one constellation, and the scalar fix of their own
# capture, whose position the simulated receiver starts from.
function fixture_decoders(signal::GPSL1CA)
    states = _precompile_states(signal, _PRECOMPILE_GPS_L1CA_STATES, identity, GPSL1CA())
    [state.decoder for state in states], states
end
function fixture_decoders(signal::GalileoE1B)
    states = _precompile_states(signal, _PRECOMPILE_GALILEO_E1B_STATES, identity, GalileoE1B())
    [state.decoder for state in states], states
end
# GPS L2CM on the GPS L1 C/A fixture satellites, with CNAV data carrying their ephemerides.
function fixture_decoders(signal::GPSL2CM)
    states = _precompile_states(signal, _PRECOMPILE_GPS_L1CA_STATES, _precompile_cnav, GPSL1CA())
    [state.decoder for state in states], states
end

# The receiver's truth: a straight line at constant velocity and a clock bias
# drifting at a constant rate, both in metres — optionally with one jerk-limited
# manoeuvre: from `maneuver_start` seconds after `t0` on, an acceleration
# `maneuver_acceleration · sin²(π τ / maneuver_duration)` over `maneuver_duration`,
# after which the receiver drives on at its new velocity.
Base.@kwdef struct SimTruth
    position::SVector{3,Float64}
    velocity::SVector{3,Float64} = SVector(8.0, -5.0, 2.0)
    clock_bias::Float64 = 1500.0
    clock_drift::Float64 = 30.0
    t0::Float64 # receiver time of week at the start
    maneuver_start::Float64 = Inf
    maneuver_duration::Float64 = 4.0
    maneuver_acceleration::SVector{3,Float64} = zero(SVector{3,Float64})
end

# The manoeuvre's velocity and position offsets `τ` seconds into it.
function maneuver_offsets(truth::SimTruth, τ)
    τ <= 0 && return zero(SVector{3,Float64}), zero(SVector{3,Float64})
    D = truth.maneuver_duration
    A = truth.maneuver_acceleration
    s = min(τ, D)
    dv = A * (s / 2 - D / (4π) * sin(2π * s / D))
    dp = A * (s^2 / 4 + D^2 / (8π^2) * (cos(2π * s / D) - 1))
    τ > D && (dp += dv * (τ - D))
    dv, dp
end

function truth_position(truth::SimTruth, t)
    _, dp = maneuver_offsets(truth, t - truth.t0 - truth.maneuver_start)
    truth.position + truth.velocity * (t - truth.t0) + dp
end
function truth_velocity(truth::SimTruth, t)
    dv, _ = maneuver_offsets(truth, t - truth.t0 - truth.maneuver_start)
    truth.velocity + dv
end
truth_clock_bias(truth::SimTruth, t) = truth.clock_bias + truth.clock_drift * (t - truth.t0)

# One simulated channel.
# The atmosphere the true ranges pass through: PositionVelocityTime's ionospheric
# `correction` (held as `Any`, the way `predict_atmospheric_delays` takes it) and the
# start of the GPS week the receiver times count in, which dates the models, or none at
# all.
struct SimAtmosphere
    enabled::Bool
    correction::Any
    week_start::TAITime
end
const NO_ATMOSPHERE = SimAtmosphere(false, nothing, TAITime(0, 0.0))

# The atmosphere over the fixture satellites of `signals`, dated by a GPS fixture `state`
# at time of week `t`.
function sim_atmosphere(signals, fixtures, state, t)
    measurement = prepare_observation(state, 2021)
    correction = prepare_observations(map(last, fixtures); approximate_year = 2021).ionospheric_model.parameters
    SimAtmosphere(true, correction, get_week_start_epoch(measurement.time_system, measurement.week))
end

mutable struct SimSat{D,E,S,L}
    signal::S
    decoder::D
    const base_tow::Float64
    const chip_rate::Float64
    const center_frequency::Float64
    state::E
    # The replica: uncorrected transmit time, carrier phase error against the signal
    # (cycles), the commands it runs on.
    replica_time::Float64
    phase_error::Float64
    carrier_doppler::Float64
    code_doppler::Float64
    previous_prompt::ComplexF64
    transmit_time::Float64 # the latest true transmit time, a warm start
    in_view::Bool          # false during a scripted outage
    amplitude::Float64     # of the signal, scripted to fade it
    timeline::NCOTimeline
    const range_bias::Float64 # a hardware delay on this signal (m)
    const atmosphere::SimAtmosphere
    # For the integrations the tracking_loop reads by itself: the navigation bits the signal
    # carries (none: a bare code), the correlator noise (σ of each component of a
    # unit-amplitude prompt per millisecond; zero: none), and the device sample the
    # next integration ends at.
    stream::Union{Nothing,LNAVStream}
    noise_std::Float64
    rng::Xoshiro
    next_end_sample::Int
    # For the pipeline, which steps the tracking_loop as a host does: the signal's own
    # state — bit clock, C/N₀ estimator, post-correlation filter — and the noise density its
    # noise-referenced C/N₀ estimator divides by (1/Hz; that of the unit-amplitude
    # prompt at the channel's C/N₀).
    loop::L
    const noise_density::Float64
end

# The atmospheric delay (m) of a satellite at `position`, seen from the true receiver
# position `r` at receiver time `t`: PositionVelocityTime's ionospheric and
# tropospheric models, which the filter corrects with.
function atmospheric_delay(sat::SimSat, r, position, t)
    sat.atmosphere.enabled || return 0.0
    measurements = [(; position, center_frequency = sat.center_frequency)]
    only(predict_atmospheric_delays([r[1], r[2], r[3], 0.0], measurements,
        IonosphericModel(sat.atmosphere.correction), sat.atmosphere.week_start + t, true))
end

# The satellite state of a replica at uncorrected transmit time `u`.
function replica_satellite_state(sat::SimSat, u)
    rate = ustrip(Hz, get_bit_rate(sat.decoder))
    elapsed = u - sat.base_tow
    num_bits = floor(Int, elapsed * rate)
    decoder = GNSSDecoderState(sat.decoder; num_bits_since_time_reference = num_bits)
    code_phase = (elapsed - num_bits / rate) * sat.chip_rate
    SignalObservation(; decoder, signal = sat.signal, code_phase, carrier_doppler = sat.carrier_doppler * Hz)
end

# The satellite's broadcast ephemeris and clock; every fixture is decoded in full.
ephemeris(sat::SimSat) = something(get_broadcast_ephemeris(sat.decoder))

# The true transmit time solved from `t_t` on for a receiver clock reading `t` (seconds
# of week), the satellite position and velocity there, and the range PositionVelocityTime
# models (m): the Earth-rotation-corrected geometric range plus the receiver clock bias.
function solve_transmit(sat::SimSat, truth::SimTruth, t, t_t)
    r = truth_position(truth, t)
    ξ = [r[1], r[2], r[3], truth_clock_bias(truth, t)]
    columns = BiasColumns([1], 1, [0], 0)
    ρ = [0.0]
    position = velocity = zero(SVector{3,Float64})
    for _ = 1:4
        orbit = calc_satellite_position_and_velocity(ephemeris(sat), t_t)
        position, velocity = orbit.position, orbit.velocity
        predict_pseudoranges!(ρ, [SVector{3,Float64}(position)], ξ, columns)
        delay = sat.range_bias + atmospheric_delay(sat, r, SVector{3,Float64}(position), t)
        t_t = t - (ρ[1] + delay) / TrackingLoops.SPEED_OF_LIGHT
    end
    t_t, SVector{3,Float64}(position), SVector{3,Float64}(velocity), ρ[1]
end

# The true corrected transmit time, satellite position, velocity and clock drift for
# a receiver clock reading `t` (seconds of week).
function true_transmit(sat::SimSat, truth::SimTruth, t)
    t_t, position, velocity, _ = solve_transmit(sat, truth, t, sat.transmit_time)
    sat.transmit_time = t_t
    t_t, position, velocity, calc_satellite_clock_drift(ephemeris(sat), t_t)
end

# The true carrier Doppler (Hz) at receiver time `t`: the rate of the range the code is
# generated from — the central difference of the modelled range — plus the satellite
# clock drift. Not the filter's own rate model, so the carrier the integrations see stays
# consistent with their code wherever that model falls short.
function true_doppler(sat::SimSat, truth::SimTruth, t; h = 0.01)
    _, _, _, ρ_after = solve_transmit(sat, truth, t + h, sat.transmit_time)
    _, _, _, ρ_before = solve_transmit(sat, truth, t - h, sat.transmit_time)
    t_t, = solve_transmit(sat, truth, t, sat.transmit_time)
    rate = -(ρ_after - ρ_before) / 2h +
           calc_satellite_clock_drift(ephemeris(sat), t_t) * TrackingLoops.SPEED_OF_LIGHT
    rate / (TrackingLoops.SPEED_OF_LIGHT / sat.center_frequency)
end

# A channel locked onto the truth at receiver time `t`: the replica on the true
# transmit time, carrier phase and Dopplers.
function SimSat(signal, decoder, tracking_loop, truth::SimTruth, t; range_bias = 0.0,
        atmosphere = NO_ATMOSPHERE, stream = nothing, cn0_dbhz = Inf, seed = decoder.prn,
        cn0_estimator = default_cn0_estimator(signal, 100),
        driver_carrier_phase_offset = get_carrier_phase_offset(signal))
    base_tow = Float64(get_broadcast_time_of_week(decoder))
    chip_rate = Float64(ustrip(Hz, get_chip_rate(signal)))
    center_frequency = Float64(ustrip(Hz, get_center_frequency(signal)))
    ratio = get_chip_rate_to_center_frequency_ratio(signal)
    noise_std = isinf(cn0_dbhz) ? 0.0 : sqrt(1 / (2 * 10^(cn0_dbhz / 10) * 1e-3))
    noise_density = 10^(-cn0_dbhz / 10)
    probe = SimSat(signal, decoder, base_tow, chip_rate, center_frequency,
        nothing, 0.0, 0.0, 0.0, 0.0, complex(0.0), t - 0.075, true, 1.0, NCOTimeline(),
        range_bias, atmosphere, stream, noise_std, Xoshiro(seed), 0, nothing, noise_density)
    t_t, = true_transmit(probe, truth, t)
    doppler = true_doppler(probe, truth, t)
    state = init_loop_state(tracking_loop, signal, doppler * Hz, doppler * ratio * Hz)
    sat = SimSat(signal, decoder, base_tow, chip_rate, center_frequency, state,
        t_t, 0.0, doppler, doppler * ratio, complex(0.0), t_t, true, 1.0, NCOTimeline(),
        range_bias, atmosphere, stream, noise_std, Xoshiro(seed), 0,
        # The simulated correlator is early-prompt-late for every signal, so the state is built for it.
        PostCorrelationState(signal, decoder.prn; cn0_estimator, driver_carrier_phase_offset,
            correlator_bank = EarlyPromptLateCorrelatorBank(SVector{3,ComplexF64}(0, 0, 0), 0.5)), noise_density)
    sat.replica_time = uncorrected_time(sat, t_t)
    reset_timeline!(sat.timeline, doppler, doppler * ratio)
    sat
end

# The satellite clock's own reading — the uncorrected transmit time its signal carries —
# at the corrected transmit time `t_t`: the broadcast clock correction inverted. It is
# inverted at every instant, not once: the correction drifts (`a_f1`, the relativistic
# term) by up to a few millimetres of range per second, and a satellite clock frozen at
# its initial offset is a range error growing linearly over the run, different for
# every satellite, which a geometry without redundancy amplifies by its DOP into a drift
# of the solution.
function uncorrected_time(sat::SimSat, t_t)
    u = t_t
    for _ = 1:3
        u += t_t - correct_clock(ephemeris(sat), something(get_group_delay(sat.decoder, sat.signal), 0.0), u)
    end
    u
end

triangle(x) = max(0.0, 1.0 - abs(x))

# The correlator bank of one integration of `num_samples` samples ending at device sample
# `sample_index` (receiver time `t_end`), from the errors at the integration's centre, and the
# commands the replica ran on: the triangle autocorrelation at the true-minus-replica code
# error, rotated by the carrier phase error, signed by the navigation bit and with the
# noise of the channel. Returns the correlator bank, the prompt, the code error and the true
# Doppler and the commands.
function simulate_correlator(sat::SimSat, truth, t_end, num_samples, sample_index, nco_history)
    fs = ustrip(Hz, SIM_FS)
    dt = num_samples / fs
    t_mid = t_end - dt / 2
    integration_start_sample_index = sample_index - num_samples
    nco_command = calc_mean_nco_command(nco_history, integration_start_sample_index, sample_index)
    carrier, code = nco_command.carrier_doppler, nco_command.code_doppler
    t_t, = true_transmit(sat, truth, t_mid)
    f_true = true_doppler(sat, truth, t_mid)
    u_true = uncorrected_time(sat, t_t)
    u_replica = sat.replica_time + dt / 2 * (1 + code / sat.chip_rate)
    code_error = (u_true - u_replica) * sat.chip_rate
    mean_phase = sat.phase_error + (f_true - carrier) * dt / 2
    correlator_template = EarlyPromptLateCorrelatorBank(SVector{3,ComplexF64}(0, 0, 0), 0.5)
    d = TrackingLoops.calc_early_late_sample_spacing(correlator_template, SIM_FS, get_chip_rate(sat.signal)) *
        sat.chip_rate / fs
    bit = isnothing(sat.stream) || iszero(get_bit_rate(sat.signal)) ? true :
        lnav_bit(sat.stream, floor(Int, u_true * 50))
    amplitude = sat.in_view ? (bit ? sat.amplitude : -sat.amplitude) : 0.0
    rotation = cis(2π * mean_phase)
    # Out of view only a trace of noise is left, never an exact zero the
    # discriminators would divide by.
    floor_ = sat.in_view ? 0.0 : 1e-3
    σ = sat.noise_std * sqrt(1e-3 / dt)
    noise() = σ == 0 ? 0.0im : σ * complex(randn(sat.rng), randn(sat.rng))
    prompt = amplitude * triangle(code_error) * rotation + floor_ * cis(1e3 * t_end) + noise()
    early = amplitude * triangle(code_error - d / 2) * rotation + floor_ * cis(2e3 * t_end) + noise()
    late = amplitude * triangle(code_error + d / 2) * rotation + floor_ * cis(3e3 * t_end) + noise()
    correlator_bank = EarlyPromptLateCorrelatorBank(SVector{3,ComplexF64}(late, prompt, early), 0.5)
    correlator_bank, prompt, code_error, f_true, carrier, code
end

# The replica moved on by an integration, and the new commands commanded.
function advance_replica!(sat::SimSat, f_true, carrier, code, dt, new_carrier, new_code,
        integration_start_sample_index, sample_index, application_sample_index)
    sat.phase_error += (f_true - carrier) * dt
    sat.replica_time += dt * (1 + code / sat.chip_rate)
    if application_sample_index == nothing
        sat.carrier_doppler = ustrip(Hz, new_carrier)
        sat.code_doppler = ustrip(Hz, new_code)
    else
        schedule_command!(sat.timeline, application_sample_index, ustrip(Hz, new_carrier), ustrip(Hz, new_code))
        advance_timeline!(sat.timeline, integration_start_sample_index)
        (; carrier_doppler, code_doppler) = get_nco_command(sat.timeline, sample_index)
        sat.carrier_doppler, sat.code_doppler = carrier_doppler, code_doppler
    end
end

sim_nco_history(sat::SimSat, application_sample_index) =
    application_sample_index == nothing ? ConstantNCOCommand(sat.carrier_doppler, sat.code_doppler) : sat.timeline

# One integration of `num_samples` samples ending at device sample `sample_index` (receiver
# time `t_end`), through the satellite's own loop alone — the per-integration step of the
# vector loop, with the navigation processor filled by hand at the epochs
# (`fill_slot!`). Returns the code error.
function simulate_integration!(sat::SimSat, tracking_loop, truth, t_end, num_samples, sample_index, application_sample_index;
        polarity = 0)
    nco_history = sim_nco_history(sat, application_sample_index)
    correlator_bank, prompt, code_error, f_true, carrier, code =
        simulate_correlator(sat, truth, t_end, num_samples, sample_index, nco_history)
    # The host reads the C/N₀ `fill_slot!` reports, and no signal during an outage.
    post_correlation_state = with_cn0(processed_state(sat.signal, correlator_bank, sat.previous_prompt, num_samples,
        sample_index, sample_index, 1, SIM_FS; carrier_phase_polarity = polarity), sat.in_view ? 45.0 : -Inf)
    sat.state, new_carrier, new_code =
        TrackingLoops._step_satellite(tracking_loop, sat.state, post_correlation_state, nco_history, application_sample_index)
    sat.previous_prompt = prompt
    advance_replica!(sat, f_true, carrier, code, num_samples / ustrip(Hz, SIM_FS),
        new_carrier, new_code, sample_index - num_samples, sample_index, application_sample_index)
    code_error
end

# ─────────────────────────────────────────────────────────────────────────────
# The navigation processor filled by hand: the satellites' loops step integration by integration, and at each
# epoch the harness writes their slots from the fixture decoders, runs the update and has
# every satellite apply it. This is the update the tracking_loop runs by itself (see the
# pipeline below), on decoders that need no bits: the fixtures of every constellation.

# Fill the slot of a channel at the measurement epoch (device sample `epoch_sample_index`), as the
# satellite's snapshot would: the replica at the epoch, from the fixture decoder.
function fill_slot!(slot, sat::SimSat, epoch_sample_index, application_sample_index, nav_processor; measurement_ready = sat.in_view)
    fs = ustrip(Hz, SIM_FS)
    epoch_state = replica_satellite_state(sat, sat.replica_time)
    slot.occupied = true
    slot.decoder = epoch_state.decoder
    slot.loop_state = sat.state
    sat.state = TrackingLoops._reset_discriminator_accumulators(sat.state)
    slot.code_phase = epoch_state.code_phase
    nco_command = calc_mean_nco_command(sim_nco_history(sat, application_sample_index), epoch_sample_index, epoch_sample_index)
    carrier, code = nco_command.carrier_doppler, nco_command.code_doppler
    slot.carrier_doppler = carrier * Hz
    slot.code_doppler = code * Hz
    slot.cn0_dbhz = 45.0
    slot.coherent_integration_time = integration_ms(sat.signal) * 1.0ms
    slot.early_late_spacing =
        TrackingLoops.calc_early_late_sample_spacing(
            EarlyPromptLateCorrelatorBank(SVector{3,ComplexF64}(0, 0, 0), 0.5),
            SIM_FS,
            get_chip_rate(sat.signal),
        ) * sat.chip_rate / fs
    slot.measurement_ready = measurement_ready
    slot.positioning_ready = measurement_ready
    slot.snapshot_epoch_index = nav_processor.pending_epoch_index
    slot.driver_last_end_sample_index = slot.decoding_last_end_sample_index = epoch_sample_index
    slot.driver_last_end_time = slot.decoding_last_end_time = epoch_sample_index / fs
    slot.chips_since_epoch = 0.0
    slot
end

# A channel no longer tracked: its slot has not seen an integration for longer than two
# updates, so the update drops it.
function drop_slot!(slot)
    slot.driver_last_end_time = slot.decoding_last_end_time = -Inf
    slot.snapshot_epoch_index = typemin(Int)
    slot
end

# The receiver of a simulation: per signal group the channels, and the tracking_loop whose
# navigation processor the harness fills.
struct SimReceiver{CH<:Tuple,E<:VectorTrackingLoop}
    channels::CH
    tracking_loop::E
    truth::SimTruth
    update_interval_ms::Int
    nominal_ms::Int
    delay_ms::Int
    # The integrations' carrier phase polarity: `0` (Costas), or ±1 for a dataless signal.
    polarity::Int
end

function Base.getproperty(receiver::SimReceiver, name::Symbol)
    name === :sats && return getfield(receiver, :channels)[1]
    name === :vt && return getfield(receiver, :tracking_loop).navigation_processor
    name === :groups && return getfield(receiver, :tracking_loop).navigation_processor.groups
    name === :group && return getfield(receiver, :tracking_loop).navigation_processor.groups[1]
    getfield(receiver, name)
end

# The slots of a group's channels.
channel_slots(group, channels) = group.slots[1:length(channels)]

per_signal(n::Tuple, signals) = n
per_signal(n, signals) = map(_ -> n, signals)

# The integration length of a signal: one code period, in milliseconds.
integration_ms(signal) = round(Int, 1000 * get_code_length(signal) / ustrip(Hz, get_chip_rate(signal)))

function SimReceiver(;
    signals = (GPSL1CA(),),
    inner = ScalarTrackingLoop(ThirdOrderAssistedBilinearLF),
    filter_config = NavFilterConfig(),
    num_sats = typemax(Int), # or one count per signal
    range_biases = map(_ -> 0.0, signals), # per signal (m)
    atmosphere = false, # delay the true ranges by the modelled atmosphere
    correct_atmosphere = atmosphere, # and let the filter correct them
    integrations_per_update = 100, # update interval in milliseconds
    nominal_integrations = integrations_per_update, # the tracking_loop's update interval
    delay_integrations = 0,       # NCO delay in milliseconds
    truth_kw = (;),
    tracking_loop_kw = (;),
    polarity = 0,
)
    fixtures = map(fixture_decoders, signals)
    gps_states = last(fixture_decoders(GPSL1CA()))
    fix = calc_pvt(gps_states; approximate_year = 2021).solution
    t0 = maximum(maximum(calc_transmit_time, last(f)) for f in fixtures) + 0.075
    truth = SimTruth(; position = SVector(fix.position.x, fix.position.y, fix.position.z), t0, truth_kw...)
    delays = atmosphere ? sim_atmosphere(signals, fixtures, first(gps_states), t0) : NO_ATMOSPHERE
    tracking_loop = VectorTrackingLoop(signals...; scalar_loop = inner, filter_config, approximate_year = 2021,
        navigation_filter_update_interval = nominal_integrations * 1.0ms,
        enable_ionospheric_correction = correct_atmosphere,
        enable_tropospheric_correction = correct_atmosphere, tracking_loop_kw...)
    nav_processor = tracking_loop.navigation_processor
    channels = map(signals, fixtures, per_signal(num_sats, signals), range_biases) do signal, (decoders, _), n, range_bias
        n = min(n, length(decoders))
        [SimSat(signal, decoders[i], tracking_loop, truth, t0; range_bias, atmosphere = delays) for i = 1:n]
    end
    # Every channel holds the slot of its index, registered as its first integration would.
    for (group, sats) in zip(nav_processor.groups, channels), (i, sat) in enumerate(sats)
        nav_processor.next_slot_generation += 1
        sat.state = TrackingLoops._registered(sat.state, i, nav_processor.next_slot_generation)
        slot = group.slots[i]
        slot.prn = sat.decoder.prn
        slot.occupied = true
        slot.slot_generation = nav_processor.next_slot_generation
        slot.loop_state = sat.state
        slot.decoder = sat.decoder
    end
    SimReceiver(channels, tracking_loop, truth, integrations_per_update, nominal_integrations, delay_integrations, polarity)
end

const SAMPLES_PER_MS = 4000

sim_time(receiver, sample) = receiver.truth.t0 + sample / ustrip(Hz, SIM_FS)

# An integration standing in for the one a satellite applies an update on: only its sampling
# frequency is read.
feedback_integration(sat::SimSat, sample) =
    processed_state(sat.signal, EarlyPromptLateCorrelatorBank(SVector{3,ComplexF64}(0, 1, 0), 0.5),
        complex(0.0), SAMPLES_PER_MS, sample, sample, 1, SIM_FS)

# The update at device sample `sample` (an epoch), and every satellite applying it, the
# corrections sized for where its command takes effect — or, with `ignore_delay`, as if it
# took effect at the epoch.
function run_update!(receiver::SimReceiver, sample, fill!; ignore_delay = false)
    nav_processor = receiver.vt
    application_sample_index = receiver.delay_ms == 0 ? Int64(sample) : Int64(sample + receiver.delay_ms * SAMPLES_PER_MS)
    sized_for = ignore_delay ? Int64(sample) : application_sample_index
    nav_processor.pending_epoch_index = round(Int, sample / (receiver.nominal_ms * SAMPLES_PER_MS))
    for (channels, group) in zip(receiver.channels, receiver.groups), (i, sat) in enumerate(channels)
        fill!(group.slots[i], sat, sample, receiver.delay_ms == 0 ? nothing : application_sample_index, nav_processor)
    end
    TrackingLoops._navigation_update!(nav_processor, sample / ustrip(Hz, SIM_FS))
    for (channels, group) in zip(receiver.channels, receiver.groups), (i, sat) in enumerate(channels)
        nco_history = receiver.delay_ms == 0 ? sim_nco_history(sat, nothing) : sat.timeline
        sat.state = TrackingLoops._apply_filter_feedback(nav_processor, group, group.slots[i], sat.state,
            feedback_integration(sat, sample), nco_history, sized_for)
    end
    nav_processor.pvt, nav_processor.status
end

# Run `num_updates` navigation filter updates from device sample `start_sample`; `outage(update_index,
# sat_index)` scripts which of the first group's satellites lose their signal (every
# group's, for `outage_all`). `fill!` fills a slot at the measurement epoch. Returns per
# update the solution, the status and every channel's latest code error, then the device
# sample reached, and whether some channel's code error left the correlation triangle,
# which ends the run.
function run_simulation!(
    receiver::SimReceiver,
    num_updates;
    start_sample = 0,
    outage = (c, i) -> false,
    outage_all = false,
    update_offset = 0,
    fill! = fill_slot!,
    ignore_delay = false,
)
    results = []
    sample = start_sample
    num_channels = sum(length, receiver.channels)
    for update_index = 1:num_updates
        code_errors = zeros(num_channels)
        for (g, channels) in enumerate(receiver.channels), (i, sat) in enumerate(channels)
            sat.in_view = !((g == 1 || outage_all) && outage(update_index + update_offset, i))
        end
        for k = 1:receiver.update_interval_ms
            sample_index = sample + k * SAMPLES_PER_MS
            application_sample_index = receiver.delay_ms == 0 ? nothing :
                Int64(sample_index + receiver.delay_ms * SAMPLES_PER_MS)
            c = 0
            for channels in receiver.channels, sat in channels
                c += 1
                n = integration_ms(sat.signal)
                k % n == 0 || continue
                code_errors[c] = simulate_integration!(sat, receiver.tracking_loop, receiver.truth,
                    sim_time(receiver, sample_index), n * SAMPLES_PER_MS, sample_index, application_sample_index;
                    receiver.polarity)
            end
            # A diverged loop ends the run: the replica has left the signal.
            all(e -> abs(e) < 1, code_errors) || return results, sample, true
        end
        sample += receiver.update_interval_ms * SAMPLES_PER_MS
        pvt, status = run_update!(receiver, sample, fill!; ignore_delay)
        # The solution's containers are reused by the next update, so what is read later
        # is copied out now.
        push!(results, (; pvt, status, code_errors, time = sim_time(receiver, sample),
            reasons = [slot.release_reason for (group, channels) in zip(receiver.groups, receiver.channels)
                       for slot in channel_slots(group, channels)],
            measured = collect(keys(pvt.observations)),
            max_rate_residual = maximum(v -> abs(v.rate_residual), values(pvt.observations); init = 0.0u"m/s")))
    end
    results, sample, false
end

position_error(receiver, result) =
    norm(SVector(result.pvt.position.x, result.pvt.position.y, result.pvt.position.z) -
         truth_position(receiver.truth, result.time))

# ─────────────────────────────────────────────────────────────────────────────
# The pipeline: GPS L1 C/A satellites broadcasting their fixture ephemerides as real
# LNAV bits, driven as a host drives any tracking_loop: per signal a `PostCorrelationState`
# advanced by `process_correlator_bank_output!`, an integration built from it, and `step_loop!` on every integration
# of every signal. The vector tracking_loop finds nothing but those integrations, and decodes
# the bits, solves the scalar PVT, seeds its filter and takes the satellites over by
# itself. Each integration spans one code period of the replica, so it ends on a code-period
# boundary to within a sample, and reports the replica's code phase there.
#
# With `pilot = true` every satellite is a pilot + data pair: the tracking_loop ranges on
# a GPS L1C-P pilot (a 10 ms code, its overlay taken as wiped off) and decodes the
# LNAV bits of the satellite's GPS L1 C/A signal, whose replica follows the pilot's
# commands, as a passenger's does. Both codes start at the same instant, as a pair's
# do. L1 C/A is not L1C-P's data component, but it shares its chip rate and stands in
# for one with the bits at hand — and its code is ten times shorter than the pilot's.

# GNSSDecoder pairs an L1C-P range with the L1C-D message; the stand-in pair ranges it
# with the LNAV message, whose group delay is the L1 C/A one. A signal instance builds
# its codes, so the one asked is built once.
const LNAV_GROUP_DELAY_SIGNAL = GPSL1CA()
GNSSDecoder.get_group_delay(
    decoder::GNSSDecoderState{<:GNSSDecoder.GPSLNAVData},
    ::GPSL1C_P,
) = get_group_delay(decoder, LNAV_GROUP_DELAY_SIGNAL)

struct PipelineReceiver{E,S,P}
    sats::Vector{S}
    # Per satellite its passenger, the data component, when the satellites are pairs.
    passengers::Vector{P}
    tracking_loop::E
    truth::SimTruth
    delay_ms::Int
    # How late the drivers' and the passengers' integrations reach the tracking_loop (ms).
    driver_lag_ms::Int
    passenger_lag_ms::Int
    # Whether the host drains the soft bits after every integration, or once per millisecond
    # (once per call, as Tracking.jl does).
    drain_per_integration::Bool
    # The receiver time (s) from which the noise reference is ready.
    noise_ready_after::Float64
    # The device sample each signal's last integration ended at.
    last_ends::Vector{Int}
    passenger_last_ends::Vector{Int}
    # The bytes `step_loop!` allocated while `measuring`.
    measuring::Base.RefValue{Bool}
    allocated::Base.RefValue{Int}
end

Base.getproperty(receiver::PipelineReceiver, name::Symbol) =
    name === :vt ? getfield(receiver, :tracking_loop).navigation_processor : getfield(receiver, name)

# Satellites whose receiver starts `lead` seconds before a subframe 1, so the whole
# clock and ephemeris is decoded `lead + 18` seconds in. `pilot` sets the receiver's
# type; a trimmed app passes it as a `Val`, so the type is known when it compiles.
PipelineReceiver(; pilot = false, kw...) = PipelineReceiver(Val(pilot); kw...)

function PipelineReceiver(
    ::Val{pilot};
    num_sats = typemax(Int),
    inner = ScalarTrackingLoop(ThirdOrderAssistedBilinearLF),
    filter_config = NavFilterConfig(),
    delay_integrations = 0,
    cn0_dbhz = 45.0,
    lead = 8.0,
    tracking_loop = nothing,
    tracking_loop_kw = (;),
    cn0_estimator = () -> default_cn0_estimator(GPSL1CA(), 100),
    driver_lag_ms = 0,
    passenger_lag_ms = 0,
    drain_per_integration = true,
    noise_ready_after = -Inf,
) where {pilot}
    decoders, states = fixture_decoders(GPSL1CA())
    fix = calc_pvt(states; approximate_year = 2021).solution
    t0 = next_subframe1_start(maximum(calc_transmit_time, states)) - lead
    truth = SimTruth(; position = SVector(fix.position.x, fix.position.y, fix.position.z), t0)
    ranging = pilot ? (GPSL1C_P() => GPSL1CA(),) : (GPSL1CA(),)
    tracking_loop = something(tracking_loop, VectorTrackingLoop(ranging...; scalar_loop = inner, filter_config, approximate_year = 2021,
        enable_ionospheric_correction = false, enable_tropospheric_correction = false,
        tracking_loop_kw...))
    decoders = decoders[1:min(num_sats, end)]
    driver_signal = pilot ? GPSL1C_P() : GPSL1CA()
    sats = [SimSat(driver_signal, decoder, tracking_loop, truth, t0; cn0_dbhz,
        stream = LNAVStream(decoder.data), cn0_estimator = cn0_estimator()) for decoder in decoders]
    passengers = if pilot
        [SimSat(GPSL1CA(), decoder, tracking_loop, truth, t0; cn0_dbhz, seed = decoder.prn + 100,
            stream = LNAVStream(decoder.data), cn0_estimator = cn0_estimator(),
            driver_carrier_phase_offset = get_carrier_phase_offset(driver_signal)) for decoder in decoders]
    else
        similar(sats, 0)
    end
    for sat in Iterators.flatten((sats, passengers))
        sat.next_end_sample = next_code_period_end(sat, 0)
    end
    PipelineReceiver(sats, passengers, tracking_loop, truth,
        delay_integrations, driver_lag_ms, passenger_lag_ms, drain_per_integration, noise_ready_after,
        zeros(Int, length(sats)), zeros(Int, length(passengers)), Ref(false), Ref(0))
end

# The device sample the replica completes its current code period at, for an integration
# starting at `sample`.
function next_code_period_end(sat::SimSat, sample)
    code_length = get_code_length(sat.signal)
    phase = mod(sat.replica_time * sat.chip_rate, code_length)
    chips = code_length - phase
    chips < 0.5 && (chips += code_length)
    rate = sat.chip_rate + sat.code_doppler
    sample + max(1, round(Int, chips / rate * ustrip(Hz, SIM_FS)))
end

# One integration of a pipeline signal, ending at its `next_end_sample`, through the host's
# steps: `process_correlator_bank_output!` on the signal's state, the integration built from it, `step_loop!`
# with the satellite's tracking_loop state — `owner`'s, the driver's — and the returned
# Dopplers applied. A driver's commands are applied to its `passenger` too, whose
# replica follows them; a passenger's own integration returns the command in force.
function pipeline_integration!(receiver::PipelineReceiver, sat::SimSat, start_sample, owner::SimSat = sat,
        passenger::Union{Nothing,SimSat} = nothing)
    sample_index = sat.next_end_sample
    num_samples = sample_index - start_sample
    application_sample_index = receiver.delay_ms == 0 ? nothing :
        Int64(sample_index + receiver.delay_ms * SAMPLES_PER_MS)
    nco_history = sim_nco_history(sat, application_sample_index)
    correlator_bank, _, code_error, f_true, carrier, code =
        simulate_correlator(sat, receiver.truth, sim_time(receiver, sample_index), num_samples, sample_index, nco_history)
    dt = num_samples / ustrip(Hz, SIM_FS)
    code_phase = mod((sat.replica_time + dt * (1 + code / sat.chip_rate)) * sat.chip_rate,
        get_code_length(sat.signal))
    # The correlator bank as a device reports it: summed over the integration's samples, and a
    # passenger's prompt in its own carrier-phase frame, which `process_correlator_bank_output!` turns
    # back onto the driver's.
    driver_phase_offset = get_carrier_phase_offset(owner.signal)
    rotation = cispi(2 * (get_carrier_phase_offset(sat.signal) - driver_phase_offset))
    raw = EarlyPromptLateCorrelatorBank(
        get_correlator_outputs(correlator_bank) .* (rotation * num_samples * get_code_amplitude(sat.signal)), 0.5)
    output = CorrelatorBankOutput(raw, num_samples, sample_index, code_phase)
    noise_ready = sim_time(receiver, sample_index) - receiver.truth.t0 >= receiver.noise_ready_after
    sat.loop = process_correlator_bank_output!(sat.loop, output, SIM_FS, sat.noise_density / Hz, noise_ready)
    owner.state, new_carrier, new_code = if receiver.measuring[]
        step_measured!(receiver.allocated, receiver.tracking_loop, owner.state, sat.loop, nco_history, application_sample_index)
    else
        step_loop!(receiver.tracking_loop, owner.state, sat.loop, nco_history, application_sample_index)
    end
    receiver.drain_per_integration && empty!(sat.loop.bit_synchronizer.soft_bits)
    if sat === owner
        advance_replica!(sat, f_true, carrier, code, dt, new_carrier, new_code,
            start_sample, sample_index, application_sample_index)
        isnothing(passenger) || follow_driver!(passenger, new_carrier, new_code, application_sample_index)
    else
        # A passenger's command is the one in force: nothing to schedule.
        sat.phase_error += (f_true - carrier) * dt
        sat.replica_time += dt * (1 + code / sat.chip_rate)
        application_sample_index == nothing || advance_timeline!(sat.timeline, start_sample)
        if application_sample_index != nothing
            (; carrier_doppler, code_doppler) = get_nco_command(sat.timeline, sample_index)
            sat.carrier_doppler, sat.code_doppler = carrier_doppler, code_doppler
        end
    end
    sat.next_end_sample = next_code_period_end(sat, sample_index)
    code_error
end

# A passenger's replica takes the driver's command.
function follow_driver!(passenger::SimSat, new_carrier, new_code, application_sample_index)
    if application_sample_index == nothing
        passenger.carrier_doppler = ustrip(Hz, new_carrier)
        passenger.code_doppler = ustrip(Hz, new_code)
    else
        schedule_command!(passenger.timeline, application_sample_index, ustrip(Hz, new_carrier), ustrip(Hz, new_code))
    end
    nothing
end

# `step_loop!`, adding the bytes it allocated to `allocated`.
function step_measured!(allocated, tracking_loop, state, post_correlation_state, nco_history, application_sample_index)
    bytes = @allocated result = step_loop!(tracking_loop, state, post_correlation_state, nco_history, application_sample_index)
    allocated[] += bytes
    result
end

# Every integration of satellite `i` that has reached the tracking_loop by device sample
# `sample`: the driver's and its passenger's, each `lag` late.
function step_satellite!(receiver::PipelineReceiver, i, sample)
    sat = receiver.sats[i]
    passenger = isempty(receiver.passengers) ? nothing : receiver.passengers[i]
    lag = receiver.driver_lag_ms * SAMPLES_PER_MS
    code_error = 0.0
    while sat.next_end_sample + lag <= sample
        integration_end_sample_index = sat.next_end_sample
        code_error = max(code_error, abs(pipeline_integration!(receiver, sat, receiver.last_ends[i], sat, passenger)))
        receiver.last_ends[i] = integration_end_sample_index
    end
    isnothing(passenger) && return code_error
    lag = receiver.passenger_lag_ms * SAMPLES_PER_MS
    while passenger.next_end_sample + lag <= sample
        integration_end_sample_index = passenger.next_end_sample
        code_error = max(code_error,
            abs(pipeline_integration!(receiver, passenger, receiver.passenger_last_ends[i], sat)))
        receiver.passenger_last_ends[i] = integration_end_sample_index
    end
    code_error
end

# Run the pipeline for `duration` seconds from device sample `start_sample`, the
# satellites interleaved millisecond by millisecond as a host steps them chunk by
# chunk. `outage(t, i)` scripts which satellites lose their signal at receiver time `t`
# (seconds since the start), `fade(t, i)` their signal amplitudes, and `paused(t, i)`
# which reach the tracking_loop late: their integrations wait for the pause to end. Returns the
# result of every update the tracking_loop ran, the sample reached, and whether a replica
# left its signal.
function run_pipeline!(receiver::PipelineReceiver, duration; start_sample = 0,
        outage = (t, i) -> false, fade = (t, i) -> 1.0, paused = (t, i) -> false,
        on_tick = sample -> nothing)
    results = []
    tracking_loop = receiver.tracking_loop
    sample = start_sample
    last_update = get_navigation_update_index(tracking_loop)
    code_errors = zeros(length(receiver.sats))
    end_sample = start_sample + round(Int, duration * ustrip(Hz, SIM_FS))
    while sample < end_sample
        sample += SAMPLES_PER_MS
        t = sample / ustrip(Hz, SIM_FS)
        for (i, sat) in enumerate(receiver.sats)
            sat.in_view = !outage(t, i)
            sat.amplitude = fade(t, i)
            if !isempty(receiver.passengers)
                receiver.passengers[i].in_view = sat.in_view
                receiver.passengers[i].amplitude = sat.amplitude
            end
            paused(t, i) && continue
            error = step_satellite!(receiver, i, sample)
            error > 0 && (code_errors[i] = error)
        end
        if !receiver.drain_per_integration
            for s in Iterators.flatten((receiver.sats, receiver.passengers))
                empty!(s.loop.bit_synchronizer.soft_bits)
            end
        end
        on_tick(sample)
        all(e -> abs(e) < 1, code_errors) || return results, sample, true
        update_index = get_navigation_update_index(tracking_loop)
        if update_index !== nothing && update_index != last_update
            last_update = update_index
            nav_processor = receiver.vt
            pvt = nav_processor.pvt
            push!(results, (; pvt = deepcopy(pvt), status = nav_processor.status, code_errors = copy(code_errors),
                time = receiver.truth.t0 + TrackingLoops._epoch_time(nav_processor, nav_processor.update_epoch_index),
                measured = collect(keys(pvt.observations)),
                reasons = [get_satellite_report(tracking_loop, sat.signal, sat.decoder.prn).release_reason for sat in receiver.sats]))
        end
    end
    results, sample, false
end
