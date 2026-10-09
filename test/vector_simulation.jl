# A closed-loop simulation of vector tracking on synthetic satellites.
#
# The ephemerides are PositionVelocityTime's decoded fixtures; everything else is
# synthesised from a chosen true trajectory, so the truth is exact: each satellite's
# true transmit time is solved from PositionVelocityTime's own range model at the
# receiver's true position and clock, and its replica — the code phase and the carrier
# Doppler the estimator commands — is propagated record by record. Every record's
# correlator is the triangle autocorrelation at the true-minus-replica code error and
# the carrier phase error, so the per-record `VectorPLLAndDLL` and the filter above it
# run on exactly what a real correlator would see, without the noise. Optionally the
# true ranges carry the atmospheric delays of PositionVelocityTime's models and a
# per-signal hardware delay, an inter-frequency bias.
#
# Two harnesses drive the estimator. The first fills its navigation engine by hand at
# every epoch from the fixture decoders, for the cycle itself on every constellation;
# the second (the pipeline, at the end) hands it nothing but records, and GPS L1 C/A
# satellites broadcasting real navigation bits.
using PositionVelocityTime:
    _precompile_states,
    _PRECOMPILE_GPS_L1CA_STATES,
    _PRECOMPILE_GALILEO_E1B_STATES,
    _precompile_cnav,
    day_of_year,
    predict_atmospheric_delays,
    satellite_measurement,
    select_ionospheric_correction,
    calc_corrected_time,
    correct_clock,
    calc_satellite_position_and_velocity,
    calc_satellite_clock_drift,
    calc_ρ_hat!,
    BiasColumns,
    SatelliteState,
    SignalGroup,
    calc_pvt
using GNSSDecoder: GNSSDecoderState, get_time_of_week
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
    states =
        _precompile_states(signal, _PRECOMPILE_GALILEO_E1B_STATES, identity, GalileoE1B())
    [state.decoder for state in states], states
end
# GPS L2CM on the GPS L1 C/A fixture satellites, with CNAV data carrying their ephemerides.
function fixture_decoders(signal::GPSL2CM)
    states =
        _precompile_states(signal, _PRECOMPILE_GPS_L1CA_STATES, _precompile_cnav, GPSL1CA())
    [state.decoder for state in states], states
end

# The receiver's truth: a straight line at constant velocity and a clock bias
# drifting at a constant rate, both in metres.
Base.@kwdef struct SimTruth
    position::SVector{3,Float64}
    velocity::SVector{3,Float64} = SVector(8.0, -5.0, 2.0)
    clock_bias::Float64 = 1500.0
    clock_drift::Float64 = 30.0
    t0::Float64 # receiver time of week at the start
end

truth_position(truth::SimTruth, t) = truth.position + truth.velocity * (t - truth.t0)
truth_clock_bias(truth::SimTruth, t) = truth.clock_bias + truth.clock_drift * (t - truth.t0)

# One simulated channel.
# The atmosphere the true ranges pass through: PositionVelocityTime's ionospheric
# `correction` (held as `Any`, the way `predict_atmospheric_delays` takes it) and the
# troposphere's day of year, or none at all.
struct SimAtmosphere
    enabled::Bool
    correction::Any
    doy::Int
end
const NO_ATMOSPHERE = SimAtmosphere(false, nothing, 1)

# The atmosphere over the fixture satellites of `signals`, dated by a GPS fixture `state`
# at time of week `t`.
function sim_atmosphere(signals, fixtures, state, t)
    row = satellite_measurement(state, 2021)
    correction = select_ionospheric_correction(
        map((signal, f) -> SignalGroup(signal, last(f)), signals, fixtures),
    )
    SimAtmosphere(true, correction, day_of_year(row.system_start_time, row.week, t))
end

mutable struct SimSat{D,E,S}
    signal::S
    decoder::D
    const base_tow::Float64
    const code_frequency::Float64
    const center_frequency::Float64
    state::E
    # The replica: uncorrected transmit time, carrier phase error against the signal
    # (cycles), the words it runs on.
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
    # For the records the estimator reads by itself: the navigation bits the signal
    # carries (none: a bare code), the correlator noise (σ of each component of a
    # unit-amplitude prompt per millisecond; zero: none), and the device sample the
    # next record ends at.
    stream::Union{Nothing,LNAVStream}
    noise_std::Float64
    rng::Xoshiro
    next_end_sample::Int
end

# The atmospheric delay (m) of a satellite at `position`, seen from the true receiver
# position `r` at receiver time `t`: PositionVelocityTime's ionospheric and
# tropospheric models, which the filter corrects with.
function atmospheric_delay(sat::SimSat, r, position, t)
    sat.atmosphere.enabled || return 0.0
    rows = [(; position, center_frequency = sat.center_frequency)]
    only(
        predict_atmospheric_delays(
            [r[1], r[2], r[3], 0.0],
            rows,
            sat.atmosphere.correction,
            t,
            sat.atmosphere.doy,
            true,
        ),
    )
end

# The satellite state of a replica at uncorrected transmit time `u`.
function replica_satellite_state(sat::SimSat, u)
    rate = ustrip(Hz, get_data_frequency(sat.decoder))
    elapsed = u - sat.base_tow
    num_bits = floor(Int, elapsed * rate)
    decoder = GNSSDecoderState(sat.decoder; num_bits_after_valid_syncro_sequence = num_bits)
    code_phase = (elapsed - num_bits / rate) * sat.code_frequency
    SatelliteState(;
        decoder,
        system = sat.signal,
        code_phase,
        carrier_doppler = sat.carrier_doppler * Hz,
    )
end

# The true corrected transmit time, satellite position, velocity and clock drift for
# a receiver clock reading `t` (seconds of week).
function true_transmit(sat::SimSat, truth::SimTruth, t)
    r = truth_position(truth, t)
    ξ = [r[1], r[2], r[3], truth_clock_bias(truth, t)]
    columns = BiasColumns([1], 1, [0], 0)
    ρ = [0.0]
    t_t = sat.transmit_time
    position = velocity = zero(SVector{3,Float64})
    for _ = 1:4
        orbit = calc_satellite_position_and_velocity(sat.decoder, t_t)
        position, velocity = orbit.position, orbit.velocity
        calc_ρ_hat!(ρ, [SVector{3,Float64}(position)], ξ, columns)
        delay = sat.range_bias + atmospheric_delay(sat, r, SVector{3,Float64}(position), t)
        t_t = t - (ρ[1] + delay) / TrackingLoops.SPEED_OF_LIGHT
    end
    sat.transmit_time = t_t
    t_t,
    SVector{3,Float64}(position),
    SVector{3,Float64}(velocity),
    calc_satellite_clock_drift(sat.decoder, t_t)
end

# The true carrier Doppler (Hz) at receiver time `t`.
function true_doppler(sat::SimSat, truth::SimTruth, t)
    _, position, velocity, drift = true_transmit(sat, truth, t)
    rate = TrackingLoops.predict_pseudorange_rate(
        truth_position(truth, t),
        truth.velocity,
        truth.clock_drift,
        position,
        velocity,
        drift,
    )
    rate / (TrackingLoops.SPEED_OF_LIGHT / sat.center_frequency)
end

# A channel locked onto the truth at receiver time `t`: the replica on the true
# transmit time, carrier phase and Dopplers.
function SimSat(
    signal,
    decoder,
    estimator,
    truth::SimTruth,
    t;
    range_bias = 0.0,
    atmosphere = NO_ATMOSPHERE,
    stream = nothing,
    cn0_dbhz = Inf,
    seed = decoder.prn,
)
    base_tow = Float64(get_time_of_week(decoder))
    code_frequency = Float64(ustrip(Hz, get_code_frequency(signal)))
    center_frequency = Float64(ustrip(Hz, get_center_frequency(signal)))
    ratio = get_code_center_frequency_ratio(signal)
    noise_std = isinf(cn0_dbhz) ? 0.0 : sqrt(1 / (2 * 10^(cn0_dbhz / 10) * 1e-3))
    probe = SimSat(
        signal,
        decoder,
        base_tow,
        code_frequency,
        center_frequency,
        nothing,
        0.0,
        0.0,
        0.0,
        0.0,
        complex(0.0),
        t - 0.075,
        true,
        1.0,
        NCOTimeline(),
        range_bias,
        atmosphere,
        stream,
        noise_std,
        Xoshiro(seed),
        0,
    )
    t_t, = true_transmit(probe, truth, t)
    doppler = true_doppler(probe, truth, t)
    state = init_estimator_state(estimator, signal, doppler * Hz, doppler * ratio * Hz)
    sat = SimSat(
        signal,
        decoder,
        base_tow,
        code_frequency,
        center_frequency,
        state,
        t_t,
        0.0,
        doppler,
        doppler * ratio,
        complex(0.0),
        t_t,
        true,
        1.0,
        NCOTimeline(),
        range_bias,
        atmosphere,
        stream,
        noise_std,
        Xoshiro(seed),
        0,
    )
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
        u += t_t - correct_clock(sat.decoder, sat.signal, u)
    end
    u
end

triangle(x) = max(0.0, 1.0 - abs(x))

# The correlator of one record of `num_samples` samples ending at device sample
# `sample_index` (receiver time `t_end`), from the errors at the record's centre, and the
# words the replica ran on: the triangle autocorrelation at the true-minus-replica code
# error, rotated by the carrier phase error, signed by the navigation bit and with the
# noise of the channel. Returns the correlator, the prompt, the code error and the true
# Doppler and the words.
function simulate_correlator(sat::SimSat, truth, t_end, num_samples, sample_index, words)
    fs = ustrip(Hz, SIM_FS)
    dt = num_samples / fs
    t_mid = t_end - dt / 2
    record_start = sample_index - num_samples
    carrier, code = mean_nco_word(words, record_start, sample_index)
    t_t, = true_transmit(sat, truth, t_mid)
    f_true = true_doppler(sat, truth, t_mid)
    u_true = uncorrected_time(sat, t_t)
    u_replica = sat.replica_time + dt / 2 * (1 + code / sat.code_frequency)
    code_error = (u_true - u_replica) * sat.code_frequency
    mean_phase = sat.phase_error + (f_true - carrier) * dt / 2
    correlator_template = EarlyPromptLateCorrelator(SVector{3,ComplexF64}(0, 0, 0), 0.5)
    d =
        TrackingLoops.get_early_late_sample_spacing(
            correlator_template,
            SIM_FS,
            get_code_frequency(sat.signal),
        ) * sat.code_frequency / fs
    bit = isnothing(sat.stream) ? true : lnav_bit(sat.stream, floor(Int, u_true * 50))
    amplitude = sat.in_view ? (bit ? sat.amplitude : -sat.amplitude) : 0.0
    rotation = cis(2π * mean_phase)
    # Out of view only a trace of noise is left, never an exact zero the
    # discriminators would divide by.
    floor_ = sat.in_view ? 0.0 : 1e-3
    σ = sat.noise_std * sqrt(1e-3 / dt)
    noise() = σ == 0 ? 0.0im : σ * complex(randn(sat.rng), randn(sat.rng))
    prompt =
        amplitude * triangle(code_error) * rotation + floor_ * cis(1e3 * t_end) + noise()
    early =
        amplitude * triangle(code_error - d / 2) * rotation +
        floor_ * cis(2e3 * t_end) +
        noise()
    late =
        amplitude * triangle(code_error + d / 2) * rotation +
        floor_ * cis(3e3 * t_end) +
        noise()
    correlator = EarlyPromptLateCorrelator(SVector{3,ComplexF64}(late, prompt, early), 0.5)
    correlator, prompt, code_error, f_true, carrier, code
end

# The replica moved on by a record, and the new words commanded.
function advance_replica!(
    sat::SimSat,
    f_true,
    carrier,
    code,
    dt,
    new_carrier,
    new_code,
    record_start,
    sample_index,
    landing_sample,
)
    sat.phase_error += (f_true - carrier) * dt
    sat.replica_time += dt * (1 + code / sat.code_frequency)
    if landing_sample == NO_LANDING_SAMPLE
        sat.carrier_doppler = ustrip(Hz, new_carrier)
        sat.code_doppler = ustrip(Hz, new_code)
    else
        schedule_word!(
            sat.timeline,
            landing_sample,
            ustrip(Hz, new_carrier),
            ustrip(Hz, new_code),
        )
        promote_words!(sat.timeline, record_start)
        sat.carrier_doppler, sat.code_doppler = nco_word_at(sat.timeline, sample_index)
    end
end

sim_words(sat::SimSat, landing_sample) =
    landing_sample == NO_LANDING_SAMPLE ?
    FixedNCOWord(sat.carrier_doppler, sat.code_doppler) : sat.timeline

# One record of `num_samples` samples ending at device sample `sample_index` (receiver
# time `t_end`), through the satellite's own loop alone — the per-record step of the
# vector loop, with the navigation engine filled by hand at the epochs
# (`fill_slot!`). Returns the code error.
function simulate_record!(
    sat::SimSat,
    estimator,
    truth,
    t_end,
    num_samples,
    sample_index,
    landing_sample,
)
    words = sim_words(sat, landing_sample)
    correlator, prompt, code_error, f_true, carrier, code =
        simulate_correlator(sat, truth, t_end, num_samples, sample_index, words)
    record = LoopRecord(
        sat.signal,
        correlator,
        sat.previous_prompt,
        num_samples,
        sample_index,
        sample_index,
        1,
        SIM_FS,
    )
    sat.state, new_carrier, new_code =
        TrackingLoops._step_satellite(estimator, sat.state, record, words, landing_sample)
    sat.previous_prompt = prompt
    advance_replica!(
        sat,
        f_true,
        carrier,
        code,
        num_samples / ustrip(Hz, SIM_FS),
        new_carrier,
        new_code,
        sample_index - num_samples,
        sample_index,
        landing_sample,
    )
    code_error
end

# ─────────────────────────────────────────────────────────────────────────────
# The engine filled by hand: the satellites' loops step record by record, and at each
# epoch the harness writes their slots from the fixture decoders, runs the cycle and has
# every satellite take it up. This is the cycle the estimator runs by itself (see the
# pipeline below), on decoders that need no bits: the fixtures of every constellation.

# Fill the slot of a channel at the cycle epoch (device sample `epoch_sample`), as the
# satellite's snapshot would: the replica at the epoch, from the fixture decoder.
function fill_slot!(
    slot,
    sat::SimSat,
    epoch_sample,
    landing_sample,
    nav;
    in_lock = sat.in_view,
)
    fs = ustrip(Hz, SIM_FS)
    epoch_state = replica_satellite_state(sat, sat.replica_time)
    slot.occupied = true
    slot.decoder = epoch_state.decoder
    slot.estimator_state = sat.state
    sat.state = TrackingLoops._reset_discriminator_accumulators(sat.state)
    slot.code_phase = epoch_state.code_phase
    carrier, code =
        mean_nco_word(sim_words(sat, landing_sample), epoch_sample, epoch_sample)
    slot.carrier_doppler = carrier * Hz
    slot.code_doppler = code * Hz
    slot.cn0_dbhz = 45.0
    slot.coherent_integration_time = record_ms(sat.signal) * 1.0ms
    slot.early_late_spacing =
        TrackingLoops.get_early_late_sample_spacing(
            EarlyPromptLateCorrelator(SVector{3,ComplexF64}(0, 0, 0), 0.5),
            SIM_FS,
            get_code_frequency(sat.signal),
        ) * sat.code_frequency / fs
    slot.in_lock = in_lock
    slot.pvt_ready = in_lock
    slot.snapshot_epoch = nav.pending_epoch
    slot.last_end_sample = epoch_sample
    slot.last_end_time = epoch_sample / fs
    slot.chips_since_epoch = 0.0
    slot
end

# A channel no longer tracked: its slot has not seen a record for longer than two
# cycles, so the cycle drops it.
function drop_slot!(slot)
    slot.last_end_time = -Inf
    slot.snapshot_epoch = typemin(Int)
    slot
end

# The receiver of a simulation: per signal group the channels, and the estimator whose
# navigation engine the harness fills.
struct SimReceiver{CH<:Tuple,E<:VectorPLLAndDLL}
    channels::CH
    estimator::E
    truth::SimTruth
    cycle_ms::Int
    nominal_ms::Int
    delay_ms::Int
end

function Base.getproperty(receiver::SimReceiver, name::Symbol)
    name === :sats && return getfield(receiver, :channels)[1]
    name === :vt && return getfield(receiver, :estimator).navigation
    name === :groups && return getfield(receiver, :estimator).navigation.groups
    name === :group && return getfield(receiver, :estimator).navigation.groups[1]
    getfield(receiver, name)
end

# The slots of a group's channels.
channel_slots(group, channels) = group.slots[1:length(channels)]

per_signal(n::Tuple, signals) = n
per_signal(n, signals) = map(_ -> n, signals)

# The record length of a signal: one code period, in milliseconds.
record_ms(signal) =
    round(Int, 1000 * get_code_length(signal) / ustrip(Hz, get_code_frequency(signal)))

function SimReceiver(;
    signals = (GPSL1CA(),),
    inner = ConventionalAssistedPLLAndDLL(),
    config = VectorTracking(),
    num_sats = typemax(Int), # or one count per signal
    range_biases = map(_ -> 0.0, signals), # per signal (m)
    atmosphere = false, # delay the true ranges by the modelled atmosphere
    correct_atmosphere = atmosphere, # and let the filter correct them
    records_per_cycle = 100, # cycle length in milliseconds
    nominal_records = records_per_cycle, # the estimator's cycle time
    delay_records = 0,       # NCO delay in milliseconds
    truth_kw = (;),
)
    fixtures = map(fixture_decoders, signals)
    gps_states = last(fixture_decoders(GPSL1CA()))
    fix = calc_pvt(SignalGroup(GPSL1CA(), gps_states); approximate_year = 2021)
    t0 = maximum(maximum(calc_corrected_time, last(f)) for f in fixtures) + 0.075
    truth = SimTruth(;
        position = SVector(fix.position.x, fix.position.y, fix.position.z),
        t0,
        truth_kw...,
    )
    delays =
        atmosphere ? sim_atmosphere(signals, fixtures, first(gps_states), t0) :
        NO_ATMOSPHERE
    estimator = VectorPLLAndDLL(
        signals...;
        inner,
        config,
        approximate_year = 2021,
        cycle_time = nominal_records * 1.0ms,
        enable_ionospheric_correction = correct_atmosphere,
        enable_tropospheric_correction = correct_atmosphere,
    )
    nav = estimator.navigation
    channels = map(
        signals,
        fixtures,
        per_signal(num_sats, signals),
        range_biases,
    ) do signal, (decoders, _), n, range_bias
        n = min(n, length(decoders))
        [
            SimSat(
                signal,
                decoders[i],
                estimator,
                truth,
                t0;
                range_bias,
                atmosphere = delays,
            ) for i = 1:n
        ]
    end
    # Every channel holds the slot of its index, registered as its first record would.
    for (group, sats) in zip(nav.groups, channels), (i, sat) in enumerate(sats)
        nav.registrations += 1
        sat.state = TrackingLoops._registered(sat.state, i, nav.registrations)
        slot = group.slots[i]
        slot.prn = sat.decoder.prn
        slot.occupied = true
        slot.registration = nav.registrations
        slot.estimator_state = sat.state
        slot.decoder = sat.decoder
    end
    SimReceiver(
        channels,
        estimator,
        truth,
        records_per_cycle,
        nominal_records,
        delay_records,
    )
end

const SAMPLES_PER_MS = 4000

sim_time(receiver, sample) = receiver.truth.t0 + sample / ustrip(Hz, SIM_FS)

# A record standing in for the one a satellite takes a cycle up on: only its sampling
# frequency is read.
take_up_record(sat::SimSat, sample) = LoopRecord(
    sat.signal,
    EarlyPromptLateCorrelator(SVector{3,ComplexF64}(0, 1, 0), 0.5),
    complex(0.0),
    SAMPLES_PER_MS,
    sample,
    sample,
    1,
    SIM_FS,
)

# The cycle at device sample `sample` (an epoch), and every satellite taking it up, the
# corrections sized for where its command lands — or, with `ignore_delay`, as if it
# landed at the epoch.
function run_cycle!(receiver::SimReceiver, sample, fill!; ignore_delay = false)
    nav = receiver.vt
    landing =
        receiver.delay_ms == 0 ? Int64(sample) :
        Int64(sample + receiver.delay_ms * SAMPLES_PER_MS)
    sized_for = ignore_delay ? Int64(sample) : landing
    nav.pending_epoch = round(Int, sample / (receiver.nominal_ms * SAMPLES_PER_MS))
    for (channels, group) in zip(receiver.channels, receiver.groups),
        (i, sat) in enumerate(channels)

        fill!(
            group.slots[i],
            sat,
            sample,
            receiver.delay_ms == 0 ? NO_LANDING_SAMPLE : landing,
            nav,
        )
    end
    TrackingLoops._navigation_cycle!(nav, sample / ustrip(Hz, SIM_FS))
    for (channels, group) in zip(receiver.channels, receiver.groups),
        (i, sat) in enumerate(channels)

        words = receiver.delay_ms == 0 ? sim_words(sat, NO_LANDING_SAMPLE) : sat.timeline
        sat.state = TrackingLoops._take_up_cycle(
            nav,
            group,
            group.slots[i],
            sat.state,
            take_up_record(sat, sample),
            words,
            sized_for,
        )
    end
    nav.pvt, nav.status
end

# Run `num_cycles` navigation cycles from device sample `start_sample`; `outage(cycle,
# sat_index)` scripts which of the first group's satellites lose their signal (every
# group's, for `outage_all`). `fill!` fills a slot at the cycle epoch. Returns per
# cycle the solution, the status and every channel's latest code error, then the device
# sample reached, and whether some channel's code error left the correlation triangle,
# which ends the run.
function run_simulation!(
    receiver::SimReceiver,
    num_cycles;
    start_sample = 0,
    outage = (c, i) -> false,
    outage_all = false,
    cycle_offset = 0,
    fill! = fill_slot!,
    ignore_delay = false,
)
    results = []
    sample = start_sample
    num_channels = sum(length, receiver.channels)
    for cycle = 1:num_cycles
        code_errors = zeros(num_channels)
        for (g, channels) in enumerate(receiver.channels), (i, sat) in enumerate(channels)
            sat.in_view = !((g == 1 || outage_all) && outage(cycle + cycle_offset, i))
        end
        for k = 1:receiver.cycle_ms
            sample_index = sample + k * SAMPLES_PER_MS
            landing =
                receiver.delay_ms == 0 ? NO_LANDING_SAMPLE :
                Int64(sample_index + receiver.delay_ms * SAMPLES_PER_MS)
            c = 0
            for channels in receiver.channels, sat in channels
                c += 1
                n = record_ms(sat.signal)
                k % n == 0 || continue
                code_errors[c] = simulate_record!(
                    sat,
                    receiver.estimator,
                    receiver.truth,
                    sim_time(receiver, sample_index),
                    n * SAMPLES_PER_MS,
                    sample_index,
                    landing,
                )
            end
            # A diverged loop ends the run: the replica has left the signal.
            all(e -> abs(e) < 1, code_errors) || return results, sample, true
        end
        sample += receiver.cycle_ms * SAMPLES_PER_MS
        pvt, status = run_cycle!(receiver, sample, fill!; ignore_delay)
        # The solution's containers are reused by the next cycle, so what is read later
        # is copied out now.
        push!(
            results,
            (;
                pvt,
                status,
                code_errors,
                time = sim_time(receiver, sample),
                reasons = [
                    slot.release_reason for
                    (group, channels) in zip(receiver.groups, receiver.channels) for
                    slot in channel_slots(group, channels)
                ],
                measured = collect(keys(pvt.sats)),
                max_rate_residual = maximum(
                    v -> abs(v.rate_residual),
                    values(pvt.sats);
                    init = 0.0u"m/s",
                ),
            ),
        )
    end
    results, sample, false
end

position_error(receiver, result) = norm(
    SVector(result.pvt.position.x, result.pvt.position.y, result.pvt.position.z) -
    truth_position(receiver.truth, result.time),
)

# ─────────────────────────────────────────────────────────────────────────────
# The pipeline: GPS L1 C/A satellites broadcasting their fixture ephemerides as real
# LNAV bits, driven through `step_loop` alone. The estimator finds the bit edges,
# decodes the bits, solves the scalar PVT, seeds its filter and takes the satellites
# over by itself. Each record spans one code period of the replica, so it ends on a
# code-block boundary to within a sample, and reports the replica's code phase there.

struct PipelineReceiver{E<:VectorPLLAndDLL,S}
    sats::Vector{S}
    estimator::E
    truth::SimTruth
    delay_ms::Int
    # The device sample each satellite's last record ended at.
    last_ends::Vector{Int}
    # The bytes `step_loop` allocated while `measuring`.
    measuring::Base.RefValue{Bool}
    allocated::Base.RefValue{Int}
end

Base.getproperty(receiver::PipelineReceiver, name::Symbol) =
    name === :vt ? getfield(receiver, :estimator).navigation : getfield(receiver, name)

# Satellites whose receiver starts `lead` seconds before a subframe 1, so the whole
# clock and ephemeris is decoded `lead + 18` seconds in.
function PipelineReceiver(;
    num_sats = typemax(Int),
    inner = ConventionalAssistedPLLAndDLL(),
    config = VectorTracking(),
    delay_records = 0,
    cn0_dbhz = 45.0,
    lead = 8.0,
    estimator_kw = (;),
)
    decoders, states = fixture_decoders(GPSL1CA())
    fix = calc_pvt(SignalGroup(GPSL1CA(), states); approximate_year = 2021)
    t0 = next_subframe1_start(maximum(calc_corrected_time, states)) - lead
    truth =
        SimTruth(; position = SVector(fix.position.x, fix.position.y, fix.position.z), t0)
    estimator = VectorPLLAndDLL(
        GPSL1CA();
        inner,
        config,
        approximate_year = 2021,
        enable_ionospheric_correction = false,
        enable_tropospheric_correction = false,
        estimator_kw...,
    )
    sats = [
        SimSat(
            GPSL1CA(),
            decoder,
            estimator,
            truth,
            t0;
            cn0_dbhz,
            stream = LNAVStream(decoder.data),
        ) for decoder in decoders[1:min(num_sats, end)]
    ]
    for sat in sats
        sat.next_end_sample = next_block_end(sat, 0)
    end
    PipelineReceiver(
        sats,
        estimator,
        truth,
        delay_records,
        zeros(Int, length(sats)),
        Ref(false),
        Ref(0),
    )
end

# The device sample the replica completes its current code period at, for a record
# starting at `sample`.
function next_block_end(sat::SimSat, sample)
    code_length = get_code_length(sat.signal)
    phase = mod(sat.replica_time * sat.code_frequency, code_length)
    chips = code_length - phase
    chips < 0.5 && (chips += code_length)
    rate = sat.code_frequency + sat.code_doppler
    sample + max(1, round(Int, chips / rate * ustrip(Hz, SIM_FS)))
end

# One record of a pipeline satellite, ending at its `next_end_sample`.
function pipeline_record!(receiver::PipelineReceiver, sat::SimSat, start_sample)
    sample_index = sat.next_end_sample
    num_samples = sample_index - start_sample
    landing =
        receiver.delay_ms == 0 ? NO_LANDING_SAMPLE :
        Int64(sample_index + receiver.delay_ms * SAMPLES_PER_MS)
    words = sim_words(sat, landing)
    correlator, prompt, code_error, f_true, carrier, code = simulate_correlator(
        sat,
        receiver.truth,
        sim_time(receiver, sample_index),
        num_samples,
        sample_index,
        words,
    )
    dt = num_samples / ustrip(Hz, SIM_FS)
    code_phase = mod(
        (sat.replica_time + dt * (1 + code / sat.code_frequency)) * sat.code_frequency,
        get_code_length(sat.signal),
    )
    record = LoopRecord(
        sat.signal,
        correlator,
        sat.previous_prompt,
        num_samples,
        sample_index,
        sample_index,
        1,
        SIM_FS;
        prn = sat.decoder.prn,
        code_phase,
    )
    sat.state, new_carrier, new_code = if receiver.measuring[]
        step_measured!(
            receiver.allocated,
            receiver.estimator,
            sat.state,
            record,
            words,
            landing,
        )
    else
        step_loop(receiver.estimator, sat.state, record, words, landing)
    end
    sat.previous_prompt = prompt
    advance_replica!(
        sat,
        f_true,
        carrier,
        code,
        dt,
        new_carrier,
        new_code,
        start_sample,
        sample_index,
        landing,
    )
    sat.next_end_sample = next_block_end(sat, sample_index)
    code_error
end

# `step_loop`, adding the bytes it allocated to `allocated`.
function step_measured!(allocated, estimator, state, record, words, landing)
    bytes = @allocated result = step_loop(estimator, state, record, words, landing)
    allocated[] += bytes
    result
end

# Run the pipeline for `duration` seconds from device sample `start_sample`, the
# satellites interleaved millisecond by millisecond as a host steps them chunk by
# chunk. `outage(t, i)` scripts which satellites lose their signal at receiver time `t`
# (seconds since the start), and `fade(t, i)` their signal amplitudes. Returns the
# result of every cycle the estimator ran, the sample reached, and whether a replica
# left its signal.
function run_pipeline!(
    receiver::PipelineReceiver,
    duration;
    start_sample = 0,
    outage = (t, i) -> false,
    fade = (t, i) -> 1.0,
    on_tick = sample -> nothing,
    record! = pipeline_record!,
    driver_signal = GPSL1CA(),
)
    results = []
    nav = receiver.vt
    sample = start_sample
    last_cycle = nav.cycle_id
    code_errors = zeros(length(receiver.sats))
    end_sample = start_sample + round(Int, duration * ustrip(Hz, SIM_FS))
    while sample < end_sample
        sample += SAMPLES_PER_MS
        t = sample / ustrip(Hz, SIM_FS)
        for (i, sat) in enumerate(receiver.sats)
            sat.in_view = !outage(t, i)
            sat.amplitude = fade(t, i)
            while sat.next_end_sample <= sample
                record_end = sat.next_end_sample
                code_errors[i] = record!(receiver, sat, receiver.last_ends[i])
                receiver.last_ends[i] = record_end
            end
        end
        on_tick(sample)
        all(e -> abs(e) < 1, code_errors) || return results, sample, true
        if nav.cycle_id != last_cycle
            last_cycle = nav.cycle_id
            pvt = nav.pvt
            push!(
                results,
                (;
                    pvt = deepcopy(pvt),
                    status = nav.status,
                    code_errors = copy(code_errors),
                    time = receiver.truth.t0 +
                           TrackingLoops._epoch_time(nav, nav.cycle_epoch),
                    measured = collect(keys(pvt.sats)),
                    reasons = [
                        release_reason(receiver.estimator, driver_signal, sat.decoder.prn)
                        for sat in receiver.sats
                    ],
                ),
            )
        end
    end
    results, sample, false
end
