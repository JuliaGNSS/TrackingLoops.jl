# A closed-loop simulation of vector tracking on synthetic satellites.
#
# The ephemerides are PositionVelocityTime's decoded fixtures; everything else is
# synthesised from a chosen true trajectory, so the truth is exact: each satellite's
# true transmit time is solved from PositionVelocityTime's own range model at the
# receiver's true position and clock, and its replica — the code phase and the carrier
# Doppler the estimator commands — is propagated record by record. Every record's
# correlator is the triangle autocorrelation at the true-minus-replica code error and
# the carrier phase error, so the per-record `VectorPLLAndDLL` and the filter above it
# run on exactly what a real correlator would see, without the noise.
using PositionVelocityTime:
    _precompile_states,
    _PRECOMPILE_GPS_L1CA_STATES,
    _PRECOMPILE_GALILEO_E1B_STATES,
    calc_corrected_time,
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
    clock_offset::Float64  # corrected − uncorrected transmit time
    in_view::Bool          # false during a scripted outage
    timeline::NCOTimeline
end

# The satellite state of a replica at uncorrected transmit time `u`.
function replica_satellite_state(sat::SimSat, u)
    rate = ustrip(Hz, get_data_frequency(sat.decoder))
    elapsed = u - sat.base_tow
    num_bits = floor(Int, elapsed * rate)
    decoder = GNSSDecoderState(sat.decoder; num_bits_after_valid_syncro_sequence = num_bits)
    code_phase = (elapsed - num_bits / rate) * sat.code_frequency
    SatelliteState(; decoder, system = sat.signal, code_phase, carrier_doppler = sat.carrier_doppler * Hz)
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
        t_t = t - ρ[1] / TrackingLoops.SPEED_OF_LIGHT
    end
    sat.transmit_time = t_t
    t_t, SVector{3,Float64}(position), SVector{3,Float64}(velocity),
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
function SimSat(signal, decoder, estimator, truth::SimTruth, t; delayed = false)
    base_tow = Float64(get_time_of_week(decoder))
    code_frequency = Float64(ustrip(Hz, get_code_frequency(signal)))
    center_frequency = Float64(ustrip(Hz, get_center_frequency(signal)))
    ratio = get_code_center_frequency_ratio(signal)
    probe = SimSat(signal, decoder, base_tow, code_frequency, center_frequency,
        nothing, 0.0, 0.0, 0.0, 0.0, complex(0.0), t - 0.075, 0.0, true, NCOTimeline())
    t_t, = true_transmit(probe, truth, t)
    doppler = true_doppler(probe, truth, t)
    state = init_estimator_state(estimator, signal, doppler * Hz, doppler * ratio * Hz)
    sat = SimSat(signal, decoder, base_tow, code_frequency, center_frequency, state,
        t_t, 0.0, doppler, doppler * ratio, complex(0.0), t_t, 0.0, true, NCOTimeline())
    # Uncorrected transmit time of the replica: invert the clock correction.
    u = t_t
    for _ = 1:3
        u += t_t - calc_corrected_time(replica_satellite_state(sat, u))
    end
    sat.replica_time = u
    sat.clock_offset = t_t - u
    reset_timeline!(sat.timeline, doppler, doppler * ratio)
    sat
end

triangle(x) = max(0.0, 1.0 - abs(x))

# One record of `num_samples` samples ending at device sample `sample_index`
# (receiver time `t_end`): the correlator from the errors at the record's centre,
# the per-record loop step, the replica propagated.
function simulate_record!(sat::SimSat, estimator, truth, t_end, num_samples, sample_index, landing_sample)
    fs = ustrip(Hz, SIM_FS)
    dt = num_samples / fs
    t_mid = t_end - dt / 2
    record_start = sample_index - num_samples
    words = landing_sample == NO_LANDING_SAMPLE ? FixedNCOWord(sat.carrier_doppler, sat.code_doppler) : sat.timeline
    carrier, code = mean_nco_word(words, record_start, sample_index)
    t_t, = true_transmit(sat, truth, t_mid)
    f_true = true_doppler(sat, truth, t_mid)
    u_true = t_t - sat.clock_offset
    u_replica = sat.replica_time + dt / 2 * (1 + code / sat.code_frequency)
    code_error = (u_true - u_replica) * sat.code_frequency
    mean_phase = sat.phase_error + (f_true - carrier) * dt / 2
    correlator_template = EarlyPromptLateCorrelator(SVector{3,ComplexF64}(0, 0, 0), 0.5)
    d = TrackingLoops.get_early_late_sample_spacing(correlator_template, SIM_FS, get_code_frequency(sat.signal)) *
        sat.code_frequency / fs
    amplitude = sat.in_view ? 1.0 : 0.0
    rotation = cis(2π * mean_phase)
    # Out of view only a trace of noise is left, never an exact zero the
    # discriminators would divide by.
    floor = sat.in_view ? 0.0 : 1e-3
    prompt = amplitude * triangle(code_error) * rotation + floor * cis(1e3 * t_end)
    early = amplitude * triangle(code_error - d / 2) * rotation + floor * cis(2e3 * t_end)
    late = amplitude * triangle(code_error + d / 2) * rotation + floor * cis(3e3 * t_end)
    correlator = EarlyPromptLateCorrelator(SVector{3,ComplexF64}(late, prompt, early), 0.5)
    record = LoopRecord(sat.signal, correlator, sat.previous_prompt, num_samples,
        sample_index, sample_index, 1, SIM_FS)
    sat.state, new_carrier, new_code = step_loop(estimator, sat.state, record, words, landing_sample)
    sat.previous_prompt = prompt
    sat.phase_error += (f_true - carrier) * dt
    sat.replica_time += dt * (1 + code / sat.code_frequency)
    if landing_sample == NO_LANDING_SAMPLE
        sat.carrier_doppler = ustrip(Hz, new_carrier)
        sat.code_doppler = ustrip(Hz, new_code)
    else
        schedule_word!(sat.timeline, landing_sample, ustrip(Hz, new_carrier), ustrip(Hz, new_code))
        promote_words!(sat.timeline, record_start)
        sat.carrier_doppler, sat.code_doppler = nco_word_at(sat.timeline, sample_index)
    end
    code_error
end

# Fill a `VTSat` from a channel at the cycle epoch (device sample `epoch_sample`,
# receiver time `t`), with the command computed now landing at `landing_sample`.
function fill_vtsat!(vtsat::VTSat, sat::SimSat, epoch_sample, landing_sample; in_lock = sat.in_view)
    fs = ustrip(Hz, SIM_FS)
    epoch_state = replica_satellite_state(sat, sat.replica_time)
    vtsat.active = true
    vtsat.decoder = epoch_state.decoder
    vtsat.estimator_state = sat.state
    vtsat.code_phase = epoch_state.code_phase
    carrier, code = landing_sample == NO_LANDING_SAMPLE ?
        (sat.carrier_doppler, sat.code_doppler) : nco_word_at(sat.timeline, epoch_sample)
    vtsat.carrier_doppler = carrier * Hz
    vtsat.code_doppler = code * Hz
    if landing_sample == NO_LANDING_SAMPLE
        vtsat.landing_lead = 0.0s
        vtsat.code_phase_at_landing = epoch_state.code_phase
        vtsat.carrier_doppler_at_landing = carrier * Hz
    else
        lead = (landing_sample - epoch_sample) / fs
        # The replica's code phase at landing under the words already committed.
        u = sat.replica_time
        step = 1000
        for a = epoch_sample:step:(landing_sample-1)
            b = min(a + step, landing_sample)
            _, word_code = mean_nco_word(sat.timeline, a, b)
            u += (b - a) / fs * (1 + word_code / sat.code_frequency)
        end
        vtsat.landing_lead = lead * s
        # The code phase at landing counts on from the epoch's.
        vtsat.code_phase_at_landing =
            epoch_state.code_phase + (u - sat.replica_time) * sat.code_frequency
        vtsat.carrier_doppler_at_landing = first(nco_word_at(sat.timeline, landing_sample)) * Hz
    end
    vtsat.cn0_dbhz = 45.0
    vtsat.coherent_integration_time = record_ms(sat.signal) * 1.0ms
    vtsat.early_late_spacing =
        TrackingLoops.get_early_late_sample_spacing(
            EarlyPromptLateCorrelator(SVector{3,ComplexF64}(0, 0, 0), 0.5),
            SIM_FS,
            get_code_frequency(sat.signal),
        ) * sat.code_frequency / fs
    vtsat.in_lock = in_lock
    vtsat.pvt_ready = in_lock
    vtsat
end

# The receiver of a simulation: per signal group the channels and their `VTSat` slots,
# and the filter. `sats` and `group` are the first group's.
struct SimReceiver{CH<:Tuple,G<:Tuple,V}
    channels::CH
    groups::G
    vt::V
    truth::SimTruth
    estimator::VectorPLLAndDLL
    cycle_ms::Int
    delay_ms::Int
end

function Base.getproperty(receiver::SimReceiver, name::Symbol)
    name === :sats && return getfield(receiver, :channels)[1]
    name === :group && return getfield(receiver, :groups)[1]
    getfield(receiver, name)
end

# The record length of a signal: one code period, in milliseconds.
record_ms(signal) = round(Int, 1000 * get_code_length(signal) / ustrip(Hz, get_code_frequency(signal)))

function SimReceiver(;
    signals = (GPSL1CA(),),
    estimator = VectorPLLAndDLL(),
    config = VectorTracking(),
    num_sats = typemax(Int),
    records_per_cycle = 100, # cycle length in milliseconds
    delay_records = 0,       # NCO delay in milliseconds
    truth_kw = (;),
)
    fixtures = map(fixture_decoders, signals)
    gps_states = last(fixture_decoders(GPSL1CA()))
    fix = calc_pvt(SignalGroup(GPSL1CA(), gps_states); approximate_year = 2021)
    t0 = maximum(maximum(calc_corrected_time, last(f)) for f in fixtures) + 0.075
    truth = SimTruth(; position = SVector(fix.position.x, fix.position.y, fix.position.z), t0, truth_kw...)
    channels = map(signals, fixtures) do signal, (decoders, _)
        n = min(num_sats, length(decoders))
        [SimSat(signal, decoders[i], estimator, truth, t0) for i = 1:n]
    end
    groups = map(signals, channels) do signal, sats
        VTSignalGroup(signal, [VTSat(sat.decoder, sat.state; prn = sat.decoder.prn) for sat in sats])
    end
    vt = VectorTrackingState(config, groups; approximate_year = 2021,
        enable_ionospheric_correction = false, enable_tropospheric_correction = false,
        integration_time = records_per_cycle * 1.0ms)
    SimReceiver(channels, groups, vt, truth, estimator, records_per_cycle, delay_records)
end

const SAMPLES_PER_MS = 4000

sim_time(receiver::SimReceiver, sample) = receiver.truth.t0 + sample / ustrip(Hz, SIM_FS)

# Run `num_cycles` navigation cycles from device sample `start_sample`; `outage(cycle,
# sat_index)` scripts which of the first group's satellites lose their signal (every
# group's, for `outage_all`). `fill!` fills a `VTSat` at the cycle epoch. Returns per
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
    fill! = fill_vtsat!,
)
    results = []
    sample = start_sample
    cycle_time = receiver.cycle_ms * 1.0ms
    num_channels = sum(length, receiver.channels)
    for cycle = 1:num_cycles
        code_errors = zeros(num_channels)
        for (g, channels) in enumerate(receiver.channels), (i, sat) in enumerate(channels)
            sat.in_view = !((g == 1 || outage_all) && outage(cycle + cycle_offset, i))
        end
        for k = 1:receiver.cycle_ms
            sample_index = sample + k * SAMPLES_PER_MS
            landing = receiver.delay_ms == 0 ? NO_LANDING_SAMPLE :
                Int64(sample_index + receiver.delay_ms * SAMPLES_PER_MS)
            c = 0
            for channels in receiver.channels, sat in channels
                c += 1
                n = record_ms(sat.signal)
                k % n == 0 || continue
                code_errors[c] = simulate_record!(sat, receiver.estimator, receiver.truth,
                    sim_time(receiver, sample_index), n * SAMPLES_PER_MS, sample_index, landing)
            end
            # A diverged loop ends the run: the replica has left the signal.
            all(e -> abs(e) < 1, code_errors) || return results, sample, true
        end
        sample += receiver.cycle_ms * SAMPLES_PER_MS
        landing = receiver.delay_ms == 0 ? NO_LANDING_SAMPLE :
            Int64(sample + receiver.delay_ms * SAMPLES_PER_MS)
        for (channels, group) in zip(receiver.channels, receiver.groups), (i, sat) in enumerate(channels)
            fill!(group.sats[i], sat, sample, landing)
        end
        pvt, status = update_navigation!(receiver.vt, receiver.groups, uconvert(s, cycle_time))
        for (channels, group) in zip(receiver.channels, receiver.groups), (i, sat) in enumerate(channels)
            sat.state = group.sats[i].estimator_state
        end
        # The solution's containers are reused by the next cycle, so what is read later
        # is copied out now.
        push!(results, (; pvt, status, code_errors, time = sim_time(receiver, sample),
            reasons = [v.release_reason for group in receiver.groups for v in group.sats],
            measured = collect(keys(pvt.sats)),
            max_rate_residual = maximum(v -> abs(v.rate_residual), values(pvt.sats); init = 0.0u"m/s")))
    end
    results, sample, false
end

position_error(receiver::SimReceiver, result) =
    norm(SVector(result.pvt.position.x, result.pvt.position.y, result.pvt.position.z) -
         truth_position(receiver.truth, result.time))
