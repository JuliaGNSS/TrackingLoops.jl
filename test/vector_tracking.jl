# The navigation cycle, cycle by cycle, on the synthetic satellites of
# `vector_simulation.jl` with the engine filled by hand: seeding from the scalar fix,
# the loop closure, membership and release, and the solution.
using PositionVelocityTime: predict_pseudoranges!, PVTSolution
using Geodesy: ENUfromECEF, wgs84

@testset "The tracking_loop's navigation engine" begin
    tracking_loop = VectorTrackingLoop(GPSL1CA(), GalileoE1B(); satellite_capacity_per_signal = 4)
    nav = tracking_loop.engine
    @test length(nav.groups) == 2
    @test nav.groups[1].driver_signal isa GPSL1CA
    @test all(group -> length(group.slots) == 4, nav.groups)
    @test all(group -> all(slot -> !slot.occupied && slot.slot_generation == 0, group.slots), nav.groups)
    @test nav.navigation_update_interval == 0.1s
    @test nav.min_tracking_cn0_dbhz == 30.0
    @test get_navigation_status(tracking_loop) == NavigationStatus()
    @test !get_navigation_status(tracking_loop).vector_tracking_active
    @test get_navigation_solution(tracking_loop) === nav.pvt
    @test isempty(get_vector_tracking_members(tracking_loop))
    @test get_release_reason(tracking_loop, GPSL1CA(), 5) == VT_NOT_RELEASED
    state = init_loop_state(tracking_loop, GPSL1CA(), 100.0Hz, 0.1Hz)
    @test (state.slot_index, state.slot_generation, state.cycle_index) == (0, 0, -1)
    @test !state.vector_controlled
    @test VectorTrackingLoop(GPSL1CA(); navigation_update_interval = 20ms).engine.navigation_update_interval == 0.02s
    @test VectorTrackingLoop(GPSL1CA(); min_tracking_cn0 = 35dBHz).engine.min_tracking_cn0_dbhz == 35.0
    @test !VectorTrackingLoop(GPSL1CA(); filter_config = nothing).engine.enabled
end

@testset "The tracking_loop rejects what it cannot track" begin
    @test_throws ArgumentError VectorTrackingLoop()
    @test_throws ArgumentError VectorTrackingLoop(GPSL1CA(), GPSL1CA())
    # A dataless pilot carries no bits to decode.
    @test_throws ArgumentError VectorTrackingLoop(GPSL1C_P())
    @test_throws ArgumentError VectorTrackingLoop(GalileoE1C())
    @test_throws ArgumentError VectorTrackingLoop(GPSL1CA(); navigation_update_interval = 0.0s)
    @test_throws ArgumentError VectorTrackingLoop(GPSL1CA(); navigation_update_interval = -0.1s)
    @test_throws ArgumentError VectorTrackingLoop(GPSL1CA(); navigation_update_interval = Inf * s)
    @test_throws ArgumentError VectorTrackingLoop(GPSL1CA(); satellite_capacity_per_signal = 0)
    # An integration of no satellite, or of a driver it was not built for.
    tracking_loop = VectorTrackingLoop(GPSL1CA())
    state = init_loop_state(tracking_loop, GPSL1CA(), 100.0Hz, 0.1Hz)
    correlator_bank = EarlyPromptLateCorrelatorBank(SVector{3,ComplexF64}(0.5, 1.0, 0.5), 0.5)
    output = CorrelatorBankOutput(correlator_bank, 4000, 4000, 0.0)
    nco_history = ConstantNCOCommand(100.0, 0.1)
    @test_throws ArgumentError step_loop!(tracking_loop, state,
        PredetectionIntegration(GPSL1CA(), correlator_bank, complex(0.0), output, 1, 4e6Hz), nco_history, nothing)
    galileo = init_loop_state(tracking_loop, GalileoE1B(), 100.0Hz, 0.1Hz)
    @test_throws ArgumentError step_loop!(tracking_loop, galileo,
        PredetectionIntegration(GalileoE1B(), correlator_bank, complex(0.0), output, 1, 4e6Hz; prn = 3), nco_history, nothing)
    # A passenger it does not decode is the host's business: it is ignored.
    @test step_loop!(tracking_loop, state,
        PredetectionIntegration(GalileoE1B(), correlator_bank, complex(0.0), output, 1, 4e6Hz; prn = 3), nco_history,
        nothing) == (state, 100.0Hz, 0.1Hz)
end

@testset "Without a configuration only the scalar PVT is solved" begin
    rx = SimReceiver(; filter_config = nothing)
    @test !rx.vt.enabled
    results, _, diverged = run_simulation!(rx, 5)
    @test !diverged
    @test all(r -> !r.status.vector_tracking_active && !r.status.seeded, results)
    @test all(r -> r.status.num_vector_tracking_members == 0, results)
    @test all(sat -> !sat.state.vector_controlled, rx.sats)
    # A scalar fix each cycle, near the truth.
    @test length(results[end].pvt.observations) == length(rx.sats)
    @test position_error(rx, results[end]) < 3.0
    @test isnan(results[end].status.position_std)
end

@testset "A fresh scalar fix seeds the filter and closes the loops" begin
    rx = SimReceiver()
    vt = rx.vt
    results, _, diverged = run_simulation!(rx, 1)
    @test !diverged
    seed = results[1]
    @test seed.status.seeded
    @test seed.status.vector_tracking_active
    @test vt.vector_tracking_active
    @test seed.status.num_vector_tracking_members == length(rx.sats)
    # The emitted solution is the scalar fix, with no per-member report of the filter.
    @test length(seed.pvt.observations) == length(rx.sats)
    @test isempty(vt.vector_tracking_members)
    # Every fix satellite took itself into the loop with corrections, and joined with
    # empty accumulators to accumulate from now on.
    for (sat, slot) in zip(rx.sats, channel_slots(rx.group, rx.sats))
        @test sat.state.vector_controlled
        @test sat.state.code_disc_sum == (0, 0.0)
        @test isfinite(sat.state.code_frequency_correction)
        @test sat.state.cycle_index == vt.cycle_index
        @test slot.command_cycle_index == vt.cycle_index
        @test slot.release_reason == VT_NOT_RELEASED
    end
    # The reference epoch is the fix's, with the reference clock starting at zero.
    @test vt.reference_epoch == seed.pvt.time
    # The seeded state is the fix.
    user_pos, _, _ = TL.nav_filter_states(vt.x, vt.model.idxs)
    @test user_pos == SVector(seed.pvt.position.x, seed.pvt.position.y, seed.pvt.position.z)

    # Without a satellite ready for the scalar solve there is no fix and nothing to seed.
    rx = SimReceiver()
    for sat in rx.sats
        sat.in_view = true
    end
    fill_unready!(v, sat, epoch, application_sample_index, nav) = (fill_slot!(v, sat, epoch, application_sample_index, nav); v.positioning_ready = false)
    results, _, _ = run_simulation!(rx, 2; fill! = fill_unready!)
    @test all(r -> !r.status.vector_tracking_active && !r.status.seeded, results)
    @test all(sat -> !sat.state.vector_controlled, rx.sats)
end

@testset "A failed scalar attempt retains the previous fix without reseeding" begin
    for filter_config in (nothing, NavigationFilterConfig())
        rx = SimReceiver(; filter_config)
        _, sample, _ = run_simulation!(rx, 1)
        previous_fix = get_navigation_solution(rx.tracking_loop)
        @test previous_fix.time !== nothing
        foreach(slot -> TL._is_member(slot) && TL._mark_release!(slot, VT_FALLBACK), rx.group.slots)
        rx.vt.vector_tracking_active = false
        fill_unready!(slot, sat, epoch, application_sample_index, nav) =
            (fill_slot!(slot, sat, epoch, application_sample_index, nav); slot.positioning_ready = false)
        results, _, _ = run_simulation!(rx, 1; start_sample = sample, fill! = fill_unready!)
        @test get_navigation_solution(rx.tracking_loop) === previous_fix
        @test !only(results).status.seeded
        @test !only(results).status.vector_tracking_active
    end
end

@testset "Without an NCO delay the corrections are the residuals at the updated state" begin
    rx = SimReceiver()
    run_simulation!(rx, 3)
    vt = rx.vt
    buffers = vt.buffers
    members = buffers.members
    idxs = vt.model.idxs
    T = 0.1
    # The corrections the cycle left, recomputed from the updated state the way
    # GNSSReceiver computed them: all members at once, from the epoch's measurements.
    ξ = TL.position_and_bias_vector(vt.x, idxs)
    predicted = predict_pseudoranges!(zeros(length(members)), [m.sat_pos for m in members], ξ,
        TL.member_bias_columns(members, vt.layout))
    user_pos, user_vel, user_drift = TL.nav_filter_states(vt.x, idxs)
    for (j, member) in enumerate(members)
        sat = rx.sats[member.slot_index]
        slot = rx.group.slots[member.slot_index]
        @test slot.member_index == j
        measured = TL.calc_pseudorange(get_time_of_week(GPST(), vt.reference_epoch), member.transmit_tow_gpst) -
                   buffers.delays[j]
        # The carrier NCO is commanded the Doppler of the rate predicted where the
        # command takes effect — at the epoch, with the satellite there; the measured rate, a
        # mean over the past cycle, does not enter it.
        ephemeris = something(GNSSDecoder.get_broadcast_ephemeris(slot.decoder))
        orbit = calc_satellite_position_and_velocity(ephemeris, member.transmit_tow)
        rate_at_epoch = TL.calc_pseudorange_rate(user_pos, user_vel, user_drift,
            member.sat_pos, orbit.velocity, calc_satellite_clock_drift(ephemeris, member.transmit_tow))
        @test SVector{3,Float64}(orbit.position) == member.sat_pos
        # The measurement the last update fused against the one the corrections use.
        @test measured === buffers.measured_pseudoranges[j]
        # The satellite took the corrections up where its command takes effect, at the epoch:
        # the range from the orbit evaluated there is the epoch's.
        @test sat.state.code_frequency_correction ≈
              TL.nco_code_correction(predicted[j], measured, member.code_frequency, T) * Hz atol = 1e-6Hz
        @test sat.state.ramp.doppler ≈
              TL.predicted_carrier_doppler(rate_at_epoch, member.wavelength) atol = 1e-6
        @test sat.state.ramp.reference_time ≈ TL._epoch_time(vt, vt.cycle_epoch_index)
        # The post-fit rate is predicted at the measurement's own instant.
        @test buffers.predicted_pseudorange_rates[j] ≈ TL.predict_pseudorange_rate(vt.x, idxs, member) atol = 1e-9
        @test sat.state.code_command_delay == 0.0s
        # The accumulators were emptied at the epoch's snapshot.
        @test sat.state.code_disc_sum == (0, 0.0)
        # Taken up once: a second take-up changes nothing.
        @test TL._take_up_cycle(vt, rx.group, slot, sat.state, take_up_integration(sat, 0),
            sim_nco_history(sat, nothing), Int64(0)) === sat.state
    end
end

@testset "A correction sized for a later application_sample_index" begin
    rx = SimReceiver()
    _, sample, _ = run_simulation!(rx, 3)
    vt = rx.vt
    slot = rx.group.slots[1]
    sat = rx.sats[1]
    j = slot.member_index
    member = vt.buffers.members[j]
    nco_history = sim_nco_history(sat, nothing)
    integration = take_up_integration(sat, sample)
    at_epoch = TL._commands_at_application(vt, rx.group, slot, integration, nco_history, Int64(sample))
    @test at_epoch[3] == 0.0
    @test at_epoch[1] ≈ ustrip(Hz, sat.state.code_frequency_correction) atol = 1e-9
    @test at_epoch[2].doppler ≈ sat.state.ramp.doppler atol = 1e-9
    # 20 ms on, under the same commands: the range has moved on with the replica, so the
    # code correction barely changes, the ramp carries on, and the lead is the 20 ms.
    later = TL._commands_at_application(vt, rx.group, slot, integration, nco_history, Int64(sample + 80_000))
    @test later[3] ≈ 0.02
    @test later[1] ≈ at_epoch[1] atol = 0.05
    @test later[2].doppler ≈ TL._doppler_at(at_epoch[2], later[2].reference_time) atol = 1e-3
    # It must take effect within 2.5 cycles of the epoch.
    @test_throws ArgumentError TL._commands_at_application(vt, rx.group, slot, integration, nco_history,
        Int64(sample + 1_100_000))
    @test_throws ArgumentError TL._commands_at_application(vt, rx.group, slot, integration, nco_history,
        Int64(sample - 4000))
end

@testset "The solution of a running cycle" begin
    rx = SimReceiver()
    results, _, _ = run_simulation!(rx, 10)
    result = results[end]
    pvt = result.pvt
    @test result.status.vector_tracking_active
    @test !result.status.seeded
    @test pvt.time isa TAITime
    @test pvt.reference_system === GPST()
    @test pvt.dop !== nothing
    @test 0 < pvt.dop.GDOP < 10
    @test length(pvt.observations) == length(rx.sats)
    @test keys(get_vector_tracking_members(rx.tracking_loop)) == keys(pvt.observations)
    @test isempty(pvt.inter_system_biases)
    @test position_error(rx, result) < 0.5
    @test norm(SVector(pvt.velocity.x, pvt.velocity.y, pvt.velocity.z) - rx.truth.velocity) < 0.05
    @test 0.0u"m" < result.status.position_std < 10.0u"m"
    @test result.status.clock_bias_std > 0.0u"m"
    # The filter's clock drift is reported in m/s, the rate of the clock bias.
    @test ustrip(u"m/s", pvt.clock_drift) ≈ rx.truth.clock_drift atol = 0.05
    # Timestamps advance by the cycle.
    @test results[end].pvt.time - results[end-1].pvt.time ≈ 0.1 atol = 1e-6
    # The same solution object is returned and kept.
    @test pvt === get_navigation_solution(rx.tracking_loop)
    # Steered members' post-fit residuals are their discriminators alone: small.
    for info in values(pvt.observations)
        @test abs(info.residual) < 1.0u"m"
        @test abs(info.rate_residual) < 0.05u"m/s"
    end
end

@testset "Membership: out of lock coasts, a dropped satellite is released" begin
    rx = SimReceiver()
    _, sample, _ = run_simulation!(rx, 5)
    # Out of lock: still a member, unmeasured, still steered, in `get_vector_tracking_members` but
    # not in the solution's `sats`.
    prn = rx.sats[1].decoder.prn
    coast!(v, sat, epoch, application_sample_index, nav) =
        (fill_slot!(v, sat, epoch, application_sample_index, nav); v.prn == prn && (v.measurement_ready = false))
    results, sample, _ = run_simulation!(rx, 1; fill! = coast!, start_sample = sample)
    key = (:GPSL1CA, prn)
    @test rx.sats[1].state.vector_controlled
    @test get_release_reason(rx.tracking_loop, GPSL1CA(), prn) == VT_NOT_RELEASED
    @test !haskey(results[1].pvt.observations, key)
    @test haskey(get_vector_tracking_members(rx.tracking_loop), key)
    @test results[1].status.num_vector_tracking_members == length(rx.sats)
    @test !results[1].status.released
    # No integration for two cycles: dropped, released as ineligible, and re-seeded from the
    # replica's Dopplers where its command takes effect.
    dropped = rx.group.slots[2]
    drop!(v, sat, epoch, application_sample_index, nav) =
        (fill_slot!(v, sat, epoch, application_sample_index, nav); v === dropped && drop_slot!(v))
    results, _, _ = run_simulation!(rx, 1; fill! = drop!, start_sample = sample)
    sat = rx.sats[2]
    @test results[1].status.released
    @test dropped.release_reason == VT_INELIGIBLE
    @test get_release_reason(rx.tracking_loop, GPSL1CA(), dropped.prn) == VT_INELIGIBLE
    @test !dropped.occupied
    @test !dropped.loop_state.vector_controlled
    @test !sat.state.vector_controlled
    @test sat.state.code_frequency_correction == 0.0Hz
    # Re-seeded at its residual PLL's estimate, a hair off the replica's command, which
    # the filter's ramp keeps on the carrier.
    @test sat.state.scalar_state.reference_carrier_doppler ≈ sat.carrier_doppler * Hz atol = 0.01Hz
    @test sat.state.scalar_state.reference_code_doppler == sat.code_doppler * Hz
    @test results[1].status.num_vector_tracking_members == length(rx.sats) - 1
    @test !haskey(get_vector_tracking_members(rx.tracking_loop), (:GPSL1CA, dropped.prn))
    @test all(v.release_reason == VT_NOT_RELEASED for v in channel_slots(rx.group, rx.sats) if v !== dropped)
end

@testset "A member that missed the epoch sits the cycle out" begin
    rx = SimReceiver()
    _, sample, _ = run_simulation!(rx, 5)
    late = rx.group.slots[3]
    miss!(v, sat, epoch, application_sample_index, nav) =
        (fill_slot!(v, sat, epoch, application_sample_index, nav); v === late && (v.snapshot_epoch_index = typemin(Int)))
    results, _, _ = run_simulation!(rx, 1; fill! = miss!, start_sample = sample)
    # Not stale, so not dropped: it keeps its place and its corrections, without new ones.
    @test late.occupied
    @test !late.in_cycle
    @test late.release_reason == VT_NOT_RELEASED
    @test late.command_cycle_index != rx.vt.cycle_index
    @test rx.sats[3].state.vector_controlled
    @test results[1].status.num_vector_tracking_members == length(rx.sats) - 1
    @test !results[1].status.released
end

@testset "An unsolvable epoch grows the starvation timer, a solvable one pays it back" begin
    rx = SimReceiver()
    _, sample, _ = run_simulation!(rx, 3)
    # Three satellites in lock are too few for position and clock.
    starve!(v, sat, epoch, application_sample_index, nav) =
        (fill_slot!(v, sat, epoch, application_sample_index, nav); v.measurement_ready = v.prn in (rx.sats[1].decoder.prn, rx.sats[2].decoder.prn, rx.sats[3].decoder.prn))
    results, sample, _ = run_simulation!(rx, 3; fill! = starve!, start_sample = sample)
    @test [r.status.measurement_deficit_duration for r in results] ≈ [0.1, 0.2, 0.3] .* s
    @test all(r -> r.status.vector_tracking_active, results)
    results, _, _ = run_simulation!(rx, 2; start_sample = sample)
    @test [r.status.measurement_deficit_duration for r in results] ≈ [0.25, 0.2] .* s
end

@testset "A released satellite takes over from the replica at application_sample_index" begin
    # Under an NCO delay the commands committed until the application sample are the vector loop's,
    # so the scalar loop re-seeds from the replica there, not from the epoch's: the
    # code from its command, the carrier from the residual PLL's estimate on the ramp there.
    rx = SimReceiver(; inner = DelayCompensatedTrackingLoop(), integrations_per_cycle = 20, delay_integrations = 15)
    _, sample, diverged = run_simulation!(rx, 20)
    @test !diverged
    dropped = rx.group.slots[2]
    sat = rx.sats[2]
    drop!(v, sat, epoch, application_sample_index, nav) =
        (fill_slot!(v, sat, epoch, application_sample_index, nav); v === dropped && drop_slot!(v))
    _, sample, _ = run_simulation!(rx, 1; fill! = drop!, start_sample = sample)
    @test dropped.release_reason == VT_INELIGIBLE
    application_carrier, application_code = get_nco_command(sat.timeline, sample + 15 * SAMPLES_PER_MS)
    @test sat.state.scalar_state.reference_carrier_doppler ≈ application_carrier * Hz atol = 0.01Hz
    @test sat.state.scalar_state.reference_code_doppler == application_code * Hz
    @test sat.state.scalar_state.reference_carrier_doppler != dropped.carrier_doppler
end

@testset "A satellite is admitted only above the admission mask" begin
    rx = SimReceiver()
    _, sample, _ = run_simulation!(rx, 5)
    vt = rx.vt
    group = rx.group
    sat = group.slots[1]
    sat.loop_state = TL._release_from_vector_loop(sat.loop_state)
    user_pos = first(TL.nav_filter_states(vt.x, vt.model.idxs))
    here = ENUfromECEF(ECEF(user_pos...), wgs84)
    antipode = ENUfromECEF(ECEF((-user_pos)...), wgs84)
    @test TL._is_above_admission_mask(group.driver_signal, sat, here)
    @test !TL._is_above_admission_mask(group.driver_signal, sat, antipode)
    # Eligible and in lock, but below the mask: not admitted, so a satellite at the
    # horizon is not admitted and released every cycle. A member stays a member; the
    # elevation mask of the running cycle releases it.
    @test !TL._update_membership!(false, group, 1, nothing, antipode)
    @test !sat.loop_state.vector_controlled
    @test all(v -> v.loop_state.vector_controlled, channel_slots(group, rx.sats)[2:end])
    TL._update_membership!(false, group, 1, nothing, here)
    @test sat.loop_state.vector_controlled
    # A running cycle admits it back at the filter's own position, and the satellite
    # takes the admission up.
    rx.sats[1].state = TL._release_from_vector_loop(rx.sats[1].state)
    results, _, _ = run_simulation!(rx, 1; start_sample = sample)
    @test rx.sats[1].state.vector_controlled
    @test results[1].status.num_vector_tracking_members == length(rx.sats)
end

@testset "A member below the horizon is released unmeasured" begin
    # The filter's solution moved to the antipode: every satellite is below its horizon.
    # The elevation mask runs before the measurement update, at the solution the cycle
    # starts from, so they are all released without being measured, and with no member
    # left the loop falls back.
    rx = SimReceiver()
    _, sample, _ = run_simulation!(rx, 5)
    pos = rx.vt.model.idxs.pos
    rx.vt.x[pos] .= -rx.vt.x[pos]
    results, _, _ = run_simulation!(rx, 1; start_sample = sample)
    @test all(==(VT_BELOW_HORIZON), results[1].reasons)
    @test isempty(results[1].measured)
    @test results[1].status.fell_back
end

@testset "Biases are reported only for what was measured" begin
    rx = SimReceiver(; signals = (GPSL1CA(), GalileoE1B()))
    # No Galileo satellite in lock: its clock coasts and is not reported.
    gps_only!(v, sat, epoch, application_sample_index, nav) =
        (fill_slot!(v, sat, epoch, application_sample_index, nav); sat.signal isa GalileoE1B && (v.measurement_ready = false))
    results, sample, _ = run_simulation!(rx, 10; fill! = gps_only!)
    @test results[end].status.vector_tracking_active
    @test !haskey(results[end].pvt.inter_system_biases, GST())
    @test all(key -> first(key) === :GPSL1CA, keys(results[end].pvt.observations))
    results, _, _ = run_simulation!(rx, 3; start_sample = sample)
    @test haskey(results[end].pvt.inter_system_biases, GST())
end

@testset "A re-seed takes the receive epoch afresh from the fix" begin
    rx = SimReceiver()
    results, sample, _ = run_simulation!(rx, 5)
    before = results[end].pvt.time
    # The scalar solve takes over from a receive epoch a week off, which the re-seed must
    # not carry over.
    rx.vt.reference_epoch -= GNSSSignals.SECONDS_PER_WEEK
    foreach(slot -> TL._is_member(slot) && TL._mark_release!(slot, VT_FALLBACK), rx.group.slots)
    rx.vt.vector_tracking_active = false
    results, _, _ = run_simulation!(rx, 2; start_sample = sample)
    @test results[1].status.seeded
    @test results[2].status.vector_tracking_active
    @test results[2].pvt.time - before ≈ 0.2 atol = 1e-3
end

@testset "The measurement-buffer cache grows without undefined slots" begin
    rx = SimReceiver()
    run_simulation!(rx, 2)
    buffers = rx.vt.buffers
    n = length(buffers.measurement_updates)
    num_states = length(rx.vt.x)
    @test size(TL._measurement_buffers!(buffers, num_states, n + 3).z) == (n + 3,)
    @test size(TL._measurement_buffers!(buffers, num_states, n + 1).z) == (n + 1,)
    @test length(buffers.measurement_updates) == n + 3
end
