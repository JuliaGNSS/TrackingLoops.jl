# The navigation cycle, cycle by cycle, on the synthetic satellites of
# `vector_simulation.jl` with the engine filled by hand: seeding from the scalar fix,
# the loop closure, membership and release, and the solution.
using PositionVelocityTime: calc_ρ_hat!, PVTSolution, TAITime
using Geodesy: ENUfromECEF, wgs84

@testset "The estimator's navigation engine" begin
    estimator = VectorPLLAndDLL(GPSL1CA(), GalileoE1B(); max_satellites_per_signal = 4)
    nav = estimator.navigation
    @test length(nav.groups) == 2
    @test nav.groups[1].signal isa GPSL1CA
    @test all(group -> length(group.slots) == 4, nav.groups)
    @test all(group -> all(slot -> !slot.occupied && slot.registration == 0, group.slots), nav.groups)
    @test nav.cycle_time == 0.1s
    @test nav.lock_cn0_threshold == 30.0
    @test navigation_status(estimator) == VTStatus()
    @test !navigation_status(estimator).running
    @test navigation_solution(estimator) === nav.pvt
    @test isempty(member_sats(estimator))
    @test release_reason(estimator, GPSL1CA(), 5) == VT_NOT_RELEASED
    state = init_estimator_state(estimator, GPSL1CA(), 100.0Hz, 0.1Hz)
    @test (state.slot, state.registration, state.cycle_id) == (0, 0, -1)
    @test !state.vt_on
    @test VectorPLLAndDLL(GPSL1CA(); cycle_time = 20ms).navigation.cycle_time == 0.02s
    @test VectorPLLAndDLL(GPSL1CA(); lock_cn0_threshold = 35dBHz).navigation.lock_cn0_threshold == 35.0
    @test !VectorPLLAndDLL(GPSL1CA(); config = nothing).navigation.enabled
end

@testset "The estimator rejects what it cannot track" begin
    @test_throws ArgumentError VectorPLLAndDLL()
    @test_throws ArgumentError VectorPLLAndDLL(GPSL1CA(), GPSL1CA())
    # A dataless pilot carries no bits to decode.
    @test_throws ArgumentError VectorPLLAndDLL(GPSL1C_P())
    @test_throws ArgumentError VectorPLLAndDLL(GalileoE1C())
    @test_throws ArgumentError VectorPLLAndDLL(GPSL1CA(); cycle_time = 0.0s)
    @test_throws ArgumentError VectorPLLAndDLL(GPSL1CA(); cycle_time = -0.1s)
    @test_throws ArgumentError VectorPLLAndDLL(GPSL1CA(); cycle_time = Inf * s)
    @test_throws ArgumentError VectorPLLAndDLL(GPSL1CA(); max_satellites_per_signal = 0)
    # A record of a signal it was not built for, or of no satellite.
    estimator = VectorPLLAndDLL(GPSL1CA())
    state = init_estimator_state(estimator, GPSL1CA(), 100.0Hz, 0.1Hz)
    correlator = EarlyPromptLateCorrelator(SVector{3,ComplexF64}(0.5, 1.0, 0.5), 0.5)
    output = CorrelatorOutput(correlator, 4000, 4000, 0.0)
    words = FixedNCOWord(100.0, 0.1)
    @test_throws ArgumentError step_loop(estimator, state,
        LoopRecord(GPSL1CA(), correlator, complex(0.0), output, 1, 4e6Hz), words, NO_LANDING_SAMPLE)
    @test_throws ArgumentError step_loop(estimator, state,
        LoopRecord(GalileoE1B(), correlator, complex(0.0), output, 1, 4e6Hz; prn = 3), words, NO_LANDING_SAMPLE)
end

@testset "Without a configuration only the scalar PVT is solved" begin
    rx = SimReceiver(; config = nothing)
    @test !rx.vt.enabled
    results, _, diverged = run_simulation!(rx, 5)
    @test !diverged
    @test all(r -> !r.status.running && !r.status.enabled, results)
    @test all(r -> r.status.num_members == 0, results)
    @test all(sat -> !sat.state.vt_on, rx.sats)
    # A scalar fix each cycle, near the truth.
    @test length(results[end].pvt.sats) == length(rx.sats)
    @test position_error(rx, results[end]) < 3.0
    @test isnan(results[end].status.position_std)
end

@testset "A fresh scalar fix seeds the filter and closes the loops" begin
    rx = SimReceiver()
    vt = rx.vt
    results, _, diverged = run_simulation!(rx, 1)
    @test !diverged
    seed = results[1]
    @test seed.status.enabled
    @test seed.status.running
    @test vt.running
    @test seed.status.num_members == length(rx.sats)
    # The emitted solution is the scalar fix, with no per-member report of the filter.
    @test length(seed.pvt.sats) == length(rx.sats)
    @test isempty(vt.member_sats)
    # Every fix satellite took itself into the loop with corrections, and joined with
    # empty accumulators to accumulate from now on.
    for (sat, slot) in zip(rx.sats, channel_slots(rx.group, rx.sats))
        @test sat.state.vt_on
        @test sat.state.code_discr_acc == (0, 0.0)
        @test isfinite(sat.state.code_freq_update)
        @test sat.state.cycle_id == vt.cycle_id
        @test slot.correction_cycle == vt.cycle_id
        @test slot.release_reason == VT_NOT_RELEASED
    end
    # The reference epoch is the latest transmit time corrected by the fix's clock.
    members = vt.buffers.members
    latest = maximum(member.time_gpst_count for member in members)
    @test ustrip(s, vt.reference_time) ≈
          latest - ustrip(u"m", seed.pvt.time_correction) / TL.SPEED_OF_LIGHT
    # The seeded state is the fix.
    user_pos, _, _ = TL.nav_filter_states(vt.x, vt.model.idxs)
    @test user_pos == SVector(seed.pvt.position.x, seed.pvt.position.y, seed.pvt.position.z)

    # Without a satellite ready for the scalar solve there is no fix and nothing to seed.
    rx = SimReceiver()
    for sat in rx.sats
        sat.in_view = true
    end
    fill_unready!(v, sat, epoch, landing, nav) = (fill_slot!(v, sat, epoch, landing, nav); v.pvt_ready = false)
    results, _, _ = run_simulation!(rx, 2; fill! = fill_unready!)
    @test all(r -> !r.status.running && !r.status.enabled, results)
    @test all(sat -> !sat.state.vt_on, rx.sats)
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
    # GNSSReceiver computed them: all members at once, from the epoch's rows.
    ξ = TL.position_and_bias_vector(vt.x, idxs)
    predicted = calc_ρ_hat!(zeros(length(members)), [m.sat_position for m in members], ξ,
        TL.vt_bias_columns(members, vt.layout))
    user_pos, user_vel, user_drift = TL.nav_filter_states(vt.x, idxs)
    for (j, member) in enumerate(members)
        sat = rx.sats[member.slot]
        slot = rx.group.slots[member.slot]
        @test slot.member_index == j
        measured = TL.pseudorange_from_tows(ustrip(s, vt.reference_time), member.time_gpst_count) -
                   buffers.delays[j]
        rate = TL.predict_pseudorange_rate(user_pos, user_vel, user_drift,
            member.sat_position, member.sat_velocity, member.sat_clock_drift)
        # The measurement the last update fused against the one the corrections use.
        @test measured === buffers.measured_pseudoranges[j]
        # The satellite took the corrections up where its command lands, at the epoch:
        # the range from the orbit evaluated there is the epoch's.
        @test sat.state.code_freq_update ≈
              TL.nco_code_correction(predicted[j], measured, member.code_frequency, T) * Hz atol = 1e-6Hz
        @test sat.state.carrier_freq_update ≈
              TL.nco_carrier_correction(buffers.predicted_pseudorange_rates[j],
                  member.pseudorange_rate, member.wavelength) * Hz atol = 1e-6Hz
        @test buffers.predicted_pseudorange_rates[j] ≈ rate atol = 1e-9
        @test sat.state.code_update_landing_lead == 0.0s
        # The accumulators were emptied at the epoch's snapshot.
        @test sat.state.code_discr_acc == (0, 0.0)
        # Taken up once: a second take-up changes nothing.
        @test TL._take_up_cycle(vt, rx.group, slot, sat.state, take_up_record(sat, 0),
            sim_words(sat, NO_LANDING_SAMPLE), Int64(0)) === sat.state
    end
end

@testset "A correction sized for a later landing" begin
    rx = SimReceiver()
    _, sample, _ = run_simulation!(rx, 3)
    vt = rx.vt
    slot = rx.group.slots[1]
    sat = rx.sats[1]
    j = slot.member_index
    member = vt.buffers.members[j]
    words = sim_words(sat, NO_LANDING_SAMPLE)
    record = take_up_record(sat, sample)
    at_epoch = TL._correction_at_landing(vt, rx.group, slot, record, words, Int64(sample))
    @test at_epoch[3] == 0.0
    @test at_epoch[1] ≈ ustrip(Hz, sat.state.code_freq_update) atol = 1e-9
    # 20 ms on, under the same words: the range has moved on with the replica, so the
    # code correction barely changes, and the lead is the 20 ms.
    later = TL._correction_at_landing(vt, rx.group, slot, record, words, Int64(sample + 80_000))
    @test later[3] ≈ 0.02
    @test later[1] ≈ at_epoch[1] atol = 0.05
    # It must land within 2.5 cycles of the epoch.
    @test_throws ArgumentError TL._correction_at_landing(vt, rx.group, slot, record, words,
        Int64(sample + 1_100_000))
    @test_throws ArgumentError TL._correction_at_landing(vt, rx.group, slot, record, words,
        Int64(sample - 4000))
end

@testset "The solution of a running cycle" begin
    rx = SimReceiver()
    results, _, _ = run_simulation!(rx, 10)
    result = results[end]
    pvt = result.pvt
    @test result.status.running
    @test !result.status.enabled
    @test pvt.time isa TAITime
    @test pvt.reference_system === GPST()
    @test pvt.dop !== nothing
    @test 0 < pvt.dop.GDOP < 10
    @test length(pvt.sats) == length(rx.sats)
    @test keys(member_sats(rx.estimator)) == keys(pvt.sats)
    @test isempty(pvt.inter_system_biases)
    @test position_error(rx, result) < 0.5
    @test norm(SVector(pvt.velocity.x, pvt.velocity.y, pvt.velocity.z) - rx.truth.velocity) < 0.05
    @test 0.0u"m" < result.status.position_std < 10.0u"m"
    @test result.status.clock_std > 0.0u"m"
    # The filter's clock drift is reported as the relative drift.
    @test pvt.relative_clock_drift * TL.SPEED_OF_LIGHT ≈ rx.truth.clock_drift atol = 0.05
    # Timestamps advance by the cycle.
    @test results[end].pvt.time - results[end-1].pvt.time ≈ 0.1 atol = 1e-6
    # The same solution object is returned and kept.
    @test pvt === navigation_solution(rx.estimator)
    # Steered members' post-fit residuals are their discriminators alone: small.
    for info in values(pvt.sats)
        @test abs(info.residual) < 1.0u"m"
        @test abs(info.rate_residual) < 0.05u"m/s"
    end
end

@testset "Membership: out of lock coasts, a dropped satellite is released" begin
    rx = SimReceiver()
    _, sample, _ = run_simulation!(rx, 5)
    # Out of lock: still a member, unmeasured, still steered, in `member_sats` but not in
    # the solution's `sats`.
    prn = rx.sats[1].decoder.prn
    coast!(v, sat, epoch, landing, nav) =
        (fill_slot!(v, sat, epoch, landing, nav); v.prn == prn && (v.in_lock = false))
    results, sample, _ = run_simulation!(rx, 1; fill! = coast!, start_sample = sample)
    key = (:GPSL1CA, prn)
    @test rx.sats[1].state.vt_on
    @test release_reason(rx.estimator, GPSL1CA(), prn) == VT_NOT_RELEASED
    @test !haskey(results[1].pvt.sats, key)
    @test haskey(member_sats(rx.estimator), key)
    @test results[1].status.num_members == length(rx.sats)
    @test !results[1].status.released
    # No record for two cycles: dropped, released as ineligible, and re-seeded from the
    # replica's Dopplers where its command lands.
    dropped = rx.group.slots[2]
    drop!(v, sat, epoch, landing, nav) =
        (fill_slot!(v, sat, epoch, landing, nav); v === dropped && drop_slot!(v))
    results, _, _ = run_simulation!(rx, 1; fill! = drop!, start_sample = sample)
    sat = rx.sats[2]
    @test results[1].status.released
    @test dropped.release_reason == VT_INELIGIBLE
    @test release_reason(rx.estimator, GPSL1CA(), dropped.prn) == VT_INELIGIBLE
    @test !dropped.occupied
    @test !dropped.estimator_state.vt_on
    @test !sat.state.vt_on
    @test sat.state.code_freq_update == 0.0Hz
    @test sat.state.inner.init_carrier_doppler == sat.carrier_doppler * Hz
    @test sat.state.inner.init_code_doppler == sat.code_doppler * Hz
    @test results[1].status.num_members == length(rx.sats) - 1
    @test !haskey(member_sats(rx.estimator), (:GPSL1CA, dropped.prn))
    @test all(v.release_reason == VT_NOT_RELEASED for v in channel_slots(rx.group, rx.sats) if v !== dropped)
end

@testset "A member that missed the epoch sits the cycle out" begin
    rx = SimReceiver()
    _, sample, _ = run_simulation!(rx, 5)
    late = rx.group.slots[3]
    miss!(v, sat, epoch, landing, nav) =
        (fill_slot!(v, sat, epoch, landing, nav); v === late && (v.snapshot_epoch = typemin(Int)))
    results, _, _ = run_simulation!(rx, 1; fill! = miss!, start_sample = sample)
    # Not stale, so not dropped: it keeps its place and its corrections, without new ones.
    @test late.occupied
    @test !late.active
    @test late.release_reason == VT_NOT_RELEASED
    @test late.correction_cycle != rx.vt.cycle_id
    @test rx.sats[3].state.vt_on
    @test results[1].status.num_members == length(rx.sats) - 1
    @test !results[1].status.released
end

@testset "An unsolvable epoch grows the starvation timer, a solvable one pays it back" begin
    rx = SimReceiver()
    _, sample, _ = run_simulation!(rx, 3)
    # Three satellites in lock are too few for position and clock.
    starve!(v, sat, epoch, landing, nav) =
        (fill_slot!(v, sat, epoch, landing, nav); v.in_lock = v.prn in (rx.sats[1].decoder.prn, rx.sats[2].decoder.prn, rx.sats[3].decoder.prn))
    results, sample, _ = run_simulation!(rx, 3; fill! = starve!, start_sample = sample)
    @test [r.status.time_with_insufficient_meas for r in results] ≈ [0.1, 0.2, 0.3] .* s
    @test all(r -> r.status.running, results)
    results, _, _ = run_simulation!(rx, 2; start_sample = sample)
    @test [r.status.time_with_insufficient_meas for r in results] ≈ [0.25, 0.2] .* s
end

@testset "A released satellite takes over from the replica at landing" begin
    # Under an NCO delay the words committed until the landing are the vector loop's,
    # so the scalar loop re-seeds from the replica there, not from the epoch's.
    rx = SimReceiver(; inner = NCOReferencedPLLAndDLL(), records_per_cycle = 20, delay_records = 15)
    _, sample, diverged = run_simulation!(rx, 20)
    @test !diverged
    dropped = rx.group.slots[2]
    sat = rx.sats[2]
    drop!(v, sat, epoch, landing, nav) =
        (fill_slot!(v, sat, epoch, landing, nav); v === dropped && drop_slot!(v))
    _, sample, _ = run_simulation!(rx, 1; fill! = drop!, start_sample = sample)
    @test dropped.release_reason == VT_INELIGIBLE
    landing_carrier, landing_code = nco_word_at(sat.timeline, sample + 15 * SAMPLES_PER_MS)
    @test sat.state.inner.init_carrier_doppler == landing_carrier * Hz
    @test sat.state.inner.init_code_doppler == landing_code * Hz
    @test sat.state.inner.init_carrier_doppler != dropped.carrier_doppler
end

@testset "A satellite is admitted only above the admission mask" begin
    rx = SimReceiver()
    _, sample, _ = run_simulation!(rx, 5)
    vt = rx.vt
    group = rx.group
    sat = group.slots[1]
    sat.estimator_state = TL._disable_vector_tracking(sat.estimator_state)
    user_pos = first(TL.nav_filter_states(vt.x, vt.model.idxs))
    here = ENUfromECEF(ECEF(user_pos...), wgs84)
    antipode = ENUfromECEF(ECEF((-user_pos)...), wgs84)
    @test TL._is_above_admission_mask(group.signal, sat, here)
    @test !TL._is_above_admission_mask(group.signal, sat, antipode)
    # Eligible and in lock, but below the mask: not admitted, so a satellite at the
    # horizon is not admitted and released every cycle. A member stays a member; the
    # elevation mask of the running cycle releases it.
    @test !TL._update_membership!(false, group, 1, nothing, antipode)
    @test !sat.estimator_state.vt_on
    @test all(v -> v.estimator_state.vt_on, channel_slots(group, rx.sats)[2:end])
    TL._update_membership!(false, group, 1, nothing, here)
    @test sat.estimator_state.vt_on
    # A running cycle admits it back at the filter's own position, and the satellite
    # takes the admission up.
    rx.sats[1].state = TL._disable_vector_tracking(rx.sats[1].state)
    results, _, _ = run_simulation!(rx, 1; start_sample = sample)
    @test rx.sats[1].state.vt_on
    @test results[1].status.num_members == length(rx.sats)
end

@testset "Biases are reported only for what was measured" begin
    rx = SimReceiver(; signals = (GPSL1CA(), GalileoE1B()))
    # No Galileo satellite in lock: its clock coasts and is not reported.
    gps_only!(v, sat, epoch, landing, nav) =
        (fill_slot!(v, sat, epoch, landing, nav); sat.signal isa GalileoE1B && (v.in_lock = false))
    results, sample, _ = run_simulation!(rx, 10; fill! = gps_only!)
    @test results[end].status.running
    @test !haskey(results[end].pvt.inter_system_biases, GST())
    @test all(key -> first(key) === :GPSL1CA, keys(results[end].pvt.sats))
    results, _, _ = run_simulation!(rx, 3; start_sample = sample)
    @test haskey(results[end].pvt.inter_system_biases, GST())
end

@testset "A re-seed reads the epoch offset afresh" begin
    rx = SimReceiver()
    results, sample, _ = run_simulation!(rx, 5)
    before = results[end].pvt.time
    # The scalar solve takes over, and the cached epoch offset is a week stale, as after
    # a GPST week rollover no running cycle saw.
    rx.vt.time_epoch_offset -= TL.SECONDS_PER_WEEK
    rx.vt.running = false
    results, _, _ = run_simulation!(rx, 2; start_sample = sample)
    @test results[1].status.enabled
    @test results[2].status.running
    @test results[2].pvt.time - before ≈ 0.2 atol = 1e-3
end

@testset "Members kept across a fallback start over at the seed" begin
    rx = SimReceiver()
    _, sample, _ = run_simulation!(rx, 5)
    @test all(sat -> sat.state.vt_on, rx.sats)
    # The scalar solve takes over while every satellite stays in the loop, one of them
    # reacquired and not decoded yet.
    rx.vt.running = false
    prn = rx.sats[1].decoder.prn
    reacquired!(v, sat, epoch, landing, nav) = begin
        fill_slot!(v, sat, epoch, landing, nav)
        if v.prn == prn
            v.decoder = GNSSDecoderState(GPSL1CA(), prn)
            v.pvt_ready = false
        end
    end
    results, _, _ = run_simulation!(rx, 1; fill! = reacquired!, start_sample = sample)
    status = results[1].status
    @test status.enabled
    @test status.released
    @test rx.group.slots[1].release_reason == VT_INELIGIBLE
    @test !rx.sats[1].state.vt_on
    @test status.num_members == length(rx.sats) - 1
    # The others start over in the loop, with nothing stale accumulated.
    @test all(sat -> sat.state.vt_on && sat.state.code_discr_acc == (0, 0.0), rx.sats[2:end])
    @test all(slot -> slot.restart_cycle == rx.vt.cycle_id, channel_slots(rx.group, rx.sats)[2:end])
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
