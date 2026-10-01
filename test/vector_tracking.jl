# `update_navigation!` cycle by cycle, on the synthetic satellites of
# `vector_simulation.jl`: seeding from the scalar fix, the loop closure, membership and
# release, the solution, and decoding.
using PositionVelocityTime: calc_ρ_hat!, PVTSolution, TAITime
using Geodesy: ENUfromECEF, wgs84

@testset "VTSat defaults" begin
    decoder = GNSSDecoderState(GPSL1CA(), 5)
    state = init_estimator_state(VectorPLLAndDLL(), GPSL1CA(), 100.0Hz, 0.1Hz)
    sat = VTSat(decoder, state)
    @test sat.prn == 5
    @test !sat.active
    @test sat.estimator_state === state
    @test sat.landing_lead == 0.0s
    @test isnan(sat.cn0_dbhz)
    @test sat.release_reason == VT_NOT_RELEASED
    @test VTSat(decoder, state; prn = 9, in_lock = true).in_lock
    group = VTSignalGroup(GPSL1CA(), [sat])
    @test group.sats[1] === sat
end

@testset "Without a configuration only the scalar PVT is solved" begin
    rx = SimReceiver(; config = nothing)
    @test !rx.vt.enabled
    results, _, diverged = run_simulation!(rx, 5)
    @test !diverged
    @test all(r -> !r.status.running && !r.status.enabled, results)
    @test all(r -> r.status.num_members == 0, results)
    @test all(sat -> !sat.estimator_state.vt_on, rx.group.sats)
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
    # Every fix satellite is in the loop with corrections, and nothing was reset: the
    # members joined with empty accumulators and accumulate from now on.
    for sat in rx.group.sats
        @test sat.estimator_state.vt_on
        @test sat.estimator_state.code_discr_acc == (0, 0.0)
        @test isfinite(sat.estimator_state.code_freq_update)
        @test sat.release_reason == VT_NOT_RELEASED
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
    fill_unready!(v, sat, epoch, landing) = (fill_vtsat!(v, sat, epoch, landing); v.pvt_ready = false)
    results, _, _ = run_simulation!(rx, 2; fill! = fill_unready!)
    @test all(r -> !r.status.running && !r.status.enabled, results)
    @test all(sat -> !sat.estimator_state.vt_on, rx.group.sats)
end

@testset "Without an NCO delay the corrections are the residuals at the updated state" begin
    rx = SimReceiver()
    run_simulation!(rx, 3)
    vt = rx.vt
    buffers = vt.buffers
    members = buffers.members
    idxs = vt.model.idxs
    T = 0.1
    # The corrections the cycle just wrote, recomputed from the updated state the way
    # GNSSReceiver computed them: all members at once, from the epoch's rows.
    ξ = TL.position_and_bias_vector(vt.x, idxs)
    predicted = calc_ρ_hat!(zeros(length(members)), [m.sat_position for m in members], ξ,
        TL.vt_bias_columns(members, vt.layout))
    user_pos, user_vel, user_drift = TL.nav_filter_states(vt.x, idxs)
    for (j, member) in enumerate(members)
        sat = rx.group.sats[member.slot]
        measured = TL.pseudorange_from_tows(ustrip(s, vt.reference_time), member.time_gpst_count) -
                   buffers.delays[j]
        rate = TL.predict_pseudorange_rate(user_pos, user_vel, user_drift,
            member.sat_position, member.sat_velocity, member.sat_clock_drift)
        # The measurement the last update fused against the one the corrections use.
        @test measured === buffers.measured_pseudoranges[j]
        # The accumulators were reset after the corrections, so read the corrections
        # themselves.
        @test sat.estimator_state.code_freq_update ===
              TL.nco_code_correction(predicted[j], measured, member.code_frequency, T) * Hz
        # The carrier correction evaluates the satellite's velocity afresh at the landing
        # (the same instant here), where the compiler may contract the orbit arithmetic
        # differently than in the row: equal to rounding, not to the bit.
        @test sat.estimator_state.carrier_freq_update ≈
              TL.nco_carrier_correction(rate, member.pseudorange_rate, member.wavelength) * Hz atol = 1e-9Hz
        @test sat.estimator_state.code_update_landing_lead == 0.0s
        @test sat.estimator_state.code_discr_acc == (0, 0.0)
    end
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
    @test keys(rx.vt.member_sats) == keys(pvt.sats)
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
    @test pvt === rx.vt.pvt
    # Steered members' post-fit residuals are their discriminators alone: small.
    for info in values(pvt.sats)
        @test abs(info.residual) < 1.0u"m"
        @test abs(info.rate_residual) < 0.05u"m/s"
    end
end

@testset "Membership: out of lock coasts, ineligible is released" begin
    rx = SimReceiver()
    _, sample, _ = run_simulation!(rx, 5)
    # Out of lock: still a member, unmeasured, still steered, in `member_sats` but not in
    # the solution's `sats`.
    coast!(v, sat, epoch, landing) =
        (fill_vtsat!(v, sat, epoch, landing); v.prn == rx.sats[1].decoder.prn && (v.in_lock = false))
    results, sample, _ = run_simulation!(rx, 1; fill! = coast!, start_sample = sample)
    key = (:GPSL1CA, rx.sats[1].decoder.prn)
    @test rx.group.sats[1].estimator_state.vt_on
    @test rx.group.sats[1].release_reason == VT_NOT_RELEASED
    @test !haskey(results[1].pvt.sats, key)
    @test haskey(rx.vt.member_sats, key)
    @test results[1].status.num_members == length(rx.sats)
    @test !results[1].status.released
    # No longer tracked: released as ineligible and re-seeded from the replica's Dopplers.
    drop!(v, sat, epoch, landing) =
        (fill_vtsat!(v, sat, epoch, landing); v.prn == rx.sats[2].decoder.prn && (v.active = false))
    results, _, _ = run_simulation!(rx, 1; fill! = drop!, start_sample = sample)
    released = rx.group.sats[2]
    @test results[1].status.released
    @test released.release_reason == VT_INELIGIBLE
    @test !released.estimator_state.vt_on
    @test released.estimator_state.code_freq_update == 0.0Hz
    @test released.estimator_state.inner.init_carrier_doppler == released.carrier_doppler
    @test released.estimator_state.inner.init_code_doppler == released.code_doppler
    @test released.carrier_doppler_at_landing == released.carrier_doppler
    @test released.code_doppler_at_landing == released.code_doppler
    @test results[1].status.num_members == length(rx.sats) - 1
    @test !haskey(rx.vt.member_sats, (:GPSL1CA, released.prn))
    @test all(v.release_reason == VT_NOT_RELEASED for v in rx.group.sats if v !== released)
end

@testset "An unsolvable epoch grows the starvation timer, a solvable one pays it back" begin
    rx = SimReceiver()
    _, sample, _ = run_simulation!(rx, 3)
    # Three satellites in lock are too few for position and clock.
    starve!(v, sat, epoch, landing) =
        (fill_vtsat!(v, sat, epoch, landing); v.in_lock = v.prn in (rx.sats[1].decoder.prn, rx.sats[2].decoder.prn, rx.sats[3].decoder.prn))
    results, sample, _ = run_simulation!(rx, 3; fill! = starve!, start_sample = sample)
    @test [r.status.time_with_insufficient_meas for r in results] ≈ [0.1, 0.2, 0.3] .* s
    @test all(r -> r.status.running, results)
    results, _, _ = run_simulation!(rx, 2; start_sample = sample)
    @test [r.status.time_with_insufficient_meas for r in results] ≈ [0.25, 0.2] .* s
end

@testset "decode_soft_bits! decodes and empties the soft bits" begin
    state = SignalLoopState(GPSL1CA())
    decoder = GNSSDecoderState(GPSL1CA(), 3)
    @test decode_soft_bits!(decoder, state) === decoder
    append!(get_soft_bits(state), Float32[1, -1, 1, 1, -1, -1, 1, -1])
    new_decoder = decode_soft_bits!(decoder, state)
    @test new_decoder isa typeof(decoder)
    @test isempty(get_soft_bits(state))
end

@testset "A released satellite takes over from the replica at landing" begin
    rx = SimReceiver()
    _, sample, _ = run_simulation!(rx, 5)
    # Under an NCO delay the words committed until the landing are the vector loop's,
    # so the scalar loop re-seeds from the replica there, not from the epoch's.
    prn = rx.sats[2].decoder.prn
    function drop_delayed!(v, sat, epoch, landing)
        fill_vtsat!(v, sat, epoch, landing)
        if v.prn == prn
            v.active = false
            v.landing_lead = 0.02s
            v.carrier_doppler_at_landing = v.carrier_doppler + 3.0Hz
            v.code_doppler_at_landing = v.code_doppler + 0.002Hz
        end
    end
    run_simulation!(rx, 1; fill! = drop_delayed!, start_sample = sample)
    released = rx.group.sats[2]
    @test released.release_reason == VT_INELIGIBLE
    @test released.estimator_state.inner.init_carrier_doppler == released.carrier_doppler + 3.0Hz
    @test released.estimator_state.inner.init_code_doppler == released.code_doppler + 0.002Hz
end

@testset "A satellite is admitted only above the admission mask" begin
    rx = SimReceiver()
    _, sample, _ = run_simulation!(rx, 5)
    vt = rx.vt
    group = rx.group
    sat = group.sats[1]
    sat.estimator_state = disable_vector_tracking(sat.estimator_state)
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
    @test all(v -> v.estimator_state.vt_on, group.sats[2:end])
    TL._update_membership!(false, group, 1, nothing, here)
    @test sat.estimator_state.vt_on
    # A running cycle admits it back at the filter's own position.
    sat.estimator_state = disable_vector_tracking(sat.estimator_state)
    results, _, _ = run_simulation!(rx, 1; start_sample = sample)
    @test results[1].status.num_members == length(rx.sats)
end

@testset "Biases are reported only for what was measured" begin
    rx = SimReceiver(; signals = (GPSL1CA(), GalileoE1B()))
    # No Galileo satellite in lock: its clock coasts and is not reported.
    gps_only!(v, sat, epoch, landing) =
        (fill_vtsat!(v, sat, epoch, landing); sat.signal isa GalileoE1B && (v.in_lock = false))
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

@testset "Stale members of a rebuilt state do not stop the seed" begin
    rx = SimReceiver()
    _, sample, _ = run_simulation!(rx, 5)
    @test all(v -> v.estimator_state.vt_on, rx.group.sats)
    # A loop process rebuilds its state over the channel states it kept, all still in
    # the loop, one of them reacquired and not decoded yet.
    vt = VectorTrackingState(VectorTracking(), rx.groups; approximate_year = 2021,
        enable_ionospheric_correction = false, enable_tropospheric_correction = false)
    rebuilt = SimReceiver(rx.channels, rx.groups, vt, rx.truth, rx.estimator, rx.cycle_ms, rx.delay_ms)
    prn = rx.sats[1].decoder.prn
    reacquired!(v, sat, epoch, landing) = begin
        fill_vtsat!(v, sat, epoch, landing)
        if v.prn == prn
            v.decoder = GNSSDecoderState(GPSL1CA(), prn)
            v.pvt_ready = false
        end
    end
    results, _, _ = run_simulation!(rebuilt, 1; fill! = reacquired!, start_sample = sample)
    status = results[1].status
    @test status.enabled
    @test status.released
    @test rx.group.sats[1].release_reason == VT_INELIGIBLE
    @test !rx.group.sats[1].estimator_state.vt_on
    @test status.num_members == length(rx.sats) - 1
    # The others start over in the loop, with nothing stale accumulated.
    @test all(v -> v.estimator_state.vt_on && v.estimator_state.code_discr_acc == (0, 0.0),
        rx.group.sats[2:end])
end

@testset "update_navigation! rejects a cycle time and landing leads it cannot use" begin
    rx = SimReceiver()
    _, sample, _ = run_simulation!(rx, 2)
    @test_throws ArgumentError update_navigation!(rx.vt, rx.groups, 0.0s)
    @test_throws ArgumentError update_navigation!(rx.vt, rx.groups, -0.1s)
    @test_throws ArgumentError update_navigation!(rx.vt, rx.groups, Inf * s)
    sat = rx.group.sats[1]
    sat.landing_lead = 0.26s
    @test_throws ArgumentError update_navigation!(rx.vt, rx.groups, 0.1s)
    sat.landing_lead = -0.01s
    @test_throws ArgumentError update_navigation!(rx.vt, rx.groups, 0.1s)
    # Nothing was touched: the loop runs on.
    results, _, diverged = run_simulation!(rx, 2; start_sample = sample)
    @test !diverged
    @test all(r -> r.status.running, results)
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
