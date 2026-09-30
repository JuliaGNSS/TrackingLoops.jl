# `update_navigation!` cycle by cycle, on the synthetic satellites of
# `vector_simulation.jl`: seeding from the scalar fix, the loop closure, membership and
# release, the solution, and decoding.
using PositionVelocityTime: calc_ρ_hat!, PVTSolution, TAITime

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
        @test sat.estimator_state.carrier_freq_update ===
              TL.nco_carrier_correction(rate, member.pseudorange_rate, member.wavelength) * Hz
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
