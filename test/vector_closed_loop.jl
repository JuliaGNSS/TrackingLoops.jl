# Vector tracking closed over many cycles on the synthetic satellites of
# `vector_simulation.jl`: convergence, an outage, the fallback after the starvation
# timeout, several constellations, and an NCO delay whose landing the corrections are
# sized for.

tail_errors(receiver, results) = maximum(r -> position_error(receiver, r), results)
tail_code_errors(results) = maximum(r -> maximum(abs, r.code_errors), results)

@testset "The vector loop converges onto the true trajectory" begin
    rx = SimReceiver()
    results, _, diverged = run_simulation!(rx, 100)
    @test !diverged
    @test all(r -> r.status.running, results)
    @test all(r -> r.status.num_members == length(rx.sats), results)
    tail = results[51:end]
    @test tail_errors(rx, tail) < 0.2
    @test tail_code_errors(tail) < 0.005
    for r in tail
        velocity = SVector(r.pvt.velocity.x, r.pvt.velocity.y, r.pvt.velocity.z)
        @test norm(velocity - rx.truth.velocity) < 0.02
    end
    # The pseudorange-only loop (VDLL) closes as well.
    rx = SimReceiver(; config = VectorTracking(; use_pseudorange_rates = false))
    results, _, diverged = run_simulation!(rx, 100)
    @test !diverged
    @test tail_errors(rx, results[51:end]) < 0.5
    @test tail_code_errors(results[51:end]) < 0.01
end

@testset "Cycles off the nominal interval are propagated by their own length" begin
    # A cycle runs on the first chunk boundary past the nominal interval: 104 ms cycles on
    # a filter built for 100 ms, under a TCXO's 600 m/s of clock drift. A process model
    # kept at 100 ms would mispredict the clock by drift · 4 ms = 2.4 m every cycle.
    rx = SimReceiver(; records_per_cycle = 104, nominal_cycle = 100.0ms,
        truth_kw = (; clock_drift = 600.0))
    results, _, diverged = run_simulation!(rx, 100)
    @test !diverged
    @test rx.vt.model.integration_time ≈ 104.0ms
    tail = results[51:end]
    @test tail_errors(rx, tail) < 0.2
    @test tail_code_errors(tail) < 0.005
end

@testset "An outage of some satellites is ridden through" begin
    rx = SimReceiver()
    # Three satellites lose their signal for five seconds and stay in the loop, steered
    # by the solution of the other six.
    outage(cycle, i) = 20 <= cycle <= 70 && i <= 3
    results, _, diverged = run_simulation!(rx, 120; outage)
    @test !diverged
    @test all(r -> r.status.running && r.status.num_members == length(rx.sats), results)
    @test all(r -> r.status.time_with_insufficient_meas == 0.0s, results)
    @test all(r -> all(==(VT_NOT_RELEASED), r.reasons), results)
    during = results[25:70]
    # The coasting members are reported but not measured.
    @test all(r -> length(r.measured) == length(rx.sats) - 3, during)
    @test tail_errors(rx, during) < 0.3
    # Their replicas are still on the signal when it comes back.
    @test tail_code_errors(during) < 0.01
    after = results[90:end]
    @test all(r -> length(r.measured) == length(rx.sats), after)
    @test tail_errors(rx, after) < 0.2
    @test tail_code_errors(after) < 0.005
end

@testset "Starved beyond the timeout, the loop falls back to scalar tracking" begin
    rx = SimReceiver()
    _, sample, _ = run_simulation!(rx, 19)
    # The receiver flags every satellite out of lock (and so not ready for a scalar
    # solve), while the signals stay: nothing can be measured.
    unlocked!(vtsat, sat, epoch, landing) =
        (fill_vtsat!(vtsat, sat, epoch, landing; in_lock = false); vtsat.pvt_ready = false)
    results, sample, diverged = run_simulation!(rx, 105; start_sample = sample, fill! = unlocked!)
    @test !diverged
    timers = [r.status.time_with_insufficient_meas for r in results]
    # Strictly beyond the ten-second timeout: the 101st unsolvable cycle.
    fallback = findfirst(r -> r.status.fell_back, results)
    @test fallback == 101
    @test all(r -> r.status.running, results[1:fallback-1])
    @test timers[fallback] > 10.0s
    @test timers[fallback-1] ≈ 10.0s
    @test results[fallback].status.released
    @test !results[fallback].status.running
    @test results[fallback].status.num_members == 0
    @test all(==(VT_FALLBACK), results[fallback].reasons)
    # A fallback cycle still emits its (predicted) solution.
    @test results[fallback].pvt.time isa PositionVelocityTime.TAITime
    @test isempty(results[fallback].measured)
    # The released satellites run their scalar loops: nothing steered, nothing accumulated.
    @test all(v -> !v.estimator_state.vt_on && v.estimator_state.code_freq_update == 0.0Hz, rx.group.sats)
    @test all(v -> v.estimator_state.code_discr_acc == (0, 0.0), rx.group.sats)
    # Not ready, no satellite enters the scalar solve, so nothing re-seeds…
    @test all(r -> !r.status.running && !r.status.enabled, results[fallback+1:end])
    # …until the receiver flags them in lock again: the first scalar fix seeds vector
    # tracking anew, from scalar loops that kept tracking.
    results, _, diverged = run_simulation!(rx, 20; start_sample = sample)
    @test !diverged
    @test results[1].status.enabled
    @test all(r -> r.status.running, results)
    @test tail_errors(rx, results[10:end]) < 3.0
end

@testset "GPS and Galileo in one filter" begin
    rx = SimReceiver(; signals = (GPSL1CA(), GalileoE1B()))
    num_sats = sum(length, rx.channels)
    results, _, diverged = run_simulation!(rx, 100)
    @test !diverged
    @test all(r -> r.status.running && r.status.num_members == num_sats, results)
    tail = results[51:end]
    @test tail_errors(rx, tail) < 0.2
    @test tail_code_errors(tail) < 0.005
    pvt = results[end].pvt
    @test pvt.reference_system === GPST()
    # Both constellations share the receiver clock, so their inter-system bias is the
    # broadcast offset between the two time scales: nanoseconds, a metre at most.
    @test abs(pvt.inter_system_biases[GST()]) < 3.0u"m"
    @test count(key -> first(key) === :GalileoE1B, results[end].measured) == length(rx.channels[2])
end

@testset "Corrections sized for their landing match the loop without a delay" begin
    # A 20 ms navigation cycle and the NCO-referenced inner loop, whose landing
    # prediction keeps the carrier loop locked under the delay. A 15 ms delay lands the
    # code correction inside the second half of the cycle, a 25 ms one after the next
    # epoch: both branches of the mid-cycle advance.
    estimator = VectorPLLAndDLL(NCOReferencedPLLAndDLL())
    run(delay; kw...) = begin
        rx = SimReceiver(; estimator, records_per_cycle = 20, delay_records = delay)
        results, _, diverged = run_simulation!(rx, 1000; kw...)
        rx, results, diverged
    end
    rate_residual(results) = maximum(r -> r.max_rate_residual, results)
    rx0, results0, diverged0 = run(0)
    @test !diverged0
    tail0 = results0[750:end]
    for delay in (15, 25)
        rx, results, diverged = run(delay)
        @test !diverged
        @test all(r -> r.status.running, results)
        tail = results[750:end]
        @test tail_errors(rx, tail) < 2 * tail_errors(rx0, tail0) + 0.05
        @test tail_code_errors(tail) < 2 * tail_code_errors(tail0) + 0.002
        # The rate residue of NCO motion the signal does not back stays at the level of
        # the undelayed loop's.
        @test rate_residual(tail) < 2 * rate_residual(tail0) + 1e-3u"m/s"
        @test all(v -> v.estimator_state.code_update_landing_lead ≈ delay * 1.0ms, rx.group.sats)
    end
    # The negative control: the same delay with the corrections sized as if they acted at
    # the epoch loses the loop.
    function fill_ignoring_delay!(vtsat, sat, epoch, landing)
        fill_vtsat!(vtsat, sat, epoch, landing)
        vtsat.landing_lead = 0.0s
        vtsat.code_phase_at_landing = vtsat.code_phase
        vtsat.carrier_doppler_at_landing = vtsat.carrier_doppler
    end
    _, results, diverged = run(25; fill! = fill_ignoring_delay!)
    @test diverged || !results[end].status.running
end

@testset "The filter corrects the atmospheric delays" begin
    # The true ranges carry PositionVelocityTime's ionospheric and tropospheric delays,
    # metres each, which the filter predicts at its own position every cycle.
    rx = SimReceiver(; atmosphere = true)
    results, _, diverged = run_simulation!(rx, 100)
    @test !diverged
    @test all(r -> r.status.running, results)
    tail = results[51:end]
    @test tail_errors(rx, tail) < 0.2
    @test tail_code_errors(tail) < 0.005
    # The control: the same delays left uncorrected bias the solution by metres.
    rx = SimReceiver(; atmosphere = true, correct_atmosphere = false)
    results, _, diverged = run_simulation!(rx, 100)
    @test !diverged
    @test tail_errors(rx, results[51:end]) > 2.0
end

@testset "Two bands share one filter and its inter-frequency bias" begin
    # GPS L1 C/A and L2CM from the same satellites, with a 4 m hardware delay on L2.
    rx = SimReceiver(; signals = (GPSL1CA(), GPSL2CM()), range_biases = (0.0, 4.0))
    num_sats = sum(length, rx.channels)
    @test rx.vt.layout.extra_bands == [:L2]
    results, _, diverged = run_simulation!(rx, 100)
    @test !diverged
    @test all(r -> r.status.running && r.status.num_members == num_sats, results)
    tail = results[51:end]
    @test tail_errors(rx, tail) < 0.2
    @test tail_code_errors(tail) < 0.005
    ifb = results[end].pvt.inter_frequency_biases[:L2]
    @test ifb.reference == :L1
    @test abs(ifb.value - 4.0u"m") < 0.05u"m"
    @test count(key -> first(key) === :GPSL2CM, results[end].measured) == length(rx.channels[2])
end

@testset "A scarce constellation's clock collapses onto GPS through the GGTO" begin
    # Three GPS satellites and one Galileo satellite cannot determine a position and two
    # clocks. The Galileo satellite broadcasts the GGTO, so its clock is tied to the GPS
    # one by a pseudo-measurement, in the scalar fix that seeds the filter and in every
    # filter cycle.
    rx = SimReceiver(; signals = (GPSL1CA(), GalileoE1B()), num_sats = (3, 1))
    galileo = only(rx.channels[2])
    galileo.decoder = with_zero_ggto(galileo.decoder)
    results, _, diverged = run_simulation!(rx, 100)
    @test !diverged
    @test results[1].status.enabled
    @test all(r -> r.status.running && r.status.num_members == 4, results)
    @test all(r -> length(r.measured) == 4, results)
    gst_clock, gpst_clock, isb = only(rx.vt.buffers.observability.hub_offset_constraints)
    @test (gst_clock, gpst_clock) == (2, 1)
    @test isb == 0.0
    tail = results[51:end]
    @test tail_errors(rx, tail) < 0.5
    @test tail_code_errors(tail) < 0.005
    @test abs(results[end].pvt.inter_system_biases[GST()]) < 0.05u"m"
    # Without the GGTO there is nothing to collapse onto, and no fix to seed from.
    rx = SimReceiver(; signals = (GPSL1CA(), GalileoE1B()), num_sats = (3, 1))
    results, _, _ = run_simulation!(rx, 5)
    @test all(r -> !r.status.running && !r.status.enabled, results)
end
