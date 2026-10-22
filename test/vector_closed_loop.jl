# Vector tracking closed over many cycles on the synthetic satellites of
# `vector_simulation.jl`, with the engine filled by hand from the fixture decoders:
# convergence, an outage, the fallback after the starvation timeout, several
# constellations and bands, and an NCO delay whose landing the corrections are sized
# for. `vector_pipeline.jl` closes the loop on records alone.

using Accessors: @set
using Statistics: mean, std

# A Galileo decoder that has decoded a GGTO of zero, the offset the simulated time
# scales have: the fixtures were captured without one.
function with_zero_ggto(decoder)
    data = decoder.data
    data = @set data.A_0G = 0.0
    data = @set data.A_1G = 0.0
    data = @set data.t_0G = 0
    data = @set data.WN_0G = data.WN
    @set decoder.data = data
end

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
    velocity_errors = map(tail) do r
        SVector(r.pvt.velocity.x, r.pvt.velocity.y, r.pvt.velocity.z) - rx.truth.velocity
    end
    @test all(e -> norm(e) < 0.02, velocity_errors)
    # Without a bias: the rate model leaving the satellite velocity in the frame of
    # transmission while it rotated the line of sight biased the velocity by 7 mm/s;
    # rotating it but leaving out the rate of the rotation and the light time, 0.6 mm/s.
    @test norm(mean(velocity_errors)) < 5e-4
    # The pseudorange-only loop (VDLL) closes as well.
    rx = SimReceiver(; filter_config = NavigationFilterConfig(; use_pseudorange_rates = false))
    results, _, diverged = run_simulation!(rx, 100)
    @test !diverged
    @test tail_errors(rx, results[51:end]) < 0.5
    @test tail_code_errors(results[51:end]) < 0.01
end

@testset "A cycle over a skipped epoch is propagated by its own length" begin
    # Every other epoch is skipped (as when the satellites resume after a gap): 200 ms
    # cycles on a filter built for 100 ms, under a TCXO's 600 m/s of clock drift. A
    # process model kept at 100 ms would mispredict the clock by drift · 100 ms = 60 m
    # every cycle.
    rx = SimReceiver(; records_per_cycle = 200, nominal_records = 100,
        truth_kw = (; clock_drift = 600.0))
    results, _, diverged = run_simulation!(rx, 60)
    @test !diverged
    @test rx.vt.model.cycle_time ≈ 200.0ms
    tail = results[31:end]
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
    unlocked!(slot, sat, epoch, landing, nav) =
        (fill_slot!(slot, sat, epoch, landing, nav; in_lock = false); slot.pvt_ready = false)
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
    @test all(sat -> !sat.state.in_vector_loop && sat.state.code_freq_update == 0.0Hz, rx.sats)
    @test all(sat -> sat.state.code_disc_sum == (0, 0.0), rx.sats)
    # Not ready, no satellite enters the scalar solve, so nothing re-seeds…
    @test all(r -> !r.status.running && !r.status.seeded, results[fallback+1:end])
    # …until the receiver flags them in lock again: the first scalar fix seeds vector
    # tracking anew, from scalar loops that kept tracking.
    results, _, diverged = run_simulation!(rx, 20; start_sample = sample)
    @test !diverged
    @test results[1].status.seeded
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

@testset "Seeded from a high-DOP fix, the filter reports the uncertainty it has" begin
    # Four GPS satellites and one Galileo: the scalar fix has a GDOP of about 45, which
    # turns the simulation's sub-metre modelling error into a fix metres off. Seeded
    # with a fixed covariance, the filter stayed about 5 m off while reporting under a
    # metre (JuliaGNSS/TrackingLoops.jl#16).
    rx = SimReceiver(; signals = (GPSL1CA(), GalileoE1B()), num_sats = (4, 1))
    results, _, diverged = run_simulation!(rx, 300)
    @test !diverged
    @test all(r -> r.status.running, results)
    @test position_error(rx, results[1]) > 3.0
    @test all(r -> position_error(rx, r) < 3 * ustrip(u"m", r.status.position_std), results)
end

@testset "Without redundancy, the filter settles onto the truth and stays there" begin
    # Four GPS satellites, a GDOP of about 40: every metre of range error left in the
    # simulated signals moves the solution by tens. With the satellite clocks held at
    # their initial broadcast correction, the solution drifted off by about 0.17 m/s
    # after a few hundred cycles, beyond 10 m by cycle 1200, while the reported
    # uncertainty levelled off (JuliaGNSS/TrackingLoops.jl#25). With the rate model short
    # of the derivative of the pseudorange model by the Earth rotation of the satellite
    # velocity, the tail was 7 m off.
    rx = SimReceiver(; signals = (GPSL1CA(),), num_sats = (4,))
    results, _, diverged = run_simulation!(rx, 1200)
    @test !diverged
    @test all(r -> r.status.running, results)
    @test all(r -> position_error(rx, r) < 3 * ustrip(u"m", r.status.position_std), results)
    @test tail_errors(rx, results[601:end]) < 0.5
end

@testset "Corrections sized for their landing match the loop without a delay" begin
    # A 20 ms navigation cycle and the NCO-referenced inner loop, whose landing
    # prediction keeps the carrier loop locked under the delay. A 15 ms delay lands the
    # code correction inside the second half of the cycle, a 25 ms one after the next
    # epoch: both branches of the mid-cycle advance.
    run(delay; kw...) = begin
        rx = SimReceiver(; inner = NCOReferencedPLLAndDLL(), records_per_cycle = 20, delay_records = delay)
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
        @test all(sat -> sat.state.code_update_landing_lead ≈ delay * 1.0ms, rx.sats)
    end
    # The negative control: the same delay with the corrections sized as if they acted at
    # the epoch loses the loop.
    _, results, diverged = run(25; ignore_delay = true)
    @test diverged || !results[end].status.running
end

@testset "The fused rates are continuous across cycles and weighted as they are noisy" begin
    # Galileo E1 at 45 dB-Hz, the C/N₀ the slots report. Each member's rate measurement,
    # against the true Doppler at the epoch, has the noise the filter weights it with and
    # the −1/2 lag-1 autocorrelation of measurements that share their boundary prompts;
    # summed over the cycles, it stays at the carrier-phase noise. The replica runs on
    # the filter's ramp alone: with the PLL's jitter in its word, the residuals were
    # nearly five times `√R`, their autocorrelation −0.15, and the integrated error
    # walked by decimetres.
    rx = SimReceiver(; signals = (GalileoE1B(),))
    noise_std = sqrt(1 / (2 * 10^4.5 * 1e-3))
    foreach(sat -> sat.noise_std = noise_std, rx.sats)
    T = 0.1
    residuals = [Float64[] for _ in rx.sats]
    variances = Float64[]
    max_code_error = 0.0
    all_measured = true
    sample = 0
    for cycle = 1:400
        for k = 4:4:100
            sample_index = sample + k * SAMPLES_PER_MS
            for sat in rx.sats
                code_error = simulate_record!(sat, rx.estimator, rx.truth, sim_time(rx, sample_index),
                    4 * SAMPLES_PER_MS, sample_index, NO_LANDING_SAMPLE)
                max_code_error = max(max_code_error, abs(code_error))
            end
        end
        sample += 100 * SAMPLES_PER_MS
        run_cycle!(rx, sample, fill_slot!)
        cycle > 50 || continue
        for member in rx.vt.buffers.members
            all_measured &= member.available
            sat = rx.sats[member.slot_index]
            push!(residuals[member.slot_index], member.pseudorange_rate -
                member.wavelength * true_doppler(sat, rx.truth, sim_time(rx, sample)))
            push!(variances, TL.pseudorange_rate_noise_variance(member, T))
        end
    end
    @test max_code_error < 0.05
    @test all_measured
    sqrt_R = sqrt(mean(variances))
    for r in residuals
        @test std(r) / sqrt_R ≈ 1.0 atol = 0.25
        y = r .- mean(r)
        @test sum(y[1:end-1] .* y[2:end]) / sum(abs2, y) ≈ -0.5 atol = 0.12
        @test maximum(abs, cumsum(r) .* T) < 0.03
    end
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

@testset "A scarce constellation's clock is tied to GPS through the GGTO" begin
    # Three GPS satellites and one Galileo satellite cannot determine a position and two
    # clocks. The Galileo satellite broadcasts the GGTO, so its clock is tied to the GPS
    # one by a pseudo-measurement, in the scalar fix that seeds the filter and in every
    # filter cycle.
    rx = SimReceiver(; signals = (GPSL1CA(), GalileoE1B()), num_sats = (3, 1))
    galileo = only(rx.channels[2])
    galileo.decoder = with_zero_ggto(galileo.decoder)
    results, _, diverged = run_simulation!(rx, 100)
    @test !diverged
    @test results[1].status.seeded
    @test all(r -> r.status.running && r.status.num_members == 4, results)
    @test all(r -> length(r.measured) == 4, results)
    gst_clock, gpst_clock, isb = only(rx.vt.buffers.time_offset_constraints)
    @test (gst_clock, gpst_clock) == (2, 1)
    @test isb == 0.0
    # The solution reports the broadcast offset it took as given, as `calc_pvt` does
    # (compared with `==`: the zero GGTO's offset is `-0.0 m`).
    function took_the_ggto(pvt)
        constraint = only(pvt.time_offset_constraints)
        (constraint.system, constraint.target) == (GST(), GPST()) && constraint.offset == 0.0u"m"
    end
    @test all(r -> took_the_ggto(r.pvt), results)
    tail = results[51:end]
    @test tail_errors(rx, tail) < 0.5
    @test tail_code_errors(tail) < 0.005
    @test abs(results[end].pvt.inter_system_biases[GST()]) < 0.05u"m"
    # Without the GGTO nothing ties the clocks, and there is no fix to seed from.
    rx = SimReceiver(; signals = (GPSL1CA(), GalileoE1B()), num_sats = (3, 1))
    results, _, _ = run_simulation!(rx, 5)
    @test all(r -> !r.status.running && !r.status.seeded, results)
end
