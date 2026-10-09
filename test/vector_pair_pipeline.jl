# Vector tracking from records alone on a pilot + data pair: each satellite ranges on a
# pilot (the GPS L1C-P stand-in of `vector_simulation.jl`) and decodes the LNAV bits of
# its data component, whose records the host hands over separately, before, after or
# well after the pilot's.

const PAIR = GPSL1C_P() => GPSL1CA()

pair_position(result) = SVector(result.pvt.position.x, result.pvt.position.y, result.pvt.position.z)

# The transmit time of each satellite in a solution, by PRN.
pair_times(result) = Dict(prn => info.time for ((_, prn), info) in pairs(result.pvt.sats))

@testset "From records alone, ranging on a pilot and decoding its data component" begin
    rx = PipelineReceiver(; signals = PAIR)
    results, sample, diverged = run_pipeline!(rx, 36.0)
    @test !diverged
    @test length(results) == 360
    estimator = rx.estimator
    seed = findfirst(r -> r.status.enabled, results)
    @test seed !== nothing
    @test 26.0 < results[seed].time - rx.truth.t0 < 27.0
    tail = results[seed+50:end]
    @test all(r -> r.status.running && r.status.num_members == length(rx.sats), tail)
    @test all(r -> length(r.measured) == length(rx.sats), tail)
    @test all(r -> all(==(VT_NOT_RELEASED), r.reasons), tail)
    @test pipeline_errors(rx, tail) < 3.0
    @test pipeline_code_errors(tail) < 0.05
    # Everything is keyed by the driver: the solution, the members and the reports.
    @test all(key -> first(key) == :GPSL1C_P, keys(navigation_solution(estimator).sats))
    for sat in rx.sats
        report = satellite_report(estimator, GPSL1C_P(), sat.decoder.prn)
        # The data component's bit clock and decoder, the pilot's C/N₀.
        @test report.bit_synced && report.in_lock && report.pvt_ready && report.in_vector_loop
        @test report.decoder isa typeof(sat.decoder)
        @test report.decoder.data.sqrt_A == sat.decoder.data.sqrt_A
        @test 40 < report.cn0_dbhz < 50
        @test satellite_report(estimator, GPSL1CA(), sat.decoder.prn) === nothing
    end
    # Warm, every record of either signal allocates nothing.
    rx.measuring[] = true
    results, _, diverged = run_pipeline!(rx, 3.0; start_sample = sample)
    @test !diverged && all(r -> r.status.running, results)
    @test rx.allocated[] == 0
end

@testset "A pilot + data pair has the transmit times of the data component alone" begin
    # Scalar fixes on clean signals, so both replicas sit on the truth: the pair's
    # symbol count comes from the data component and its code phase from the pilot,
    # and together they make the transmit time the data component makes alone. What is
    # left between them is the two loops' tracking error — the pilot's 10 ms records
    # against C/A's 1 ms ones — of a few centimetres, where a symbol, a C/A code block
    # or a chip miscounted would be 6000 km, 300 km or 293 m.
    alone = PipelineReceiver(; config = nothing, cn0_dbhz = 80.0)
    paired = PipelineReceiver(; config = nothing, cn0_dbhz = 80.0, signals = PAIR)
    alone_results, _, alone_diverged = run_pipeline!(alone, 30.0)
    paired_results, _, paired_diverged = run_pipeline!(paired, 30.0)
    @test !alone_diverged && !paired_diverged
    @test length(alone_results) == length(paired_results)
    fixes = findall(r -> !isempty(r.pvt.sats), alone_results)
    @test length(fixes) > 20
    @test fixes == findall(r -> !isempty(r.pvt.sats), paired_results)
    for i in fixes
        alone_times = pair_times(alone_results[i])
        paired_times = pair_times(paired_results[i])
        @test keys(alone_times) == keys(paired_times)
        @test maximum(prn -> abs(alone_times[prn] - paired_times[prn]), keys(alone_times)) *
              TrackingLoops.SPEED_OF_LIGHT < 0.3
        @test norm(pair_position(alone_results[i]) - pair_position(paired_results[i])) < 0.5
    end
end

@testset "The data component's records may come before, after or well after the pilot's" begin
    # The scalar solve, whose loops do not depend on when a cycle runs: every epoch's
    # snapshot must be the same to the bit, whether a satellite's data-component record
    # reaches the epoch before its pilot's, right after it, or 25 ms — two and a half
    # pilot records — later. (Late, a satellite registers on its pilot's first record
    # rather than its data component's, so the satellites take other slots and the solve
    # sums them in another order: its solution agrees to rounding, not to the bit.)
    run(kw) = first(run_pipeline!(PipelineReceiver(; config = nothing, signals = PAIR, kw...), 30.0))
    before = run((;))
    @test count(r -> !isempty(r.pvt.sats), before) > 20
    for kw in ((; passenger_after = true), (; passenger_lag_ms = 25))
        other = run(kw)
        # The late records leave the last epoch pending at most.
        @test length(before) - 1 <= length(other) <= length(before)
        for (a, b) in zip(before, other)
            @test a.time == b.time
            @test pair_times(a) == pair_times(b)
            @test norm(pair_position(a) - pair_position(b)) < 1e-4
        end
        # In order, the solution is the same to the bit.
        haskey(kw, :passenger_after) &&
            @test all(((a, b),) -> pair_position(a) == pair_position(b), zip(before, other))
    end
end

@testset "The soft bits are decoded once, however often the host drains them" begin
    for signals in (GPSL1CA(), PAIR)
        per_record, _, = run_pipeline!(PipelineReceiver(; signals, num_sats = 6), 30.0)
        per_tick_rx = PipelineReceiver(; signals, num_sats = 6, drain_per_record = false)
        per_tick, _, = run_pipeline!(per_tick_rx, 30.0)
        @test any(r -> r.status.running, per_tick)
        @test length(per_record) == length(per_tick)
        @test all(((a, b),) -> pair_position(a) == pair_position(b), zip(per_record, per_tick))
        for sat in per_tick_rx.sats
            @test TrackingLoops.is_decoding_completed_for_positioning(
                satellite_report(per_tick_rx.estimator, sat.signal, sat.decoder.prn).decoder)
        end
    end
end
