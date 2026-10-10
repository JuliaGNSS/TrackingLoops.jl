# Vector tracking from records alone: GPS L1 C/A satellites broadcasting real LNAV
# bits, every one stepped through `step_loop` and nothing else (the pipeline harness of
# `vector_simulation.jl`). The estimator syncs to the bits, decodes them, solves the
# scalar PVT, seeds its filter and takes the satellites over by itself.

pipeline_errors(receiver, results) = maximum(r -> position_error(receiver, r), results)
pipeline_code_errors(results) = maximum(r -> maximum(abs, r.code_errors), results)
pipeline_velocity_errors(receiver, results) = maximum(results) do r
    norm(SVector(r.pvt.velocity.x, r.pvt.velocity.y, r.pvt.velocity.z) - receiver.truth.velocity)
end
pipeline_slots(receiver) = receiver.vt.groups[1].slots[1:length(receiver.sats)]

# A receiver started cold and run until vector tracking has run for a few seconds.
const WARM_PIPELINE = let
    rx = PipelineReceiver(; num_sats = 8, estimator_kw = (; max_satellites_per_signal = 8))
    results, sample, diverged = run_pipeline!(rx, 30.0)
    @assert !diverged && results[end].status.running
    (; rx, results, sample)
end

# A copy of the warm receiver to run on. On Julia 1.10 `deepcopy` keeps no spare
# capacity of an empty vector, so the soft-bit buffers get theirs back: the first soft
# bit of every satellite would allocate otherwise.
function warm_pipeline()
    rx = deepcopy(WARM_PIPELINE.rx)
    for sat in rx.sats
        sizehint!(sat.loop.bit_buffer.soft_bits, 64)
    end
    rx
end

@testset "From records alone: bit sync, decoding, the scalar fix, the filter" begin
    rx = PipelineReceiver()
    results, sample, diverged = run_pipeline!(rx, 36.0)
    @test !diverged
    nav = rx.vt
    # A cycle on every epoch: the records' grid starts at sample 0.
    @test length(results) == 360
    @test [r.time - rx.truth.t0 for r in results] ≈ 0.0:0.1:35.9
    # Every satellite found its bit edges, decoded its clock and ephemeris and is in
    # lock; the first fix came once the last of them had decoded subframe 3, 8 + 18 s
    # in, and seeded the filter.
    for slot in pipeline_slots(rx)
        @test slot.occupied
        @test slot.decoding_bit_synced
        @test TL.is_decoding_completed_for_positioning(slot.running_decoder)
        @test slot.in_lock && slot.pvt_ready
        @test 40 < slot.cn0_dbhz < 50
    end
    seed = findfirst(r -> r.status.enabled, results)
    @test 26.0 < results[seed].time - rx.truth.t0 < 27.0
    @test all(r -> !r.status.running && isempty(r.measured), results[1:seed-1])
    @test position_error(rx, results[seed]) < 10.0
    # Then the filter has every satellite: the solution near the truth, each replica
    # on its signal.
    tail = results[seed+50:end]
    @test all(r -> r.status.running && r.status.num_members == length(rx.sats), tail)
    @test all(r -> length(r.measured) == length(rx.sats), tail)
    @test all(r -> all(==(VT_NOT_RELEASED), r.reasons), tail)
    @test pipeline_errors(rx, tail) < 3.0
    @test pipeline_code_errors(tail) < 0.05
    @test all(sat -> sat.state.vt_on, rx.sats)
    @test 0.0u"m" < navigation_status(rx.estimator).position_std < 3.0u"m"
    @test keys(member_sats(rx.estimator)) == keys(navigation_solution(rx.estimator).sats)
    # Each satellite took up every cycle once.
    @test all(sat -> sat.state.cycle_id == nav.cycle_id, rx.sats)
    @test nav.registrations == length(rx.sats)
    # What a consumer reads off the estimator: the cycle count and its epoch on the
    # records' grid, the solution with its DOP, and per satellite what the engine
    # decoded and measured.
    estimator = rx.estimator
    @test navigation_cycle(estimator) == length(results)
    @test navigation_epoch(estimator) ≈ 35.9s
    @test navigation_epoch(estimator) ≈ (results[end].time - rx.truth.t0) * s
    solution = navigation_solution(estimator)
    @test solution.dop !== nothing && 0 < solution.dop.GDOP < 10
    for sat in rx.sats
        report = satellite_report(estimator, GPSL1CA(), sat.decoder.prn)
        @test report isa SatelliteReport
        @test report.prn == sat.decoder.prn
        @test report.tracked && report.bit_synced && report.in_lock && report.pvt_ready
        @test report.in_vector_loop
        @test report.release_reason == VT_NOT_RELEASED
        @test report.epoch ≈ navigation_epoch(estimator)
        @test 40 < report.cn0_dbhz < 50
        @test TL.is_decoding_completed_for_positioning(report.decoder)
        @test report.decoder.data.sqrt_A == sat.decoder.data.sqrt_A
    end
    @test satellite_report(estimator, GPSL1CA(), 32) === nothing
    @test satellite_report(estimator, GalileoE1B(), rx.sats[1].decoder.prn) === nothing
    report_of(e, signal, prn) = @allocated satellite_report(e, signal, prn)
    signal = GPSL1CA()
    report_of(estimator, signal, rx.sats[1].decoder.prn)
    @test report_of(estimator, signal, rx.sats[1].decoder.prn) == 0
end

@testset "From records alone, on a clean signal" begin
    rx = PipelineReceiver(; cn0_dbhz = 80.0)
    results, _, diverged = run_pipeline!(rx, 36.0)
    @test !diverged
    seed = findfirst(r -> r.status.enabled, results)
    @test seed !== nothing
    tail = results[seed+50:end]
    @test all(r -> r.status.running, tail)
    @test pipeline_errors(rx, tail) < 0.3
    @test pipeline_velocity_errors(rx, tail) < 0.2
    @test pipeline_code_errors(tail) < 0.005
end

@testset "From records alone, an outage of some satellites is ridden through" begin
    rx = warm_pipeline()
    start = WARM_PIPELINE.sample
    t_start = start / 4e6
    # Three satellites lose their signal for four seconds: they drop out of lock and
    # stay in the loop, steered by the solution of the other five. (The C/N₀ estimate of
    # bare noise sits just below the lock threshold and now and then crosses it, which
    # measures a coasting satellite for a cycle.)
    outage(t, i) = t_start + 1 <= t <= t_start + 5 && i <= 3
    results, _, diverged = run_pipeline!(rx, 12.0; start_sample = start, outage)
    @test !diverged
    @test all(r -> r.status.running && r.status.num_members == length(rx.sats), results)
    @test all(r -> all(==(VT_NOT_RELEASED), r.reasons), results)
    during = filter(r -> t_start + 3 <= r.time - rx.truth.t0 <= t_start + 5, results)
    @test all(r -> length(r.measured) < length(rx.sats), during)
    @test count(r -> length(r.measured) == length(rx.sats) - 3, during) > length(during) / 2
    @test pipeline_errors(rx, during) < 6.0
    # Their replicas are still on the signal when it comes back.
    @test pipeline_code_errors(during) < 0.1
    after = filter(r -> r.time - rx.truth.t0 >= t_start + 9, results)
    @test all(r -> length(r.measured) == length(rx.sats), after)
    @test pipeline_errors(rx, after) < 6.0
    @test pipeline_code_errors(after) < 0.05
end

@testset "From records alone, a starved filter falls back and seeds again" begin
    # A one-second starvation timeout, and six satellites faded 12 dB for two seconds,
    # below the 35 dB-Hz lock threshold: two measured are too few, so the loop falls back
    # to scalar tracking, whose loops keep the faded signals, and seeds anew from the
    # first scalar fix once they are back above it.
    rx = PipelineReceiver(; num_sats = 8, config = VectorTracking(; insufficient_meas_timeout = 1.0s),
        estimator_kw = (; lock_cn0_threshold = 35dBHz))
    results, start, diverged = run_pipeline!(rx, 30.0)
    @test !diverged && results[end].status.running
    t_start = start / 4e6
    fade(t, i) = t_start + 0.5 <= t <= t_start + 2.5 && i <= 6 ? 0.25 : 1.0
    results, _, diverged = run_pipeline!(rx, 6.0; start_sample = start, fade)
    @test !diverged
    fallback = findfirst(r -> r.status.fell_back, results)
    @test fallback !== nothing
    @test results[fallback].status.released
    @test all(==(VT_FALLBACK), results[fallback].reasons)
    @test !results[fallback].status.running
    reseed = findfirst(r -> r.status.enabled, results)
    @test reseed !== nothing && reseed > fallback
    @test results[end].status.running
    @test all(sat -> sat.state.vt_on, rx.sats)
end

@testset "From records alone, under an NCO delay" begin
    # A 20 ms navigation cycle and the NCO-referenced inner loop, the commands landing
    # 15 ms after the record that computed them: each satellite sizes its corrections
    # for its own landing.
    kw = (; num_sats = 8, inner = NCOReferencedPLLAndDLL(), estimator_kw = (; cycle_time = 20ms))
    rx0 = PipelineReceiver(; kw...)
    results0, _, diverged0 = run_pipeline!(rx0, 33.0)
    @test !diverged0
    rx = PipelineReceiver(; kw..., delay_records = 15)
    results, _, diverged = run_pipeline!(rx, 33.0)
    @test !diverged
    seed = findfirst(r -> r.status.enabled, results)
    @test seed !== nothing
    tail = results[seed+250:end]
    tail0 = results0[findfirst(r -> r.status.enabled, results0)+250:end]
    @test all(r -> r.status.running, tail)
    @test pipeline_errors(rx, tail) < 2 * pipeline_errors(rx0, tail0) + 0.5
    @test pipeline_code_errors(tail) < 2 * pipeline_code_errors(tail0) + 0.01
    # The lead is the delay plus the wait for the next record after the cycle.
    @test all(sat -> 0.015 <= ustrip(s, sat.state.code_update_landing_lead) <= 0.018, rx.sats)
end

@testset "A dropped satellite frees its slot for the next" begin
    rx = warm_pipeline()
    nav = rx.vt
    start = WARM_PIPELINE.sample
    slots = nav.groups[1].slots
    @test length(slots) == 8
    gone = pop!(rx.sats)
    pop!(rx.last_ends)
    gone_slot = slots[gone.state.slot]
    # Two cycles without a record: dropped, released as ineligible, its slot free with
    # its storage kept.
    rx.measuring[] = true
    results, sample, diverged = run_pipeline!(rx, 0.5; start_sample = start)
    @test !diverged
    @test !gone_slot.occupied
    @test !satellite_report(rx.estimator, GPSL1CA(), gone.decoder.prn).tracked
    @test any(r -> r.status.released, results)
    @test release_reason(rx.estimator, GPSL1CA(), gone.decoder.prn) in (VT_INELIGIBLE, VT_NOT_RELEASED)
    @test all(r -> r.status.num_members == 7, results[end-2:end])
    # A satellite never seen before takes the free slot: no slot is added, and its
    # decoder starts from nothing.
    decoders, _ = fixture_decoders(GPSL1CA())
    newcomer_decoder = decoders[9]
    newcomer = SimSat(GPSL1CA(), newcomer_decoder, rx.estimator, rx.truth, sim_time(rx, sample);
        cn0_dbhz = 45.0, stream = LNAVStream(newcomer_decoder.data))
    newcomer.next_end_sample = next_block_end(newcomer, sample)
    push!(rx.sats, newcomer)
    push!(rx.last_ends, sample)
    results, sample, diverged = run_pipeline!(rx, 1.0; start_sample = sample)
    @test !diverged
    @test length(slots) == 8
    @test newcomer.state.slot == gone.state.slot
    @test gone_slot.occupied
    @test gone_slot.prn == newcomer_decoder.prn
    @test gone_slot.running_decoder.prn == newcomer_decoder.prn
    @test !TL.is_decoding_completed_for_positioning(gone_slot.running_decoder)
    @test nav.registrations == 9
    # Dropping, freeing and taking the slot over allocated nothing, and neither did the
    # warm records and cycles around it.
    @test rx.allocated[] == 0
    @test all(r -> r.status.running, results)
    # With every slot taken, one more satellite grows the group.
    another = SimSat(GPSL1CA(), gone.decoder, rx.estimator, rx.truth, sim_time(rx, sample);
        cn0_dbhz = 45.0, stream = gone.stream)
    another.next_end_sample = next_block_end(another, sample)
    push!(rx.sats, another)
    push!(rx.last_ends, sample)
    rx.measuring[] = false
    _, _, diverged = run_pipeline!(rx, 0.3; start_sample = sample)
    @test !diverged
    @test length(slots) == 9
    @test another.state.slot == 9
end

@testset "A re-acquired satellite gets its slot and its decoded data back" begin
    rx = warm_pipeline()
    nav = rx.vt
    lost = rx.sats[2]
    index = lost.state.slot
    prn = lost.decoder.prn
    # Lost for half a second, no record at all: dropped and released.
    deleteat!(rx.sats, 2)
    deleteat!(rx.last_ends, 2)
    _, sample, diverged = run_pipeline!(rx, 0.5; start_sample = WARM_PIPELINE.sample)
    @test !diverged
    slot = nav.groups[1].slots[index]
    @test !slot.occupied
    # Re-acquired: a fresh state on the same PRN, which registers on its old slot.
    registrations = nav.registrations
    reacquired = SimSat(GPSL1CA(), lost.decoder, rx.estimator, rx.truth, sim_time(rx, sample);
        cn0_dbhz = 45.0, stream = lost.stream)
    reacquired.next_end_sample = next_block_end(reacquired, sample)
    push!(rx.sats, reacquired)
    push!(rx.last_ends, sample)
    results, _, diverged = run_pipeline!(rx, 14.0; start_sample = sample)
    @test !diverged
    @test nav.registrations == registrations + 1
    @test reacquired.state.slot == index
    @test slot.occupied && slot.prn == prn
    # Its decoder restarted its sync but kept what it had decoded, so once the bit clock
    # has found the edges again the next subframe makes it ready — not a whole frame —
    # and the filter takes it back.
    @test slot.pvt_ready
    @test reacquired.state.vt_on
    @test results[end].status.num_members == length(rx.sats)
    @test all(r -> r.status.running, results)
end

@testset "A warm minute allocates nothing" begin
    rx = warm_pipeline()
    rx.measuring[] = true
    results, _, diverged = run_pipeline!(rx, 60.0; start_sample = WARM_PIPELINE.sample)
    @test !diverged
    @test length(results) == 600
    @test all(r -> r.status.running, results)
    @test rx.allocated[] == 0
end

# ─────────────────────────────────────────────────────────────────────────────
# The host's bit clock and C/N₀ estimator, and pilot + data pairs.

using PositionVelocityTime: calc_uncorrected_time

# Per satellite, the transmit time of its latest snapshot minus the true one at that
# epoch, as a range (m).
function snapshot_range_errors(rx)
    nav = rx.vt
    group = nav.groups[1]
    map(rx.sats) do sat
        slot = group.slots[sat.state.slot]
        t = rx.truth.t0 + TL._epoch_time(nav, slot.snapshot_epoch)
        u_true = uncorrected_time(sat, first(true_transmit(sat, rx.truth, t)))
        (calc_uncorrected_time(TL._satellite_state(group.signal, slot)) - u_true) *
            TL.SPEED_OF_LIGHT
    end
end

first_fix(results) = results[findfirst(r -> r.status.enabled, results)]

@testset "A pilot + data pair ranges on the pilot and decodes the data" begin
    rx = PipelineReceiver(; pilot = true)
    results, sample, diverged = run_pipeline!(rx, 36.0)
    @test !diverged
    estimator = rx.estimator
    group = rx.vt.groups[1]
    @test group.signal isa GPSL1C_P && group.decoding_signal isa GPSL1CA
    # The data's bit clock synced and its bits decoded; the pilot's C/N₀ sets the lock.
    for sat in rx.sats
        slot = group.slots[sat.state.slot]
        @test slot.decoding_bit_synced
        @test TL.is_decoding_completed_for_positioning(slot.running_decoder)
        @test slot.in_lock && slot.pvt_ready
        report = satellite_report(estimator, GPSL1C_P(), sat.decoder.prn)
        @test report.bit_synced && report.in_vector_loop
        @test report.cn0_dbhz ≈ ustrip(estimate_cn0(sat.loop, 1ms)) atol = 1.0
        @test satellite_report(estimator, GPSL1CA(), sat.decoder.prn) === nothing
        @test release_reason(estimator, GPSL1C_P(), sat.decoder.prn) == VT_NOT_RELEASED
    end
    # The first fix comes once every satellite's data is decoded, as on the data alone.
    seed = findfirst(r -> r.status.enabled, results)
    @test 26.0 < results[seed].time - rx.truth.t0 < 27.0
    @test position_error(rx, results[seed]) < 10.0
    tail = results[seed+50:end]
    @test all(r -> r.status.running && r.status.num_members == length(rx.sats), tail)
    @test pipeline_errors(rx, tail) < 3.0
    @test pipeline_code_errors(tail) < 0.2
    # The solution and the members are keyed by the driver.
    @test all(key -> first(key) == :GPSL1C_P, keys(member_sats(estimator)))
    @test all(key -> first(key) == :GPSL1C_P, keys(navigation_solution(estimator).sats))
    # Warm, a pair's records step without allocating.
    for sat in Iterators.flatten((rx.sats, rx.passengers))
        sizehint!(sat.loop.bit_buffer.soft_bits, 64)
    end
    rx.measuring[] = true
    _, _, diverged = run_pipeline!(rx, 2.0; start_sample = sample)
    @test !diverged
    @test rx.allocated[] == 0
end

@testset "A pair's transmit time is the data's own, on a code $(pilot ? "ten times longer" : "of its own")" for pilot in (false, true)
    # On a clean signal the snapshot's transmit time is the true one to within the
    # sample the record ends on, although the pilot's code is ten data codes long: the
    # data's bit clock counts the symbols, the pilot's replica places the epoch in
    # one. And the scalar fix is the one the data alone gives.
    rx = PipelineReceiver(; pilot, cn0_dbhz = 80.0, num_sats = 6)
    results, _, diverged = run_pipeline!(rx, 27.0)
    @test !diverged
    @test all(e -> abs(e) < 1.0, snapshot_range_errors(rx))
    @test position_error(rx, first_fix(results)) < 2.0
end

@testset "At an epoch the data's record may come before or after the pilot's" begin
    # The data's records reach the estimator 25 ms after the pilot's, more than one
    # 10 ms pilot record late — or the pilot's 25 ms after the data's. Every epoch's
    # snapshot is the same either way, and so is every scalar fix up to the one that
    # seeds the filter; the commands land 30 ms after their record, after the late
    # records, so the replicas do not depend on when the records are stepped. (The
    # satellites register in the order their first records arrive, so the solve sums
    # them in another order: the fixes agree to rounding.)
    kw = (; pilot = true, num_sats = 6, inner = NCOReferencedPLLAndDLL(), delay_records = 30)
    runs = map(((0, 0), (0, 25), (25, 0))) do (driver_lag_ms, passenger_lag_ms)
        rx = PipelineReceiver(; kw..., driver_lag_ms, passenger_lag_ms)
        results, _, diverged = run_pipeline!(rx, 28.0)
        @test !diverged
        results
    end
    seeds = map(results -> findfirst(r -> r.status.enabled, results), runs)
    @test seeds[1] !== nothing && allequal(seeds)
    for results in runs[2:end]
        @test all(zip(results[1:seeds[1]], runs[1][1:seeds[1]])) do (a, b)
            norm(a.pvt.position - b.pvt.position) < 1e-6 && Set(a.measured) == Set(b.measured)
        end
        @test all(zip(results[1:seeds[1]], runs[1][1:seeds[1]])) do (a, b)
            a.time == b.time
        end
    end
end

@testset "The soft bits decode the same whenever the host drains them" begin
    # After every record, as a loop process publishes them, or once per millisecond
    # of records, as Tracking.jl's `track!` leaves them for its consumer: the engine
    # decodes exactly the bits each record added, either way.
    runs = map((true, false)) do drain_per_record
        rx = PipelineReceiver(; num_sats = 6, drain_per_record)
        results, _, diverged = run_pipeline!(rx, 28.0)
        @test !diverged
        rx, results
    end
    (rx1, results1), (rx2, results2) = runs
    @test any(r -> r.status.enabled, results1)
    @test [r.pvt.position for r in results1] == [r.pvt.position for r in results2]
    for (a, b) in zip(rx1.vt.groups[1].slots, rx2.vt.groups[1].slots)
        @test a.running_decoder.num_bits_after_valid_syncro_sequence ==
              b.running_decoder.num_bits_after_valid_syncro_sequence
    end
end

@testset "The engine reads the C/N₀ estimator the host configured: $name" for (name, cn0_estimator) in (
    ("moments", () -> MomentsCN0Estimator(100)),
    ("NWPR", () -> NWPRCN0Estimator()),
)
    rx = PipelineReceiver(; num_sats = 6, cn0_estimator)
    results, _, diverged = run_pipeline!(rx, 28.0)
    @test !diverged
    @test any(r -> r.status.enabled, results)
    for sat in rx.sats
        @test sat.loop.cn0_estimator isa typeof(cn0_estimator())
        report = satellite_report(rx.estimator, GPSL1CA(), sat.decoder.prn)
        @test report.in_lock
        @test 40 < report.cn0_dbhz < 50
    end
end

@testset "Before the noise reference is ready a satellite is out of lock" begin
    # The noise-referenced C/N₀ estimator reads -Inf dB-Hz until the host has a noise
    # density for the signal: every satellite decodes, but none is in lock, so there
    # is no fix until the reference is there.
    rx = PipelineReceiver(; num_sats = 6, noise_ready_after = 28.0)
    results, sample, diverged = run_pipeline!(rx, 27.9)
    @test !diverged
    @test all(r -> isempty(r.measured) && !r.status.enabled, results)
    for sat in rx.sats
        report = satellite_report(rx.estimator, GPSL1CA(), sat.decoder.prn)
        @test report.bit_synced && !report.in_lock && !report.pvt_ready
        @test report.cn0_dbhz == -Inf
        @test TL.is_decoding_completed_for_positioning(report.decoder)
    end
    results, _, diverged = run_pipeline!(rx, 1.0; start_sample = sample)
    @test !diverged
    @test any(r -> r.status.enabled, results)
end

@testset "One host loop drives $name" for (name, estimator, pilot) in (
    ("a scalar loop on pairs", ConventionalAssistedPLLAndDLL(), true),
    ("a scalar loop on plain signals", NCOReferencedPLLAndDLL(), false),
    ("the vector loop on plain signals", nothing, false),
    ("the vector loop on pairs", nothing, true),
)
    # `pipeline_record!` calls the same functions on every record of every signal,
    # whichever estimator it was given.
    rx = PipelineReceiver(; num_sats = 4, pilot, estimator)
    results, _, diverged = run_pipeline!(rx, 3.0)
    @test !diverged
    if rx.estimator isa VectorPLLAndDLL
        @test length(results) == 30
    else
        @test isempty(results)
        @test navigation_solution(rx.estimator) === nothing
        @test navigation_cycle(rx.estimator) === nothing
    end
end
