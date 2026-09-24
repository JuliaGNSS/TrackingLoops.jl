epl(p) = EarlyPromptLateCorrelator(SVector{3,ComplexF64}(0.5p, p, 0.5p), 0.5)

# GPS L1 C/A at 4 MHz: one code block is 4000 samples, 20 blocks form a bit.
function synced_l1ca_state(; kwargs...)
    signal = GPSL1CA()
    state = SignalLoopState(signal; kwargs...)
    for k = 1:200
        p = 2000.0 * (isodd(div(k - 1, 20)) ? -1.0 : 1.0)
        state, = apply_record(state, signal, 7, CorrelatorOutput(epl(p), 4000, 4000k), 4e6Hz, 1e-6 / Hz, true)
    end
    @test has_bit_or_secondary_code_been_found(state)
    state
end

@testset "A record that crosses the bit boundary is reported, not logged" begin
    signal = GPSL1CA()
    state = synced_l1ca_state()
    # Three-block records walk the bit accumulator past the 20-block boundary
    # within a bit.
    reports = Bool[]
    for k = 1:7
        output = CorrelatorOutput(epl(2000.0), 12_000, 800_000 + 12_000k)
        was_synced = has_bit_or_secondary_code_been_found(state)
        state, _, _, _, overshoot = @test_logs apply_record(state, signal, 7, output, 4e6Hz, 1e-6 / Hz, true)
        push!(reports, overshoot)
        @test overshoot == (was_synced && !has_bit_or_secondary_code_been_found(state))
        overshoot && break
    end
    @test last(reports)
    @test count(reports) == 1
    @test !has_bit_or_secondary_code_been_found(state)
end

@testset "A secondary code is found through apply_record" begin
    # Galileo E1C: a 4 ms primary code under a 25-chip secondary code, one
    # record per primary code period at 4.092 MHz.
    signal = GalileoE1C()
    code = get_secondary_code(signal).code
    n = 16_368
    state = SignalLoopState(signal)
    for k = 1:100
        p = 2000.0 * code[mod1(k + 7, 25)]
        state, _, _, blocks = apply_record(state, signal, 11, CorrelatorOutput(epl(p), n, n * k), 4.092e6Hz, 1e-6 / Hz, true)
        @test blocks == 1
    end
    @test has_bit_or_secondary_code_been_found(state)
end

@testset "Re-arming a channel forgets the previous satellite" begin
    for cn0_estimator in (NWPRCN0Estimator(; num_records = 20), MomentsCN0Estimator(20), NoiseRefCN0Estimator(; num_records = 20))
        state = synced_l1ca_state(; cn0_estimator)
        fresh = reset_signal_state(state)
        @test !has_bit_or_secondary_code_been_found(fresh)
        @test isempty(get_soft_bits(fresh))
        @test fresh.last_filtered_prompt == 0
        # The fresh estimator is indistinguishable from a newly built one.
        new = cn0_estimator
        for name in fieldnames(typeof(new))
            name === :fallback && continue
            @test getfield(fresh.cn0_estimator, name) == getfield(new, name)
        end
        if fresh.cn0_estimator isa NWPRCN0Estimator
            @test state.cn0_estimator.filled_ratio_length > 0
            @test fresh.cn0_estimator.filled_ratio_length == 0
            @test fresh.cn0_estimator.fallback.buffer_current_index == 0
        end
        # And it reuses the old state's vectors instead of allocating new ones.
        @test fresh.bit_buffer.soft_bits === state.bit_buffer.soft_bits
    end
end

struct HistoryCN0Estimator <: AbstractCN0Estimator end

@testset "Re-arming refuses a C/N₀ estimator it cannot reset" begin
    state = SignalLoopState(GPSL1CA(); cn0_estimator = HistoryCN0Estimator())
    @test_throws ArgumentError reset_signal_state(state)
end
