epl(p) = EarlyPromptLateCorrelatorBank(SVector{3,ComplexF64}(0.5p, p, 0.5p), 0.5)

# GPS L1 C/A at 4 MHz: one code period is 4000 samples, 20 code periods form a bit.
function synced_l1ca_state(; kwargs...)
    signal = GPSL1CA()
    state = PostCorrelationState(signal; kwargs...)
    for k = 1:200
        p = 2000.0 * (isodd(div(k - 1, 20)) ? -1.0 : 1.0)
        state, = process_correlator_bank_output!(state, signal, 7, CorrelatorBankOutput(epl(p), 4000, 4000k), 4e6Hz, 1e-6 / Hz, true)
    end
    @test is_bit_or_secondary_code_synced(state)
    state
end

@testset "An integration that crosses the bit boundary is reported, not logged" begin
    signal = GPSL1CA()
    state = synced_l1ca_state()
    # Three-period integrations walk the bit accumulator past the 20-period boundary
    # within a bit.
    reports = Bool[]
    for k = 1:7
        output = CorrelatorBankOutput(epl(2000.0), 12_000, 800_000 + 12_000k)
        was_synced = is_bit_or_secondary_code_synced(state)
        state, _, _, _, overshoot = @test_logs process_correlator_bank_output!(state, signal, 7, output, 4e6Hz, 1e-6 / Hz, true)
        push!(reports, overshoot)
        @test overshoot == (was_synced && !is_bit_or_secondary_code_synced(state))
        overshoot && break
    end
    @test last(reports)
    @test count(reports) == 1
    @test !is_bit_or_secondary_code_synced(state)
end

@testset "A secondary code is found through process_correlator_bank_output!" begin
    # Galileo E1C: a 4 ms primary code under a 25-chip secondary code, one
    # integration per primary code period at 4.092 MHz.
    signal = GalileoE1C()
    code = get_secondary_code(signal).code
    n = 16_368
    state = PostCorrelationState(signal)
    for k = 1:100
        p = 2000.0 * code[mod1(k + 7, 25)]
        state, _, _, blocks = process_correlator_bank_output!(state, signal, 11, CorrelatorBankOutput(epl(p), n, n * k), 4.092e6Hz, 1e-6 / Hz, true)
        @test blocks == 1
    end
    @test is_bit_or_secondary_code_synced(state)
end

@testset "Re-arming a channel forgets the previous satellite" begin
    for cn0_estimator in (NWPRCN0Estimator(; num_integrations = 20), MomentsCN0Estimator(; num_integrations = 20), NoiseReferencedCN0Estimator(; num_integrations = 20))
        state = synced_l1ca_state(; cn0_estimator)
        fresh = reset_post_correlation!(state)
        @test !is_bit_or_secondary_code_synced(fresh)
        @test isempty(get_soft_bits(fresh))
        @test fresh.last_filtered_prompt == 0
        # The fresh estimator is indistinguishable from a newly built one.
        new = cn0_estimator
        for name in fieldnames(typeof(new))
            name === :fallback && continue
            @test getfield(fresh.cn0_estimator, name) == getfield(new, name)
        end
        if fresh.cn0_estimator isa NWPRCN0Estimator
            @test state.cn0_estimator.num_filled > 0
            @test fresh.cn0_estimator.num_filled == 0
            @test fresh.cn0_estimator.fallback.last_write_index == 0
        end
        # And it reuses the old state's vectors instead of allocating new ones.
        @test fresh.bit_synchronizer.soft_bits === state.bit_synchronizer.soft_bits
    end
end

struct HistoryCN0Estimator <: AbstractCN0Estimator end

@testset "Re-arming refuses a C/N₀ estimator it cannot reset" begin
    state = PostCorrelationState(GPSL1CA(); cn0_estimator = HistoryCN0Estimator())
    @test_throws ArgumentError reset_post_correlation!(state)
end
