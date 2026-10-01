# The navigation-bit buffer and its soft / hard sync detectors, driven directly
# (ported from Tracking.jl's `test/bit_buffer.jl`).

using Random: MersenneTwister
using TrackingLoops:
    PhaseAccumulators,
    _norm_quantile,
    _t_quantile,
    _cfar_decide,
    _detect_bit_edge_cfar,
    _detect_secondary_code_cfar,
    _seed_phase_accumulators!,
    _update_phase_accumulators!,
    _update_secondary_accumulators!,
    _secondary_code_search,
    _packed_secondary_code,
    _advance_secondary_phase

@testset "A SyncResult carries found, phase and polarity" begin
    r = @inferred SyncResult(false, 0, Int8(0))
    @test r.found == false
    @test r.phase == 0
    @test r.polarity == 0
end

@testset "The normal quantile matches tabulated values" begin
    @test _norm_quantile(0.5) == 0.0
    @test _norm_quantile(0.975) ≈ 1.959964 atol = 1e-6
    @test _norm_quantile(0.999) ≈ 3.090232 atol = 1e-6
    @test _norm_quantile(0.025) ≈ -_norm_quantile(0.975) atol = 1e-12
    # The open-interval contract: `_t_quantile` clamps before calling in, so
    # the ±Inf endpoints are documented behaviour rather than an error.
    @test _norm_quantile(1.0) == Inf
    @test _norm_quantile(0.0) == -Inf
end

# `@allocated` at global scope picks up boxing from untyped global lookups, so
# the measurement goes through a typed function.
_bb_measure_t_quantile_alloc(probability::Float64, dof::Int) =
    (_t_quantile(probability, dof); @allocated _t_quantile(probability, dof))

@testset "The Student-t quantile matches tables and is allocation-free" begin
    # Median is exactly 0, and returns directly rather than self-recursing.
    @test _t_quantile(0.5, 1) === 0.0
    @test _t_quantile(0.5, 30) === 0.0
    # dof = 1 is the standard Cauchy, quantile(p) = tan(π(p − 0.5)).
    @test _t_quantile(0.75, 1) ≈ 1.0 atol = 1e-6
    @test _t_quantile(0.9, 1) ≈ tan(pi * 0.4) atol = 1e-6
    # Symmetric about 0.
    @test _t_quantile(0.25, 1) ≈ -_t_quantile(0.75, 1) atol = 1e-9
    @test _t_quantile(0.001, 7) ≈ -_t_quantile(0.999, 7) atol = 1e-6
    # Accuracy against standard Student-t tables: exact closed forms at
    # dof 1 and 2, Hill's series (≲2e-4 relative) above.
    @test _t_quantile(0.975, 1) ≈ 12.706205 rtol = 1e-7
    @test _t_quantile(0.975, 2) ≈ 4.302653 rtol = 1e-7
    @test _t_quantile(0.995, 3) ≈ 5.840909 rtol = 1e-4
    @test _t_quantile(0.95, 5) ≈ 2.015048 rtol = 1e-4
    @test _t_quantile(0.975, 10) ≈ 2.228139 rtol = 1e-4
    @test _t_quantile(0.999, 20) ≈ 3.551808 rtol = 1e-4
    @test _t_quantile(0.975, 30) ≈ 2.042272 rtol = 1e-4
    @test _t_quantile(0.975, 120) ≈ 1.979930 rtol = 1e-4
    # The deep-tail branch of Hill's series (small `y`): t(0.9995; 3).
    @test _t_quantile(0.9995, 3) ≈ 12.924 rtol = 1e-3
    # Heavier tails than the normal, converging to it as dof → ∞.
    p = 1 - (1 - 0.999) / (20 - 1)   # the detector's Bonferroni-split argument, L1 C/A
    normal_limit = 3.87813           # tabulated Φ⁻¹(p)
    @test _t_quantile(p, 1) > _t_quantile(p, 10) > _t_quantile(p, 1000) > normal_limit
    @test _t_quantile(p, 100_000) ≈ normal_limit atol = 1e-3
    @test _t_quantile(p, 1) > 1000
    @test _t_quantile(p, 69) < 4.2
    # Monotone in dof over the range the detector sweeps — no branch seam.
    @test issorted([_t_quantile(p, dof) for dof = 1:400]; rev = true)
    @test all(_bb_measure_t_quantile_alloc(p, dof) == 0 for dof = 1:400)
    @test _bb_measure_t_quantile_alloc(0.25, 7) == 0
    # The clamped endpoint still produces a finite threshold rather than NaN.
    @test isfinite(_t_quantile(prevfloat(1.0), 1))
    @test isfinite(_t_quantile(prevfloat(1.0), 40))
end

const BB_L1CA_BLOCKS_PER_BIT = 20

# A noiseless ±1 soft-prompt stream from a list of data bits, one bit = 20
# blocks, scaled by `amp`.
_bb_bitstream(bits; amp = 1.0) =
    ComplexF64[amp * (b == 1 ? 1.0 : -1.0) for b in bits for _ = 1:BB_L1CA_BLOCKS_PER_BIT]

# Fold a prompt stream into fresh phase accumulators (as `_buffer_find_bit`
# does) and run the bit-edge detector after each block. `upto = 0` returns the
# 1-based block of the first lock (0 if never); otherwise the SyncResult after
# exactly `upto` blocks.
function _bb_detect_over(prompts, confidence; upto = 0)
    accumulators = PhaseAccumulators()
    _seed_phase_accumulators!(accumulators, BB_L1CA_BLOCKS_PER_BIT)
    for (block_number, prompt) in enumerate(prompts)
        _update_phase_accumulators!(
            accumulators,
            ComplexF64(prompt),
            block_number - 1,
            BB_L1CA_BLOCKS_PER_BIT,
        )
        result =
            _detect_bit_edge_cfar(accumulators, BB_L1CA_BLOCKS_PER_BIT, confidence, block_number)
        upto == 0 && result.found && return block_number
        upto == block_number && return result
    end
    return upto == 0 ? 0 :
           _detect_bit_edge_cfar(accumulators, BB_L1CA_BLOCKS_PER_BIT, confidence, length(prompts))
end

@testset "The soft bit-edge detector needs at least two bins" begin
    @test _bb_detect_over(_bb_bitstream([0, 1])[1:39], 0.999; upto = 39).found == false
end

@testset "A noiseless bit-edge lock fires at the true bit boundary, not one early" begin
    # Data 0,0,1: the first transition is preceded by a repeated bit (the
    # issue-#124 trigger). The true edge is at block 60.
    @test _bb_detect_over(_bb_bitstream([0, 0, 1])[1:59], 0.999; upto = 59).found == false
    res = _bb_detect_over(_bb_bitstream([0, 0, 1]), 0.999; upto = 60)
    @test res.found == true
    @test res.phase == 0
    @test res.polarity == +1
    res = _bb_detect_over(_bb_bitstream([1, 1, 0]), 0.999; upto = 60)
    @test res.found == true
    @test res.polarity == -1
end

@testset "A bit stream without a data transition never locks" begin
    @test _bb_detect_over(_bb_bitstream(fill(1, 5)), 0.999) == 0
end

@testset "A higher bit-edge confidence never locks earlier in noise" begin
    function lock_block(confidence, seed)
        rng = MersenneTwister(seed)
        clean = _bb_bitstream([bit % 2 for bit = 0:39]; amp = 8.0)
        noisy = ComplexF64[prompt + complex(randn(rng), randn(rng)) for prompt in clean]
        _bb_detect_over(noisy, confidence)
    end
    for seed = 1:5
        low_confidence_lock = lock_block(0.95, seed)
        high_confidence_lock = lock_block(0.99999, seed)
        @test low_confidence_lock > 0
        @test high_confidence_lock > 0
        @test high_confidence_lock >= low_confidence_lock
        @test low_confidence_lock % BB_L1CA_BLOCKS_PER_BIT == 0
        @test high_confidence_lock % BB_L1CA_BLOCKS_PER_BIT == 0
    end
end

@testset "A bit-edge confidence of 1.0 stays conservative rather than locking instantly" begin
    rng = MersenneTwister(7)
    clean = _bb_bitstream([bit % 2 for bit = 0:39]; amp = 3.0)
    noisy = ComplexF64[prompt + complex(randn(rng), randn(rng)) for prompt in clean]
    low_confidence_lock = _bb_detect_over(noisy, 0.99)
    high_confidence_lock = _bb_detect_over(noisy, 1.0)
    @test low_confidence_lock > 0
    @test high_confidence_lock == 0 || high_confidence_lock > low_confidence_lock
end

@testset "A long near-constant run never false-locks the bit-edge detector" begin
    # Welford keeps the variance stable at large bin counts, so a tiny drift
    # with no real bit edge never reads as infinite confidence.
    accumulators = PhaseAccumulators()
    _seed_phase_accumulators!(accumulators, BB_L1CA_BLOCKS_PER_BIT)
    found = false
    for block_number = 1:5000
        _update_phase_accumulators!(
            accumulators,
            complex(1e4 * (1 + 1e-7 * (block_number - 1)), 0.0),
            block_number - 1,
            BB_L1CA_BLOCKS_PER_BIT,
        )
        if _detect_bit_edge_cfar(accumulators, BB_L1CA_BLOCKS_PER_BIT, 0.999, block_number).found
            found = true
            break
        end
    end
    @test !found
end

@testset "The CFAR decision core compares the peak against its runner-up" begin
    period = 5
    # Fewer than two bins on any hypothesis: no runner-up can exist yet.
    @test _cfar_decide([10.0, 1.0, 1.0, 1.0, 1.0], zeros(5), 2 * period - 1, period, 0.999) ==
          (false, -1, 0)
    # Noiseless separation locks at the peak.
    accepted, peak, count = _cfar_decide([10.0, 1.0, 1.0, 1.0, 1.0], zeros(5), 2 * period, period, 0.999)
    @test accepted
    @test peak == 0
    @test count == 2
    # No separation never locks.
    @test !_cfar_decide(fill(5.0, 5), zeros(5), 2 * period, period, 0.999)[1]
    # The peak need not be the hypothesis with the most energy overall: the
    # best one with a single bin is the runner-up of the best with two bins.
    accepted, peak, _ = _cfar_decide([1.0, 2.0, 1.0, 1.0, 10.0], zeros(5), 2 * period + 3, period, 0.999)
    @test peak == 1
    @test !accepted
    # A large peak variance suppresses the lock.
    mean = [10.0, 1.0, 1.0, 1.0, 1.0]
    @test _cfar_decide(mean, zeros(5), 2 * period, period, 0.999)[1]
    @test !_cfar_decide(mean, [1.0e6, 0.0, 0.0, 0.0, 0.0], 2 * period, period, 0.999)[1]
end

# The ±1 secondary chips in time order for `signal` / `prn`.
_bb_secondary_chips(signal, prn) = [
    Int(GNSSSignals.secondary_value(get_secondary_code(signal), prn, k)) for
    k = 0:(get_secondary_code_length(signal)-1)
]

# Block `i` (0-based) carries secondary chip `(i + start_chip) % N` times a
# per-period data symbol times `amp`, plus optional complex Gaussian noise.
function _bb_secondary_stream(signal, prn, nblocks; start_chip = 0, amp = 5.0, noise = 0.0, seed = 1)
    N = get_secondary_code_length(signal)
    chips = _bb_secondary_chips(signal, prn)
    rng = MersenneTwister(seed)
    data = 1
    prompts = ComplexF64[]
    for i = 0:(nblocks-1)
        chip = mod(i + start_chip, N)
        chip == 0 && (data = rand(rng, (-1, 1)))
        p = ComplexF64(amp * data * chips[chip+1])
        noise > 0 && (p += noise * complex(randn(rng), randn(rng)))
        push!(prompts, p)
    end
    prompts
end

function _bb_secondary_detect_over(prompts, signal, prn, confidence; upto = 0)
    N = get_secondary_code_length(signal)
    accumulators = PhaseAccumulators()
    _seed_phase_accumulators!(accumulators, N)
    local result
    for (block_number, prompt) in enumerate(prompts)
        _update_secondary_accumulators!(accumulators, ComplexF64(prompt), block_number - 1, N, signal, prn)
        result = _detect_secondary_code_cfar(accumulators, N, confidence, block_number)
        upto == 0 && result.found && return block_number
        upto == block_number && return result
    end
    return upto == 0 ? 0 : result
end

@testset "The correct secondary-code rotation wipes the overlay and dominates the energy" begin
    signal = GPSL5I()
    prn = 1
    N = get_secondary_code_length(signal)
    amp = 5.0
    accumulators = PhaseAccumulators()
    _seed_phase_accumulators!(accumulators, N)
    for (i, p) in enumerate(_bb_secondary_stream(signal, prn, 2N; amp))
        _update_secondary_accumulators!(accumulators, ComplexF64(p), i - 1, N, signal, prn)
    end
    energies = accumulators.mean_bin_energy
    @test argmax(energies) - 1 == 0
    @test energies[1] ≈ (N * amp)^2
    @test energies[1] > 20 * maximum(energies[2:end])
end

@testset "The soft secondary-code detector locks at the period boundary" begin
    signal = GPSL5I()
    prn = 1
    N = get_secondary_code_length(signal)
    prompts = _bb_secondary_stream(signal, prn, 2N - 1; amp = 8.0)
    @test _bb_secondary_detect_over(prompts, signal, prn, 0.999; upto = 2N - 1).found == false
    for start_chip = 0:(N-1)
        prompts = _bb_secondary_stream(signal, prn, 4N; start_chip, amp = 8.0)
        synced = _bb_secondary_detect_over(prompts, signal, prn, 0.999)
        @test synced >= 2N
        @test (start_chip + synced - 1) % N == N - 1
        res = _bb_secondary_detect_over(prompts, signal, prn, 0.999; upto = synced)
        @test res.found
        @test res.phase == 0
    end
end

@testset "The soft secondary-code polarity follows the winning period's sign" begin
    signal = GPSL5I()
    prn = 1
    N = get_secondary_code_length(signal)
    chips = _bb_secondary_chips(signal, prn)
    pos = ComplexF64[8.0 * c for c in chips]
    neg = ComplexF64[-8.0 * c for c in chips]
    @test _bb_secondary_detect_over(vcat(pos, pos, pos), signal, prn, 0.999; upto = 3N).polarity == +1
    @test _bb_secondary_detect_over(vcat(neg, neg, neg), signal, prn, 0.999; upto = 3N).polarity == -1
end

@testset "The soft secondary-code detector does not lock on pure noise" begin
    # The hard exact-match sign template accepts random 10-bit windows a few
    # percent of the time; the soft CFAR detector on the same noise never locks.
    signal = GPSL5I()
    prn = 1
    N = get_secondary_code_length(signal)
    reference = _packed_secondary_code(UInt32, signal, prn)
    hard_false_locks = 0
    soft_locks = 0
    for seed = 1:200
        rng = MersenneTwister(1000 + seed)
        noise = ComplexF64[complex(randn(rng), randn(rng)) for _ = 1:4N]
        _bb_secondary_detect_over(noise, signal, prn, 0.999) != 0 && (soft_locks += 1)
        window = rand(rng, UInt32) & UInt32((1 << N) - 1)
        _secondary_code_search(window, reference, N, 0).found && (hard_false_locks += 1)
    end
    @test soft_locks == 0
    @test hard_false_locks > 0
end

# Feed one-block prompts through `buffer` and return the 1-based block at which
# the detector locked (0 if never) together with the final bit buffer.
# With `stop_at_lock` the feed ends on the locking block.
function _bb_feed_prompts(prompts; signal = GPSL1CA(), prn = 1, stop_at_lock = false)
    bit_buffer = BitBuffer{get_code_block_buffer_type(signal)}()
    found_at = 0
    for (i, prompt) in enumerate(prompts)
        bit_buffer = TrackingLoops.buffer(signal, prn, bit_buffer, 1, prompt)
        if found_at == 0 && has_bit_or_secondary_code_been_found(bit_buffer)
            found_at = i
            stop_at_lock && break
        end
    end
    found_at, bit_buffer
end

@testset "The L1 C/A bit-edge lock through buffer is not one block early" begin
    found_at, bit_buffer = _bb_feed_prompts([fill(-1.0 + 0.0im, 40); fill(1.0 + 0.0im, 20)])
    @test found_at == 60
    @test bit_buffer.polarity == +1
    found_at, bit_buffer = _bb_feed_prompts([fill(1.0 + 0.0im, 40); fill(-1.0 + 0.0im, 20)])
    @test found_at == 60
    @test bit_buffer.polarity == -1
    found_at, _ = _bb_feed_prompts(fill(1.0 + 0.0im, 80))
    @test found_at == 0
end

@testset "A fresh bit buffer is empty and unsynced" begin
    for bit_buffer in (@inferred(BitBuffer()), @inferred(BitBuffer{UInt8}()))
        @test bit_buffer.code_block_buffer == 0
        @test bit_buffer.code_block_buffer_length == 0
        @test has_bit_or_secondary_code_been_found(bit_buffer) == false
        @test bit_buffer.secondary_phase == 0
        @test bit_buffer.polarity == 0
        @test isempty(get_soft_bits(bit_buffer))
        @test get_soft_bits(bit_buffer) isa Vector{Float32}
        @test length(bit_buffer) == 0
        @test bit_buffer.prompt_accumulator == complex(0, 0)
        @test bit_buffer.prompt_accumulator_integrated_code_blocks == 0
    end
    @test BitBuffer() isa BitBuffer{UInt128}
end

@testset "Buffering more than one block before sync is an error" begin
    @test_throws "The number code blocks must be equal to 1" TrackingLoops.buffer(
        GPSL1CA(),
        1,
        BitBuffer(),
        2,
        2 + 0im,
    )
end

@testset "A pre-sync block shifts its prompt sign into the search window" begin
    next_bit_buffer = @inferred TrackingLoops.buffer(GPSL1CA(), 1, BitBuffer(), 1, 2 + 0im)
    @test length(next_bit_buffer) == 0
    @test next_bit_buffer.code_block_buffer == 1
    @test next_bit_buffer.code_block_buffer_length == 1
end

@testset "The pre-sync bits are recovered with the lock polarity applied" begin
    # Block signs +,+,- lock at negative polarity and decode as soft -,-,+;
    # the whole-signal sign flip locks at positive polarity and decodes the same.
    for (prompts, polarity) in (
        ([fill(1.0 + 0.0im, 40); fill(-1.0 + 0.0im, 20)], -1),
        ([fill(-1.0 + 0.0im, 40); fill(1.0 + 0.0im, 20)], +1),
    )
        found_at, bit_buffer = _bb_feed_prompts(prompts)
        @test found_at == 60
        @test bit_buffer.found == true
        @test bit_buffer.polarity == polarity
        @test length(bit_buffer) == 3
        @test bit_buffer.code_block_buffer_length == 60
        soft = get_soft_bits(bit_buffer)
        @test soft[1] < 0 && soft[2] < 0 && soft[3] > 0
    end
end

@testset "The recovered pre-sync soft bits are scaled by the prompt amplitude" begin
    _, bit_buffer = _bb_feed_prompts([fill(2.0 + 0.0im, 40); fill(-2.0 + 0.0im, 20)])
    @test get_soft_bits(bit_buffer) == Float32[-40.0, -40.0, 40.0]
    _, unit_buffer = _bb_feed_prompts([fill(1.0 + 0.0im, 40); fill(-1.0 + 0.0im, 20)])
    @test get_soft_bits(unit_buffer) == Float32[-20.0, -20.0, 20.0]
end

@testset "A post-sync prompt accumulates until the bit completes" begin
    code_blocks_buffer = 0xfffffffffff0000
    code_blocks_buffer_length = ndigits(code_blocks_buffer; base = 2)
    signal = GPSL1CA()

    bit_buffer = BitBuffer(code_blocks_buffer, code_blocks_buffer_length, true, complex(-1, 0), 1)
    next_bit_buffer = @inferred TrackingLoops.buffer(signal, 1, bit_buffer, 1, -2 + 0im)
    @test isempty(get_soft_bits(next_bit_buffer))
    @test next_bit_buffer.prompt_accumulator == -3 + 0im
    @test next_bit_buffer.prompt_accumulator_integrated_code_blocks == 2

    # The 20th block completes the bit; the soft bit is the real part of the sum
    # and lands after the seeded soft bits.
    bit_buffer = BitBuffer(
        code_blocks_buffer,
        code_blocks_buffer_length,
        true,
        complex(-10.0, 2.0),
        19,
        Float32[-1.0, 1.0],
    )
    next_bit_buffer = @inferred TrackingLoops.buffer(signal, 1, bit_buffer, 1, -2 + 0im)
    @test get_soft_bits(next_bit_buffer) == Float32[-1.0, 1.0, -12.0]
    @test next_bit_buffer.prompt_accumulator == 0 + 0im
    @test next_bit_buffer.prompt_accumulator_integrated_code_blocks == 0

    # A multi-block record completes the bit as well.
    bit_buffer = BitBuffer(
        code_blocks_buffer,
        code_blocks_buffer_length,
        true,
        complex(10, 2),
        10,
        Float32[1.0, 1.0],
    )
    next_bit_buffer = @inferred TrackingLoops.buffer(signal, 1, bit_buffer, 10, 10 + 1im)
    @test get_soft_bits(next_bit_buffer) == Float32[1.0, 1.0, 20.0]
    @test next_bit_buffer.prompt_accumulator_integrated_code_blocks == 0
end

@testset "A negative-polarity lock flips the post-sync soft bits" begin
    _, bit_buffer = _bb_feed_prompts([fill(1.0 + 0.0im, 40); fill(-1.0 + 0.0im, 20)])
    @test bit_buffer.polarity == -1
    bit_buffer = TrackingLoops.buffer(GPSL1CA(), 1, bit_buffer, 20, -20.0 + 0.0im)
    @test last(get_soft_bits(bit_buffer)) == 20.0f0
end

@testset "A record that carries the accumulator past the bit boundary drops sync" begin
    signal = GPSL1CA()
    soft_bits = Float32[1.0, -1.0]
    bit_buffer = BitBuffer(UInt64(0xff), 40, true, complex(18.0, 0.0), 18, soft_bits)
    resynced = TrackingLoops.buffer(signal, 1, bit_buffer, 3, 3.0 + 0.0im)
    @test !has_bit_or_secondary_code_been_found(resynced)
    @test resynced.code_block_buffer == 0
    @test resynced.code_block_buffer_length == 0
    @test resynced.polarity == 0
    @test resynced.prompt_accumulator == 0
    @test resynced.prompt_accumulator_integrated_code_blocks == 0
    # The bits completed before the bad record are kept, in the same vector.
    @test get_soft_bits(resynced) === soft_bits
    @test soft_bits == Float32[1.0, -1.0]

    # A synced soft-detector buffer also has its hypothesis statistics zeroed,
    # so the next pre-sync block starts a clean search.
    _, synced = _bb_feed_prompts([fill(1.0 + 0.0im, 40); fill(-1.0 + 0.0im, 20)])
    @test any(!iszero, synced.phase_acc.mean_bin_energy)
    resynced = TrackingLoops.buffer(signal, 1, synced, 21, 1.0 + 0.0im)
    @test !has_bit_or_secondary_code_been_found(resynced)
    @test length(resynced.phase_acc.mean_bin_energy) == BB_L1CA_BLOCKS_PER_BIT
    @test all(iszero, resynced.phase_acc.mean_bin_energy)
    @test all(iszero, resynced.phase_acc.last_bin_polarity)
end

@testset "Bit accumulation is unbounded" begin
    signal = GPSL1CA()
    bit_buffer = BitBuffer(UInt128(0), 0, true, complex(0.0, 0.0), 0)
    for _ = 1:(200*20)
        bit_buffer = TrackingLoops.buffer(signal, 1, bit_buffer, 1, 1.0 + 0.0im)
    end
    @test length(bit_buffer) == 200
    @test all(>(0), get_soft_bits(bit_buffer))
end

@testset "Reset empties the soft bits but keeps the vector and the lock" begin
    _, bit_buffer = _bb_feed_prompts([fill(1.0 + 0.0im, 40); fill(-1.0 + 0.0im, 20)])
    soft_bits = get_soft_bits(bit_buffer)
    @test !isempty(soft_bits)
    reset_bit_buffer = TrackingLoops.reset(bit_buffer)
    @test isempty(get_soft_bits(reset_bit_buffer))
    @test get_soft_bits(reset_bit_buffer) === soft_bits
    @test has_bit_or_secondary_code_been_found(reset_bit_buffer)
    @test reset_bit_buffer.polarity == bit_buffer.polarity
end

@testset "A secondary-code lock through buffer seeds the bit accumulator with the phase" begin
    # GPS L5I: NH10 under 50 bps data — one bit is one NH10 period. Start the
    # stream three chips into the period; the soft detector locks at a period
    # boundary, so the upcoming integration is chip 0 and no pre-sync bits are
    # recovered from the overlay-modulated signs.
    signal = GPSL5I()
    N = get_secondary_code_length(signal)
    prompts = _bb_secondary_stream(signal, 1, 6N; start_chip = 3, amp = 8.0)
    found_at, bit_buffer = _bb_feed_prompts(prompts; signal, stop_at_lock = true)
    @test found_at > 0
    @test (3 + found_at - 1) % N == N - 1
    @test bit_buffer.secondary_phase == 0
    @test bit_buffer.prompt_accumulator_integrated_code_blocks == 0
    @test isempty(get_soft_bits(bit_buffer))
    @test bit_buffer.polarity in (-1, 1)
    # One whole bit later the first post-sync soft bit is emitted.
    bit_buffer = TrackingLoops.buffer(signal, 1, bit_buffer, N, 8.0 * N + 0.0im)
    @test length(get_soft_bits(bit_buffer)) == 1
end

@testset "A hard secondary-code lock through buffer leaves a pilot's post-sync buffer alone" begin
    # GPS L1C-P: the 1800-chip overlay on a dataless pilot, hard rotation sweep.
    # Feed exactly one overlay period of prompt signs starting at chip 0.
    signal = GPSL1C_P()
    prn = 1
    chips = _bb_secondary_chips(signal, prn)
    found_at, bit_buffer = _bb_feed_prompts(ComplexF64.(chips); signal, prn)
    @test found_at == 1800
    @test bit_buffer.polarity == +1
    @test bit_buffer.secondary_phase == 0
    @test bit_buffer isa BitBuffer{TrackingLoops.UInt1800}
    # Post-sync a pilot has no data bits to decode: the buffer is returned as is.
    @test TrackingLoops.buffer(signal, prn, bit_buffer, 5, 1.0 + 0.0im) === bit_buffer
end

@testset "A one-symbol-per-code-block signal locks on its first block" begin
    # GPS L2CM: the hard detector reports found immediately, and the single
    # buffered block is recovered as the first soft bit.
    signal = GPSL2CM()
    bit_buffer = TrackingLoops.buffer(signal, 1, BitBuffer{UInt8}(), 1, -3.0 + 0.0im)
    @test has_bit_or_secondary_code_been_found(bit_buffer)
    @test bit_buffer.polarity == +1
    @test get_soft_bits(bit_buffer) == Float32[-3.0]
    bit_buffer = TrackingLoops.buffer(signal, 1, bit_buffer, 1, 2.0 + 0.0im)
    @test get_soft_bits(bit_buffer) == Float32[-3.0, 2.0]
end

@testset "A dataless signal without overlay locks without recovering bits" begin
    # Galileo E5a-QP: neither data nor a secondary code.
    signal = GalileoE5aQP()
    bit_buffer = TrackingLoops.buffer(signal, 1, BitBuffer{UInt8}(), 1, 1.0 + 0.0im)
    @test has_bit_or_secondary_code_been_found(bit_buffer)
    @test bit_buffer.secondary_phase == 0
    @test isempty(get_soft_bits(bit_buffer))
    @test TrackingLoops.buffer(signal, 1, bit_buffer, 31, 1.0 + 0.0im) === bit_buffer
end

@testset "A dataless signal without a sync feature never locks" begin
    signal = GPSL2CL()
    _, bit_buffer = _bb_feed_prompts(fill(1.0 + 0.0im, 50); signal)
    @test !has_bit_or_secondary_code_been_found(bit_buffer)
    @test bit_buffer.code_block_buffer_length == 50
end

@testset "The secondary phase advances modulo the secondary-code length" begin
    bit_buffer = BitBuffer(UInt32(0), 0, true, complex(0.0, 0.0), 0)
    advanced = _advance_secondary_phase(GPSL5I(), bit_buffer, 13)
    @test advanced.secondary_phase == 3
    @test _advance_secondary_phase(GPSL5I(), advanced, 7).secondary_phase == 0
    @test advanced.found
    # A signal without a secondary code leaves the buffer untouched.
    @test _advance_secondary_phase(GPSL1CA(), bit_buffer, 13) === bit_buffer
end

@testset "The hard secondary-code search rejects windows past the error budget" begin
    reference = _packed_secondary_code(UInt32, GPSL5I(), 1)
    @test _secondary_code_search(reference, reference, 10, 0) == SyncResult(true, 0, Int8(1))
    @test _secondary_code_search(reference ⊻ UInt32(1), reference, 10, 0).found == false
    @test _secondary_code_search(reference ⊻ UInt32(1), reference, 10, 1).found == true
    # Bits above the N-bit window are masked off.
    @test _secondary_code_search(reference | (UInt32(1) << 20), reference, 10, 0).found
end
