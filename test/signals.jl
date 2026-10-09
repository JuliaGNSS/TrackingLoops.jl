# Per-signal sync traits and detectors for GPS, Galileo and BeiDou (ported from
# Tracking.jl's `test/gps_*.jl`, `test/galileo_*.jl`, `test/beidou_*.jl` and
# `test/signal_coverage.jl`; the `track`-based parts live in Tracking.jl).

using TrackingLoops: detect_bit_or_secondary_code_sync, _packed_secondary_code, UInt1800
using Unitful: ms, s
using Random: MersenneTwister, randperm
using InteractiveUtils: subtypes

# Rotate the low `N` bits of `x` left by `r` (emulates a prompt buffer whose
# upcoming integration sits `r` secondary chips into the period).
_sig_rotl(x::T, r, N) where {T} =
    r == 0 ? x : ((x << r) | (x >> (N - r))) & ((one(T) << N) - one(T))
# The same for the exact-width 1800-chip buffer, where the mask is all ones.
_sig_rotl1800(x, r) = r == 0 ? x : ((x << r) | (x >> (1800 - r)))
const SIG_ALL_ONES_1800 = (UInt1800(1) << 1799) | ((UInt1800(1) << 1799) - one(UInt1800))

# Flip `n` distinct random bits of an 1800-chip buffer.
function _sig_flip1800(x, n, rng)
    for idx in randperm(rng, 1800)[1:n]
        x ⊻= UInt1800(1) << (idx - 1)
    end
    x
end

# A signal that broadcasts one symbol per primary code period reports `found`
# from the very first call, whatever the buffer holds.
function _sig_test_symbol_is_code_block(signal, prn, B)
    for (bits, n) in ((B(0x0), 0), (B(0x1), 1), (B(0xff), 32))
        res = @inferred detect_bit_or_secondary_code_sync(signal, prn, bits, n)
        @test res.found == true
        @test res.phase == 0
        @test res.polarity == +1
    end
end

# A short-secondary-code signal: below one period the hard detector declines;
# from one period on it recovers every tested rotation at positive polarity and
# the complement at negative polarity.
function _sig_test_secondary_search(signal, prn, B, rotations)
    N = get_secondary_code_length(signal)
    @test @inferred(detect_bit_or_secondary_code_sync(signal, prn, B(0x0), N - 1)).found ==
          false
    reference = _packed_secondary_code(B, signal, prn)
    for r in rotations
        res = @inferred detect_bit_or_secondary_code_sync(
            signal,
            prn,
            _sig_rotl(reference, r, N),
            N,
        )
        @test res.found == true
        @test res.phase == r
        @test res.polarity == +1
    end
    negated = reference ⊻ ((one(B) << N) - one(B))
    res = @inferred detect_bit_or_secondary_code_sync(signal, prn, negated, N)
    @test res.found == true
    @test res.phase == 0
    @test res.polarity == -1
end

# An 1800-chip overlay pilot on the hard rotation sweep.
function _sig_test_overlay_search(signal, prn)
    @test uses_soft_secondary_code_detection(signal) == false
    for n in (0, 1, 1799)
        @test @inferred(
            detect_bit_or_secondary_code_sync(signal, prn, UInt1800(0xffffffff), n)
        ).found == false
    end
    reference = _packed_secondary_code(UInt1800, signal, prn)
    for r in (0, 137, 1799)
        res = @inferred detect_bit_or_secondary_code_sync(
            signal,
            prn,
            _sig_rotl1800(reference, r),
            1800,
        )
        @test res.found == true
        @test res.phase == r
        @test res.polarity == +1
    end
    res = @inferred detect_bit_or_secondary_code_sync(
        signal,
        prn,
        reference ⊻ SIG_ALL_ONES_1800,
        1800,
    )
    @test res.found == true
    @test res.phase == 0
    @test res.polarity == -1

    # Up to 2.5 % of 1800 = 45 flipped chips still lock at the unrotated phase;
    # one more rejects (the fixed seed pins that no other rotation is in range).
    max_errors = floor(Int, get_bit_edge_or_secondary_code_tolerance(signal) * 1800)
    @test max_errors == 45
    rng = MersenneTwister(42)
    for n_errors in (0, 1, 10, max_errors)
        res = detect_bit_or_secondary_code_sync(
            signal,
            prn,
            _sig_flip1800(reference, n_errors, rng),
            1800,
        )
        @test res.found == true
        @test res.phase == 0
    end
    res = detect_bit_or_secondary_code_sync(
        signal,
        prn,
        _sig_flip1800(reference, max_errors + 1, rng),
        1800,
    )
    @test res.found == false
end

function _sig_test_correlator(signal, C)
    @test @inferred(get_default_correlator(signal, NumAnts(1))) ==
          C(; num_ants = NumAnts(1))
    @test @inferred(get_default_correlator(signal, NumAnts(3))) ==
          C(; num_ants = NumAnts(3))
    @test get_default_correlator(signal) == C(; num_ants = NumAnts(1))
end

# `carrier` is the default carrier bandwidth as capped for one primary code period.
function _sig_test_bandwidths(signal, carrier)
    @test @inferred(default_wide_carrier_loop_filter_bandwidth(signal)) == 50.0Hz
    @test @inferred(default_narrow_carrier_loop_filter_bandwidth(signal)) == 18.0Hz
    @test @inferred(default_fll_assist_loop_filter_bandwidth(signal)) == 5.0Hz
    primary_period = get_code_length(signal) / get_code_frequency(signal)
    @test @inferred(
        wide_cap(default_wide_carrier_loop_filter_bandwidth(signal), primary_period)
    ) ≈ carrier
    @test @inferred(default_code_loop_filter_bandwidth(signal)) ≈ 1.0Hz
end

@testset "GPS L1 C/A uses the soft bit-edge detector" begin
    signal = GPSL1CA()
    _sig_test_correlator(signal, EarlyPromptLateCorrelator)
    _sig_test_bandwidths(signal, 50.0Hz)
    @test @inferred(get_code_block_buffer_type(signal)) === UInt64
    @test @inferred(uses_soft_bit_edge_detection(signal)) == true
    @test uses_soft_secondary_code_detection(signal) == false
    @test @inferred(get_bit_edge_detection_confidence(signal)) ≈ 0.999
end

@testset "GPS L1C-D reports one symbol per code block" begin
    signal = GPSL1C_D()
    _sig_test_symbol_is_code_block(signal, 1, UInt8)
    _sig_test_correlator(signal, VeryEarlyPromptLateCorrelator)
    _sig_test_bandwidths(signal, 9.0Hz)
    @test @inferred(get_code_block_buffer_type(signal)) === UInt8
end

@testset "GPS L1C-P locks its 1800-chip overlay with the hard rotation sweep" begin
    signal = GPSL1C_P()
    _sig_test_overlay_search(signal, 1)
    _sig_test_correlator(signal, VeryEarlyPromptLateCorrelator)
    _sig_test_bandwidths(signal, 9.0Hz)
    @test @inferred(get_code_block_buffer_type(signal)) === UInt1800
end

@testset "GPS L2CM reports one symbol per code block" begin
    signal = GPSL2CM()
    _sig_test_symbol_is_code_block(signal, 1, UInt8)
    _sig_test_correlator(signal, EarlyPromptLateCorrelator)
    _sig_test_bandwidths(signal, 4.5Hz)
    # The 1 Hz DLL default is capped to 0.9 Hz at a 20 ms integration.
    @test @inferred(
        effective_code_loop_filter_bandwidth(
            default_code_loop_filter_bandwidth(signal),
            20ms,
        )
    ) ≈ 0.9Hz
    @test @inferred(get_code_block_buffer_type(signal)) === UInt8
    @test get_band_id(get_band(signal)) == :L2
end

@testset "GPS L2CL is a dataless pilot that never syncs" begin
    signal = GPSL2CL()
    for (bits, n) in ((UInt8(0x0), 0), (UInt8(0x1), 10), (UInt8(0xff), 1000))
        @test @inferred(detect_bit_or_secondary_code_sync(signal, 1, bits, n)).found ==
              false
    end
    _sig_test_correlator(signal, EarlyPromptLateCorrelator)
    _sig_test_bandwidths(signal, 0.06Hz)
    @test @inferred(
        effective_code_loop_filter_bandwidth(
            default_code_loop_filter_bandwidth(signal),
            1.5s,
        )
    ) ≈ 0.012Hz
    @test @inferred(get_code_block_buffer_type(signal)) === UInt8
    @test get_band_id(get_band(signal)) == :L2
end

@testset "GPS L5I locks its NH10 secondary code" begin
    signal = GPSL5I()
    prn = 1
    # NH10 packed newest-first from `get_secondary_code`: 1111001010.
    @test _packed_secondary_code(UInt32, signal, prn) == UInt32(0x3ca)
    res = @inferred detect_bit_or_secondary_code_sync(signal, prn, UInt32(0x3ca), 50)
    @test res.found == true
    @test res.polarity == +1
    @test !@inferred(detect_bit_or_secondary_code_sync(signal, prn, UInt32(0x3ca), 5)).found
    res = @inferred detect_bit_or_secondary_code_sync(signal, prn, UInt32(0x035), 10)
    @test res.found == true
    @test res.polarity == -1
    _sig_test_secondary_search(signal, prn, UInt32, 0:9)
    # 2.5 % of a 10-block window floors to an exact match.
    one_chip_off = UInt32(0x3ca) ⊻ UInt32(0x1)
    @test !detect_bit_or_secondary_code_sync(signal, prn, one_chip_off, 10).found
    _sig_test_correlator(signal, EarlyPromptLateCorrelator)
    _sig_test_bandwidths(signal, 50.0Hz)
    @test @inferred(get_code_block_buffer_type(signal)) === UInt32
    @test uses_soft_secondary_code_detection(signal) == true
end

@testset "GPS L5Q locks its NH20 secondary code" begin
    signal = GPSL5Q()
    prn = 1
    @test get_secondary_code_length(signal) == 20
    _sig_test_secondary_search(signal, prn, UInt32, (0, 7, 19))
    reference = _packed_secondary_code(UInt32, signal, prn)
    @test !detect_bit_or_secondary_code_sync(signal, prn, reference ⊻ UInt32(0x1), 20).found
    _sig_test_correlator(signal, EarlyPromptLateCorrelator)
    _sig_test_bandwidths(signal, 50.0Hz)
    @test @inferred(get_code_block_buffer_type(signal)) === UInt32
    @test uses_soft_secondary_code_detection(signal) == true
end

const E1BS = (GalileoE1B(), GalileoE1B_BOC11())
@testset "E1B: a symbol per code block ($(nameof(typeof(signal))))" for signal in E1BS
    _sig_test_symbol_is_code_block(signal, 1, UInt8)
    _sig_test_correlator(signal, VeryEarlyPromptLateCorrelator)
    _sig_test_bandwidths(signal, 22.5Hz)
    @test @inferred(get_code_block_buffer_type(signal)) === UInt8
end

const E1CS = (GalileoE1C(), GalileoE1C_BOC11())
@testset "E1C locks its CS25 secondary code ($(nameof(typeof(signal))))" for signal in E1CS
    @test get_secondary_code_length(signal) == 25
    _sig_test_secondary_search(signal, 1, UInt32, (0, 11, 24))
    _sig_test_correlator(signal, VeryEarlyPromptLateCorrelator)
    _sig_test_bandwidths(signal, 22.5Hz)
    @test @inferred(get_code_block_buffer_type(signal)) === UInt32
    @test uses_soft_secondary_code_detection(signal) == true
end

@testset "Galileo E5a-I locks its CS20 secondary code" begin
    signal = GalileoE5aI()
    @test get_secondary_code_length(signal) == 20
    _sig_test_secondary_search(signal, 1, UInt32, (0, 9, 19))
    _sig_test_correlator(signal, EarlyPromptLateCorrelator)
    _sig_test_bandwidths(signal, 50.0Hz)
    @test @inferred(get_code_block_buffer_type(signal)) === UInt32
    @test uses_soft_secondary_code_detection(signal) == true
end

@testset "Galileo E5a-Q locks its per-PRN CS100 secondary code" begin
    signal = GalileoE5aQ()
    @test get_secondary_code_length(signal) == 100
    _sig_test_secondary_search(signal, 1, UInt128, (0, 37, 99))
    @test _packed_secondary_code(UInt128, signal, 1) !=
          _packed_secondary_code(UInt128, signal, 2)
    _sig_test_correlator(signal, EarlyPromptLateCorrelator)
    _sig_test_bandwidths(signal, 50.0Hz)
    @test @inferred(get_code_block_buffer_type(signal)) === UInt128
    @test uses_soft_secondary_code_detection(signal) == true
end

@testset "Galileo E5a-QP locks on its first block and integrates whole code cycles" begin
    signal = GalileoE5aQP()
    _sig_test_symbol_is_code_block(signal, 1, UInt8)
    @test get_secondary_code_length(signal) == 1
    @test uses_soft_secondary_code_detection(signal) == false
    @test uses_soft_bit_edge_detection(signal) == false
    _sig_test_correlator(signal, EarlyPromptLateCorrelator)
    @test @inferred(get_code_block_buffer_type(signal)) === UInt8
    @test max_num_code_blocks_to_integrate(signal) == 31
    @test default_num_code_blocks_to_integrate(signal) == 31
end

@testset "Galileo E5b-I locks its CS4 secondary code" begin
    signal = GalileoE5bI()
    prn = 1
    N = get_secondary_code_length(signal)
    @test N == 4
    # CS4 is `1110`, shared across SVIDs.
    reference = _packed_secondary_code(UInt32, signal, prn)
    @test reference == UInt32(0b1110)
    _sig_test_secondary_search(signal, prn, UInt32, 0:(N-1))
    # Any single flip inside the 4-bit window rejects.
    for bit = 0:(N-1)
        @test detect_bit_or_secondary_code_sync(
            signal,
            prn,
            reference ⊻ (UInt32(1) << bit),
            N,
        ).found == false
    end
    _sig_test_correlator(signal, EarlyPromptLateCorrelator)
    _sig_test_bandwidths(signal, 50.0Hz)
    @test @inferred(get_code_block_buffer_type(signal)) === UInt32
    @test uses_soft_secondary_code_detection(signal) == true
end

@testset "Galileo E5b-Q locks its CS100 secondary code" begin
    signal = GalileoE5bQ()
    @test get_secondary_code_length(signal) == 100
    _sig_test_secondary_search(signal, 1, UInt128, (0, 37, 99))
    # E5b-Q draws the upper half of the CS100 table, E5a-Q the lower.
    @test _packed_secondary_code(UInt128, signal, 1) !=
          _packed_secondary_code(UInt128, GalileoE5aQ(), 1)
    _sig_test_correlator(signal, EarlyPromptLateCorrelator)
    _sig_test_bandwidths(signal, 50.0Hz)
    @test @inferred(get_code_block_buffer_type(signal)) === UInt128
    @test uses_soft_secondary_code_detection(signal) == true
end

@testset "Galileo E6-B reports one symbol per code block" begin
    signal = GalileoE6B()
    @test get_secondary_code_length(signal) == 1
    _sig_test_symbol_is_code_block(signal, 1, UInt8)
    _sig_test_correlator(signal, EarlyPromptLateCorrelator)
    _sig_test_bandwidths(signal, 50.0Hz)
    @test @inferred(get_code_block_buffer_type(signal)) === UInt8
    @test uses_soft_secondary_code_detection(signal) == false
    @test uses_soft_bit_edge_detection(signal) == false
    @test get_carrier_phase_offset(signal) == 0.0
end

@testset "Galileo E6-C locks its CS100 secondary code" begin
    signal = GalileoE6C()
    @test get_secondary_code_length(signal) == 100
    _sig_test_secondary_search(signal, 1, UInt128, (0, 61, 99))
    # E6-C draws the same CS100 half as E5a-Q.
    @test _packed_secondary_code(UInt128, signal, 1) ==
          _packed_secondary_code(UInt128, GalileoE5aQ(), 1)
    _sig_test_correlator(signal, EarlyPromptLateCorrelator)
    _sig_test_bandwidths(signal, 50.0Hz)
    @test @inferred(get_code_block_buffer_type(signal)) === UInt128
    @test uses_soft_secondary_code_detection(signal) == true
    @test get_carrier_phase_offset(signal) ≈ π
end

@testset "BeiDou B1C data reports one symbol per code block" begin
    signal = BeiDouB1C_D()
    @test get_secondary_code_length(signal) == 1
    _sig_test_symbol_is_code_block(signal, 1, UInt8)
    @test get_band_id(signal) === :L1
    _sig_test_correlator(signal, VeryEarlyPromptLateCorrelator)
    _sig_test_bandwidths(signal, 9.0Hz)
    @test @inferred(get_code_block_buffer_type(signal)) === UInt8
    @test uses_soft_secondary_code_detection(signal) == false
    @test uses_soft_bit_edge_detection(signal) == false
end

@testset "BeiDou B1C pilot locks its 1800-chip overlay with the hard rotation sweep" begin
    signal = BeiDouB1C_P()
    @test get_secondary_code_length(signal) == 1800
    _sig_test_overlay_search(signal, 1)
    @test get_band_id(signal) === :L1
    _sig_test_correlator(signal, VeryEarlyPromptLateCorrelator)
    _sig_test_bandwidths(signal, 9.0Hz)
    @test @inferred(get_code_block_buffer_type(signal)) === UInt1800
end

const SIG_BEIDOU_MEO_PRN = 6   # MEO/IGSO (D1) — carries NH20
const SIG_BEIDOU_GEO_PRN = 1   # GEO (D2) — carries no overlay
const SIG_B1I_SYMBOL_OFFSET = 7

# Drive the live soft path for B1I with one navigation symbol every
# `blocks_per_symbol` blocks, the PRN's overlay chip on top and unit-variance
# complex noise. Returns the block and rotation of the lock, or `nothing`.
function _sig_b1i_soft_sync(prn, blocks_per_symbol; nblocks, amplitude = 3.0, seed = 1)
    signal = BeiDouB1I()
    N = get_secondary_code_length(signal)
    overlay = get_secondary_code(signal)
    rng = MersenneTwister(seed)
    accumulators = TrackingLoops.PhaseAccumulators()
    TrackingLoops._seed_phase_accumulators!(accumulators, N)
    symbol = 1.0
    for i = 0:(nblocks-1)
        (i - SIG_B1I_SYMBOL_OFFSET) % blocks_per_symbol == 0 &&
            (symbol = rand(rng, (-1.0, 1.0)))
        chip = GNSSSignals.secondary_value(overlay, prn, mod(i - SIG_B1I_SYMBOL_OFFSET, N))
        prompt = amplitude * symbol * chip + randn(rng, ComplexF64)
        TrackingLoops._update_secondary_accumulators!(
            accumulators,
            ComplexF64(prompt),
            i,
            N,
            signal,
            prn,
        )
        result = TrackingLoops._detect_secondary_code_cfar(accumulators, N, 0.999, i + 1)
        result.found && return (block = i + 1, rotation = mod(i + 1, N))
    end
    nothing
end

@testset "BeiDou B1I locks NH20 on MEO satellites and stays pre-sync on GEO" begin
    signal = BeiDouB1I()
    N = get_secondary_code_length(signal)
    @test N == 20
    reference = _packed_secondary_code(UInt32, signal, SIG_BEIDOU_MEO_PRN)
    @test reference == UInt32(0b00000100110101001110)
    _sig_test_secondary_search(signal, SIG_BEIDOU_MEO_PRN, UInt32, (0, 9, 19))

    # GEO satellites carry an all-ones column, which is rotation-invariant.
    mask = (one(UInt32) << N) - one(UInt32)
    for prn in (1, 5, 59, 63)
        @test _packed_secondary_code(UInt32, signal, prn) == mask
    end
    geo_reference = _packed_secondary_code(UInt32, signal, SIG_BEIDOU_GEO_PRN)
    @test all(_sig_rotl(geo_reference, r, N) == geo_reference for r = 0:(N-1))

    # The soft path: MEO locks at the true period boundary; the hypothetical GEO
    # at the D1 rate reduces to a bit-edge search and still finds the boundary;
    # real GEO (D2, 2 blocks per symbol) never locks.
    meo = _sig_b1i_soft_sync(SIG_BEIDOU_MEO_PRN, 20; nblocks = 2000)
    @test meo !== nothing
    @test meo.rotation == SIG_B1I_SYMBOL_OFFSET
    geo_at_d1_rate = _sig_b1i_soft_sync(SIG_BEIDOU_GEO_PRN, 20; nblocks = 4000)
    @test geo_at_d1_rate !== nothing
    @test geo_at_d1_rate.rotation == SIG_B1I_SYMBOL_OFFSET
    for seed = 1:3
        @test _sig_b1i_soft_sync(SIG_BEIDOU_GEO_PRN, 2; nblocks = 20000, seed) === nothing
    end

    @test uses_soft_secondary_code_detection(signal) == true
    @test uses_soft_bit_edge_detection(signal) == false
    _sig_test_correlator(signal, EarlyPromptLateCorrelator)
    _sig_test_bandwidths(signal, 50.0Hz)
    @test @inferred(get_code_block_buffer_type(signal)) === UInt32
end

@testset "BeiDou B2a data locks its 5-chip secondary code" begin
    signal = BeiDouB2aI()
    N = get_secondary_code_length(signal)
    @test N == 5
    @test _packed_secondary_code(UInt32, signal, 1) == UInt32(0b00010)
    _sig_test_secondary_search(signal, 1, UInt32, 0:(N-1))
    @test get_band_id(signal) === :L5
    _sig_test_correlator(signal, EarlyPromptLateCorrelator)
    _sig_test_bandwidths(signal, 50.0Hz)
    @test @inferred(get_code_block_buffer_type(signal)) === UInt32
    @test uses_soft_secondary_code_detection(signal) == true
end

@testset "BeiDou B2a pilot locks its per-PRN 100-chip secondary code" begin
    signal = BeiDouB2aQ()
    @test get_secondary_code_length(signal) == 100
    _sig_test_secondary_search(signal, 1, UInt128, (0, 44, 99))
    @test get_band_id(signal) === :L5
    _sig_test_correlator(signal, EarlyPromptLateCorrelator)
    _sig_test_bandwidths(signal, 50.0Hz)
    @test @inferred(get_code_block_buffer_type(signal)) === UInt128
    @test uses_soft_secondary_code_detection(signal) == true
end

@testset "BeiDou B2b-I reports one symbol per code block" begin
    signal = BeiDouB2bI()
    prn = 6   # the ICD defines ranging codes for PRN 6-58 only
    @test get_secondary_code_length(signal) == 1
    _sig_test_symbol_is_code_block(signal, prn, UInt8)
    @test get_band_id(signal) === :E5b
    _sig_test_correlator(signal, EarlyPromptLateCorrelator)
    _sig_test_bandwidths(signal, 50.0Hz)
    @test @inferred(get_code_block_buffer_type(signal)) === UInt8
    @test uses_soft_secondary_code_detection(signal) == false
    @test uses_soft_bit_edge_detection(signal) == false
end

@testset "BeiDou B3I locks the same NH20 as B1I" begin
    signal = BeiDouB3I()
    N = get_secondary_code_length(signal)
    @test N == 20
    reference = _packed_secondary_code(UInt32, signal, SIG_BEIDOU_MEO_PRN)
    @test reference == _packed_secondary_code(UInt32, BeiDouB1I(), SIG_BEIDOU_MEO_PRN)
    _sig_test_secondary_search(signal, SIG_BEIDOU_MEO_PRN, UInt32, (0, 13, 19))
    @test _packed_secondary_code(UInt32, signal, SIG_BEIDOU_GEO_PRN) ==
          (one(UInt32) << N) - one(UInt32)
    @test uses_soft_secondary_code_detection(signal) == true
    @test uses_soft_bit_edge_detection(signal) == false
    _sig_test_correlator(signal, EarlyPromptLateCorrelator)
    _sig_test_bandwidths(signal, 50.0Hz)
    @test @inferred(get_code_block_buffer_type(signal)) === UInt32
end

# Concrete leaves of the `AbstractGNSSSignal` tree that GNSSSignals itself
# defines; the module filter keeps out any fake signal types tests define.
function _sig_signal_types()
    leaves(T) = (subs = subtypes(T); isempty(subs) ? [T] : reduce(vcat, leaves.(subs)))
    types = filter(T -> Base.typename(T).module === GNSSSignals, leaves(AbstractGNSSSignal))
    sort(types; by = T -> string(nameof(T)))
end

const SIG_SUPPORTED = [Base.typename(T).wrapper() for T in _sig_signal_types()]

@testset "Every GNSSSignals signal type has a sync API" begin
    @test !isempty(SIG_SUPPORTED)
    for T in _sig_signal_types()
        @test hasmethod(get_default_correlator, Tuple{Base.typename(T).wrapper,NumAnts})
    end
end

@testset "Sync API complete for $(get_signal_name(signal))" for signal in SIG_SUPPORTED
    for num_ants in (1, 3)
        correlator = @inferred get_default_correlator(signal, NumAnts(num_ants))
        @test correlator isa AbstractCorrelator
        @test get_num_ants(correlator) == num_ants
    end

    # The sync-search buffer must hold one whole secondary-code period.
    B = @inferred get_code_block_buffer_type(signal)
    @test B <: Unsigned
    @test sizeof(B) * 8 >= get_secondary_code_length(signal)

    # A signal routes to at most one soft detector.
    @test !(
        uses_soft_bit_edge_detection(signal) && uses_soft_secondary_code_detection(signal)
    )
    if hasmethod(detect_bit_or_secondary_code_sync, Tuple{typeof(signal),Int,B,Int})
        @test detect_bit_or_secondary_code_sync(signal, 6, zero(B), 0) isa SyncResult
    else
        # Only the soft bit-edge path has no hard detector method.
        @test uses_soft_bit_edge_detection(signal)
    end

    # A few unsynced blocks through the bit buffer run whichever detector the
    # signal routes to, at its own buffer width.
    bit_buffer = BitBuffer{B}()
    for k = 1:3
        bit_buffer = TrackingLoops.buffer(
            signal,
            6,
            bit_buffer,
            1,
            complex(isodd(k) ? 1.0 : -1.0, 0.0),
        )
    end
    @test bit_buffer isa BitBuffer{B}
    # A lock on the first block freezes the search window at one block.
    @test bit_buffer.code_block_buffer_length ==
          (has_bit_or_secondary_code_been_found(bit_buffer) ? 1 : 3)
end
