# C/N₀ estimators, driven directly through `update` / `estimate_cn0`. Ported
# from Tracking.jl's `test/cn0_estimators/*.jl` and the estimator-only parts of
# `test/cn0_estimator_comparison.jl`; the parts that ran a `TrackState` through
# `track` / `track!` belong to Tracking and are not repeated here.

using Random: Xoshiro, randn
using Statistics: mean, std
using Unitful: ms, kHz, MHz
using TrackingLoops: get_prompt_buffer, get_current_index, get_fallback_cn0_estimator

# Float64 throughout, matching what the tracking loop produces
# (`integrated_samples / sampling_frequency`); `1ms` would make the density a
# `Rational{Int64}`, a type combination the loop never creates.
const CN0_T = 1.0ms
# `E|P|² = N₀/T`, so a unit-noise-power prompt stream is measured against `N₀ = T`.
const CN0_N₀ = uconvert(Hz^-1, CN0_T)

# Post-correlation prompt model: `λ = (C/N₀)·T` is the per-record SNR, noise is
# CN(0,1), the phase is perfect.
cn0_prompts(λ, n, rng) = sqrt(λ) .+ (randn(rng, n) .+ im .* randn(rng, n)) ./ sqrt(2)

cn0_db(x) = ustrip(uconvert(dBHz, x))

cn0_median(xs) = (v = sort(collect(xs)); v[div(length(v) + 1, 2)])

cn0_context(; noise_density = CN0_N₀, integration_time = CN0_T) =
    CN0UpdateContext(GPSL1CA(), BitBuffer{UInt64}(), 1; noise_density, integration_time)

cn0_fold(estimator, prompts) = foldl(update, prompts; init = estimator)
cn0_fold(estimator, prompts, context) =
    foldl((e, p) -> update(e, p, context), prompts; init = estimator)

function cn0_fold_many(estimator, prompt, n)
    for _ = 1:n
        estimator = update(estimator, prompt)
    end
    estimator
end

function cn0_fold_many(estimator, prompt, context, n)
    for _ = 1:n
        estimator = update(estimator, prompt, context)
    end
    estimator
end

# A minimal custom estimator: proves the abstract interface's defaults route to
# its two-argument `update` and to its own `estimate_cn0`.
struct CountingCN0Estimator <: AbstractCN0Estimator
    num_prompts::Int
end
TrackingLoops.update(estimator::CountingCN0Estimator, prompt) =
    CountingCN0Estimator(estimator.num_prompts + 1)
TrackingLoops.estimate_cn0(estimator::CountingCN0Estimator, integration_time) =
    estimator.num_prompts * dBHz

@testset "A custom CN0 estimator plugs into the interface" begin
    estimator = CountingCN0Estimator(0)
    # The three-argument form the loop calls drops the context and forwards.
    estimator = @inferred update(estimator, 1.0 + 0.0im, cn0_context())
    estimator = update(estimator, 1.0 + 0.0im)
    @test estimator.num_prompts == 2
    @test estimate_cn0(estimator, CN0_T) == 2dBHz
    # Reads no density unless it says so, on the instance and on the type.
    @test !requires_noise_density(estimator)
    @test !requires_noise_density(CountingCN0Estimator)
end

@testset "default_cn0_estimator is a noise-referenced one" begin
    default = default_cn0_estimator(GPSL1CA(), 40)
    @test default isa NoiseRefCN0Estimator
    @test default.num_records == 40
    @test Base.length(default.buffered_cn0) == 40
    @test requires_noise_density(default)
    @test requires_noise_density(default_cn0_estimator(GPSL1C_P(), 100))
    @test requires_noise_density(NoiseRefCN0Estimator)
end

@testset "CN0UpdateContext derives the bit grid" begin
    bit_buffer = BitBuffer{UInt64}()
    # Positional form: noise density and integration time default to `nothing`.
    positional = @inferred CN0UpdateContext(GPSL1CA(), 2, 20, 3, bit_buffer)
    @test positional.num_code_blocks == 2
    @test positional.num_code_blocks_per_bit == 20
    @test positional.bit_code_block_index == 3
    @test positional.noise_density === nothing
    @test positional.integration_time === nothing

    # Derived form: the grid comes from the signal and the bit buffer, and is
    # marked unknown while sync has not been found.
    context_l1ca = @inferred CN0UpdateContext(GPSL1CA(), bit_buffer, 1)
    @test context_l1ca.num_code_blocks_per_bit == 20
    @test context_l1ca.bit_code_block_index == -1
    synced_buffer = BitBuffer{UInt64}(
        zero(UInt64),
        40,
        true,
        0,
        Int8(1),
        complex(0.0, 0.0),
        3,
        Float32[],
        PhaseAccumulators(),
    )
    @test CN0UpdateContext(GPSL1CA(), synced_buffer, 1).bit_code_block_index == 3
    # Records of a fold that follow a sync detected in that same fold were
    # correlated with pre-sync replicas — their alignment is not trustworthy.
    @test CN0UpdateContext(GPSL1CA(), synced_buffer, 1, false).bit_code_block_index == -1
    # The keywords land in the context.
    with_density = CN0UpdateContext(
        GPSL1CA(),
        synced_buffer,
        1;
        noise_density = CN0_N₀,
        integration_time = CN0_T,
    )
    @test with_density.noise_density == CN0_N₀
    @test with_density.integration_time == CN0_T
end

@testset "Moments CN0 estimator" begin
    cn0_estimator = MomentsCN0Estimator(20)
    @test @inferred(get_prompt_buffer(cn0_estimator)) == zeros(ComplexF64, 20)
    @test @inferred(get_current_index(cn0_estimator)) == 0
    @test @inferred(Base.length(cn0_estimator)) == 0
    # An empty buffer reports 0 dB-Hz rather than dividing by zero.
    @test @inferred(estimate_cn0(cn0_estimator, CN0_T)) == 0.0dBHz
    @test !requires_noise_density(cn0_estimator)

    next_cn0_estimator = @inferred update(cn0_estimator, 1 + 2im)
    @test @inferred(get_prompt_buffer(next_cn0_estimator))[1] == 1 + 2im
    @test @inferred(get_current_index(next_cn0_estimator)) == 1
    @test @inferred(Base.length(next_cn0_estimator)) == 1

    # The ring wraps from the last slot back to the first.
    cn0_estimator = MomentsCN0Estimator(ones(ComplexF64, 20), 20, 20)
    next_cn0_estimator = @inferred update(cn0_estimator, 1 + 2im)
    @test @inferred(get_prompt_buffer(next_cn0_estimator))[1] == 1 + 2im
    @test @inferred(get_current_index(next_cn0_estimator)) == 1
    @test @inferred(Base.length(next_cn0_estimator)) == 20

    cn0_estimator = MomentsCN0Estimator(ones(ComplexF64, 20), 19, 20)
    @test @inferred(get_current_index(cn0_estimator)) == 19
    @test @inferred(Base.length(cn0_estimator)) == 20
end

@testset "Moments CN0 estimation" begin
    # Tracking ran this through its correlator; here the prompts are drawn from
    # the post-correlation model directly: a true 45 dB-Hz over 1 ms records.
    rng = Xoshiro(1234)
    λ = 10^(45 / 10) * 1e-3
    estimator = cn0_fold(MomentsCN0Estimator(100), cn0_prompts(λ, 100, rng))
    @test @inferred(get_current_index(estimator)) == 100
    @test @inferred(Base.length(estimator)) == 100
    cn0_estimate = @inferred estimate_cn0(estimator, 1ms)
    @test cn0_estimate ≈ 45dBHz atol = 1.0dBHz
end

@testset "Moments CN0 estimate divides by the record's integration time" begin
    # A record spanning N code blocks arrives with N times the SNR of a one-block
    # record, so the same shape at two block counts must report the same C/N₀
    # once the divisor is N × the code period.
    rng = Xoshiro(4321)
    one_block = [1.0 + 0.2 * randn(rng, ComplexF64) for _ = 1:100]
    twenty_blocks = [1.0 + 0.2 / sqrt(20) * randn(rng, ComplexF64) for _ = 1:100]
    cn0_1 = estimate_cn0(cn0_fold(MomentsCN0Estimator(100), one_block), 1ms)
    cn0_20 = estimate_cn0(cn0_fold(MomentsCN0Estimator(100), twenty_blocks), 20ms)
    @test cn0_20 ≈ cn0_1 atol = 1.5dBHz
    # And the divisor is actually used: the same prompts at 20 ms read 13 dB lower.
    same = cn0_fold(MomentsCN0Estimator(100), one_block)
    linear(x) = ustrip(Hz, Unitful.linear(x))
    @test 10 * log10(linear(estimate_cn0(same, 1ms)) / linear(estimate_cn0(same, 20ms))) ≈
          10 * log10(20) atol = 0.01
end

@testset "NoCN0Estimator measures nothing and says so" begin
    estimator = NoCN0Estimator()
    # Stateless: the same instance comes back, whichever `update` form is used.
    @test @inferred(update(estimator, 1.0 + 0.0im)) === estimator
    context = CN0UpdateContext(GPSL1CA(), BitBuffer{UInt64}(), 1)
    @test @inferred(update(estimator, 1.0 + 0.0im, context)) === estimator
    cn0_fold_many(estimator, 1.0 + 0.0im, 10)
    @test @allocated(cn0_fold_many(estimator, 1.0 + 0.0im, 1000)) == 0
    @test !requires_noise_density(estimator)

    # `-Inf dB-Hz`, not `NaN dB-Hz`: `NaN dB-Hz >= threshold` is `true` for every
    # threshold with Unitful's `Level` comparison, so a NaN would clear every lock
    # detector it met.
    @test @inferred(estimate_cn0(estimator, 1ms)) == -Inf * dBHz
    @test !(estimate_cn0(estimator, 1ms) >= 20dBHz)
    @test NaN * dBHz >= 20dBHz          # ... which is why NaN is not used

    # It is a legal NWPR `fallback`: it replaces the moment ratio's noise floor
    # with "no estimate" for the phases that admit no coherent window.
    honest = NWPRCN0Estimator(; num_narrowband_code_blocks = 20, fallback = estimator)
    @test @inferred(estimate_cn0(honest, 1ms)) == -Inf * dBHz
    for _ = 1:19
        honest = update(honest, 1.0 + 0.0im)
    end
    @test estimate_cn0(honest, 1ms) == -Inf * dBHz     # window still open
    honest = update(honest, 1.0 + 0.0im)
    @test estimate_cn0(honest, 1ms) == Inf * dBHz      # first window closed
end

@testset "NoiseRef: one record plus a density is already an estimate" begin
    @test_throws ArgumentError NoiseRefCN0Estimator(; num_records = 0)
    estimator = NoiseRefCN0Estimator(; num_records = 4)
    @test @inferred(Base.length(estimator)) == 0
    @test get_current_index(estimator) == 0
    # Empty ring: `-Inf dB-Hz`, the house convention for "not measured".
    @test @inferred(estimate_cn0(estimator, CN0_T)) == -Inf * dBHz

    # No warm-up, no window, no `M`: the very first record reports. A noiseless
    # prompt of power `p` against `N₀ = T` reads `(p − 1)/T`.
    one = @inferred update(estimator, complex(2.0, 0.0), cn0_context())
    @test Base.length(one) == 1
    @test get_current_index(one) == 1
    @test cn0_db(estimate_cn0(one, CN0_T)) ≈ 10log10((4.0 - 1.0) / 1e-3)

    # `estimate_cn0`'s `integration_time` argument is ignored — `T` was applied
    # per record, where each record's own value was known.
    @test estimate_cn0(one, 20ms) == estimate_cn0(one, CN0_T)

    # The ring wraps at `num_records` and keeps only the newest.
    filled = cn0_fold(
        NoiseRefCN0Estimator(; num_records = 4),
        fill(complex(2.0, 0.0), 10),
        cn0_context(),
    )
    @test Base.length(filled) == 4
    @test get_current_index(filled) == 2
    @test cn0_db(estimate_cn0(filled, CN0_T)) ≈ 10log10((4.0 - 1.0) / 1e-3)

    @test requires_noise_density(estimator)
    @test !requires_noise_density(NWPRCN0Estimator())
    @test !requires_noise_density(MomentsCN0Estimator(10))
    @test !requires_noise_density(NoCN0Estimator())
end

@testset "NoiseRef: records of different length are each divided by their own T" begin
    # Against a fixed `N₀` the sample-normalised prompt power of a record at
    # C/N₀ = γ is `N₀·(γ + 1/T)`, so a 20 ms record carrying the *same* γ has a
    # visibly different power.
    γ = 3000.0                                   # ≈34.8 dB-Hz
    power(t_ms) = 1e-3 * (γ + 1 / (t_ms * 1e-3))
    short = update(NoiseRefCN0Estimator(), complex(sqrt(power(1)), 0.0), cn0_context())
    long = update(
        NoiseRefCN0Estimator(),
        complex(sqrt(power(20)), 0.0),
        cn0_context(; integration_time = 20.0ms),
    )
    @test cn0_db(estimate_cn0(short, CN0_T)) ≈ 10log10(γ) atol = 1e-9
    @test cn0_db(estimate_cn0(long, 20ms)) ≈ 10log10(γ) atol = 1e-9

    # ... and a ring holding both together still averages to the same γ.
    both = update(
        short,
        complex(sqrt(power(20)), 0.0),
        cn0_context(; integration_time = 20.0ms),
    )
    @test cn0_db(estimate_cn0(both, CN0_T)) ≈ 10log10(γ) atol = 1e-9
end

@testset "NoiseRef: per-record terms are not clamped, so the mean is unbiased" begin
    rng = Xoshiro(20260806)
    noise_only = cn0_fold(
        NoiseRefCN0Estimator(; num_records = 2000),
        cn0_prompts(0.0, 2000, rng),
        cn0_context(),
    )
    terms = noise_only.buffered_cn0
    @test count(<(0), terms) > 500          # ≈63 % of a unit-mean exponential
    # ... and they cancel: one σ of the mean is `1/(√2000 · T)` ≈ 22 Hz.
    @test abs(mean(terms)) < 150
    # A whole ring of pure noise never reports a positive floor.
    @test cn0_db(estimate_cn0(noise_only, CN0_T)) < 25
end

@testset "NoiseRef: bias and σ match the pinned noise-reference columns" begin
    num_records = 100
    trials = 1500
    for (cn0, σ_bound) in ((30.0, 0.9), (40.0, 0.30), (50.0, 0.10))
        λ = 10^(cn0 / 10) * 1e-3
        rng = Xoshiro(20260806 + round(Int, cn0))
        estimates = map(1:trials) do _
            estimator = cn0_fold(
                NoiseRefCN0Estimator(; num_records),
                cn0_prompts(λ, num_records, rng),
                cn0_context(),
            )
            10^(cn0_db(estimate_cn0(estimator, CN0_T)) / 10) * 1e-3
        end
        @test all(isfinite, estimates)
        @test all(>(0), estimates)
        @test abs(10log10(mean(estimates) / λ)) < 0.05
        @test 4.342944819 * std(estimates) / λ < σ_bound
    end
end

@testset "NoiseRef: a missing source or T is a loud, static error" begin
    estimator = NoiseRefCN0Estimator()
    # No density and no T: named as the missing *source*, the root cause.
    err_both = try
        update(
            estimator,
            complex(1.0, 0.0),
            CN0UpdateContext(GPSL1CA(), BitBuffer{UInt64}(), 1),
        )
        nothing
    catch e
        e
    end
    @test err_both isa ArgumentError
    @test occursin("AbstractNoiseEstimator is configured", err_both.msg)

    # No density, but a T.
    err_density = try
        update(estimator, complex(1.0, 0.0), cn0_context(; noise_density = nothing))
        nothing
    catch e
        e
    end
    @test err_density isa ArgumentError
    @test occursin("AbstractNoiseEstimator is configured", err_density.msg)

    # A density without a T is the same kind of error, not a MethodError.
    err_time = try
        update(estimator, complex(1.0, 0.0), cn0_context(; integration_time = nothing))
        nothing
    catch e
        e
    end
    @test err_time isa ArgumentError
    @test occursin("integration time", err_time.msg)

    # And there is no bare-prompt form at all.
    @test_throws ArgumentError update(estimator, complex(1.0, 0.0))
end

@testset "NoiseRef: a NaN never leaves estimate_cn0" begin
    # A zero measured floor with a zero prompt — a front-end dropout.
    poisoned = update(
        NoiseRefCN0Estimator(; num_records = 4),
        complex(0.0, 0.0),
        cn0_context(; noise_density = 0.0 * CN0_N₀),
    )
    @test isnan(poisoned.buffered_cn0[1])                # the per-record term is NaN ...
    @test estimate_cn0(poisoned, CN0_T) == -Inf * dBHz   # ... and the estimate is not
    @test !(estimate_cn0(poisoned, CN0_T) >= 30dBHz)

    nan_prompt = update(NoiseRefCN0Estimator(), complex(NaN, 0.0), cn0_context())
    @test estimate_cn0(nan_prompt, CN0_T) == -Inf * dBHz

    # An `Inf` term (a zero floor under a non-zero prompt) is not a C/N₀ either.
    inf_term = update(
        NoiseRefCN0Estimator(),
        complex(1.0, 0.0),
        cn0_context(; noise_density = 0.0 * CN0_N₀),
    )
    @test estimate_cn0(inf_term, CN0_T) == -Inf * dBHz
end

@testset "NoiseRef: update is allocation-free in steady state and inferred" begin
    estimator = NoiseRefCN0Estimator()
    context = cn0_context()
    prompt = complex(2.0, 0.0)
    cn0_fold_many(estimator, prompt, context, 200)          # warm up
    # A genuine per-fold allocation would grow a hundredfold between the two
    # fold counts; a fixed per-call harness cost (Julia 1.10) does not move.
    few = @allocated cn0_fold_many(estimator, prompt, context, 10_000)
    many = @allocated cn0_fold_many(estimator, prompt, context, 1_000_000)
    @test many == few
    @test few <= 128
    @test @inferred(update(estimator, prompt, context)) isa NoiseRefCN0Estimator
    @test @inferred(estimate_cn0(estimator, CN0_T)) isa typeof(0.0dBHz)
end

# JuliaGNSS/Tracking.jl#217
@testset "The signals NWPR cannot serve, the noise reference can" begin
    # One symbol per code block, or a secondary code before sync: NWPR admits no
    # coherent window ever and reports its fallback; the noise reference has no
    # window, so these are ordinary records to it.
    bit_buffer = BitBuffer{UInt64}()
    cases = (
        ("GPS L1C-D, synced", GPSL1C_D(), 1, 0),
        ("Galileo E1B, synced", GalileoE1B(), 1, 0),
        ("GPS L5I, pre-sync", GPSL5I(), 10, -1),
    )
    for (name, signal, blocks_per_bit, block_index) in cases
        @testset "$name" begin
            context = CN0UpdateContext(
                signal,
                1,
                blocks_per_bit,
                block_index,
                bit_buffer,
                CN0_N₀,
                CN0_T,
            )
            rng = Xoshiro(20260806)
            prompts = cn0_prompts(10^3.0 * 1e-3, 400, rng)     # a true 30 dB-Hz

            nwpr = cn0_fold(NWPRCN0Estimator(), prompts, context)
            reference =
                cn0_fold(NoiseRefCN0Estimator(; num_records = 400), prompts, context)
            @test Base.length(nwpr) == 0
            @test estimate_cn0(nwpr, CN0_T) ==
                  estimate_cn0(get_fallback_cn0_estimator(nwpr), CN0_T)
            @test cn0_db(estimate_cn0(reference, CN0_T)) ≈ 30 atol = 1.0

            # On pure noise NWPR's moment-ratio fallback manufactures signal power;
            # the reference divides by a floor it was told.
            noise = cn0_prompts(0.0, 400, rng)
            nwpr_noise = cn0_fold(NWPRCN0Estimator(), noise, context)
            reference_noise =
                cn0_fold(NoiseRefCN0Estimator(; num_records = 400), noise, context)
            @test cn0_db(estimate_cn0(nwpr_noise, CN0_T)) > 18
            @test cn0_db(estimate_cn0(reference_noise, CN0_T)) < 15
        end
    end
end

@testset "NWPR's coherent window is sized from the signal's code period" begin
    @test NWPRCN0Estimator(GPSL1CA()).num_narrowband_code_blocks == 5
    @test NWPRCN0Estimator(GPSL1C_P()).num_narrowband_code_blocks == 2
    @test NWPRCN0Estimator(GPSL1CA(); num_records = 40).num_records == 40
    # An explicit window still wins, and other keywords are forwarded.
    explicit = NWPRCN0Estimator(GPSL1C_P(); num_narrowband_code_blocks = 7)
    @test explicit.num_narrowband_code_blocks == 7
    @test NWPRCN0Estimator(GPSL1CA(); fallback = NoCN0Estimator()).fallback isa
          NoCN0Estimator
end

@testset "NWPR CN0 estimator on a bare prompt stream" begin
    @test_throws ArgumentError NWPRCN0Estimator(; num_records = 1)
    @test_throws ArgumentError NWPRCN0Estimator(; num_narrowband_code_blocks = 0)
    @test_throws ArgumentError NWPRCN0Estimator(; num_presync_narrowband_code_blocks = -1)

    # One block per record, back-to-back windows of `num_narrowband_code_blocks`.
    # Before the first window closes the fallback's empty-buffer value shows.
    estimator = NWPRCN0Estimator(; num_records = 100, num_narrowband_code_blocks = 20)
    @test @inferred(Base.length(estimator)) == 0
    @test get_current_index(estimator) == 0
    @test @inferred(estimate_cn0(estimator, 1ms)) == 0.0dBHz
    for _ = 1:19
        estimator = @inferred update(estimator, 1.0 + 0.0im)
    end
    @test Base.length(estimator) == 0            # window still open
    estimator = update(estimator, 1.0 + 0.0im)   # 20th record closes it
    @test Base.length(estimator) == 1
    @test get_current_index(estimator) == 1
    @test estimator.num_records_per_ratio == 20
    # 20 identical prompts: the coherent sum holds all the power, µ̂ = M, the
    # noise-free limit of the expression.
    @test @inferred(estimate_cn0(estimator, 1ms)) == Inf * dBHz
    # The ring holds `num_records ÷ M` ratios, so the memory stays ~100 records.
    for _ = 1:(20*10)
        estimator = update(estimator, 1.0 + 0.0im)
    end
    @test Base.length(estimator) == 5

    # Alternating signs cancel in the coherent sum: µ̂ ≤ 1 is "no signal".
    alternating = NWPRCN0Estimator(; num_records = 100, num_narrowband_code_blocks = 2)
    for k = 1:10
        alternating = update(alternating, complex(isodd(k) ? 1.0 : -1.0, 0.0))
    end
    @test Base.length(alternating) == 5
    @test estimate_cn0(alternating, 1ms) == -Inf * dBHz

    # All-zero windows are the division guard: nothing is buffered and the
    # fallback reports. (`NoCN0Estimator` as the fallback, because the default
    # `MomentsCN0Estimator` reads `NaN dB-Hz` on an all-zero buffer.)
    zeros_only = NWPRCN0Estimator(;
        num_records = 100,
        num_narrowband_code_blocks = 2,
        fallback = NoCN0Estimator(),
    )
    for _ = 1:10
        zeros_only = update(zeros_only, complex(0.0, 0.0))
    end
    @test Base.length(zeros_only) == 0
    @test estimate_cn0(zeros_only, 1ms) == -Inf * dBHz

    # A window of one record carries no information (NBP == WBP by construction),
    # so nothing is ever buffered and the fallback keeps reporting.
    single = NWPRCN0Estimator(; num_records = 100, num_narrowband_code_blocks = 1)
    for _ = 1:50
        single = update(single, 1.0 + 0.0im)
    end
    @test Base.length(single) == 0
    @test estimate_cn0(single, 1ms) == estimate_cn0(get_fallback_cn0_estimator(single), 1ms)
end

# JuliaGNSS/Tracking.jl#217
@testset "NWPR CN0 estimator noise floor beats the moment ratio" begin
    # The moment ratio manufactures signal power out of noise at a finite
    # window; NWPR does not. Amplitude √(C/N₀·T) in unit-variance complex noise.
    prompts(cn0, n, seed) =
        (isnothing(cn0) ? 0.0 : sqrt(10^(cn0 / 10) * 1e-3)) .+
        randn(Xoshiro(seed), ComplexF64, n)
    moments(cn0, seed) = cn0_db(
        estimate_cn0(cn0_fold(MomentsCN0Estimator(100), prompts(cn0, 100, seed)), 1ms),
    )
    nwpr(cn0, seed) = cn0_db(
        estimate_cn0(
            cn0_fold(
                NWPRCN0Estimator(; num_records = 100, num_narrowband_code_blocks = 20),
                prompts(cn0, 100, seed),
            ),
            1ms,
        ),
    )
    seeds = 1:200

    @test cn0_median(moments(nothing, s) for s in seeds) > 25
    @test cn0_median(nwpr(nothing, s) for s in seeds) < 15
    # False alarms against a 25 dB-Hz code-lock threshold on pure noise.
    @test count(s -> moments(nothing, s) >= 25, seeds) / length(seeds) > 0.5
    @test count(s -> nwpr(nothing, s) >= 25, seeds) == 0

    # An estimate, not just a detector: unbiased to ~1 dB from 20 dB-Hz up.
    for cn0 in (20.0, 25.0, 30.0, 45.0)
        @test cn0_median(nwpr(cn0, s) for s in seeds) ≈ cn0 atol = 1.0
    end
    @test cn0_median(moments(20.0, s) for s in seeds) - 20 > 5
end

@testset "NWPR CN0 estimator follows the navigation-bit grid" begin
    # Feed a constant prompt and walk the bit grid by hand, so only the window
    # logic is under test.
    bit_buffer = BitBuffer{UInt64}()
    context(signal, blocks_per_bit, bit_block_index) =
        CN0UpdateContext(signal, 1, blocks_per_bit, bit_block_index, bit_buffer)
    fresh() = NWPRCN0Estimator(; num_records = 100, num_narrowband_code_blocks = 20)

    # GPS L1 C/A, synced: one window per 20-block navigation bit, and it only
    # starts on a bit boundary.
    estimator = fresh()
    for bit_block_index in [10:19; repeat(0:19, 2)]
        estimator = update(estimator, 1.0 + 0.0im, context(GPSL1CA(), 20, bit_block_index))
    end
    @test Base.length(estimator) == 2
    @test estimator.num_records_per_ratio == 20

    # The window is capped by `num_narrowband_code_blocks`: shorter windows tile
    # the bit from its start.
    capped = NWPRCN0Estimator(; num_records = 100, num_narrowband_code_blocks = 5)
    for bit_block_index in repeat(0:19, 2)
        capped = update(capped, 1.0 + 0.0im, context(GPSL1CA(), 20, bit_block_index))
    end
    @test Base.length(capped) == 8
    @test capped.num_records_per_ratio == 5
    # A window that would run past the end of the bit is not opened at all.
    ragged = NWPRCN0Estimator(; num_records = 100, num_narrowband_code_blocks = 6)
    for bit_block_index in repeat(0:19, 2)
        ragged = update(ragged, 1.0 + 0.0im, context(GPSL1CA(), 20, bit_block_index))
    end
    @test Base.length(ragged) == 6
    @test ragged.num_records_per_ratio == 6
    # A cap above the bit period cannot lengthen the window past the bit.
    wide = NWPRCN0Estimator(; num_records = 100, num_narrowband_code_blocks = 50)
    for bit_block_index in repeat(0:19, 2)
        wide = update(wide, 1.0 + 0.0im, context(GPSL1CA(), 20, bit_block_index))
    end
    @test Base.length(wide) == 2
    @test wide.num_records_per_ratio == 20

    # Not synced yet: a short unaligned window, five blocks by default.
    estimator = fresh()
    for _ = 1:20
        estimator = update(estimator, 1.0 + 0.0im, context(GPSL1CA(), 20, -1))
    end
    @test Base.length(estimator) == 4
    @test estimator.num_records_per_ratio == 5
    # An explicit pre-sync length is honoured.
    two_block = NWPRCN0Estimator(;
        num_records = 100,
        num_narrowband_code_blocks = 20,
        num_presync_narrowband_code_blocks = 2,
    )
    for _ = 1:8
        two_block = update(two_block, 1.0 + 0.0im, context(GPSL1CA(), 20, -1))
    end
    @test Base.length(two_block) == 4
    @test two_block.num_records_per_ratio == 2
    # Switching to the post-sync window restarts the ring.
    for bit_block_index = 0:19
        estimator = update(estimator, 1.0 + 0.0im, context(GPSL1CA(), 20, bit_block_index))
    end
    @test Base.length(estimator) == 1
    @test estimator.num_records_per_ratio == 20

    # Even at an unchanged window length, the buffered pre-sync windows are
    # dropped when the first bit-aligned one completes.
    same_length = NWPRCN0Estimator(; num_records = 100, num_narrowband_code_blocks = 5)
    for _ = 1:20
        same_length = update(same_length, 1.0 + 0.0im, context(GPSL1CA(), 20, -1))
    end
    @test Base.length(same_length) == 4
    @test same_length.num_records_per_ratio == 5
    @test !same_length.ratios_are_bit_aligned
    for bit_block_index = 0:4
        same_length =
            update(same_length, 1.0 + 0.0im, context(GPSL1CA(), 20, bit_block_index))
    end
    @test same_length.num_records_per_ratio == 5     # window length unchanged ...
    @test same_length.ratios_are_bit_aligned
    @test Base.length(same_length) == 1              # ... yet the ring restarted

    # A pre-sync window still open when sync arrives is dropped, not carried
    # into the bit-aligned window.
    estimator = fresh()
    for _ = 1:9
        estimator = update(estimator, 1.0 + 0.0im, context(GPSL1CA(), 20, -1))
    end
    @test estimator.num_accumulated_records == 4
    estimator = update(estimator, 1.0 + 0.0im, context(GPSL1CA(), 20, 5))
    @test estimator.num_accumulated_records == 0     # dropped, not continued
    for bit_block_index in [6:19; 0:19]
        estimator = update(estimator, 1.0 + 0.0im, context(GPSL1CA(), 20, bit_block_index))
    end
    @test estimator.num_records_per_ratio == 20
    @test Base.length(estimator) == 1

    # A grid that jumps to the start of a new window mid-window restarts it
    # there rather than dropping the record.
    restarted = NWPRCN0Estimator(; num_records = 100, num_narrowband_code_blocks = 5)
    for bit_block_index = 0:2
        restarted = update(restarted, 1.0 + 0.0im, context(GPSL1CA(), 20, bit_block_index))
    end
    @test restarted.num_accumulated_records == 3
    restarted = update(restarted, 1.0 + 0.0im, context(GPSL1CA(), 20, 5))
    @test restarted.num_accumulated_records == 1
    @test restarted.num_accumulated_code_blocks == 1

    # A signal carrying a secondary code has no pre-sync coherence at all, so no
    # window is opened and the fallback is reported.
    estimator = fresh()
    for _ = 1:40
        estimator = update(estimator, 1.0 + 0.0im, context(GPSL5I(), 10, -1))
    end
    @test Base.length(estimator) == 0
    @test estimate_cn0(estimator, 1ms) ==
          estimate_cn0(get_fallback_cn0_estimator(estimator), 1ms)
    # Once synced its bit is 10 blocks long, and the replica wipes the overlay.
    for bit_block_index in repeat(0:9, 3)
        estimator = update(estimator, 1.0 + 0.0im, context(GPSL5I(), 10, bit_block_index))
    end
    @test Base.length(estimator) == 3
    @test estimator.num_records_per_ratio == 10

    # One symbol per code block (GPS L1C-D, Galileo E1B): no coherent window.
    estimator = fresh()
    for _ = 1:40
        estimator = update(estimator, 1.0 + 0.0im, context(GPSL1C_D(), 1, 0))
    end
    @test Base.length(estimator) == 0

    # A pilot has no bit grid to respect post-sync: windows run back to back.
    estimator = NWPRCN0Estimator(; num_records = 100, num_narrowband_code_blocks = 5)
    for _ = 1:20
        estimator = update(estimator, 1.0 + 0.0im, context(GPSL1C_P(), 0, 0))
    end
    @test Base.length(estimator) == 4
    @test estimator.num_records_per_ratio == 5
end

@testset "NWPR CN0 estimator when a record outgrows its own window" begin
    bit_buffer = BitBuffer{UInt64}()
    context(num_code_blocks, bit_block_index) =
        CN0UpdateContext(GPSL1CA(), num_code_blocks, 20, bit_block_index, bit_buffer)

    # Records as long as a whole navigation bit close every window on a single
    # record; the windows buffered before the switch have to go with it.
    rng = Xoshiro(1)
    noisy() = sqrt(10^3.5 * 1e-3) + randn(rng, ComplexF64)   # a true ~35 dB-Hz
    estimator = NWPRCN0Estimator(; num_records = 100, num_narrowband_code_blocks = 5)
    for _ = 1:5, bit_block_index = 0:19
        estimator = update(estimator, noisy(), context(1, bit_block_index))
    end
    @test Base.length(estimator) == 20
    @test estimator.num_records_per_ratio == 5
    @test isfinite(cn0_db(estimate_cn0(estimator, 1ms)))
    for _ = 1:20
        estimator = update(estimator, noisy(), context(20, 0))
    end
    @test Base.length(estimator) == 0
    @test estimate_cn0(estimator, 20ms) ==
          estimate_cn0(get_fallback_cn0_estimator(estimator), 20ms)
end

@testset "NWPR CN0 estimator skips a record spanning no whole code block" begin
    # The fractional record right after a sync phase-snap leaves the open
    # window untouched; its prompt still reaches the fallback.
    bit_buffer = BitBuffer{UInt64}()
    context(num_code_blocks, bit_block_index) =
        CN0UpdateContext(GPSL1CA(), num_code_blocks, 20, bit_block_index, bit_buffer)

    estimator = NWPRCN0Estimator(; num_records = 100, num_narrowband_code_blocks = 5)
    for bit_block_index = 0:2
        estimator = update(estimator, 1.0 + 0.0im, context(1, bit_block_index))
    end
    @test estimator.num_accumulated_records == 3
    estimator = update(estimator, 1.0 + 0.0im, context(0, 3))
    @test estimator.num_accumulated_records == 3
    @test estimator.num_accumulated_code_blocks == 3
    for bit_block_index = 3:4
        estimator = update(estimator, 1.0 + 0.0im, context(1, bit_block_index))
    end
    @test Base.length(estimator) == 1
    @test estimator.num_records_per_ratio == 5
    @test Base.length(get_fallback_cn0_estimator(estimator)) == 6
end

# Reduced form of Tracking.jl's `test/cn0_estimator_comparison.jl`: the same
# post-correlation prompt model, but one seed and fewer trials, so only the
# structural claims with a wide statistical margin are pinned. NWPR is driven
# through its bare-prompt-stream `update`, the noise reference is the shipped
# `NoiseRefCN0Estimator` against the exactly known floor.
function cn0_comparison_sweep(cn0, trials)
    λ = 10^(cn0 / 10) * 1e-3
    rng = Xoshiro(20260806 + cn0)
    nwpr = Vector{Float64}(undef, trials)
    noise_ref = Vector{Float64}(undef, trials)
    coherent = Vector{Float64}(undef, trials)
    context = cn0_context()
    M = 5
    for t = 1:trials
        prompts = cn0_prompts(λ, 100, rng)
        nwpr_estimator = cn0_fold(
            NWPRCN0Estimator(; num_records = 100, num_narrowband_code_blocks = M),
            prompts,
        )
        nwpr[t] = 10^(cn0_db(estimate_cn0(nwpr_estimator, CN0_T)) / 10) * 1e-3
        ref_estimator =
            cn0_fold(NoiseRefCN0Estimator(; num_records = 100), prompts, context)
        noise_ref[t] = mean(ref_estimator.buffered_cn0) * 1e-3
        # Coherent reference: `M` records summed is one `M`-times-longer record.
        coherent[t] = mean(
            (abs2(sum(@view prompts[((w-1)*M+1):(w*M)])) - M) / M^2 for w = 1:div(100, M)
        )
    end
    stats(estimates) = begin
        usable = filter(x -> isfinite(x) && !iszero(x), estimates)
        (;
            σ_db = 4.342944819 * std(usable) / λ,
            degenerate = 1 - length(usable) / length(estimates),
        )
    end
    (; nwpr = stats(nwpr), noise_ref = stats(noise_ref), coherent = stats(coherent))
end

@testset "NWPR against a noise reference on the prompt model" begin
    sweep = Dict(cn0 => cn0_comparison_sweep(cn0, 1000) for cn0 in (20, 40, 45, 50))
    # NWPR reports degenerate estimates at low C/N₀ (≈5.6 % at 20 dB-Hz); a
    # noise reference never does.
    @test sweep[20].nwpr.degenerate > 0.01
    for cn0 in keys(sweep)
        @test sweep[cn0].noise_ref.degenerate == 0
        @test sweep[cn0].coherent.degenerate == 0
    end
    # A coherent noise reference beats NWPR. Pinned from 40 dB-Hz up only: at
    # 20 dB-Hz the margin is ~7 %, too close to this trial count's Monte-Carlo
    # error to be a test of the estimators rather than of one RNG stream.
    for cn0 in (40, 45, 50)
        @test sweep[cn0].coherent.σ_db < 0.5 * sweep[cn0].nwpr.σ_db
    end
    # NWPR's relative error saturates at the `µ̂ → M` ceiling, a noise
    # reference's keeps falling.
    @test sweep[50].nwpr.σ_db > 0.8 * sweep[45].nwpr.σ_db
    @test sweep[50].noise_ref.σ_db < 0.7 * sweep[45].noise_ref.σ_db
    # Non-coherent squaring loss is real at low C/N₀ and gone by 40 dB-Hz.
    @test sweep[20].noise_ref.σ_db > sweep[20].nwpr.σ_db
    @test sweep[40].noise_ref.σ_db < sweep[40].nwpr.σ_db
end
