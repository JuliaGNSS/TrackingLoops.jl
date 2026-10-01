using Random: Xoshiro, randn
using StaticArrays: SMatrix
using Unitful: s, ms, ustrip, uconvert

# One dump of `n` samples of CN(0, σ²) noise despread by a ±1 code, as the three
# shapes a producer can report it in. `σ²` is the per-sample variance, so the
# density every one of them must land on is `σ²/f_s`.
function white_noise_dump(σ², n, seed)
    rng = Xoshiro(seed)
    accumulation = complex(0.0, 0.0)
    sample_power = 0.0
    for _ = 1:n
        x = sqrt(σ² / 2) * complex(randn(rng), randn(rng))
        code = rand(rng, (-1.0, 1.0))
        accumulation += x * code
        sample_power += abs2(x)
    end
    (; accumulation, sample_power)
end

# A hardware-style source that keeps no window: it reports no density and no
# look count, and relies on every interface default.
struct ExternalOnlyNoiseEstimator <: AbstractNoiseEstimator end

# A stand-in for a software backend: Tracking.jl implements `despread_noise!` for
# its downconvert-and-correlate backends; here it appends one power-monitor
# observation over the slice so the forwarding of `update_noise!` is observable.
struct FakeDespreadBackend end

function TrackingLoops.despread_noise!(
    ::FakeDespreadBackend,
    estimator,
    measurement,
    first_sample,
    last_sample,
    context,
)
    slice = view(measurement, first_sample:last_sample)
    append_noise_observation!(
        estimator,
        noise_observation_from_samples(sum(abs2, slice), length(slice), 4e6Hz),
    )
end

@testset "The three noise builders land on the same N₀ scale" begin
    fs = 4e6Hz
    σ² = 3.0
    n = 4000

    # The power monitor is the low-variance builder — `M = n` looks — so it pins
    # the scale on a single dump.
    d = white_noise_dump(σ², n, 20260806)
    from_samples = noise_observation_from_samples(d.sample_power, n, fs)
    @test ustrip(Hz^-1, from_samples.noise_density) ≈ ustrip(Hz^-1, σ² / fs) rtol = 0.05
    @test from_samples.num_sub_integrations == n
    @test from_samples.duration ≈ uconvert(s, n / fs)
    @test from_samples.prn == 0

    # A single despread dump is one look with 100 % relative error, so average
    # many before comparing.
    estimator = CorrelatorNoiseEstimator(; window_duration = 10.0s)
    for seed = 1:400
        dump = white_noise_dump(σ², 400, seed)
        append_noise_observation!(estimator, noise_observation(dump.accumulation, 400, fs))
    end
    @test ustrip(Hz^-1, get_noise_density(estimator)) ≈ ustrip(Hz^-1, σ² / fs) rtol = 0.15

    # Pre-summing is the same density, and reduces to the single-dump builder at M = 1.
    one_dump = noise_observation(d.accumulation, n, fs; prn = 7)
    @test one_dump.prn == 7
    presummed = noise_observation_from_correlator(abs2(d.accumulation), 1, n, fs; prn = 7)
    @test presummed.noise_density == one_dump.noise_density
    @test presummed.num_sub_integrations == one_dump.num_sub_integrations
    @test presummed.duration == one_dump.duration
    @test presummed.prn == one_dump.prn

    # `code_amplitude` undoes a multi-level code's integer scale.
    scaled = noise_observation(3.0 * d.accumulation, n, fs; code_amplitude = 3)
    @test ustrip(Hz^-1, scaled.noise_density) ≈ ustrip(Hz^-1, one_dump.noise_density)
    scaled_corr = noise_observation_from_correlator(
        abs2(3.0 * d.accumulation),
        1,
        n,
        fs;
        code_amplitude = 3,
    )
    @test ustrip(Hz^-1, scaled_corr.noise_density) ≈ ustrip(Hz^-1, one_dump.noise_density)

    # The sampling frequency may be spelled in any unit or number type.
    @test typeof(noise_observation(d.accumulation, n, 4.0e3Hz * 1000)) ===
          typeof(noise_observation(d.accumulation, n, fs))
end

@testset "Simultaneous looks report their own span, not M times it" begin
    fs = 4e6Hz
    obs = noise_observation_from_correlator(3.0, 3, 3 * 4000, fs; duration = 4000 / fs)
    @test obs.num_sub_integrations == 3
    @test obs.duration ≈ uconvert(s, 4000 / fs)
    @test ustrip(Hz^-1, obs.noise_density) ≈ ustrip(Hz^-1, 3.0 / (3 * 4000 * fs))
    # The default duration assumes consecutive sub-integrations.
    consecutive = noise_observation_from_correlator(3.0, 3, 3 * 4000, fs)
    @test consecutive.duration ≈ 3 * obs.duration
end

@testset "The noise window is bounded in time and weighted by M" begin
    fs = 4e6Hz
    estimator = CorrelatorNoiseEstimator(; window_duration = 10.0ms)
    for k = 1:50
        append_noise_observation!(
            estimator,
            noise_observation_from_samples(4000.0 * k, 4000, fs),
        )
    end
    @test length(estimator) == 10
    # Equal `M`, so the plain mean of `k = 41 … 50`, i.e. 45.5 / f_s.
    @test ustrip(Hz^-1, get_noise_density(estimator)) ≈ ustrip(Hz^-1, 45.5 / fs)
    @test noise_window_looks(estimator) == 10 * 4000

    # An observation longer than the window is still kept.
    long = CorrelatorNoiseEstimator(; window_duration = 1.0ms)
    append_noise_observation!(long, noise_observation_from_samples(1.0e5, 100_000, fs))
    @test length(long) == 1
    @test !isnothing(get_noise_density(long))
    # ... and is replaced once a newer one covers the window on its own.
    append_noise_observation!(long, noise_observation_from_samples(2.0e5, 100_000, fs))
    @test length(long) == 1
    @test ustrip(Hz^-1, get_noise_density(long)) ≈ ustrip(Hz^-1, 2.0 / fs)

    # `M` makes observations from different producers combinable.
    mixed = CorrelatorNoiseEstimator(; window_duration = 10.0s)
    append_noise_observation!(mixed, noise_observation_from_samples(4000.0, 4000, fs))
    append_noise_observation!(
        mixed,
        noise_observation_from_correlator(4000.0 * 1000, 1, 4000, fs),
    )
    @test ustrip(Hz^-1, get_noise_density(mixed)) ≈
          ustrip(Hz^-1, (4000 * 1.0 + 1 * 1000.0) / 4001 / fs)
    @test noise_window_looks(mixed) == 4001
end

@testset "CorrelatorNoiseEstimator validates its configuration" begin
    estimator = CorrelatorNoiseEstimator(;
        window_duration = 500ms,
        tap_code_shift = 2,
        carrier_dither = 1000Hz,
        rng = Xoshiro(3),
    )
    @test estimator.window_duration === 0.5s
    @test estimator.tap_code_shift === 2.0
    @test estimator.carrier_dither === 1000.0Hz
    @test estimator.rng == Xoshiro(3)
    @test CorrelatorNoiseEstimator(; carrier_dither = 0.0Hz).carrier_dither == 0.0Hz
    @test_throws ArgumentError CorrelatorNoiseEstimator(; window_duration = 0.0s)
    @test_throws ArgumentError CorrelatorNoiseEstimator(; window_duration = -1.0ms)
    @test_throws ArgumentError CorrelatorNoiseEstimator(; tap_code_shift = 0.0)
    @test_throws ArgumentError CorrelatorNoiseEstimator(; carrier_dither = -1.0Hz)
end

@testset "An empty noise window has no density" begin
    estimator = CorrelatorNoiseEstimator()
    @test isnothing(get_noise_density(estimator))
    @test length(estimator) == 0
    @test noise_window_looks(estimator) == 0
    append_noise_observation!(
        estimator,
        noise_observation_from_samples(4000.0, 4000, 4e6Hz),
    )
    @test !isnothing(get_noise_density(estimator))

    # The public reader carries the `nothing`; the fold's splitter must not.
    D = noise_density_type(estimator)
    @test D === NoiseDensity
    @test Base.return_types(get_noise_density, (typeof(estimator),)) == [Union{Nothing,D}]
    empty_one = CorrelatorNoiseEstimator()
    @test @inferred(TrackingLoops._noise_density_and_ready(empty_one)) == (zero(D), false)
    @test !TrackingLoops._noise_window_filling(empty_one)
    density, ready = @inferred TrackingLoops._noise_density_and_ready(estimator)
    @test ready
    @test density == get_noise_density(estimator)
    @test !TrackingLoops._noise_window_filling(estimator)
end

@testset "A zero measured floor is not a floor to divide by" begin
    dead = CorrelatorNoiseEstimator()
    append_noise_observation!(dead, noise_observation_from_samples(0.0, 4000, 4e6Hz))
    D = noise_density_type(dead)
    @test get_noise_density(dead) == zero(D)
    @test @inferred(TrackingLoops._noise_density_and_ready(dead)) == (zero(D), false)
    @test !TrackingLoops._noise_window_filling(dead)

    for bad in (NaN, Inf)
        e = CorrelatorNoiseEstimator()
        append_noise_observation!(e, NoiseObservation(bad / 1.0Hz, 1, 1.0e-3s, Int16(1)))
        @test @inferred(TrackingLoops._noise_density_and_ready(e)) == (zero(D), false)
        @test !TrackingLoops._noise_window_filling(e)
    end
end

@testset "A noise observation lands whatever number type the producer used" begin
    f32 = noise_observation(complex(1.0f0, 0.0f0), 4000, 4.0f6Hz)
    @test f32 isa NoiseObservation{NoiseDensity,typeof(1.0s)}
    e32 = CorrelatorNoiseEstimator()
    append_noise_observation!(e32, f32)
    @test length(e32) == 1
    @test ustrip(Hz^-1, get_noise_density(e32)) ≈ 6.25e-11 rtol = 1e-6

    mixed = noise_observation_from_correlator(1.0, 1, 4000, 4_000_000Hz; duration = 1ms)
    @test mixed isa NoiseObservation{NoiseDensity,typeof(1.0s)}
    @test mixed.duration === 1.0e-3s

    # A hand-assembled observation is retyped on append rather than dropped.
    hand = NoiseObservation(1.0f-10 / 1.0f0Hz, 1, 1.0f-3s, Int16(3))
    @test !(hand isa NoiseObservation{NoiseDensity,typeof(1.0s)})
    converted = convert(NoiseObservation{NoiseDensity,typeof(1.0s)}, hand)
    @test converted isa NoiseObservation{NoiseDensity,typeof(1.0s)}
    @test converted.prn == 3
    # Converting an already-canonical observation is the identity.
    @test convert(NoiseObservation{NoiseDensity,typeof(1.0s)}, converted) === converted
    ehand = CorrelatorNoiseEstimator()
    append_noise_observation!(ehand, hand)
    @test length(ehand) == 1
    @test ustrip(Hz^-1, get_noise_density(ehand)) ≈ 1.0e-10 rtol = 1e-6
end

@testset "The noise estimator is mutated in place, never rebuilt" begin
    estimator = CorrelatorNoiseEstimator()
    obs = noise_observation_from_samples(4000.0, 4000, 4e6Hz)
    @test append_noise_observation!(estimator, obs) === estimator
    for T in (CorrelatorNoiseEstimator, NoiseObservation, TrackingLoops.NoiseUpdateContext)
        @test !ismutabletype(T)
    end
end

@testset "The noise window's running totals stay exact" begin
    # The cached sums must equal the walk they replace, including after thousands
    # of mixed-size appends and drops (the case the periodic refresh exists for).
    exact_span(e) = sum(o.duration for o in e.buffered)
    exact_looks(e) = sum(o.num_sub_integrations for o in e.buffered)
    exact_density(e) =
        sum(o.num_sub_integrations * o.noise_density for o in e.buffered) / exact_looks(e)

    rng = Xoshiro(17)
    window = 50.0ms
    estimator = CorrelatorNoiseEstimator(; window_duration = window)
    for i = 1:5000
        duration = uconvert(s, (rand(rng) < 0.05 ? 200.0 : 1.0) * rand(rng) * ms)
        append_noise_observation!(
            estimator,
            NoiseObservation(
                (1.0 + 5rand(rng)) * 1e-10 / 1.0Hz,
                rand(rng, 1:64),
                duration,
                Int16(rand(rng, 1:32)),
            ),
        )
        i % 500 == 0 || continue
        @test estimator.totals[].span ≈ exact_span(estimator) rtol = 1e-10
        @test estimator.totals[].looks == exact_looks(estimator)
        @test noise_window_looks(estimator) == exact_looks(estimator)
        @test get_noise_density(estimator) ≈ exact_density(estimator) rtol = 1e-10
    end
    # Minimal, but never short of the configured span.
    @test estimator.totals[].span >= window
    @test estimator.totals[].span - first(estimator.buffered).duration < window
end

@testset "Appending to and reading the noise window is allocation-free" begin
    # Two very different counts separate a per-push allocation (which would grow
    # a hundredfold) from a fixed per-call one of the measurement harness.
    function push_many(estimator, obs, n)
        for _ = 1:n
            append_noise_observation!(estimator, obs)
        end
        estimator
    end
    function read_many(estimator, n)
        d = get_noise_density(estimator)
        for _ = 1:n
            d = get_noise_density(estimator)
        end
        d
    end
    estimator = CorrelatorNoiseEstimator()
    obs = noise_observation_from_samples(4000.0, 4000, 4e6Hz)
    push_many(estimator, obs, 5_000)
    few = @allocated push_many(estimator, obs, 10_000)
    many = @allocated push_many(estimator, obs, 200_000)
    @test many == few
    @test few <= 128

    read_many(estimator, 100)
    few_reads = @allocated read_many(estimator, 1_000)
    many_reads = @allocated read_many(estimator, 10_000)
    @test many_reads == few_reads
    @test few_reads <= 128
end

@testset "The abstract noise estimator interface has no-op defaults" begin
    estimator = ExternalOnlyNoiseEstimator()
    @test update_noise!(estimator, nothing, 1, 10, nothing) === estimator
    obs = noise_observation_from_samples(4000.0, 4000, 4e6Hz)
    @test append_noise_observation!(estimator, obs) === estimator
    @test isnothing(get_noise_density(estimator))
    @test noise_density_type(estimator) === NoiseDensity
    @test isnothing(noise_window_looks(estimator))
    @test TrackingLoops._noise_density_and_ready(estimator) == (zero(NoiseDensity), false)
    @test !TrackingLoops._noise_window_filling(estimator)
    @test (GPSL1CA = CorrelatorNoiseEstimator(), E1B = estimator) isa NoiseEstimators
end

@testset "update_noise! forwards to the backend's despread_noise!" begin
    estimator = CorrelatorNoiseEstimator()
    samples = fill(complex(1.0, 1.0), 8000)
    context = TrackingLoops.NoiseUpdateContext(GPSL1CA(), 3, FakeDespreadBackend())
    @test context.chunk_index == 3
    @test update_noise!(estimator, samples, 1, 4000, context) === estimator
    @test length(estimator) == 1
    # |x|² = 2 per sample ⇒ N₀ = 2 / f_s.
    @test ustrip(Hz^-1, get_noise_density(estimator)) ≈ 2 / 4e6
end

@testset "The software reference's helpers pool taps and rotate PRNs" begin
    # Single antenna: the scalar power of the taps.
    taps = SVector(1.0 + 1.0im, 2.0 + 0.0im, 0.0 - 3.0im)
    @test TrackingLoops._pool_taps(taps, NumAnts(1)) == 2.0 + 4.0 + 9.0
    # An array: the summed outer products, one dimension up.
    array_taps = SVector(
        SVector(1.0 + 0.0im, 0.0 + 1.0im),
        SVector(2.0 + 0.0im, 1.0 + 0.0im),
        SVector(0.0 + 0.0im, 1.0 - 1.0im),
    )
    pooled = TrackingLoops._pool_taps(array_taps, NumAnts(2))
    @test pooled isa SMatrix{2,2,ComplexF64}
    @test pooled ≈ sum(t * t' for t in array_taps)
    @test pooled ≈ pooled'

    # The rotation position is carried in the newest observation.
    signal = GPSL1CA()
    num_prns = size(GNSSSignals.get_codes(signal), 2)
    estimator = CorrelatorNoiseEstimator()
    @test TrackingLoops._next_noise_prn(estimator, signal) == 1
    append_noise_observation!(
        estimator,
        noise_observation_from_samples(4000.0, 4000, 4e6Hz; prn = 5),
    )
    @test TrackingLoops._next_noise_prn(estimator, signal) == 6
    append_noise_observation!(
        estimator,
        noise_observation_from_samples(4000.0, 4000, 4e6Hz; prn = num_prns),
    )
    @test TrackingLoops._next_noise_prn(estimator, signal) == 1
end

@testset "The noise builders take per-antenna input" begin
    fs = 4e6Hz
    n = 4000
    M = 3

    b = SVector{M,ComplexF64}(2.0 + 0.0im, 0.0 + 1.0im, 1.0 - 1.0im)
    obs = @inferred noise_observation(b, n, fs)
    R = obs.noise_density
    @test R isa SMatrix{M,M}
    @test obs.num_sub_integrations == 1
    for m = 1:M, k = 1:M
        @test R[m, k] ≈ (b[m] * conj(b[k])) / (n * fs)
    end
    for m = 1:M
        @test real(R[m, m]) ≈ noise_observation(b[m], n, fs).noise_density
    end

    pre_summed = b * b' + (2 * b) * (2 * b)'
    from_corr = @inferred noise_observation_from_correlator(pre_summed, 2, 2n, fs)
    @test from_corr.noise_density isa SMatrix{M,M}
    @test from_corr.noise_density ≈ pre_summed / (2n * fs)
    @test from_corr.num_sub_integrations == 2

    from_samples = @inferred noise_observation_from_samples(pre_summed, n, fs)
    @test from_samples.noise_density isa SMatrix{M,M}
    @test from_samples.noise_density ≈ pre_summed / (n * fs)
    @test from_samples.num_sub_integrations == n

    estimator = CorrelatorNoiseEstimator(; num_ants = NumAnts(M))
    @test noise_density_type(estimator) <: SMatrix{M,M}
    @test TrackingLoops._num_ants(estimator) === NumAnts(M)
    @test TrackingLoops._num_ants(CorrelatorNoiseEstimator()) === NumAnts(1)
    @test TrackingLoops._num_ants_of_density_type(NoiseDensity) === NumAnts(1)
    append_noise_observation!(estimator, obs)
    @test get_noise_density(estimator) ≈ R

    hand_built = NoiseObservation(
        SMatrix{M,M,ComplexF32,M * M}(b * b') / 4.0f6Hz,
        1,
        1.0ms,
        Int16(3),
    )
    append_noise_observation!(estimator, hand_built)
    @test length(estimator) == 2
end

@testset "A noise covariance is withheld until it spans its own dimensions" begin
    # One software observation pools three taps, i.e. three looks; an `M×M`
    # covariance needs `M` looks before it can answer every `wᴴR̂w`.
    fs = 4e6Hz
    function observation(M, seed)
        rng = Xoshiro(seed)
        taps = [SVector{M}(randn(rng, ComplexF64, M)) for _ = 1:3]
        covariance = sum(t * t' for t in taps)
        noise_observation_from_correlator(covariance, 3, 3 * 4000, fs; duration = 1ms)
    end
    for (M, expected_observations) in ((1, 1), (3, 1), (4, 2), (8, 3))
        estimator = CorrelatorNoiseEstimator(; num_ants = NumAnts(M))
        ready_at = 0
        for k = 1:4
            if M == 1
                append_noise_observation!(
                    estimator,
                    noise_observation_from_correlator(3.0, 3, 3 * 4000, fs),
                )
            else
                append_noise_observation!(estimator, observation(M, k))
            end
            _, ready = TrackingLoops._noise_density_and_ready(estimator)
            if ready && ready_at == 0
                ready_at = k
            end
        end
        @test ready_at == expected_observations
    end

    estimator = CorrelatorNoiseEstimator(; num_ants = NumAnts(4))
    append_noise_observation!(estimator, observation(4, 1))
    @test noise_window_looks(estimator) == 3
    @test !isnothing(get_noise_density(estimator))
    @test TrackingLoops._noise_window_filling(estimator)
    density, ready = TrackingLoops._noise_density_and_ready(estimator)
    @test !ready
    @test density == zero(noise_density_type(estimator))

    # A dead input at four antennas measures a zero floor, which is not filling.
    dead = CorrelatorNoiseEstimator(; num_ants = NumAnts(4))
    append_noise_observation!(
        dead,
        noise_observation_from_correlator(zero(SMatrix{4,4,ComplexF64,16}), 3, 12000, fs),
    )
    @test !TrackingLoops._noise_window_filling(dead)
    @test !last(TrackingLoops._noise_density_and_ready(dead))
    # And a non-finite element is rejected outright.
    bad = CorrelatorNoiseEstimator(; num_ants = NumAnts(2))
    append_noise_observation!(
        bad,
        noise_observation_from_correlator(
            SMatrix{2,2,ComplexF64,4}(1.0, 0.0, 0.0, NaN),
            3,
            12000,
            fs,
        ),
    )
    @test !last(TrackingLoops._noise_density_and_ready(bad))

    @test TrackingLoops._sufficient_looks(zero(SMatrix{4,4,ComplexF64,16}), nothing)
    @test !TrackingLoops._sufficient_looks(zero(SMatrix{4,4,ComplexF64,16}), 3)
    @test TrackingLoops._sufficient_looks(zero(SMatrix{4,4,ComplexF64,16}), 4)
    @test TrackingLoops._sufficient_looks(1.0 / 1.0Hz, 1)
end
