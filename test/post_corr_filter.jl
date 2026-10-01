# Post-correlation filter. Ported from Tracking.jl's `test/post_corr_filter.jl`.

using StaticArrays: SMatrix
using TrackingLoops: _combine_antennas, _combine_correlator, _reduce_noise_density

@testset "Post correlation filter" begin
    default_post_corr_filter = DefaultPostCorrFilter()

    @test update(default_post_corr_filter, 1.0 + 2.0im) == default_post_corr_filter

    # Single antenna: unity weight, and combining is the identity on the tap.
    w1 = @inferred get_weights(default_post_corr_filter, NumAnts(1))
    @test w1 === 1.0 + 0.0im
    @test @inferred(_combine_antennas(w1, complex(1.2, 1.3))) === complex(1.2, 1.3)

    # Multi antenna: the last element is selected.
    w2 = @inferred get_weights(default_post_corr_filter, NumAnts(2))
    @test w2 === SVector{2,ComplexF64}(0.0 + 0.0im, 1.0 + 0.0im)
    tap = SVector(complex(1.2, 1.3), complex(2.2, 4.3))
    @test @inferred(_combine_antennas(w2, tap)) === complex(2.2, 4.3)
    w3 = get_weights(default_post_corr_filter, NumAnts(3))
    @test w3 == SVector{3,ComplexF64}(0, 0, 1)

    # The weights are conjugated: wᴴ·b.
    @test _combine_antennas(1.0im, 1.0 + 0.0im) == -1.0im
    @test _combine_antennas(SVector{2,ComplexF64}(1.0im, 0), SVector{2,ComplexF64}(1, 5)) ==
          -1.0im

    # A whole multi-antenna correlator reduces to the single-channel one.
    correlator = EarlyPromptLateCorrelator(
        SVector(
            SVector{2,ComplexF64}(1, 2),
            SVector{2,ComplexF64}(3, 4),
            SVector{2,ComplexF64}(5, 6),
        ),
        0.5,
    )
    combined = @inferred _combine_correlator(correlator, w2)
    @test get_accumulators(combined) == SVector{3,ComplexF64}(2, 4, 6)
end

@testset "Noise density reduces through the same weights" begin
    # No estimator configured passes straight through, so the C/N₀ context's
    # type parameter stays `Nothing` and the wiring mistake still throws.
    @test _reduce_noise_density(nothing, 1.0 + 0.0im) === nothing
    @test _reduce_noise_density(nothing, SVector{2,ComplexF64}(0, 1)) === nothing

    # Unity weight leaves a scalar density bit-identical.
    density = 1.234 / 1.0Hz
    @test @inferred(_reduce_noise_density(density, 1.0 + 0.0im)) === density
    # A scalar gain of g scales the floor by |g|².
    @test _reduce_noise_density(density, 2.0 + 0.0im) ≈ 4 * density

    # A covariance reduces to wᴴRw. Under `DefaultPostCorrFilter`'s weights that
    # is exactly the last antenna's own floor.
    R = SMatrix{2,2,ComplexF64,4}(4.0, 1.0 - 2.0im, 1.0 + 2.0im, 9.0) / 1.0Hz
    w_last = get_weights(DefaultPostCorrFilter(), NumAnts(2))
    @test @inferred(_reduce_noise_density(R, w_last)) ≈ real(R[2, 2])

    # An averaging beamformer sees the floor its weights actually produce, which
    # for correlated antennas is not the mean of the diagonal.
    w_mean = SVector{2,ComplexF64}(0.5, 0.5)
    @test _reduce_noise_density(R, w_mean) ≈
          real(0.25 * (R[1, 1] + R[1, 2] + R[2, 1] + R[2, 2]))

    # Spatially white noise: wᴴ(σ²I)w == ‖w‖²σ².
    white = SMatrix{2,2,ComplexF64,4}(3.0, 0.0, 0.0, 3.0) / 1.0Hz
    @test _reduce_noise_density(white, w_mean) ≈ 0.5 * 3.0 / 1.0Hz
end
