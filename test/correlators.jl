@testset "Correlator initial accumulators" begin
    @test @inferred(TrackingLoops.get_initial_accumulator(NumAnts(1), NumAccumulators(3))) isa
          SVector{3,ComplexF64}
    @test @inferred(TrackingLoops.get_initial_accumulator(NumAnts(4), NumAccumulators(3))) isa
          SVector{3,SVector{4,ComplexF64}}
    @test @inferred(TrackingLoops.get_initial_accumulator(NumAnts(1), 3)) isa Vector{ComplexF64}
    @test @inferred(TrackingLoops.get_initial_accumulator(NumAnts(4), 3)) isa
          Vector{SVector{4,ComplexF64}}
end

@testset "Correlator constructors and accessors" begin
    correlator = @inferred EarlyPromptLateCorrelator()
    @test @inferred(get_early(correlator)) == 0.0
    @test @inferred(get_prompt(correlator)) == 0.0
    @test @inferred(get_late(correlator)) == 0.0
    @test correlator.preferred_early_late_to_prompt_code_shift == 0.5
    @test get_num_ants(correlator) == 1
    @test TrackingLoops._num_ants_val(correlator) === NumAnts(1)

    correlator = @inferred EarlyPromptLateCorrelator(num_ants = NumAnts(1))
    @test @inferred(get_prompt(correlator)) == 0.0
    @test get_num_ants(correlator) == 1

    correlator = @inferred EarlyPromptLateCorrelator(num_ants = NumAnts(2))
    @test @inferred(get_early(correlator)) == SVector(0.0 + 0.0im, 0.0 + 0.0im)
    @test @inferred(get_prompt(correlator)) == SVector(0.0 + 0.0im, 0.0 + 0.0im)
    @test @inferred(get_late(correlator)) == SVector(0.0 + 0.0im, 0.0 + 0.0im)
    @test correlator.preferred_early_late_to_prompt_code_shift == 0.5
    @test get_num_ants(correlator) == 2
    @test TrackingLoops._num_ants_val(correlator) === NumAnts(2)

    correlator = EarlyPromptLateCorrelator(SVector(1.0 + 0.0im, 2.0 + 0.0im, 3.0 + 0.0im), 0.5)
    @test @inferred(get_early(correlator)) == 3.0
    @test @inferred(get_prompt(correlator)) == 2.0
    @test @inferred(get_late(correlator)) == 1.0
    @test get_prompt_index(correlator) == 2
    @test get_accumulators(correlator) == SVector(1.0 + 0.0im, 2.0 + 0.0im, 3.0 + 0.0im)
    @test get_num_ants(correlator) == 1

    correlator = @inferred VeryEarlyPromptLateCorrelator()
    @test correlator.preferred_early_late_to_prompt_code_shift == 0.15
    @test correlator.preferred_very_early_late_to_prompt_code_shift == 0.6
    @test @inferred(get_very_early(correlator)) == 0.0
    @test @inferred(get_very_late(correlator)) == 0.0
    @test get_num_ants(correlator) == 1

    correlator = @inferred VeryEarlyPromptLateCorrelator(num_ants = NumAnts(3))
    @test @inferred(get_very_early(correlator)) == zero(SVector{3,ComplexF64})
    @test get_num_ants(correlator) == 3

    correlator = VeryEarlyPromptLateCorrelator(
        SVector(1.0 + 0.0im, 2.0 + 0.0im, 3.0 + 0.0im, 4.0 + 0.0im, 5.0 + 0.0im),
        0.15,
        0.6,
    )
    @test @inferred(get_very_early(correlator)) == 5.0
    @test @inferred(get_early(correlator)) == 4.0
    @test @inferred(get_prompt(correlator)) == 3.0
    @test @inferred(get_late(correlator)) == 2.0
    @test @inferred(get_very_late(correlator)) == 1.0
    @test get_prompt_index(correlator) == 3
    @test get_num_ants(correlator) == 1

    multi = VeryEarlyPromptLateCorrelator(
        SVector(ntuple(k -> SVector(k + 0.0im, -k + 0.0im), 5)),
        0.15,
        0.6,
    )
    @test get_num_ants(multi) == 2
    @test get_very_early(multi) == SVector(5.0 + 0.0im, -5.0 + 0.0im)
    @test get_very_late(multi) == SVector(1.0 + 0.0im, -1.0 + 0.0im)
end

@testset "A VEML correlator refuses taps that all miss the correlation peak" begin
    # Both shifts at a chip or more: the BOC(1,1) envelope is zero at every tap.
    @test_throws ArgumentError VeryEarlyPromptLateCorrelator(;
        preferred_early_late_to_prompt_code_shift = 1.0,
        preferred_very_early_late_to_prompt_code_shift = 1.5,
    )
    @test_throws ArgumentError VeryEarlyPromptLateCorrelator(;
        preferred_early_late_to_prompt_code_shift = 1.2,
        preferred_very_early_late_to_prompt_code_shift = 1.0,
    )
    # One tap on the peak is enough for a finite slope.
    @test VeryEarlyPromptLateCorrelator(;
        preferred_early_late_to_prompt_code_shift = 0.15,
        preferred_very_early_late_to_prompt_code_shift = 1.2,
    ) isa VeryEarlyPromptLateCorrelator
    @test VeryEarlyPromptLateCorrelator(;
        preferred_early_late_to_prompt_code_shift = 0.99,
        preferred_very_early_late_to_prompt_code_shift = 0.99,
    ) isa VeryEarlyPromptLateCorrelator
end

@testset "Correlator sample shifts" begin
    gpsl1 = GPSL1CA()
    code_frequency = get_code_frequency(gpsl1)
    sampling_frequency = code_frequency * 4
    correlator = EarlyPromptLateCorrelator()
    @test @inferred(
        get_correlator_sample_shifts(correlator, sampling_frequency, code_frequency)
    ) == -2:2:2

    galileo_e1b = GalileoE1B()
    sampling_frequency = get_code_frequency(galileo_e1b) * 4
    correlator = VeryEarlyPromptLateCorrelator()
    @test @inferred(
        get_correlator_sample_shifts(correlator, sampling_frequency, code_frequency)
    ) == -2:1:2

    sampling_frequency = get_code_frequency(galileo_e1b) * 8
    @test @inferred(
        get_correlator_sample_shifts(correlator, sampling_frequency, code_frequency)
    ) == [-5, -1, 0, 1, 5]

    # At least one sample is always shifted, however coarse the sampling.
    @test get_correlator_sample_shifts(
        EarlyPromptLateCorrelator(; preferred_early_late_to_prompt_code_shift = 0.01),
        code_frequency,
        code_frequency,
    ) == [-1, 0, 1]
end

@testset "Correlator number of accumulators" begin
    @test @inferred(get_num_accumulators(EarlyPromptLateCorrelator())) == 3
    @test @inferred(get_num_accumulators(VeryEarlyPromptLateCorrelator())) == 5
end

@testset "is_zero correlator" begin
    zero_corr = EarlyPromptLateCorrelator(SVector(0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im), 0.5)
    @test TrackingLoops.is_zero(zero_corr)
    nonzero_corr =
        EarlyPromptLateCorrelator(SVector(0.0 + 0.0im, 1.0 + 0.0im, 0.0 + 0.0im), 0.5)
    @test !TrackingLoops.is_zero(nonzero_corr)
end

@testset "Zeroing correlator" begin
    correlator = @inferred EarlyPromptLateCorrelator(
        SVector(
            SVector(1.0 + 0.0im, 1.0 + 0.0im),
            SVector(1.0 + 0.0im, 1.0 + 0.0im),
            SVector(1.0 + 0.0im, 1.0 + 0.0im),
        ),
        0.5,
    )
    @test @inferred(zero(correlator)) == EarlyPromptLateCorrelator(
        SVector(
            SVector(0.0 + 0.0im, 0.0 + 0.0im),
            SVector(0.0 + 0.0im, 0.0 + 0.0im),
            SVector(0.0 + 0.0im, 0.0 + 0.0im),
        ),
        0.5,
    )

    correlator =
        @inferred EarlyPromptLateCorrelator(SVector(1.0 + 0.0im, 1.0 + 0.0im, 1.0 + 0.0im), 0.5)
    @test @inferred(zero(correlator)) ==
          EarlyPromptLateCorrelator(SVector(0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im), 0.5)

    vepl = VeryEarlyPromptLateCorrelator(SVector(ntuple(k -> complex(k, 0.0), 5)), 0.2, 0.7)
    zeroed = @inferred zero(vepl)
    @test get_accumulators(zeroed) == zero(SVector{5,ComplexF64})
    @test zeroed.preferred_early_late_to_prompt_code_shift == 0.2
    @test zeroed.preferred_very_early_late_to_prompt_code_shift == 0.7
end

@testset "Filter correlator" begin
    correlator = @inferred EarlyPromptLateCorrelator(
        SVector(
            SVector(1.0 + 0.0im, 1.0 + 1.0im),
            SVector(1.0 + 0.0im, 2.0 + 0.0im),
            SVector(1.0 + 1.0im, 1.0 + 3.0im),
        ),
        0.5,
    )
    # `DefaultPostCorrFilter`'s weights select the last antenna.
    weights = @inferred get_weights(DefaultPostCorrFilter(), NumAnts(2))
    filtered_correlator = @inferred TrackingLoops.apply(
        tap -> TrackingLoops._combine_antennas(weights, tap),
        correlator,
    )
    @test filtered_correlator ==
          EarlyPromptLateCorrelator(SVector(1.0 + 1.0im, 2.0 + 0.0im, 1.0 + 3.0im), 0.5)

    correlator =
        @inferred EarlyPromptLateCorrelator(SVector(1.0 + 0.0im, 1.0 + 0.0im, 1.0 + 0.0im), 0.5)
    filtered_correlator = @inferred TrackingLoops.apply(x -> 2 * x, correlator)
    @test filtered_correlator ==
          EarlyPromptLateCorrelator(SVector(2.0 + 0.0im, 2.0 + 0.0im, 2.0 + 0.0im), 0.5)

    vepl = VeryEarlyPromptLateCorrelator(SVector(ntuple(k -> complex(k, 0.0), 5)), 0.15, 0.6)
    @test get_accumulators(TrackingLoops.apply(x -> 2 * x, vepl)) ==
          SVector(ntuple(k -> complex(2k, 0.0), 5))
end

@testset "Early late sample spacing" begin
    code_frequency = get_code_frequency(GPSL1CA())
    two_ants = SVector(1.0 + 0.0im, 1.0 + 0.0im)
    correlator = @inferred EarlyPromptLateCorrelator(SVector(two_ants, two_ants, two_ants), 0.5)
    @test get_early_late_sample_spacing(correlator, 2e6Hz, code_frequency) == 2
    @test get_early_late_sample_spacing(correlator, 4e6Hz, code_frequency) == 4

    correlator = @inferred VeryEarlyPromptLateCorrelator(
        SVector(two_ants, two_ants, two_ants, two_ants, two_ants),
        0.15,
        0.6,
    )
    @test get_early_late_sample_spacing(correlator, 4e6Hz, code_frequency) == 2
end

@testset "Normalize correlator" begin
    correlator =
        @inferred EarlyPromptLateCorrelator(SVector(1.0 + 0.0im, 1.0 + 0.0im, 1.0 + 0.0im), 0.5)
    @test @inferred(normalize(correlator, 10)) ==
          EarlyPromptLateCorrelator(SVector(0.1 + 0.0im, 0.1 + 0.0im, 0.1 + 0.0im), 0.5)

    correlator = EarlyPromptLateCorrelator(
        SVector(
            SVector(1.0 + 0.0im, 1.0 + 0.0im),
            SVector(1.0 + 0.0im, 1.0 + 0.0im),
            SVector(1.0 + 0.0im, 1.0 + 0.0im),
        ),
        0.5,
    )
    @test @inferred(normalize(correlator, 10)) == EarlyPromptLateCorrelator(
        SVector(
            SVector(0.1 + 0.0im, 0.1 + 0.0im),
            SVector(0.1 + 0.0im, 0.1 + 0.0im),
            SVector(0.1 + 0.0im, 0.1 + 0.0im),
        ),
        0.5,
    )

    # The optional third argument divides out the code amplitude (CBOC).
    correlator =
        EarlyPromptLateCorrelator(SVector(20.0 + 0.0im, 20.0 + 0.0im, 20.0 + 0.0im), 0.5)
    @test @inferred(normalize(correlator, 10, 20.0)) ==
          EarlyPromptLateCorrelator(SVector(0.1 + 0.0im, 0.1 + 0.0im, 0.1 + 0.0im), 0.5)
    @test normalize(correlator, 10, 1) == normalize(correlator, 10)
end

@testset "Correlator update_accumulator keeps the configuration" begin
    epl = EarlyPromptLateCorrelator(; preferred_early_late_to_prompt_code_shift = 0.25)
    updated = update_accumulator(epl, SVector(1.0 + 0.0im, 2.0 + 0.0im, 3.0 + 0.0im))
    @test updated.preferred_early_late_to_prompt_code_shift == 0.25
    @test get_prompt(updated) == 2.0

    vepl = VeryEarlyPromptLateCorrelator(;
        preferred_early_late_to_prompt_code_shift = 0.1,
        preferred_very_early_late_to_prompt_code_shift = 0.5,
    )
    updated = update_accumulator(vepl, SVector(ntuple(k -> complex(k, 0.0), 5)))
    @test updated.preferred_early_late_to_prompt_code_shift == 0.1
    @test updated.preferred_very_early_late_to_prompt_code_shift == 0.5
    @test get_prompt(updated) == 3.0
end

@testset "CorrelatorOutput carries the raw correlator" begin
    correlator = EarlyPromptLateCorrelator(SVector(1.0 + 0.0im, 2.0 + 0.0im, 3.0 + 0.0im), 0.5)
    output = CorrelatorOutput(correlator, 4000, 8000)
    @test output.correlator === correlator
    @test output.integrated_samples == 4000
    @test output.sample_index == 8000
end
