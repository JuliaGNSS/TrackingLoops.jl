@testset "Correlator bank initial correlator outputs" begin
    @test @inferred(TrackingLoops.zero_correlator_outputs(NumAnts(1), NumCorrelators(3))) isa
          SVector{3,ComplexF64}
    @test @inferred(TrackingLoops.zero_correlator_outputs(NumAnts(4), NumCorrelators(3))) isa
          SVector{3,SVector{4,ComplexF64}}
    @test @inferred(TrackingLoops.zero_correlator_outputs(NumAnts(1), 3)) isa Vector{ComplexF64}
    @test @inferred(TrackingLoops.zero_correlator_outputs(NumAnts(4), 3)) isa
          Vector{SVector{4,ComplexF64}}
end

@testset "Correlator bank constructors and accessors" begin
    correlator_bank = @inferred EarlyPromptLateCorrelatorBank()
    @test @inferred(get_early(correlator_bank)) == 0.0
    @test @inferred(get_prompt(correlator_bank)) == 0.0
    @test @inferred(get_late(correlator_bank)) == 0.0
    @test correlator_bank.preferred_early_late_to_prompt_spacing == 0.5
    @test get_num_ants(correlator_bank) == 1
    @test NumAnts(correlator_bank) === NumAnts(1)

    correlator_bank = @inferred EarlyPromptLateCorrelatorBank(num_ants = NumAnts(1))
    @test @inferred(get_prompt(correlator_bank)) == 0.0
    @test get_num_ants(correlator_bank) == 1

    correlator_bank = @inferred EarlyPromptLateCorrelatorBank(num_ants = NumAnts(2))
    @test @inferred(get_early(correlator_bank)) == SVector(0.0 + 0.0im, 0.0 + 0.0im)
    @test @inferred(get_prompt(correlator_bank)) == SVector(0.0 + 0.0im, 0.0 + 0.0im)
    @test @inferred(get_late(correlator_bank)) == SVector(0.0 + 0.0im, 0.0 + 0.0im)
    @test correlator_bank.preferred_early_late_to_prompt_spacing == 0.5
    @test get_num_ants(correlator_bank) == 2
    @test NumAnts(correlator_bank) === NumAnts(2)

    correlator_bank = EarlyPromptLateCorrelatorBank(SVector(1.0 + 0.0im, 2.0 + 0.0im, 3.0 + 0.0im), 0.5)
    @test @inferred(get_early(correlator_bank)) == 3.0
    @test @inferred(get_prompt(correlator_bank)) == 2.0
    @test @inferred(get_late(correlator_bank)) == 1.0
    @test get_prompt_index(correlator_bank) == 2
    @test get_correlator_outputs(correlator_bank) == SVector(1.0 + 0.0im, 2.0 + 0.0im, 3.0 + 0.0im)
    @test get_num_ants(correlator_bank) == 1

    correlator_bank = @inferred VeryEarlyPromptLateCorrelatorBank()
    @test correlator_bank.preferred_early_late_to_prompt_spacing == 0.15
    @test correlator_bank.preferred_very_early_late_to_prompt_spacing == 0.6
    @test @inferred(get_very_early(correlator_bank)) == 0.0
    @test @inferred(get_very_late(correlator_bank)) == 0.0
    @test get_num_ants(correlator_bank) == 1

    correlator_bank = @inferred VeryEarlyPromptLateCorrelatorBank(num_ants = NumAnts(3))
    @test @inferred(get_very_early(correlator_bank)) == zero(SVector{3,ComplexF64})
    @test get_num_ants(correlator_bank) == 3

    correlator_bank = VeryEarlyPromptLateCorrelatorBank(
        SVector(1.0 + 0.0im, 2.0 + 0.0im, 3.0 + 0.0im, 4.0 + 0.0im, 5.0 + 0.0im),
        0.15,
        0.6,
    )
    @test @inferred(get_very_early(correlator_bank)) == 5.0
    @test @inferred(get_early(correlator_bank)) == 4.0
    @test @inferred(get_prompt(correlator_bank)) == 3.0
    @test @inferred(get_late(correlator_bank)) == 2.0
    @test @inferred(get_very_late(correlator_bank)) == 1.0
    @test get_prompt_index(correlator_bank) == 3
    @test get_num_ants(correlator_bank) == 1

    multi = VeryEarlyPromptLateCorrelatorBank(
        SVector(ntuple(k -> SVector(k + 0.0im, -k + 0.0im), 5)),
        0.15,
        0.6,
    )
    @test get_num_ants(multi) == 2
    @test get_very_early(multi) == SVector(5.0 + 0.0im, -5.0 + 0.0im)
    @test get_very_late(multi) == SVector(1.0 + 0.0im, -1.0 + 0.0im)
end

@testset "A VEML correlator bank refuses correlators that all miss the correlation peak" begin
    # Both shifts at a chip or more: the BOC(1,1) envelope is zero at every correlator.
    @test_throws ArgumentError VeryEarlyPromptLateCorrelatorBank(;
        preferred_early_late_to_prompt_spacing = 1.0,
        preferred_very_early_late_to_prompt_spacing = 1.5,
    )
    @test_throws ArgumentError VeryEarlyPromptLateCorrelatorBank(;
        preferred_early_late_to_prompt_spacing = 1.2,
        preferred_very_early_late_to_prompt_spacing = 1.0,
    )
    # One correlator on the peak is enough for a finite slope.
    @test VeryEarlyPromptLateCorrelatorBank(;
        preferred_early_late_to_prompt_spacing = 0.15,
        preferred_very_early_late_to_prompt_spacing = 1.2,
    ) isa VeryEarlyPromptLateCorrelatorBank
    @test VeryEarlyPromptLateCorrelatorBank(;
        preferred_early_late_to_prompt_spacing = 0.99,
        preferred_very_early_late_to_prompt_spacing = 0.99,
    ) isa VeryEarlyPromptLateCorrelatorBank
end

@testset "Correlator sample shifts" begin
    gpsl1 = GPSL1CA()
    code_frequency = get_chip_rate(gpsl1)
    sampling_frequency = code_frequency * 4
    correlator_bank = EarlyPromptLateCorrelatorBank()
    @test @inferred(
        calc_correlator_sample_offsets(correlator_bank, sampling_frequency, code_frequency)
    ) == -2:2:2

    galileo_e1b = GalileoE1B()
    sampling_frequency = get_chip_rate(galileo_e1b) * 4
    correlator_bank = VeryEarlyPromptLateCorrelatorBank()
    @test @inferred(
        calc_correlator_sample_offsets(correlator_bank, sampling_frequency, code_frequency)
    ) == -2:1:2

    sampling_frequency = get_chip_rate(galileo_e1b) * 8
    @test @inferred(
        calc_correlator_sample_offsets(correlator_bank, sampling_frequency, code_frequency)
    ) == [-5, -1, 0, 1, 5]

    # At least one sample is always shifted, however coarse the sampling.
    @test calc_correlator_sample_offsets(
        EarlyPromptLateCorrelatorBank(; preferred_early_late_to_prompt_spacing = 0.01),
        code_frequency,
        code_frequency,
    ) == [-1, 0, 1]
end

@testset "Correlator bank number of correlators" begin
    @test @inferred(get_num_correlators(EarlyPromptLateCorrelatorBank())) == 3
    @test @inferred(get_num_correlators(VeryEarlyPromptLateCorrelatorBank())) == 5
end

@testset "Zeroing correlator bank" begin
    correlator_bank = @inferred EarlyPromptLateCorrelatorBank(
        SVector(
            SVector(1.0 + 0.0im, 1.0 + 0.0im),
            SVector(1.0 + 0.0im, 1.0 + 0.0im),
            SVector(1.0 + 0.0im, 1.0 + 0.0im),
        ),
        0.5,
    )
    @test @inferred(zero(correlator_bank)) == EarlyPromptLateCorrelatorBank(
        SVector(
            SVector(0.0 + 0.0im, 0.0 + 0.0im),
            SVector(0.0 + 0.0im, 0.0 + 0.0im),
            SVector(0.0 + 0.0im, 0.0 + 0.0im),
        ),
        0.5,
    )

    correlator_bank =
        @inferred EarlyPromptLateCorrelatorBank(SVector(1.0 + 0.0im, 1.0 + 0.0im, 1.0 + 0.0im), 0.5)
    @test @inferred(zero(correlator_bank)) ==
          EarlyPromptLateCorrelatorBank(SVector(0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im), 0.5)

    vepl = VeryEarlyPromptLateCorrelatorBank(SVector(ntuple(k -> complex(k, 0.0), 5)), 0.2, 0.7)
    zeroed = @inferred zero(vepl)
    @test get_correlator_outputs(zeroed) == zero(SVector{5,ComplexF64})
    @test zeroed.preferred_early_late_to_prompt_spacing == 0.2
    @test zeroed.preferred_very_early_late_to_prompt_spacing == 0.7
end

@testset "Filter correlator bank" begin
    correlator_bank = @inferred EarlyPromptLateCorrelatorBank(
        SVector(
            SVector(1.0 + 0.0im, 1.0 + 1.0im),
            SVector(1.0 + 0.0im, 2.0 + 0.0im),
            SVector(1.0 + 1.0im, 1.0 + 3.0im),
        ),
        0.5,
    )
    # `DefaultPostCorrFilter`'s weights select the last antenna.
    weights = @inferred get_weights(DefaultPostCorrFilter(), NumAnts(2))
    filtered_correlator_bank = @inferred TrackingLoops.apply(
        correlator_output -> TrackingLoops._filter_correlator_output(correlator_output, weights),
        correlator_bank,
    )
    @test filtered_correlator_bank ==
          EarlyPromptLateCorrelatorBank(SVector(1.0 + 1.0im, 2.0 + 0.0im, 1.0 + 3.0im), 0.5)

    correlator_bank =
        @inferred EarlyPromptLateCorrelatorBank(SVector(1.0 + 0.0im, 1.0 + 0.0im, 1.0 + 0.0im), 0.5)
    filtered_correlator_bank = @inferred TrackingLoops.apply(x -> 2 * x, correlator_bank)
    @test filtered_correlator_bank ==
          EarlyPromptLateCorrelatorBank(SVector(2.0 + 0.0im, 2.0 + 0.0im, 2.0 + 0.0im), 0.5)

    vepl = VeryEarlyPromptLateCorrelatorBank(SVector(ntuple(k -> complex(k, 0.0), 5)), 0.15, 0.6)
    @test get_correlator_outputs(TrackingLoops.apply(x -> 2 * x, vepl)) ==
          SVector(ntuple(k -> complex(2k, 0.0), 5))
end

@testset "Early late sample spacing" begin
    code_frequency = get_chip_rate(GPSL1CA())
    two_ants = SVector(1.0 + 0.0im, 1.0 + 0.0im)
    correlator_bank = @inferred EarlyPromptLateCorrelatorBank(SVector(two_ants, two_ants, two_ants), 0.5)
    @test calc_early_late_sample_spacing(correlator_bank, 2e6Hz, code_frequency) == 2
    @test calc_early_late_sample_spacing(correlator_bank, 4e6Hz, code_frequency) == 4

    correlator_bank = @inferred VeryEarlyPromptLateCorrelatorBank(
        SVector(two_ants, two_ants, two_ants, two_ants, two_ants),
        0.15,
        0.6,
    )
    @test calc_early_late_sample_spacing(correlator_bank, 4e6Hz, code_frequency) == 2
end

@testset "Normalize correlator bank" begin
    correlator_bank =
        @inferred EarlyPromptLateCorrelatorBank(SVector(1.0 + 0.0im, 1.0 + 0.0im, 1.0 + 0.0im), 0.5)
    @test @inferred(normalize(correlator_bank, 10)) ==
          EarlyPromptLateCorrelatorBank(SVector(0.1 + 0.0im, 0.1 + 0.0im, 0.1 + 0.0im), 0.5)

    correlator_bank = EarlyPromptLateCorrelatorBank(
        SVector(
            SVector(1.0 + 0.0im, 1.0 + 0.0im),
            SVector(1.0 + 0.0im, 1.0 + 0.0im),
            SVector(1.0 + 0.0im, 1.0 + 0.0im),
        ),
        0.5,
    )
    @test @inferred(normalize(correlator_bank, 10)) == EarlyPromptLateCorrelatorBank(
        SVector(
            SVector(0.1 + 0.0im, 0.1 + 0.0im),
            SVector(0.1 + 0.0im, 0.1 + 0.0im),
            SVector(0.1 + 0.0im, 0.1 + 0.0im),
        ),
        0.5,
    )

    # The optional third argument divides out the code amplitude (CBOC).
    correlator_bank =
        EarlyPromptLateCorrelatorBank(SVector(20.0 + 0.0im, 20.0 + 0.0im, 20.0 + 0.0im), 0.5)
    @test @inferred(normalize(correlator_bank, 10, 20.0)) ==
          EarlyPromptLateCorrelatorBank(SVector(0.1 + 0.0im, 0.1 + 0.0im, 0.1 + 0.0im), 0.5)
    @test normalize(correlator_bank, 10, 1) == normalize(correlator_bank, 10)
end

@testset "Correlator set_correlator_outputs keeps the configuration" begin
    epl = EarlyPromptLateCorrelatorBank(; preferred_early_late_to_prompt_spacing = 0.25)
    updated = set_correlator_outputs(epl, SVector(1.0 + 0.0im, 2.0 + 0.0im, 3.0 + 0.0im))
    @test updated.preferred_early_late_to_prompt_spacing == 0.25
    @test get_prompt(updated) == 2.0

    vepl = VeryEarlyPromptLateCorrelatorBank(;
        preferred_early_late_to_prompt_spacing = 0.1,
        preferred_very_early_late_to_prompt_spacing = 0.5,
    )
    updated = set_correlator_outputs(vepl, SVector(ntuple(k -> complex(k, 0.0), 5)))
    @test updated.preferred_early_late_to_prompt_spacing == 0.1
    @test updated.preferred_very_early_late_to_prompt_spacing == 0.5
    @test get_prompt(updated) == 3.0
end

@testset "CorrelatorBankOutput carries the raw correlator bank" begin
    correlator_bank = EarlyPromptLateCorrelatorBank(SVector(1.0 + 0.0im, 2.0 + 0.0im, 3.0 + 0.0im), 0.5)
    output = CorrelatorBankOutput(correlator_bank, 4000, 8000)
    @test output.correlator_bank === correlator_bank
    @test output.num_integrated_samples == 4000
    @test output.sample_index == 8000
    # A producer that does not report the replica's code phase leaves it `NaN`.
    @test isnan(output.code_phase)
    @test CorrelatorBankOutput(correlator_bank, 4000, 8000, 1022.75).code_phase == 1022.75
end
