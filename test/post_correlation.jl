using Random: MersenneTwister

@testset "process_correlator_bank_output! advances the bit synchronizer, C/N0 and post-correlation filter" begin
    signal = GPSL1CA()
    fs = 4e6Hz
    state = PostCorrelationState(signal, 7)
    @test !is_bit_or_secondary_code_synced(state)
    @test isempty(get_soft_bits(state))
    density = 1.0e-6 / Hz
    for k = 1:200
        sgn = isodd(div(k - 1, 20)) ? -1.0 : 1.0
        p = 2000.0 * sgn
        output = CorrelatorBankOutput(EarlyPromptLateCorrelatorBank(SVector{3,ComplexF64}(0.5p, p, 0.5p), 0.5), 4000, 4000k)
        state = process_correlator_bank_output!(state, output, fs, density, true)
        @test get_prompt(state.last_filtered_correlator_bank) == p / 4000
        @test state.last_num_code_periods == 1
    end
    @test is_bit_or_secondary_code_synced(state)
    @test length(get_soft_bits(state)) >= 8
    # The host's read-outs: one code period per integration, the bit position, no overlay.
    @test get_last_num_code_periods(state) == 1
    @test get_num_code_periods_into_bit(state) == state.bit_synchronizer.num_code_periods_into_bit
    @test 0 <= get_num_code_periods_into_bit(state) < 20
    @test get_secondary_code_phase(state) == 0
    @test get_bit_polarity(state) in (-1, 1)
    @test get_bit_polarity(PostCorrelationState(signal, 7)) == 0
    @test all(abs.(get_soft_bits(state)) .> 0)
    # C/N0: |P|²/N0 − 1/T with |P| = 0.5 and N0 = 1e-6/Hz over 1 ms integrations.
    cn0 = estimate_cn0(state, 1023 / get_chip_rate(signal))
    @test 10 * log10(ustrip(Hz, Unitful.linear(cn0))) ≈ 10 * log10(0.25 / 1e-6 - 1000) atol = 1e-6
    # A restarted bit clock forgets sync and keeps the rest.
    restarted = reset_sync!(state)
    @test !is_bit_or_secondary_code_synced(restarted)
    @test restarted.cn0_estimator === state.cn0_estimator
    @test restarted.last_filtered_correlator_bank == state.last_filtered_correlator_bank
end

@testset "A signal's state is updated by keyword, the other fields kept" begin
    state = PostCorrelationState(GPSL1CA(), 7)
    cn0_estimator = MomentsCN0Estimator(; num_averaged_integrations = 50)
    updated = @inferred PostCorrelationState(state; cn0_estimator, last_num_code_periods = 2)
    @test updated.cn0_estimator === cn0_estimator
    @test updated.last_num_code_periods == 2
    for field in fieldnames(PostCorrelationState)
        field in (:cn0_estimator, :last_num_code_periods) && continue
        @test getfield(updated, field) === getfield(state, field)
    end
    # A prompt and a polarity of another number type are converted.
    updated = PostCorrelationState(state; previous_filtered_prompt = 1 + 2im, last_carrier_phase_polarity = 1)
    @test updated.previous_filtered_prompt === complex(1.0, 2.0)
    @test updated.last_carrier_phase_polarity === Int8(1)
end

@testset "A passenger component is de-rotated onto the driver's phase" begin
    # GPS L5Q (pilot, driver) and L5I (data) are in quadrature; the data prompt
    # is rotated onto the real axis before it reaches the bit synchronizer.
    driver, data = GPSL5Q(), GPSL5I()
    fs = 25e6Hz
    state = PostCorrelationState(data, 3; driver_carrier_phase_offset = get_carrier_phase_offset(driver))
    p = 2000.0im   # the data component sits on the imaginary axis of the driver's frame
    output = CorrelatorBankOutput(EarlyPromptLateCorrelatorBank(SVector{3,ComplexF64}(0.5p, p, 0.5p), 0.5), 25_000, 25_000)
    state = process_correlator_bank_output!(state, output, fs, 1e-6 / Hz, true)
    prompt = get_prompt(state.last_filtered_correlator_bank)
    # The recorded prompt is the un-rotated one; the rotation only reaches the bit synchronizer.
    @test prompt == p / 25_000
end

@testset "A coherent integration identifies its satellite and replica phase" begin
    correlator_bank = EarlyPromptLateCorrelatorBank(SVector{3,ComplexF64}(0.5, 1.0, 0.5), 0.5)
    output = CorrelatorBankOutput(correlator_bank, 4000, 8000, 3.25)
    state = processed_state(GPSL1CA(), correlator_bank, cis(0.1), output, 1, 4e6Hz)
    @test state.prn == 0
    @test state.last_code_phase == 3.25
    @test state.last_sample_index == 8000
    @test state.last_command_issue_sample_index == 8000
    @test state.sampling_frequency === 4.0e6Hz
    # The host's per-measurement origin moves onto the band's common one.
    shifted = processed_state(GPSL1CA(), correlator_bank, cis(0.1), output, 1, 4e6Hz;
        command_issue_sample_index = 12_000, prn = 7, sample_index_offset = 40_000)
    @test shifted.prn == 7
    @test shifted.last_sample_index == 48_000
    @test shifted.last_command_issue_sample_index == 52_000
    @test shifted.last_num_integrated_samples == 4000
    # The positional form keeps its eight arguments.
    plain = processed_state(GPSL1CA(), correlator_bank, cis(0.1), 4000, 8000, 8000, 1, 4e6Hz)
    @test plain.prn == 0 && isnan(plain.last_code_phase)
    @test processed_state(GPSL1CA(), correlator_bank, cis(0.1), 4000, 8000, 8000, 1, 4e6Hz;
        prn = 3, code_phase = 0.5).prn == 3
end

@testset "The state's type is fixed at construction" begin
    # The latest filtered correlator bank before any integration has the type of every later one,
    # also for a multi-antenna correlator bank, whose filtered bank has one antenna.
    state = PostCorrelationState(GPSL1CA(), 7)
    @test get_prompt(state.last_filtered_correlator_bank) == 0
    @test isempty(get_soft_bits(state)) && state.last_num_new_soft_bits == 0
    output = CorrelatorBankOutput(EarlyPromptLateCorrelatorBank(SVector{3,ComplexF64}(1000, 2000, 1000), 0.5), 4000, 4000)
    @test @inferred(process_correlator_bank_output!(state, output, 4e6Hz, 1e-6 / Hz, true)) isa typeof(state)
    array_bank = EarlyPromptLateCorrelatorBank(; num_ants = NumAnts(2))
    array_state = PostCorrelationState(GPSL1CA(), 7; correlator_bank = array_bank, cn0_estimator = NoCN0Estimator())
    array_output = CorrelatorBankOutput(EarlyPromptLateCorrelatorBank(
        SVector{3,SVector{2,ComplexF64}}(SVector(1000, 900), SVector(2000, 1800), SVector(1000, 900)), 0.5), 4000, 4000)
    @test process_correlator_bank_output!(array_state, array_output, 4e6Hz, nothing, false) isa typeof(array_state)
    @test get_num_ants(array_state.last_filtered_correlator_bank) == 1
end

@testset "The state's latest integration carries its bit clock and C/N₀" begin
    signal = GPSL1CA()
    fs = 4e6Hz
    state = PostCorrelationState(signal, 7)
    epl(p) = EarlyPromptLateCorrelatorBank(SVector{3,ComplexF64}(0.5p, p, 0.5p), 0.5)
    changes = SyncEvent[]
    for k = 1:200
        p = 2000.0 * (isodd(div(k - 1, 20)) ? -1.0 : 1.0)
        output = CorrelatorBankOutput(epl(p), 4000, 4000k, 0.0)
        previous_prompt = get_prompt(state.last_filtered_correlator_bank)
        num_bits = length(get_soft_bits(state))
        state = process_correlator_bank_output!(state, output, fs, 1e-6 / Hz, true)
        @test state.previous_filtered_prompt == previous_prompt
        # Exactly the bits this integration added, however many the soft-bit vector holds.
        @test state.last_num_new_soft_bits == length(get_soft_bits(state)) - num_bits
        push!(changes, state.last_sync_change)
        if is_bit_or_secondary_code_synced(state)
            @test get_num_code_periods_into_bit(state) == state.bit_synchronizer.num_code_periods_into_bit
            # The sign the sync found the soft bits at.
            @test get_bit_polarity(state) == state.bit_synchronizer.bit_polarity != 0
            # Drained after every other integration: the next integration's bits are still its own,
            # and the drained ones are no longer reported.
            if iseven(k)
                drain_soft_bits!(state)
                @test isempty(get_soft_bits(state))
            end
        else
            @test get_num_code_periods_into_bit(state) == 0
            @test state.last_num_new_soft_bits == 0
            @test get_bit_polarity(state) == 0
        end
    end
    @test count(==(SYNC_FOUND), changes) == 1
    @test count(==(SYNC_LOST), changes) == 0
    # A latest integration set without processing: no sync, no bits, no C/N₀.
    plain = processed_state(signal, epl(1.0), 0.0, 4000, 4000, 4000, 1, fs)
    @test !is_bit_or_secondary_code_synced(plain) && plain.last_sync_change == SYNC_UNCHANGED && get_bit_polarity(plain) == 0
    @test plain.last_num_new_soft_bits == 0 && plain.cn0_estimator isa NoCN0Estimator
    # A restarted bit clock reports no new bits.
    restarted = reset_sync!(state)
    @test isempty(get_soft_bits(restarted)) && restarted.last_num_new_soft_bits == 0
end

@testset "A lost sync is reported on the integration" begin
    signal = GPSL1CA()
    state = PostCorrelationState(signal, 7)
    epl(p) = EarlyPromptLateCorrelatorBank(SVector{3,ComplexF64}(0.5p, p, 0.5p), 0.5)
    for k = 1:200
        p = 2000.0 * (isodd(div(k - 1, 20)) ? -1.0 : 1.0)
        state = process_correlator_bank_output!(state, CorrelatorBankOutput(epl(p), 4000, 4000k), 4e6Hz, 1e-6 / Hz, true)
    end
    @test is_bit_or_secondary_code_synced(state)
    # Three-period integrations step the bit accumulator past the boundary.
    for k = 1:7
        output = CorrelatorBankOutput(epl(6000.0), 12_000, 800_000 + 12_000k, 0.0)
        state = process_correlator_bank_output!(state, output, 4e6Hz, 1e-6 / Hz, true)
        if !is_bit_or_secondary_code_synced(state)
            @test state.last_sync_change == SYNC_LOST
            break
        end
        @test state.last_sync_change == SYNC_UNCHANGED
    end
end

@testset "A synced pilot's integrations carry the prompt's sign for the four-quadrant discriminators" begin
    # Galileo E1C, whose secondary chip 0 is negative for every PRN: before the
    # sync the replica is primary-only, so the prompts carry the overlay; after it
    # the replica wipes the overlay.
    signal = GalileoE1C()
    fs = 4e6Hz
    code = get_secondary_code(signal)
    epl(p) = EarlyPromptLateCorrelatorBank(SVector{3,ComplexF64}(0.5p, p, 0.5p), 0.5)
    rng = MersenneTwister(2)
    for prn in (1, 3), sign in (1.0, -1.0)
        chip(k) = GNSSSignals.secondary_value(code, prn, mod(k, 25))
        @test chip(0) == -1
        state = PostCorrelationState(signal, prn)
        synced_integrations = 0
        for k = 0:300
            synced = is_bit_or_secondary_code_synced(state)
            p = (synced ? sign : sign * chip(k)) * 4000.0 * cis(0.05) +
                200 * randn(rng, ComplexF64)
            output = CorrelatorBankOutput(epl(p), 16_000, 16_000 * (k + 1))
            state = process_correlator_bank_output!(state, output, fs, 1e-6 / Hz, true)
            filtered = state.last_filtered_correlator_bank
            if synced
                synced_integrations += 1
                # The prompt's sign, so the four-quadrant PLL reads the small error,
                # not one half a cycle off. The noise spreads it by ~0.006 cycles RMS,
                # so the bound is about 9σ over the few hundred integrations per run.
                @test state.last_carrier_phase_polarity == sign
                @test pll_disc(signal, filtered; polarity = state.last_carrier_phase_polarity) ≈
                      0.05 / 2π atol = 0.05
                # The first wiped-off integration does not chain from a prompt correlated
                # before the sync.
                @test iszero(state.previous_filtered_prompt) == (synced_integrations == 1)
            else
                @test state.last_carrier_phase_polarity == 0
            end
        end
        @test synced_integrations > 100
        # A host whose replica has not wiped the overlay under an integration says so.
        output = CorrelatorBankOutput(epl(sign * 4000.0), 16_000, 16_000 * 302)
        unwiped = process_correlator_bank_output!(state, output, fs, 1e-6 / Hz, true; secondary_code_wiped = false)
        @test unwiped.last_carrier_phase_polarity == 0
    end
end

@testset "The prompt's sign is known only for a dataless signal synced to its secondary code" begin
    synced(signal, polarity) = TrackingLoops.BitSynchronizer{prompt_sign_history_type(signal)}(
        zero(prompt_sign_history_type(signal)),
        0,
        true,
        0,
        Int8(polarity),
        complex(0.0),
        0,
        Float32[],
        TrackingLoops.SyncHypothesisAccumulators(),
    )
    polarity(signal, bit_synchronizer, secondary_code_wiped = true) =
        TrackingLoops._carrier_phase_polarity(signal, bit_synchronizer, secondary_code_wiped)
    # The sync polarity, whatever the overlay's chip 0 (positive for GPS L5Q,
    # negative for Galileo E1C).
    @test polarity(GPSL5Q(), synced(GPSL5Q(), -1)) == -1
    @test polarity(GPSL5Q(), synced(GPSL5Q(), 1)) == 1
    @test polarity(GalileoE1C(), synced(GalileoE1C(), 1)) == 1
    @test polarity(GalileoE1C(), synced(GalileoE1C(), -1)) == -1
    # Not yet synced, or the overlay not wiped under the integration.
    @test polarity(GPSL5Q(), TrackingLoops.BitSynchronizer{prompt_sign_history_type(GPSL5Q())}()) == 0
    @test polarity(GPSL5Q(), synced(GPSL5Q(), 1), false) == 0
    # Data signals, and (for now) a pilot without a secondary code.
    @test polarity(GPSL5I(), synced(GPSL5I(), 1)) == 0
    @test polarity(GPSL1CA(), synced(GPSL1CA(), 1)) == 0
    @test polarity(GalileoE5aQP(), synced(GalileoE5aQP(), 1)) == 0
    @test @inferred(polarity(GPSL5Q(), synced(GPSL5Q(), 1))) isa Int8
end

@testset "An integration whose replica has not wiped the overlay yet stays out of the bit sum" begin
    # GPS L5I, synced to its NH10 secondary code, 1 ms integrations at 10.23 MHz.
    signal = GPSL5I()
    B = prompt_sign_history_type(signal)
    synced = TrackingLoops.BitSynchronizer{B}(
        zero(B),
        0,
        true,
        3,
        Int8(1),
        complex(0.0),
        0,
        Float32[],
        TrackingLoops.SyncHypothesisAccumulators(),
    )
    output = CorrelatorBankOutput(
        EarlyPromptLateCorrelatorBank(SVector{3,ComplexF64}(0.5, 1.0, 0.5) .* 10230, 0.5),
        10230,
        10230,
    )
    state = PostCorrelationState(PostCorrelationState(signal, 1; cn0_estimator = NoCN0Estimator());
        bit_synchronizer = synced)
    process(secondary_code_wiped) = process_correlator_bank_output!(
        state,
        output,
        10.23e6Hz,
        1.0 / Hz,
        true;
        secondary_code_wiped,
    )
    wiped = process(true).bit_synchronizer
    unwiped = process(false).bit_synchronizer
    # Wiped, the prompt joins the bit's coherent sum; not yet wiped, only its
    # period counts, and the secondary-code anchor moves along.
    @test wiped.bit_prompt_sum != 0
    @test unwiped.bit_prompt_sum == 0
    @test wiped.num_code_periods_into_bit ==
          unwiped.num_code_periods_into_bit == 1
    @test wiped.secondary_code_phase == 3
    @test unwiped.secondary_code_phase == 4
    # By default the integration follows the sync with its overlay wiped.
    @test process_correlator_bank_output!(
        state,
        output,
        10.23e6Hz,
        1.0 / Hz,
        true,
    ).bit_synchronizer.bit_prompt_sum == wiped.bit_prompt_sum
end

@testset "The dataless signals with a secondary code" begin
    for signal in (GPSL1C_P(), GPSL5Q(), GalileoE1C(), GalileoE5aQ(), GalileoE5bQ(),
        GalileoE6C(), BeiDouB1C_P(), BeiDouB2aQ())
        @test @inferred(is_dataless_with_secondary_code(signal))
    end
    for signal in (GPSL1CA(), GPSL5I(), GalileoE1B(), GalileoE5aI(), BeiDouB1I(),
        GPSL2CL(), GalileoE5aQP())
        @test !is_dataless_with_secondary_code(signal)
    end
end

function process_correlator_bank_output_allocation_probe(state, signal, output)
    next_state = process_correlator_bank_output!(state, output, 4e6Hz, 1e-6 / Hz, true; command_issue_sample_index = 12_000, sample_index_offset = 40_000)
    next_state, next_state.last_sample_index + next_state.last_command_issue_sample_index + next_state.last_num_new_soft_bits
end

function integration_processing_allocations(state, signal, output)
    process_correlator_bank_output_allocation_probe(state, signal, output)
    @allocated process_correlator_bank_output_allocation_probe(state, signal, output)
end

@testset "Integration processing shares the sample grid and reuses its storage" begin
    signal = GPSL1CA()
    state = PostCorrelationState(signal, 7)
    output = CorrelatorBankOutput(EarlyPromptLateCorrelatorBank(
        SVector{3,ComplexF64}(1000, 2000, 1000), 0.5), 4000, 8000, 3.25)
    state = @inferred process_correlator_bank_output!(state, output, 4e6Hz, 1e-6 / Hz, true; command_issue_sample_index = 12_000, sample_index_offset = 40_000)
    @test state.last_sample_index == 48_000
    @test state.last_command_issue_sample_index == 52_000
    @test state.prn == 7 && state.last_code_phase == 3.25
    # The soft bits are the bit synchronizer's own vector, which the vector loop views.
    @test get_soft_bits(state) === state.bit_synchronizer.soft_bits
    @test integration_processing_allocations(state, signal, output) == 0
end
