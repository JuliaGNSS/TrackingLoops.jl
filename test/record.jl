using Random: MersenneTwister

@testset "apply_record advances the bit buffer, C/N0 and prompt filter" begin
    signal = GPSL1CA()
    fs = 4e6Hz
    state = SignalLoopState(signal)
    @test !has_bit_or_secondary_code_been_found(state)
    @test isempty(get_soft_bits(state))
    density = 1.0e-6 / Hz
    for k = 1:200
        sgn = isodd(div(k - 1, 20)) ? -1.0 : 1.0
        p = 2000.0 * sgn
        output = CorrelatorOutput(EarlyPromptLateCorrelator(SVector{3,ComplexF64}(0.5p, p, 0.5p), 0.5), 4000, 4000k)
        state, prompt, filtered, blocks = apply_record(state, signal, 7, output, fs, density, true)
        @test prompt == p / 4000
        @test get_prompt(filtered) == prompt
        @test blocks == 1
    end
    @test has_bit_or_secondary_code_been_found(state)
    @test length(get_soft_bits(state)) >= 8
    @test all(abs.(get_soft_bits(state)) .> 0)
    # C/N0: |P|²/N0 − 1/T with |P| = 0.5 and N0 = 1e-6/Hz over 1 ms records.
    cn0 = estimate_cn0(state, 1023 / get_code_frequency(signal))
    @test 10 * log10(ustrip(Hz, Unitful.linear(cn0))) ≈ 10 * log10(0.25 / 1e-6 - 1000) atol = 1e-6
    # A restarted bit clock forgets sync and keeps the rest.
    restarted = restart_bit_clock(state)
    @test !has_bit_or_secondary_code_been_found(restarted)
    @test restarted.cn0_estimator === state.cn0_estimator
    @test restarted.last_filtered_prompt == state.last_filtered_prompt
end

@testset "A signal's state is updated by keyword, the other fields kept" begin
    state = SignalLoopState(GPSL1CA())
    cn0_estimator = MomentsCN0Estimator(50)
    updated = @inferred SignalLoopState(state; cn0_estimator, last_num_code_blocks = 2)
    @test updated.cn0_estimator === cn0_estimator
    @test updated.last_num_code_blocks == 2
    for field in fieldnames(SignalLoopState)
        field in (:cn0_estimator, :last_num_code_blocks) && continue
        @test getfield(updated, field) === getfield(state, field)
    end
    # A prompt and a polarity of another number type are converted.
    updated = SignalLoopState(state; last_previous_prompt = 1 + 2im, last_carrier_phase_polarity = 1)
    @test updated.last_previous_prompt === complex(1.0, 2.0)
    @test updated.last_carrier_phase_polarity === Int8(1)
end

@testset "A passenger component is de-rotated onto the driver's phase" begin
    # GPS L5Q (pilot, driver) and L5I (data) are in quadrature; the data prompt
    # is rotated onto the real axis before it reaches the bit buffer.
    driver, data = GPSL5Q(), GPSL5I()
    fs = 25e6Hz
    state = SignalLoopState(data)
    p = 2000.0im   # the data component sits on the imaginary axis of the driver's frame
    output = CorrelatorOutput(EarlyPromptLateCorrelator(SVector{3,ComplexF64}(0.5p, p, 0.5p), 0.5), 25_000, 25_000)
    state, prompt, _, _ = apply_record(state, data, 3, output, fs, 1e-6 / Hz, true, get_carrier_phase_offset(driver))
    @test prompt == p / 25_000
    # The recorded prompt is the un-rotated one; the rotation only reaches the bit buffer.
    @test state.last_filtered_prompt == prompt
end

@testset "A loop record identifies its satellite and replica phase" begin
    correlator = EarlyPromptLateCorrelator(SVector{3,ComplexF64}(0.5, 1.0, 0.5), 0.5)
    output = CorrelatorOutput(correlator, 4000, 8000, 3.25)
    record = LoopRecord(GPSL1CA(), correlator, cis(0.1), output, 1, 4e6Hz)
    @test record.prn == 0
    @test record.code_phase == 3.25
    @test record.sample_index == 8000
    @test record.fold_end == 8000
    # The host's per-measurement origin moves onto the band's common one.
    shifted = LoopRecord(GPSL1CA(), correlator, cis(0.1), output, 1, 4e6Hz;
        fold_end = 12_000, prn = 7, sample_offset = 40_000)
    @test shifted.prn == 7
    @test shifted.sample_index == 48_000
    @test shifted.fold_end == 52_000
    @test shifted.integrated_samples == 4000
    # The positional form keeps its eight arguments.
    plain = LoopRecord(GPSL1CA(), correlator, cis(0.1), 4000, 8000, 8000, 1, 4e6Hz)
    @test plain.prn == 0 && isnan(plain.code_phase)
    @test LoopRecord(GPSL1CA(), correlator, cis(0.1), 4000, 8000, 8000, 1, 4e6Hz;
        prn = 3, code_phase = 0.5).prn == 3
end

@testset "A record built from the signal's state carries its bit clock and C/N₀" begin
    signal = GPSL1CA()
    fs = 4e6Hz
    state = SignalLoopState(signal)
    epl(p) = EarlyPromptLateCorrelator(SVector{3,ComplexF64}(0.5p, p, 0.5p), 0.5)
    records = LoopRecord[]
    changes = SyncChange[]
    for k = 1:200
        p = 2000.0 * (isodd(div(k - 1, 20)) ? -1.0 : 1.0)
        output = CorrelatorOutput(epl(p), 4000, 4000k, 0.0)
        previous_prompt = state.last_filtered_prompt
        state, _, filtered, = apply_record(state, signal, 7, output, fs, 1e-6 / Hz, true)
        record = LoopRecord(signal, filtered, output, state, fs; prn = 7)
        @test record.previous_prompt == previous_prompt
        @test record.integrated_code_blocks == state.last_num_code_blocks
        @test record.cn0_estimator === state.cn0_estimator
        @test record.bit_synced == has_bit_or_secondary_code_been_found(state)
        # Exactly the bits this record added, however many the buffer holds.
        @test record.new_soft_bits == get_soft_bits(state)[end-state.last_num_new_soft_bits+1:end]
        push!(changes, record.sync_change)
        if record.bit_synced
            @test record.blocks_into_symbol == state.bit_buffer.prompt_accumulator_integrated_code_blocks
            # Drained after every other record: the next record's bits are still its own.
            iseven(k) && empty!(get_soft_bits(state))
        else
            @test record.blocks_into_symbol == 0
            @test isempty(record.new_soft_bits)
        end
    end
    @test count(==(SYNC_FOUND), changes) == 1
    @test count(==(SYNC_LOST), changes) == 0
    # A record without the state: no sync, no bits, no C/N₀.
    plain = LoopRecord(signal, epl(1.0), 0.0, 4000, 4000, 4000, 1, fs)
    @test !plain.bit_synced && plain.sync_change == SYNC_UNCHANGED
    @test isempty(plain.new_soft_bits) && plain.cn0_estimator isa NoCN0Estimator
    # A restarted bit clock reports no new bits.
    @test restart_bit_clock(state).last_num_new_soft_bits == 0
end

@testset "A lost sync is reported on the record" begin
    signal = GPSL1CA()
    state = SignalLoopState(signal)
    epl(p) = EarlyPromptLateCorrelator(SVector{3,ComplexF64}(0.5p, p, 0.5p), 0.5)
    for k = 1:200
        p = 2000.0 * (isodd(div(k - 1, 20)) ? -1.0 : 1.0)
        state, = apply_record(state, signal, 7, CorrelatorOutput(epl(p), 4000, 4000k), 4e6Hz, 1e-6 / Hz, true)
    end
    @test has_bit_or_secondary_code_been_found(state)
    # Three-block records step the bit accumulator past the boundary.
    for k = 1:7
        output = CorrelatorOutput(epl(6000.0), 12_000, 800_000 + 12_000k, 0.0)
        state, _, filtered, = apply_record(state, signal, 7, output, 4e6Hz, 1e-6 / Hz, true)
        record = LoopRecord(signal, filtered, output, state, 4e6Hz)
        if !has_bit_or_secondary_code_been_found(state)
            @test record.sync_change == SYNC_LOST && !record.bit_synced
            break
        end
        @test record.sync_change == SYNC_UNCHANGED
    end
end

@testset "A synced pilot's records carry the prompt's sign for the four-quadrant discriminators" begin
    # Galileo E1C, whose secondary chip 0 is negative for every PRN: before the
    # sync the replica is primary-only, so the prompts carry the overlay; after it
    # the replica wipes the overlay.
    signal = GalileoE1C()
    fs = 4e6Hz
    code = get_secondary_code(signal)
    epl(p) = EarlyPromptLateCorrelator(SVector{3,ComplexF64}(0.5p, p, 0.5p), 0.5)
    rng = MersenneTwister(2)
    for prn in (1, 3), sign in (1.0, -1.0)
        chip(k) = GNSSSignals.secondary_value(code, prn, mod(k, 25))
        @test chip(0) == -1
        state = SignalLoopState(signal)
        synced_records = 0
        for k = 0:300
            synced = has_bit_or_secondary_code_been_found(state)
            p = (synced ? sign : sign * chip(k)) * 4000.0 * cis(0.05) +
                200 * randn(rng, ComplexF64)
            output = CorrelatorOutput(epl(p), 16_000, 16_000 * (k + 1))
            state, _, filtered, = apply_record(state, signal, prn, output, fs, 1e-6 / Hz, true)
            record = LoopRecord(signal, filtered, output, state, fs; prn)
            if synced
                synced_records += 1
                # The prompt's sign, so the four-quadrant PLL reads the small error,
                # not one half a cycle off. The noise spreads it by ~0.006 cycles RMS,
                # so the bound is about 9σ over the few hundred records per run.
                @test record.carrier_phase_polarity == sign
                @test pll_disc(signal, filtered; polarity = record.carrier_phase_polarity) ≈
                      0.05 / 2π atol = 0.05
                # The first wiped-off record does not chain from a prompt correlated
                # before the sync.
                @test iszero(record.previous_prompt) == (synced_records == 1)
            else
                @test record.carrier_phase_polarity == 0
            end
        end
        @test synced_records > 100
        # A host whose replica has not wiped the overlay under a record says so.
        output = CorrelatorOutput(epl(sign * 4000.0), 16_000, 16_000 * 302)
        unwiped, = apply_record(state, signal, prn, output, fs, 1e-6 / Hz, true; overlay_wiped = false)
        @test unwiped.last_carrier_phase_polarity == 0
    end
end

@testset "The prompt's sign is known only for a dataless signal synced to its secondary code" begin
    synced(signal, polarity) = TrackingLoops.BitBuffer{get_code_block_buffer_type(signal)}(
        zero(get_code_block_buffer_type(signal)),
        0,
        true,
        0,
        Int8(polarity),
        complex(0.0),
        0,
        Float32[],
        TrackingLoops.PhaseAccumulators(),
    )
    polarity(signal, bit_buffer, overlay_wiped = true) =
        TrackingLoops._carrier_phase_polarity(signal, bit_buffer, overlay_wiped)
    # The sync polarity, whatever the overlay's chip 0 (positive for GPS L5Q,
    # negative for Galileo E1C).
    @test polarity(GPSL5Q(), synced(GPSL5Q(), -1)) == -1
    @test polarity(GPSL5Q(), synced(GPSL5Q(), 1)) == 1
    @test polarity(GalileoE1C(), synced(GalileoE1C(), 1)) == 1
    @test polarity(GalileoE1C(), synced(GalileoE1C(), -1)) == -1
    # Not yet synced, or the overlay not wiped under the record.
    @test polarity(GPSL5Q(), TrackingLoops.BitBuffer{get_code_block_buffer_type(GPSL5Q())}()) == 0
    @test polarity(GPSL5Q(), synced(GPSL5Q(), 1), false) == 0
    # Data signals, and (for now) a pilot without a secondary code.
    @test polarity(GPSL5I(), synced(GPSL5I(), 1)) == 0
    @test polarity(GPSL1CA(), synced(GPSL1CA(), 1)) == 0
    @test polarity(GalileoE5aQP(), synced(GalileoE5aQP(), 1)) == 0
    @test @inferred(polarity(GPSL5Q(), synced(GPSL5Q(), 1))) isa Int8
end

@testset "A record whose replica has not wiped the overlay yet stays out of the bit sum" begin
    # GPS L5I, synced to its NH10 secondary code, 1 ms records at 10.23 MHz.
    signal = GPSL5I()
    B = get_code_block_buffer_type(signal)
    synced = TrackingLoops.BitBuffer{B}(
        zero(B),
        0,
        true,
        3,
        Int8(1),
        complex(0.0),
        0,
        Float32[],
        TrackingLoops.PhaseAccumulators(),
    )
    output = CorrelatorOutput(
        EarlyPromptLateCorrelator(SVector{3,ComplexF64}(0.5, 1.0, 0.5) .* 10230, 0.5),
        10230,
        10230,
    )
    fold(overlay_wiped) = TrackingLoops.fold_record(
        signal,
        1,
        synced,
        NoCN0Estimator(),
        DefaultPostCorrFilter(),
        output,
        10.23e6Hz,
        1.0 / Hz,
        true,
        get_carrier_phase_offset(signal);
        overlay_wiped,
    )
    wiped = first(fold(true))
    unwiped = first(fold(false))
    # Wiped, the prompt joins the bit's coherent sum; not yet wiped, only its
    # block counts, and the secondary-code anchor moves along.
    @test wiped.prompt_accumulator != 0
    @test unwiped.prompt_accumulator == 0
    @test wiped.prompt_accumulator_integrated_code_blocks ==
          unwiped.prompt_accumulator_integrated_code_blocks == 1
    @test wiped.secondary_phase == 3
    @test unwiped.secondary_phase == 4
    # By default the record follows the sync with its overlay wiped.
    @test first(TrackingLoops.fold_record(
        signal,
        1,
        synced,
        NoCN0Estimator(),
        DefaultPostCorrFilter(),
        output,
        10.23e6Hz,
        1.0 / Hz,
        true,
        get_carrier_phase_offset(signal),
    )).prompt_accumulator == wiped.prompt_accumulator
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
