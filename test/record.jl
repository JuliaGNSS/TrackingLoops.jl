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

@testset "A loop record summarises the signal's bit clock and C/N₀ estimator" begin
    signal = GPSL1CA()
    fs = 4e6Hz
    state = SignalLoopState(signal)
    correlator = EarlyPromptLateCorrelator(SVector{3,ComplexF64}(0.5, 1.0, 0.5), 0.5)
    # Without a signal state: no sync, no soft bits, no C/N₀.
    bare = LoopRecord(signal, correlator, cis(0.1), 4000, 8000, 8000, 1, fs)
    @test !bare.bit_synced && bare.sync_change == SYNC_UNCHANGED && bare.blocks_into_symbol == 0
    @test isempty(bare.new_soft_bits)
    @test estimate_cn0(bare.cn0_estimator, 1e-3s) == estimate_cn0(NoCN0Estimator(), 1e-3s)
    changes = SyncChange[]
    for k = 1:200
        p = 2000.0 * (isodd(div(k - 1, 20)) ? -1.0 : 1.0)
        output = CorrelatorOutput(EarlyPromptLateCorrelator(SVector{3,ComplexF64}(0.5p, p, 0.5p), 0.5), 4000, 4000k)
        num_bits = length(get_soft_bits(state))
        state, _, filtered, blocks = apply_record(state, signal, 7, output, fs, 1e-6 / Hz, true)
        record = LoopRecord(signal, filtered, complex(0.0), output, blocks, fs; prn = 7, signal_state = state)
        push!(changes, record.sync_change)
        # Exactly the soft bits this record appended, as a view into the buffer.
        @test record.new_soft_bits == get_soft_bits(state)[num_bits+1:end]
        @test parent(record.new_soft_bits) === get_soft_bits(state)
        @test record.bit_synced == has_bit_or_secondary_code_been_found(state)
        @test record.blocks_into_symbol ==
              (record.bit_synced ? state.bit_buffer.prompt_accumulator_integrated_code_blocks : 0)
        @test record.cn0_estimator === state.cn0_estimator
    end
    # The sync was found on one record, never lost.
    @test count(==(SYNC_FOUND), changes) == 1
    @test !(SYNC_LOST in changes)
    # Bits that were drained before the record is built are not handed on again.
    num_bits = length(get_soft_bits(state))
    @test num_bits > 0
    empty!(get_soft_bits(state))
    output = CorrelatorOutput(EarlyPromptLateCorrelator(SVector{3,ComplexF64}(1000, 2000, 1000), 0.5), 4000, 4000 * 201)
    state, _, filtered, blocks = apply_record(state, signal, 7, output, fs, 1e-6 / Hz, true)
    record = LoopRecord(signal, filtered, complex(0.0), output, blocks, fs; signal_state = state)
    @test record.new_soft_bits == get_soft_bits(state)
    # A record that overshoots the bit boundary loses the sync.
    output = CorrelatorOutput(EarlyPromptLateCorrelator(SVector{3,ComplexF64}(1000, 2000, 1000), 0.5), 4000 * 30, 4000 * 231)
    state, = apply_record(state, signal, 7, output, fs, 1e-6 / Hz, true)
    @test state.sync_change == SYNC_LOST
    # A restarted or re-armed state hands on nothing.
    @test restart_bit_clock(state).sync_change == SYNC_UNCHANGED
    @test isempty(TrackingLoops._new_soft_bits(reset_signal_state(state)))
end
