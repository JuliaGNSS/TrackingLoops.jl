using Accessors: setproperties

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

@testset "A loop record built from the loop state follows the record contract" begin
    correlator(p) = EarlyPromptLateCorrelator(SVector{3,ComplexF64}(0.5p, p, 0.5p), 0.5)
    fs = 4e6Hz
    signal = GPSL1CA()
    loop = SignalLoopState(signal)
    # The first record has nothing to compare with.
    output = CorrelatorOutput(correlator(2000.0), 4000, 4000)
    folded, _, filtered, blocks = apply_record(loop, signal, 7, output, fs, 1e-6 / Hz, true)
    first_record = LoopRecord(loop, signal, filtered, output, blocks, fs; prn = 7)
    @test iszero(first_record.previous_prompt)
    @test first_record.prn == 7
    @test !first_record.wiped_off && first_record.polarity == 0
    # A record of the same length chains from the last filtered prompt.
    output = CorrelatorOutput(correlator(2000.0im), 4000, 8000)
    _, _, filtered, blocks = apply_record(folded, signal, 7, output, fs, 1e-6 / Hz, true)
    record = LoopRecord(folded, signal, filtered, output, blocks, fs; prn = 7, sample_offset = 100)
    @test record.previous_prompt == folded.last_filtered_prompt
    @test record.sample_index == 8100
    # One of another length does not.
    @test iszero(LoopRecord(folded, signal, filtered, output, 2, fs; prn = 7).previous_prompt)
    # Nor does one whose wipe-off differs: a pilot's first record after its
    # secondary-code sync, which also reads with the sync's polarity.
    pilot = GPSL5Q()
    unsynced = SignalLoopState(pilot)
    @test !unsynced.last_wiped_off
    synced = SignalLoopState(
        setproperties(unsynced.bit_buffer, (; found = true, polarity = Int8(1))),
        unsynced.cn0_estimator,
        unsynced.post_corr_filter,
        cis(0.3),
        1,
        false,
    )
    output = CorrelatorOutput(correlator(2000.0), 25_000, 25_000)
    record = LoopRecord(synced, pilot, correlator(0.08), output, 1, 25e6Hz; prn = 1)
    @test record.wiped_off
    @test record.polarity == get_sync_polarity(pilot, synced.bit_buffer, 1)
    @test iszero(record.previous_prompt)
    # The fold remembers the wipe-off the record was correlated with.
    folded, _, filtered, blocks = apply_record(synced, pilot, 1, output, 25e6Hz, 1e-6 / Hz, true)
    @test folded.last_wiped_off
    @test LoopRecord(folded, pilot, filtered, output, blocks, 25e6Hz; prn = 1).previous_prompt ==
          folded.last_filtered_prompt
    # A record after a sync found earlier in its fold was correlated with the pre-sync
    # replica: not wiped off, no polarity, and remembered as such.
    pre_sync = LoopRecord(synced, pilot, correlator(0.08), output, 1, 25e6Hz; prn = 1,
        correlated_pre_sync = true)
    @test !pre_sync.wiped_off && pre_sync.polarity == 0
    @test pre_sync.previous_prompt == synced.last_filtered_prompt
    folded, = apply_record(synced, pilot, 1, output, 25e6Hz, 1e-6 / Hz, true;
        correlated_pre_sync = true)
    @test !folded.last_wiped_off
    # A pilot without a secondary code is wiped off whether or not it synced.
    l2cl = GPSL2CL()
    @test LoopRecord(SignalLoopState(l2cl), l2cl, correlator(0.08), output, 1, 25e6Hz;
        prn = 1, correlated_pre_sync = true).wiped_off
end
