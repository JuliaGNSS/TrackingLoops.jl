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
