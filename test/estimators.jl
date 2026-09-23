# The loop model of GNSSReceiver's delay-aware tests: a carrier whose phase
# error advances by `2π (f_true − w) Δt` per record under the replica word `w`,
# the word the estimator commands landing `d` records after the record that
# produced it.
const LOOP_SIGNAL = GPSL1CA()
const LOOP_FS = 4e6Hz
const LOOP_N = 4000

loop_epl(late, prompt, early) =
    EarlyPromptLateCorrelator(SVector{3,ComplexF64}(late, prompt, early), 0.5)

function simulate_delayed_loop(estimator, d; f_true = 130.0, handover = 100.0, phi0 = 0.8, steps = 600)
    code_doppler = handover * Hz * get_code_center_frequency_ratio(LOOP_SIGNAL)
    state = init_estimator_state(estimator, LOOP_SIGNAL, handover * Hz, code_doppler)
    timeline = NCOTimeline()
    reset_timeline!(timeline, handover, ustrip(Hz, code_doppler))
    dt = LOOP_N / ustrip(Hz, LOOP_FS)
    phase = phi0
    phases = Float64[]
    words = Float64[]
    previous_prompt = complex(0.0, 0.0)
    for k = 0:steps-1
        a = k * LOOP_N
        b = a + LOOP_N
        w, _ = nco_word_at(timeline, a)
        mean_phase = phase + π * (f_true - w) * dt
        phase += 2π * (f_true - w) * dt
        push!(phases, wrap_half_cycle(mean_phase))
        push!(words, w)
        p = cis(mean_phase)
        output = CorrelatorOutput(loop_epl(0.5p, p, 0.5p), LOOP_N, b)
        landing = Int64(b + d * LOOP_N)
        record = LoopRecord(LOOP_SIGNAL, output.correlator, previous_prompt, output, 1, LOOP_FS)
        state, carrier, code = step_loop(estimator, state, record, timeline, landing)
        previous_prompt = p
        schedule_word!(timeline, landing, ustrip(Hz, carrier), ustrip(Hz, code))
        promote_words!(timeline, b - LOOP_N)
    end
    phases, words
end

@testset "With no delay the NCO-referenced loop is the conventional loop" begin
    conventional = simulate_delayed_loop(ConventionalAssistedPLLAndDLL(), 0)
    referenced = simulate_delayed_loop(NCOReferencedPLLAndDLL(), 0)
    @test referenced[1] == conventional[1]
    @test referenced[2] == conventional[2]
    control = simulate_delayed_loop(NCOReferencedPLLAndDLL(; predict_landing = false), 0)
    @test control[2] == conventional[2]
    @test all(abs.(referenced[1][400:end]) .< 0.2)
    @test all(abs.(referenced[2][400:end] .- 130.0) .< 1.0)
end

@testset "The NCO-referenced loop holds lock through $d records of delay" for d in 1:6
    phases, words = simulate_delayed_loop(NCOReferencedPLLAndDLL(), d; steps = 1200)
    @test all(abs.(phases[1000:end]) .< 0.05)
    @test all(abs.(words[1000:end] .- 130.0) .< 0.3)
    @test maximum(abs.(phases[100:end])) < 1.0
end

@testset "The conventional loop and the negative control limit-cycle at four records of delay" begin
    for estimator in (ConventionalAssistedPLLAndDLL(), NCOReferencedPLLAndDLL(; predict_landing = false))
        phases, words = simulate_delayed_loop(estimator, 4; steps = 1200)
        tail = words[600:end] .- 130.0
        @test sqrt(sum(abs2, tail) / length(tail)) > 20
        @test maximum(abs.(phases[600:end])) > 1.0
    end
end

@testset "Estimator state construction and reset" begin
    estimator = NCOReferencedPLLAndDLL()
    state = init_estimator_state(estimator, GPSL1CA(), 1234.0Hz, 0.8Hz)
    @test state isa SatNCOReferencedPLLAndDLL
    @test state.init_carrier_doppler == 1234.0Hz
    @test state.carrier_loop_filter_bandwidth == 18.0Hz
    @test isnan(state.previous_record_center)
    narrow = NCOReferencedPLLAndDLL(; carrier_loop_filter_bandwidth = 12.0Hz)
    @test init_estimator_state(narrow, GPSL1CA(), 0.0Hz, 0.0Hz).carrier_loop_filter_bandwidth == 12.0Hz
    reset_state = reset_estimator_state(narrow, init_estimator_state(narrow, GPSL1CA(), 0.0Hz, 0.0Hz), 50.0Hz, 0.03Hz)
    @test reset_state.init_carrier_doppler == 50.0Hz
    @test reset_state.carrier_loop_filter_bandwidth == 12.0Hz
    @test reset_state.carrier_loop_filter.x1 == 0.0Hz
    conv = init_estimator_state(ConventionalAssistedPLLAndDLL(), GalileoE1B(), 0.0Hz, 0.0Hz)
    @test conv isa SatConventionalPLLAndDLL
    @test conv.carrier_loop_filter_bandwidth == default_carrier_loop_filter_bandwidth(GalileoE1B())
    @test TrackingLoops.estimator_state_type(ConventionalAssistedPLLAndDLL(), GPSL1CA()) === typeof(conv)
end
