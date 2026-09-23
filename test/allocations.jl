# The per-record and per-step paths must allocate nothing in steady state.
# `AllocCheck` proves it statically for the arithmetic — the loop-filter step,
# the timeline and the discriminators — and a dynamic `@allocated` check covers
# the whole record fold, whose bit buffer pushes soft bits into a vector that a
# static analysis has to count as growable.
@testset "The per-record paths are allocation-free" begin
    signal = GPSL1CA()
    fs = 4e6Hz
    density = 1.0e-6 / Hz
    estimator = NCOReferencedPLLAndDLL()
    est_state = init_estimator_state(estimator, signal, 100.0Hz, 0.1Hz)
    timeline = NCOTimeline()
    reset_timeline!(timeline, 100.0, 0.1)
    state_ref = Ref(SignalLoopState(signal))
    est_ref = Ref(est_state)
    # Keeps everything in `Ref`s so the loop returns nothing and the measurement
    # sees only what the fold itself allocates. The loop's own variables are
    # named apart from the testset's: a closure assigning a name that exists in
    # the enclosing scope captures and boxes that variable, and the boxed
    # dispatch would be charged to the fold.
    function run_records!(state_ref, est_ref, first_k, n)
        local sgn, p, output, previous, prompt, filtered, blocks, record, carrier, code
        st = state_ref[]
        es = est_ref[]
        for k = first_k:(first_k+n-1)
            sgn = isodd(div(k - 1, 20)) ? -1.0 : 1.0
            p = 2000.0 * sgn * cis(0.01)
            output = CorrelatorOutput(EarlyPromptLateCorrelator(SVector{3,ComplexF64}(0.5p, p, 0.5p), 0.5), 4000, 4000k)
            previous = st.last_filtered_prompt
            st, prompt, filtered, blocks = apply_record(st, signal, 7, output, fs, density, true)
            record = LoopRecord(signal, filtered, previous, output, blocks, fs)
            es, carrier, code = step_loop(estimator, es, record, timeline, Int64(4000k + 8000))
            schedule_word!(timeline, 4000k + 8000, ustrip(Hz, carrier), ustrip(Hz, code))
            promote_words!(timeline, 4000k - 4000)
        end
        state_ref[] = st
        est_ref[] = es
        nothing
    end
    # Warm up: sync found, soft-bit vector grown to its working size.
    run_records!(state_ref, est_ref, 1, 400)
    @test has_bit_or_secondary_code_been_found(state_ref[])
    # The soft bits are drained by their consumer once per fold; here once.
    empty!(get_soft_bits(state_ref[]))
    run_records!(state_ref, est_ref, 401, 1)
    allocated = @allocated run_records!(state_ref, est_ref, 402, 200)
    @test allocated == 0
    state = state_ref[]
    est_state = est_ref[]

    output = CorrelatorOutput(EarlyPromptLateCorrelator(SVector{3,ComplexF64}(0.5, 1.0, 0.5), 0.5), 4000, 4000)
    record = LoopRecord(signal, output.correlator, complex(0.0, 0.0), output, 1, fs)
    step_sig = Tuple{typeof(estimator),typeof(est_state),typeof(record),typeof(timeline),Int64}
    @test isempty(AllocCheck.check_allocs(step_loop, step_sig; ignore_throw = true))
    conv = ConventionalAssistedPLLAndDLL()
    conv_state = init_estimator_state(conv, signal, 100.0Hz, 0.1Hz)
    @test isempty(AllocCheck.check_allocs(step_loop, Tuple{typeof(conv),typeof(conv_state),typeof(record),FixedNCOWord,Int64}; ignore_throw = true))
    @test isempty(AllocCheck.check_allocs(mean_nco_word, Tuple{NCOTimeline,Int64,Int64}; ignore_throw = true))
    @test isempty(AllocCheck.check_allocs(schedule_word!, Tuple{NCOTimeline,Int64,Float64,Float64}; ignore_throw = true))
    @test isempty(AllocCheck.check_allocs(promote_words!, Tuple{NCOTimeline,Int64}; ignore_throw = true))
    @test isempty(AllocCheck.check_allocs(pll_disc, Tuple{typeof(signal),typeof(output.correlator)}; ignore_throw = true))
    @test isempty(AllocCheck.check_allocs(dll_disc, Tuple{typeof(signal),typeof(output.correlator),typeof(0.1Hz),typeof(fs)}; ignore_throw = true))
end
