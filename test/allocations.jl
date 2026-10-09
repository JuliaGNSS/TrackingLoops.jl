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

@testset "The vector loop's per-record path is allocation-free" begin
    signal = GPSL1CA()
    fs = 4e6Hz
    for inner in (ConventionalAssistedPLLAndDLL(), NCOReferencedPLLAndDLL())
        estimator = VectorPLLAndDLL(signal; inner)
        state = init_estimator_state(estimator, signal, 100.0Hz, 0.1Hz)
        output = CorrelatorOutput(EarlyPromptLateCorrelator(SVector{3,ComplexF64}(0.5, 1.0, 0.5), 0.5), 4000, 4000, 0.0)
        record = LoopRecord(signal, output.correlator, cis(0.1), output, 1, fs; prn = 3,
            signal_state = SignalLoopState(signal))
        nav = estimator.navigation
        group = first(nav.groups)
        slot = first(group.slots)
        # The pieces of a record that never allocate, as far as AllocCheck sees: not the
        # registration, which may grow a group past its preallocated slots by design, nor
        # the bit clock's decoding, whose bounded vote tally and logging it cannot see
        # through. The run below measures the whole record.
        for words_type in (FixedNCOWord, NCOTimeline)
            sig = Tuple{typeof(estimator),typeof(state),typeof(record),words_type,Int64}
            @test isempty(AllocCheck.check_allocs(TrackingLoops._step_satellite, sig; ignore_throw = true))
            @test isempty(AllocCheck.check_allocs(TrackingLoops._snapshot_driver_half!,
                Tuple{typeof(nav),typeof(group),typeof(slot),typeof(state),typeof(record),words_type};
                ignore_throw = true))
            @test isempty(AllocCheck.check_allocs(TrackingLoops._snapshot_decoding_half!,
                Tuple{typeof(nav),typeof(group),typeof(slot),typeof(record),words_type};
                ignore_throw = true))
        end
        @test isempty(AllocCheck.check_allocs(TrackingLoops._complete_snapshot!,
            Tuple{typeof(nav),typeof(group),typeof(slot)}; ignore_throw = true))
        # A run through the whole estimator as a host drives it, the states kept in
        # `Ref`s: each record folded into the signal's state, then stepped. The
        # satellite registers, finds the bit edges, decodes bits that never sync a frame
        # and the engine cycles every 100 ms.
        timeline = NCOTimeline()
        reset_timeline!(timeline, 100.0, 0.1)
        function run_vector_records!(state_ref, signal_ref, estimator, timeline, first_k, n)
            local p, out, rec, carrier, code, filtered
            st = state_ref[]
            sig = signal_ref[]
            for k = first_k:(first_k+n-1)
                p = 4000 * cis(0.01k) * (isodd(k ÷ 20) ? 1 : -1)
                out = CorrelatorOutput(EarlyPromptLateCorrelator(SVector{3,ComplexF64}(0.5p, p, 0.5p), 0.5), 4000, 4000k, 0.0)
                previous = sig.last_filtered_prompt
                sig, _, filtered, blocks = apply_record(sig, signal, 3, out, fs, 1e-4 / Hz, true)
                rec = LoopRecord(signal, filtered, previous, out, blocks, fs; prn = 3, signal_state = sig)
                st, carrier, code = step_loop(estimator, st, rec, timeline, Int64(4000k + 8000))
                schedule_word!(timeline, 4000k + 8000, ustrip(Hz, carrier), ustrip(Hz, code))
                promote_words!(timeline, 4000k - 4000)
                # The host drains the soft bits after every record.
                empty!(get_soft_bits(sig))
            end
            state_ref[] = st
            signal_ref[] = sig
            nothing
        end
        state_ref = Ref(state)
        signal_ref = Ref(SignalLoopState(signal))
        run_vector_records!(state_ref, signal_ref, estimator, timeline, 1, 400)
        @test has_bit_or_secondary_code_been_found(signal_ref[])
        @test (@allocated run_vector_records!(state_ref, signal_ref, estimator, timeline, 401, 400)) == 0
        @test satellite_report(estimator, signal, 3).bit_synced
        @test isfinite(satellite_report(estimator, signal, 3).cn0_dbhz)
        @test nav.cycle_id >= 7
    end
end
