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
            output = CorrelatorOutput(
                EarlyPromptLateCorrelator(SVector{3,ComplexF64}(0.5p, p, 0.5p), 0.5),
                4000,
                4000k,
            )
            previous = st.last_filtered_prompt
            st, prompt, filtered, blocks =
                apply_record(st, signal, 7, output, fs, density, true)
            record = LoopRecord(signal, filtered, previous, output, blocks, fs)
            es, carrier, code =
                step_loop(estimator, es, record, timeline, Int64(4000k + 8000))
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

    output = CorrelatorOutput(
        EarlyPromptLateCorrelator(SVector{3,ComplexF64}(0.5, 1.0, 0.5), 0.5),
        4000,
        4000,
    )
    record = LoopRecord(signal, output.correlator, complex(0.0, 0.0), output, 1, fs)
    step_sig =
        Tuple{typeof(estimator),typeof(est_state),typeof(record),typeof(timeline),Int64}
    @test isempty(AllocCheck.check_allocs(step_loop, step_sig; ignore_throw = true))
    conv = ConventionalAssistedPLLAndDLL()
    conv_state = init_estimator_state(conv, signal, 100.0Hz, 0.1Hz)
    @test isempty(
        AllocCheck.check_allocs(
            step_loop,
            Tuple{typeof(conv),typeof(conv_state),typeof(record),FixedNCOWord,Int64};
            ignore_throw = true,
        ),
    )
    @test isempty(
        AllocCheck.check_allocs(
            mean_nco_word,
            Tuple{NCOTimeline,Int64,Int64};
            ignore_throw = true,
        ),
    )
    @test isempty(
        AllocCheck.check_allocs(
            schedule_word!,
            Tuple{NCOTimeline,Int64,Float64,Float64};
            ignore_throw = true,
        ),
    )
    @test isempty(
        AllocCheck.check_allocs(
            promote_words!,
            Tuple{NCOTimeline,Int64};
            ignore_throw = true,
        ),
    )
    @test isempty(
        AllocCheck.check_allocs(
            pll_disc,
            Tuple{typeof(signal),typeof(output.correlator)};
            ignore_throw = true,
        ),
    )
    @test isempty(
        AllocCheck.check_allocs(
            dll_disc,
            Tuple{typeof(signal),typeof(output.correlator),typeof(0.1Hz),typeof(fs)};
            ignore_throw = true,
        ),
    )
end

@testset "The vector loop's per-record path is allocation-free" begin
    signal = GPSL1CA()
    fs = 4e6Hz
    for inner in (ConventionalAssistedPLLAndDLL(), NCOReferencedPLLAndDLL())
        estimator = VectorPLLAndDLL(signal; inner)
        state = init_estimator_state(estimator, signal, 100.0Hz, 0.1Hz)
        output = CorrelatorOutput(
            EarlyPromptLateCorrelator(SVector{3,ComplexF64}(0.5, 1.0, 0.5), 0.5),
            4000,
            4000,
            0.0,
        )
        record = LoopRecord(signal, output.correlator, cis(0.1), output, 1, fs; prn = 3)
        nav = estimator.navigation
        group = first(nav.groups)
        slot = first(group.slots)
        # The pieces of a record that never allocate, as far as AllocCheck sees: not the
        # registration, which may grow a group past its preallocated slots by design, nor
        # the bit clock's decoding, whose bounded vote tally and logging it cannot see
        # through. The run below measures the whole record.
        for words_type in (FixedNCOWord, NCOTimeline)
            sig = Tuple{typeof(estimator),typeof(state),typeof(record),words_type,Int64}
            @test isempty(
                AllocCheck.check_allocs(
                    TrackingLoops._step_satellite,
                    sig;
                    ignore_throw = true,
                ),
            )
            @test isempty(
                AllocCheck.check_allocs(
                    TrackingLoops._snapshot_epoch!,
                    Tuple{
                        typeof(nav),
                        typeof(group),
                        typeof(slot),
                        typeof(state),
                        typeof(record),
                        words_type,
                    };
                    ignore_throw = true,
                ),
            )
        end
        # A run through the whole estimator, the state kept in a `Ref`: the satellite
        # registers, syncs to nothing and the engine cycles every 100 ms.
        timeline = NCOTimeline()
        reset_timeline!(timeline, 100.0, 0.1)
        function run_vector_records!(state_ref, estimator, timeline, first_k, n)
            local p, out, rec, carrier, code
            st = state_ref[]
            previous = complex(0.0, 0.0)
            for k = first_k:(first_k+n-1)
                p = cis(0.01k) * (isodd(k ÷ 20) ? 1 : -1)
                out = CorrelatorOutput(
                    EarlyPromptLateCorrelator(SVector{3,ComplexF64}(0.5p, p, 0.5p), 0.5),
                    4000,
                    4000k,
                    0.0,
                )
                rec = LoopRecord(signal, out.correlator, previous, out, 1, fs; prn = 3)
                st, carrier, code =
                    step_loop(estimator, st, rec, timeline, Int64(4000k + 8000))
                schedule_word!(
                    timeline,
                    4000k + 8000,
                    ustrip(Hz, carrier),
                    ustrip(Hz, code),
                )
                promote_words!(timeline, 4000k - 4000)
                previous = p
            end
            state_ref[] = st
            nothing
        end
        state_ref = Ref(state)
        run_vector_records!(state_ref, estimator, timeline, 1, 400)
        @test (@allocated run_vector_records!(state_ref, estimator, timeline, 401, 400)) ==
              0
        @test nav.cycle_id >= 7
    end
end

@testset "A very-early-prompt-late record runs through the vector loop alloc-free" begin
    # Its tap spacing and DLL variance factor are read per record by the engine.
    signal = GalileoE1B()
    fs = 4.092e6Hz
    estimator = VectorPLLAndDLL(signal)
    taps = SVector(0.3, 0.6, 1.0, 0.6, 0.3)
    timeline = NCOTimeline()
    reset_timeline!(timeline, 100.0, 0.1)
    function run_veml_vector!(state_ref, first_k, n)
        local p, out, rec, carrier, code
        st = state_ref[]
        for k = first_k:(first_k+n-1)
            p = cis(0.01k)
            out = CorrelatorOutput(
                update_accumulator(get_default_correlator(signal), p .* taps),
                16368,
                16368k,
                0.0,
            )
            rec = LoopRecord(signal, out.correlator, cis(0.01(k - 1)), out, 1, fs; prn = 3)
            st, carrier, code =
                step_loop(estimator, st, rec, timeline, Int64(16368k + 32736))
            schedule_word!(timeline, 16368k + 32736, ustrip(Hz, carrier), ustrip(Hz, code))
            promote_words!(timeline, 16368k - 16368)
        end
        state_ref[] = st
        nothing
    end
    state_ref = Ref(init_estimator_state(estimator, signal, 100.0Hz, 0.1Hz))
    run_veml_vector!(state_ref, 1, 200)
    @test (@allocated run_veml_vector!(state_ref, 201, 100)) == 0
end

@testset "A very-early-prompt-late record folds and steps without allocating" begin
    # Galileo E1B's default correlator has five taps; its fold, discriminators and
    # step take other methods than the three-tap ones above.
    signal = GalileoE1B()
    fs = 4.092e6Hz
    density = 1.0e-6 / Hz
    @test get_default_correlator(signal) isa VeryEarlyPromptLateCorrelator
    estimator = ConventionalAssistedPLLAndDLL()
    state_ref = Ref(SignalLoopState(signal))
    est_ref = Ref(init_estimator_state(estimator, signal, 100.0Hz, 0.1Hz))
    taps = SVector(0.3, 0.6, 1.0, 0.6, 0.3)
    function run_veml_records!(state_ref, est_ref, first_k, n)
        local p, output, previous, filtered, blocks, record, carrier, code
        st = state_ref[]
        es = est_ref[]
        for k = first_k:(first_k+n-1)
            p = 1000.0 * cis(0.01k)
            output = CorrelatorOutput(
                update_accumulator(get_default_correlator(signal), p .* taps),
                16368,
                16368k,
            )
            previous = st.last_filtered_prompt
            st, _, filtered, blocks = apply_record(st, signal, 7, output, fs, density, true)
            record = LoopRecord(signal, filtered, previous, output, blocks, fs)
            es, carrier, code =
                step_loop(estimator, es, record, FixedNCOWord(0.0, 0.0), NO_LANDING_SAMPLE)
        end
        state_ref[] = st
        est_ref[] = es
        nothing
    end
    run_veml_records!(state_ref, est_ref, 1, 50)
    empty!(get_soft_bits(state_ref[]))
    run_veml_records!(state_ref, est_ref, 51, 1)
    empty!(get_soft_bits(state_ref[]))
    @test (@allocated run_veml_records!(state_ref, est_ref, 52, 1)) == 0
end
