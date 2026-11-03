# The per-integration and per-step paths must allocate nothing in steady state.
# `AllocCheck` proves it statically for the arithmetic — the loop-filter step,
# the timeline and the discriminators — and a dynamic `@allocated` check covers
# the whole post-correlation processing, whose bit synchronizer pushes soft bits into a vector that a
# static analysis has to count as growable.
@testset "The per-integration paths are allocation-free" begin
    signal = GPSL1CA()
    fs = 4e6Hz
    density = 1.0e-6 / Hz
    tracking_loop = DelayCompensatedTrackingLoop()
    est_state = init_loop_state(tracking_loop, signal, 100.0Hz, 0.1Hz)
    timeline = NCOTimeline()
    reset_timeline!(timeline, 100.0, 0.1)
    state_ref = Ref(PostCorrelationState(signal))
    est_ref = Ref(est_state)
    # Keeps everything in `Ref`s so the loop returns nothing and the measurement
    # sees only what the processing itself allocates. The loop's own variables are
    # named apart from the testset's: a closure assigning a name that exists in
    # the enclosing scope captures and boxes that variable, and the boxed
    # dispatch would be charged to the processing.
    function run_integrations!(state_ref, est_ref, first_k, n)
        local sgn, p, output, prompt, filtered, code_periods, integration, carrier, code
        st = state_ref[]
        es = est_ref[]
        for k = first_k:(first_k+n-1)
            sgn = isodd(div(k - 1, 20)) ? -1.0 : 1.0
            p = 2000.0 * sgn * cis(0.01)
            output = CorrelatorBankOutput(EarlyPromptLateCorrelatorBank(SVector{3,ComplexF64}(0.5p, p, 0.5p), 0.5), 4000, 4000k)
            st, prompt, filtered, code_periods = process_correlator_bank_output!(st, signal, 7, output, fs, density, true)
            integration = LoopRecord(signal, filtered, output, st, fs)
            es, carrier, code = step_loop!(tracking_loop, es, integration, timeline, Int64(4000k + 8000))
            schedule_command!(timeline, 4000k + 8000, ustrip(Hz, carrier), ustrip(Hz, code))
            advance_timeline!(timeline, 4000k - 4000)
        end
        state_ref[] = st
        est_ref[] = es
        nothing
    end
    # Warm up: sync found, soft-bit vector grown to its working size.
    run_integrations!(state_ref, est_ref, 1, 400)
    @test is_bit_or_secondary_code_synced(state_ref[])
    # The soft bits are drained by their consumer once per chunk; here once.
    empty!(get_soft_bits(state_ref[]))
    run_integrations!(state_ref, est_ref, 401, 1)
    allocated = @allocated run_integrations!(state_ref, est_ref, 402, 200)
    @test allocated == 0
    state = state_ref[]
    est_state = est_ref[]

    output = CorrelatorBankOutput(EarlyPromptLateCorrelatorBank(SVector{3,ComplexF64}(0.5, 1.0, 0.5), 0.5), 4000, 4000)
    integration = LoopRecord(signal, output.correlator_bank, complex(0.0, 0.0), output, 1, fs)
    step_sig = Tuple{typeof(tracking_loop),typeof(est_state),typeof(integration),typeof(timeline),Int64}
    @test isempty(AllocCheck.check_allocs(step_loop!, step_sig; ignore_throw = true))
    conv = ConventionalAssistedPLLAndDLL()
    conv_state = init_loop_state(conv, signal, 100.0Hz, 0.1Hz)
    @test isempty(AllocCheck.check_allocs(step_loop!, Tuple{typeof(conv),typeof(conv_state),typeof(integration),ConstantNCOCommand,Int64}; ignore_throw = true))
    @test isempty(AllocCheck.check_allocs(calc_mean_nco_command, Tuple{NCOTimeline,Int64,Int64}; ignore_throw = true))
    @test isempty(AllocCheck.check_allocs(schedule_command!, Tuple{NCOTimeline,Int64,Float64,Float64}; ignore_throw = true))
    @test isempty(AllocCheck.check_allocs(advance_timeline!, Tuple{NCOTimeline,Int64}; ignore_throw = true))
    @test isempty(AllocCheck.check_allocs(confirm_command!, Tuple{NCOTimeline,Int64,Int64}; ignore_throw = true))
    @test isempty(AllocCheck.check_allocs(cancel_command!, Tuple{NCOTimeline,Int64}; ignore_throw = true))
    @test isempty(AllocCheck.check_allocs(pll_disc, Tuple{typeof(signal),typeof(output.correlator_bank)}; ignore_throw = true))
    @test isempty(AllocCheck.check_allocs(dll_disc, Tuple{typeof(signal),typeof(output.correlator_bank),typeof(0.1Hz),typeof(fs)}; ignore_throw = true))
end

@testset "The vector loop's per-integration path is allocation-free" begin
    signal = GPSL1CA()
    fs = 4e6Hz
    for inner in (ConventionalAssistedPLLAndDLL(), DelayCompensatedTrackingLoop())
        tracking_loop = VectorTrackingLoop(signal; inner)
        state = init_loop_state(tracking_loop, signal, 100.0Hz, 0.1Hz)
        output = CorrelatorBankOutput(EarlyPromptLateCorrelatorBank(SVector{3,ComplexF64}(0.5, 1.0, 0.5), 0.5), 4000, 4000, 0.0)
        integration = LoopRecord(signal, output.correlator_bank, cis(0.1), output, 1, fs; prn = 3)
        nav_processor = tracking_loop.navigation
        group = first(nav_processor.groups)
        slot = first(group.slots)
        # The pieces of an integration that never allocate, as far as AllocCheck sees: not the
        # slot_generation, which may grow a group past its preallocated slots by design, nor
        # the decoding, whose bounded vote tally and logging it cannot see through. The
        # run below measures the whole integration.
        for words_type in (ConstantNCOCommand, NCOTimeline)
            sig = Tuple{typeof(tracking_loop),typeof(state),typeof(integration),words_type,Int64}
            @test isempty(AllocCheck.check_allocs(TrackingLoops._step_satellite, sig; ignore_throw = true))
            @test isempty(AllocCheck.check_allocs(TrackingLoops._snapshot_driver!,
                Tuple{typeof(nav_processor),typeof(group),typeof(slot),typeof(state),typeof(integration),words_type};
                ignore_throw = true))
            @test isempty(AllocCheck.check_allocs(TrackingLoops._snapshot_decoding!,
                Tuple{typeof(nav_processor),typeof(group),typeof(slot),typeof(integration),words_type};
                ignore_throw = true))
            @test isempty(AllocCheck.check_allocs(TrackingLoops._advance_driver!,
                Tuple{typeof(group),typeof(slot),typeof(integration),words_type};
                ignore_throw = true))
        end
        @test isempty(AllocCheck.check_allocs(TrackingLoops._complete_snapshot!,
            Tuple{typeof(nav_processor),typeof(group),typeof(slot)}; ignore_throw = true))
        # A run through the whole tracking_loop, the state kept in a `Ref`: the satellite
        # registers, syncs to nothing and the navigation processor updates every 100 ms.
        timeline = NCOTimeline()
        reset_timeline!(timeline, 100.0, 0.1)
        function run_vector_integrations!(state_ref, tracking_loop, timeline, first_k, n)
            local p, out, rec, carrier, code
            st = state_ref[]
            previous = complex(0.0, 0.0)
            for k = first_k:(first_k+n-1)
                p = cis(0.01k) * (isodd(k ÷ 20) ? 1 : -1)
                out = CorrelatorBankOutput(EarlyPromptLateCorrelatorBank(SVector{3,ComplexF64}(0.5p, p, 0.5p), 0.5), 4000, 4000k, 0.0)
                rec = LoopRecord(signal, out.correlator_bank, previous, out, 1, fs; prn = 3)
                st, carrier, code = step_loop!(tracking_loop, st, rec, timeline, Int64(4000k + 8000))
                schedule_command!(timeline, 4000k + 8000, ustrip(Hz, carrier), ustrip(Hz, code))
                advance_timeline!(timeline, 4000k - 4000)
                previous = p
            end
            state_ref[] = st
            nothing
        end
        state_ref = Ref(state)
        run_vector_integrations!(state_ref, tracking_loop, timeline, 1, 400)
        @test (@allocated run_vector_integrations!(state_ref, tracking_loop, timeline, 401, 400)) == 0
        @test nav_processor.update_index >= 7
    end
end
