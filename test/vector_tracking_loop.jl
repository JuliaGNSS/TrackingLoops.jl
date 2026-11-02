using TrackingLoopFilters: filter_loop, ThirdOrderBilinearLF
const TL = TrackingLoops

# The satellite's own loop under the vector loop, without the navigation processor around
# it: what `step_loop!` runs after the navigation processor's part.
step_satellite(tracking_loop::VectorTrackingLoop, args...) = TL._step_satellite(tracking_loop, args...)
step_satellite(tracking_loop, args...) = step_loop!(tracking_loop, args...)

# A host's C/N₀ estimator that reads `cn0_dbhz` whatever the prompts.
struct ReportedCN0 <: TrackingLoops.AbstractCN0Estimator
    cn0_dbhz::Float64
end
TrackingLoops.estimate_cn0(estimator::ReportedCN0, coherent_integration_time) = estimator.cn0_dbhz * dBHz

# `post_correlation_state` as read by a host whose C/N₀ estimator reads `cn0_dbhz`.
with_cn0(post_correlation_state::PostCorrelationState, cn0_dbhz) =
    PostCorrelationState(post_correlation_state; cn0_estimator = ReportedCN0(cn0_dbhz))

# One integration at `sample_index` with the given raw correlator outputs, normalised the
# way `process_correlator_bank_output!` does before the loop filters see it (the post-correlation filter
# is the identity for the first prompt), its host reading `cn0_dbhz`.
function vector_integration(signal, correlator_outputs, num_samples, fs; previous_prompt = complex(0.0, 0.0),
        sample_index = num_samples, cn0_dbhz = 45.0)
    correlator_bank = set_correlator_outputs(get_default_correlator_bank(signal), SVector(correlator_outputs...))
    output = CorrelatorBankOutput(correlator_bank, num_samples, sample_index)
    normalized = normalize(correlator_bank, num_samples, get_code_amplitude(signal))
    with_cn0(processed_state(signal, normalized, previous_prompt, output, 1, fs), cn0_dbhz)
end

# An integration whose prompt is `prompt`, its early and late correlators half of it.
prompt_integration(signal, prompt, num_samples, fs; sample_index = num_samples, previous_prompt = complex(0.0),
        cn0_dbhz = 45.0) =
    vector_integration(signal, (500.0prompt, 1000.0prompt, 500.0prompt), num_samples, fs; previous_prompt,
        sample_index, cn0_dbhz)

# A satellite in the vector loop, commanded the Doppler `ramp` and the code correction
# `code_update`, its residual PLL taking over from the scalar command `scalar_doppler`.
function commanded(state, ramp; code_update = 0.0Hz, lead = 0.0s, scalar_doppler = ramp.doppler, fs = 1e6)
    TL._set_vector_commands(TL._add_to_vector_loop(state), code_update, ramp, lead, ConstantNCOCommand(scalar_doppler, 0.0), 0, fs)
end

@testset "Vector PLL and DLL state" begin
    tracking_loop = VectorTrackingLoop(GPSL1CA())
    @test tracking_loop.scalar_loop isa ScalarTrackingLoop{ThirdOrderAssistedBilinearLF}
    state = init_loop_state(tracking_loop, GPSL1CA(), 500.0Hz, 100.0Hz)
    @test state isa VectorLoopState
    @test state.scalar_state == init_loop_state(ScalarTrackingLoop(ThirdOrderAssistedBilinearLF), GPSL1CA(), 500.0Hz, 100.0Hz)
    @test state.vector_controlled == false
    @test state.code_disc_sum == (0, 0.0)
    @test state.fll_disc_sum == (0, 0.0Hz)
    @test all(isnan, state.previous_replica_tail)
    @test state.code_frequency_correction == 0.0Hz
    @test state.code_frequency_correction_history == (0.0Hz, 0.0Hz, 0.0Hz)
    @test state.code_command_delay == 0.0s
    @test isnan(state.ramp.reference_time)
    @test !TL._has_carrier_ramp(state)
    @test state.residual_pll.bandwidth == 10.0Hz
    @test loop_state_type(tracking_loop, GPSL1CA()) === typeof(state)
    @test TL._disc_mean(state.code_disc_sum) === nothing
    @test TL._disc_mean(state.fll_disc_sum) === nothing
end

@testset "Vector PLL and DLL seeding" begin
    # Auto bandwidths resolve from the driver signal, sized exactly like the
    # inner (scalar fallback) loop.
    state = init_loop_state(VectorTrackingLoop(GPSL1CA()), GPSL1CA(), 100.0Hz, 0.0Hz)
    @test state.scalar_state.bandwidths.wide_carrier == 50.0Hz
    @test state.scalar_state.bandwidths.narrow_carrier == 18.0Hz
    @test state.scalar_state.bandwidths.fll_assist == 5.0Hz
    @test state.scalar_state.bandwidths.code == 1.0Hz
    @test state.scalar_state.reference_carrier_doppler == 100.0Hz
    @test get_carrier_loop_stage(state) == FLL_ASSISTED_PLL
    explicit = ScalarTrackingLoop(ThirdOrderAssistedBilinearLF; narrow_carrier_loop_filter_bandwidth = 12.0Hz, code_loop_filter_bandwidth = 1.0Hz)
    explicit_state = init_loop_state(VectorTrackingLoop(GPSL1CA(); scalar_loop = explicit), GPSL1CA(), 100.0Hz, 0.0Hz)
    @test explicit_state.scalar_state.bandwidths.narrow_carrier == 12.0Hz
    @test explicit_state.scalar_state.bandwidths.code == 1.0Hz
    bumped = VectorTrackingLoop(GPSL1CA(); scalar_loop = ScalarTrackingLoop(explicit; narrow_carrier_loop_filter_bandwidth = 15.0Hz))
    @test bumped.scalar_loop.bandwidths.narrow_carrier_loop_filter_bandwidth == 15.0Hz
    @test bumped.scalar_loop.bandwidths.code_loop_filter_bandwidth == 1.0Hz
    referenced = init_loop_state(VectorTrackingLoop(GalileoE1B(); scalar_loop = DelayCompensatedTrackingLoop()), GalileoE1B(), 0.0Hz, 0.0Hz)
    @test referenced.scalar_state isa ScalarLoopState
    @test referenced.scalar_state.bandwidths.narrow_carrier ==
          default_narrow_carrier_loop_filter_bandwidth(GalileoE1B())
    # The residual PLL's bandwidth is the tracking_loop's, for either inner loop; the
    # inner loop's own bandwidths stay.
    for inner in (explicit, DelayCompensatedTrackingLoop())
        tracking_loop = VectorTrackingLoop(GPSL1CA(); scalar_loop = inner, residual_pll_loop_filter_bandwidth = 5.0Hz)
        vector_state = init_loop_state(tracking_loop, GPSL1CA(), 100.0Hz, 0.0Hz)
        @test vector_state.residual_pll.bandwidth == 5.0Hz
        @test vector_state.scalar_state == init_loop_state(inner, GPSL1CA(), 100.0Hz, 0.0Hz)
        # A satellite keeps it across the hand-overs.
        @test TL._add_to_vector_loop(vector_state).residual_pll.bandwidth == 5.0Hz
        @test TL._hand_over_to_scalar_loop(TL._add_to_vector_loop(vector_state), ConstantNCOCommand(1.0, 0.0), 0, 1e6).residual_pll.bandwidth == 5.0Hz
    end
    @test_throws ArgumentError VectorTrackingLoop(GPSL1CA(); residual_pll_loop_filter_bandwidth = 0.0Hz)
end

@testset "The filter alone sets the carrier NCO, with $(nameof(typeof(inner)))" for inner in (
    ScalarTrackingLoop(ThirdOrderAssistedBilinearLF),
    DelayCompensatedTrackingLoop(),
)
    # The command is the filter's ramp at the middle of the integration it takes effect in, whatever
    # the prompt reads; the code is commanded from it, carrier-aided, plus the filter's
    # code correction.
    signal = GPSL1CA()
    fs = 5e6Hz
    tracking_loop = VectorTrackingLoop(signal; scalar_loop = inner)
    ramp = TL.CarrierDopplerRamp(110.0, 4.0, 0.0)
    state = commanded(init_loop_state(tracking_loop, signal, 100.0Hz, 0.1Hz), ramp; code_update = -0.25Hz)
    nco_history = ConstantNCOCommand(100.0, 0.1)
    ratio = get_chip_rate_to_center_frequency_ratio(signal)
    for application_sample_index in (nothing, Int64(15000))
        commands = map((cis(0.0), cis(0.2), -cis(0.1))) do prompt
            _, carrier, code = step_satellite(tracking_loop, state, prompt_integration(signal, prompt, 5000, fs), nco_history, application_sample_index)
            carrier, code
        end
        @test allequal(commands)
        carrier, code = first(commands)
        middle = ((application_sample_index == nothing ? 5000 : 15000) + 2500) / 5e6
        @test carrier ≈ (110.0 + 4.0 * middle) * Hz
        @test code ≈ 0.1Hz - 0.25Hz + (carrier - 100.0Hz) * ratio
    end
    # The inner loop's filters are frozen.
    stepped, = step_satellite(tracking_loop, state, prompt_integration(signal, cis(0.2), 5000, fs), nco_history, nothing)
    @test stepped.scalar_state.carrier_loop_filter == state.scalar_state.carrier_loop_filter
    @test stepped.scalar_state.code_loop_filter == state.scalar_state.code_loop_filter
    @test get_carrier_loop_stage(stepped) == VECTOR_LOOP
    @test get_carrier_loop_stage(stepped.scalar_state) == get_carrier_loop_stage(state.scalar_state)
end

@testset "A vector loop needs an FLL-assisted inner loop" begin
    @test_throws ArgumentError VectorTrackingLoop(GPSL1CA(); scalar_loop = ScalarTrackingLoop())
    @test_throws ArgumentError VectorTrackingLoop(GPSL1CA(); scalar_loop = ScalarTrackingLoop(ThirdOrderBilinearLF))
    @test_throws ArgumentError VectorTrackingLoop(GPSL1CA(); scalar_loop = VectorTrackingLoop(GPSL1CA()))
    @test VectorTrackingLoop(GPSL1CA(); scalar_loop = ScalarTrackingLoop(ThirdOrderAssistedBilinearLF)) isa VectorTrackingLoop
    @test VectorTrackingLoop(GPSL1CA(); scalar_loop = DelayCompensatedTrackingLoop()) isa VectorTrackingLoop
end

@testset "Out of the vector loop it is the inner loop, with $(nameof(typeof(inner)))" for inner in (
    ScalarTrackingLoop(ThirdOrderAssistedBilinearLF),
    DelayCompensatedTrackingLoop(),
)
    # Under an NCO delay too: the delayed simulation of `tracking_loops.jl`.
    for d in (0, 3)
        @test simulate_delayed_loop(VectorTrackingLoop(GPSL1CA(); scalar_loop = inner), d; steps = 300) ==
              simulate_delayed_loop(inner, d; steps = 300)
    end
    # And integration by integration on one fixed command: nothing is accumulated and the
    # commands are never applied.
    signal = GPSL1CA()
    fs = 5e6Hz
    vector = VectorTrackingLoop(GPSL1CA(); scalar_loop = inner)
    vector_state = TL._set_vector_commands(init_loop_state(vector, signal, 100.0Hz, 0.1Hz), 0.5Hz,
        TL.CarrierDopplerRamp(120.0, 0.0, 0.0), 0.0s, ConstantNCOCommand(100.0, 0.0), 0, 5e6)
    inner_state = init_loop_state(inner, signal, 100.0Hz, 0.1Hz)
    post_correlation_state = vector_integration(signal, (1000.0 + 10im, 2000.0 + 20im, 750.0 + 10im), 5000, fs; previous_prompt = cis(0.1))
    nco_history = ConstantNCOCommand(100.0, 0.1)
    new_vector_state, vector_carrier, vector_code = step_satellite(vector, vector_state, post_correlation_state, nco_history, nothing)
    new_inner_state, inner_carrier, inner_code = step_satellite(inner, inner_state, post_correlation_state, nco_history, nothing)
    @test vector_carrier == inner_carrier
    @test vector_code == inner_code
    @test new_vector_state.scalar_state == new_inner_state
    @test new_vector_state.code_disc_sum == (0, 0.0)
    @test new_vector_state.fll_disc_sum == (0, 0.0Hz)
    # The replica's last half integration is kept, for the span a later vector loop opens
    # with: half of the 1 ms integration on the 100 Hz command.
    @test new_vector_state.previous_replica_tail == (100.0 * 0.0005, 0.0005s)
    @test new_vector_state.code_frequency_correction == 0.5Hz
    @test vector_carrier != 120.0Hz
end

@testset "A member's integration: the DLL accumulated, the residual PLL stepped, with $(nameof(typeof(inner)))" for inner in (
    ScalarTrackingLoop(ThirdOrderAssistedBilinearLF),
    DelayCompensatedTrackingLoop(),
)
    sampling_frequency = 5e6Hz
    gpsl1 = GPSL1CA()
    carrier_doppler = 100.0Hz
    reference_code_doppler = carrier_doppler * get_chip_rate_to_center_frequency_ratio(gpsl1)
    num_samples = 5000
    correlator_outputs = (1000.0 + 10im, 2000.0 + 20im, 750.0 + 10im)
    tracking_loop = VectorTrackingLoop(GPSL1CA(); scalar_loop = inner)
    state = init_loop_state(tracking_loop, gpsl1, carrier_doppler, reference_code_doppler)
    state = commanded(state, TL.CarrierDopplerRamp(100.0, 0.0, 0.0); code_update = -0.25Hz)
    post_correlation_state = vector_integration(gpsl1, correlator_outputs, num_samples, sampling_frequency)
    nco_history = ConstantNCOCommand(ustrip(Hz, carrier_doppler), ustrip(Hz, reference_code_doppler))
    state, = step_satellite(tracking_loop, state, post_correlation_state, nco_history, nothing)
    normalized_correlator = set_correlator_outputs(
        get_default_correlator_bank(gpsl1),
        SVector(correlator_outputs...) ./ num_samples,
    )
    dll_discriminator = dll_disc(gpsl1, normalized_correlator, reference_code_doppler, sampling_frequency)
    @test state.scalar_state.code_loop_filter == SecondOrderBilinearLF()
    # The DLL discriminator is accumulated. Without a previous prompt there is no
    # frequency measurement, so no carrier span opens: not a span of zero rotation.
    @test state.code_disc_sum == (1, dll_discriminator)
    @test state.fll_disc_sum == (0, 0.0Hz)
    @test TL._disc_mean(state.code_disc_sum) == dll_discriminator
    @test TL._disc_mean(state.fll_disc_sum) === nothing
    @test !TL.has_disc_sums(state)
    @test TL._has_carrier_ramp(state)
    @test state.residual_pll.last_prompt_sample_index ≈ 0.0005 * ustrip(Hz, sampling_frequency)
    # The replica's second half integration is kept for the next span.
    @test state.previous_replica_tail == (ustrip(Hz, carrier_doppler) * 0.0005, 0.0005s)
    @test state.code_frequency_correction == -0.25Hz
    # The residual PLL reads the phase lock on the prompts it rotated, carrying on from the
    # scalar loop's averages; the staging stands still.
    @test state.residual_pll.lock.lock_detector.num_integrations == 1
    @test state.scalar_state.staging == init_loop_state(inner, gpsl1, carrier_doppler, reference_code_doppler).staging

    state = TL._reset_discriminator_accumulators(state)
    @test state.code_disc_sum == (0, 0.0)
    @test state.fll_disc_sum == (0, 0.0Hz)
    @test state.previous_replica_tail == (ustrip(Hz, carrier_doppler) * 0.0005, 0.0005s)
    @test TL._disc_mean(state.code_disc_sum) === nothing
    @test TL._disc_mean(state.fll_disc_sum) === nothing
    @test state.vector_controlled
    @test state.code_frequency_correction == -0.25Hz
end

@testset "The vector loop accumulates the FLL discriminator" begin
    signal = GPSL1CA()
    fs = 5e6Hz
    tracking_loop = VectorTrackingLoop(GPSL1CA())
    state = commanded(init_loop_state(tracking_loop, signal, 100.0Hz, 0.1Hz), TL.CarrierDopplerRamp(100.0, 0.0, 0.0))
    previous = cis(0.2)
    correlator_outputs = (1000.0 + 10im, 2000.0 + 400im, 750.0 + 10im)
    integrations = [vector_integration(signal, correlator_outputs, 5000, fs; previous_prompt = previous, sample_index = 5000k)
               for k = 1:2]
    fll = fll_disc(signal, integrations[1].last_filtered_correlator_bank, previous, 5000 / fs)
    @test fll != 0.0Hz
    state, = step_satellite(tracking_loop, state, integrations[1], ConstantNCOCommand(100.0, 0.1), nothing)
    state, = step_satellite(tracking_loop, state, integrations[2], ConstantNCOCommand(120.0, 0.1), nothing)
    @test state.fll_disc_sum == (2, fll + fll)
    @test TL._disc_mean(state.fll_disc_sum) == (fll + fll) / 2
    # The replica's last half integration, which the residual PLL propagates by.
    @test state.previous_replica_tail == (120.0 * 0.0005, 0.0005s)
    # An integration without a previous prompt (a changed carrier phase polarity) has no
    # frequency measurement, so it is left out rather than taken as 0 Hz.
    no_prompt = vector_integration(signal, correlator_outputs, 5000, fs; sample_index = 15000)
    skipped, = step_satellite(tracking_loop, state, no_prompt, ConstantNCOCommand(160.0, 0.1), nothing)
    @test skipped.fll_disc_sum == state.fll_disc_sum
    @test skipped.previous_replica_tail == (160.0 * 0.0005, 0.0005s)
end

@testset "The vector loop accumulates the FLL discriminator in Hz from a MHz sampling frequency" begin
    # The FLL reading took the sampling frequency's unit, and the `Hz`-typed
    # accumulator refused it as soon as the satellite was in the vector loop.
    signal = GPSL1CA()
    tracking_loop = VectorTrackingLoop(GPSL1CA())
    function accumulated(fs)
        state = commanded(init_loop_state(tracking_loop, signal, 100.0Hz, 0.1Hz), TL.CarrierDopplerRamp(100.0, 0.0, 0.0))
        post_correlation_state = vector_integration(signal, (1000.0 + 10im, 2000.0 + 400im, 750.0 + 10im), 5000, fs;
            previous_prompt = cis(0.2))
        state, = step_satellite(tracking_loop, state, post_correlation_state, ConstantNCOCommand(100.0, 0.1), nothing)
        state.fll_disc_sum
    end
    count, carrier_sum = accumulated(5.0u"MHz")
    @test count == 1
    @test carrier_sum != 0.0Hz
    @test carrier_sum ≈ last(accumulated(5e6Hz))
end

@testset "In the vector loop either inner loop runs the same" begin
    # The inner loop's discriminators and filters are not used in the vector loop: the
    # filter commands the carrier, the residual PLL measures it.
    signal = GPSL1CA()
    fs = 4e6Hz
    function run(inner, application_offset)
        tracking_loop = VectorTrackingLoop(GPSL1CA(); scalar_loop = inner)
        state = init_loop_state(tracking_loop, signal, 100.0Hz, 0.1Hz)
        state = commanded(state, TL.CarrierDopplerRamp(100.5, 1.0, 0.0); code_update = 0.1Hz, scalar_doppler = 100.0)
        timeline = NCOTimeline()
        reset_timeline!(timeline, 100.0, 0.1)
        previous = complex(0.0, 0.0)
        carriers = typeof(1.0Hz)[]
        for k = 1:50
            p = cis(0.3 + 0.01k)
            output = CorrelatorBankOutput(loop_epl(0.5p, p, 0.5p), 4000, 4000k)
            post_correlation_state = processed_state(signal, output.correlator_bank, previous, output, 1, fs)
            application_sample_index = application_offset == 0 ? nothing : Int64(4000k + application_offset)
            state, carrier, code = step_satellite(tracking_loop, state, post_correlation_state, timeline, application_sample_index)
            schedule_command!(timeline, 4000k + application_offset, ustrip(Hz, carrier), ustrip(Hz, code))
            advance_timeline!(timeline, 4000k - 4000)
            push!(carriers, carrier)
            previous = p
        end
        carriers, state
    end
    for application_offset in (0, 12000)
        referenced, referenced_state = run(DelayCompensatedTrackingLoop(), application_offset)
        conventional, conventional_state = run(ScalarTrackingLoop(ThirdOrderAssistedBilinearLF), application_offset)
        @test referenced == conventional
        @test referenced_state.code_disc_sum == conventional_state.code_disc_sum
        @test referenced_state.fll_disc_sum == conventional_state.fll_disc_sum
        @test referenced_state.residual_pll == conventional_state.residual_pll
        # The first integration has no previous prompt.
        @test first(conventional_state.fll_disc_sum) == 49
    end
end

@testset "Vector tracking state management" begin
    tracking_loop = VectorTrackingLoop(GPSL1CA())
    state = init_loop_state(tracking_loop, GPSL1CA(), 100.0Hz, 0.1Hz)
    joined = TL._add_to_vector_loop(state)
    @test joined.vector_controlled
    ramp = TL.CarrierDopplerRamp(110.0, 2.0, 0.01)
    # A member joining again changes nothing, so the members can join every cycle.
    steered = VectorLoopState(
        TL._set_vector_commands(joined, 1.0Hz, ramp, 0.02s, ConstantNCOCommand(100.0, 0.0), 0, 1e6);
        code_disc_sum = (3, 0.3),
        fll_disc_sum = (3, 0.3Hz),
        previous_replica_tail = (0.1, 0.0005s),
    )
    @test TL._add_to_vector_loop(steered) === steered
    # The first ramp starts the residual PLL from the scalar loop's command, against the ramp.
    @test steered.ramp == ramp
    @test TL._has_carrier_ramp(steered)
    @test steered.residual_pll.carrier_frequency_correction ≈ 100.0 - 110.0
    @test TL._residual_doppler(steered.residual_pll) ≈ 100.0 - 110.0
    @test steered.code_frequency_correction == 1.0Hz
    @test steered.code_frequency_correction_history == (1.0Hz, 0.0Hz, 0.0Hz)
    @test steered.code_command_delay == 0.02s
    # A later one keeps the residual PLL's estimates of the incoming Doppler and its rate.
    next_ramp = TL.CarrierDopplerRamp(105.0, 0.0, 0.11)
    next = TL._set_vector_commands(steered, 2.0Hz, next_ramp, 0.0s, ConstantNCOCommand(999.0, 0.0), 0, 1e6)
    @test TL._doppler_at(next_ramp, 0.01) + next.residual_pll.carrier_frequency_correction ≈ TL._doppler_at(ramp, 0.01) + steered.residual_pll.carrier_frequency_correction
    @test TL._doppler_at(next_ramp, 0.01) + TL._residual_doppler(next.residual_pll) ≈
          TL._doppler_at(ramp, 0.01) + TL._residual_doppler(steered.residual_pll)
    @test next_ramp.doppler_rate + TL._residual_doppler_rate(next.residual_pll) ≈
          ramp.doppler_rate + TL._residual_doppler_rate(steered.residual_pll)
    @test next.code_frequency_correction_history == (2.0Hz, 1.0Hz, 0.0Hz)
    @test TL._set_vector_commands(next, 3.0Hz, next_ramp, 0.0s, ConstantNCOCommand(0.0, 0.0), 0, 1e6).code_frequency_correction_history ==
          (3.0Hz, 2.0Hz, 1.0Hz)
    @test next.code_frequency_correction == 2.0Hz
    @test next.code_command_delay == 0.0s
    @test next.code_disc_sum == (3, 0.3)
    # Leaving zeroes everything the vector loop owned, so a satellite that joins again
    # never runs on stale values.
    released = TL._release_from_vector_loop(next)
    @test !released.vector_controlled
    @test released.scalar_state == next.scalar_state
    @test released.code_disc_sum == (0, 0.0)
    @test released.fll_disc_sum == (0, 0.0Hz)
    @test released.code_frequency_correction == 0.0Hz
    @test released.code_frequency_correction_history == (0.0Hz, 0.0Hz, 0.0Hz)
    @test released.code_command_delay == 0.0s
    @test isnan(released.ramp.reference_time)
    @test !TL._has_carrier_ramp(released)
    # The replica's last half integration is the replica's: it survives the hand-over both ways.
    @test released.previous_replica_tail == (0.1, 0.0005s)
    @test TL._add_to_vector_loop(released).previous_replica_tail == (0.1, 0.0005s)
    @test TL._add_to_vector_loop(released).code_frequency_correction == 0.0Hz
end

@testset "Resetting keeps the vector flag and stops applying the code correction" begin
    tracking_loop = VectorTrackingLoop(GPSL1CA(); scalar_loop = ScalarTrackingLoop(ThirdOrderAssistedBilinearLF; narrow_carrier_loop_filter_bandwidth = 12.0Hz))
    state = init_loop_state(tracking_loop, GPSL1CA(), 100.0Hz, 0.1Hz)
    post_correlation_state = vector_integration(GPSL1CA(), (1000.0 + 10im, 2000.0 + 200im, 750.0 + 10im), 5000, 5e6Hz; previous_prompt = cis(0.1))
    ramp = TL.CarrierDopplerRamp(104.0, 0.0, 0.0)
    state = commanded(state, ramp; code_update = 0.5Hz)
    state, = step_satellite(tracking_loop, state, post_correlation_state, ConstantNCOCommand(100.0, 0.1), nothing)
    state = TL._set_vector_commands(state, 0.6Hz, ramp, 0.01s, ConstantNCOCommand(0.0, 0.0), 0, 5e6)
    reset = reinit_loop_state(tracking_loop, state, 150.0Hz, 0.2Hz)
    @test reset isa typeof(state)
    @test reset.vector_controlled
    @test reset.code_disc_sum == (0, 0.0)
    @test reset.fll_disc_sum == (0, 0.0Hz)
    @test reset.previous_replica_tail == state.previous_replica_tail
    # The re-seeded Dopplers carry the last code correction, so it is no longer applied…
    @test reset.code_frequency_correction == 0.0Hz
    # …but the replica is still steered by it, so the next code measurement is moved to
    # the epoch exactly as without the reset.
    @test reset.code_frequency_correction_history == (0.6Hz, 0.5Hz, 0.0Hz)
    @test reset.code_command_delay == 0.01s
    @test TrackingLoops.code_phase_advance(reset, 0.1) == TrackingLoops.code_phase_advance(state, 0.1)
    @test TrackingLoops.code_phase_advance(reset, 0.1) != 0.0
    # The carrier command is the filter's Doppler itself, and the residual PLL carries on.
    @test reset.ramp == state.ramp
    @test reset.residual_pll == state.residual_pll
    @test reset.scalar_state.carrier_loop_filter == ThirdOrderAssistedBilinearLF()
    @test reset.scalar_state.bandwidths.narrow_carrier == 12.0Hz
    @test get_carrier_loop_stage(reset) == VECTOR_LOOP
    @test reset.scalar_state.reference_carrier_doppler == 150.0Hz
    @test reset.scalar_state.reference_code_doppler == 0.2Hz
    # With no code correction, the code Doppler is the re-seeded one plus the carrier aiding.
    _, carrier_doppler, code_doppler =
        step_satellite(tracking_loop, reset, post_correlation_state, ConstantNCOCommand(150.0, 0.2), nothing)
    @test carrier_doppler == 104.0Hz
    @test code_doppler ≈ 0.2Hz + (carrier_doppler - 150.0Hz) * get_chip_rate_to_center_frequency_ratio(GPSL1CA())
end

@testset "Releasing re-seeds the inner loop, with $(nameof(typeof(inner)))" for inner in (
    ScalarTrackingLoop(ThirdOrderAssistedBilinearLF; narrow_carrier_loop_filter_bandwidth = 12.0Hz),
    DelayCompensatedTrackingLoop(; narrow_carrier_loop_filter_bandwidth = 12.0Hz),
)
    tracking_loop = VectorTrackingLoop(GPSL1CA(); scalar_loop = inner)
    state = init_loop_state(tracking_loop, GPSL1CA(), 100.0Hz, 0.1Hz)
    post_correlation_state = vector_integration(GPSL1CA(), (1000.0 + 10im, 2000.0 + 200im, 750.0 + 10im), 5000, 5e6Hz; previous_prompt = cis(0.1))
    ramp = TL.CarrierDopplerRamp(104.0, 2.0, 0.0)
    state = commanded(state, ramp; code_update = 0.5Hz, lead = 0.01s)
    state, = step_satellite(tracking_loop, state, post_correlation_state, ConstantNCOCommand(100.0, 0.1), nothing)
    # A residual PLL out of lock knows no better than the filter: the inner loop starts over
    # from the handed-over replica and the ramp's Doppler rate.
    unlocked = TL.VectorLoopState(state; residual_pll = TL.ResidualPLL(state.residual_pll; lock = TL.ResidualPLLLock()))
    released = TL._hand_over_to_scalar_loop(unlocked, ConstantNCOCommand(150.0, 0.2), 10_000, 5e6)
    @test released isa typeof(state)
    @test !released.vector_controlled
    # The inner loop starts over with its own configuration kept: the per-satellite
    # state carries it. Unlike a reset, the release restarts the staging.
    reset_inner = reinit_loop_state(inner, state.scalar_state, 150.0Hz, 0.2Hz)
    @test released.scalar_state == Base.typename(typeof(reset_inner)).wrapper(
        reset_inner;
        carrier_loop_filter = TL._carrier_filter_at_rate(reset_inner.carrier_loop_filter, 2.0),
        staging = TL.CarrierLoopStaging(typeof(reset_inner.carrier_loop_filter)),
    )
    @test released.scalar_state.carrier_loop_filter.x2 == 2.0Hz^2
    @test released.scalar_state.reference_carrier_doppler == 150.0Hz
    @test released.scalar_state.reference_code_doppler == 0.2Hz
    # A residual PLL in lock hands over its estimate where the release takes effect: the ramp plus
    # its residual Doppler, moved on by its rate, and the Doppler rate of both.
    residual_pll = state.residual_pll
    residual_pll = TL.ResidualPLL(residual_pll.bandwidth, 1.5, TL._residual_pll_loop_filter(1.5, 0.5), residual_pll.last_prompt_sample_index,
        residual_pll.carrier_phase_polarity, residual_pll.predicted_prompt_phase,
        TL.ResidualPLLLock(TL.PhaseLockDetector(0.0, 0.0, 0.0, 0, 1.0), true, 0))
    locked = TL._hand_over_to_scalar_loop(TL.VectorLoopState(state; residual_pll), ConstantNCOCommand(150.0, 0.2), 10_000, 5e6)
    @test locked.scalar_state.reference_carrier_doppler ≈
          (TL._doppler_at(ramp, 0.002) + 1.5 + 0.5 * (0.002 - residual_pll.last_prompt_sample_index / 5e6)) * Hz
    @test locked.scalar_state.carrier_loop_filter.x2 ≈ 2.5Hz^2
    @test locked.scalar_state.reference_code_doppler == 0.2Hz
    @test released.scalar_state.bandwidths.narrow_carrier == 12.0Hz
    # The carrier loop pulls in afresh, FLL-assisted.
    @test get_carrier_loop_stage(state) == VECTOR_LOOP
    @test get_carrier_loop_stage(released) == FLL_ASSISTED_PLL
    # Nothing the vector loop owned survives.
    @test released.code_disc_sum == (0, 0.0)
    @test released.fll_disc_sum == (0, 0.0Hz)
    @test released.previous_replica_tail == state.previous_replica_tail
    @test released.code_frequency_correction == 0.0Hz
    @test released.code_frequency_correction_history == (0.0Hz, 0.0Hz, 0.0Hz)
    @test !TL._has_carrier_ramp(released)
    # Out of the vector loop it steps exactly as the re-seeded inner loop.
    @test step_satellite(tracking_loop, released, post_correlation_state, ConstantNCOCommand(150.0, 0.2), nothing)[2:3] ==
          step_satellite(inner, released.scalar_state, post_correlation_state, ConstantNCOCommand(150.0, 0.2), nothing)[2:3]
end

# ─────────────────────────────────────────────────────────────────────────────
# The pseudorange-rate observable of a satellite in the vector loop, on a carrier of
# known phase: the replica commanded a ramp `offset` Hz off the carrier's Doppler, the
# residual PLL following the rest integration by integration, and the accumulators read and
# emptied at every navigation epoch, as the navigation processor's snapshot does — before the integration
# that crosses the epoch, with the command that integration runs on as the replica's Doppler.

using Random: Xoshiro, randn
using Statistics: mean, std

const RATE_SIGNAL = GPSL1CA()
const RATE_FS = 4e6
const RATE_WAVELENGTH = TL.SPEED_OF_LIGHT / ustrip(Hz, get_center_frequency(RATE_SIGNAL))

# Track the carrier of phase `phase(t)` (cycles) with integrations of `integration_samples`,
# the first ending at `first_end` (samples), up to `duration` seconds, the epochs every
# `cycle` seconds. Each prompt is the mean of the carrier against the replica over its
# integration, with complex noise of `noise_std` per component. Returns per epoch `E` the
# rate observable (Hz: the replica's command there plus the mean FLL) and the integrations it
# averaged.
function track_carrier(phase, doppler0, rate; offset = 0.5, integration_samples = 16000, first_end = 16000,
        cycle = 0.1, duration = 2.0, noise_std = 0.0, rng = Xoshiro(1))
    tracking_loop = VectorTrackingLoop(RATE_SIGNAL)
    state = init_loop_state(tracking_loop, RATE_SIGNAL, doppler0 * Hz, 0.0Hz)
    state = commanded(state, TL.CarrierDopplerRamp(doppler0 + offset, rate, 0.0); scalar_doppler = doppler0)
    carrier = doppler0
    replica_phase = phase(0.0) # cycles, at the start of the integration ahead
    previous_prompt = complex(0.0)
    epochs = NamedTuple[]
    epoch = 1
    integration_end_sample_index = first_end
    integration_start_sample_index = 0
    while integration_end_sample_index / RATE_FS <= duration
        t0, t1 = integration_start_sample_index / RATE_FS, integration_end_sample_index / RATE_FS
        # The navigation processor's snapshot: the accumulators up to the epoch, emptied before the
        # integration that crosses it, whose command is the replica's there.
        if t1 > epoch * cycle
            push!(epochs, (; epoch = epoch * cycle,
                doppler = carrier + TL.mean_fll_disc(state),
                count = first(state.fll_disc_sum)))
            state = TL._reset_discriminator_accumulators(state)
            epoch += 1
        end
        n = 64
        prompt = sum(cis(2π * (phase(t0 + (i - 0.5) / n * (t1 - t0)) -
                                replica_phase - carrier * (i - 0.5) / n * (t1 - t0))) for i = 1:n) / n
        prompt += noise_std * complex(randn(rng), randn(rng))
        output = CorrelatorBankOutput(loop_epl(0.5prompt, prompt, 0.5prompt), integration_end_sample_index - integration_start_sample_index, integration_end_sample_index)
        post_correlation_state = processed_state(RATE_SIGNAL, output.correlator_bank, previous_prompt, output, 1, RATE_FS * Hz)
        state, new_carrier, = TL._step_satellite(tracking_loop, state, post_correlation_state,
            ConstantNCOCommand(carrier, 0.0), nothing)
        replica_phase += carrier * (t1 - t0)
        carrier = ustrip(Hz, new_carrier)
        previous_prompt = prompt
        integration_start_sample_index = integration_end_sample_index
        integration_end_sample_index += integration_samples
    end
    epochs
end

@testset "The rate is the Doppler at the epoch" begin
    # A line-of-sight acceleration of 5 m/s², on 4 ms integrations ending 1.3 ms past the
    # epochs, the replica commanded 0.5 Hz off on a ramp of the carrier's slope. The
    # mean FLL is the cycle's mean offset from the replica, so with the replica's command
    # at the epoch it is the incoming Doppler there, to within the half integration that
    # command stands for — not the cycle's mean Doppler, half a cycle earlier.
    acceleration = 5.0
    ramp = acceleration / RATE_WAVELENGTH # Hz/s
    doppler(t) = 1200.0 + ramp * t
    phase(t) = 0.3 + 1200.0 * t + ramp * t^2 / 2
    epochs = track_carrier(phase, 1200.0, ramp; first_end = 16000 + 5200, duration = 3.0)
    for e in epochs[11:end]
        @test RATE_WAVELENGTH * abs(e.doppler - doppler(e.epoch)) < acceleration * 0.002 + 0.001
        @test RATE_WAVELENGTH * abs(e.doppler - doppler(e.epoch - 0.05)) > 0.2
    end
end

@testset "The rate measurements are continuous across cycles" begin
    # Noisy prompts at 45 dB-Hz, 4.1 ms integrations, so the 100 ms cycles take 24 or 25 of
    # them, under a slow Doppler ramp. The FLL compares consecutive prompts and chains
    # across the cycles, so the integrated rate error stays at the noise of the
    # boundary prompts — no walk, however many cycles — each cycle's error has the
    # variance the filter weights the rate with, and consecutive cycles, sharing a
    # boundary prompt, a lag-1 autocorrelation of −1/2.
    cn0_lin = 10^4.5
    integration_samples = 16400
    T_coh = integration_samples / RATE_FS
    ramp = 0.5 / RATE_WAVELENGTH
    doppler(t) = 800.0 + ramp * t
    phase(t) = 0.1 + 800.0 * t + ramp * t^2 / 2
    epochs = track_carrier(phase, 800.0, ramp; integration_samples, first_end = integration_samples, duration = 60.0,
        noise_std = sqrt(1 / (2 * cn0_lin * T_coh)), rng = Xoshiro(7))
    measured = epochs[11:end]
    @test all(e -> e.count in (24, 25), measured)
    @test any(e -> e.count == 24, measured) && any(e -> e.count == 25, measured)
    residuals = [RATE_WAVELENGTH * (e.doppler - doppler(e.epoch)) for e in measured]
    integrated_error = cumsum(residuals) .* 0.1
    σ_φ² = 1 / (2 * cn0_lin * T_coh) * (1 + 1 / (2 * cn0_lin * T_coh))
    σ_integrated = RATE_WAVELENGTH * sqrt(2σ_φ²) / 2π
    @test maximum(abs, integrated_error) < 5σ_integrated
    @test std(integrated_error[1:end÷2]) ≈ std(integrated_error[end÷2:end]) rtol = 0.3
    @test std(residuals) / (RATE_WAVELENGTH * sqrt(2σ_φ²) / (2π * 0.1)) ≈ 1.0 atol = 0.15
    r = residuals .- mean(residuals)
    @test sum(r[1:end-1] .* r[2:end]) / sum(abs2, r) ≈ -0.5 atol = 0.1
end
