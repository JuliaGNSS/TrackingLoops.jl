# The loop model of GNSSReceiver's delay-aware tests: a carrier whose phase
# error advances by `2π (f_true − w) Δt` per integration under the replica command `w`,
# the command the tracking_loop issues, applied `d` integrations after the integration that
# produced it.
const LOOP_SIGNAL = GPSL1CA()
const LOOP_FS = 4e6Hz
const LOOP_N = 4000

# The delay a loop tolerates shrinks as its bandwidth grows: at 18 Hz the
# conventional loop holds through about 20 integrations of delay, at its default 50 Hz
# wide bandwidth through three. The delay tests run the loops at 85 Hz, wide and
# narrow, where 5 integrations make it limit-cycle. Once the staging drops the FLL the
# loop tolerates an integration less than with it: the NCO-referenced loop holds
# through 5 integrations there, not 6.
const WIDE_LOOP = (;
    wide_carrier_loop_filter_bandwidth = 85.0Hz,
    narrow_carrier_loop_filter_bandwidth = 85.0Hz,
)

# The NCO-referenced loop pulls in at the narrow bandwidth by default; the
# conventional loop configured alike is the loop it extends.
const NCO_REFERENCED_WIDE = (; wide_carrier_loop_filter_bandwidth = 18.0Hz)

loop_epl(late, prompt, early) =
    EarlyPromptLateCorrelatorBank(SVector{3,ComplexF64}(late, prompt, early), 0.5)

function simulate_delayed_loop(tracking_loop, d; f_true = 130.0, handover = 100.0, phi0 = 0.8, steps = 600)
    code_doppler = handover * Hz * get_chip_rate_to_center_frequency_ratio(LOOP_SIGNAL)
    state = init_loop_state(tracking_loop, LOOP_SIGNAL, handover * Hz, code_doppler)
    timeline = NCOTimeline()
    reset_timeline!(timeline, handover, ustrip(Hz, code_doppler))
    dt = LOOP_N / ustrip(Hz, LOOP_FS)
    phase = phi0
    phases = Float64[]
    nco_history = Float64[]
    previous_prompt = complex(0.0, 0.0)
    for k = 0:steps-1
        a = k * LOOP_N
        b = a + LOOP_N
        w = get_nco_command(timeline, a).carrier_doppler
        mean_phase = phase + π * (f_true - w) * dt
        phase += 2π * (f_true - w) * dt
        # In radians, folded as the Costas discriminator reads it.
        push!(phases, 2π * TrackingLoops.wrap_phase_error(mean_phase / 2π))
        push!(nco_history, w)
        p = cis(mean_phase)
        output = CorrelatorBankOutput(loop_epl(0.5p, p, 0.5p), LOOP_N, b)
        application_sample_index = Int64(b + d * LOOP_N)
        integration = CoherentIntegration(LOOP_SIGNAL, output.correlator_bank, previous_prompt, output, 1, LOOP_FS; prn = 1)
        state, carrier, code = step_loop!(tracking_loop, state, integration, timeline, application_sample_index)
        previous_prompt = p
        schedule_command!(timeline, application_sample_index, ustrip(Hz, carrier), ustrip(Hz, code))
        advance_timeline!(timeline, b - LOOP_N)
    end
    phases, nco_history
end

@testset "With no delay the NCO-referenced loop is the conventional loop" begin
    conventional =
        simulate_delayed_loop(ConventionalAssistedPLLAndDLL(; NCO_REFERENCED_WIDE...), 0; steps = 1200)
    referenced = simulate_delayed_loop(DelayCompensatedTrackingLoop(), 0; steps = 1200)
    @test referenced[1] == conventional[1]
    @test referenced[2] == conventional[2]
    control = simulate_delayed_loop(DelayCompensatedTrackingLoop(; compensate_delay = false), 0; steps = 1200)
    @test control[2] == conventional[2]
    @test all(abs.(referenced[1][900:end]) .< 0.2)
    @test all(abs.(referenced[2][900:end] .- 130.0) .< 1.0)
end

@testset "The NCO-referenced loop holds lock through $d integrations of delay" for d in 1:5
    phases, nco_history = simulate_delayed_loop(DelayCompensatedTrackingLoop(; WIDE_LOOP...), d; steps = 2000)
    @test all(abs.(phases[1800:end]) .< 0.05)
    @test all(abs.(nco_history[1800:end] .- 130.0) .< 0.3)
    # The pull-in transient stays clear of the ±π/2 edge where the BPSK
    # discriminator slips.
    @test maximum(abs.(phases[100:end])) < 1.4
end

@testset "The default loop holds lock through three integrations of delay" begin
    phases, nco_history = simulate_delayed_loop(ConventionalAssistedPLLAndDLL(), 3; steps = 1200)
    @test all(abs.(phases[1100:end]) .< 0.05)
    @test all(abs.(nco_history[1100:end] .- 130.0) .< 0.3)
end

@testset "The conventional loop and the negative control limit-cycle at five integrations of delay" begin
    for tracking_loop in (
        ConventionalAssistedPLLAndDLL(; WIDE_LOOP...),
        DelayCompensatedTrackingLoop(; compensate_delay = false, WIDE_LOOP...),
    )
        phases, nco_history = simulate_delayed_loop(tracking_loop, 5; steps = 1200)
        tail = nco_history[600:end] .- 130.0
        @test sqrt(sum(abs2, tail) / length(tail)) > 20
        @test maximum(abs.(phases[600:end])) > 1.0
    end
end

@testset "Estimator state construction and reset" begin
    tracking_loop = DelayCompensatedTrackingLoop()
    state = init_loop_state(tracking_loop, GPSL1CA(), 1234.0Hz, 0.8Hz)
    @test state isa SatNCOReferencedPLLAndDLL
    @test state.reference_carrier_doppler == 1234.0Hz
    # The NCO-referenced loop pulls in at the narrow bandwidth by default.
    @test state.bandwidths.wide_carrier == 18.0Hz
    @test state.bandwidths.narrow_carrier == 18.0Hz
    @test state.bandwidths.fll_assist == 5.0Hz
    @test get_carrier_loop_stage(state) == FLL_ASSISTED_PLL
    @test isnan(state.previous_record_center)
    narrow = DelayCompensatedTrackingLoop(;
        wide_carrier_loop_filter_bandwidth = 30.0Hz,
        narrow_carrier_loop_filter_bandwidth = 12.0Hz,
        fll_assist_loop_filter_bandwidth = 3.0Hz,
    )
    narrow_state = init_loop_state(narrow, GPSL1CA(), 0.0Hz, 0.0Hz)
    @test narrow_state.bandwidths.wide_carrier == 30.0Hz
    @test narrow_state.bandwidths.narrow_carrier == 12.0Hz
    @test narrow_state.bandwidths.fll_assist == 3.0Hz
    reset_state = reinit_loop_state(narrow, narrow_state, 50.0Hz, 0.03Hz)
    @test reset_state.reference_carrier_doppler == 50.0Hz
    @test reset_state.bandwidths.narrow_carrier == 12.0Hz
    @test reset_state.carrier_loop_filter.x1 == 0.0Hz
    conv = init_loop_state(ConventionalAssistedPLLAndDLL(), GalileoE1B(), 0.0Hz, 0.0Hz)
    @test conv isa ScalarLoopState
    @test conv.bandwidths.wide_carrier ==
          default_wide_carrier_loop_filter_bandwidth(GalileoE1B()) == 50.0Hz
    @test conv.bandwidths.narrow_carrier ==
          default_narrow_carrier_loop_filter_bandwidth(GalileoE1B()) == 18.0Hz
    @test conv.bandwidths.fll_assist ==
          default_fll_assist_loop_filter_bandwidth(GalileoE1B()) == 5.0Hz
    @test get_carrier_loop_stage(conv) == FLL_ASSISTED_PLL
    # A carrier filter without an FLL path starts at the wide PLL.
    @test get_carrier_loop_stage(init_loop_state(ScalarTrackingLoop(), GPSL1CA(), 0.0Hz, 0.0Hz)) ==
          WIDE_PLL
    @test loop_state_type(ConventionalAssistedPLLAndDLL(), GPSL1CA()) === typeof(conv)
    # The state integrations its driver by key, so it is of one type whatever the driver.
    @test conv.driver == TrackingLoops._signal_key(GalileoE1B()) != TrackingLoops._signal_key(GPSL1CA())
    @test ScalarLoopState(; reference_carrier_doppler = 0.0Hz, reference_code_doppler = 0.0Hz).driver == 0
end

@testset "Without a application_sample_index sample the NCO-referenced loop is the conventional loop" begin
    # The software receiver's path: one fixed command per chunk and no application sample
    # sample, so the prediction branch is never entered.
    signal = LOOP_SIGNAL
    conventional = ConventionalAssistedPLLAndDLL(; NCO_REFERENCED_WIDE...)
    referenced = DelayCompensatedTrackingLoop()
    conv_state = init_loop_state(conventional, signal, 100.0Hz, 0.1Hz)
    ref_state = init_loop_state(referenced, signal, 100.0Hz, 0.1Hz)
    carrier, code = 100.0, 0.1
    previous_prompt = complex(0.0, 0.0)
    for k = 1:300
        p = cis(0.3 + 0.002k)
        output = CorrelatorBankOutput(loop_epl(0.45p, p, 0.55p), LOOP_N, LOOP_N * k)
        integration = CoherentIntegration(signal, output.correlator_bank, previous_prompt, output, 1, LOOP_FS)
        nco_history = ConstantNCOCommand(carrier, code)
        conv_state, conv_carrier, conv_code = step_loop!(conventional, conv_state, integration, nco_history, NO_LANDING_SAMPLE)
        ref_state, ref_carrier, ref_code = step_loop!(referenced, ref_state, integration, nco_history, NO_LANDING_SAMPLE)
        @test ref_carrier == conv_carrier
        @test ref_code == conv_code
        carrier, code = ustrip(Hz, conv_carrier), ustrip(Hz, conv_code)
        previous_prompt = p
    end
end

@testset "The negative control is the conventional loop at any delay" for d in (2, 4)
    conventional =
        simulate_delayed_loop(ConventionalAssistedPLLAndDLL(; NCO_REFERENCED_WIDE...), d; steps = 300)
    control = simulate_delayed_loop(DelayCompensatedTrackingLoop(; compensate_delay = false), d; steps = 300)
    @test control == conventional
end

@testset "Resetting the conventional loop keeps its bandwidths" begin
    tracking_loop = ConventionalAssistedPLLAndDLL(; code_loop_filter_bandwidth = 0.5Hz)
    state = init_loop_state(tracking_loop, LOOP_SIGNAL, 0.0Hz, 0.0Hz)
    output = CorrelatorBankOutput(loop_epl(0.4, cis(0.2), 0.6), LOOP_N, LOOP_N)
    integration = CoherentIntegration(LOOP_SIGNAL, output.correlator_bank, cis(0.1), output, 1, LOOP_FS)
    state, = step_loop!(tracking_loop, state, integration, ConstantNCOCommand(0.0, 0.0), NO_LANDING_SAMPLE)
    @test state.carrier_loop_filter.x1 != 0.0Hz
    reset = reinit_loop_state(tracking_loop, state, 42.0Hz, 0.02Hz)
    @test reset isa typeof(state)
    @test reset.reference_carrier_doppler == 42.0Hz
    @test reset.reference_code_doppler == 0.02Hz
    @test reset.carrier_loop_filter == ThirdOrderAssistedBilinearLF()
    @test reset.code_loop_filter == SecondOrderBilinearLF()
    @test reset.bandwidths.wide_carrier == state.bandwidths.wide_carrier
    @test reset.bandwidths.narrow_carrier == state.bandwidths.narrow_carrier
    @test reset.bandwidths.fll_assist == state.bandwidths.fll_assist
    @test reset.bandwidths.code == 0.5Hz
end

@testset "$(nameof(typeof(tracking_loop))) is driven through the tracking_loop interface alone" for tracking_loop in (
    ConventionalAssistedPLLAndDLL(),
    DelayCompensatedTrackingLoop(),
    VectorTrackingLoop(LOOP_SIGNAL),
    VectorTrackingLoop(LOOP_SIGNAL; inner = DelayCompensatedTrackingLoop()),
)
    # What a host calls, and nothing else: build, step an integration, reset.
    state = init_loop_state(tracking_loop, LOOP_SIGNAL, 100.0Hz, 0.0Hz)
    @test state isa loop_state_type(tracking_loop, LOOP_SIGNAL)
    output = CorrelatorBankOutput(loop_epl(0.5, 1.0, 0.5), LOOP_N, LOOP_N)
    # The integration names its satellite, which the vector loop's navigation engine needs.
    integration = CoherentIntegration(LOOP_SIGNAL, output.correlator_bank, complex(0.0, 0.0), output, 1, LOOP_FS; prn = 5)
    stepped, carrier_doppler, code_doppler =
        @inferred step_loop!(tracking_loop, state, integration, ConstantNCOCommand(100.0, 0.0), NO_LANDING_SAMPLE)
    @test stepped isa typeof(state)
    @test carrier_doppler isa typeof(1.0Hz)
    @test code_doppler isa typeof(1.0Hz)
    @test @inferred(reinit_loop_state(tracking_loop, stepped, carrier_doppler, code_doppler)) isa
          typeof(state)
    # What it knows of the navigation solution: nothing, for a scalar loop.
    if tracking_loop isa VectorTrackingLoop
        @test get_navigation_cycle_index(tracking_loop) isa Int
        @test get_navigation_solution(tracking_loop) !== nothing
        @test get_satellite_report(tracking_loop, LOOP_SIGNAL, 5) isa SatelliteReport
    else
        @test get_navigation_solution(tracking_loop) === nothing
        @test get_navigation_status(tracking_loop) === nothing
        @test get_navigation_cycle_index(tracking_loop) === nothing
        @test get_navigation_elapsed_time(tracking_loop) === nothing
        @test get_satellite_report(tracking_loop, LOOP_SIGNAL, 5) === nothing
    end
end

@testset "$(nameof(typeof(tracking_loop))) leaves a passenger's integration to the host" for tracking_loop in (
    ConventionalAssistedPLLAndDLL(),
    DelayCompensatedTrackingLoop(),
)
    # GPS L5: the pilot L5Q drives, the data component L5I rides along.
    driver, passenger = GPSL5Q(), GPSL5I()
    fs = 25e6Hz
    n = 25_000
    state = init_loop_state(tracking_loop, driver, 100.0Hz, 0.1Hz)
    output = CorrelatorBankOutput(loop_epl(0.5, 1.0, 0.5), n, n)
    integration = CoherentIntegration(passenger, output.correlator_bank, complex(0.0), output, 1, fs; prn = 5)
    # The state unchanged, and the command in force…
    stepped, carrier, code = step_loop!(tracking_loop, state, integration, ConstantNCOCommand(123.0, 0.12), NO_LANDING_SAMPLE)
    @test stepped === state
    @test (carrier, code) == (123.0Hz, 0.12Hz)
    # …at the application sample.
    timeline = NCOTimeline()
    reset_timeline!(timeline, 100.0, 0.1)
    schedule_command!(timeline, 2n, 130.0, 0.13)
    @test step_loop!(tracking_loop, state, integration, timeline, Int64(3n))[2:3] == (130.0Hz, 0.13Hz)
    @test step_loop!(tracking_loop, state, integration, timeline, Int64(n))[2:3] == (100.0Hz, 0.1Hz)
    # Passenger integrations interleaved with the driver's leave its Doppler sequence as it
    # is, to the bit.
    function dopplers(with_passengers)
        state = init_loop_state(tracking_loop, driver, 100.0Hz, 0.1Hz)
        carrier, code = 100.0, 0.1
        previous_prompt = complex(0.0, 0.0)
        sequence = Tuple{typeof(1.0Hz),typeof(1.0Hz)}[]
        for k = 1:200
            p = cis(0.3 + 0.002k)
            output = CorrelatorBankOutput(loop_epl(0.45p, p, 0.55p), n, n * k)
            if with_passengers
                passenger_integration = CoherentIntegration(passenger, output.correlator_bank, complex(0.0), output, 1, fs)
                state, = step_loop!(tracking_loop, state, passenger_integration, ConstantNCOCommand(carrier, code), NO_LANDING_SAMPLE)
            end
            integration = CoherentIntegration(driver, output.correlator_bank, previous_prompt, output, 1, fs)
            state, c, d = step_loop!(tracking_loop, state, integration, ConstantNCOCommand(carrier, code), NO_LANDING_SAMPLE)
            push!(sequence, (c, d))
            carrier, code = ustrip(Hz, c), ustrip(Hz, d)
            previous_prompt = p
        end
        sequence, state
    end
    plain, plain_state = dopplers(false)
    interleaved, interleaved_state = dopplers(true)
    @test plain == interleaved
    @test plain_state == interleaved_state
    @test get_navigation_solution(tracking_loop) === nothing
end

@testset "$(nameof(typeof(tracking_loop))) reads an integration of known sign four-quadrant" for tracking_loop in (
    ConventionalAssistedPLLAndDLL(),
    DelayCompensatedTrackingLoop(),
)
    state = init_loop_state(tracking_loop, LOOP_SIGNAL, 0.0Hz, 0.0Hz)
    # The prompt 0.4 cycles ahead of the replica and of the previous prompt.
    output = CorrelatorBankOutput(loop_epl(0.5, cis(0.8π), 0.5), LOOP_N, LOOP_N)
    coherent_integration_time = LOOP_N / LOOP_FS
    discriminators(; kwargs...) = TrackingLoops._discriminators(
        tracking_loop,
        state,
        CoherentIntegration(LOOP_SIGNAL, output.correlator_bank, complex(1.0), output, 1, LOOP_FS; kwargs...),
        ConstantNCOCommand(0.0, 0.0),
        NO_LANDING_SAMPLE;
        form_fll = true,
    )
    costas = discriminators()
    @test costas.phase_error ≈ -0.1
    @test costas.frequency_error ≈ -0.1 / coherent_integration_time
    four_quadrant = discriminators(; carrier_phase_polarity = 1)
    @test four_quadrant.phase_error ≈ 0.4
    @test four_quadrant.frequency_error ≈ 0.4 / coherent_integration_time
    # The prompt's sign negative: the error read around the inverted prompt.
    @test discriminators(; carrier_phase_polarity = -1).phase_error ≈ -0.1
end

@testset "A phase error is folded into the PLL discriminator's range" begin
    # The Costas PLL reads ±1/4 cycle, the four-quadrant one ±1/2.
    @test @inferred(TrackingLoops.wrap_phase_error(0.3)) ≈ -0.2
    @test TrackingLoops.wrap_phase_error(0.1) == 0.1
    @test @inferred(TrackingLoops.wrap_phase_error(0.3; polarity = Int8(1))) ≈ 0.3
    @test TrackingLoops.wrap_phase_error(-0.3; polarity = -1) ≈ -0.3
    @test TrackingLoops.wrap_phase_error(0.7; polarity = 1) ≈ -0.3
end

@testset "$(nameof(typeof(tracking_loop))) drops the FLL at phase lock and then narrows" for tracking_loop in (
    ConventionalAssistedPLLAndDLL(),
    DelayCompensatedTrackingLoop(),
)
    # The loop model of `simulate_delayed_loop` without delay: a 30 Hz handover error.
    state = init_loop_state(tracking_loop, LOOP_SIGNAL, 100.0Hz, 0.0Hz)
    dt = LOOP_N / ustrip(Hz, LOOP_FS)
    phase = 0.8
    carrier_command = 100.0
    previous_prompt = complex(0.0, 0.0)
    stages = CarrierLoopStage[]
    for k = 0:1499
        mean_phase = phase + π * (130.0 - carrier_command) * dt
        phase += 2π * (130.0 - carrier_command) * dt
        p = cis(mean_phase)
        output = CorrelatorBankOutput(loop_epl(0.5p, p, 0.5p), LOOP_N, LOOP_N * (k + 1))
        integration = CoherentIntegration(LOOP_SIGNAL, output.correlator_bank, previous_prompt, output, 1, LOOP_FS)
        uses_fll = get_carrier_loop_stage(state) == FLL_ASSISTED_PLL
        # The FLL discriminator is formed only while the loop uses it.
        discriminators = TrackingLoops._discriminators(
            tracking_loop,
            state,
            integration,
            ConstantNCOCommand(carrier_command, 0.0),
            NO_LANDING_SAMPLE;
            form_fll = uses_fll,
        )
        uses_fll || @test discriminators.frequency_error == 0.0Hz
        state, carrier, = step_loop!(tracking_loop, state, integration, ConstantNCOCommand(carrier_command, 0.0), NO_LANDING_SAMPLE)
        push!(stages, get_carrier_loop_stage(state))
        carrier_command = ustrip(Hz, carrier)
        previous_prompt = p
    end
    @test issorted(stages)
    # The NCO-referenced loop pulls in at 18 Hz, more slowly than the conventional
    # loop's 50 Hz.
    @test FLL_ASSISTED_PLL < stages[900] && last(stages) == NARROW_PLL
    @test get_phase_lock_indicator(state) > 0.99 && is_phase_locked(state)
    @test abs(carrier_command - 130.0) < 0.1
    # A reset keeps the staging, a fresh state starts it over.
    reset = reinit_loop_state(tracking_loop, state, 130.0Hz, 0.0Hz)
    @test reset.staging == state.staging
    @test get_carrier_loop_stage(reset) == NARROW_PLL
    # Before the averages span their time the indicator is `NaN`: not locked.
    @test !is_phase_locked(init_loop_state(tracking_loop, LOOP_SIGNAL, 130.0Hz, 0.0Hz))
    @test get_carrier_loop_stage(init_loop_state(tracking_loop, LOOP_SIGNAL, 130.0Hz, 0.0Hz)) ==
          FLL_ASSISTED_PLL
end
