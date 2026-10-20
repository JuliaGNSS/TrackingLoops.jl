# The carrier of a satellite in the vector loop: the navigation filter sets its NCO, the
# FLL measures the rate against it, and a residual PLL follows the carrier on the
# prompts the NCO leaves and gives the phase the engine rotates the prompts it decodes
# by. On the synthetic satellites of `vector_simulation.jl`, and the engine's
# decoding record by record.

using Random: Xoshiro
using Statistics: mean, std

# The per-millisecond noise σ of a unit-amplitude prompt at `cn0_dbhz`.
noise_std_at(cn0_dbhz) = sqrt(1 / (2 * 10^(cn0_dbhz / 10) * 1e-3))

# A slot filled as `fill_slot!` fills it, reporting `cn0_dbhz` — whatever the signal
# does: a host's C/N₀ estimator that does not notice a fade.
reporting(cn0_dbhz) =
    (slot, sat, epoch, landing, nav; in_lock = sat.in_view) ->
        (fill_slot!(slot, sat, epoch, landing, nav; in_lock); slot.cn0_dbhz = cn0_dbhz; slot)

# Run `cycles` 100 ms cycles of 1 ms ticks, each channel's noise σ set by
# `noise(cycle, i)` before the cycle. `on_record(cycle, i, sat, phase_error_before,
# record_seconds)` after every record, `on_cycle(cycle)` after every navigation cycle.
function run_noisy!(rx, cycles; noise, fill! = fill_slot!, vector = true,
        on_record = (c, i, sat, before, T) -> nothing, on_cycle = c -> nothing)
    sample = 0
    for cycle = 1:cycles
        foreach(((i, sat),) -> sat.noise_std = noise(cycle, i), enumerate(rx.sats))
        for k = 1:100
            sample_index = sample + k * SAMPLES_PER_MS
            for (i, sat) in enumerate(rx.sats)
                n = record_ms(sat.signal)
                k % n == 0 || continue
                before = sat.phase_error
                simulate_record!(sat, rx.estimator, rx.truth, sim_time(rx, sample_index),
                    n * SAMPLES_PER_MS, sample_index, NO_LANDING_SAMPLE; rx.polarity)
                on_record(cycle, i, sat, before, n * 1e-3)
            end
        end
        sample += 100 * SAMPLES_PER_MS
        vector && run_cycle!(rx, sample, fill!)
        on_cycle(cycle)
    end
    sample
end

cycle_time(rx, cycle) = rx.truth.t0 + cycle * 0.1
velocity_error(rx, cycle) =
    norm(SVector(rx.vt.pvt.velocity.x, rx.vt.pvt.velocity.y, rx.vt.pvt.velocity.z) -
         truth_velocity(rx.truth, cycle_time(rx, cycle)))
solution_error(rx, cycle) = position_error(rx, (; pvt = rx.vt.pvt, time = cycle_time(rx, cycle)))

# A record's true mean residual phase (cycles) against the residual PLL's prediction of it,
# folded into the Costas range: the error the prompt the engine decodes is rotated with.
residual_pll_error(sat, before) =
    rem((before + sat.phase_error) / 2 - sat.state.residual_pll.prompt_phase, 0.5, RoundNearest)

@testset "The filter alone commands the carrier, the residual PLL follows the rest" begin
    rx = SimReceiver(; signals = (GPSL1CA(), GalileoE1B()))
    results, _, diverged = run_simulation!(rx, 60)
    @test !diverged
    for channels in rx.channels, sat in channels
        state = sat.state
        @test state.in_vector_loop && TL._has_carrier_ramp(state)
        # The replica's word is the filter's ramp at the middle of the next record…
        T = record_ms(sat.signal) * 1e-3
        @test sat.carrier_doppler ≈ TL._doppler_at(state.ramp, state.residual_pll.prompt_center + T) atol = 1e-4
        # …which the residual PLL's residual brings onto the incoming carrier.
        @test abs(TL._residual_doppler(state.residual_pll)) < 0.1
        @test phase_lock_indicator(state) > 0.9
    end
    @test rx.sats[1].state.residual_pll.bandwidth == 10.0Hz
    narrow = SimReceiver(; estimator_kw = (; residual_pll_loop_filter_bandwidth = 7.0Hz))
    @test narrow.sats[1].state.residual_pll.bandwidth == 7.0Hz
end

@testset "The residual PLL's phase jitter at 40 dB-Hz is the scalar PLL's or less, on $(nameof(typeof(signal)))" for signal in (
    GPSL1CA(),
    GalileoE1B(),
)
    # The 18 Hz PLL alone, and the residual PLL at its default 10 Hz, on the same noisy
    # signal: each record's true mean phase error against what the loop predicted for it —
    # the replica's phase for the scalar loop, the residual PLL's prediction for the
    # residual PLL, which the engine rotates the prompts it decodes by. A third-order loop's
    # jitter goes as `√B_L`, so 10 Hz is `√(10/18) = 0.75` times the 18 Hz loop's.
    noise = (c, i) -> noise_std_at(40.0)
    scalar = SimReceiver(; signals = (signal,))
    scalar_errors = Float64[]
    run_noisy!(scalar, 100; noise, vector = false, on_record = (c, i, sat, before, T) ->
        c > 30 && push!(scalar_errors, rem((before + sat.phase_error) / 2, 0.5, RoundNearest)))
    @test all(sat -> carrier_loop_stage(sat.state) == NARROW_PLL, scalar.sats)
    rx = SimReceiver(; signals = (signal,))
    errors = Float64[]
    run_noisy!(rx, 100; noise, on_record = (c, i, sat, before, T) ->
        c > 30 && push!(errors, residual_pll_error(sat, before)))
    @test all(sat -> sat.state.in_vector_loop, rx.sats)
    @test length(errors) == length(scalar_errors)
    @test std(errors) / std(scalar_errors) < 0.85
end

@testset "The FLL's rates on the filter-set replica are weighted as they are noisy, $(nameof(typeof(signal))) at $cn0_dbhz dB-Hz$(polarity == 0 ? "" : ", four-quadrant")" for (signal, cn0_dbhz, polarity) in (
    (GalileoE1B(), 45.0, 0),
    (GalileoE1B(), 37.0, 0),
    (GPSL1CA(), 45.0, 0),
    (GalileoE1B(), 34.0, 1),
)
    # Each member's rate — the replica's word at the epoch plus the mean FLL, against a
    # replica the filter alone steers — against the true Doppler at the epoch: the noise
    # the filter weights it with, the −1/2 lag-1 autocorrelation of measurements that
    # share their boundary prompts, and an integrated error bounded by the boundary
    # prompts' phase noise. No residual beyond 6σ. Not on GPS L1 C/A's 1 ms records
    # below about 40 dB-Hz: there a phase difference between two prompts wraps at the
    # Costas discriminator's ±90° often enough to break the telescoping, as it did
    # with the FLL branch.
    rx = SimReceiver(; signals = (signal,), polarity)
    T = 0.1
    residuals = [Float64[] for _ in rx.sats]
    variances = Float64[]
    run_noisy!(rx, 400; noise = (c, i) -> noise_std_at(cn0_dbhz), fill! = reporting(cn0_dbhz),
        on_cycle = c -> begin
            c > 50 || return
            for member in rx.vt.buffers.members
                @test member.available
                sat = rx.sats[member.slot_index]
                push!(residuals[member.slot_index], member.pseudorange_rate -
                    member.wavelength * true_doppler(sat, rx.truth, cycle_time(rx, c)))
                push!(variances, TL.pseudorange_rate_noise_variance(member, T))
            end
        end)
    sqrt_R = sqrt(mean(variances))
    for r in residuals
        @test std(r) / sqrt_R ≈ 1.0 atol = 0.2
        y = r .- mean(r)
        @test sum(y[1:end-1] .* y[2:end]) / sum(abs2, y) ≈ -0.5 atol = 0.15
        @test maximum(abs, cumsum(r) .* T) < 0.06
        @test maximum(abs, r) < 6 * sqrt_R
    end
end

@testset "The residual PLL follows a manoeuvre the filter's ramp does not model, $(nameof(typeof(signal)))" for signal in (
    GPSL1CA(),
    GalileoE1B(),
)
    # A jerk-limited manoeuvre at 40 dB-Hz: 0.5 g for 4 s (3.9 m/s³ of jerk) — the default
    # motion model's 5 m/s² — and 1 g for 2 s (16 m/s³). The constant-velocity filter's ramp
    # leaves the acceleration to the residual PLL, whose phase error stays well inside the
    # Costas discriminator's ±1/4 cycle at the default 10 Hz.
    for (acceleration, duration, max_residual) in ((5.0, 4.0, 6.0), (10.0, 2.0, 10.0))
        truth_kw = (; maneuver_start = 10.0, maneuver_duration = duration,
            maneuver_acceleration = acceleration * SVector(0.6, 0.8, 0.0))
        rx = SimReceiver(; signals = (signal,), truth_kw)
        max_frequency = 0.0
        errors = Float64[]
        run_noisy!(rx, 250; noise = (c, i) -> noise_std_at(40.0), fill! = reporting(40.0),
            on_record = (c, i, sat, before, T) -> begin
                c > 30 || return
                max_frequency = max(max_frequency, abs(TL._residual_doppler(sat.state.residual_pll)))
                push!(errors, residual_pll_error(sat, before))
            end,
            on_cycle = c -> begin
                c >= 90 || return
                @test solution_error(rx, c) < 2.0
                @test velocity_error(rx, c) < 0.8
            end)
        @test max_frequency < max_residual
        @test maximum(abs, errors) < 0.1
        @test all(sat -> sat.state.in_vector_loop, rx.sats)
    end
end

# A member of GPS L1 C/A on the ramp `ramp`, its residual PLL taking over at
# `scalar_doppler`, stepped with 1 ms records at 1 MHz whose residual phase against the
# replica's word, the ramp's, is `residual(t)` (cycles), at amplitude `amplitude(t)` plus
# noise of σ `noise_std`, the host reading no signal while the amplitude is zero; the
# records end at `ends` (samples). `on_record(state, k)` after the `k`-th.
function step_residual(residual, ends; ramp = TL.CarrierRamp(100.0, 0.0, 0.0), scalar_doppler = ramp.doppler,
        unlocked = false, amplitude = t -> 1.0, noise_std = 0.0, on_record = (state, k) -> nothing)
    rng = Xoshiro(1)
    estimator = VectorPLLAndDLL(GPSL1CA())
    state = commanded(init_estimator_state(estimator, GPSL1CA(), ramp.doppler * Hz, 0.0Hz), ramp; scalar_doppler)
    unlocked && (state = TL.SatVectorPLLAndDLL(state; residual_pll = TL.ResidualPLL(state.residual_pll; lock = TL.ResidualPLLLock())))
    words = FixedNCOWord(ramp.doppler, 0.0)
    previous_prompt = complex(0.0)
    for (k, sample_index) in enumerate(ends)
        center = (sample_index - 500) / 1e6
        prompt = amplitude(center) * cis(2π * residual(center)) + noise_std * complex(randn(rng), randn(rng))
        record = prompt_record(GPSL1CA(), prompt, 1000, 1e6Hz; sample_index, previous_prompt,
            cn0_dbhz = iszero(amplitude(center)) ? -Inf : 45.0)
        state, = step_satellite(estimator, state, record, words, NO_LANDING_SAMPLE)
        previous_prompt = get_prompt(record.filtered_correlator)
        on_record(state, k)
    end
    state
end

@testset "Out of phase lock with the signal there, the residual PLL pulls in by itself" begin
    # The incoming carrier 8 Hz off the ramp at 40 dB-Hz, the residual PLL out of lock: the
    # signal is there, so its loop runs on — frequency and rate too — and pulls the
    # residual Doppler in; held at the ramp, its phase alone would not. (Its estimate
    # rings for about two seconds after the pull-in.)
    state = step_residual(t -> 0.1 + 8.0 * t, 1000:1000:4_000_000; unlocked = true,
        noise_std = noise_std_at(40.0))
    @test state.residual_pll.lock.locked
    @test TL._residual_doppler(state.residual_pll) ≈ 8.0 atol = 0.1
    @test abs(rem(state.residual_pll.prompt_phase - (0.1 + 8.0 * state.residual_pll.prompt_center), 0.5, RoundNearest)) < 0.05
end

@testset "Without the signal the residual PLL holds at the ramp" begin
    # Locked on a carrier 2 Hz off the ramp, then 4 s of noise alone, then the signal
    # back 1 Hz off the ramp: the noise does not walk the residual Doppler off, and the
    # residual PLL locks again from the ramp.
    outage(t) = 1.0 <= t < 5.0
    residual(t) = t < 1.0 ? 2.0 * (t - 0.0005) : 1.0 * t
    during = Float64[]
    relock = Ref(NaN)
    state = step_residual(residual, 1000:1000:7_000_000; scalar_doppler = 102.0,
        amplitude = t -> outage(t) ? 0.0 : 1.0, noise_std = noise_std_at(40.0),
        on_record = (state, k) -> begin
            t = state.residual_pll.prompt_center
            1.5 <= t < 5.0 && push!(during, TL._residual_doppler(state.residual_pll))
            t >= 5.0 && isnan(relock[]) && state.residual_pll.lock.locked && (relock[] = t - 5.0)
        end)
    @test count(iszero, during) > 0.95 * length(during)
    @test maximum(abs, during) < 1.0
    @test relock[] < 0.3
    @test TL._residual_doppler(state.residual_pll) ≈ 1.0 atol = 0.1
end

@testset "A gap between records: the replica ran on through it" begin
    # The incoming carrier 2 Hz off the ramp; 2.5 ms of samples skipped after the tenth
    # record, over which the replica advanced by a quarter cycle on its 100 Hz word.
    ends = vcat(1000:1000:10_000, 12_500 .+ (1000:1000:10_000))
    # Zero at the first prompt, as the scalar loop hands over phase-locked.
    residual(t) = 2.0 * (t - 0.0005)
    errors = Float64[]
    fll = Float64[]
    previous_acc = Ref((0, 0.0Hz))
    step_residual(residual, ends; scalar_doppler = 102.0, on_record = (state, k) -> begin
        push!(errors, rem(state.residual_pll.prompt_phase - residual(state.residual_pll.prompt_center), 0.5, RoundNearest))
        count, total = state.fll_disc_sum
        count > previous_acc[][1] && push!(fll, ustrip(Hz, total - previous_acc[][2]))
        previous_acc[] = state.fll_disc_sum
    end)
    # The prediction across the gap stays on the incoming carrier…
    @test maximum(abs, errors) < 1e-3
    # …and the FLL reads the residual Doppler over the prompts' real spacing.
    @test all(f -> isapprox(f, 2.0; atol = 1e-6), fll)
    @test length(fll) == length(ends) - 1
end

@testset "Hand-over: joined at the scalar loop's Doppler, released at the residual PLL's estimate" begin
    rx = SimReceiver()
    _, sample, _ = run_simulation!(rx, 30)
    sat = rx.sats[2]
    slot = rx.group.slots[2]
    # The residual PLL took over at the scalar loop's word, so its estimate of the incoming
    # Doppler continued the scalar loop's (here both on the truth).
    residual_pll = sat.state.residual_pll
    @test TL._has_carrier_ramp(sat.state)
    t = residual_pll.prompt_center
    @test abs(TL._doppler_at(sat.state.ramp, t) + TL._residual_doppler(residual_pll) -
              true_doppler(sat, rx.truth, sim_time(rx, round(Int, t * ustrip(Hz, SIM_FS))))) < 0.5
    # The scalar loop is re-seeded at the residual PLL's estimate of the incoming Doppler
    # where the release lands — on the truth — and at its Doppler rate, and its staging
    # restarts.
    @test residual_pll.lock.locked
    landing_time = sample / ustrip(Hz, SIM_FS)
    estimate = TL._doppler_at(sat.state.ramp, landing_time) + TL._residual_doppler(residual_pll) +
               TL._residual_doppler_rate(residual_pll) * (landing_time - residual_pll.prompt_center)
    rate = sat.state.ramp.rate + TL._residual_doppler_rate(residual_pll)
    TL._mark_release!(slot, VT_INELIGIBLE)
    rx.vt.cycle_id += 1
    sat.state = TL._take_up_cycle(rx.vt, rx.group, slot, sat.state, take_up_record(sat, sample),
        sim_words(sat, NO_LANDING_SAMPLE), Int64(sample))
    @test !sat.state.in_vector_loop
    @test sat.state.inner.init_carrier_doppler ≈ estimate * Hz
    @test abs(estimate - true_doppler(sat, rx.truth, sim_time(rx, sample))) < 0.5
    @test sat.state.inner.carrier_loop_filter.x2 ≈ rate * Hz^2
    @test carrier_loop_stage(sat.state) == FLL_ASSISTED_PLL
    @test !TL._has_carrier_ramp(sat.state)
end

# A decoding record of `signal` at `sample_index` whose prompt is `prompt`: synced, the
# host's bit clock having appended `host_bits`, signed by `bit_polarity`.
function decoding_record(signal, prompt, sample_index; host_bits = Float32[], bit_polarity = 1,
        sync_change = TL.SYNC_UNCHANGED, fold_end = sample_index)
    correlator = EarlyPromptLateCorrelator(SVector{3,ComplexF64}(0.5prompt, prompt, 0.5prompt), 0.5)
    LoopRecord(signal, correlator, complex(0.0), 4000, sample_index, fold_end, 1, 4e6Hz, 5, NaN,
        true, sync_change, 0, view(host_bits, 1:length(host_bits)), NoCN0Estimator(), Int8(0),
        Int8(bit_polarity))
end

# A residual PLL of a satellite in the vector loop that wiped its latest prompt, at
# `center` (s), with `prompt_phase`, its loop filter putting out `carrier_freq_update`.
residual_pll_at(center, prompt_phase; carrier_freq_update = 0.0) = TL.ResidualPLL(
    10.0Hz, carrier_freq_update, TL._residual_pll_loop_filter(carrier_freq_update, 0.0), center, Int8(0), prompt_phase,
    TL.ResidualPLLLock(TL.PhaseLockDetector(0.0, 0.0, 0.0, 0, 1.0), true, 0))

@testset "The engine decodes the prompts it rotated onto the incoming carrier" begin
    signal = GPSL1CA()
    estimator = VectorPLLAndDLL(signal)
    nav = estimator.navigation
    group = first(nav.groups)
    slot = first(group.slots)
    words = FixedNCOWord(100.0, 0.0)
    state = TL._add_to_vector_loop(init_estimator_state(estimator, signal, 100.0Hz, 0.0Hz))
    sentinel = 12345.0f0
    slot.soft_bit[1] = sentinel
    # The prompts of one data symbol of value `-1` turn by 0.1 cycle a record at the
    # filter's Doppler error, and the residual PLL's prediction turns with them: rotated
    # back, they sum to the real axis, which the host's bit clock, signed by its sync
    # polarity, turns into a soft bit at the record that ends the symbol.
    feed!(k, phase; host_bits = Float32[], bit_polarity = 1, unwiped = false) = TL._decode_record!(
        group, slot, decoding_record(signal, -cis(2π * phase), 4000k; host_bits, bit_polarity),
        SatVectorPLLAndDLL(state; residual_pll = residual_pll_at((4000k - 2000) / 4e6, phase)), words, true, unwiped)
    # A symbol the satellite joins the loop within is the host's.
    feed!(1, 0.1; host_bits = Float32[0.5])
    @test slot.soft_bit[1] == sentinel
    @test slot.symbol_from_edge
    # The next ones are the engine's: three records, the last ending the symbol.
    feed!(2, 0.2)
    feed!(3, 0.3)
    @test slot.symbol_accumulator ≈ -2.0
    feed!(4, 0.4; host_bits = Float32[0.7])
    @test slot.soft_bit[1] ≈ -3.0f0
    @test slot.symbol_accumulator == 0.0
    feed!(5, 0.5)
    feed!(6, 0.6; host_bits = Float32[0.7], bit_polarity = -1)
    @test slot.soft_bit[1] ≈ 2.0f0
    # A record correlated before the overlay was wiped breaks the symbol: the host's
    # soft bits until the next one begins.
    slot.soft_bit[1] = sentinel
    feed!(7, 0.7; unwiped = true)
    @test !slot.symbol_from_edge && slot.symbol_accumulator == 0.0
    feed!(8, 0.8; host_bits = Float32[0.7])
    @test slot.soft_bit[1] == sentinel
    @test slot.symbol_from_edge
    feed!(9, 0.9; host_bits = Float32[0.7])
    @test slot.soft_bit[1] ≈ -1.0f0
    # Out of the vector loop the engine decodes the host's soft bits.
    slot.soft_bit[1] = sentinel
    out = init_estimator_state(estimator, signal, 100.0Hz, 0.0Hz)
    TL._decode_record!(group, slot, decoding_record(signal, cis(0.3), 40000; host_bits = Float32[0.7]),
        out, words, true, false)
    @test slot.soft_bit[1] == sentinel
    @test !slot.symbol_from_edge
end

@testset "A data component's prompt is rotated by the phase the residual PLL estimated for its pilot" begin
    # The host may step the pilot's records before the data's: the data's record finds
    # the residual PLL past it, and takes the phase it estimated at the pilot's record of
    # the same span, moved on by its residual Doppler where the spans differ. A record
    # past the residual PLL's latest prompt takes its prediction over the words the
    # replica ran on.
    signal = GalileoE1B()
    residual_pll = residual_pll_at(0.0045, 0.31; carrier_freq_update = 2.0)
    history = TL.PromptHistory(8)
    TL._push_prompt!(history, residual_pll_at(0.0005, 0.2))
    TL._push_prompt!(history, residual_pll)
    ramp = TL.CarrierRamp(1000.0, 0.0, 0.0)
    words = FixedNCOWord(1000.0, 0.0)
    half = (1000.0 * 0.0005, 0.0005s)
    record(start, stop) = decoding_record(signal, cis(0.0), stop; fold_end = stop) |>
        r -> LoopRecord(r.signal, r.filtered_correlator, r.previous_prompt, stop - start, stop, stop, 1,
            r.sampling_frequency, r.prn, r.code_phase, true, r.sync_change, 0, r.new_soft_bits,
            r.cn0_estimator, Int8(0), Int8(1))
    phase(start, stop; history = history) =
        TL._symbol_prompt_phase(residual_pll, ramp, half, history, record(start, stop), words, false)
    @test phase(16000, 20000) ≈ 0.31
    @test phase(0, 4000) ≈ 0.2
    @test phase(0, 8000) ≈ 0.2 + 2.0 * 0.0005
    # The driver's own record is the prediction it was wiped with.
    @test TL._symbol_prompt_phase(residual_pll, ramp, half, history, record(16000, 20000), words, true) == 0.31
    # Past the latest prompt: the residual Doppler carries it on, the replica on the ramp.
    @test phase(20000, 24000) ≈ 0.31 + 2.0 * 0.001
    # Older than the history reaches by more than a driver record: no phase, rather than
    # one moved back over records the loop filter's output did not hold for.
    short = TL.PromptHistory(1)
    TL._push_prompt!(short, residual_pll)
    @test phase(12000, 16000; history = short) ≈ 0.31 - 2.0 * 0.001
    @test isnan(phase(4000, 8000; history = short))
    # Forgotten, as while the filter does not command the satellite: no phase either.
    @test isnan(phase(16000, 20000; history = TL._clear_prompts!(short)))
end

@testset "A slot remembers half a navigation cycle of the driver's prompts" begin
    # The most a host steps one of a satellite's signals ahead of the other, in the
    # driver's shortest records, one code period each, and the record straddling it.
    @test TL._prompt_history_length(GPSL5Q(), GPSL5I(), 100.0ms) == 51
    @test TL._prompt_history_length(GalileoE1C(), GalileoE1B(), 100.0ms) == 14
    @test TL._prompt_history_length(GPSL1C_P(), GPSL1CA(), 200.0ms) == 11
    # A plain signal decodes its own records.
    @test TL._prompt_history_length(GPSL1CA(), GPSL1CA(), 100.0ms) == 0
    estimator = VectorPLLAndDLL(GPSL5Q() => GPSL5I(), GPSL1CA())
    @test [length(group.slots[1].prompt_history.centers) for group in estimator.navigation.groups] == [51, 0]
    # A ring: the newest prompts are kept.
    history = TL.PromptHistory(3)
    for k = 1:5
        TL._push_prompt!(history, residual_pll_at(0.001k, 0.1k))
    end
    @test history.count == 3
    @test sort(history.centers) ≈ [0.003, 0.004, 0.005]
    @test history.phases[history.newest] ≈ 0.5
end

@testset "Decoding through the filter-set carrier, and a −25 dB fade" begin
    # GPS L1 C/A broadcasting LNAV, stepped as a host steps it (the pipeline harness): the
    # engine decodes the prompts it rotated by the residual PLLs, and the decoders go on
    # decoding. One satellite fades by 25 dB for 8 s, which the host's own C/N₀ estimator
    # follows: it leaves the update and rejoins within seconds.
    rx = PipelineReceiver(; num_sats = 8, estimator_kw = (; max_satellites_per_signal = 8))
    fade = (40.0, 48.0)
    results, _, diverged = run_pipeline!(rx, 66.0; fade = (t, i) -> i == 1 && fade[1] <= t < fade[2] ? 10^(-25 / 20) : 1.0)
    @test !diverged
    seed = findfirst(r -> r.status.seeded, results)
    @test 26.0 < results[seed].time - rx.truth.t0 < 27.0
    @test all(r -> r.status.running, results[seed:end])
    slots = rx.vt.groups[1].slots[1:length(rx.sats)]
    @test all(slot -> slot.decoding_bit_synced && slot.in_lock, slots)
    # Every decoder decoded the subframes broadcast while the filter set the carrier, up
    # to a minute after the receiver started, from the engine's own soft bits.
    @test all(slot -> slot.symbol_from_edge, slots)
    @test all(slot -> TL.GNSSDecoder.get_time_of_week(slot.running_decoder) >=
        TL.GNSSDecoder.get_time_of_week(rx.sats[1].decoder) + 60, slots)
    prn = rx.sats[1].decoder.prn
    elapsed(r) = r.time - rx.truth.t0
    measured(r) = (:GPSL1CA, prn) in r.measured
    during = filter(r -> fade[1] + 1.5 <= elapsed(r) < fade[2], results)
    @test !any(measured, during)
    back = findfirst(r -> elapsed(r) >= fade[2] && measured(r), results)
    @test back !== nothing && elapsed(results[back]) - fade[2] < 3.0
    settled = filter(r -> elapsed(r) >= fade[2] + 5, results)
    @test all(measured, settled)
    @test sqrt(mean(abs2, map(r -> position_error(rx, r), settled))) < 1.0
    @test maximum(r -> maximum(abs, r.code_errors), settled) < 0.05
end
