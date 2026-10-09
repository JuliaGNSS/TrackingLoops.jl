# The per-record side of the navigation engine piece by piece: the epoch grid, the
# snapshot of a satellite at an epoch, registration and the slots, staleness.
using GNSSDecoder: GNSSDecoderState

engine_correlator(p = 1.0 + 0.0im) = EarlyPromptLateCorrelator(SVector{3,ComplexF64}(0.5p, p, 0.5p), 0.5)

# A record of `n` samples ending at `sample_index` on the 4 MHz grid.
engine_record(prn, sample_index; n = 4000, code_phase = 0.0, p = 1.0 + 0.0im, signal = GPSL1CA()) =
    LoopRecord(signal, engine_correlator(p), complex(0.0), n, sample_index, sample_index, 1, 4e6Hz;
        prn, code_phase)

@testset "The navigation epochs are the multiples of the cycle time" begin
    nav = VectorPLLAndDLL(GPSL1CA()).navigation
    @test TL._epoch_at_or_after(nav, 0.0) == 0
    @test TL._epoch_at_or_after(nav, 0.3) == 3
    @test TL._epoch_at_or_after(nav, 0.3 + 1e-9) == 4
    @test TL._epoch_at_or_after(nav, 0.3 - 1e-9) == 3
    @test TL._epoch_time(nav, 7) ≈ 0.7
end

@testset "A satellite registers on its first record" begin
    estimator = VectorPLLAndDLL(GPSL1CA(); max_satellites_per_signal = 2)
    nav = estimator.navigation
    group = nav.groups[1]
    words = FixedNCOWord(100.0, 0.1)
    state = init_estimator_state(estimator, GPSL1CA(), 100.0Hz, 0.1Hz)
    state, = step_loop(estimator, state, engine_record(7, 4000), words, NO_LANDING_SAMPLE)
    @test (state.slot, state.registration) == (1, 1)
    slot = group.slots[1]
    @test slot.occupied && slot.prn == 7 && slot.running_decoder.prn == 7
    # The record started at sample 0: the first epoch it can snapshot is the one there,
    # and as the only satellite it ran that epoch's cycle at once.
    @test slot.first_epoch == 0
    @test nav.cycle_epoch == 0 && nav.pending_epoch == 1
    @test slot.last_end_sample == 4000
    # Later records keep the slot.
    state, = step_loop(estimator, state, engine_record(7, 8000), words, NO_LANDING_SAMPLE)
    @test nav.registrations == 1
    # Another satellite takes the next slot.
    other = init_estimator_state(estimator, GPSL1CA(), 50.0Hz, 0.05Hz)
    other, = step_loop(estimator, other, engine_record(9, 8000), words, NO_LANDING_SAMPLE)
    @test other.slot == 2 && group.slots[2].prn == 9
    # A fresh state of a PRN already held — the satellite re-acquired — takes its slot
    # over and restarts its bit clock; the old state, of an older registration,
    # registers anew.
    reacquired = init_estimator_state(estimator, GPSL1CA(), 100.0Hz, 0.1Hz)
    reacquired, = step_loop(estimator, reacquired, engine_record(7, 12_000), words, NO_LANDING_SAMPLE)
    @test reacquired.slot == 1 && reacquired.registration == 3
    @test slot.registration == 3
    @test isnothing(slot.running_decoder.num_bits_after_valid_syncro_sequence)
    # A third satellite finds no free slot and grows the group.
    third = init_estimator_state(estimator, GPSL1CA(), 10.0Hz, 0.0Hz)
    third, = step_loop(estimator, third, engine_record(11, 12_000), words, NO_LANDING_SAMPLE)
    @test third.slot == 3 && length(group.slots) == 3
end

@testset "The snapshot reads the code phase from the last symbol edge" begin
    signal = GPSL1CA()
    nav = VectorPLLAndDLL(signal).navigation
    group = nav.groups[1]
    slot = group.slots[1]
    decoder = GNSSDecoderState(GNSSDecoderState(signal, 3); num_bits_after_valid_syncro_sequence = 10)
    code_frequency = 1.023e6
    fs = 4e6
    function take_snapshot!(record, words)
        TL._snapshot_driver_half!(nav, group, slot, group.prototype, record, words)
        TL._snapshot_decoding_half!(nav, group, slot, record, words)
        TL._complete_snapshot!(nav, group, slot)
    end
    function snapshot(blocks, fraction, end_sample, epoch_sample; code_doppler = 0.0)
        nav.pending_epoch = 1
        nav.num_snapshots = 0
        slot.occupied = true
        slot.first_epoch = 0
        slot.snapshot_epoch = slot.driver_epoch = slot.decoding_epoch = typemin(Int)
        slot.running_decoder = decoder
        slot.bit_synced = true
        slot.blocks_into_symbol = blocks
        slot.last_code_phase_fraction = slot.decoding_last_code_phase_fraction = fraction
        slot.last_end_sample = slot.decoding_last_end_sample = end_sample
        slot.last_end_time = slot.decoding_last_end_time = end_sample / fs
        take_snapshot!(engine_record(3, epoch_sample + 4000), FixedNCOWord(0.0, code_doppler))
        slot.decoder.num_bits_after_valid_syncro_sequence, slot.code_phase
    end
    # The epoch (sample 400 000) lies 1000 samples past the last record end: five
    # blocks into the symbol, 0.2 chips past the block boundary, and the chips the
    # replica runs on to the epoch.
    num_bits, code_phase = snapshot(5, 0.2, 399_000, 400_000)
    @test num_bits == 10
    @test code_phase ≈ 5 * 1023 + 0.2 + 1000 / fs * code_frequency
    @test slot.snapshot_epoch == 1 && nav.num_snapshots == 1
    # The code Doppler counts.
    _, with_doppler = snapshot(5, 0.2, 399_000, 400_000; code_doppler = 2.0)
    @test with_doppler - code_phase ≈ 1000 / fs * 2.0
    # A record that ended just short of the symbol edge it is counted up to: the phase
    # is the one before the edge, read against one symbol fewer.
    num_bits, code_phase = snapshot(0, -0.3, 400_000 - 1, 400_000)
    @test num_bits == 9
    @test code_phase ≈ 20460 - 0.3 + 1 / fs * code_frequency
    # A bit edge exactly at the last record end.
    num_bits, code_phase = snapshot(0, 0.0, 399_000, 400_000)
    @test num_bits == 10
    @test code_phase ≈ 1000 / fs * code_frequency
    # The replica run on past the next edge before the epoch: one symbol more.
    num_bits, code_phase = snapshot(19, 0.5, 400_000 - 4000, 400_000)
    @test num_bits == 11
    @test code_phase ≈ 19 * 1023 + 0.5 + 4000 / fs * code_frequency - 20460
    # A record that does not cross the pending epoch takes no snapshot.
    nav.num_snapshots = 0
    slot.snapshot_epoch = slot.driver_epoch = slot.decoding_epoch = typemin(Int)
    slot.last_end_sample = slot.decoding_last_end_sample = 300_000
    slot.last_end_time = slot.decoding_last_end_time = 300_000 / fs
    take_snapshot!(engine_record(3, 304_000), FixedNCOWord(0.0, 0.0))
    @test nav.num_snapshots == 0
    @test slot.snapshot_epoch == typemin(Int)
end

@testset "A satellite without records for two cycles is dropped" begin
    signal = GPSL1CA()
    nav = VectorPLLAndDLL(signal).navigation
    group = nav.groups[1]
    fresh, stale, member = group.slots[1], group.slots[2], group.slots[3]
    nav.pending_epoch = 10
    for slot in (fresh, stale, member)
        slot.occupied = true
        slot.first_epoch = 0
    end
    fresh.snapshot_epoch = fresh.driver_epoch = fresh.decoding_epoch = 10
    fresh.last_end_time = fresh.decoding_last_end_time = 1.0
    stale.last_end_time = stale.decoding_last_end_time = 0.75
    member.last_end_time = member.decoding_last_end_time = 0.7
    member.estimator_state = TL._enable_vector_tracking(group.prototype)
    # The stale ones are not waited for…
    @test TL._all_snapshotted(true, group, nav, 1.0)
    stale.last_end_time = stale.decoding_last_end_time = 0.85
    @test !TL._all_snapshotted(true, group, nav, 1.0)
    stale.last_end_time = stale.decoding_last_end_time = 0.75
    # …and dropped by the cycle, the member released.
    @test TL._prepare_slots!(false, group, nav, 1.0)
    @test fresh.active && fresh.occupied
    @test !stale.occupied && stale.release_reason == VT_NOT_RELEASED
    @test !member.occupied && member.release_reason == VT_INELIGIBLE
    @test !member.estimator_state.vt_on
    @test release_reason(VectorPLLAndDLL(signal), signal, 3) == VT_NOT_RELEASED
end

@testset "Satellites stepped in turn: the last to reach the epoch runs its cycle" begin
    estimator = VectorPLLAndDLL(GPSL1CA())
    nav = estimator.navigation
    words = FixedNCOWord(100.0, 0.1)
    states = [init_estimator_state(estimator, GPSL1CA(), 100.0Hz, 0.1Hz) for _ = 1:3]
    prns = (3, 5, 7)
    for k = 1:99, i = 1:3
        states[i], = step_loop(estimator, states[i], engine_record(prns[i], 4000k), words, NO_LANDING_SAMPLE)
    end
    # The first epoch is at sample 0, the start of the first records; the next at 0.1 s.
    @test nav.cycle_id == 1 && nav.cycle_epoch == 0
    for i = 1:2
        states[i], = step_loop(estimator, states[i], engine_record(prns[i], 400_000 + 4000), words, NO_LANDING_SAMPLE)
        @test nav.cycle_id == 1
        @test nav.num_snapshots == i
    end
    states[3], = step_loop(estimator, states[3], engine_record(prns[3], 400_000 + 4000), words, NO_LANDING_SAMPLE)
    @test nav.cycle_id == 2 && nav.cycle_epoch == 1 && nav.pending_epoch == 2
    # The last one took its cycle up on the spot; the others do on their next record.
    @test states[3].cycle_id == 2
    @test states[1].cycle_id == 1
    # A satellite that falls behind holds the cycle up until the others are past the
    # epoch after: then the overdue cycle runs without it.
    for k = 102:300, i = 1:2
        states[i], = step_loop(estimator, states[i], engine_record(prns[i], 4000k), words, NO_LANDING_SAMPLE)
    end
    @test nav.cycle_id == 2
    @test nav.num_snapshots == 2
    states[1], = step_loop(estimator, states[1], engine_record(prns[1], 4000 * 301), words, NO_LANDING_SAMPLE)
    @test nav.cycle_id == 3 && nav.cycle_epoch == 2
    @test !nav.groups[1].slots[3].active
end

@testset "A satellite joining past the pending epoch does not skip it" begin
    estimator = VectorPLLAndDLL(GPSL1CA())
    nav = estimator.navigation
    words = FixedNCOWord(100.0, 0.1)
    a = init_estimator_state(estimator, GPSL1CA(), 100.0Hz, 0.1Hz)
    for k = 1:99
        a, = step_loop(estimator, a, engine_record(3, 4000k), words, NO_LANDING_SAMPLE)
    end
    @test nav.pending_epoch == 1
    # Stepped first in a chunk that spans the epoch, a newcomer starts past it…
    b = init_estimator_state(estimator, GPSL1CA(), 50.0Hz, 0.05Hz)
    b, = step_loop(estimator, b, engine_record(5, 4000 * 102), words, NO_LANDING_SAMPLE)
    @test nav.pending_epoch == 1
    @test nav.groups[1].slots[2].first_epoch == 2
    # …and the satellite that was due at it still runs it.
    a, = step_loop(estimator, a, engine_record(3, 4000 * 101), words, NO_LANDING_SAMPLE)
    @test nav.cycle_epoch == 1 && nav.pending_epoch == 2
    # After a gap in every satellite's records the epochs move on to where they resume.
    a, = step_loop(estimator, a, engine_record(3, 4000 * 1000), words, NO_LANDING_SAMPLE)
    b, = step_loop(estimator, b, engine_record(5, 4000 * 1000), words, NO_LANDING_SAMPLE)
    a, = step_loop(estimator, a, engine_record(3, 4000 * 1001), words, NO_LANDING_SAMPLE)
    b, = step_loop(estimator, b, engine_record(5, 4000 * 1001), words, NO_LANDING_SAMPLE)
    @test nav.cycle_epoch == 10
end

@testset "A member dropped while its records stopped rejoins released" begin
    estimator = VectorPLLAndDLL(GPSL1CA())
    nav = estimator.navigation
    group = nav.groups[1]
    words = FixedNCOWord(100.0, 0.1)
    a = init_estimator_state(estimator, GPSL1CA(), 100.0Hz, 0.1Hz)
    b = init_estimator_state(estimator, GPSL1CA(), 50.0Hz, 0.05Hz)
    for k = 1:10
        a, = step_loop(estimator, a, engine_record(3, 4000k), words, NO_LANDING_SAMPLE)
        b, = step_loop(estimator, b, engine_record(5, 4000k), words, NO_LANDING_SAMPLE)
    end
    # `b` is a member of the vector loop, in the slot and in the state its host keeps.
    slot = group.slots[b.slot]
    b = TL._enable_vector_tracking(b)
    slot.estimator_state = b
    # Its records stop for longer than two cycles: a cycle drops it and releases it, but
    # only in the slot — the host never hands the satellite the release.
    for k = 11:400
        a, = step_loop(estimator, a, engine_record(3, 4000k), words, NO_LANDING_SAMPLE)
    end
    @test !slot.occupied
    @test !slot.estimator_state.vt_on
    @test b.vt_on
    # Its records resume: it registers anew, and takes up the release it missed, its
    # scalar loop re-seeded from the replica's Dopplers.
    resumed = FixedNCOWord(120.0, 0.12)
    b, = step_loop(estimator, b, engine_record(5, 4000 * 401), resumed, NO_LANDING_SAMPLE)
    @test slot.occupied
    @test !b.vt_on
    @test !slot.estimator_state.vt_on
    @test b.code_freq_update == 0.0Hz
    @test b.inner.init_carrier_doppler == 120.0Hz
    @test b.inner.init_code_doppler == 0.12Hz
end

# A fresh slot of `signals` (a driver, or a `driver => decoding signal` pair) at pending
# epoch 1 (sample 400 000 on the 4 MHz grid), both signals' last records ending at
# `end_sample`, the decoding signal synced `blocks` blocks into its symbol.
function epoch_slot(signals; blocks = 0, end_sample = 399_000, num_bits = 10, cn0_threshold = 30dBHz)
    nav = VectorPLLAndDLL(signals; lock_cn0_threshold = cn0_threshold).navigation
    group = nav.groups[1]
    slot = group.slots[1]
    decoder = GNSSDecoderState(GNSSDecoderState(group.decoding_signal, 3);
        num_bits_after_valid_syncro_sequence = num_bits)
    nav.pending_epoch = 1
    slot.occupied = true
    slot.first_epoch = 0
    slot.running_decoder = decoder
    slot.bit_synced = true
    slot.blocks_into_symbol = blocks
    slot.last_end_sample = slot.decoding_last_end_sample = end_sample
    slot.last_end_time = slot.decoding_last_end_time = end_sample / 4e6
    slot.last_code_phase_fraction = slot.decoding_last_code_phase_fraction = 0.0
    nav, group, slot
end

# The prompts a C/N₀ estimator was fed, as the host's state hands it on.
function cn0_signal_state(cn0_estimator; num_records = 50)
    state = SignalLoopState(GPSL1CA(); cn0_estimator)
    for k = 1:num_records
        p = 4000 * (1.0 + 0.05 * sin(k))
        output = CorrelatorOutput(engine_correlator(p), 4000, 4000k)
        state, = apply_record(state, GPSL1CA(), 3, output, 4e6Hz, 1e-5 / Hz, true)
    end
    state
end

@testset "The C/N₀ is the driver's, from the estimator the host configured" begin
    for (cn0_estimator, expected_lock) in (
        (NoiseRefCN0Estimator(; num_records = 20), true),
        (MomentsCN0Estimator(20), true),
        (NWPRCN0Estimator(; num_records = 20), true),
        # A noise reference that is not ready yet: no estimate, out of lock.
        (NoiseRefCN0Estimator(; num_records = 20), false),
    )
        nav, group, slot = epoch_slot(GPSL1CA())
        state = expected_lock ? cn0_signal_state(cn0_estimator) : SignalLoopState(GPSL1CA(); cn0_estimator)
        record = LoopRecord(GPSL1CA(), engine_correlator(), complex(0.0), 4000, 403_000, 403_000, 1, 4e6Hz;
            prn = 3, signal_state = state)
        TL._snapshot_driver_half!(nav, group, slot, group.prototype, record, FixedNCOWord(0.0, 0.0))
        TL._snapshot_decoding_half!(nav, group, slot, record, FixedNCOWord(0.0, 0.0))
        TL._complete_snapshot!(nav, group, slot)
        @test slot.snapshot_epoch == 1
        expected = min(ustrip(estimate_cn0(state.cn0_estimator, 1e-3s)), TL.MAX_CN0_DBHZ)
        @test slot.cn0_dbhz == expected
        @test slot.in_lock == expected_lock
        expected_lock || @test slot.cn0_dbhz == -Inf
        # What the measurement is weighted by follows it.
        @test TL.linear_cn0_floor(slot.cn0_dbhz) == max(10^(expected / 10), 1.0)
    end
end

@testset "A pair's snapshot puts the decoding signal's symbol edge on the driver's code phase" begin
    # GPS L2CL ranges, L2CM decodes: one 10 230-chip block per CNAV symbol, while the
    # pilot's code runs for 75 symbols. Its code phase past the nearest block boundary
    # is far from the phase within the symbol; only its residue modulo a symbol is.
    signals = GPSL2CL() => GPSL2CM()
    code_frequency = 511_500.0
    symbol = 10_230
    @test TL._driver_code_period(GPSL2CL(), GNSSDecoderState(GPSL2CM(), 3)) == symbol
    function snapshot(symbol_fraction, driver_fraction; blocks = 0)
        nav, group, slot = epoch_slot(signals; blocks)
        slot.decoding_last_code_phase_fraction = symbol_fraction
        slot.last_code_phase_fraction = driver_fraction
        record = engine_record(3, 404_000; signal = GPSL2CL())
        decoding = engine_record(3, 404_000; signal = GPSL2CM())
        TL._snapshot_driver_half!(nav, group, slot, group.prototype, record, FixedNCOWord(0.0, 0.0))
        @test slot.snapshot_epoch == typemin(Int)
        TL._snapshot_decoding_half!(nav, group, slot, decoding, FixedNCOWord(0.0, 0.0))
        TL._complete_snapshot!(nav, group, slot)
        @test slot.snapshot_epoch == 1
        slot.decoder.num_bits_after_valid_syncro_sequence, slot.code_phase
    end
    chips = 1000 / 4e6 * code_frequency
    # The same phase within the symbol, the pilot 37 symbols into its code, or 70 (past
    # the middle of its code, so the nearest code boundary is the next one).
    num_bits, code_phase = snapshot(0.2, 0.25 + 37 * symbol)
    @test num_bits == 10
    @test code_phase ≈ 0.25 + chips
    num_bits, code_phase = snapshot(0.2, 0.25 + 70 * symbol - 767_250)
    @test num_bits == 10
    @test code_phase ≈ 0.25 + chips
    # The decoding signal's replica a little off: the driver's phase counts.
    num_bits, code_phase = snapshot(3.0, 0.25 - 20 * symbol)
    @test num_bits == 10
    @test code_phase ≈ 0.25 + chips
    # The driver just short of a symbol edge the decoding signal counts from: the phase
    # before the edge, read against one symbol fewer.
    num_bits, code_phase = snapshot(0.3 - chips, -0.2 - chips - 5 * symbol)
    @test num_bits == 9
    @test code_phase ≈ symbol - 0.2
end

@testset "A pilot + data pair: arrival order and staleness" begin
    signals = GalileoE1C() => GalileoE1B()
    driver_record(sample) = engine_record(3, sample; n = 16_368, signal = GalileoE1C())
    data_record(sample) = engine_record(3, sample; n = 16_368, signal = GalileoE1B())
    words = FixedNCOWord(0.0, 0.0)
    # The halves taken in either order give the same snapshot.
    snapshots = map((:driver_first, :data_first)) do order
        nav, group, slot = epoch_slot(signals; blocks = 0, end_sample = 399_000)
        slot.decoding_last_code_phase_fraction = 0.1
        slot.last_code_phase_fraction = 0.15
        take_driver() = TL._snapshot_driver_half!(nav, group, slot, group.prototype, driver_record(415_368), words)
        take_data() = TL._snapshot_decoding_half!(nav, group, slot, data_record(415_368), words)
        if order === :driver_first
            take_driver(); TL._complete_snapshot!(nav, group, slot)
            @test nav.num_snapshots == 1 && slot.snapshot_epoch == typemin(Int)
            take_data()
        else
            take_data(); TL._complete_snapshot!(nav, group, slot)
            @test nav.num_snapshots == 1 && slot.snapshot_epoch == typemin(Int)
            take_driver()
        end
        TL._complete_snapshot!(nav, group, slot)
        @test nav.num_snapshots == 1 && slot.snapshot_epoch == 1
        (slot.decoder.num_bits_after_valid_syncro_sequence, slot.code_phase, slot.in_lock)
    end
    @test snapshots[1] == snapshots[2]
    # The code phase is the driver's: 0.15 chips past the block, plus the chips to the epoch.
    @test snapshots[1][2] ≈ 0.15 + 1000 / 4e6 * 1.023e6
    # A satellite whose data component has had no record for two cycles is stale, however
    # fresh its driver.
    nav, group, slot = epoch_slot(signals)
    slot.last_end_time = 1.0
    slot.decoding_last_end_time = 0.75
    @test TL._is_stale(nav, slot, 1.0)
    slot.decoding_last_end_time = 0.85
    @test !TL._is_stale(nav, slot, 1.0)
end

@testset "A pair's records step the satellite by its driver" begin
    estimator = VectorPLLAndDLL(GPSL1CA(), GalileoE1C() => GalileoE1B())
    nav = estimator.navigation
    words = FixedNCOWord(100.0, 0.1)
    state = init_estimator_state(estimator, GalileoE1C(), 100.0Hz, 0.1Hz)
    # The data component's record comes first: it registers the satellite and returns
    # the command in force, without taking up a cycle.
    data = engine_record(5, 16_368; n = 16_368, signal = GalileoE1B())
    state, carrier, code = step_loop(estimator, state, data, words, NO_LANDING_SAMPLE)
    @test (carrier, code) == (100.0Hz, 0.1Hz)
    @test (state.slot, state.registration, state.cycle_id) == (1, 1, -1)
    slot = nav.groups[2].slots[1]
    @test slot.prn == 5 && slot.running_decoder.prn == 5
    @test slot.decoding_last_end_sample == 16_368
    # The driver's record steps the loop on the same slot.
    driver = engine_record(5, 16_368; n = 16_368, signal = GalileoE1C())
    state, = step_loop(estimator, state, driver, words, NO_LANDING_SAMPLE)
    @test state.slot == 1 && nav.registrations == 1
    @test slot.last_end_sample == 16_368
    @test satellite_report(estimator, GalileoE1C(), 5) isa SatelliteReport
    @test satellite_report(estimator, GalileoE1B(), 5) === nothing
end
