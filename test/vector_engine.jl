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

# Set `slot` up for a snapshot of epoch 1 by a record ending past it, the driver's and
# the decoding signal's last records having ended at `end_sample` (the decoding one at
# `decoding_end`), and take both halves and the snapshot. Returns the decoder's symbol
# count and the code phase from its last symbol edge.
function take_snapshot!(nav, group, slot, decoder, epoch_sample; blocks, fraction, end_sample,
        driver_phase = fraction, decoding_end = end_sample, code_doppler = 0.0, synced = true)
    fs = 4e6
    nav.pending_epoch = 1
    nav.num_snapshots = 0
    slot.occupied = true
    slot.first_epoch = 0
    slot.driver_epoch = slot.decoding_epoch = slot.snapshot_epoch = typemin(Int)
    slot.running_decoder = decoder
    slot.decoding_bit_synced = synced
    slot.decoding_blocks_into_symbol = blocks
    slot.decoding_code_phase_fraction = fraction
    slot.decoding_last_end_sample = decoding_end
    slot.decoding_last_end_time = decoding_end / fs
    slot.last_code_phase = driver_phase
    slot.last_end_sample = end_sample
    slot.last_end_time = end_sample / fs
    words = FixedNCOWord(0.0, code_doppler)
    record = engine_record(3, epoch_sample + 4000; signal = group.signal)
    TL._snapshot_driver!(nav, group, slot, group.prototype, record, words)
    TL._snapshot_decoding!(nav, group, slot, record, words)
    TL._complete_snapshot!(nav, group, slot)
    slot.decoder.num_bits_after_valid_syncro_sequence, slot.code_phase
end

@testset "The snapshot reads the code phase from the last symbol edge" begin
    signal = GPSL1CA()
    nav = VectorPLLAndDLL(signal).navigation
    group = nav.groups[1]
    slot = group.slots[1]
    decoder = GNSSDecoderState(GNSSDecoderState(signal, 3); num_bits_after_valid_syncro_sequence = 10)
    code_frequency = 1.023e6
    fs = 4e6
    snapshot(blocks, fraction, end_sample, epoch_sample; kw...) =
        take_snapshot!(nav, group, slot, decoder, epoch_sample; blocks, fraction, end_sample, kw...)
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
    # Without the replica's code phase the record is taken to end on a block boundary.
    num_bits, code_phase = snapshot(5, NaN, 399_000, 400_000)
    @test code_phase ≈ 5 * 1023 + 1000 / fs * code_frequency
    # Out of sync there is no lock, and no symbol edge to count from.
    snapshot(5, 0.2, 399_000, 400_000; synced = false)
    @test !slot.in_lock
    # A record that does not cross the pending epoch takes no snapshot.
    nav.num_snapshots = 0
    slot.driver_epoch = slot.decoding_epoch = slot.snapshot_epoch = typemin(Int)
    slot.last_end_sample = slot.decoding_last_end_sample = 300_000
    slot.last_end_time = slot.decoding_last_end_time = 300_000 / fs
    record = engine_record(3, 304_000)
    TL._snapshot_driver!(nav, group, slot, group.prototype, record, FixedNCOWord(0.0, 0.0))
    TL._snapshot_decoding!(nav, group, slot, record, FixedNCOWord(0.0, 0.0))
    TL._complete_snapshot!(nav, group, slot)
    @test nav.num_snapshots == 0
    @test slot.driver_epoch == slot.decoding_epoch == typemin(Int)
end

@testset "A pair's snapshot takes the symbol count from the data, the phase from the pilot" begin
    # GPS L2C: the satellite ranges on the 1.5 s CL code and decodes CM, whose 20 ms
    # code is one data symbol. The driver's code phase pins the symbol phase modulo
    # the CM code, which its 75 times longer code is a multiple of.
    nav = VectorPLLAndDLL(GPSL2CL() => GPSL2CM()).navigation
    group = nav.groups[1]
    @test group.decoding_signal isa GPSL2CM
    slot = group.slots[1]
    @test slot.running_decoder isa typeof(GNSSDecoderState(GPSL2CM(), 1))
    decoder = GNSSDecoderState(GNSSDecoderState(GPSL2CM(), 3); num_bits_after_valid_syncro_sequence = 10)
    code_frequency = 511.5e3
    fs = 4e6
    chips = 1000 / fs * code_frequency
    snapshot(fraction, driver_phase; kw...) = take_snapshot!(nav, group, slot, decoder, 400_000;
        blocks = 0, fraction, driver_phase, end_sample = 399_000, kw...)
    # The data record ended 0.2 chips past a CM code boundary, the pilot's replica
    # 0.25 chips past one, 3 CM codes into the CL code: the pilot's phase is taken.
    num_bits, code_phase = snapshot(0.2, 3 * 10230 + 0.25)
    @test num_bits == 10
    @test code_phase ≈ 0.25 + chips
    # At an epoch one sample after the records' end, the data record 0.1 chips past
    # the boundary but the pilot's replica, at the end of its code, 0.3 chips short of
    # it: one symbol fewer, read up to the edge.
    one_sample = code_frequency / fs
    num_bits, code_phase = take_snapshot!(nav, group, slot, decoder, 400_000;
        blocks = 0, fraction = 0.1, driver_phase = 767_250 - 0.3, end_sample = 399_999)
    @test num_bits == 9
    @test code_phase ≈ 10230 - 0.3 + one_sample
    # The data record behind the pilot's: its last record ended 4000 samples before the
    # pilot's, and the symbol it counts in moved on by those chips.
    num_bits, code_phase = snapshot(0.2, 0.25 + 4000 / fs * code_frequency; decoding_end = 395_000)
    @test num_bits == 10
    @test code_phase ≈ 0.25 + 4000 / fs * code_frequency + chips
    # Without the pilot's code phase the data's count stands.
    _, code_phase = snapshot(0.2, NaN)
    @test code_phase ≈ 0.2 + chips
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
    ended!(slot, t) = (slot.last_end_time = slot.decoding_last_end_time = t)
    fresh.snapshot_epoch = 10
    ended!(fresh, 1.0)
    ended!(stale, 0.75)
    ended!(member, 0.7)
    member.estimator_state = TL._enable_vector_tracking(group.prototype)
    # The stale ones are not waited for…
    @test TL._all_snapshotted(true, group, nav, 1.0)
    ended!(stale, 0.85)
    @test !TL._all_snapshotted(true, group, nav, 1.0)
    # …and a satellite is stale once either of its signals is.
    stale.decoding_last_end_time = 0.75
    @test TL._all_snapshotted(true, group, nav, 1.0)
    ended!(stale, 0.75)
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

@testset "Vector tracking ranges on drivers and decodes their data components" begin
    pair = VectorPLLAndDLL(GPSL1CA(), GalileoE1C() => GalileoE1B())
    groups = pair.navigation.groups
    @test groups[1].signal isa GPSL1CA && groups[1].decoding_signal isa GPSL1CA
    @test groups[2].signal isa GalileoE1C && groups[2].decoding_signal isa GalileoE1B
    @test groups[2].slots[1].running_decoder isa typeof(GNSSDecoderState(GalileoE1B(), 1))
    @test pair.navigation.layout.signal_id_by_group == [:GPSL1CA, :GalileoE1C]
    # A dataless driver needs a data component to decode; that one must carry data and
    # have the driver's chip rate, and no signal may be given twice.
    @test_throws "pair it with its data component" VectorPLLAndDLL(GalileoE1C())
    @test_throws "carries none" VectorPLLAndDLL(GPSL5Q() => GalileoE1C())
    @test_throws "chip rate" VectorPLLAndDLL(GPSL5Q() => GPSL1CA())
    @test_throws "only once" VectorPLLAndDLL(GPSL1CA(), GPSL1C_P() => GPSL1CA())
    @test_throws "only once" VectorPLLAndDLL(GalileoE1C() => GalileoE1B(), GalileoE1B())
    # Code lengths may differ.
    @test VectorPLLAndDLL(GPSL2CL() => GPSL2CM()) isa VectorPLLAndDLL
end

@testset "A record of each role" begin
    estimator = VectorPLLAndDLL(GPSL1C_P() => GPSL1CA())
    nav = estimator.navigation
    words = FixedNCOWord(100.0, 0.1)
    state = init_estimator_state(estimator, GPSL1C_P(), 100.0Hz, 0.1Hz)
    # The data component's record comes first: it registers the satellite and leaves
    # its loop alone, returning the command in force.
    stepped, carrier, code = step_loop(estimator, state, engine_record(7, 4000; signal = GPSL1CA()), words, NO_LANDING_SAMPLE)
    @test (stepped.slot, stepped.registration) == (1, 1)
    @test stepped.inner === state.inner
    @test (carrier, code) == (100.0Hz, 0.1Hz)
    @test nav.groups[1].slots[1].decoding_last_end_sample == 4000
    # Under a landing sample the command in force is the one there.
    timeline = NCOTimeline()
    reset_timeline!(timeline, 100.0, 0.1)
    schedule_word!(timeline, 6000, 120.0, 0.12)
    _, carrier, code = step_loop(estimator, stepped, engine_record(7, 5000; signal = GPSL1CA()), timeline, Int64(7000))
    @test (carrier, code) == (120.0Hz, 0.12Hz)
    # The pilot's record steps the loop on the same slot.
    driven, = step_loop(estimator, stepped, engine_record(7, 40_000; n = 40_000, signal = GPSL1C_P()), words, NO_LANDING_SAMPLE)
    @test driven.slot == 1 && nav.registrations == 1
    @test nav.groups[1].slots[1].last_end_sample == 40_000
    # A passenger the estimator does not decode is ignored; a driver it does not range
    # on is an error, as is a satellite stepped with another satellite's driver.
    plain = VectorPLLAndDLL(GPSL1CA())
    l1 = init_estimator_state(plain, GPSL1CA(), 100.0Hz, 0.1Hz)
    ignored, carrier, = step_loop(plain, l1, engine_record(7, 4000; signal = GalileoE1B()), words, NO_LANDING_SAMPLE)
    @test ignored === l1 && carrier == 100.0Hz
    galileo = init_estimator_state(plain, GalileoE1B(), 100.0Hz, 0.1Hz)
    @test_throws "not built for GalileoE1B" step_loop(plain, galileo, engine_record(7, 4000; signal = GalileoE1B()), words, NO_LANDING_SAMPLE)
    @test_throws "ranges on as a driver" step_loop(plain, galileo, engine_record(7, 4000), words, NO_LANDING_SAMPLE)
end

# A C/N₀ estimator that always reads the same.
struct FixedCN0Estimator <: AbstractCN0Estimator
    dbhz::Float64
end
TL.update(estimator::FixedCN0Estimator, prompt, context::CN0UpdateContext) = estimator
TL.estimate_cn0(estimator::FixedCN0Estimator, integration_time) = estimator.dbhz * dBHz

@testset "The C/N₀ is the driver's, from the estimator the host configured" begin
    for (configured, read) in ((42.0, 42.0), (95.0, 80.0))
        estimator = VectorPLLAndDLL(GPSL1CA())
        nav = estimator.navigation
        loop = SignalLoopState(GPSL1CA(); cn0_estimator = FixedCN0Estimator(configured))
        state = init_estimator_state(estimator, GPSL1CA(), 100.0Hz, 0.1Hz)
        for k = 1:101
            output = CorrelatorOutput(engine_correlator(4000.0), 4000, 4000k, 0.0)
            record = LoopRecord(GPSL1CA(), engine_correlator(), complex(0.0), output, loop, 4e6Hz; prn = 7)
            state, = step_loop(estimator, state, record, FixedNCOWord(100.0, 0.1), NO_LANDING_SAMPLE)
        end
        slot = nav.groups[1].slots[1]
        @test slot.snapshot_epoch == 1
        # Capped at 80 dB-Hz, and out of lock without the bit sync.
        @test slot.cn0_dbhz == read
        @test !slot.in_lock
    end
end
