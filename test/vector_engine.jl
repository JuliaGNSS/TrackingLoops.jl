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
    function snapshot(blocks, fraction, end_sample, epoch_sample; code_doppler = 0.0)
        nav.pending_epoch = 1
        nav.num_snapshots = 0
        slot.occupied = true
        slot.first_epoch = 0
        slot.snapshot_epoch = typemin(Int)
        slot.running_decoder = decoder
        slot.bit_buffer = TL.BitBuffer{UInt64}(UInt64(0), 0, true, 0, Int8(1), complex(0.0), blocks,
            slot.bit_buffer.soft_bits, slot.bit_buffer.phase_acc)
        slot.last_code_phase_fraction = fraction
        slot.last_end_sample = end_sample
        slot.last_end_time = end_sample / fs
        state = group.prototype
        record = engine_record(3, epoch_sample + 4000)
        TL._snapshot_epoch!(nav, group, slot, state, record, FixedNCOWord(0.0, code_doppler))
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
    slot.snapshot_epoch = typemin(Int)
    slot.last_end_sample = 300_000
    slot.last_end_time = 300_000 / fs
    TL._snapshot_epoch!(nav, group, slot, group.prototype, engine_record(3, 304_000), FixedNCOWord(0.0, 0.0))
    @test nav.num_snapshots == 0
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
    fresh.snapshot_epoch = 10
    fresh.last_end_time = 1.0
    stale.last_end_time = 0.75
    member.last_end_time = 0.7
    member.estimator_state = TL._enable_vector_tracking(group.prototype)
    # The stale ones are not waited for…
    @test TL._all_snapshotted(true, group, nav, 1.0)
    stale.last_end_time = 0.85
    @test !TL._all_snapshotted(true, group, nav, 1.0)
    stale.last_end_time = 0.75
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
