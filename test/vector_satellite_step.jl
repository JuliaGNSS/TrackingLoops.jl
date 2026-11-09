# The per-integration side of the navigation processor piece by piece: the epoch grid, the
# snapshot of a satellite at an epoch, slot_generation and the slots, staleness.
using GNSSDecoder: GNSSDecoder, GNSSDecoderState, decode!

nav_processor_correlator_bank(p = 1.0 + 0.0im) = EarlyPromptLateCorrelatorBank(SVector{3,ComplexF64}(0.5p, p, 0.5p), 0.5)

# An integration of `n` samples ending at `sample_index` on the 4 MHz grid.
nav_processor_integration(prn, sample_index; n = 4000, code_phase = 0.0, p = 1.0 + 0.0im, signal = GPSL1CA()) =
    processed_state(signal, nav_processor_correlator_bank(p), complex(0.0), n, sample_index, sample_index, 1, 4e6Hz;
        prn, code_phase)

@testset "The navigation epochs are the multiples of the update interval" begin
    nav_processor = VectorTrackingLoop(GPSL1CA()).navigation_processor
    @test TL._epoch_at_or_after(nav_processor, 0.0) == 0
    @test TL._epoch_at_or_after(nav_processor, 0.3) == 3
    @test TL._epoch_at_or_after(nav_processor, 0.3 + 1e-9) == 4
    @test TL._epoch_at_or_after(nav_processor, 0.3 - 1e-9) == 3
    @test TL._epoch_time(nav_processor, 7) ≈ 0.7
    # The first epoch whose time, as the integrations' ends are compared with it, is not
    # before `t`: 129 × 30 ms comes out just short of 3.87 s, which 3.87 / 0.03 rounds
    # to 129.
    nav_processor = VectorTrackingLoop(GPSL1CA(); navigation_filter_update_interval = 30ms).navigation_processor
    @test TL._epoch_time(nav_processor, 129) < 3.87
    @test TL._epoch_at_or_after(nav_processor, 3.87) == 130
    @test all((0:16_000_000) ./ 4e6) do t
        epoch = TL._epoch_at_or_after(nav_processor, t)
        TL._epoch_time(nav_processor, epoch - 1) < t <= TL._epoch_time(nav_processor, epoch)
    end
end

@testset "A satellite registers on its first integration" begin
    tracking_loop = VectorTrackingLoop(GPSL1CA(); satellite_capacity_per_signal = 2)
    nav_processor = tracking_loop.navigation_processor
    group = nav_processor.groups[1]
    nco_history = ConstantNCOCommand(100.0, 0.1)
    state = init_loop_state(tracking_loop, GPSL1CA(), 100.0Hz, 0.1Hz)
    state, = step_loop!(tracking_loop, state, nav_processor_integration(7, 4000), nco_history, nothing)
    @test (state.slot_index, state.slot_generation) == (1, 1)
    slot = group.slots[1]
    @test slot.occupied && slot.prn == 7 && slot.running_decoder.prn == 7
    # The integration started at sample 0: the first epoch it can snapshot is the one there,
    # and as the only satellite it ran that epoch's update at once.
    @test slot.first_epoch_index == 0
    @test nav_processor.update_epoch_index == 0 && nav_processor.pending_epoch_index == 1
    @test slot.driver_last_end_sample_index == 4000
    # Later integrations keep the slot.
    state, = step_loop!(tracking_loop, state, nav_processor_integration(7, 8000), nco_history, nothing)
    @test nav_processor.next_slot_generation == 1
    # Another satellite takes the next slot.
    other = init_loop_state(tracking_loop, GPSL1CA(), 50.0Hz, 0.05Hz)
    other, = step_loop!(tracking_loop, other, nav_processor_integration(9, 8000), nco_history, nothing)
    @test other.slot_index == 2 && group.slots[2].prn == 9
    # A fresh state of a PRN already held — the satellite re-acquired — takes its slot
    # over and restarts its bit clock; the old state, of an older slot_generation,
    # registers anew.
    reacquired = init_loop_state(tracking_loop, GPSL1CA(), 100.0Hz, 0.1Hz)
    reacquired, = step_loop!(tracking_loop, reacquired, nav_processor_integration(7, 12_000), nco_history, nothing)
    @test reacquired.slot_index == 1 && reacquired.slot_generation == 3
    @test slot.slot_generation == 3
    @test isnothing(slot.running_decoder.num_bits_since_time_reference)
    # A third satellite finds no free slot and grows the group.
    third = init_loop_state(tracking_loop, GPSL1CA(), 10.0Hz, 0.0Hz)
    third, = step_loop!(tracking_loop, third, nav_processor_integration(11, 12_000), nco_history, nothing)
    @test third.slot_index == 3 && length(group.slots) == 3
end

# B-CNAV1 subframe 1 of satellite `prn` at SOH `soh`: its BCH PRN and SOH codewords as
# ±1 soft bits, the first bit each codeword's bit 0.
b1c_subframe1(prn, soh) = Float32[
    (bit -> bit ? -1 : 1).(digits(Bool, GNSSDecoder.b1c_prn_codeword(prn); base = 2, pad = 21))
    (bit -> bit ? -1 : 1).(digits(Bool, GNSSDecoder.b1c_soh_codeword(soh); base = 2, pad = 51))
]

@testset "A satellite taking over another's slot decodes on its own PRN" begin
    # The slot's decoder is rebuilt on the containers of the satellite that held it.
    # B1C syncs on a PRN codeword, which must be the new satellite's.
    tracking_loop = VectorTrackingLoop(BeiDouB1C_P() => BeiDouB1C_D(); satellite_capacity_per_signal = 1)
    group = tracking_loop.navigation_processor.groups[1]
    nco_history = ConstantNCOCommand(100.0, 0.1)
    state = init_loop_state(tracking_loop, BeiDouB1C_P(), 100.0Hz, 0.1Hz)
    state, = step_loop!(tracking_loop, state, nav_processor_integration(19, 4000; signal = BeiDouB1C_P()), nco_history, nothing)
    slot = group.slots[1]
    @test slot.running_decoder.prn == 19
    slot.occupied = false
    other = init_loop_state(tracking_loop, BeiDouB1C_P(), 100.0Hz, 0.1Hz)
    other, = step_loop!(tracking_loop, other, nav_processor_integration(30, 8000; signal = BeiDouB1C_P()), nco_history, nothing)
    @test other.slot_index == 1 && slot.running_decoder.prn == 30
    # Subframe 1 at both ends of a frame is all the sync reads.
    frame = [b1c_subframe1(30, 10); ones(Float32, 1728); b1c_subframe1(30, 11)]
    decoder = decode!(slot.running_decoder, frame, length(frame))
    @test decoder.raw_data.SOH == 10
end

# Set `slot` up for a snapshot of epoch 1 by an integration ending past it, the driver's and
# the decoding signal's last integrations having ended at `end_sample` (the decoding one at
# `decoding_end`), and take both halves and the snapshot. Returns the decoder's bit
# count and the code phase from its last bit edge.
function take_snapshot!(nav_processor, group, slot, decoder, epoch_sample_index; code_periods, fraction, end_sample,
        driver_phase = fraction, decoding_end = end_sample, code_doppler = 0.0, synced = true)
    fs = 4e6
    nav_processor.pending_epoch_index = 1
    nav_processor.num_snapshots = 0
    slot.occupied = true
    slot.first_epoch_index = 0
    slot.driver_epoch_index = slot.decoding_epoch_index = slot.snapshot_epoch_index = typemin(Int)
    slot.running_decoder = decoder
    slot.decoding_bit_synced = synced
    slot.decoding_num_code_periods_into_bit = code_periods
    slot.decoding_code_phase_fraction = fraction
    slot.decoding_last_end_sample_index = decoding_end
    slot.decoding_last_end_time = decoding_end / fs
    slot.last_code_phase = driver_phase
    slot.driver_last_end_sample_index = end_sample
    slot.driver_last_end_time = end_sample / fs
    nco_history = ConstantNCOCommand(0.0, code_doppler)
    post_correlation_state = nav_processor_integration(3, epoch_sample_index + 4000; signal = group.driver_signal)
    TL._snapshot_driver!(nav_processor, group, slot, group.prototype, post_correlation_state, nco_history)
    TL._snapshot_decoding!(nav_processor, group, slot, post_correlation_state, nco_history)
    TL._complete_snapshot!(nav_processor, group, slot)
    slot.decoder.num_bits_since_time_reference, slot.code_phase
end

@testset "The snapshot reads the code phase from the last bit edge" begin
    signal = GPSL1CA()
    nav_processor = VectorTrackingLoop(signal).navigation_processor
    group = nav_processor.groups[1]
    slot = group.slots[1]
    decoder = GNSSDecoderState(GNSSDecoderState(signal, 3); num_bits_since_time_reference = 10)
    chip_rate = 1.023e6
    fs = 4e6
    snapshot(code_periods, fraction, end_sample, epoch_sample_index; kw...) =
        take_snapshot!(nav_processor, group, slot, decoder, epoch_sample_index; code_periods, fraction, end_sample, kw...)
    # The epoch (sample 400 000) lies 1000 samples past the last integration end: five
    # code periods into the bit, 0.2 chips past the code period boundary, and the chips the
    # replica runs on to the epoch.
    num_bits, code_phase = snapshot(5, 0.2, 399_000, 400_000)
    @test num_bits == 10
    @test code_phase ≈ 5 * 1023 + 0.2 + 1000 / fs * chip_rate
    @test slot.snapshot_epoch_index == 1 && nav_processor.num_snapshots == 1
    # The code Doppler counts.
    _, with_doppler = snapshot(5, 0.2, 399_000, 400_000; code_doppler = 2.0)
    @test with_doppler - code_phase ≈ 1000 / fs * 2.0
    # An integration that ended just short of the bit edge it is counted up to: the phase
    # is the one before the edge, read against one bit fewer.
    num_bits, code_phase = snapshot(0, -0.3, 400_000 - 1, 400_000)
    @test num_bits == 9
    @test code_phase ≈ 20460 - 0.3 + 1 / fs * chip_rate
    # A bit edge exactly at the last integration end.
    num_bits, code_phase = snapshot(0, 0.0, 399_000, 400_000)
    @test num_bits == 10
    @test code_phase ≈ 1000 / fs * chip_rate
    # The replica run on past the next edge before the epoch: one bit more.
    num_bits, code_phase = snapshot(19, 0.5, 400_000 - 4000, 400_000)
    @test num_bits == 11
    @test code_phase ≈ 19 * 1023 + 0.5 + 4000 / fs * chip_rate - 20460
    # Without the replica's code phase the integration is taken to end on a code period boundary.
    num_bits, code_phase = snapshot(5, NaN, 399_000, 400_000)
    @test code_phase ≈ 5 * 1023 + 1000 / fs * chip_rate
    # Out of sync there is no lock, and no bit edge to count from.
    snapshot(5, 0.2, 399_000, 400_000; synced = false)
    @test !slot.measurement_ready
    # An integration that does not cross the pending epoch takes no snapshot.
    nav_processor.num_snapshots = 0
    slot.driver_epoch_index = slot.decoding_epoch_index = slot.snapshot_epoch_index = typemin(Int)
    slot.driver_last_end_sample_index = slot.decoding_last_end_sample_index = 300_000
    slot.driver_last_end_time = slot.decoding_last_end_time = 300_000 / fs
    post_correlation_state = nav_processor_integration(3, 304_000)
    TL._snapshot_driver!(nav_processor, group, slot, group.prototype, post_correlation_state, ConstantNCOCommand(0.0, 0.0))
    TL._snapshot_decoding!(nav_processor, group, slot, post_correlation_state, ConstantNCOCommand(0.0, 0.0))
    TL._complete_snapshot!(nav_processor, group, slot)
    @test nav_processor.num_snapshots == 0
    @test slot.driver_epoch_index == slot.decoding_epoch_index == typemin(Int)
end

@testset "A pair's snapshot takes the bit count from the data, the phase from the pilot" begin
    # GPS L2C: the satellite ranges on the 1.5 s CL code and decodes CM, whose 20 ms
    # code is one data bit. The driver's code phase pins the bit phase modulo
    # the CM code, which its 75 times longer code is a multiple of.
    nav_processor = VectorTrackingLoop(GPSL2CL() => GPSL2CM()).navigation_processor
    group = nav_processor.groups[1]
    @test group.decoding_signal isa GPSL2CM
    slot = group.slots[1]
    @test slot.running_decoder isa typeof(GNSSDecoderState(GPSL2CM(), 1))
    decoder = GNSSDecoderState(GNSSDecoderState(GPSL2CM(), 3); num_bits_since_time_reference = 10)
    chip_rate = 511.5e3
    fs = 4e6
    chips = 1000 / fs * chip_rate
    snapshot(fraction, driver_phase; kw...) = take_snapshot!(nav_processor, group, slot, decoder, 400_000;
        code_periods = 0, fraction, driver_phase, end_sample = 399_000, kw...)
    # The data integration ended 0.2 chips past a CM code boundary, the pilot's replica
    # 0.25 chips past one, 3 CM codes into the CL code: the pilot's phase is taken.
    num_bits, code_phase = snapshot(0.2, 3 * 10230 + 0.25)
    @test num_bits == 10
    @test code_phase ≈ 0.25 + chips
    # At an epoch one sample after the integrations' end, the data integration 0.1 chips past
    # the boundary but the pilot's replica, at the end of its code, 0.3 chips short of
    # it: one bit fewer, read up to the edge.
    one_sample = chip_rate / fs
    num_bits, code_phase = take_snapshot!(nav_processor, group, slot, decoder, 400_000;
        code_periods = 0, fraction = 0.1, driver_phase = 767_250 - 0.3, end_sample = 399_999)
    @test num_bits == 9
    @test code_phase ≈ 10230 - 0.3 + one_sample
    # The data integration behind the pilot's: its last integration ended 4000 samples before the
    # pilot's, and the bit it counts in moved on by those chips.
    num_bits, code_phase = snapshot(0.2, 0.25 + 4000 / fs * chip_rate; decoding_end = 395_000)
    @test num_bits == 10
    @test code_phase ≈ 0.25 + 4000 / fs * chip_rate + chips
    # Without the pilot's code phase the data's count stands.
    _, code_phase = snapshot(0.2, NaN)
    @test code_phase ≈ 0.2 + chips
end

@testset "A satellite without integrations for two update intervals is dropped" begin
    signal = GPSL1CA()
    nav_processor = VectorTrackingLoop(signal).navigation_processor
    group = nav_processor.groups[1]
    fresh, stale, member = group.slots[1], group.slots[2], group.slots[3]
    nav_processor.pending_epoch_index = 10
    for slot in (fresh, stale, member)
        slot.occupied = true
        slot.first_epoch_index = 0
    end
    ended!(slot, t) = (slot.driver_last_end_time = slot.decoding_last_end_time = t)
    fresh.snapshot_epoch_index = 10
    ended!(fresh, 1.0)
    ended!(stale, 0.75)
    ended!(member, 0.7)
    member.loop_state = TL._add_to_vector_loop(group.prototype)
    # The stale ones are not waited for…
    @test TL._all_snapshotted(true, group, nav_processor, 1.0)
    ended!(stale, 0.85)
    @test !TL._all_snapshotted(true, group, nav_processor, 1.0)
    # …and a satellite is stale once either of its signals is.
    stale.decoding_last_end_time = 0.75
    @test TL._all_snapshotted(true, group, nav_processor, 1.0)
    # …nor one past the epoch without its snapshot, which it can no longer take — but
    # one with the driver's half taken is, while its decoding signal is short of it…
    ended!(stale, 1.001)
    @test TL._all_snapshotted(true, group, nav_processor, 1.002)
    stale.driver_epoch_index = 10
    stale.decoding_last_end_time = 0.999
    @test !TL._all_snapshotted(true, group, nav_processor, 1.002)
    stale.driver_epoch_index = typemin(Int)
    ended!(stale, 0.75)
    # …and dropped by the update, the member released.
    @test TL._prepare_slots!(false, group, nav_processor, 1.0)
    @test fresh.in_update && fresh.occupied
    @test !stale.occupied && stale.release_reason == VT_NOT_RELEASED
    @test !member.occupied && member.release_reason == VT_INELIGIBLE
    @test !member.loop_state.vector_controlled
    @test get_satellite_report(VectorTrackingLoop(signal), signal, 3) === nothing
end

@testset "Satellites stepped in turn: the last to reach the epoch runs its update" begin
    tracking_loop = VectorTrackingLoop(GPSL1CA())
    nav_processor = tracking_loop.navigation_processor
    nco_history = ConstantNCOCommand(100.0, 0.1)
    states = [init_loop_state(tracking_loop, GPSL1CA(), 100.0Hz, 0.1Hz) for _ = 1:3]
    prns = (3, 5, 7)
    for k = 1:99, i = 1:3
        states[i], = step_loop!(tracking_loop, states[i], nav_processor_integration(prns[i], 4000k), nco_history, nothing)
    end
    # The first epoch is at sample 0, the start of the first integrations; the next at 0.1 s.
    @test nav_processor.update_index == 1 && nav_processor.update_epoch_index == 0
    for i = 1:2
        states[i], = step_loop!(tracking_loop, states[i], nav_processor_integration(prns[i], 400_000 + 4000), nco_history, nothing)
        @test nav_processor.update_index == 1
        @test nav_processor.num_snapshots == i
    end
    states[3], = step_loop!(tracking_loop, states[3], nav_processor_integration(prns[3], 400_000 + 4000), nco_history, nothing)
    @test nav_processor.update_index == 2 && nav_processor.update_epoch_index == 1 && nav_processor.pending_epoch_index == 2
    # The last one applied its update on the spot; the others do on their next integration.
    @test states[3].update_index == 2
    @test states[1].update_index == 1
    # A satellite that falls behind holds the update up until the others are past the
    # epoch after: then the overdue update runs without it.
    for k = 102:300, i = 1:2
        states[i], = step_loop!(tracking_loop, states[i], nav_processor_integration(prns[i], 4000k), nco_history, nothing)
    end
    @test nav_processor.update_index == 2
    @test nav_processor.num_snapshots == 2
    states[1], = step_loop!(tracking_loop, states[1], nav_processor_integration(prns[1], 4000 * 301), nco_history, nothing)
    @test nav_processor.update_index == 3 && nav_processor.update_epoch_index == 2
    @test !nav_processor.groups[1].slots[3].in_update
end

@testset "A satellite joining past the pending epoch does not skip it" begin
    tracking_loop = VectorTrackingLoop(GPSL1CA())
    nav_processor = tracking_loop.navigation_processor
    nco_history = ConstantNCOCommand(100.0, 0.1)
    a = init_loop_state(tracking_loop, GPSL1CA(), 100.0Hz, 0.1Hz)
    for k = 1:99
        a, = step_loop!(tracking_loop, a, nav_processor_integration(3, 4000k), nco_history, nothing)
    end
    @test nav_processor.pending_epoch_index == 1
    # Stepped first in a chunk that spans the epoch, a newcomer starts past it…
    b = init_loop_state(tracking_loop, GPSL1CA(), 50.0Hz, 0.05Hz)
    b, = step_loop!(tracking_loop, b, nav_processor_integration(5, 4000 * 102), nco_history, nothing)
    @test nav_processor.pending_epoch_index == 1
    @test nav_processor.groups[1].slots[2].first_epoch_index == 2
    # …and the satellite that was due at it still runs it.
    a, = step_loop!(tracking_loop, a, nav_processor_integration(3, 4000 * 101), nco_history, nothing)
    @test nav_processor.update_epoch_index == 1 && nav_processor.pending_epoch_index == 2
    # After a gap in every satellite's integrations the epochs move on to where they resume.
    a, = step_loop!(tracking_loop, a, nav_processor_integration(3, 4000 * 1000), nco_history, nothing)
    b, = step_loop!(tracking_loop, b, nav_processor_integration(5, 4000 * 1000), nco_history, nothing)
    a, = step_loop!(tracking_loop, a, nav_processor_integration(3, 4000 * 1001), nco_history, nothing)
    b, = step_loop!(tracking_loop, b, nav_processor_integration(5, 4000 * 1001), nco_history, nothing)
    @test nav_processor.update_epoch_index == 10
end

@testset "Satellites stepped past an epoch before it is pending are not locked out" begin
    # GNSSReceiver steps the satellites one after another through each chunk: here
    # 4 ms, four 1 ms integrations each, the integrations' boundaries on the epochs.
    tracking_loop = VectorTrackingLoop(GPSL1CA())
    nav_processor = tracking_loop.navigation_processor
    slots = nav_processor.groups[1].slots
    nco_history = ConstantNCOCommand(100.0, 0.1)
    prns = (3, 5, 7)
    states = [init_loop_state(tracking_loop, GPSL1CA(), 100.0Hz, 0.1Hz) for _ in prns]
    updates = []
    function step_chunk!(k, sats = 1:3)
        for i in sats, j = k-3:k
            id = nav_processor.update_index
            states[i], = step_loop!(tracking_loop, states[i], nav_processor_integration(prns[i], 4000j), nco_history, nothing)
            nav_processor.update_index == id ||
                push!(updates, (; epoch = nav_processor.update_epoch_index, time = j / 1000, in_update = [s.in_update for s in slots[1:3]]))
        end
    end
    foreach(step_chunk!, 4:4:500)
    @test nav_processor.update_epoch_index == 4 && nav_processor.pending_epoch_index == 5
    # The first two lose their signal for an epoch; the third snapshots it, and its
    # update waits for them…
    foreach(k -> step_chunk!(k, 3:3), 504:4:600)
    @test nav_processor.pending_epoch_index == 5 && slots[3].snapshot_epoch_index == 5
    # …until they are re-acquired, on integrations that start exactly at the next epoch:
    # registered, they are due from that epoch on, and the second runs the waiting update.
    states[1] = init_loop_state(tracking_loop, GPSL1CA(), 100.0Hz, 0.1Hz)
    states[2] = init_loop_state(tracking_loop, GPSL1CA(), 100.0Hz, 0.1Hz)
    step_chunk!(604)
    @test nav_processor.update_epoch_index == 5
    @test slots[1].first_epoch_index == slots[2].first_epoch_index == 6
    # By then the first is past the next epoch, which it can no longer snapshot: the
    # epochs move on to the one after, which no satellite is past. Every satellite
    # snapshots it and every one after it, and every update runs as soon as the last
    # satellite crosses its epoch, not an epoch late.
    @test nav_processor.pending_epoch_index == 7
    foreach(step_chunk!, 608:4:1500)
    @test nav_processor.update_epoch_index == 14
    recovered = filter(c -> c.epoch > 5, updates)
    @test [c.epoch for c in recovered] == 7:14
    @test all(c -> all(c.in_update), recovered)
    @test all(c -> c.time - c.epoch * 0.1 < 0.01, recovered)
    @test all(s -> s.snapshot_epoch_index == 14, slots[1:3])
end

@testset "A member dropped while its integrations stopped rejoins released" begin
    tracking_loop = VectorTrackingLoop(GPSL1CA())
    nav_processor = tracking_loop.navigation_processor
    group = nav_processor.groups[1]
    nco_history = ConstantNCOCommand(100.0, 0.1)
    a = init_loop_state(tracking_loop, GPSL1CA(), 100.0Hz, 0.1Hz)
    b = init_loop_state(tracking_loop, GPSL1CA(), 50.0Hz, 0.05Hz)
    for k = 1:10
        a, = step_loop!(tracking_loop, a, nav_processor_integration(3, 4000k), nco_history, nothing)
        b, = step_loop!(tracking_loop, b, nav_processor_integration(5, 4000k), nco_history, nothing)
    end
    # `b` is a member of the vector loop, in the slot and in the state its host keeps.
    slot = group.slots[b.slot_index]
    b = TL._add_to_vector_loop(b)
    slot.loop_state = b
    # Its integrations stop for longer than two update intervals: an update drops it and releases it, but
    # only in the slot — the host never hands the satellite the release.
    for k = 11:400
        a, = step_loop!(tracking_loop, a, nav_processor_integration(3, 4000k), nco_history, nothing)
    end
    @test !slot.occupied
    @test !slot.loop_state.vector_controlled
    @test b.vector_controlled
    # Its integrations resume: it registers anew, and applies the release it missed, its
    # scalar loop re-seeded from the replica's Dopplers.
    resumed = ConstantNCOCommand(120.0, 0.12)
    b, = step_loop!(tracking_loop, b, nav_processor_integration(5, 4000 * 401), resumed, nothing)
    @test slot.occupied
    @test !b.vector_controlled
    @test !slot.loop_state.vector_controlled
    @test b.code_frequency_correction == 0.0Hz
    @test b.scalar_state.initial_carrier_doppler == 120.0Hz
    @test b.scalar_state.initial_code_doppler == 0.12Hz
end

@testset "Vector tracking ranges on drivers and decodes their data components" begin
    pair = VectorTrackingLoop(GPSL1CA(), GalileoE1C() => GalileoE1B())
    groups = pair.navigation_processor.groups
    @test groups[1].driver_signal isa GPSL1CA && groups[1].decoding_signal isa GPSL1CA
    @test groups[2].driver_signal isa GalileoE1C && groups[2].decoding_signal isa GalileoE1B
    @test groups[2].slots[1].running_decoder isa typeof(GNSSDecoderState(GalileoE1B(), 1))
    @test pair.navigation_processor.nav_filter.layout.signal_id_by_group == [:GPSL1CA, :GalileoE1C]
    # A dataless driver needs a data component to decode; that one must carry data and
    # have the driver's chip rate, and no signal may be given twice.
    @test_throws "pair it with its data component" VectorTrackingLoop(GalileoE1C())
    @test_throws "carries none" VectorTrackingLoop(GPSL5Q() => GalileoE1C())
    @test_throws "chip rate" VectorTrackingLoop(GPSL5Q() => GPSL1CA())
    @test_throws "only once" VectorTrackingLoop(GPSL1CA(), GPSL1C_P() => GPSL1CA())
    @test_throws "only once" VectorTrackingLoop(GalileoE1C() => GalileoE1B(), GalileoE1B())
    # Code lengths may differ.
    @test VectorTrackingLoop(GPSL2CL() => GPSL2CM()) isa VectorTrackingLoop
end

@testset "An integration of each role" begin
    tracking_loop = VectorTrackingLoop(GPSL1C_P() => GPSL1CA())
    nav_processor = tracking_loop.navigation_processor
    nco_history = ConstantNCOCommand(100.0, 0.1)
    state = init_loop_state(tracking_loop, GPSL1C_P(), 100.0Hz, 0.1Hz)
    # The data component's integration comes first: it registers the satellite and leaves
    # its loop alone, returning the command in force.
    stepped, carrier, code = step_loop!(tracking_loop, state, nav_processor_integration(7, 4000; signal = GPSL1CA()), nco_history, nothing)
    @test (stepped.slot_index, stepped.slot_generation) == (1, 1)
    @test stepped.scalar_state === state.scalar_state
    @test (carrier, code) == (100.0Hz, 0.1Hz)
    @test nav_processor.groups[1].slots[1].decoding_last_end_sample_index == 4000
    # Under an application sample the command in force is the one there.
    timeline = NCOTimeline()
    reset_timeline!(timeline, 100.0, 0.1)
    schedule_command!(timeline, 6000, 120.0, 0.12)
    _, carrier, code = step_loop!(tracking_loop, stepped, nav_processor_integration(7, 5000; signal = GPSL1CA()), timeline, Int64(7000))
    @test (carrier, code) == (120.0Hz, 0.12Hz)
    # The pilot's integration steps the loop on the same slot.
    driven, = step_loop!(tracking_loop, stepped, nav_processor_integration(7, 40_000; n = 40_000, signal = GPSL1C_P()), nco_history, nothing)
    @test driven.slot_index == 1 && nav_processor.next_slot_generation == 1
    @test nav_processor.groups[1].slots[1].driver_last_end_sample_index == 40_000
    # A passenger the tracking_loop does not decode is ignored; a driver it does not range
    # on is an error, as is a satellite stepped with another satellite's driver.
    plain = VectorTrackingLoop(GPSL1CA())
    l1 = init_loop_state(plain, GPSL1CA(), 100.0Hz, 0.1Hz)
    ignored, carrier, = step_loop!(plain, l1, nav_processor_integration(7, 4000; signal = GalileoE1B()), nco_history, nothing)
    @test ignored === l1 && carrier == 100.0Hz
    galileo = init_loop_state(plain, GalileoE1B(), 100.0Hz, 0.1Hz)
    @test_throws "not built for GalileoE1B" step_loop!(plain, galileo, nav_processor_integration(7, 4000; signal = GalileoE1B()), nco_history, nothing)
    @test_throws "ranges on as a driver" step_loop!(plain, galileo, nav_processor_integration(7, 4000), nco_history, nothing)
end

@testset "Another pair's data component is a passenger, not a decoding integration" begin
    # L1C-D is the data component the L1C pair decodes, but here it rides along on a
    # satellite L1 C/A drives: its integrations must leave the L1C group alone, or the
    # satellite would re-register on the two groups by turns.
    tracking_loop = VectorTrackingLoop(GPSL1CA(), GPSL1C_P() => GPSL1C_D())
    nav_processor = tracking_loop.navigation_processor
    nco_history = ConstantNCOCommand(100.0, 0.1)
    state = init_loop_state(tracking_loop, GPSL1CA(), 100.0Hz, 0.1Hz)
    state, = step_loop!(tracking_loop, state, nav_processor_integration(7, 4000), nco_history, nothing)
    for k = 1:300
        passenger = nav_processor_integration(7, 40_000k; n = 40_000, signal = GPSL1C_D())
        stepped, carrier, code = step_loop!(tracking_loop, state, passenger, nco_history, nothing)
        @test stepped === state
        @test (carrier, code) == (100.0Hz, 0.1Hz)
        for j = 1:10
            state, = step_loop!(tracking_loop, state, nav_processor_integration(7, 40_000(k - 1) + 4000j), nco_history, nothing)
        end
    end
    @test nav_processor.next_slot_generation == 1
    @test (state.slot_index, state.slot_generation) == (1, 1)
    @test !any(slot -> slot.occupied, nav_processor.groups[2].slots)
    # The satellite keeps snapshotting the epochs, so its updates keep running: one
    # every 100 ms up to the last integration, 3 s in.
    @test nav_processor.groups[1].slots[1].snapshot_epoch_index == nav_processor.update_epoch_index == 29
end

# A C/N₀ estimator that always reads the same.
struct FixedCN0Estimator <: AbstractCN0Estimator
    dbhz::Float64
end
TL.update!(estimator::FixedCN0Estimator, prompt, context::CN0UpdateContext) = estimator
TL.estimate_cn0(estimator::FixedCN0Estimator, coherent_integration_time) = estimator.dbhz * dBHz

@testset "The C/N₀ is the driver's, from the tracking_loop the host configured" begin
    for (configured, read) in ((42.0, 42.0), (95.0, 80.0))
        tracking_loop = VectorTrackingLoop(GPSL1CA())
        nav_processor = tracking_loop.navigation_processor
        loop = PostCorrelationState(GPSL1CA(), 7; cn0_estimator = FixedCN0Estimator(configured))
        state = init_loop_state(tracking_loop, GPSL1CA(), 100.0Hz, 0.1Hz)
        for k = 1:101
            output = CorrelatorBankOutput(nav_processor_correlator_bank(4000.0), 4000, 4000k, 0.0)
            loop = process_correlator_bank_output!(loop, output, 4e6Hz, 1e-6 / Hz, true)
            state, = step_loop!(tracking_loop, state, loop, ConstantNCOCommand(100.0, 0.1), nothing)
        end
        slot = nav_processor.groups[1].slots[1]
        @test slot.snapshot_epoch_index == 1
        # Capped at 80 dB-Hz, and out of lock without the bit sync.
        @test slot.cn0_dbhz == read
        @test !slot.measurement_ready
    end
end
