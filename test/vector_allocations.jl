# Every kind of navigation cycle allocates nothing once its buffers have grown: the
# scalar solve, a running cycle, a release, a rejoin, the fallback, and a re-seed from
# a fresh scalar fix, for one constellation and for two, with the engine filled by hand
# (`vector_simulation.jl`). The pipeline's records and cycles, registrations and freed
# slots are checked in `vector_pipeline.jl`.

# Fill every channel's slot for the epoch at `sample`, as `run_cycle!` does, then
# measure the cycle alone.
function measure_navigation(rx::SimReceiver, sample; fill! = fill_slot!)
    nav = rx.vt
    nav.pending_epoch = round(Int, sample / (rx.nominal_ms * SAMPLES_PER_MS))
    for (channels, group) in zip(rx.channels, rx.groups), (i, sat) in enumerate(channels)
        fill!(group.slots[i], sat, sample, NO_LANDING_SAMPLE, nav)
    end
    now = sample / ustrip(Hz, SIM_FS)
    @allocated TL._navigation_cycle!(nav, now)
end

# The satellites take the latest cycle up, as they would on their next record.
function take_up!(rx::SimReceiver, sample)
    for (channels, group) in zip(rx.channels, rx.groups), (i, sat) in enumerate(channels)
        sat.state = TL._take_up_cycle(rx.vt, group, group.slots[i], sat.state,
            take_up_record(sat, sample), sim_words(sat, NO_LANDING_SAMPLE), Int64(sample))
    end
end

@testset "Navigation cycles are allocation-free, $(length(signals)) signal group(s)" for signals in (
    (GPSL1CA(),),
    (GPSL1CA(), GalileoE1B()),
)
    cycle = 100 * SAMPLES_PER_MS
    scalar = SimReceiver(; signals, config = nothing)
    _, sample, _ = run_simulation!(scalar, 3)
    @test measure_navigation(scalar, sample + cycle) == 0

    rx = SimReceiver(; signals)
    _, sample, _ = run_simulation!(rx, 20)
    @test rx.vt.running
    sample += cycle
    @test measure_navigation(rx, sample) == 0
    take_up!(rx, sample)
    # A skipped epoch rebuilds the process model in place.
    sample += 2cycle
    @test measure_navigation(rx, sample) == 0
    @test rx.vt.model.integration_time == 0.2s
    take_up!(rx, sample)

    slot = rx.group.slots[2]
    sample += cycle
    @test measure_navigation(rx, sample; fill! = (v, sat, e, l, nav) ->
        (fill_slot!(v, sat, e, l, nav); v === slot && drop_slot!(v))) == 0
    @test slot.release_reason == VT_INELIGIBLE
    take_up!(rx, sample)
    sample += cycle
    @test measure_navigation(rx, sample) == 0
    take_up!(rx, sample)
    @test rx.sats[2].state.vt_on

    unlocked!(v, sat, e, l, nav) = (fill_slot!(v, sat, e, l, nav); v.in_lock = false; v.pvt_ready = false)
    rx.vt.time_with_insufficient_meas = 10.0s
    sample += cycle
    @test measure_navigation(rx, sample; fill! = unlocked!) == 0
    @test !rx.vt.running
    take_up!(rx, sample)
    sample += cycle
    @test measure_navigation(rx, sample; fill! = unlocked!) == 0
    take_up!(rx, sample)

    sample += cycle
    @test measure_navigation(rx, sample) == 0
    @test rx.vt.running
    @test navigation_status(rx.estimator).enabled
end

@testset "Navigation cycles are allocation-free with $name" for (name, kw) in (
    ("the atmospheric corrections", (; atmosphere = true)),
    ("two bands", (; signals = (GPSL1CA(), GPSL2CM()), range_biases = (0.0, 4.0))),
)
    cycle = 100 * SAMPLES_PER_MS
    scalar = SimReceiver(; kw..., config = nothing)
    _, sample, _ = run_simulation!(scalar, 3)
    @test measure_navigation(scalar, sample + cycle) == 0
    rx = SimReceiver(; kw...)
    _, sample, _ = run_simulation!(rx, 20)
    @test rx.vt.running
    @test measure_navigation(rx, sample + cycle) == 0
end

@testset "Cycles with a decoded GGTO are allocation-free" begin
    # Real Galileo satellites broadcast the GGTO and the fixtures do not, so it is set by
    # hand. Reading it allocated before GNSSDecoder 5.0.2 (JuliaGNSS/GNSSDecoder.jl#101),
    # and still does on Julia 1.10, which GNSSDecoder does not check for allocations
    # either. The allocation-free loop process is built with juliac on 1.12 or later.
    # Here the GGTO also collapses the Galileo clock onto the GPS one.
    for config in (nothing, VectorTracking())
        rx = SimReceiver(; signals = (GPSL1CA(), GalileoE1B()), num_sats = (3, 1), config)
        galileo = only(rx.channels[2])
        galileo.decoder = with_zero_ggto(galileo.decoder)
        _, sample, _ = run_simulation!(rx, 20)
        @test rx.vt.running == !isnothing(config)
        isnothing(config) ||
            @test !isempty(rx.vt.buffers.observability.hub_offset_constraints)
        @test measure_navigation(rx, sample + 100 * SAMPLES_PER_MS) == 0 skip = VERSION < v"1.11"
    end
end

@testset "A satellite takes a cycle up without allocating" begin
    rx = SimReceiver()
    _, sample, _ = run_simulation!(rx, 5)
    sample += 100 * SAMPLES_PER_MS
    measure_navigation(rx, sample)
    group = rx.group
    sat = rx.sats[1]
    slot = group.slots[1]
    record = take_up_record(sat, sample)
    words = sim_words(sat, NO_LANDING_SAMPLE)
    state = sat.state
    take_up(state) = @allocated TL._take_up_cycle(rx.vt, group, slot, state, record, words, Int64(sample))
    take_up(state)
    @test take_up(state) == 0
end
