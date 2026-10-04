# Every kind of navigation cycle allocates nothing once its buffers have grown: the
# scalar solve, a running cycle, a release, a rejoin, the fallback, and a re-seed from
# a fresh scalar fix, for one constellation and for two. The one allocation a run makes
# on purpose — the unscented update's intermediate, built once per measurement count —
# is paid by the warm-up.
measure_navigation(vt, groups, cycle_time) = @allocated update_navigation!(vt, groups, cycle_time)

@testset "Navigation cycles are allocation-free, $(length(signals)) signal group(s)" for signals in (
    (GPSL1CA(),),
    (GPSL1CA(), GalileoE1B()),
)
    scalar = SimReceiver(; signals, config = nothing)
    run_simulation!(scalar, 3)
    @test measure_navigation(scalar.vt, scalar.groups, 0.1s) == 0

    rx = SimReceiver(; signals)
    run_simulation!(rx, 20)
    @test rx.vt.running
    @test measure_navigation(rx.vt, rx.groups, 0.1s) == 0
    # A measured interval off the nominal one rebuilds the process model in place.
    @test measure_navigation(rx.vt, rx.groups, 0.13s) == 0
    @test rx.vt.model.integration_time == 0.13s

    sat = rx.group.sats[2]
    sat.active = false
    @test measure_navigation(rx.vt, rx.groups, 0.1s) == 0
    @test sat.release_reason == VT_INELIGIBLE
    sat.active = true
    @test measure_navigation(rx.vt, rx.groups, 0.1s) == 0
    @test sat.estimator_state.vt_on

    for group in rx.groups, v in group.sats
        v.in_lock = false
        v.pvt_ready = false
    end
    rx.vt.time_with_insufficient_meas = 10.0s
    @test measure_navigation(rx.vt, rx.groups, 0.1s) == 0
    @test !rx.vt.running
    @test measure_navigation(rx.vt, rx.groups, 0.1s) == 0

    for group in rx.groups, v in group.sats
        v.in_lock = true
        v.pvt_ready = true
    end
    @test measure_navigation(rx.vt, rx.groups, 0.1s) == 0
    @test rx.vt.running
end

@testset "Navigation cycles are allocation-free with $name" for (name, kw) in (
    ("the atmospheric corrections", (; atmosphere = true)),
    ("two bands", (; signals = (GPSL1CA(), GPSL2CM()), range_biases = (0.0, 4.0))),
)
    scalar = SimReceiver(; kw..., config = nothing)
    run_simulation!(scalar, 3)
    @test measure_navigation(scalar.vt, scalar.groups, 0.1s) == 0
    rx = SimReceiver(; kw...)
    run_simulation!(rx, 20)
    @test rx.vt.running
    @test measure_navigation(rx.vt, rx.groups, 0.1s) == 0
end

@testset "A Galileo satellite's decoded GGTO allocates" begin
    # GNSSDecoder 5.0.1 allocates while reading a decoded GGTO (`galileo_ggto_offset`,
    # JuliaGNSS/GNSSDecoder.jl#101),
    # once per Galileo satellite and cycle, in the scalar solve and in the filter alike.
    # Real satellites broadcast the GGTO; the fixtures do not, which is why the cycles
    # above stay clean. Here it also collapses the Galileo clock onto the GPS one.
    rx = SimReceiver(; signals = (GPSL1CA(), GalileoE1B()), num_sats = (3, 1))
    galileo = only(rx.channels[2])
    galileo.decoder = with_zero_ggto(galileo.decoder)
    run_simulation!(rx, 20)
    @test rx.vt.running
    @test !isempty(rx.vt.buffers.observability.hub_offset_constraints)
    @test_broken measure_navigation(rx.vt, rx.groups, 0.1s) == 0
end

@testset "decode_soft_bits! is allocation-free" begin
    state = SignalLoopState(GPSL1CA())
    decoder = GNSSDecoderState(GPSL1CA(), 3)
    decode_twice!(decoder, state, bits) = @allocated begin
        append!(get_soft_bits(state), bits)
        decoder = decode_soft_bits!(decoder, state)
    end
    bits = Float32[isodd(i) ? 1 : -1 for i = 1:20]
    decode_twice!(decoder, state, bits)
    @test decode_twice!(decoder, state, bits) == 0
end
