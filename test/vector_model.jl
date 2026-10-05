# The navigation filter's model, ported from GNSSReceiver's `test/vector_tracking.jl`:
# the configuration, the layout and state indices, the process model, the measurement
# model and its noise, the bias observability and the loop-closure maths.
using LinearAlgebra: diag, eigvals, Symmetric, I, Diagonal, norm, isposdef
using Accessors: @set
using GNSSDecoder: GNSSDecoderState
using PositionVelocityTime: PositionVelocityTime, SPEED_OF_LIGHT, BiasColumns, calc_ρ_hat!,
    calc_H!, calc_DOP!, time_scale_offset_to_gpst, broadcast_time_offset,
    CANDIDATE_HUB_SYSTEMS, NO_TIME_OFFSET, PVTSolution, TAITime, InterFrequencyBias,
    SignalGroup
using Geodesy: ECEF

const TL = TrackingLoops

# A `VTMember` with hand-picked geometry for the measurement-model tests; the
# defaults describe a GPS L1 C/A satellite straight overhead along +x, at 45 dB-Hz,
# with a one-chip early-late spacing and one code period of coherent integration.
function _test_member(;
    group = 1,
    slot = 1,
    prn = 1,
    clock_bias_index = 1,
    ifb_index = 0,
    signal = GPSL1CA(),
    available = true,
    time = 0.0,
    sat_position = SVector(2.6e7, 0.0, 0.0),
    sat_velocity = SVector(0.0, 0.0, 0.0),
    sat_clock_drift = 0.0,
    pseudorange_rate = 0.0,
    code_discriminator = 0.0,
    carrier_discriminator = 0.0,
    cn0 = 10^4.5, # 45 dB-Hz
    early_late_spacing = 1.0,
    coherent_integration_time = nothing,
    decoder = nothing,
)
    c = SPEED_OF_LIGHT
    code_frequency = Float64(ustrip(Hz, get_code_frequency(signal)))
    tcoh = something(coherent_integration_time, get_code_length(signal) / code_frequency)
    time_offsets =
        isnothing(decoder) ? ntuple(_ -> NO_TIME_OFFSET, 3) :
        map(hub -> broadcast_time_offset(decoder, hub; approximate_year = 2021), CANDIDATE_HUB_SYSTEMS)
    TL.VTMember(
        group,
        slot,
        prn,
        clock_bias_index,
        ifb_index,
        c / code_frequency,
        c / ustrip(Hz, get_center_frequency(signal)),
        code_frequency,
        available,
        time,
        # The same instant on the GPS Time count: differs from `time` by the
        # signal's defined scale offset (0 for GPST/GST, −14 s for BDT).
        time - time_scale_offset_to_gpst(get_time_system(signal)),
        sat_position,
        sat_velocity,
        sat_clock_drift,
        pseudorange_rate,
        code_discriminator,
        carrier_discriminator,
        cn0,
        early_late_spacing,
        tcoh,
        time_offsets,
    )
end

@testset "Vector-loop times difference on the GPS Time count" begin
    # One physical instant: a BDT seconds-of-week reads 14 s below the GPS time
    # of week (GPST is TAI−19, BDT is TAI−33). Everything the vector loop
    # *differences* across members — `reference_time`, the pseudoranges — must
    # therefore use `time_gpst_count`, not `time`; mixing raw counts hands every
    # BeiDou member 14 s × c ≈ 4.2×10⁹ m of structural pseudorange offset that
    # the ns-scale BGTO collapse constraint then fights rather than absorbs.
    gps = _test_member(; time = 100.0)
    bds = _test_member(; group = 2, signal = BeiDouB2aI(), clock_bias_index = 2, time = 86.0)
    @test gps.time_gpst_count == 100.0
    @test bds.time_gpst_count ≈ gps.time_gpst_count
    # The own-scale transmit time stays untouched — it is what the ephemeris,
    # the clock polynomial and `calc_steering_offset` are evaluated at.
    @test bds.time == 86.0
end

@testset "VectorTracking configuration" begin
    config = VectorTracking()
    @test config.use_pseudorange_rates # VDFLL by default
    @test config.motion_model_order == 2
    @test config.clock_model_order == 2
    @test config.insufficient_meas_timeout == 10.0s

    vdll = VectorTracking(use_pseudorange_rates = false)
    @test !vdll.use_pseudorange_rates

    @test_throws ArgumentError VectorTracking(motion_model_order = 0)
    @test_throws ArgumentError VectorTracking(motion_model_order = 4)
    @test_throws ArgumentError VectorTracking(clock_model_order = 0)
    @test_throws ArgumentError VectorTracking(clock_model_order = 3)
end

@testset "Navigation filter layout from the configured signals" begin
    # Single constellation, single band: one clock bias, no IFBs.
    layout = TL.NavFilterLayout((GPSL1CA(),))
    @test layout.time_systems == [GPST()]
    @test isempty(layout.extra_bands)
    @test layout.clock_bias_index_by_group == [1]
    @test layout.ifb_index_by_group == [0]
    @test layout.signal_id_by_group == [:GPSL1CA]

    # Two constellations sharing L1: two clock biases, still no IFBs.
    layout = TL.NavFilterLayout((GPSL1CA(), GalileoE1B()))
    @test layout.time_systems == [GPST(), GST()]
    @test isempty(layout.extra_bands)
    @test layout.clock_bias_index_by_group == [1, 2]

    # GPS on two bands: one clock bias, one inter-frequency bias for the band
    # beyond the reference.
    layout = TL.NavFilterLayout((GPSL1CA(), GPSL5I()))
    @test layout.time_systems == [GPST()]
    @test length(layout.extra_bands) == 1
    @test sort(layout.ifb_index_by_group) == [0, 1]
    @test layout.band_by_group == [get_band_id(GPSL1CA()), get_band_id(GPSL5I())]

    # A band stranded alone on its constellation gets no IFB column — its
    # delay folds into that constellation's clock (PositionVelocityTime's
    # observability-driven layout).
    layout = TL.NavFilterLayout((GPSL1CA(), GalileoE5aI()))
    @test layout.time_systems == [GPST(), GST()]
    @test isempty(layout.extra_bands)
end

@testset "Navigation filter state indices" begin
    idxs = TL.NavFilterIndices(2, 2, 1, 0)
    @test idxs.pos == [1, 3, 5]
    @test idxs.vel == [2, 4, 6]
    @test isempty(idxs.acc)
    @test idxs.clock_biases == [7]
    @test idxs.clock_drift == 8
    @test isempty(idxs.ifb)

    # Two clock biases sharing one drift, plus one inter-frequency bias.
    idxs = TL.NavFilterIndices(2, 2, 2, 1)
    @test idxs.clock_biases == [7, 8]
    @test idxs.clock_drift == 9
    @test idxs.ifb == [10]

    idxs = TL.NavFilterIndices(1, 1, 1, 0)
    @test idxs.pos == [1, 2, 3]
    @test isempty(idxs.vel)
    @test idxs.clock_biases == [4]
    @test idxs.clock_drift == 0

    idxs = TL.NavFilterIndices(3, 2, 1, 0)
    @test idxs.pos == [1, 4, 7]
    @test idxs.vel == [2, 5, 8]
    @test idxs.acc == [3, 6, 9]
    @test idxs.clock_biases == [10]
    @test idxs.clock_drift == 11

    idxs = TL.NavFilterIndices(2, 2, 2, 1)
    x = zeros(10)
    x[idxs.pos] = [1.0, 2.0, 3.0]
    x[idxs.vel] = [4.0, 5.0, 6.0]
    x[idxs.clock_biases] = [7.0, 8.0]
    x[idxs.clock_drift] = 9.0
    x[idxs.ifb] = [10.0]
    pos, vel, clock_drift = TL.nav_filter_states(x, idxs)
    @test pos == [1.0, 2.0, 3.0]
    @test vel == [4.0, 5.0, 6.0]
    @test clock_drift == 9.0
    # The [x, y, z, tc₁.., ifb₁..] sub-vector PositionVelocityTime consumes.
    @test TL.position_and_bias_vector(x, idxs) == [1.0, 2.0, 3.0, 7.0, 8.0, 10.0]

    # Unmodelled derivatives read as zero.
    idxs1 = TL.NavFilterIndices(1, 1, 1, 0)
    pos, vel, clock_drift = TL.nav_filter_states([1.0, 2.0, 3.0, 4.0], idxs1)
    @test pos == [1.0, 2.0, 3.0]
    @test vel == zeros(3)
    @test clock_drift == 0.0
end

@testset "Navigation filter process model" begin
    config = VectorTracking()
    T = 0.1
    layout = TL.NavFilterLayout((GPSL1CA(), GalileoE1B()))
    nav = TL.NavFilterModel(config, layout, 100.0ms)
    @test size(nav.F) == (9, 9)
    # Per-axis constant-velocity blocks propagate x += T * ẋ.
    for i in [1, 3, 5]
        @test nav.F[i, i] == 1.0
        @test nav.F[i, i+1] == T
        @test nav.F[i+1, i+1] == 1.0
        @test nav.F[i+1, i] == 0.0
    end
    # Both clock biases integrate the single oscillator drift.
    @test nav.F[7, 9] == T
    @test nav.F[8, 9] == T
    @test nav.F[9, 9] == 1.0
    # Process noise is symmetric positive semi-definite with positive variances,
    # and the drift random walk is fully correlated across the biases (one
    # oscillator).
    @test nav.Q ≈ nav.Q'
    @test all(diag(nav.Q) .> 0)
    @test all(eigvals(Symmetric(nav.Q)) .> -1e-12)
    @test nav.Q[7, 8] > 0

    # Position-only, bias-only model: pure identity dynamics.
    nav1 = TL.NavFilterModel(
        VectorTracking(; motion_model_order = 1, clock_model_order = 1),
        TL.NavFilterLayout((GPSL1CA(),)),
        100.0ms,
    )
    @test nav1.F == I(4)
    @test size(nav1.Q) == (4, 4)

    # An inter-frequency bias state is (nearly) constant.
    nav_ifb = TL.NavFilterModel(config, TL.NavFilterLayout((GPSL1CA(), GPSL5I())), 100.0ms)
    ifb_index = nav_ifb.idxs.ifb[1]
    @test nav_ifb.F[ifb_index, ifb_index] == 1.0
    @test all(nav_ifb.F[ifb_index, 1:end .!= ifb_index] .== 0.0)
    # The IFB random walk is driven by the configured density (m/√s), so a front end whose
    # bands drift apart can be described without touching the source.
    @test nav_ifb.Q[ifb_index, ifb_index] ≈ ustrip(u"m/sqrt(s)", config.ifb_noise_density)^2 * T
    loose_ifb = TL.NavFilterModel(
        VectorTracking(; ifb_noise_density = 0.05u"m/sqrt(s)"),
        TL.NavFilterLayout((GPSL1CA(), GPSL5I())),
        100.0ms,
    )
    @test loose_ifb.Q[ifb_index, ifb_index] ≈ 25 * nav_ifb.Q[ifb_index, ifb_index]

    # Integration-time management: the process model follows the measured interval,
    # however small the deviation (a chunk's 4 ms on 100 ms is metres of clock bias at a
    # TCXO's drift), rebuilt in place.
    nav = TL.NavFilterModel(config, TL.NavFilterLayout((GPSL1CA(),)), 100.0ms)
    F = nav.F
    Q = copy(nav.Q)
    TL.ensure_nav_filter_integration_time!(nav, config, 100.0ms)
    @test nav.Q == Q
    TL.ensure_nav_filter_integration_time!(nav, config, 0.104s)
    @test nav.integration_time == 0.104s
    @test nav.F[1, 2] == 0.104
    @test nav.Q == TL.NavFilterModel(config, TL.NavFilterLayout((GPSL1CA(),)), 0.104s).Q
    TL.ensure_nav_filter_integration_time!(nav, config, 0.2s)
    @test nav.integration_time == 0.2s
    @test nav.F[1, 2] ≈ 0.2
    @test nav.F === F
end

@testset "Motion process noise models the first unmodelled derivative" begin
    # `acceleration_noise_std` is the platform's agility at every motion model order: the
    # axis block is always `Γ Γᵀ σ²` for the first derivative the state does *not* carry,
    # with the orders that do not model the acceleration deriving their figure from it
    # through `MANOEUVRE_TIME`. Checked against the gain vectors written out by hand.
    acc = 5.0u"m/s^2"
    σ_a = ustrip(u"m/s^2", acc)
    τ = ustrip(s, TL.MANOEUVRE_TIME)
    σ_v = σ_a * τ   # velocity change over one manoeuvre
    σ_j = σ_a / τ   # jerk making up one manoeuvre
    T = 0.1
    layout = TL.NavFilterLayout((GPSL1CA(),))
    config(order) = VectorTracking(; motion_model_order = order, acceleration_noise_std = acc)
    axis_block(order) = TL.NavFilterModel(config(order), layout, T * s).Q[1:order, 1:order]

    @test TL.MANOEUVRE_TIME == 2.0s
    @test axis_block(1) ≈ [T] * [T]' * σ_v^2
    @test axis_block(2) ≈ [T^2 / 2, T] * [T^2 / 2, T]' * σ_a^2
    @test axis_block(3) ≈ [T^3 / 6, T^2 / 2, T] * [T^3 / 6, T^2 / 2, T]' * σ_j^2

    # Order 2 is the one whose unmodelled derivative *is* the acceleration, so the manoeuvre
    # time constant does not enter: its driving noise is the configured figure itself.
    @test TL.motion_noise_model(config(2), T)[2] == σ_a
    # The two derived figures sit either side of it by the same factor, so one number
    # describes the platform at all three orders.
    @test TL.motion_noise_model(config(1), T)[2] / σ_a ≈ τ
    @test σ_a / TL.motion_noise_model(config(3), T)[2] ≈ τ

    # The physical figure is what is held fixed across filter intervals, not the resulting
    # variance: at every order the noise recovered from `Q` is the same number at 100 ms as
    # at 1 s.
    for (order, expected) in ((1, σ_v), (2, σ_a), (3, σ_j))
        recovered(interval) = let
            nav = TL.NavFilterModel(config(order), layout, interval)
            # The last modelled derivative's own gain is `T` at every order.
            sqrt(nav.Q[order, order]) / ustrip(s, interval)
        end
        @test recovered(0.1s) ≈ expected
        @test recovered(1.0s) ≈ expected
    end
end

@testset "The state propagated to a landing" begin
    idxs = TL.NavFilterIndices(3, 2, 2, 1)
    x = collect(1.0:13.0)
    x_τ = similar(x)
    # No lead: a plain copy, bit for bit.
    @test TL.propagate_state!(x_τ, x, idxs, 0.0) == x
    τ = 0.25
    TL.propagate_state!(x_τ, x, idxs, τ)
    for i = 1:3
        p, v, a = idxs.pos[i], idxs.vel[i], idxs.acc[i]
        @test x_τ[p] ≈ x[p] + x[v] * τ + x[a] * τ^2 / 2
        @test x_τ[v] ≈ x[v] + x[a] * τ
        @test x_τ[a] == x[a]
    end
    @test x_τ[idxs.clock_biases] ≈ x[idxs.clock_biases] .+ x[idxs.clock_drift] * τ
    @test x_τ[idxs.clock_drift] == x[idxs.clock_drift]
    @test x_τ[idxs.ifb] == x[idxs.ifb]
    # Position-only: nothing moves but the biases, with their drift.
    idxs1 = TL.NavFilterIndices(1, 1, 1, 0)
    @test TL.propagate_state!(zeros(4), [1.0, 2.0, 3.0, 4.0], idxs1, τ) == [1.0, 2.0, 3.0, 4.0]
end

@testset "Measurement prediction" begin
    config = VectorTracking()
    layout = TL.NavFilterLayout((GPSL1CA(), GalileoE1B()))
    nav = TL.NavFilterModel(config, layout, 100.0ms)
    idxs = nav.idxs

    # One GPS satellite along +x, one Galileo along +z: each pseudorange
    # carries its own system's clock bias.
    members = [
        _test_member(; prn = 1, clock_bias_index = 1, sat_position = SVector(2.6e7, 0.0, 0.0)),
        _test_member(;
            group = 2,
            prn = 2,
            clock_bias_index = 2,
            signal = GalileoE1B(),
            sat_position = SVector(0.0, 0.0, 2.6e7),
        ),
    ]
    bias_columns = TL.vt_bias_columns(members, layout)
    @test bias_columns.clock_bias_indices == [1, 2]
    @test bias_columns.num_clock_biases == 2
    @test bias_columns.num_ifb == 0
    @test TL.vt_bias_columns(members, layout, [2]).clock_bias_indices == [2]

    user_pos = SVector(6.378e6, 0.0, 0.0)
    x = zeros(9)
    x[idxs.pos] = user_pos
    x[idxs.clock_biases] = [100.0, 200.0]
    ξ = TL.position_and_bias_vector(x, idxs)
    positions = [member.sat_position for member in members]

    psr = calc_ρ_hat!(zeros(2), positions, ξ, bias_columns)
    @test psr[1] ≈ 2.6e7 - 6.378e6 + 100.0
    @test psr[2] ≈ norm(user_pos - [0.0, 0.0, 2.6e7]) + 200.0 rtol = 1e-5

    # A satellite closing head-on with 1000 m/s reads +1000 m/s; a receiver
    # clock drift of +10 m/s reduces the predicted rate accordingly.
    rate = TL.predict_pseudorange_rate(
        user_pos,
        SVector(0.0, 0.0, 0.0),
        0.0,
        SVector(2.6e7, 0.0, 0.0),
        SVector(-1000.0, 0.0, 0.0),
        0.0,
    )
    @test rate ≈ 1000.0 atol = 1e-6
    rate_with_drift = TL.predict_pseudorange_rate(
        user_pos,
        SVector(0.0, 0.0, 0.0),
        10.0,
        SVector(2.6e7, 0.0, 0.0),
        SVector(-1000.0, 0.0, 0.0),
        0.0,
    )
    @test rate_with_drift ≈ 990.0 atol = 1e-6

    # The measurement model writes the pseudoranges, then the rates, then one row per
    # hub constraint: the difference of the two clock states.
    constraints = [(2, 1, -0.3)]
    velocities = [SVector(-1000.0, 0.0, 0.0), SVector(0.0, 0.0, 0.0)]
    h! = TL.VTMeasurementModel(idxs, zeros(length(ξ)), positions, velocities, [0.0, 0.0],
        bias_columns, true, constraints)
    y = zeros(5)
    h!(y, x)
    @test y[1:2] == psr
    @test y[3] ≈ 1000.0 atol = 1e-6
    @test y[4] ≈ 0.0 atol = 1e-9
    @test y[5] == 200.0 - 100.0
    # Without the rates the hub row follows the pseudoranges directly.
    h_vdll! = TL.VTMeasurementModel(idxs, zeros(length(ξ)), positions, velocities, [0.0, 0.0],
        bias_columns, false, constraints)
    y = zeros(3)
    h_vdll!(y, x)
    @test y == [psr; 100.0]
end

@testset "Measurement noise" begin
    l1 = _test_member(; prn = 1)
    l5 = _test_member(; prn = 2, signal = GPSL5I(), group = 2)
    T = 0.1
    # The DLL thermal noise scales with the squared chip length: GPS L5's chips
    # are 10× shorter than L1 C/A's, so its thermal variance is 100× smaller.
    # (L1 C/A and L5I share a 1 ms coherent dump, so the squaring loss cancels here.)
    @test TL.pseudorange_noise_variance(l1, T) / TL.pseudorange_noise_variance(l5, T) ≈
          (l1.chip_length / l5.chip_length)^2
    # The rate rows scale with the squared carrier wavelength.
    @test TL.pseudorange_rate_noise_variance(l1, T) / TL.pseudorange_rate_noise_variance(l5, T) ≈
          (l1.wavelength / l5.wavelength)^2

    # Both rows use each member's coherent integration time: GPS L1 C/A integrates 1 ms per
    # dump, Galileo E1B 4 ms, so C/A is the noisier measurement in both — the rate through
    # the ATAN FLL phase noise, the range through the noncoherent DLL squaring loss.
    ca = _test_member(; prn = 1, signal = GPSL1CA())
    e1b = _test_member(; prn = 1, signal = GalileoE1B(), group = 2)
    @test ca.coherent_integration_time ≈ 1e-3
    @test e1b.coherent_integration_time ≈ 4e-3
    # Averaged noncoherent DLL variance d/(4·C/N0·T)·(1 + 2/((2−d)·C/N0·T_coh)), in metres².
    function range_var(m, T = T)
        d = m.early_late_spacing
        squaring_loss = 1 + 2 / ((2 - d) * m.cn0 * m.coherent_integration_time)
        d / (4 * T * m.cn0) * squaring_loss * m.chip_length^2
    end
    # Mean-FLL variance 2·σ_φ²/(2π·T)² with σ_φ² = 1/(2·C/N0·T_coh)(1+1/(2·C/N0·T_coh)),
    # expressed as a rate (λ²). No inflation factor: the derived single-cycle variance is the
    # right one for this estimator, and the lag-1 correlation the telescoping creates makes a
    # white-noise filter pessimistic rather than overconfident.
    rate_var(m) = let ct = m.cn0 * m.coherent_integration_time
        m.wavelength^2 * (1 / (2 * ct) * (1 + 1 / (2 * ct))) / (2 * π^2 * T^2)
    end
    @test TL.pseudorange_noise_variance(ca, T) ≈ range_var(ca)
    @test TL.pseudorange_noise_variance(e1b, T) ≈ range_var(e1b)
    @test TL.pseudorange_rate_noise_variance(ca, T) ≈ rate_var(ca)
    @test TL.pseudorange_rate_noise_variance(e1b, T) ≈ rate_var(e1b)
    # shorter-T_coh C/A is the noisier rate measurement
    @test TL.pseudorange_rate_noise_variance(ca, T) > TL.pseudorange_rate_noise_variance(e1b, T)
    # E1B's chips are 293 m against C/A's 293 m, so at equal C/N0 and spacing the only
    # difference in the range rows is the squaring loss — C/A's shorter dump costs it more.
    @test ca.chip_length ≈ e1b.chip_length
    @test TL.pseudorange_noise_variance(ca, T) > TL.pseudorange_noise_variance(e1b, T)

    # The squaring loss is pinned to the coherent dump, not the filter interval: lengthening
    # the filter interval averages the thermal term down but cannot buy back the loss.
    @test TL.pseudorange_noise_variance(ca, 0.1) / TL.pseudorange_noise_variance(ca, 1.0) ≈ 10.0
    # A weak-signal member is dominated by the squaring loss, which no filter interval fixes.
    weak = _test_member(; cn0 = 10^2.0) # 20 dB-Hz
    weak_loss = 1 + 2 / ((2 - weak.early_late_spacing) * weak.cn0 * weak.coherent_integration_time)
    @test weak_loss > 10
    @test TL.pseudorange_noise_variance(weak, 0.1) ≈ range_var(weak, 0.1)

    # Narrowing the correlator at equal C/N0 lowers the thermal term proportionally, which
    # is the whole point of a narrow spacing.
    narrow = _test_member(; early_late_spacing = 0.5)
    @test TL.pseudorange_noise_variance(narrow, 0.1) ≈ range_var(narrow, 0.1)
    @test TL.pseudorange_noise_variance(narrow, 0.1) < range_var(ca, 0.1)

    # The C/N₀ floor keeps a starved or NaN estimate from degenerating the weights.
    @test TL.linear_cn0_floor(45.0) ≈ 10^4.5
    @test TL.linear_cn0_floor(-10.0) == 1.0
    @test TL.linear_cn0_floor(NaN) == 1.0
end

# A decoder with a decoded broadcast offset, set by hand.
function with_ggto(decoder, a_0g)
    data = decoder.data
    data = @set data.A_0G = a_0g
    data = @set data.A_1G = 0.0
    data = @set data.t_0G = 0
    data = @set data.WN_0G = 0
    data = @set data.WN = 0
    @set decoder.data = data
end
# The legacy D1 message broadcasts the BGTO as a bare `A_0GPS`/`A_1GPS` pair with no
# reference epoch, so at `time = 0.0` the offset is `A_0GPS` itself.
function with_bgto(decoder, a_0gps)
    data = decoder.data
    data = @set data.A_0GPS = a_0gps
    data = @set data.A_1GPS = 0.0
    data = @set data.WN = 0
    @set decoder.data = data
end
function with_gal_bgto(decoder, a_0gal)
    data = decoder.data
    data = @set data.A_0Gal = a_0gal
    data = @set data.A_1Gal = 0.0
    data = @set data.WN = 0
    @set decoder.data = data
end

@testset "Bias observability and the broadcast-clock collapse" begin
    assess(layout, members, candidates) = TL.assess_bias_observability!(
        TL.ObservabilityWorkspace(),
        layout,
        members,
        candidates,
    )
    function assess_with_constraints(layout, members, candidates)
        ws = TL.ObservabilityWorkspace()
        obs = TL.assess_bias_observability!(ws, layout, members, candidates)
        obs, copy(ws.hub_offset_constraints)
    end
    # GPS on L1 + Galileo on E1B and E5a: 2 clock biases, 1 inter-frequency
    # bias, so a full independent layout needs 3 + 2 + 1 = 6 pseudoranges.
    layout = TL.NavFilterLayout((GPSL1CA(), GalileoE1B(), GalileoE5aI()))
    gps_clock = layout.clock_bias_index_by_group[1]
    gal_clock = layout.clock_bias_index_by_group[2]
    l5_ifb = layout.ifb_index_by_group[3]
    @test l5_ifb != 0

    gps(prn) = _test_member(; prn, group = 1, clock_bias_index = gps_clock)
    e1b(prn) = _test_member(; prn, group = 2, signal = GalileoE1B(), clock_bias_index = gal_clock)
    e5a(prn) = _test_member(;
        prn,
        group = 3,
        signal = GalileoE5aI(),
        clock_bias_index = gal_clock,
        ifb_index = l5_ifb,
    )

    # Only the bias states a measurement actually touches are unknowns: GPS
    # alone on L1 leaves the Galileo clock and the L5 bias out of the count.
    obs, constraints = assess_with_constraints(layout, [gps(i) for i = 1:5], 1:5)
    @test obs.num_unknowns == 4          # 3 position + 1 GPS clock
    @test isempty(constraints) # nothing to collapse

    # All three signals present and plentiful: everything is estimated
    # independently, with no broadcast GGTO error entering the solution.
    members = [gps(1), gps(2), gps(3), gps(4), e1b(5), e1b(6), e5a(7), e5a(8)]
    obs, constraints = assess_with_constraints(layout, members, 1:8)
    @test obs.num_unknowns == 6          # 3 + 2 clocks + 1 IFB
    @test obs.num_distinct_sats_required == 5 # 3 position + 2 clocks
    @test obs.num_distinct_sats == 8
    @test TL.is_epoch_solvable(obs, 8)
    @test isempty(constraints)

    # The same layout with only five measurements is one short. Without a
    # decoded GGTO there is nothing to collapse onto, so the epoch stays
    # under-determined and the count says so.
    members = [gps(1), gps(2), gps(3), e1b(4), e5a(5)]
    obs, constraints = assess_with_constraints(layout, members, 1:5)
    @test obs.num_unknowns == 6
    @test length(members) < obs.num_unknowns
    @test !TL.is_epoch_solvable(obs, length(members))
    @test isempty(constraints)

    # With the GGTO decoded, the Galileo clock collapses onto the GPS one: one
    # unknown fewer, and the collapse is handed back as the pseudo-measurement
    # `clk_GST - clk_GPST = -c * GGTO`.
    ggto_decoder = with_ggto(GNSSDecoderState(GalileoE1B(), 4), 1e-9)
    ggto_e1b = _test_member(;
        prn = 4,
        group = 2,
        signal = GalileoE1B(),
        clock_bias_index = gal_clock,
        decoder = ggto_decoder,
    )
    @test PositionVelocityTime.time_offset_available(ggto_decoder, GPST())
    members = [gps(1), gps(2), gps(3), ggto_e1b, e5a(5)]
    obs, constraints = assess_with_constraints(layout, members, 1:5)
    @test obs.num_unknowns == 5
    @test length(members) >= obs.num_unknowns
    gst_state, gpst_state, isb = only(constraints)
    @test gst_state == gal_clock
    @test gpst_state == gps_clock
    @test isb ≈ -SPEED_OF_LIGHT * 1e-9

    # A disconnected coverage graph — GPS only on L1, Galileo only on E5a, no
    # E1B to link the bands — leaves the L5 bias collinear with the Galileo
    # clock however many satellites there are. Without a collapse the unknowns are
    # counted on the epoch's own coverage graph, as `decide_bias_layout` counts them:
    # with the two bands in separate components neither carries an observable
    # inter-frequency bias, so the E5a delay folds into the Galileo clock and the epoch
    # has 3 + 2 unknowns, not 6.
    members = [gps(1), gps(2), gps(3), gps(4), gps(5), gps(6), e5a(7)]
    obs, constraints = assess_with_constraints(layout, members, 1:7)
    @test obs.num_unknowns == 5
    @test isempty(constraints) # this E5a member carries no GGTO
    members[7] = _test_member(;
        prn = 7,
        group = 3,
        signal = GalileoE5aI(),
        clock_bias_index = gal_clock,
        ifb_index = l5_ifb,
        decoder = with_ggto(GNSSDecoderState(GalileoE5aI(), 7), 2e-9),
    )
    # Merging Galileo onto GPS reconnects the two bands, which both removes a clock unknown
    # *and* makes the E5a inter-frequency bias observable (and countable) again — so the
    # merged count is recomputed on the merged graph rather than decremented: 3 + 1 clock +
    # 1 IFB. The two effects happen to cancel here; they are separately real.
    obs, constraints = assess_with_constraints(layout, members, 1:7)
    @test obs.num_unknowns == 5
    @test length(constraints) == 1

    # A band whose component reference carries no measurement this cycle is not an unknown
    # of the cycle: with the L1 satellites gone, the L5 delay is collinear with the (single)
    # GPS clock and folds into it, so four L5 satellites determine the epoch.
    dual_band = TL.NavFilterLayout((GPSL1CA(), GPSL5I()))
    @test dual_band.ifb_index_by_group[2] != 0
    l5(prn) = _test_member(;
        prn,
        group = 2,
        signal = GPSL5I(),
        clock_bias_index = dual_band.clock_bias_index_by_group[2],
        ifb_index = dual_band.ifb_index_by_group[2],
    )
    obs, constraints = assess_with_constraints(dual_band, [l5(i) for i = 1:4], 1:4)
    @test obs.num_unknowns == 4
    @test obs.num_distinct_sats == 4
    @test TL.is_epoch_solvable(obs, 4)
    @test isempty(constraints)

    # With both bands present the bias is observable again and does count.
    l1(prn) = _test_member(; prn, group = 1)
    both_bands = [l1(1), l1(2), l1(3), l5(1), l5(2)]
    @test assess(dual_band, both_bands, 1:5).num_unknowns == 5

    # The measurement count alone is not enough: three satellites tracked on two bands each
    # make six measurements against five unknowns, so the count passes — but a satellite's
    # second band is not a second line of sight. `decide_bias_layout` makes the same
    # distinction with the same `(time system, PRN)` identification.
    three_sats_two_bands = [l1(1), l1(2), l1(3), l5(1), l5(2), l5(3)]
    obs = assess(dual_band, three_sats_two_bands, 1:6)
    @test obs.num_unknowns == 5
    @test 6 >= obs.num_unknowns              # the plain count is satisfied …
    @test obs.num_distinct_sats == 3         # … by three satellites' worth of geometry
    @test obs.num_distinct_sats_required == 4
    @test !TL.is_epoch_solvable(obs, 6)

    # A fourth satellite on either band supplies the missing line of sight.
    obs = assess(dual_band, [three_sats_two_bands; l1(4)], 1:7)
    @test obs.num_distinct_sats == 4
    @test TL.is_epoch_solvable(obs, 7)
    # Only the candidates count.
    @test assess(dual_band, [three_sats_two_bands; l1(4)], 1:6).num_distinct_sats == 3

    # A mixed epoch collapses each non-GPS clock through its own broadcast offset —
    # Galileo through the GGTO, BeiDou through the BGTO — independently and in one cycle.
    # B1I is on its own carrier, so the independent layout's coverage graph is
    # disconnected and the collapse is what reconnects it: 3 + 3 clocks + 0 IFBs recounted
    # as 3 + 1 clock + 1 IFB.
    tri = TL.NavFilterLayout((GPSL1CA(), GalileoE1B(), BeiDouB1I()))
    gps_tri, gal_tri, bds_tri = tri.clock_bias_index_by_group
    gal_member = _test_member(;
        prn = 4,
        group = 2,
        signal = GalileoE1B(),
        clock_bias_index = gal_tri,
        decoder = with_ggto(GNSSDecoderState(GalileoE1B(), 4), 1e-9),
    )
    bds_decoder = with_bgto(GNSSDecoderState(BeiDouB1I(), 21), 3e-9)
    bds_member = _test_member(;
        prn = 21,
        group = 3,
        signal = BeiDouB1I(),
        clock_bias_index = bds_tri,
        decoder = bds_decoder,
    )
    @test PositionVelocityTime.time_offset_available(bds_decoder, GPST())
    gps_of(prn) = _test_member(; prn, group = 1, clock_bias_index = gps_tri)
    mixed = [gps_of(1), gps_of(2), gps_of(3), gal_member, bds_member]
    obs, constraints = assess_with_constraints(tri, mixed, 1:5)
    @test obs.num_unknowns == 5              # 3 position + the surviving GPS clock + 1 IFB
    @test obs.num_distinct_sats_required == 4
    @test TL.is_epoch_solvable(obs, 5)
    @test length(constraints) == 2
    by_state = Dict(state => (ref, isb) for (state, ref, isb) in constraints)
    @test by_state[gal_tri][1] == gps_tri
    @test by_state[gal_tri][2] ≈ -SPEED_OF_LIGHT * 1e-9
    @test by_state[bds_tri][1] == gps_tri
    # Explicit tolerance: `calc_steering_offset` recovers the ~ns BDT steering residual by
    # subtracting the defined 14 s back out of `A_0`, which costs one ULP at 14.
    @test by_state[bds_tri][2] ≈ -SPEED_OF_LIGHT * 3e-9 atol = 1e-5

    # Only the systems that actually carry an offset collapse.
    mixed[5] = _test_member(; prn = 21, group = 3, signal = BeiDouB1I(), clock_bias_index = bds_tri)
    obs, constraints = assess_with_constraints(tri, mixed, 1:5)
    @test length(constraints) == 1
    @test only(constraints)[1] == gal_tri
    @test obs.num_unknowns == 5               # 3 position + GPS clock + BeiDou clock

    # The hub is not GPS-specific: with GPS absent, the same cycle collapses onto
    # Galileo instead, in the same fixed hub order as `decide_bias_layout`.
    duo = TL.NavFilterLayout((GalileoE1B(), BeiDouB1I()))
    gal_duo, bds_duo = duo.clock_bias_index_by_group
    bds_gal_decoder = with_gal_bgto(GNSSDecoderState(BeiDouB1I(), 22), 4e-9)
    @test PositionVelocityTime.time_offset_available(bds_gal_decoder, GST())
    bds_gal_member = _test_member(;
        prn = 22,
        group = 2,
        signal = BeiDouB1I(),
        clock_bias_index = bds_duo,
        decoder = bds_gal_decoder,
    )
    gal_of(prn) = _test_member(; prn, group = 1, signal = GalileoE1B(), clock_bias_index = gal_duo)
    obs, constraints = assess_with_constraints(duo, [gal_of(1), gal_of(2), gal_of(3), bds_gal_member], 1:4)
    gst_constraint = only(constraints)
    @test gst_constraint[1] == bds_duo
    @test gst_constraint[2] == gal_duo
    @test gst_constraint[3] ≈ -SPEED_OF_LIGHT * 4e-9 atol = 1e-5
    # The collapse reconnects the disjoint L1/B1I split: 3 position + one merged
    # clock + the now-observable B1I inter-frequency bias.
    @test obs.num_unknowns == 5

    # The position uncertainty reported with the solution is the root sum of the position
    # variances, read off the position block of P.
    idxs = TL.NavFilterIndices(VectorTracking(), layout)
    P = zeros(10, 10)
    P[idxs.pos, idxs.pos] = Diagonal([9.0, 16.0, 144.0])
    @test TL.position_uncertainty(P, idxs) ≈ sqrt(169.0)
end

@testset "Dense bias columns for DOP" begin
    # The filter's fixed bias layout carries a clock for every configured
    # constellation, but a DOP design matrix must only carry the columns the
    # epoch can determine — an empty column would make `HᵀH` singular and
    # `calc_DOP!` report -1 whenever a constellation is missing.
    layout = TL.NavFilterLayout((GPSL1CA(), GalileoE1B(), GalileoE5aI()))
    gal_clock = layout.clock_bias_index_by_group[2]
    l5_ifb = layout.ifb_index_by_group[3]
    members = [
        _test_member(; prn, group = 2, signal = GalileoE1B(), clock_bias_index = gal_clock) for
        prn = 1:5
    ]
    columns, primary = TL.dense_bias_columns(members, 1, 2, 1)
    @test columns.num_clock_biases == 1
    @test columns.num_ifb == 0
    @test all(columns.clock_bias_indices .== 1)
    @test primary == 1 # the GPS primary is not present; falls back to the first

    # Adding an E5a member brings its band's bias back as a real column.
    push!(
        members,
        _test_member(;
            prn = 9,
            group = 3,
            signal = GalileoE5aI(),
            clock_bias_index = gal_clock,
            ifb_index = l5_ifb,
        ),
    )
    columns, _ = TL.dense_bias_columns(members, gal_clock, 2, 1)
    @test columns.num_clock_biases == 1
    @test columns.num_ifb == 1
    @test columns.ifb_indices == [0, 0, 0, 0, 0, 1]

    # A rank-deficient design is what `calc_DOP!` reports with its sentinel, a determined
    # one a real geometry.
    lone = [_test_member(; prn = 1)]
    columns, _ = TL.dense_bias_columns(lone, 1, 1, 0)
    H = calc_H!(zeros(1, 4), [lone[1].sat_position], [6.378e6, 0.0, 0.0, 0.0], columns)
    @test calc_DOP!(zeros(4, 4), H, ECEF(6.378e6, 0.0, 0.0), 1).GDOP < 0
end

@testset "Solution epoch from the cached week offset" begin
    # Until an epoch offset has been resolved there is no epoch to report, so the solution
    # carries no timestamp and reads as "no fix" — as it must.
    @test isnothing(TL.vt_time(nothing, 123.25s, 0.0))

    # With the offset known, the epoch is the clock-bias-corrected reference time on top of
    # it: a receiver clock one light-second fast puts the same reference time one second
    # earlier.
    week_offset = 2052 * TL.SECONDS_PER_WEEK
    epoch = TL.vt_time(week_offset, 123.25s, 0.0)
    @test epoch == TAITime(week_offset + 123, 0.25)
    @test epoch - TL.vt_time(week_offset, 123.25s, SPEED_OF_LIGHT) ≈ 1.0 rtol = 1e-9

    # The offset is read off a primary-system row: week, start epoch and the scale offset
    # that puts a BDT count onto the GPS Time count.
    states = PositionVelocityTime._precompile_states(
        GPSL1CA(), PositionVelocityTime._PRECOMPILE_GPS_L1CA_STATES, identity, GPSL1CA())
    row = only(PositionVelocityTime.collect_measurements(
        SignalGroup(GPSL1CA(), states[1:1]);
        approximate_year = 2021,
    )[1])
    @test TL.time_epoch_offset(row) ==
          row.week * TL.SECONDS_PER_WEEK + row.system_start_epoch.second

    # A week rollover takes a week off the time of week, so the offset — the only other place
    # the run's absolute epoch is held — has to gain one, or every later solution would be
    # timestamped exactly one week in the past.
    @test TL.rolled_over_time_epoch_offset(week_offset, week_offset, true) ==
          week_offset + TL.SECONDS_PER_WEEK
    @test TL.rolled_over_time_epoch_offset(week_offset, week_offset, false) == week_offset
    # An offset first resolved on the very cycle that wraps comes from a decoder that has
    # rolled its own week number over already, so it must not be advanced a second time.
    @test TL.rolled_over_time_epoch_offset(week_offset, nothing, true) == week_offset
    @test isnothing(TL.rolled_over_time_epoch_offset(nothing, nothing, true))

    # End to end across the boundary: one cycle before the wrap and one after must be a
    # single integration time apart, not a week.
    before = TL.vt_time(week_offset, (TL.SECONDS_PER_WEEK - 0.1) * s, 0.0)
    after = TL.vt_time(TL.rolled_over_time_epoch_offset(week_offset, week_offset, true), 0.0s, 0.0)
    @test after - before ≈ 0.1 rtol = 1e-6
end

@testset "Reporting primary clock selection" begin
    layout = TL.NavFilterLayout((GPSL1CA(), GalileoE1B()))
    gps_clock, gal_clock = layout.clock_bias_index_by_group
    gps(prn) = _test_member(; prn, group = 1, clock_bias_index = gps_clock)
    e1b(prn) = _test_member(; prn, group = 2, signal = GalileoE1B(), clock_bias_index = gal_clock)
    both = [gps(1), gps(2), e1b(3)]

    # Kept while its own time system still contributes a measurement — no flicker.
    @test TL.report_primary_clock_index(layout, both, 1:3, gal_clock) == gal_clock
    @test TL.report_primary_clock_index(layout, both, 1:3, gps_clock) == gps_clock
    # Re-picked when the current primary's system is absent (index 0 is never
    # present): GPST is preferred when GPS is in the fix.
    @test TL.report_primary_clock_index(layout, both, 1:3, 0) == gps_clock
    # Else the most-populated time system (GPS absent ⇒ Galileo).
    @test TL.report_primary_clock_index(layout, [e1b(2), e1b(3)], 1:2, gps_clock) == gal_clock
    # No included measurements at all: keep whatever it was.
    @test TL.report_primary_clock_index(layout, both, Int[], gps_clock) == gps_clock
end

@testset "NCO loop-closure corrections" begin
    member = _test_member(; prn = 7, pseudorange_rate = 100.0)
    c = SPEED_OF_LIGHT
    # A predicted pseudorange 10 m beyond the measured one must speed the code
    # NCO up by f_code · 10 m / (T · c): the replica is 10 m late. A predicted
    # rate 5 m/s above the measured one maps to +5/λ Hz on the carrier NCO.
    @test TL.nco_code_correction(1000.0 + 10.0, 1000.0, member.code_frequency, 0.1) ≈
          -10.0 * member.code_frequency / (0.1 * c)
    @test TL.nco_carrier_correction(105.0, member.pseudorange_rate, member.wavelength) ≈
          5.0 / member.wavelength

    # The prediction is evaluated at the updated state, so it *is* the whole
    # correction — the measurement update's state correction must not be added
    # on top of it a second time. A satellite the filter moved 10 m towards
    # therefore gets the same single-counted NCO update as above, not double.
    layout = TL.NavFilterLayout((GPSL1CA(),))
    idxs = TL.NavFilterIndices(VectorTracking(), layout)
    user_pos = SVector(6.378e6, 0.0, 0.0)
    x = zeros(8)
    x[idxs.pos] = user_pos
    x_updated = copy(x)
    x_updated[idxs.pos] = user_pos .+ [10.0, 0.0, 0.0] # +x is towards the satellite
    columns = TL.vt_bias_columns([member], layout)
    predict(x) = calc_ρ_hat!(zeros(1), [member.sat_position], TL.position_and_bias_vector(x, idxs), columns)[1]
    @test predict(x_updated) - predict(x) ≈ -10.0 rtol = 1e-3
    @test TL.nco_code_correction(predict(x_updated), predict(x), member.code_frequency, 0.1) ≈
          10.0 * member.code_frequency / (0.1 * c) rtol = 1e-3
end

@testset "Post-fit pseudorange and range-rate residuals" begin
    # Two members: one whose replica the loop has steered onto the navigation solution
    # (the model at the updated state reproduces both its measured pseudorange and its
    # replica's own Doppler), and one the solution predicts 30 m short and 2 m/s slow of
    # what it measures.
    steered = _test_member(;
        prn = 4,
        pseudorange_rate = 800.0,
        code_discriminator = 0.02,   # chips
        carrier_discriminator = 5.0, # Hz
    )
    diverged = _test_member(; prn = 9, pseudorange_rate = -600.0)
    residual, rate_residual = TL.vt_post_fit_residuals(steered, 2.05e7, 2.05e7, steered.pseudorange_rate)
    # Both go straight into `SatInfo`'s unit-typed fields, so the units are part of the
    # contract: metres and metres per second.
    @test residual isa typeof(1.0u"m")
    @test rate_residual isa typeof(1.0u"m/s")
    # The steered member's residuals are its discriminators alone — chips × chip
    # length and Hz × wavelength — which is what makes them the vector loop's
    # per-satellite tracking-error readout.
    @test ustrip(u"m", residual) ≈ 0.02 * steered.chip_length
    @test ustrip(u"m/s", rate_residual) ≈ -5.0 * steered.wavelength
    # The other member keeps the full disagreement. Both are observed − computed in
    # `calc_pvt`'s conventions, and its rate observable (the geometric range rate) runs
    # opposite to this loop's Doppler-signed one, so a measurement beyond the prediction
    # reads positive in the range domain and negative in the rate domain.
    residual, rate_residual =
        TL.vt_post_fit_residuals(diverged, 2.10e7, 2.10e7 - 30.0, diverged.pseudorange_rate - 2.0)
    @test ustrip(u"m", residual) ≈ 30.0
    @test ustrip(u"m/s", rate_residual) ≈ -2.0
end

@testset "FLL discriminator sign into the rate measurement" begin
    # `fll_disc` measures f_incoming − f_replica, so a positive mean residual must
    # ADD to the pseudorange-rate measurement (`λ · carrier_doppler + λ · mean_fll`);
    # the discriminator enters with its own sign, not flipped. Pins the sign so a
    # re-introduced negation is caught.
    base = init_estimator_state(VectorPLLAndDLL(GPSL1CA()), GPSL1CA(), 20.0Hz, 0.0Hz)
    # Mean = sum / count: (2, 6 Hz) → +3 Hz, (1, −4 Hz) → −4 Hz.
    pos = SatVectorPLLAndDLL(base; carrier_discr_acc = (2, 6.0Hz))
    neg = SatVectorPLLAndDLL(base; carrier_discr_acc = (1, -4.0Hz))
    empty_acc = SatVectorPLLAndDLL(base; carrier_discr_acc = (0, 0.0Hz))
    @test TL.accumulated_carrier_discriminator(pos) ≈ 3.0
    @test TL.accumulated_carrier_discriminator(neg) ≈ -4.0
    @test TL.accumulated_carrier_discriminator(empty_acc) == 0.0
    # The DLL mean enters negated.
    @test TL.accumulated_code_discriminator(SatVectorPLLAndDLL(base; code_discr_acc = (2, 0.1)), 0.1) ≈ -0.05
end

@testset "The code measurement is moved from mid-cycle to the epoch" begin
    base = init_estimator_state(VectorPLLAndDLL(GPSL1CA()), GPSL1CA(), 20.0Hz, 0.0Hz)
    T = 0.1
    # Without an NCO delay the advance is the correction's `c·T/2`, bit for bit.
    for c_new in (0.37, -2.5, 1e-3), c_old in (0.0, 4.0, -1.1)
        state = TL._set_vector_corrections(TL._set_vector_corrections(base, c_old * Hz, 0.0Hz), c_new * Hz, 0.0Hz)
        @test state.code_freq_update_history == (c_new * Hz, c_old * Hz, 0.0Hz)
        @test TL.code_phase_advance(state, T) === c_new * T / 2
        state = SatVectorPLLAndDLL(state; code_discr_acc = (4, 0.2))
        @test TL.accumulated_code_discriminator(state, T) === -0.05 + c_new * T / 2
    end
    # Under an NCO delay the second half-cycle ran on the corrections in effect before
    # the newest landed: on `c₁` while the newest lands inside it or after the epoch, and
    # on `c₂` while `c₁` itself lands only after mid-cycle (`L > 1.5T`).
    c₂, c₁, c₀ = 0.5, 1.0, 2.0
    older = TL._set_vector_corrections(TL._set_vector_corrections(base, c₂ * Hz, 0.0Hz), c₁ * Hz, 0.0Hz)
    for (L, expected) in (
        (0.03, c₀ * 0.05),
        (0.075, c₀ * 0.025 + c₁ * 0.025),
        (0.1, c₁ * 0.05),
        (0.15, c₁ * 0.05),
        (0.175, c₁ * 0.025 + c₂ * 0.025),
        (0.2, c₂ * 0.05),
        (0.25, c₂ * 0.05),
    )
        state = TL._set_vector_corrections(older, c₀ * Hz, 0.0Hz, L * s)
        @test TL.code_phase_advance(state, T) ≈ expected
    end
end

@testset "A cycle without an accumulated discriminator withholds the member" begin
    # The `accumulated_*` helpers substitute a zero when nothing was accumulated, which the
    # navigation filter cannot distinguish from a genuine zero residual measured at full
    # weight. `has_accumulated_discriminators` is the guard that keeps such a member out of
    # the measurement set.
    base = init_estimator_state(VectorPLLAndDLL(GPSL1CA()), GPSL1CA(), 20.0Hz, 0.0Hz)
    accumulated = SatVectorPLLAndDLL(base; code_discr_acc = (3, 0.06), carrier_discr_acc = (3, 6.0Hz))
    nothing_accumulated = SatVectorPLLAndDLL(base; code_discr_acc = (0, 0.0), carrier_discr_acc = (0, 0.0Hz))
    @test TL.has_accumulated_discriminators(accumulated)
    @test !TL.has_accumulated_discriminators(nothing_accumulated)
    # The zero the helpers would otherwise hand to the filter is indistinguishable from a
    # measured zero — which is exactly why the guard is needed rather than the fallback.
    @test TL.accumulated_carrier_discriminator(nothing_accumulated) == 0.0
    @test TL.accumulated_code_discriminator(nothing_accumulated, 0.1) == 0.0
end

@testset "Pseudorange differencing across a GNSS week rollover" begin
    c = SPEED_OF_LIGHT
    # Receive just after the week rollover (TOW 0.05 s), transmit just before it
    # (TOW 604799.97 s): the 0.08 s light-travel difference straddles 604800 s and
    # must fold back rather than read as a ~604800 s (≈1.8e11 m) pseudorange.
    @test TL.pseudorange_from_tows(0.05, 604799.97) ≈ 0.08 * c rtol = 1e-6
    # Mid-week (no wrap) is unaffected.
    @test TL.pseudorange_from_tows(100.08, 100.0) ≈ 0.08 * c rtol = 1e-6
end

@testset "Initial navigation state from a scalar fix" begin
    layout = TL.NavFilterLayout((GPSL1CA(), GalileoE1B(), GPSL5I()))
    config = VectorTracking()
    idxs = TL.NavFilterIndices(config, layout)
    n = TL.num_nav_states(config, layout)
    pvt = PVTSolution(;
        position = ECEF(4.0e6, 4.0e5, 4.9e6),
        velocity = ECEF(1.0, 2.0, 3.0),
        relative_clock_drift = 1e-7,
        reference_system = GST(),
        inter_system_biases = Dict{PositionVelocityTime.SupportedTimeSystem,typeof(1.0u"m")}(GPST() => 12.0u"m"),
        inter_frequency_biases = Dict(:L5 => InterFrequencyBias(3.0u"m", :L1)),
    )
    x, P = zeros(n), zeros(n, n)
    primary = TL.initial_nav_state!(x, P, layout, idxs, pvt)
    @test primary == 2 # Galileo's clock
    @test TL.nav_filter_states(x, idxs)[1] == [4.0e6, 4.0e5, 4.9e6]
    @test TL.nav_filter_states(x, idxs)[2] == [1.0, 2.0, 3.0]
    @test x[idxs.clock_drift] ≈ 1e-7 * SPEED_OF_LIGHT
    @test x[idxs.clock_biases] == [12.0, 0.0]
    @test P[idxs.clock_biases[1], idxs.clock_biases[1]] == P[idxs.clock_biases[2], idxs.clock_biases[2]]
    if !isempty(idxs.ifb)
        # Seeded only when measured against the layout's own reference band.
        seeded = layout.reference_bands[1] == :L1
        @test x[idxs.ifb[1]] == (seeded ? 3.0 : 0.0)
    end
    # An unseen system starts with a generous clock variance.
    layout3 = TL.NavFilterLayout((GPSL1CA(), GalileoE1B(), BeiDouB1I()))
    idxs3 = TL.NavFilterIndices(config, layout3)
    n3 = TL.num_nav_states(config, layout3)
    x3, P3 = zeros(n3), zeros(n3, n3)
    TL.initial_nav_state!(x3, P3, layout3, idxs3, pvt)
    @test P3[idxs3.clock_biases[3], idxs3.clock_biases[3]] == 100.0^2
    # A band the fix did not measure starts from zero with the generous RF-chain variance.
    pvt_l1 = @set pvt.inter_frequency_biases = Dict{Symbol,InterFrequencyBias}()
    TL.initial_nav_state!(x, P, layout, idxs, pvt_l1)
    @test x[idxs.ifb[1]] == 0.0
    @test P[idxs.ifb[1], idxs.ifb[1]] == 30.0^2
    # A third-order motion model starts with zero acceleration, of its own variance.
    config3 = VectorTracking(; motion_model_order = 3)
    idxs_acc = TL.NavFilterIndices(config3, layout)
    n_acc = TL.num_nav_states(config3, layout)
    x_acc, P_acc = zeros(n_acc), zeros(n_acc, n_acc)
    TL.initial_nav_state!(x_acc, P_acc, layout, idxs_acc, pvt)
    @test length(idxs_acc.acc) == 3
    @test all(iszero, x_acc[idxs_acc.acc])
    @test all(a -> P_acc[a, a] > 0, idxs_acc.acc)
    @test x_acc[idxs_acc.pos] == x[idxs.pos]
    @test x_acc[idxs_acc.vel] == x[idxs.vel]
end

@testset "The seeded covariance is the fix's, scaled by its geometry" begin
    layout = TL.NavFilterLayout((GPSL1CA(), GalileoE1B()))
    config = VectorTracking()
    idxs = TL.NavFilterIndices(config, layout)
    n = TL.num_nav_states(config, layout)
    user = SVector(6.378e6, 0.0, 0.0)
    # Four GPS satellites and one Galileo satellite, the Galileo clock determined by its
    # lone satellite alone — the inter-system bias is as wrong as the position along it.
    directions = [(1.0, 0.0, 0.0), (0.6, 0.8, 0.0), (0.6, -0.4, 0.7), (0.5, -0.3, -0.8), (0.7, 0.5, 0.5)]
    members = [
        _test_member(; group = k == 5 ? 2 : 1, slot = k, prn = k, clock_bias_index = k == 5 ? 2 : 1,
            sat_position = user + 2.0e7 * SVector(d) / norm(SVector(d))) for (k, d) in enumerate(directions)
    ]
    positions = [member.sat_position for member in members]
    gst_bias = Dict{PositionVelocityTime.SupportedTimeSystem,typeof(1.0u"m")}(GST() => 2.0u"m")
    pvt = PVTSolution(; position = ECEF(user...), reference_system = GPST(), inter_system_biases = gst_bias)
    clock_used, ifb_used = zeros(Int, 2), zeros(Int, 0)
    columns, primary = TL.dense_bias_columns!(Int[], Int[], clock_used, ifb_used, members, eachindex(members), 1)
    H = calc_H!(zeros(5, 5), positions, [user..., 0.0, 0.0], columns)
    dop = calc_DOP!(zeros(5, 5), H, ECEF(user...), primary)

    x, P = zeros(n), zeros(n, n)
    TL.initial_nav_state!(x, P, layout, idxs, pvt)
    seeded = copy(P)
    @test TL.seed_fix_covariance!(P, idxs, layout, pvt, 1, H, zeros(5, 5), clock_used, ifb_used)
    block = [idxs.pos; idxs.clock_biases]
    @test P[block, block] ≈ TL.FIX_PSEUDORANGE_STD^2 * inv(H' * H)
    @test TL.position_uncertainty(P, idxs) ≈ TL.FIX_PSEUDORANGE_STD * dop.PDOP
    @test P ≈ P'
    @test isposdef(Symmetric(P))
    @test P[idxs.clock_biases[2], idxs.pos[1]] != 0
    # Velocity and clock drift keep their seed.
    others = setdiff(1:n, block)
    @test P[others, others] == seeded[others, others]
    @test all(iszero, P[others, block])

    # A Galileo clock the fix did not seed keeps its generous, uncorrelated variance; the
    # position is still uncertain by the geometry that leaves that clock unknown.
    pvt_gps = @set pvt.inter_system_biases = empty(gst_bias)
    x2, P2 = zeros(n), zeros(n, n)
    TL.initial_nav_state!(x2, P2, layout, idxs, pvt_gps)
    @test TL.seed_fix_covariance!(P2, idxs, layout, pvt_gps, 1, H, zeros(5, 5), clock_used, ifb_used)
    gal = idxs.clock_biases[2]
    @test P2[gal, gal] == 100.0^2
    @test all(iszero, P2[gal, setdiff(1:n, gal)])
    @test P2[idxs.pos, idxs.pos] ≈ P[idxs.pos, idxs.pos]

    # A rank-deficient design leaves the seed as it was.
    P3 = copy(seeded)
    H_degenerate = calc_H!(zeros(5, 5), fill(positions[1], 5), [user..., 0.0, 0.0], columns)
    @test !TL.seed_fix_covariance!(P3, idxs, layout, pvt, 1, H_degenerate, zeros(5, 5), clock_used, ifb_used)
    @test P3 == seeded
end

@testset "A singular design the factorization lets through leaves the seed as it was" begin
    # Galileo seen on E5a alone: the L5 inter-frequency-bias column duplicates the
    # Galileo clock column, so `HᵀH` is singular. For this geometry rounding can leave its
    # last Cholesky pivot slightly positive rather than failing the factorization, and
    # the inverse would seed the Galileo clock with a variance of about 10¹⁵ m².
    layout = TL.NavFilterLayout((GPSL1CA(), GalileoE1B(), GalileoE5aI()))
    config = VectorTracking()
    idxs = TL.NavFilterIndices(config, layout)
    n = TL.num_nav_states(config, layout)
    user = SVector(6.378e6, 0.0, 0.0)
    directions = [(1.0, 0.0, 0.0), (0.6, 0.8, 0.0), (0.6, -0.4, 0.7), (0.5, -0.3, -0.8), (0.7, 0.5, 0.5), (0.4, -0.7, -0.6)]
    members = [
        _test_member(; group = k <= 4 ? 1 : 3, slot = k, prn = k, clock_bias_index = k <= 4 ? 1 : 2,
            ifb_index = k <= 4 ? 0 : 1, signal = k <= 4 ? GPSL1CA() : GalileoE5aI(),
            sat_position = user + 2.0e7 * SVector(d) / norm(SVector(d))) for (k, d) in enumerate(directions)
    ]
    positions = [member.sat_position for member in members]
    gst_bias = Dict{PositionVelocityTime.SupportedTimeSystem,typeof(1.0u"m")}(GST() => 2.0u"m")
    pvt = PVTSolution(; position = ECEF(user...), reference_system = GPST(), inter_system_biases = gst_bias)
    clock_used, ifb_used = zeros(Int, 2), zeros(Int, 1)
    columns, _ = TL.dense_bias_columns!(Int[], Int[], clock_used, ifb_used, members, eachindex(members), 1)
    H = calc_H!(zeros(6, 6), positions, [user..., 0.0, 0.0, 0.0], columns)
    x, P = zeros(n), zeros(n, n)
    TL.initial_nav_state!(x, P, layout, idxs, pvt)
    seeded = copy(P)
    @test !TL.seed_fix_covariance!(P, idxs, layout, pvt, 1, H, zeros(6, 6), clock_used, ifb_used)
    @test P == seeded
end
