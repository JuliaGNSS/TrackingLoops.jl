# The navigation filter's model, ported from GNSSReceiver's `test/vector_tracking.jl`:
# the configuration, the layout and state indices, the process model, the measurement
# model and its noise, the bias observability and the loop-closure maths.
using LinearAlgebra: diag, eigvals, Symmetric, I, Diagonal, norm, isposdef
using Accessors: @set
using GNSSDecoder: GNSSDecoder, GNSSDecoderState
using Dictionaries: Dictionary
using PositionVelocityTime: PositionVelocityTime, SPEED_OF_LIGHT, BiasColumns, predict_pseudoranges!,
    calc_design_matrix!, calc_dop!, has_broadcast_time_offset,
    PVTSolution, InterFrequencyBias,
    SignalGroup, SatelliteMeasurement
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
    sat_pos = SVector(2.6e7, 0.0, 0.0),
    sat_vel = SVector(0.0, 0.0, 0.0),
    sat_clock_drift = 0.0,
    code_discriminator = 0.0,
    pseudorange_rate = 0.0,
    cn0_lin = 10^4.5, # 45 dB-Hz
    early_late_spacing = 1.0,
    coherent_integration_time = nothing,
)
    c = SPEED_OF_LIGHT
    code_frequency = Float64(ustrip(Hz, get_chip_rate(signal)))
    tcoh = something(coherent_integration_time, get_code_length(signal) / code_frequency)
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
        # The same instant on the GPS Time count: differs from `time` by the leap
        # seconds between the two (0 for GPST/GST, −14 s for BDT).
        time - ustrip(s, get_leap_second_offset(GPST(), get_time_system(signal))),
        sat_pos,
        sat_vel,
        sat_clock_drift,
        code_discriminator,
        pseudorange_rate,
        cn0_lin,
        early_late_spacing,
        tcoh,
    )
end

# A measurement for the bias-layout and reference-clock tests, which read only its
# classification: the PRN, the signal's time system and band, and `decoder`'s broadcast
# offsets, and the DOP and seed tests its position. The rest are placeholders.
function _test_measurement(;
    prn = 1,
    signal = GPSL1CA(),
    time = 0.0,
    decoder = nothing,
    sat_pos = SVector(2.6e7, 0.0, 0.0),
)
    time_system = get_time_system(signal)
    time_offset(target) =
        isnothing(decoder) ? nothing : GNSSDecoder.get_broadcast_time_offset(decoder, target, time)
    SatelliteMeasurement(
        prn,
        get_signal_id(signal),
        get_band_id(signal),
        time_system,
        time,
        sat_pos,
        SVector(0.0, 0.0, 0.0),
        0.0,
        Float64(ustrip(Hz, get_center_frequency(signal))),
        0.0,
        0,
        time_offset(GPST()),
        time_offset(GST()),
        time_offset(BDT()),
    )
end

@testset "Vector-loop times difference on the GPS Time count" begin
    # One physical instant: a BDT seconds-of-week reads 14 s below the GPS time
    # of week (GPST is TAI−19, BDT is TAI−33). Everything the vector loop
    # *differences* across members — `reference_time`, the pseudoranges — must
    # therefore use `time_gpst_count`, not `time`; mixing raw counts hands every
    # BeiDou member 14 s × c ≈ 4.2×10⁹ m of structural pseudorange offset that
    # the ns-scale BGTO constraint then fights rather than absorbs.
    gps = _test_member(; time = 100.0)
    bds = _test_member(; group = 2, signal = BeiDouB2aI(), clock_bias_index = 2, time = 86.0)
    @test gps.time_gpst_count == 100.0
    @test bds.time_gpst_count ≈ gps.time_gpst_count
    # The own-scale transmit time stays untouched — it is what the ephemeris,
    # the clock polynomial and the broadcast time offsets are evaluated at.
    @test bds.time == 86.0
end

@testset "NavigationFilterConfig configuration" begin
    config = NavigationFilterConfig()
    @test config.use_pseudorange_rates # VDFLL by default
    @test config.motion_model_order == 2
    @test config.clock_model_order == 2
    @test config.insufficient_measurements_timeout == 10.0s

    vdll = NavigationFilterConfig(use_pseudorange_rates = false)
    @test !vdll.use_pseudorange_rates

    @test_throws ArgumentError NavigationFilterConfig(motion_model_order = 0)
    @test_throws ArgumentError NavigationFilterConfig(motion_model_order = 4)
    @test_throws ArgumentError NavigationFilterConfig(clock_model_order = 0)
    @test_throws ArgumentError NavigationFilterConfig(clock_model_order = 3)
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
    config = NavigationFilterConfig()
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
        NavigationFilterConfig(; motion_model_order = 1, clock_model_order = 1),
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
        NavigationFilterConfig(; ifb_noise_density = 0.05u"m/sqrt(s)"),
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
    TL.ensure_nav_filter_cycle_time!(nav, config, 100.0ms)
    @test nav.Q == Q
    TL.ensure_nav_filter_cycle_time!(nav, config, 0.104s)
    @test nav.cycle_time == 0.104s
    @test nav.F[1, 2] == 0.104
    @test nav.Q == TL.NavFilterModel(config, TL.NavFilterLayout((GPSL1CA(),)), 0.104s).Q
    TL.ensure_nav_filter_cycle_time!(nav, config, 0.2s)
    @test nav.cycle_time == 0.2s
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
    config(order) = NavigationFilterConfig(; motion_model_order = order, acceleration_noise_std = acc)
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
    config = NavigationFilterConfig()
    layout = TL.NavFilterLayout((GPSL1CA(), GalileoE1B()))
    nav = TL.NavFilterModel(config, layout, 100.0ms)
    idxs = nav.idxs

    # One GPS satellite along +x, one Galileo along +z: each pseudorange
    # carries its own system's clock bias.
    members = [
        _test_member(; prn = 1, clock_bias_index = 1, sat_pos = SVector(2.6e7, 0.0, 0.0)),
        _test_member(;
            group = 2,
            prn = 2,
            clock_bias_index = 2,
            signal = GalileoE1B(),
            sat_pos = SVector(0.0, 0.0, 2.6e7),
        ),
    ]
    bias_columns = TL.member_bias_columns(members, layout)
    @test bias_columns.clock_columns == [1, 2]
    @test bias_columns.num_clock_biases == 2
    @test bias_columns.num_ifbs == 0
    @test TL.member_bias_columns(members, layout, [2]).clock_columns == [2]

    user_pos = SVector(6.378e6, 0.0, 0.0)
    x = zeros(9)
    x[idxs.pos] = user_pos
    x[idxs.clock_biases] = [100.0, 200.0]
    ξ = TL.position_and_bias_vector(x, idxs)
    positions = [member.sat_pos for member in members]

    psr = predict_pseudoranges!(zeros(2), positions, ξ, bias_columns)
    @test psr[1] ≈ 2.6e7 - 6.378e6 + 100.0
    @test psr[2] ≈ norm(user_pos - [0.0, 0.0, 2.6e7]) + 200.0 rtol = 1e-5

    # The prediction is Doppler-signed: a satellite closing head-on with 1000 m/s reads
    # +1000 m/s (scaled by the light time, a receiver second spanning `1 + 1000 / c`
    # seconds of its transmission), the negative of `calc_pseudorange_rate`.
    closing = _test_member(; sat_pos = SVector(2.6e7, 0.0, 0.0),
        sat_vel = SVector(-1000.0, 0.0, 0.0))
    @test TL.predict_pseudorange_rate(x, idxs, closing) ≈
          1000.0 * (1 + 1000.0 / SPEED_OF_LIGHT) atol = 1e-6

    # The rate is predicted at the epoch, from the state as it stands, whatever the
    # motion model.
    sat_pos = SVector(2.6e7, 0.0, 0.0)
    user_pos = SVector(6.378e6, 0.0, 0.0)
    for order in (1, 2, 3)
        idxs = TL.NavFilterIndices(NavigationFilterConfig(; motion_model_order = order), layout)
        x = zeros(TL.num_nav_states(NavigationFilterConfig(; motion_model_order = order), layout))
        x[idxs.pos] = user_pos
        order >= 2 && (x[idxs.vel] = [3.0, 4.0, 0.0])
        order >= 3 && (x[idxs.acc] = [5.0, 0.0, 0.0])
        x[idxs.clock_drift] = 10.0
        member = _test_member(; sat_pos,
            sat_vel = SVector(-1000.0, 0.0, 0.0), sat_clock_drift = 1e-9)
        pos, vel, drift = TL.nav_filter_states(x, idxs)
        @test TL.predict_pseudorange_rate(x, idxs, member) ==
              -TL.calc_pseudorange_rate(pos, vel, drift, member.sat_pos,
                  member.sat_vel, member.sat_clock_drift)
    end
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
        squaring_loss = 1 + 2 / ((2 - d) * m.cn0_lin * m.coherent_integration_time)
        d / (4 * T * m.cn0_lin) * squaring_loss * m.chip_length^2
    end
    # Mean-FLL variance 2·σ_φ²/(2π·T)² with σ_φ² = 1/(2·C/N0·T_coh)(1+1/(2·C/N0·T_coh)),
    # expressed as a rate (λ²). No inflation factor: the derived single-cycle variance is the
    # right one for this estimator, and the lag-1 correlation the telescoping creates makes a
    # white-noise filter pessimistic rather than overconfident.
    rate_var(m) = let ct = m.cn0_lin * m.coherent_integration_time
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
    weak = _test_member(; cn0_lin = 10^2.0) # 20 dB-Hz
    weak_loss = 1 + 2 / ((2 - weak.early_late_spacing) * weak.cn0_lin * weak.coherent_integration_time)
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

@testset "A cycle's bias layout and its broadcast time offsets" begin
    # The decision itself is `decide_bias_layout!`'s; these check what the vector loop
    # makes of it: the candidate subset, and its constraints on the filter's clock states.
    function assess(layout, measurements, candidates)
        bias_layout = TL.BiasLayout()
        solvable = TL.decide_bias_layout!(bias_layout, view(measurements, candidates))
        constraints = TL.time_offset_constraints!(Tuple{Int,Int,Float64}[], layout, bias_layout)
        solvable, constraints
    end
    layout = TL.NavFilterLayout((GPSL1CA(), GalileoE1B(), GalileoE5aI()))
    gps_clock = layout.clock_bias_index_by_group[1]
    gal_clock = layout.clock_bias_index_by_group[2]
    gps(prn) = _test_measurement(; prn)
    e1b(prn) = _test_measurement(; prn, signal = GalileoE1B())
    e5a(prn) = _test_measurement(; prn, signal = GalileoE5aI())

    # Only the candidates count.
    @test assess(layout, [gps(i) for i = 1:4], 1:4) == (true, [])
    @test assess(layout, [gps(i) for i = 1:4], 1:3) == (false, [])

    # Five measurements are one short of 3 + 2 clocks + 1 IFB, and without a decoded GGTO
    # nothing can tie the two clocks.
    measurements = [gps(1), gps(2), gps(3), e1b(4), e5a(5)]
    @test assess(layout, measurements, 1:5) == (false, [])

    # With the GGTO decoded, the Galileo clock is tied to the GPS one by the
    # pseudo-measurement `clk_GST - clk_GPST = -c * GGTO`.
    ggto_decoder = with_ggto(GNSSDecoderState(GalileoE1B(), 4), 1e-9)
    measurements[4] = _test_measurement(; prn = 4, signal = GalileoE1B(), decoder = ggto_decoder)
    @test has_broadcast_time_offset(measurements[4], GPST())
    solvable, constraints = assess(layout, measurements, 1:5)
    @test solvable
    gst_state, gpst_state, isb = only(constraints)
    @test gst_state == gal_clock
    @test gpst_state == gps_clock
    @test isb ≈ -SPEED_OF_LIGHT * 1e-9

    # A mixed epoch ties each non-GPS clock through its own broadcast offset, Galileo
    # through the GGTO and BeiDou through the BGTO, in one cycle: B1I is on its own band,
    # and only both offsets together connect the coverage graph.
    tri = TL.NavFilterLayout((GPSL1CA(), GalileoE1B(), BeiDouB1I()))
    gps_tri, gal_tri, bds_tri = tri.clock_bias_index_by_group
    gal_measurement = _test_measurement(;
        prn = 4,
        signal = GalileoE1B(),
        decoder = with_ggto(GNSSDecoderState(GalileoE1B(), 4), 1e-9),
    )
    bds_decoder = with_bgto(GNSSDecoderState(BeiDouB1I(), 21), 3e-9)
    bds_measurement = _test_measurement(; prn = 21, signal = BeiDouB1I(), decoder = bds_decoder)
    @test has_broadcast_time_offset(bds_measurement, GPST())
    solvable, constraints = assess(tri, [gps(1), gps(2), gps(3), gal_measurement, bds_measurement], 1:5)
    @test solvable
    @test length(constraints) == 2
    by_state = Dict(state => (ref, isb) for (state, ref, isb) in constraints)
    @test by_state[gal_tri][1] == gps_tri
    @test by_state[gal_tri][2] ≈ -SPEED_OF_LIGHT * 1e-9
    @test by_state[bds_tri][1] == gps_tri
    @test by_state[bds_tri][2] ≈ -SPEED_OF_LIGHT * 3e-9

    # Without GPS, BeiDou's offset toward Galileo System Time ties the two.
    duo = TL.NavFilterLayout((GalileoE1B(), BeiDouB1I()))
    gal_duo, bds_duo = duo.clock_bias_index_by_group
    bds_gal_decoder = with_gal_bgto(GNSSDecoderState(BeiDouB1I(), 22), 4e-9)
    bds_gal_measurement = _test_measurement(; prn = 22, signal = BeiDouB1I(), decoder = bds_gal_decoder)
    @test has_broadcast_time_offset(bds_gal_measurement, GST())
    measurements = [e1b(1), e1b(2), e1b(3), e1b(4), bds_gal_measurement]
    solvable, constraints = assess(duo, measurements, 1:5)
    @test solvable
    gst_constraint = only(constraints)
    @test gst_constraint[1] == bds_duo
    @test gst_constraint[2] == gal_duo
    @test gst_constraint[3] ≈ -SPEED_OF_LIGHT * 4e-9
    # An epoch that even the constraint leaves short is unsolvable, but keeps it, so the
    # clocks stay tied.
    @test assess(duo, measurements, 2:5) == (false, [gst_constraint])

    # The layout is reused from cycle to cycle, and the constraints follow the latest.
    bias_layout = TL.BiasLayout()
    constraints = Tuple{Int,Int,Float64}[]
    function cycle_constraints(candidates)
        TL.decide_bias_layout!(bias_layout, view(measurements, candidates))
        TL.time_offset_constraints!(constraints, duo, bias_layout)
    end
    @test length(cycle_constraints(1:5)) == 1
    # A cycle without candidates is decided too, with no constraint left over.
    @test isempty(cycle_constraints(1:0))
    # Without a broadcast offset there is nothing to keep: the E1B satellites alone.
    @test isempty(cycle_constraints(1:3))

    # The outage of the review: two GPS satellites and one Galileo satellite with a decoded
    # GGTO cannot determine the state even with the clocks tied, but the GGTO still ties
    # them.
    pair = TL.NavFilterLayout((GPSL1CA(), GalileoE1B()))
    gps_pair, gal_pair = pair.clock_bias_index_by_group
    outage = [gps(1), gps(2), _test_measurement(; prn = 3, signal = GalileoE1B(), decoder = ggto_decoder)]
    solvable, constraints = assess(pair, outage, 1:3)
    @test !solvable
    clock, target_clock, isb = only(constraints)
    @test (clock, target_clock) == (gal_pair, gps_pair)
    @test isb ≈ -SPEED_OF_LIGHT * 1e-9

    # The position uncertainty reported with the solution is the root sum of the position
    # variances, read off the position block of P.
    idxs = TL.NavFilterIndices(NavigationFilterConfig(), layout)
    P = zeros(10, 10)
    P[idxs.pos, idxs.pos] = Diagonal([9.0, 16.0, 144.0])
    @test TL.position_uncertainty(P, idxs) ≈ sqrt(169.0)
end

@testset "The DOP's clock column of the reference system" begin
    # The DOP takes the cycle's decided columns, numbered in order of first appearance:
    # Galileo's clock first here.
    layout = TL.NavFilterLayout((GPSL1CA(), GalileoE1B(), GalileoE5aI()))
    measurements = [
        [_test_measurement(; prn, signal = GalileoE1B()) for prn = 1:3]
        [_test_measurement(; prn) for prn = 4:6]
    ]
    bias_layout = TL.BiasLayout()
    function column(time_system, candidates)
        TL.decide_bias_layout!(bias_layout, view(measurements, candidates))
        TL.clock_column(bias_layout, view(measurements, candidates), time_system)
    end
    @test column(GST(), 1:6) == 1
    @test column(GPST(), 1:6) == 2
    # Systems a constraint ties together share their group's column.
    measurements[1] = _test_measurement(;
        prn = 1,
        signal = GalileoE1B(),
        decoder = with_ggto(GNSSDecoderState(GalileoE1B(), 1), 1e-9),
    )
    @test column(GST(), [1, 4, 5, 6]) == column(GPST(), [1, 4, 5, 6]) == 1
    # A system without a measurement is a caller's error.
    @test_throws ArgumentError column(BDT(), 1:6)
end

@testset "The receive epoch crosses into the next week by itself" begin
    # The time of week the pseudoranges difference against is read off the receive epoch,
    # so one cycle before a week crossover and one after are one cycle apart, the time of
    # week starting over.
    before = get_week_start_epoch(GPST(), 2052) + (GNSSSignals.SECONDS_PER_WEEK - 0.05)
    after = before + 0.1
    @test get_time_of_week(GPST(), before) ≈ GNSSSignals.SECONDS_PER_WEEK - 0.05
    @test get_time_of_week(GPST(), after) ≈ 0.05
    @test after - before ≈ 0.1

    # A measurement's transmit time on the GPS Time count: a BDT count reads 14 s below.
    states = PositionVelocityTime._precompile_states(
        GPSL1CA(), PositionVelocityTime._PRECOMPILE_GPS_L1CA_STATES, identity, GPSL1CA())
    measurement = only(PositionVelocityTime.collect_measurements(
        SignalGroup(GPSL1CA(), states[1:1]);
        approximate_year = 2021,
    )[1])
    @test TL.calc_leap_second_correction(measurement, GPST()) == 0
    beidou = @set measurement.time_system = BDT()
    @test TL.calc_leap_second_correction(beidou, GPST()) == 14
end

@testset "Reporting reference clock selection" begin
    layout = TL.NavFilterLayout((GPSL1CA(), GalileoE1B(), BeiDouB1I()))
    gps_clock, gal_clock, bds_clock = layout.clock_bias_index_by_group
    gps(prn) = _test_measurement(; prn)
    e1b(prn) = _test_measurement(; prn, signal = GalileoE1B())
    galileo_heavy = [gps(1), e1b(2), e1b(3)]
    function select(measurements, current_index)
        bias_layout = TL.BiasLayout()
        TL.decide_bias_layout!(bias_layout, measurements)
        TL.select_reference_clock_index(layout, measurements, bias_layout, current_index)
    end

    # Kept while its own time system still contributes a measurement — no flicker.
    @test select(galileo_heavy, gps_clock) == gps_clock
    # Re-picked when its system is absent: the system with the most satellites…
    @test select(galileo_heavy, bds_clock) == gal_clock
    @test select([e1b(2), e1b(3)], gps_clock) == gal_clock
    # …ties going to the first to appear.
    @test select([e1b(2), gps(1)], bds_clock) == gal_clock
    @test select([gps(1), e1b(2)], bds_clock) == gps_clock
    # No included measurements at all: keep whatever it was.
    @test select(SatelliteMeasurement[], gps_clock) == gps_clock
end

@testset "NCO loop-closure corrections" begin
    member = _test_member(; prn = 7, pseudorange_rate = 100.0)
    c = SPEED_OF_LIGHT
    # A predicted pseudorange 10 m beyond the measured one must speed the code
    # NCO up by f_code · 10 m / (T · c): the replica is 10 m late. A predicted
    # rate of 105 m/s is a carrier Doppler of 105/λ Hz.
    @test TL.nco_code_correction(1000.0 + 10.0, 1000.0, member.code_frequency, 0.1) ≈
          -10.0 * member.code_frequency / (0.1 * c)
    @test TL.predicted_carrier_doppler(105.0, member.wavelength) ≈ 105.0 / member.wavelength

    # The prediction is evaluated at the updated state, so it *is* the whole
    # correction — the measurement update's state correction must not be added
    # on top of it a second time. A satellite the filter moved 10 m towards
    # therefore gets the same single-counted NCO update as above, not double.
    layout = TL.NavFilterLayout((GPSL1CA(),))
    idxs = TL.NavFilterIndices(NavigationFilterConfig(), layout)
    user_pos = SVector(6.378e6, 0.0, 0.0)
    x = zeros(8)
    x[idxs.pos] = user_pos
    x_updated = copy(x)
    x_updated[idxs.pos] = user_pos .+ [10.0, 0.0, 0.0] # +x is towards the satellite
    columns = TL.member_bias_columns([member], layout)
    predict(x) = predict_pseudoranges!(zeros(1), [member.sat_pos], TL.position_and_bias_vector(x, idxs), columns)[1]
    @test predict(x_updated) - predict(x) ≈ -10.0 rtol = 1e-3
    @test TL.nco_code_correction(predict(x_updated), predict(x), member.code_frequency, 0.1) ≈
          10.0 * member.code_frequency / (0.1 * c) rtol = 1e-3
end

@testset "Post-fit pseudorange and range-rate residuals" begin
    # Two members: one whose replica the loop has steered onto the navigation solution
    # (the model at the updated state reproduces both its measured pseudorange and its
    # replica's own Doppler, 800 m/s, against which its FLL reads 5 Hz), and one the
    # solution predicts 30 m short and 2 m/s slow of what it measures.
    steered = _test_member(; prn = 4, code_discriminator = 0.02) # chips
    steered = @set steered.pseudorange_rate = 800.0 + 5.0 * steered.wavelength
    diverged = _test_member(; prn = 9, pseudorange_rate = -600.0)
    residual, rate_residual = TL.post_fit_residuals(steered, 2.05e7, 2.05e7, 800.0)
    # Both go straight into `SatelliteInfo`'s unit-typed fields, so the units are part of the
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
        TL.post_fit_residuals(diverged, 2.10e7, 2.10e7 - 30.0, diverged.pseudorange_rate - 2.0)
    @test ustrip(u"m", residual) ≈ 30.0
    @test ustrip(u"m/s", rate_residual) ≈ -2.0
end

@testset "FLL discriminator sign into the rate measurement" begin
    # `fll_disc` measures f_incoming − f_replica, so a positive reading must ADD to the
    # replica Doppler the rate is measured from; the discriminator enters with its own
    # sign, not flipped. Pins the sign so a re-introduced negation is caught.
    base = init_estimator_state(VectorPLLAndDLL(GPSL1CA()), GPSL1CA(), 20.0Hz, 0.0Hz)
    pos = SatVectorPLLAndDLL(base; fll_disc_sum = (1, 3.0Hz))
    neg = SatVectorPLLAndDLL(base; fll_disc_sum = (2, -8.0Hz))
    empty_acc = SatVectorPLLAndDLL(base; fll_disc_sum = (0, 0.0Hz))
    @test TL.mean_fll_disc(pos) ≈ 3.0
    @test TL.mean_fll_disc(neg) ≈ -4.0
    @test TL.mean_fll_disc(empty_acc) == 0.0
    # The DLL mean enters negated.
    @test TL.mean_code_disc_at_epoch(SatVectorPLLAndDLL(base; code_disc_sum = (2, 0.1)), 0.1) ≈ -0.05
end

@testset "The code measurement is moved from mid-cycle to the epoch" begin
    base = init_estimator_state(VectorPLLAndDLL(GPSL1CA()), GPSL1CA(), 20.0Hz, 0.0Hz)
    T = 0.1
    # Without an NCO delay the advance is the correction's `c·T/2`, bit for bit.
    for c_new in (0.37, -2.5, 1e-3), c_old in (0.0, 4.0, -1.1)
        state = TL._set_vector_commands(TL._set_vector_commands(base, c_old * Hz, TL.CarrierRamp(20.0, 0.0, 0.0), 0.0s, FixedNCOWord(20.0, 0.0), 0), c_new * Hz, TL.CarrierRamp(20.0, 0.0, 0.0), 0.0s, FixedNCOWord(20.0, 0.0), 0)
        @test state.code_freq_update_history == (c_new * Hz, c_old * Hz, 0.0Hz)
        @test TL.code_phase_advance(state, T) === c_new * T / 2
        state = SatVectorPLLAndDLL(state; code_disc_sum = (4, 0.2))
        @test TL.mean_code_disc_at_epoch(state, T) === -0.05 + c_new * T / 2
    end
    # Under an NCO delay the second half-cycle ran on the corrections in effect before
    # the newest landed: on `c₁` while the newest lands inside it or after the epoch, and
    # on `c₂` while `c₁` itself lands only after mid-cycle (`L > 1.5T`).
    c₂, c₁, c₀ = 0.5, 1.0, 2.0
    older = TL._set_vector_commands(TL._set_vector_commands(base, c₂ * Hz, TL.CarrierRamp(20.0, 0.0, 0.0), 0.0s, FixedNCOWord(20.0, 0.0), 0), c₁ * Hz, TL.CarrierRamp(20.0, 0.0, 0.0), 0.0s, FixedNCOWord(20.0, 0.0), 0)
    for (L, expected) in (
        (0.03, c₀ * 0.05),
        (0.075, c₀ * 0.025 + c₁ * 0.025),
        (0.1, c₁ * 0.05),
        (0.15, c₁ * 0.05),
        (0.175, c₁ * 0.025 + c₂ * 0.025),
        (0.2, c₂ * 0.05),
        (0.25, c₂ * 0.05),
    )
        state = TL._set_vector_commands(older, c₀ * Hz, TL.CarrierRamp(20.0, 0.0, 0.0), L * s, FixedNCOWord(20.0, 0.0), 0)
        @test TL.code_phase_advance(state, T) ≈ expected
    end
end

@testset "A cycle without both a code and a carrier measurement withholds the member" begin
    # `mean_code_disc_at_epoch` and `mean_fll_disc` substitute a zero when nothing was
    # accumulated, which the navigation filter cannot distinguish from a genuine zero
    # residual measured at full weight. `has_disc_sums` is the guard that keeps such a
    # member out of the measurement set.
    base = init_estimator_state(VectorPLLAndDLL(GPSL1CA()), GPSL1CA(), 20.0Hz, 0.0Hz)
    accumulated = SatVectorPLLAndDLL(base; code_disc_sum = (3, 0.06),
        fll_disc_sum = (3, 6.0Hz))
    nothing_accumulated = SatVectorPLLAndDLL(base; code_disc_sum = (0, 0.0),
        fll_disc_sum = (0, 0.0Hz))
    @test TL.has_disc_sums(accumulated)
    @test !TL.has_disc_sums(nothing_accumulated)
    # A cycle without a frequency measurement — its only record the first after a
    # (re)start — has no rate to measure, and is withheld too.
    no_fll = SatVectorPLLAndDLL(base; code_disc_sum = (1, 0.02),
        fll_disc_sum = (0, 0.0Hz))
    @test !TL.has_disc_sums(no_fll)
    # The zero the helpers would otherwise hand to the filter is indistinguishable from a
    # measured zero — which is exactly why the guard is needed rather than the fallback.
    @test TL.mean_fll_disc(nothing_accumulated) == 0.0
    @test TL.mean_code_disc_at_epoch(nothing_accumulated, 0.1) == 0.0
end

@testset "Pseudorange differencing across a GNSS week rollover" begin
    c = SPEED_OF_LIGHT
    # Receive just after the week rollover (TOW 0.05 s), transmit just before it
    # (TOW 604799.97 s): the 0.08 s light-travel difference straddles 604800 s and
    # must fold back rather than read as a ~604800 s (≈1.8e11 m) pseudorange.
    @test TL.calc_pseudorange(0.05, 604799.97) ≈ 0.08 * c rtol = 1e-6
    # Mid-week (no wrap) is unaffected.
    @test TL.calc_pseudorange(100.08, 100.0) ≈ 0.08 * c rtol = 1e-6
end

@testset "Initial navigation state from a scalar fix" begin
    layout = TL.NavFilterLayout((GPSL1CA(), GalileoE1B(), GPSL5I()))
    config = NavigationFilterConfig()
    idxs = TL.NavFilterIndices(config, layout)
    n = TL.num_nav_states(config, layout)
    pvt = PVTSolution(;
        position = ECEF(4.0e6, 4.0e5, 4.9e6),
        velocity = ECEF(1.0, 2.0, 3.0),
        clock_drift = 1e-7 * SPEED_OF_LIGHT * u"m/s",
        reference_system = GST(),
        inter_system_biases = Dictionary{PositionVelocityTime.SupportedTimeSystem,typeof(1.0u"m")}([GPST()], [12.0u"m"]),
        inter_frequency_biases = Dictionary([:L5], [InterFrequencyBias(3.0u"m", :L1)]),
    )
    x, P = zeros(n), zeros(n, n)
    reference = TL.initial_nav_state!(x, P, layout, idxs, pvt)
    @test reference == 2 # Galileo's clock
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
    pvt_l1 = @set pvt.inter_frequency_biases = Dictionary{Symbol,InterFrequencyBias}()
    TL.initial_nav_state!(x, P, layout, idxs, pvt_l1)
    @test x[idxs.ifb[1]] == 0.0
    @test P[idxs.ifb[1], idxs.ifb[1]] == 30.0^2
    # A third-order motion model starts with zero acceleration, of its own variance.
    config3 = NavigationFilterConfig(; motion_model_order = 3)
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

# The fix's bias layout, as `_seed_fix_covariance!` decides it, and its design at `user`.
function _fix_design(layout, measurements, user)
    fix_bias_layout = TL.BiasLayout()
    @test TL.decide_bias_layout!(fix_bias_layout, measurements)
    columns = TL.fit_bias_columns(fix_bias_layout)
    num_columns = TL.num_parameters(columns)
    ξ = [user; zeros(num_columns - 3)]
    H = calc_design_matrix!(zeros(length(measurements), num_columns), [measurement.position for measurement in measurements], ξ, columns)
    fix_bias_layout, H
end

# Satellites 2·10⁷ m from `user` along `directions`.
_fix_positions(user, directions) = [user + 2.0e7 * SVector(d) / norm(SVector(d)) for d in directions]

@testset "The seeded covariance is the fix's, scaled by its geometry" begin
    layout = TL.NavFilterLayout((GPSL1CA(), GalileoE1B()))
    config = NavigationFilterConfig()
    idxs = TL.NavFilterIndices(config, layout)
    n = TL.num_nav_states(config, layout)
    user = SVector(6.378e6, 0.0, 0.0)
    # Four GPS satellites and one Galileo satellite, the Galileo clock determined by its
    # lone satellite alone — the inter-system bias is as wrong as the position along it.
    directions = [(1.0, 0.0, 0.0), (0.6, 0.8, 0.0), (0.6, -0.4, 0.7), (0.5, -0.3, -0.8), (0.7, 0.5, 0.5)]
    measurements = [
        _test_measurement(; prn = k, signal = k == 5 ? GalileoE1B() : GPSL1CA(), sat_pos) for
        (k, sat_pos) in enumerate(_fix_positions(user, directions))
    ]
    gst_bias = Dictionary{PositionVelocityTime.SupportedTimeSystem,typeof(1.0u"m")}([GST()], [2.0u"m"])
    pvt = PVTSolution(; position = ECEF(user...), reference_system = GPST(), inter_system_biases = gst_bias)
    fix_bias_layout, H = _fix_design(layout, measurements, user)
    dop = calc_dop!(zeros(5, 5), H, ECEF(user...), 1)

    x, P = zeros(n), zeros(n, n)
    TL.initial_nav_state!(x, P, layout, idxs, pvt)
    seeded = copy(P)
    @test TL.seed_fix_covariance!(P, idxs, layout, pvt, 1, H, zeros(5, 5), fix_bias_layout)
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
    @test TL.seed_fix_covariance!(P2, idxs, layout, pvt_gps, 1, H, zeros(5, 5), fix_bias_layout)
    gal = idxs.clock_biases[2]
    @test P2[gal, gal] == 100.0^2
    @test all(iszero, P2[gal, setdiff(1:n, gal)])
    @test P2[idxs.pos, idxs.pos] ≈ P[idxs.pos, idxs.pos]

    # A rank-deficient design leaves the seed as it was.
    P3 = copy(seeded)
    H_degenerate = calc_design_matrix!(zeros(5, 5), fill(measurements[1].position, 5), [user..., 0.0, 0.0], TL.fit_bias_columns(fix_bias_layout))
    @test !TL.seed_fix_covariance!(P3, idxs, layout, pvt, 1, H_degenerate, zeros(5, 5), fix_bias_layout)
    @test P3 == seeded
end

@testset "A bias the fix's epoch does not determine keeps its seed" begin
    # Galileo seen on E5a alone: the L5 inter-frequency bias is collinear with the
    # Galileo clock, so the fix's layout has no column for it, and the seed fills the
    # position and both clocks while the bias keeps its variance.
    layout = TL.NavFilterLayout((GPSL1CA(), GalileoE1B(), GalileoE5aI()))
    config = NavigationFilterConfig()
    idxs = TL.NavFilterIndices(config, layout)
    n = TL.num_nav_states(config, layout)
    user = SVector(6.378e6, 0.0, 0.0)
    directions = [(1.0, 0.0, 0.0), (0.6, 0.8, 0.0), (0.6, -0.4, 0.7), (0.5, -0.3, -0.8), (0.7, 0.5, 0.5), (0.4, -0.7, -0.6)]
    measurements = [
        _test_measurement(; prn = k, signal = k <= 4 ? GPSL1CA() : GalileoE5aI(), sat_pos) for
        (k, sat_pos) in enumerate(_fix_positions(user, directions))
    ]
    gst_bias = Dictionary{PositionVelocityTime.SupportedTimeSystem,typeof(1.0u"m")}([GST()], [2.0u"m"])
    pvt = PVTSolution(; position = ECEF(user...), reference_system = GPST(), inter_system_biases = gst_bias)
    fix_bias_layout, H = _fix_design(layout, measurements, user)
    @test TL.fit_bias_columns(fix_bias_layout).num_ifbs == 0
    x, P = zeros(n), zeros(n, n)
    TL.initial_nav_state!(x, P, layout, idxs, pvt)
    seeded = copy(P)
    @test TL.seed_fix_covariance!(P, idxs, layout, pvt, 1, H, zeros(5, 5), fix_bias_layout)
    block = [idxs.pos; idxs.clock_biases]
    @test P[block, block] ≈ TL.FIX_PSEUDORANGE_STD^2 * inv(H' * H)
    l5 = idxs.ifb[1]
    @test P[l5, :] == seeded[l5, :]

    # A design with that bias as a column anyway is singular. Rounding can leave the last
    # Cholesky pivot of this geometry slightly positive rather than failing the
    # factorization, and the inverse would seed the Galileo clock with a variance of about
    # 10¹⁵ m²; the pivot check leaves the seed as it was.
    columns = BiasColumns([1, 1, 1, 1, 2, 2], 2, [0, 0, 0, 0, 1, 1], 1)
    H_singular = calc_design_matrix!(zeros(6, 6), [measurement.position for measurement in measurements], [user..., 0.0, 0.0, 0.0], columns)
    P2 = copy(seeded)
    @test !TL.seed_fix_covariance!(P2, idxs, layout, pvt, 1, H_singular, zeros(6, 6), fix_bias_layout)
    @test P2 == seeded
end
