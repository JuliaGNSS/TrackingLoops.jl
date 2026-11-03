# ─────────────────────────────────────────────────────────────────────────────
# Vector tracking (VDLL / VDFLL): the navigation filter's model
#
# A central navigation Kalman filter closes all satellites' loops at once: each
# satellite's accumulated DLL (and FLL) discriminators become pseudorange (and
# pseudorange-rate) measurements, and the predicted line-of-sight dynamics set every
# satellite's NCOs — a VDLL, or a VDFLL with `use_pseudorange_rates`.
#
# The bias model follows `PositionVelocityTime`: one clock bias per GNSS time system
# (all driven by one oscillator drift) and one inter-frequency bias per band beyond a
# reference band; pseudoranges are corrected for the broadcast ionosphere and the
# Saastamoinen troposphere. Which biases an epoch can determine is decided per update by
# `decide_bias_layout!` (see `clock_constraints!`).
#
# The per-integration `VectorTrackingLoop` accumulates discriminators and applies the filter's
# commands; this file holds the model above it (the update is in `vector/navigation_processor.jl`):
# plain functions that allocate nothing once their buffers exist.
# ─────────────────────────────────────────────────────────────────────────────

"""
    NavFilterConfig(; use_pseudorange_rates = true,
                           motion_model_order = 2, clock_model_order = 2,
                           acceleration_noise_std = 5.0m/s^2,
                           h0 = 2e-19, hm2 = 2e-20,
                           ifb_noise_density = 0.01m/sqrt(1.0s),
                           measurement_deficit_timeout = 10.0s)

Configuration of the vector-tracking navigation filter of a [`VectorTrackingLoop`](@ref).
The defaults suit a vehicle with a consumer-grade front end.

# Fields
- `use_pseudorange_rates`: `true` (VDFLL) also fuses the FLL pseudorange rates; with
  `false` (VDLL) only pseudoranges enter, and the carrier Doppler the NCOs are set to
  comes from the code measurements alone.
- `motion_model_order`: per-axis states — `1` position, `2` + velocity,
  `3` + acceleration.
- `clock_model_order`: `1` clock biases only (one per GNSS time system), `2` biases +
  one common drift.
- `acceleration_noise_std`: how hard the platform manoeuvres. `5 m/s²` suits road
  vehicles; ~`1 m/s²` pedestrians or ships, tens of `m/s²` aircraft. It means the same
  at every order: orders 1 and 3 drive a velocity (`σ_a·τ`) or jerk (`σ_a/τ`) noise
  over a `τ = 2 s` manoeuvre (`MANOEUVRE_TIME`).
- `h0`, `hm2`: oscillator Allan-variance coefficients ``h_0`` (s) and ``h_{-2}`` (1/s);
  ``h_0 = 2τσ_y²(τ)`` at short and ``h_{-2} = 3σ_y²(τ)/(2π²τ)`` at long averaging
  times. The defaults model a TCXO; an OCXO is far tighter, a bare crystal looser.
- `ifb_noise_density`: random-walk density of the inter-frequency biases (`m/√s`),
  covering the front end's thermal drift: by default a bias wanders ~`0.01 m` (1σ) in
  one second. Raise it for bands that are less thermally coupled, lower it for a
  calibrated front end.
- `measurement_deficit_timeout`: how long the filter may coast on unsolvable epochs (too few
  measurements or distinct satellites for the unknowns) before falling back to scalar
  tracking. Solution uncertainty is reported, not policed: see
  [`VectorTrackingStatus`](@ref)'s `position_std`.
"""
struct NavFilterConfig
    use_pseudorange_rates::Bool
    motion_model_order::Int
    clock_model_order::Int
    acceleration_noise_std::typeof(1.0m / s^2)
    h0::Float64
    hm2::Float64
    ifb_noise_density::typeof(1.0m / sqrt(1.0s))
    measurement_deficit_timeout::typeof(1.0s)
end

function NavFilterConfig(;
    use_pseudorange_rates = true,
    motion_model_order = 2,
    clock_model_order = 2,
    acceleration_noise_std = 5.0m / s^2,
    h0 = 2e-19,
    hm2 = 2e-20,
    ifb_noise_density = 0.01m / sqrt(1.0s),
    measurement_deficit_timeout = 10.0s,
)
    1 <= motion_model_order <= 3 ||
        throw(ArgumentError("motion_model_order must be 1, 2 or 3"))
    1 <= clock_model_order <= 2 || throw(ArgumentError("clock_model_order must be 1 or 2"))
    NavFilterConfig(
        use_pseudorange_rates,
        motion_model_order,
        clock_model_order,
        acceleration_noise_std,
        h0,
        hm2,
        ifb_noise_density,
        measurement_deficit_timeout,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Navigation filter model

# Which clock-bias and IFB state each signal group (by position in the signal tuple)
# maps to. Built from the *configured* signals so the state dimension is fixed for the
# run (a constellation without measurements coasts on its process noise); IFB columns
# follow `ifb_layout` (a band alone on its constellation folds into that clock).
struct NavFilterLayout
    time_systems::Vector{SupportedTimeSystem} # clock-bias state order
    extra_bands::Vector{Symbol}               # inter-frequency-bias state order
    reference_bands::Vector{Symbol}           # anchor band per IFB state
    clock_bias_index_by_group::Vector{Int}    # group ⇒ clock-bias state
    ifb_index_by_group::Vector{Int}           # group ⇒ IFB state (0 = reference band)
    band_by_group::Vector{Symbol}             # group ⇒ frequency band
    signal_id_by_group::Vector{Symbol}        # group ⇒ signal id, the `observations` key
end

function NavFilterLayout(signals::Tuple{Vararg{AbstractGNSSSignal}})
    time_system_per_group = collect(SupportedTimeSystem, map(get_time_system, signals))
    band_per_group = collect(Symbol, map(get_band_id, signals))
    time_systems = unique(time_system_per_group)
    ifb_indices, extra_bands, reference_bands, _ =
        ifb_layout(time_system_per_group, band_per_group)
    NavFilterLayout(
        time_systems,
        extra_bands,
        reference_bands,
        Int[something(findfirst(other -> other === ts, time_systems)) for ts in time_system_per_group],
        ifb_indices,
        band_per_group,
        collect(Symbol, map(get_signal_id, signals)),
    )
end

num_clock_biases(layout::NavFilterLayout) = length(layout.time_systems)
num_ifbs(layout::NavFilterLayout) = length(layout.extra_bands)

# Indices of the state vector
# `[x, ẋ, (ẍ), y, ẏ, (ÿ), z, ż, (z̈), clock_bias_1..M, (clock_drift), ifb_1..B]`;
# positions and biases in m, derivatives in m/s (m/s²).
struct NavFilterIndices
    pos::Vector{Int}
    vel::Vector{Int}          # empty for motion_model_order == 1
    acc::Vector{Int}          # empty for motion_model_order <= 2
    clock_biases::Vector{Int} # one per GNSS time system
    clock_drift::Int          # 0 for clock_model_order == 1; shared by all biases
    ifb::Vector{Int}          # one per frequency band beyond its reference
end

function NavFilterIndices(
    motion_model_order::Int,
    clock_model_order::Int,
    num_clock_biases::Int,
    num_ifbs::Int,
)
    pos = [1 + (i - 1) * motion_model_order for i = 1:3]
    vel = motion_model_order >= 2 ? pos .+ 1 : Int[]
    acc = motion_model_order >= 3 ? pos .+ 2 : Int[]
    clock_biases = 3 * motion_model_order .+ (1:num_clock_biases)
    clock_drift = clock_model_order >= 2 ? clock_biases[end] + 1 : 0
    ifb_offset = max(clock_drift, clock_biases[end])
    ifb = ifb_offset .+ (1:num_ifbs)
    NavFilterIndices(pos, vel, acc, collect(clock_biases), clock_drift, collect(ifb))
end

NavFilterIndices(config::NavFilterConfig, layout::NavFilterLayout) = NavFilterIndices(
    config.motion_model_order,
    config.clock_model_order,
    num_clock_biases(layout),
    num_ifbs(layout),
)

num_nav_states(config::NavFilterConfig, layout::NavFilterLayout) =
    3 * config.motion_model_order +
    num_clock_biases(layout) +
    (config.clock_model_order >= 2 ? 1 : 0) +
    num_ifbs(layout)

# (position, velocity, clock drift) of a state vector, zeros for unmodelled ones.
@inline function nav_filter_states(x, idxs::NavFilterIndices)
    pos = idxs.pos
    user_pos = SVector{3,Float64}(x[pos[1]], x[pos[2]], x[pos[3]])
    vel = idxs.vel
    user_vel =
        isempty(vel) ? zero(SVector{3,Float64}) :
        SVector{3,Float64}(x[vel[1]], x[vel[2]], x[vel[3]])
    user_clock_drift = idxs.clock_drift == 0 ? 0.0 : x[idxs.clock_drift]
    user_pos, user_vel, user_clock_drift
end

# The `[x, y, z, tc₁..tc_M, ifb₁..ifb_B]` sub-vector PositionVelocityTime's
# measurement helpers (`predict_pseudoranges!`, `calc_design_matrix!`) expect, written into `ξ`.
function position_and_bias_vector!(ξ, x, idxs::NavFilterIndices)
    for i = 1:3
        ξ[i] = x[idxs.pos[i]]
    end
    num_clocks = length(idxs.clock_biases)
    for (i, index) in enumerate(idxs.clock_biases)
        ξ[3+i] = x[index]
    end
    for (i, index) in enumerate(idxs.ifb)
        ξ[3+num_clocks+i] = x[index]
    end
    ξ
end

# Process model F: a kinematic block per axis, every clock bias integrates the one
# drift, IFBs are constant.
function nav_filter_process_model!(F, config::NavFilterConfig, idxs::NavFilterIndices, Δt_nav)
    m_ord = config.motion_model_order
    fill!(F, 0.0)
    for i in axes(F, 1)
        F[i, i] = 1.0
    end
    axis = SMatrix{3,3,Float64}(1, 0, 0, Δt_nav, 1, 0, Δt_nav^2 / 2, Δt_nav, 1)
    for d = 0:2, i = 1:m_ord, j = 1:m_ord
        F[d*m_ord+i, d*m_ord+j] = axis[i, j]
    end
    if idxs.clock_drift != 0
        for bias in idxs.clock_biases
            F[bias, idxs.clock_drift] = Δt_nav
        end
    end
    F
end

# How long one manoeuvre lasts (τ): it converts `acceleration_noise_std` into the velocity
# (`σ_a·τ`) and jerk (`σ_a/τ`) figures of orders 1 and 3. Two seconds is a road vehicle's
# lane change or 0.5 g stop from 36 km/h. A constant, not a keyword, because it is inert
# at the default order 2; change it here for much shorter or longer manoeuvres.
const MANOEUVRE_TIME = 2.0s

# Per-axis process-noise gain `Γ` (zero-padded to three entries) and driving std `σ`; the
# axis block of `Q` is `Γ Γᵀ σ²`, driven by the first unmodelled derivative (Bar-Shalom,
# Li & Kirubarajan, "Estimation with Applications to Tracking and Navigation", Wiley
# 2001, §6.3.2):
#
#   order 1 (p)     → velocity:     Γ = [Δt_nav],             σ = σ_v
#   order 2 (p,v)   → acceleration: Γ = [Δt_nav²/2, Δt_nav],       σ = σ_a
#   order 3 (p,v,a) → jerk:         Γ = [Δt_nav³/6, Δt_nav²/2, Δt_nav], σ = σ_j
#
# Each `Γ` carries its own `Δt_nav` dependence, so this stays right when the interval changes
# (a fixed rescaling of the order-2 model would not).
function motion_noise_model(config::NavFilterConfig, Δt_nav)
    acc_std = ustrip(m / s^2, config.acceleration_noise_std)
    τ = ustrip(s, MANOEUVRE_TIME)
    config.motion_model_order == 1 && return SVector(Δt_nav, 0.0, 0.0), acc_std * τ
    config.motion_model_order == 2 && return SVector(Δt_nav^2 / 2, Δt_nav, 0.0), acc_std
    SVector(Δt_nav^3 / 6, Δt_nav^2 / 2, Δt_nav), acc_std / τ
end

# Process noise Q: motion noise (`motion_noise_model`), Allan-variance clock noise and
# an IFB random walk. All clock biases ride one oscillator, so the drift random walk
# (Sg) is fully correlated across them; only white frequency noise (Sf) is per bias.
function nav_filter_process_noise_covariance!(
    Q,
    config::NavFilterConfig,
    idxs::NavFilterIndices,
    Δt_nav,
)
    m_ord = config.motion_model_order
    c = SPEED_OF_LIGHT
    fill!(Q, 0.0)

    Γ, driving_std = motion_noise_model(config, Δt_nav)
    for d = 0:2, i = 1:m_ord, j = 1:m_ord
        Q[d*m_ord+i, d*m_ord+j] = (Γ[i] * Γ[j]) * driving_std^2
    end

    # Sf [m²/s] from h0, Sg [m²/s³] from h-2.
    Sf = c^2 * config.h0 / 2
    Sg = c^2 * 2 * π^2 * config.hm2
    for bias_i in idxs.clock_biases, bias_j in idxs.clock_biases
        Q[bias_i, bias_j] = Sg * Δt_nav^3 / 3 + (bias_i == bias_j ? Sf * Δt_nav : 0.0)
    end
    if idxs.clock_drift != 0
        for bias in idxs.clock_biases
            Q[bias, idxs.clock_drift] = Sg * Δt_nav^2 / 2
            Q[idxs.clock_drift, bias] = Sg * Δt_nav^2 / 2
        end
        Q[idxs.clock_drift, idxs.clock_drift] = Sg * Δt_nav
    end

    ifb_density = ustrip(m / sqrt(s), config.ifb_noise_density)
    for ifb in idxs.ifb
        Q[ifb, ifb] = ifb_density^2 * Δt_nav
    end
    Q
end

"""
    NavFilterModel

The navigation filter's process model `F`, `Q` for one navigation filter update, rebuilt in
place when the update interval changes (`ensure_nav_filter_update_interval!`).
"""
mutable struct NavFilterModel
    update_interval::typeof(1.0s)
    const F::Matrix{Float64}
    const Q::Matrix{Float64}
    const idxs::NavFilterIndices
end

function NavFilterModel(config::NavFilterConfig, layout::NavFilterLayout, update_interval)
    n = num_nav_states(config, layout)
    model = NavFilterModel(
        update_interval,
        Matrix{Float64}(undef, n, n),
        Matrix{Float64}(undef, n, n),
        NavFilterIndices(config, layout),
    )
    rebuild_nav_filter_model!(model, config, update_interval)
end

function rebuild_nav_filter_model!(model::NavFilterModel, config::NavFilterConfig, update_interval)
    Δt_nav = ustrip(s, update_interval)
    nav_filter_process_model!(model.F, config, model.idxs, Δt_nav)
    nav_filter_process_noise_covariance!(model.Q, config, model.idxs, Δt_nav)
    model.update_interval = update_interval
    model
end

# Rebuild `F`/`Q` whenever the measured update interval changes at all: the receive epoch
# advances by it, so a nominal-interval model mispredicts the clock bias by `drift·ΔT`.
# Chunk quantisation alone overshoots by up to a chunk (e.g. 4 ms on 100 ms), metres at a
# TCXO's hundreds of m/s of drift against centimetres of clock process noise. The rebuild
# allocates nothing.
function ensure_nav_filter_update_interval!(
    model::NavFilterModel,
    config::NavFilterConfig,
    update_interval,
)
    update_interval == model.update_interval && return model
    rebuild_nav_filter_model!(model, config, update_interval)
end

# The state propagated kinematically by `τ` seconds into `x_τ` (a plain copy for
# `τ = 0`); sizes commands that take effect `τ` after the measurement epoch.
function propagate_state!(x_τ, x, idxs::NavFilterIndices, τ)
    copyto!(x_τ, x)
    has_vel = !isempty(idxs.vel)
    has_acc = !isempty(idxs.acc)
    for i = 1:3
        p = idxs.pos[i]
        if has_acc
            v, a = idxs.vel[i], idxs.acc[i]
            x_τ[p] = x[p] + x[v] * τ + x[a] * τ^2 / 2
            x_τ[v] = x[v] + x[a] * τ
        elseif has_vel
            x_τ[p] = x[p] + x[idxs.vel[i]] * τ
        end
    end
    if idxs.clock_drift != 0
        drift = x[idxs.clock_drift]
        for bias in idxs.clock_biases
            x_τ[bias] = x[bias] + drift * τ
        end
    end
    x_τ
end

# ─────────────────────────────────────────────────────────────────────────────
# Measurement gathering

# What the navigation filter needs about one vector-loop member for one update. All
# downstream per-satellite buffers align with the member vector. Plain data, so it is
# stored inline and refilled without allocating.
struct VectorTrackingObservation
    group::Int             # position of the member's signal group
    slot_index::Int        # position in that group's satellite vector
    prn::Int
    clock_bias_index::Int  # clock-bias state of this member's time system
    ifb_index::Int         # IFB state of this member's band (0 = reference)
    chip_length::Float64   # m
    wavelength::Float64    # m
    chip_rate::Float64 # Hz
    available::Bool        # usable as a measurement this update (in code lock)
    # Corrected transmit time (own system's time of week, s), at which the broadcast
    # polynomials are evaluated. Differenced across members on the reference system's
    # count, as in `calc_pvt` (see `_measure_pseudoranges!`).
    time::Float64
    sat_pos::SVector{3,Float64}
    sat_vel::SVector{3,Float64}
    sat_clock_drift::Float64 # s/s
    code_discriminator::Float64 # accumulated DLL output, moved to the epoch (chips)
    pseudorange_rate::Float64 # measured −λ·(replica Doppler + mean FLL) at the epoch (m/s)
    cn0_lin::Float64         # C/N₀, linear (Hz); C/N₀ without `_lin` is in dB-Hz
    early_late_spacing::Float64 # chips
    coherent_integration_time::Float64 # s, per coherent integration (sets DLL/FLL noise)
end

# Code phase (chips) the code correction steered out over the second half of the update interval,
# `∫_{Δt_nav/2}^{Δt_nav} c(t) dt`: moves the mean code discriminator from mid-interval to the epoch.
# `c₀, c₁, c₂` are the last three corrections, newest first, each reaching the replica
# `L` after its measurement epoch; `L = 0` gives `c₀·Δt_nav/2`. Valid for `L ≤ 2.5Δt_nav`
# (`MAX_FEEDBACK_DELAY_UPDATES`).
function code_phase_advance(loop_state::VectorLoopState, Δt_nav)
    c₀, c₁, c₂ = map(c -> ustrip(Hz, c), loop_state.code_frequency_correction_history)
    L = ustrip(s, loop_state.feedback_delay)
    c₀ * Δt_nav / 2 - (c₀ - c₁) * clamp(L - Δt_nav / 2, 0.0, Δt_nav / 2) -
    (c₁ - c₂) * clamp(L - 3Δt_nav / 2, 0.0, Δt_nav / 2)
end

# Longest NCO delay (update intervals) `code_phase_advance` covers; longer ones throw.
const MAX_FEEDBACK_DELAY_UPDATES = 2.5

# Mean DLL discriminator (chips), advanced to the epoch by `code_phase_advance`; negated
# because `dll_disc`'s sense is reversed. Zero when nothing was accumulated (see
# `has_disc_sums`).
function calc_mean_code_disc_at_epoch(loop_state::VectorLoopState, Δt_nav)
    mean = _disc_mean(loop_state.code_disc_sum)
    isnothing(mean) ? 0.0 : -mean + code_phase_advance(loop_state, Δt_nav)
end

# Mean FLL discriminator (Hz), `f_incoming − f_replica`, so it enters with its own sign:
# the rate observable at the epoch is `carrier_doppler + mean_fll`. `0.0` when nothing
# was accumulated (see `has_disc_sums`).
#
# Deliberately NOT moved to mid-interval, unlike the code: the mean FLL is
# `mean(f_in) − mean(f_replica)`, and as the replica follows the filter's Doppler ramp,
# `f_replica(epoch) − mean(f_replica)` is the predicted Doppler's rise over half the
# update, the incoming carrier's to first order: the sum is the incoming Doppler at the
# epoch. The ramp is the filter's prediction alone, so no loop's jitter enters the
# replica's command; with the PLL on the NCO, one integration's jitter
# (`f_NCO,end − mean(f_NCO)`) would make the integrated rate error walk. A command
# applied inside the update interval (an NCO delay) leaves a share of the filter's own correction
# in the measurement: on the simulator the residuals stay at `√R` and the integrated
# rate error at a centimetre, but the lag-1 autocorrelation shortens from −1/2 (−0.3 at
# 20 ms of delay, −0.1 at 50 ms).
function calc_mean_fll_disc(loop_state::VectorLoopState)
    mean = _disc_mean(loop_state.fll_disc_sum)
    isnothing(mean) ? 0.0 : ustrip(Hz, mean)
end

# Whether a member accumulated both a code and a carrier measurement this update (none
# without a full coherent integration, e.g. admitted at the update's end or starved of samples). The zero
# `calc_mean_code_disc_at_epoch` and `calc_mean_fll_disc` substitute would be a confidently zero
# residual at full `R` weight, pulling the state toward the NCO, so such members are
# withheld from the measurements (still commanded, like a member out of code lock). The
# carrier sum counts only integrations with a previous prompt, so it can be empty alone (a
# update whose only integration is the first after a (re)start); that member is withheld too.
has_disc_sums(loop_state::VectorLoopState) = first(loop_state.code_disc_sum) > 0 && first(loop_state.fll_disc_sum) > 0

# Linear C/N₀ (Hz) from dB-Hz, floored to 1 against a degenerate weight; a NaN estimate
# (empty correlator bank) maps to 1 so it cannot corrupt the Kalman update.
function calc_linear_cn0_floor(cn0_dbhz)
    cn0_lin = 10^(cn0_dbhz / 10)
    isnan(cn0_lin) ? 1.0 : max(cn0_lin, 1.0)
end

# ─────────────────────────────────────────────────────────────────────────────
# Measurement prediction

# Clock/IFB columns of the members at `indices` as PositionVelocityTime's `BiasColumns`,
# so its `predict_pseudoranges!` / `calc_design_matrix!` model the pseudoranges.
function member_bias_columns!(columns::BiasColumns, members, indices)
    resize!(columns.clock_columns, length(indices))
    resize!(columns.ifb_columns, length(indices))
    for (k, j) in enumerate(indices)
        columns.clock_columns[k] = members[j].clock_bias_index
        columns.ifb_columns[k] = members[j].ifb_index
    end
    columns
end

# The pseudorange rate (m/s) the epoch state `x` predicts for `member`: the derivative of
# the pseudorange `predict_pseudoranges!` models, Earth rotation included (PositionVelocityTime's
# `calc_pseudorange_rate`), positive while receding.
@inline function predict_pseudorange_rate(x, idxs::NavFilterIndices, member::VectorTrackingObservation)
    user_pos, user_vel, user_clock_drift = nav_filter_states(x, idxs)
    calc_pseudorange_rate(
        user_pos,
        user_vel,
        user_clock_drift,
        member.sat_pos,
        member.sat_vel,
        member.sat_clock_drift,
    )
end

# Measurement model `h!(y, x)`: the used members' pseudoranges, then their rates when
# fused, then one row per time-offset constraint (the offset between two clock states). A
# callable struct, not a closure, so the update is concrete and allocation-free.
struct NavFilterMeasurementModel
    idxs::NavFilterIndices
    ξ::Vector{Float64}
    positions::Vector{SVector{3,Float64}}
    members::Vector{VectorTrackingObservation}
    used::Vector{Int}
    columns::BiasColumns
    use_rates::Bool
    constraints::Vector{Tuple{Int,Int,Float64}}
end

function (model::NavFilterMeasurementModel)(y, x)
    idxs = model.idxs
    num_observations = length(model.positions)
    position_and_bias_vector!(model.ξ, x, idxs)
    predict_pseudoranges!(y, model.positions, model.ξ, model.columns)
    offset = num_observations
    if model.use_rates
        for (k, j) in enumerate(model.used)
            y[num_observations+k] = predict_pseudorange_rate(x, idxs, model.members[j])
        end
        offset += num_observations
    end
    for (i, (clock_index, target_clock_index, _)) in enumerate(model.constraints)
        y[offset+i] = x[idxs.clock_biases[clock_index]] - x[idxs.clock_biases[target_clock_index]]
    end
    y
end

# Measurement-noise variances, each the discriminator variance of one coherent
# integration of length `T_coh` propagated through the mean of the
# `N = Δt_nav/T_coh` coherent integrations that the filter's measurement is.
#
# Code: `dll_disc` is the noncoherent early-minus-late discriminator (Kaplan & Hegarty,
# "Understanding GPS: Principles and Applications", 2nd ed., Artech House 2006, §5.5.2;
# Betz & Kolodziejski, "Generalized Theory of Code Tracking with an Early-Late
# Discriminator, Part II", IEEE Trans. AES 45(4), 2009, pp. 1557-1564), with their loop
# bandwidth `B_n` set to one integration's open-loop `1/(2·T_coh)`. Integrations share no
# samples, so their noise is white and
#     var(mean) = d/(4·C/N0·Δt_nav) · (1 + 2/((2 − d)·C/N0·T_coh))   [chips²]:
# the thermal term averages down over `Δt_nav`, the squaring loss stays pinned to `T_coh`.
# Times `chip_length²` for metres. For VEML `d` is the inner pair's spacing, making this
# EPL model mildly conservative (the outer correlators average a little more noise away).
# Assumes `d < 2` chips (the default is 0.5).
function pseudorange_noise_variance(member::VectorTrackingObservation, Δt_nav)
    cn0_tcoh = member.cn0_lin * member.coherent_integration_time
    d = member.early_late_spacing
    squaring_loss = 1 + 2 / ((2 - d) * cn0_tcoh)
    d / (4 * Δt_nav * member.cn0_lin) * squaring_loss * member.chip_length^2
end

# Rate: the ATAN FLL (`fll_disc`) phase jitter per coherent integration is
#     σ_φ² = 1/(2·C/N0·T_coh)·(1 + 1/(2·C/N0·T_coh))   [rad²].
# Each integration's reading is `(θ_k − θ_{k-1})/(2π·T_coh)`, so the mean telescopes to
# `(θ_N − θ_0)/(2π·Δt_nav)`:
#     var(mean) = 2·σ_φ² / (2π·Δt_nav)²   [Hz²],   times λ² for (m/s)².
# Consecutive updates share a boundary phase (`previous_prompt` chains across them) with
# opposite signs: cov = −σ_φ²/(2π·Δt_nav)² against var = 2·σ_φ²/(2π·Δt_nav)², so their rate
# errors have lag-1 correlation −1/2, which the filter cannot represent. Being
# negative, it only makes the filter pessimistic about the rates, so `R` is deliberately
# not inflated; its cost, a pessimistic velocity / clock-drift uncertainty, only
# measurement differencing (Bryson-Henrikson) or a boundary-phase state would remove.
# Nor is the variance `N` times larger, as independent integrations would give: the telescoping
# earns the tighter figure. The ≈ −0.5 lag-1 autocorrelation of the rate residuals is by
# construction and must not be tuned against. One well short of it, with residuals well
# above `√R`, is the signature of a replica term that does not telescope (see
# `calc_mean_fll_disc`).
function pseudorange_rate_noise_variance(member::VectorTrackingObservation, Δt_nav)
    cn0_tcoh = member.cn0_lin * member.coherent_integration_time
    sigma_phi2 = 1 / (2 * cn0_tcoh) * (1 + 1 / (2 * cn0_tcoh))
    member.wavelength^2 * sigma_phi2 / (2 * π^2 * Δt_nav^2)
end

# ─────────────────────────────────────────────────────────────────────────────
# The bias layout of an update

# Accuracy (m) credited to a broadcast GGTO / BGTO when it ties two clocks: the Galileo OS
# SDD (Issue 1.2, Nov 2021) commits the GGTO to < 20 ns (95 %), σ ≈ 10 ns ≈ 3 m; the BGTO,
# with no published commitment, is credited the same. Too coarse to use when the geometry
# observes the offset, hence a constraint only where the satellites cannot separate the
# clocks.
const BROADCAST_INTER_SYSTEM_BIAS_STD = 3.0

# An update's bias layout is `decide_bias_layout!`'s decision on the used members' measurements, a
# `BiasLayout`, so it agrees with `calc_pvt` on solvability, the broadcast time offsets it
# takes as given and the DOP columns. The filter keeps a clock state per configured time
# system; each `ClockConstraint` of the layout enters it as the pseudo-measurement
# `x[clock] - x[target_clock] = offset` (`clock_constraints!`), so the clocks stay
# separate states, available for epochs that observe them. An unsolvable epoch's layout
# still ties every pair of clocks a broadcast offset can tie, so the offsets keep the
# clocks tied through the outage the measurement deficit watchdog times.

# The clock-bias state of `time_system`. Every measurement's time system has one, so a missing one
# is a caller's error and throws.
clock_bias_index(layout::NavFilterLayout, time_system) =
    something(findfirst(other -> other === time_system, layout.time_systems))

# The inter-frequency-bias state of `band`, `0` for a band without one: a reference band,
# or a band the configured layout gives no state.
ifb_index(layout::NavFilterLayout, band) =
    something(findfirst(==(band), layout.extra_bands), 0)

# The constraints of `bias_layout`, into `constraints`, as `(clock, target_clock, offset)`
# over the filter's clock states.
function clock_constraints!(constraints, layout::NavFilterLayout, bias_layout)
    empty!(constraints)
    for constraint in bias_layout.constraints
        clock = clock_bias_index(layout, bias_layout.clock_systems[constraint.column])
        target_clock = clock_bias_index(layout, bias_layout.clock_systems[constraint.target_column])
        push!(constraints, (clock, target_clock, constraint.value))
    end
    constraints
end

# 1σ 3-D position uncertainty (m) of a navigation-filter covariance.
function calc_position_std(P, idxs::NavFilterIndices)
    variance = 0.0
    for i in idxs.pos
        variance += P[i, i]
    end
    sqrt(variance)
end

# ─────────────────────────────────────────────────────────────────────────────
# Loop closure

# Commands of one member: the predicted minus the replica's pseudorange as a code
# frequency offset removing it over the next interval `Δt_nav`; the carrier NCO is set to the
# predicted Doppler itself (`_commands_at_application`), formed where it takes effect rather than
# from the measured rate, which reaches the NCO only through the updated state. Predict
# from the *updated* state only: adding the update's projected state correction too
# would double-count it (loop gain 2, oscillation).
calc_nco_code_correction(predicted_pseudorange, measured_pseudorange, chip_rate, Δt_nav) =
    -(predicted_pseudorange - measured_pseudorange) * chip_rate / (Δt_nav * SPEED_OF_LIGHT)

# The carrier Doppler (Hz, positive closing) of a pseudorange rate (m/s, positive receding).
predict_carrier_doppler(predicted_pseudorange_rate, wavelength) = -predicted_pseudorange_rate / wavelength

# Post-fit residuals of one member, from predictions at the *updated* state:
# `(pseudorange in m, range rate in m/s)`, observed − computed as `calc_pvt` (and RTKLIB)
# report them, with pseudorange rate positive while receding.
#
# A well-tracked member's residuals reduce to its own discriminators; one the solution
# predicts poorly (diverging, or out of lock) keeps a large residual, so members outside
# the update are reported too. Under VDLL the rate residual tests the solution against
# rates it never fused.
function calc_post_fit_residuals(
    member::VectorTrackingObservation,
    measured_pseudorange,
    predicted_pseudorange,
    predicted_pseudorange_rate,
)
    (measured_pseudorange + member.code_discriminator * member.chip_length -
     predicted_pseudorange) * m,
    (member.pseudorange_rate - predicted_pseudorange_rate) * (m / s)
end

# ─────────────────────────────────────────────────────────────────────────────
# PVT solution from the navigation filter

# The clock-bias state to report against: `select_reference_system`'s pick for the update's
# `bias_layout`, with the current reference as the previous one, so it is kept while its
# system has a measurement and no broadcast offset ties it to a better-observed system
# (no flicker; the filter keeps a sparsely observed clock determined). Picking only
# present systems keeps a week number, and so a timestamp, available. Without
# measurements it is kept; `_seed!` sets it before the first update.
function select_reference_clock_index(layout::NavFilterLayout, bias_layout, current_index)
    isempty(bias_layout.clock_systems) && return current_index
    prev_reference_system = layout.time_systems[current_index]
    clock_bias_index(layout, select_reference_system(bias_layout; prev_reference_system))
end

# The DOP design column of `time_system`'s clock among `measurements`: its linked group in
# `bias_layout`, shared by the systems a constraint ties together.
function clock_column(bias_layout, measurements, time_system)
    for (k, measurement) in enumerate(measurements)
        measurement.time_system === time_system && return bias_layout.clock_group_columns[k]
    end
    throw(ArgumentError("no measurement is of the time system"))
end

# ─────────────────────────────────────────────────────────────────────────────
# Initialization

# Initial `x` and `P` from a scalar PVT fix; returns the reference clock index. The reference
# clock bias starts at zero (the reference epoch is already corrected by it); other biases
# are seeded where the fix observed them, else zero with a generous variance. The fixed
# variances are a fallback for `seed_fix_covariance!`.
function initial_nav_state!(x, P, layout::NavFilterLayout, idxs::NavFilterIndices, pvt::PVTSolution)
    c = SPEED_OF_LIGHT

    init_std_pos = 1.0                # m
    init_std_vel = 1.0                # m/s
    init_std_acc = 1.0                # m/s²
    init_std_clock_bias = 5e-9 * c    # 5 ns in m
    init_std_clock_drift = 5e-9 * c   # 5 ns/s in m/s
    init_std_unseeded_clock_bias = 100.0 # m — inter-system offsets are tens of m at most
    init_std_seeded_ifb = 1.0         # m
    init_std_unseeded_ifb = 30.0      # m — RF-chain delays are below ~100 ns

    fill!(x, 0.0)
    fill!(P, 0.0)
    position = SVector(pvt.position.x, pvt.position.y, pvt.position.z)
    velocity = SVector(pvt.velocity.x, pvt.velocity.y, pvt.velocity.z)
    for i = 1:3
        p = idxs.pos[i]
        x[p] = position[i]
        P[p, p] = init_std_pos^2
        if !isempty(idxs.vel)
            v = idxs.vel[i]
            x[v] = velocity[i]
            P[v, v] = init_std_vel^2
        end
        if !isempty(idxs.acc)
            a = idxs.acc[i]
            P[a, a] = init_std_acc^2
        end
    end
    if idxs.clock_drift != 0
        x[idxs.clock_drift] = ustrip(m / s, pvt.clock_drift)
        P[idxs.clock_drift, idxs.clock_drift] = init_std_clock_drift^2
    end

    reference_system = pvt.reference_system
    # A reference system the layout lacks falls back to the first clock.
    reference_clock_index =
        isnothing(reference_system) ? 1 :
        something(findfirst(other -> other === reference_system, layout.time_systems), 1)
    for (index, time_system) in enumerate(layout.time_systems)
        state_index = idxs.clock_biases[index]
        if index == reference_clock_index
            P[state_index, state_index] = init_std_clock_bias^2
        elseif _fix_seeds_clock(pvt, layout, reference_clock_index, index)
            x[state_index] = ustrip(m, pvt.inter_system_biases[time_system])
            P[state_index, state_index] = init_std_clock_bias^2
        else
            P[state_index, state_index] = init_std_unseeded_clock_bias^2
        end
    end
    for index in eachindex(layout.extra_bands)
        state_index = idxs.ifb[index]
        if _fix_seeds_ifb(pvt, layout, index)
            x[state_index] = ustrip(m, pvt.inter_frequency_biases[layout.extra_bands[index]].value)
            P[state_index, state_index] = init_std_seeded_ifb^2
        else
            P[state_index, state_index] = init_std_unseeded_ifb^2
        end
    end
    reference_clock_index
end

# Whether the fix estimated clock-bias state `index` (the reference clock, or via an inter-system bias).
_fix_seeds_clock(pvt::PVTSolution, layout::NavFilterLayout, reference_clock_index, index) =
    index == reference_clock_index || haskey(pvt.inter_system_biases, layout.time_systems[index])

# Whether the fix seeds IFB state `index`: only against the layout's reference band.
function _fix_seeds_ifb(pvt::PVTSolution, layout::NavFilterLayout, index)
    band = layout.extra_bands[index]
    haskey(pvt.inter_frequency_biases, band) &&
        pvt.inter_frequency_biases[band].reference_band == layout.reference_bands[index]
end

# UERE (m) a scalar fix's covariance is scaled by in `seed_fix_covariance!`.
const FIX_PSEUDORANGE_STD = 1.0

# Overwrite the position/bias block of `P` with the fix's covariance `σ² (HᵀH)⁻¹`,
# `σ = FIX_PSEUDORANGE_STD`, for the fix's design `H` over `fix_bias_layout`'s fit columns. A
# geometry-blind variance would trust a GDOP-40 fix like a GDOP-2 one
# (JuliaGNSS/TrackingLoops.jl#16), and the full block keeps the position–clock
# correlations. Unseeded bias states keep their variance, uncorrelated, though the
# position's variance still counts them as unknowns. `normal_matrix` is square scratch of
# `H`'s column count. Returns `false`, leaving `P` unchanged, for a rank-deficient `H`, by
# the test PositionVelocityTime's `calc_fit_covariance!` and `calc_dop` share.
function seed_fix_covariance!(
    P,
    idxs::NavFilterIndices,
    layout::NavFilterLayout,
    pvt::PVTSolution,
    reference_clock_index,
    H,
    normal_matrix,
    fix_bias_layout::BiasLayout,
)
    calc_fit_covariance!(normal_matrix, H) || return false
    covariance = Symmetric(normal_matrix, :U)
    σ² = FIX_PSEUDORANGE_STD^2
    num_columns = size(covariance, 1)
    for column = 1:num_columns, row = 1:num_columns
        i = _fix_state_index(idxs, layout, pvt, reference_clock_index, fix_bias_layout, row)
        j = _fix_state_index(idxs, layout, pvt, reference_clock_index, fix_bias_layout, column)
        i == 0 || j == 0 || (P[i, j] = σ² * covariance[row, column])
    end
    true
end

# The state behind `column` of the fix's design, `0` for an unseeded bias.
function _fix_state_index(idxs, layout, pvt, reference_clock_index, fix_bias_layout, column)
    column <= 3 && return idxs.pos[column]
    columns = fit_bias_columns(fix_bias_layout)
    if column - 3 <= columns.num_clock_biases
        index = clock_bias_index(layout, fix_bias_layout.clock_systems[column-3])
        return _fix_seeds_clock(pvt, layout, reference_clock_index, index) ?
               idxs.clock_biases[index] : 0
    end
    index = ifb_index(layout, fix_bias_layout.extra_bands[column-3-columns.num_clock_biases])
    index == 0 || !_fix_seeds_ifb(pvt, layout, index) ? 0 : idxs.ifb[index]
end
