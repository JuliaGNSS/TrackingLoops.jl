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
# Saastamoinen troposphere. Which biases an epoch can determine is decided per cycle by
# `decide_bias_layout!` (see `time_offset_constraints!`).
#
# The per-record `VectorPLLAndDLL` accumulates discriminators and applies the filter's
# commands; this file holds the model above it (the cycle is in `vector/tracking.jl`):
# plain functions that allocate nothing once their buffers exist.
# ─────────────────────────────────────────────────────────────────────────────

"""
    NavigationFilterConfig(; use_pseudorange_rates = true,
                           motion_model_order = 2, clock_model_order = 2,
                           acceleration_noise_std = 5.0m/s^2,
                           h0 = 2e-19, hm2 = 2e-20,
                           ifb_noise_density = 0.01m/sqrt(1.0s),
                           insufficient_meas_timeout = 10.0s)

Configuration of the vector-tracking navigation filter of a [`VectorPLLAndDLL`](@ref).
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
- `insufficient_meas_timeout`: how long the filter may coast on unsolvable epochs (too few
  measurements or distinct satellites for the unknowns) before falling back to scalar
  tracking. Solution uncertainty is reported, not policed: see
  [`NavigationStatus`](@ref)'s `position_std`.
"""
struct NavigationFilterConfig
    use_pseudorange_rates::Bool
    motion_model_order::Int
    clock_model_order::Int
    acceleration_noise_std::typeof(1.0m / s^2)
    h0::Float64
    hm2::Float64
    ifb_noise_density::typeof(1.0m / sqrt(1.0s))
    insufficient_meas_timeout::typeof(1.0s)
end

function NavigationFilterConfig(;
    use_pseudorange_rates = true,
    motion_model_order = 2,
    clock_model_order = 2,
    acceleration_noise_std = 5.0m / s^2,
    h0 = 2e-19,
    hm2 = 2e-20,
    ifb_noise_density = 0.01m / sqrt(1.0s),
    insufficient_meas_timeout = 10.0s,
)
    1 <= motion_model_order <= 3 ||
        throw(ArgumentError("motion_model_order must be 1, 2 or 3"))
    1 <= clock_model_order <= 2 || throw(ArgumentError("clock_model_order must be 1 or 2"))
    NavigationFilterConfig(
        use_pseudorange_rates,
        motion_model_order,
        clock_model_order,
        acceleration_noise_std,
        h0,
        hm2,
        ifb_noise_density,
        insufficient_meas_timeout,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Navigation filter model

# Which clock-bias and IFB state each signal group (by position in the signal tuple)
# maps to. Built from the *configured* signals so the state dimension is fixed for the
# run (a constellation without measurements coasts on its process noise); IFB columns
# follow `band_ifb_layout` (a band alone on its constellation folds into that clock).
struct NavFilterLayout
    time_systems::Vector{SupportedTimeSystem} # clock-bias state order
    extra_bands::Vector{Symbol}               # inter-frequency-bias state order
    reference_bands::Vector{Symbol}           # anchor band per IFB state
    clock_bias_index_by_group::Vector{Int}    # group ⇒ clock-bias state
    ifb_index_by_group::Vector{Int}           # group ⇒ IFB state (0 = reference band)
    band_by_group::Vector{Symbol}             # group ⇒ frequency band
    signal_id_by_group::Vector{Symbol}        # group ⇒ signal id, the `sats` key
end

function NavFilterLayout(signals::Tuple{Vararg{AbstractGNSSSignal}})
    time_system_per_group = collect(SupportedTimeSystem, map(get_time_system, signals))
    band_per_group = collect(Symbol, map(get_band_id, signals))
    time_systems = unique(time_system_per_group)
    ifb_indices, extra_bands, reference_bands, _ =
        band_ifb_layout(time_system_per_group, band_per_group)
    NavFilterLayout(
        time_systems,
        collect(Symbol, extra_bands),
        collect(Symbol, reference_bands),
        Int[something(time_system_index(time_systems, ts)) for ts in time_system_per_group],
        collect(Int, ifb_indices),
        band_per_group,
        collect(Symbol, map(get_signal_id, signals)),
    )
end

num_clock_biases(layout::NavFilterLayout) = length(layout.time_systems)
num_ifb(layout::NavFilterLayout) = length(layout.extra_bands)

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
    num_ifb::Int,
)
    pos = [1 + (i - 1) * motion_model_order for i = 1:3]
    vel = motion_model_order >= 2 ? pos .+ 1 : Int[]
    acc = motion_model_order >= 3 ? pos .+ 2 : Int[]
    clock_biases = 3 * motion_model_order .+ (1:num_clock_biases)
    clock_drift = clock_model_order >= 2 ? clock_biases[end] + 1 : 0
    ifb_offset = max(clock_drift, clock_biases[end])
    ifb = ifb_offset .+ (1:num_ifb)
    NavFilterIndices(pos, vel, acc, collect(clock_biases), clock_drift, collect(ifb))
end

NavFilterIndices(config::NavigationFilterConfig, layout::NavFilterLayout) = NavFilterIndices(
    config.motion_model_order,
    config.clock_model_order,
    num_clock_biases(layout),
    num_ifb(layout),
)

num_nav_states(config::NavigationFilterConfig, layout::NavFilterLayout) =
    3 * config.motion_model_order +
    num_clock_biases(layout) +
    (config.clock_model_order >= 2 ? 1 : 0) +
    num_ifb(layout)

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
# measurement helpers (`calc_ρ_hat!`, `calc_H!`) expect, written into `ξ`.
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

position_and_bias_vector(x, idxs::NavFilterIndices) = position_and_bias_vector!(
    Vector{Float64}(undef, 3 + length(idxs.clock_biases) + length(idxs.ifb)),
    x,
    idxs,
)

# Process model F: a kinematic block per axis, every clock bias integrates the one
# drift, IFBs are constant.
function nav_filter_process_model!(F, config::NavigationFilterConfig, idxs::NavFilterIndices, T)
    m_ord = config.motion_model_order
    fill!(F, 0.0)
    for i in axes(F, 1)
        F[i, i] = 1.0
    end
    axis = SMatrix{3,3,Float64}(1, 0, 0, T, 1, 0, T^2 / 2, T, 1)
    for d = 0:2, i = 1:m_ord, j = 1:m_ord
        F[d*m_ord+i, d*m_ord+j] = axis[i, j]
    end
    if idxs.clock_drift != 0
        for bias in idxs.clock_biases
            F[bias, idxs.clock_drift] = T
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
#   order 1 (p)     → velocity:     Γ = [T],             σ = σ_v
#   order 2 (p,v)   → acceleration: Γ = [T²/2, T],       σ = σ_a
#   order 3 (p,v,a) → jerk:         Γ = [T³/6, T²/2, T], σ = σ_j
#
# Each `Γ` carries its own `T` dependence, so this stays right when the interval changes
# (a fixed rescaling of the order-2 model would not).
function motion_noise_model(config::NavigationFilterConfig, T)
    acc_std = ustrip(m / s^2, config.acceleration_noise_std)
    τ = ustrip(s, MANOEUVRE_TIME)
    config.motion_model_order == 1 && return SVector(T, 0.0, 0.0), acc_std * τ
    config.motion_model_order == 2 && return SVector(T^2 / 2, T, 0.0), acc_std
    SVector(T^3 / 6, T^2 / 2, T), acc_std / τ
end

# Process noise Q: motion noise (`motion_noise_model`), Allan-variance clock noise and
# an IFB random walk. All clock biases ride one oscillator, so the drift random walk
# (Sg) is fully correlated across them; only white frequency noise (Sf) is per bias.
function nav_filter_process_noise_covariance!(
    Q,
    config::NavigationFilterConfig,
    idxs::NavFilterIndices,
    T,
)
    m_ord = config.motion_model_order
    c = SPEED_OF_LIGHT
    fill!(Q, 0.0)

    Γ, driving_std = motion_noise_model(config, T)
    for d = 0:2, i = 1:m_ord, j = 1:m_ord
        Q[d*m_ord+i, d*m_ord+j] = (Γ[i] * Γ[j]) * driving_std^2
    end

    # Sf [m²/s] from h0, Sg [m²/s³] from h-2.
    Sf = c^2 * config.h0 / 2
    Sg = c^2 * 2 * π^2 * config.hm2
    for bias_i in idxs.clock_biases, bias_j in idxs.clock_biases
        Q[bias_i, bias_j] = Sg * T^3 / 3 + (bias_i == bias_j ? Sf * T : 0.0)
    end
    if idxs.clock_drift != 0
        for bias in idxs.clock_biases
            Q[bias, idxs.clock_drift] = Sg * T^2 / 2
            Q[idxs.clock_drift, bias] = Sg * T^2 / 2
        end
        Q[idxs.clock_drift, idxs.clock_drift] = Sg * T
    end

    ifb_density = ustrip(m / sqrt(s), config.ifb_noise_density)
    for ifb in idxs.ifb
        Q[ifb, ifb] = ifb_density^2 * T
    end
    Q
end

"""
    NavFilterModel

The navigation filter's process model `F`, `Q` for one navigation cycle, rebuilt in
place when the cycle time changes (`ensure_nav_filter_cycle_time!`).
"""
mutable struct NavFilterModel
    cycle_time::typeof(1.0s)
    const F::Matrix{Float64}
    const Q::Matrix{Float64}
    const idxs::NavFilterIndices
end

function NavFilterModel(config::NavigationFilterConfig, layout::NavFilterLayout, cycle_time)
    n = num_nav_states(config, layout)
    model = NavFilterModel(
        cycle_time,
        Matrix{Float64}(undef, n, n),
        Matrix{Float64}(undef, n, n),
        NavFilterIndices(config, layout),
    )
    rebuild_nav_filter_model!(model, config, cycle_time)
end

function rebuild_nav_filter_model!(model::NavFilterModel, config::NavigationFilterConfig, cycle_time)
    T = ustrip(s, cycle_time)
    nav_filter_process_model!(model.F, config, model.idxs, T)
    nav_filter_process_noise_covariance!(model.Q, config, model.idxs, T)
    model.cycle_time = cycle_time
    model
end

# Rebuild `F`/`Q` whenever the measured cycle time changes at all: `reference_time`
# advances by it, so a nominal-interval model mispredicts the clock bias by `drift·ΔT`.
# Chunk quantisation alone overshoots by up to a chunk (e.g. 4 ms on 100 ms), metres at a
# TCXO's hundreds of m/s of drift against centimetres of clock process noise. The rebuild
# allocates nothing.
function ensure_nav_filter_cycle_time!(
    model::NavFilterModel,
    config::NavigationFilterConfig,
    cycle_time,
)
    cycle_time == model.cycle_time && return model
    rebuild_nav_filter_model!(model, config, cycle_time)
end

# The state propagated kinematically by `τ` seconds into `x_τ` (a plain copy for
# `τ = 0`); sizes commands that land `τ` after the cycle epoch.
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

# What the navigation filter needs about one vector-loop member for one cycle. All
# downstream per-satellite buffers align with the member vector. Plain data, so it is
# stored inline and refilled without allocating.
struct VTMember
    group::Int             # position of the member's signal group
    slot_index::Int        # position in that group's satellite vector
    prn::Int
    clock_bias_index::Int  # clock-bias state of this member's time system
    ifb_index::Int         # IFB state of this member's band (0 = reference)
    chip_length::Float64   # m
    wavelength::Float64    # m
    code_frequency::Float64 # Hz
    available::Bool        # usable as a measurement this cycle (in code lock)
    time::Float64          # corrected transmit time (own system's time of week, s)
    # `time` on the GPS Time count (BDT reads 14 s below GPST). Used wherever times are
    # differenced across members (`reference_time`, pseudoranges), as in `calc_pvt`;
    # broadcast polynomials are evaluated at the own-scale `time`.
    time_gpst_count::Float64
    sat_position::SVector{3,Float64}
    sat_velocity::SVector{3,Float64}
    sat_clock_drift::Float64 # s/s
    code_discriminator::Float64 # accumulated DLL output, moved to the epoch (chips)
    pseudorange_rate::Float64 # measured λ·(replica Doppler + mean FLL) at the epoch (m/s)
    cn0_lin::Float64         # C/N₀, linear (Hz); C/N₀ without `_lin` is in dB-Hz
    early_late_spacing::Float64 # chips
    coherent_integration_time::Float64 # s, per coherent correlator dump (sets DLL/FLL noise)
end

# Code phase (chips) the code correction steered out over the cycle's second half,
# `∫_{T/2}^{T} c(t) dt`: moves the mean code discriminator from mid-cycle to the epoch.
# `c₀, c₁, c₂` are the last three corrections, newest first, each reaching the replica
# `L` after its cycle's epoch; `L = 0` gives `c₀·T/2`. Valid for `L ≤ 2.5T`
# (`MAX_LANDING_LEAD_CYCLES`).
function code_phase_advance(state::SatVectorPLLAndDLL, T)
    c₀, c₁, c₂ = map(c -> ustrip(Hz, c), state.code_freq_update_history)
    L = ustrip(s, state.code_update_landing_lead)
    c₀ * T / 2 - (c₀ - c₁) * clamp(L - T / 2, 0.0, T / 2) -
    (c₁ - c₂) * clamp(L - 3T / 2, 0.0, T / 2)
end

# Longest NCO delay (cycles) `code_phase_advance` covers; longer ones throw.
const MAX_LANDING_LEAD_CYCLES = 2.5

# Mean DLL discriminator (chips), advanced to the epoch by `code_phase_advance`; negated
# because `dll_disc`'s sense is reversed. Zero when nothing was accumulated (see
# `has_disc_sums`).
function mean_code_disc_at_epoch(state::SatVectorPLLAndDLL, T)
    mean = _disc_mean(state.code_disc_sum)
    isnothing(mean) ? 0.0 : -mean + code_phase_advance(state, T)
end

# Mean FLL discriminator (Hz), `f_incoming − f_replica`, so it enters with its own sign:
# the rate observable at the epoch is `carrier_doppler + mean_fll`. `0.0` when nothing
# was accumulated (see `has_disc_sums`).
#
# Deliberately NOT moved to mid-cycle, unlike the code: the mean FLL is
# `mean(f_in) − mean(f_replica)`, and as the replica follows the filter's Doppler ramp,
# `f_replica(epoch) − mean(f_replica)` is the predicted Doppler's rise over half the
# cycle, the incoming carrier's to first order: the sum is the incoming Doppler at the
# epoch. The ramp is the filter's prediction alone, so no loop's jitter enters the
# replica's word; with the PLL on the NCO, one record's jitter
# (`f_NCO,end − mean(f_NCO)`) would make the integrated rate error walk. A command
# landing inside the cycle (an NCO delay) leaves a share of the filter's own correction
# in the measurement: on the simulator the residuals stay at `√R` and the integrated
# rate error at a centimetre, but the lag-1 autocorrelation shortens from −1/2 (−0.3 at
# 20 ms of delay, −0.1 at 50 ms).
function mean_fll_disc(state::SatVectorPLLAndDLL)
    mean = _disc_mean(state.fll_disc_sum)
    isnothing(mean) ? 0.0 : ustrip(Hz, mean)
end

# Whether a member accumulated both a code and a carrier measurement this cycle (none
# without a full dump, e.g. admitted at the cycle's end or starved of samples). The zero
# `mean_code_disc_at_epoch` and `mean_fll_disc` substitute would be a confidently zero
# residual at full `R` weight, pulling the state toward the NCO, so such members are
# withheld from the measurements (still commanded, like a member out of code lock). The
# carrier sum counts only records with a previous prompt, so it can be empty alone (a
# cycle whose only record is the first after a (re)start); that member is withheld too.
has_disc_sums(state::SatVectorPLLAndDLL) = first(state.code_disc_sum) > 0 && first(state.fll_disc_sum) > 0

# Linear C/N₀ (Hz) from dB-Hz, floored to 1 against a degenerate weight; a NaN estimate
# (empty correlator) maps to 1 so it cannot corrupt the Kalman update.
function linear_cn0_floor(cn0_dbhz)
    cn0_lin = 10^(cn0_dbhz / 10)
    isnan(cn0_lin) ? 1.0 : max(cn0_lin, 1.0)
end

# Pseudorange (m) from receive and transmit times of week, folded across a week
# rollover.
pseudorange_from_tows(receive_tow, transmit_tow) =
    fold_week_crossover(receive_tow - transmit_tow) * SPEED_OF_LIGHT

# ─────────────────────────────────────────────────────────────────────────────
# Measurement prediction

# Clock/IFB columns of the members at `indices` as PositionVelocityTime's `BiasColumns`,
# so its `calc_ρ_hat!` / `calc_H!` model the pseudoranges.
function member_bias_columns!(columns::BiasColumns, members, indices)
    resize!(columns.clock_bias_indices, length(indices))
    resize!(columns.ifb_indices, length(indices))
    for (k, j) in enumerate(indices)
        columns.clock_bias_indices[k] = members[j].clock_bias_index
        columns.ifb_indices[k] = members[j].ifb_index
    end
    columns
end

member_bias_columns(members, layout::NavFilterLayout, indices = eachindex(members)) =
    member_bias_columns!(
        BiasColumns(Int[], num_clock_biases(layout), Int[], num_ifb(layout)),
        members,
        indices,
    )

# The constraints of `dense_bias_columns!`'s columns: shared and never pushed to, so
# building them allocates nothing (the four-argument `BiasColumns` builds a new vector).
const NO_CLOCK_CONSTRAINTS = ClockConstraint[]

# The same columns restricted to the bias states these members occupy, renumbered
# densely, plus `primary_clock_index` in that numbering (1 if absent). For DOP: an
# all-zero column of a missing constellation makes `HᵀH` singular (`calc_DOP!` → −1).
# `clock_columns` / `ifb_columns` are scratch of the layout's clock and IFB counts, left
# holding each state's dense column (0 if unoccupied).
function dense_bias_columns!(
    clock_indices::Vector{Int},
    ifb_indices::Vector{Int},
    clock_columns::Vector{Int},
    ifb_columns::Vector{Int},
    members,
    indices,
    primary_clock_index::Int,
)
    fill!(clock_columns, 0)
    fill!(ifb_columns, 0)
    for j in indices
        clock_columns[members[j].clock_bias_index] = 1
        members[j].ifb_index != 0 && (ifb_columns[members[j].ifb_index] = 1)
    end
    num_clocks = 0
    for i in eachindex(clock_columns)
        clock_columns[i] == 0 && continue
        num_clocks += 1
        clock_columns[i] = num_clocks
    end
    num_ifbs = 0
    for i in eachindex(ifb_columns)
        ifb_columns[i] == 0 && continue
        num_ifbs += 1
        ifb_columns[i] = num_ifbs
    end
    resize!(clock_indices, length(indices))
    resize!(ifb_indices, length(indices))
    for (k, j) in enumerate(indices)
        clock_indices[k] = clock_columns[members[j].clock_bias_index]
        ifb_indices[k] = members[j].ifb_index == 0 ? 0 : ifb_columns[members[j].ifb_index]
    end
    primary =
        1 <= primary_clock_index <= length(clock_columns) &&
        clock_columns[primary_clock_index] != 0 ? clock_columns[primary_clock_index] : 1
    BiasColumns(clock_indices, num_clocks, ifb_indices, num_ifbs, NO_CLOCK_CONSTRAINTS), primary
end

function dense_bias_columns(members, primary_clock_index::Int, num_clocks, num_ifbs)
    dense_bias_columns!(
        Int[],
        Int[],
        zeros(Int, num_clocks),
        zeros(Int, num_ifbs),
        members,
        eachindex(members),
        primary_clock_index,
    )
end

# The pseudorange rate (m/s) the epoch state `x` predicts for `member`: the derivative of
# the pseudorange `calc_ρ_hat!` models, Earth rotation included (PositionVelocityTime's
# `calc_pseudorange_rate`), negated to the sign of the measured `λ · carrier_doppler`
# (positive while closing).
@inline function predict_pseudorange_rate(x, idxs::NavFilterIndices, member::VTMember)
    user_pos, user_vel, user_clock_drift = nav_filter_states(x, idxs)
    -calc_pseudorange_rate(
        user_pos,
        user_vel,
        user_clock_drift,
        member.sat_position,
        member.sat_velocity,
        member.sat_clock_drift,
    )
end

# Measurement model `h!(y, x)`: the candidates' pseudoranges, then their rates when
# fused, then one row per time-offset constraint (the offset between two clock states). A
# callable struct, not a closure, so the update is concrete and allocation-free.
struct VTMeasurementModel
    idxs::NavFilterIndices
    ξ::Vector{Float64}
    positions::Vector{SVector{3,Float64}}
    members::Vector{VTMember}
    candidates::Vector{Int}
    columns::BiasColumns
    use_rates::Bool
    constraints::Vector{Tuple{Int,Int,Float64}}
end

function (model::VTMeasurementModel)(y, x)
    idxs = model.idxs
    num_sats = length(model.positions)
    position_and_bias_vector!(model.ξ, x, idxs)
    calc_ρ_hat!(y, model.positions, model.ξ, model.columns)
    offset = num_sats
    if model.use_rates
        for (k, j) in enumerate(model.candidates)
            y[num_sats+k] = predict_pseudorange_rate(x, idxs, model.members[j])
        end
        offset += num_sats
    end
    for (i, (clock_index, target_clock_index, _)) in enumerate(model.constraints)
        y[offset+i] = x[idxs.clock_biases[clock_index]] - x[idxs.clock_biases[target_clock_index]]
    end
    y
end

# Measurement-noise variances, each the per-dump discriminator variance for a coherent
# integration time `T_coh` propagated through the mean of the `N = T/T_coh` dumps that
# the filter's measurement is.
#
# Code: `dll_disc` is the noncoherent early-minus-late discriminator (Kaplan & Hegarty,
# "Understanding GPS: Principles and Applications", 2nd ed., Artech House 2006, §5.5.2;
# Betz & Kolodziejski, "Generalized Theory of Code Tracking with an Early-Late
# Discriminator, Part II", IEEE Trans. AES 45(4), 2009, pp. 1557-1564), with their loop
# bandwidth `B_n` set to one dump's open-loop `1/(2·T_coh)`. Dumps share no samples,
# so their noise is white and
#     var(mean) = d/(4·C/N0·T) · (1 + 2/((2 − d)·C/N0·T_coh))   [chips²]:
# the thermal term averages down over `T`, the squaring loss stays pinned to `T_coh`.
# Times `chip_length²` for metres. For VEML `d` is the inner pair's spacing, making this
# EPL model mildly conservative (the outer taps average a little more noise away).
# Assumes `d < 2` chips (the default is 0.5).
function pseudorange_noise_variance(member::VTMember, T)
    cn0_tcoh = member.cn0_lin * member.coherent_integration_time
    d = member.early_late_spacing
    squaring_loss = 1 + 2 / ((2 - d) * cn0_tcoh)
    d / (4 * T * member.cn0_lin) * squaring_loss * member.chip_length^2
end

# Rate: the ATAN FLL (`fll_disc`) phase jitter per dump is
#     σ_φ² = 1/(2·C/N0·T_coh)·(1 + 1/(2·C/N0·T_coh))   [rad²].
# Each dump is `(θ_k − θ_{k-1})/(2π·T_coh)`, so the mean telescopes to
# `(θ_N − θ_0)/(2π·T)`:
#     var(mean) = 2·σ_φ² / (2π·T)²   [Hz²],   times λ² for (m/s)².
# Consecutive cycles share a boundary phase (`previous_prompt` chains across them) with
# opposite signs: cov = −σ_φ²/(2π·T)² against var = 2·σ_φ²/(2π·T)², so their rate
# errors have lag-1 correlation −1/2, which the filter cannot represent. Being
# negative, it only makes the filter pessimistic about the rates, so `R` is deliberately
# not inflated; its cost, a pessimistic velocity / clock-drift uncertainty, only
# measurement differencing (Bryson-Henrikson) or a boundary-phase state would remove.
# Nor is the variance `N` times larger, as independent dumps would give: the telescoping
# earns the tighter figure. The ≈ −0.5 lag-1 autocorrelation of the rate residuals is by
# construction and must not be tuned against. One well short of it, with residuals well
# above `√R`, is the signature of a replica term that does not telescope (see
# `mean_fll_disc`).
function pseudorange_rate_noise_variance(member::VTMember, T)
    cn0_tcoh = member.cn0_lin * member.coherent_integration_time
    sigma_phi2 = 1 / (2 * cn0_tcoh) * (1 + 1 / (2 * cn0_tcoh))
    member.wavelength^2 * sigma_phi2 / (2 * π^2 * T^2)
end

# ─────────────────────────────────────────────────────────────────────────────
# The bias layout of a cycle

# Accuracy (m) credited to a broadcast GGTO / BGTO when it ties two clocks: the Galileo OS
# SDD (Issue 1.2, Nov 2021) commits the GGTO to < 20 ns (95 %), σ ≈ 10 ns ≈ 3 m; the BGTO,
# with no published commitment, is credited the same. Too coarse to use when the geometry
# observes the offset, hence a constraint only where the satellites cannot separate the
# clocks.
const TIME_OFFSET_STD = 3.0

# A cycle's bias layout is `decide_bias_layout!`'s decision on the candidates' rows, a
# `BiasLayout`, so it agrees with `calc_pvt` on solvability, the broadcast time offsets it
# takes as given. The filter keeps a clock state per configured time system; each `ClockConstraint` of the layout enters it as the pseudo-measurement
# `x[clock] - x[target_clock] = offset` (`time_offset_constraints!`), so the clocks stay
# separate states, available for epochs that observe them. An unsolvable epoch's layout
# still ties every pair of clocks a broadcast offset can tie, so the offsets keep the
# clocks tied through the outage the starvation watchdog times.

# The clock-bias state of `time_system`. Every row's time system has one, so a missing one
# is a caller's error and throws.
clock_bias_index(layout::NavFilterLayout, time_system) =
    something(time_system_index(layout.time_systems, time_system))

# The constraints of `bias_layout`, into `constraints`, as `(clock, target_clock, offset)`
# over the filter's clock states.
function time_offset_constraints!(constraints, layout::NavFilterLayout, bias_layout)
    empty!(constraints)
    for constraint in bias_layout.constraints
        clock = clock_bias_index(layout, bias_layout.clock_systems[constraint.column])
        target_clock = clock_bias_index(layout, bias_layout.clock_systems[constraint.target_column])
        push!(constraints, (clock, target_clock, constraint.offset))
    end
    constraints
end

# 1σ 3-D position uncertainty (m) of a navigation-filter covariance.
function position_uncertainty(P, idxs::NavFilterIndices)
    variance = 0.0
    for i in idxs.pos
        variance += P[i, i]
    end
    sqrt(variance)
end

# ─────────────────────────────────────────────────────────────────────────────
# Loop closure

# Commands of one member: the predicted minus the replica's pseudorange as a code
# frequency offset removing it over the next interval `T`; the carrier NCO is set to the
# predicted Doppler itself (`_commands_at_landing`), formed where it lands rather than
# from the measured rate, which reaches the NCO only through the updated state. Predict
# from the *updated* state only: adding the update's projected state correction too
# would double-count it (loop gain 2, oscillation).
nco_code_correction(predicted_pseudorange, measured_pseudorange, code_frequency, T) =
    -(predicted_pseudorange - measured_pseudorange) * code_frequency / (T * SPEED_OF_LIGHT)

# The carrier Doppler (Hz) of a predicted pseudorange rate (m/s, Doppler-signed).
predicted_carrier_doppler(predicted_pseudorange_rate, wavelength) = predicted_pseudorange_rate / wavelength

# Post-fit residuals of one member, from predictions at the *updated* state:
# `(pseudorange in m, range rate in m/s)`, observed − computed as `calc_pvt` (and RTKLIB)
# report them, sign included. Hence opposite subtraction orders: the rate residual is
# `h(x) - z` because this loop measures `+λ · doppler` (positive while closing) while
# `calc_pvt` residuates the geometric range rate (positive while receding). Making the
# two read alike would silently invert the rate residual against every scalar fix.
#
# A well-tracked member's residuals reduce to its own discriminators; one the solution
# predicts poorly (diverging, or out of lock) keeps a large residual, so members outside
# the update are reported too. Under VDLL the rate residual tests the solution against
# rates it never fused.
function post_fit_residuals(
    member::VTMember,
    measured_pseudorange,
    predicted_pseudorange,
    predicted_pseudorange_rate,
)
    (measured_pseudorange + member.code_discriminator * member.chip_length -
     predicted_pseudorange) * m,
    (predicted_pseudorange_rate - member.pseudorange_rate) * (m / s)
end

# ─────────────────────────────────────────────────────────────────────────────
# PVT solution from the navigation filter

# The clock-bias state to report against, among the cycle's `candidate_rows`: kept while
# its system has a row (no flicker), else the reference system of the cycle's
# `bias_layout`, `select_reference_system`'s pick, the one `calc_pvt` reports against.
# Picking only present systems keeps a week number, and so a timestamp, available.
# Without rows it is kept; `_seed!` sets it before the first cycle.
function select_primary_clock_index(
    layout::NavFilterLayout,
    candidate_rows,
    bias_layout,
    current_index,
)
    isempty(candidate_rows) && return current_index
    current = layout.time_systems[current_index]
    for row in candidate_rows
        row.time_system === current && return current_index
    end
    clock_bias_index(layout, select_reference_system(bias_layout))
end

# The epoch offset for the next cycle, advanced by a week when `reference_time` has just
# wrapped (else every later `pvt.time` is a week early). Only an already cached offset
# is advanced; a freshly resolved one comes from a decoder that has rolled over too.
rolled_over_time_epoch_offset(resolved, cached, week_rollover) =
    week_rollover && !isnothing(cached) ? cached + SECONDS_PER_WEEK : resolved

# The constant part of the solution's epoch from a primary-system measurement row: week
# in seconds + start epoch + the scale offset to the GPS Time count `reference_time`
# runs on (`VTMember.time_gpst_count`). Resolved once and cached, so a solution cannot
# lose its timestamp (by which consumers tell a fix from a non-fix); the cache
# survives a change of primary clock, since with the scale offset all systems' offsets
# agree to within their broadcast steering (ns). Rows exist only for
# satellites decoded for positioning, with the week rollover resolved.
time_epoch_offset(row::SatelliteMeasurement) =
    row.week * SECONDS_PER_WEEK +
    row.system_start_epoch.second +
    round(Int, row.count_offset_to_gpst)

# Absolute solution epoch: clock-corrected `reference_time` plus the cached offset, or
# `nothing` without one.
function solution_time(time_epoch_offset, reference_time, primary_clock_bias)
    isnothing(time_epoch_offset) && return nothing
    corrected_reference_time = ustrip(s, reference_time) - primary_clock_bias / SPEED_OF_LIGHT
    TAITime(
        time_epoch_offset + floor(Int, corrected_reference_time),
        corrected_reference_time - floor(corrected_reference_time),
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Initialization

# Initial `x` and `P` from a scalar PVT fix; returns the primary clock index. The primary
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
        x[idxs.clock_drift] = pvt.relative_clock_drift * c
        P[idxs.clock_drift, idxs.clock_drift] = init_std_clock_drift^2
    end

    reference_system = pvt.reference_system
    # A reference system the layout lacks falls back to the first clock.
    primary_clock_index =
        isnothing(reference_system) ? 1 :
        something(time_system_index(layout.time_systems, reference_system), 1)
    for (index, time_system) in enumerate(layout.time_systems)
        state_index = idxs.clock_biases[index]
        if index == primary_clock_index
            P[state_index, state_index] = init_std_clock_bias^2
        elseif _fix_seeds_clock(pvt, layout, primary_clock_index, index)
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
    primary_clock_index
end

# Whether the fix estimated clock-bias state `index` (primary, or via an inter-system bias).
_fix_seeds_clock(pvt::PVTSolution, layout::NavFilterLayout, primary_clock_index, index) =
    index == primary_clock_index || haskey(pvt.inter_system_biases, layout.time_systems[index])

# Whether the fix seeds IFB state `index`: only against the layout's reference band.
function _fix_seeds_ifb(pvt::PVTSolution, layout::NavFilterLayout, index)
    band = layout.extra_bands[index]
    haskey(pvt.inter_frequency_biases, band) &&
        pvt.inter_frequency_biases[band].reference == layout.reference_bands[index]
end

# UERE (m) a scalar fix's covariance is scaled by in `seed_fix_covariance!`.
const FIX_PSEUDORANGE_STD = 1.0

# Smallest Cholesky pivot ratio (smallest / largest diagonal) taken as full rank. An
# exactly singular `HᵀH` (e.g. an IFB duplicating a single-band constellation's clock)
# often factors with a tiny positive pivot, giving 10¹⁵ m² variances; 10⁻⁶ admits
# condition numbers up to ~10¹².
const FIX_MIN_PIVOT_RATIO = 1e-6

# Overwrite the position/bias block of `P` with the fix's covariance `σ² (HᵀH)⁻¹`,
# `σ = FIX_PSEUDORANGE_STD`, for the fix's design `H` over `dense_bias_columns!`'s
# columns (state maps `clock_columns`, `ifb_columns`). A geometry-blind variance would
# trust a GDOP-40 fix like a GDOP-2 one (JuliaGNSS/TrackingLoops.jl#16), and the full
# block keeps the position–clock correlations. Unseeded bias states keep their variance,
# uncorrelated, though the position's variance still counts them as unknowns.
# `normal_matrix` is square scratch of `H`'s column count. Returns `false`, leaving `P`
# unchanged, for a rank-deficient `H`.
function seed_fix_covariance!(
    P,
    idxs::NavFilterIndices,
    layout::NavFilterLayout,
    pvt::PVTSolution,
    primary_clock_index,
    H,
    normal_matrix,
    clock_columns,
    ifb_columns,
)
    mul!(normal_matrix, H', H)
    factorization = cholesky!(Symmetric(normal_matrix); check = false)
    _has_full_rank(factorization) || return false
    covariance = LinearAlgebra.inv!(factorization)
    σ² = FIX_PSEUDORANGE_STD^2
    num_columns = size(covariance, 1)
    for column = 1:num_columns, row = 1:num_columns
        i = _fix_state_index(idxs, layout, pvt, primary_clock_index, clock_columns, ifb_columns, row)
        j = _fix_state_index(idxs, layout, pvt, primary_clock_index, clock_columns, ifb_columns, column)
        i == 0 || j == 0 || (P[i, j] = σ² * covariance[row, column])
    end
    true
end

# Whether a normal-matrix Cholesky succeeded with no negligible pivot
# (`FIX_MIN_PIVOT_RATIO`).
function _has_full_rank(factorization)
    issuccess(factorization) || return false
    factors = factorization.factors
    smallest, largest = Inf, 0.0
    for k in axes(factors, 1)
        smallest = min(smallest, factors[k, k])
        largest = max(largest, factors[k, k])
    end
    smallest > FIX_MIN_PIVOT_RATIO * largest
end

# The state behind `column` of the fix's dense design, `0` for an unseeded bias.
function _fix_state_index(idxs, layout, pvt, primary_clock_index, clock_columns, ifb_columns, column)
    column <= 3 && return idxs.pos[column]
    for (index, dense) in enumerate(clock_columns)
        dense == column - 3 || continue
        return _fix_seeds_clock(pvt, layout, primary_clock_index, index) ?
               idxs.clock_biases[index] : 0
    end
    num_clocks = count(!iszero, clock_columns)
    for (index, dense) in enumerate(ifb_columns)
        dense == column - 3 - num_clocks || continue
        return _fix_seeds_ifb(pvt, layout, index) ? idxs.ifb[index] : 0
    end
    0
end
