# ─────────────────────────────────────────────────────────────────────────────
# Vector tracking (VDLL / VDFLL): the navigation filter's model
#
# A central navigation Kalman filter closes all satellites' loops at once: each
# satellite's accumulated DLL (and FLL) discriminators become pseudorange (and
# pseudorange-rate) measurements, and the predicted line-of-sight dynamics are fed back
# as per-satellite NCO corrections — a VDLL, or a VDFLL with `use_pseudorange_rates`.
#
# The bias model follows `PositionVelocityTime`: one clock bias per GNSS time system
# (all driven by one oscillator drift) and one inter-frequency bias per band beyond a
# reference band; pseudoranges are corrected for the broadcast ionosphere and the
# Saastamoinen troposphere. Which biases an epoch can determine is decided per cycle as
# `decide_bias_layout` does (see `assess_bias_observability!`).
#
# The per-record `VectorPLLAndDLL` accumulates discriminators and applies the NCO
# corrections; this file holds the model above it (the cycle is in `vector/tracking.jl`):
# plain functions that allocate nothing once their buffers exist.
# ─────────────────────────────────────────────────────────────────────────────

"""
    VectorTracking(; use_pseudorange_rates = true,
                   motion_model_order = 2, clock_model_order = 2,
                   acceleration_noise_std = 5.0m/s^2,
                   h0 = 2e-19, hm2 = 2e-20,
                   ifb_noise_density = 0.01m/sqrt(1.0s),
                   insufficient_meas_timeout = 10.0s)

Configuration of the vector-tracking navigation filter of a [`VectorPLLAndDLL`](@ref).
The defaults suit a vehicle with a consumer-grade front end.

# Fields
- `use_pseudorange_rates`: `true` (VDFLL) also fuses the FLL pseudorange rates; with
  `false` (VDLL) only pseudoranges enter and the carrier corrections come purely from
  the predicted velocity and clock drift.
- `motion_model_order`: per-axis states — `1` position, `2` + velocity,
  `3` + acceleration.
- `clock_model_order`: `1` clock biases only (one per GNSS time system), `2` biases +
  one common drift.
- `acceleration_noise_std`: how hard the platform manoeuvres. `5 m/s²` suits road
  vehicles; ~`1 m/s²` pedestrians or ships, tens of `m/s²` aircraft. It means the same
  at every order; see `motion_noise_model` and `MANOEUVRE_TIME`.
- `h0`, `hm2`: oscillator Allan-variance coefficients ``h_0`` (s) and ``h_{-2}`` (1/s);
  ``h_0 = 2τσ_y²(τ)`` at short and ``h_{-2} = 3σ_y²(τ)/(2π²τ)`` at long averaging
  times. The defaults model a TCXO; an OCXO is far tighter, a bare crystal looser.
- `ifb_noise_density`: random-walk density of the inter-frequency biases (`m/√s`),
  covering the front end's thermal drift: about `0.01 m` per second by default. Raise
  it for bands that are less thermally coupled, lower it for a calibrated front end.
- `insufficient_meas_timeout`: how long the filter may coast on unsolvable epochs (fewer
  measurements than unknowns) before falling back to scalar tracking. Solution
  uncertainty is reported, not policed: see [`VTStatus`](@ref)'s `position_std`.
"""
struct VectorTracking
    use_pseudorange_rates::Bool
    motion_model_order::Int
    clock_model_order::Int
    acceleration_noise_std::typeof(1.0m / s^2)
    h0::Float64
    hm2::Float64
    ifb_noise_density::typeof(1.0m / sqrt(1.0s))
    insufficient_meas_timeout::typeof(1.0s)
end

function VectorTracking(;
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
    VectorTracking(
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
# run; IFB columns follow `band_ifb_layout`.
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
        Int[_time_system_index(time_systems, ts) for ts in time_system_per_group],
        collect(Int, ifb_indices),
        band_per_group,
        collect(Symbol, map(get_signal_id, signals)),
    )
end

# The position of `time_system` in `time_systems`, `0` if absent.
function _time_system_index(time_systems, time_system)
    for (index, other) in enumerate(time_systems)
        other === time_system && return index
    end
    0
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

NavFilterIndices(config::VectorTracking, layout::NavFilterLayout) = NavFilterIndices(
    config.motion_model_order,
    config.clock_model_order,
    num_clock_biases(layout),
    num_ifb(layout),
)

num_nav_states(config::VectorTracking, layout::NavFilterLayout) =
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
function nav_filter_process_model!(F, config::VectorTracking, idxs::NavFilterIndices, T)
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
# at the default order 2.
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
function motion_noise_model(config::VectorTracking, T)
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
    config::VectorTracking,
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

The navigation filter's process model `F`, `Q` for one integration interval, rebuilt in
place when the interval changes (`ensure_nav_filter_integration_time!`).
"""
mutable struct NavFilterModel
    integration_time::typeof(1.0s)
    const F::Matrix{Float64}
    const Q::Matrix{Float64}
    const idxs::NavFilterIndices
end

function NavFilterModel(config::VectorTracking, layout::NavFilterLayout, integration_time)
    n = num_nav_states(config, layout)
    model = NavFilterModel(
        integration_time,
        Matrix{Float64}(undef, n, n),
        Matrix{Float64}(undef, n, n),
        NavFilterIndices(config, layout),
    )
    rebuild_nav_filter_model!(model, config, integration_time)
end

function rebuild_nav_filter_model!(
    model::NavFilterModel,
    config::VectorTracking,
    integration_time,
)
    T = ustrip(s, integration_time)
    nav_filter_process_model!(model.F, config, model.idxs, T)
    nav_filter_process_noise_covariance!(model.Q, config, model.idxs, T)
    model.integration_time = integration_time
    model
end

# Rebuild `F`/`Q` whenever the measured interval changes at all: `reference_time`
# advances by it, so a nominal-interval model mispredicts the clock bias by `drift·ΔT`.
# Chunk quantisation alone overshoots by up to a chunk (e.g. 4 ms on 100 ms), metres at a
# TCXO's hundreds of m/s of drift. The rebuild allocates nothing.
function ensure_nav_filter_integration_time!(
    model::NavFilterModel,
    config::VectorTracking,
    integration_time,
)
    integration_time == model.integration_time && return model
    rebuild_nav_filter_model!(model, config, integration_time)
end

# The state propagated kinematically by `τ` seconds into `x_τ` (a plain copy for
# `τ = 0`); sizes NCO corrections that land `τ` after the cycle epoch.
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
    slot::Int              # position in that group's satellite vector
    prn::Int
    clock_bias_index::Int  # clock-bias state of this member's time system
    ifb_index::Int         # IFB state of this member's band (0 = reference)
    chip_length::Float64   # m
    wavelength::Float64    # m
    code_frequency::Float64 # Hz
    available::Bool        # usable as a measurement this cycle (in code lock)
    rate_available::Bool   # its FLL accumulated a reading this cycle (the rate row)
    time::Float64          # corrected transmit time (own system's time of week, s)
    # `time` on the GPS Time count (BDT reads 14 s below GPST). Used wherever times are
    # differenced across members (`reference_time`, pseudoranges), as in `calc_pvt`;
    # broadcast polynomials are evaluated at the own-scale `time`.
    time_gpst_count::Float64
    sat_position::SVector{3,Float64}
    sat_velocity::SVector{3,Float64}
    sat_clock_drift::Float64 # s/s
    pseudorange_rate::Float64 # measured λ·doppler (m/s)
    code_discriminator::Float64    # accumulated DLL output, moved to the epoch (chips)
    carrier_discriminator::Float64 # accumulated FLL output (Hz)
    # Variances (m², m²/s²) fused over the satellite's signals (`_member_measurements`).
    code_variance::Float64
    rate_variance::Float64
    cn0::Float64             # linear carrier-to-noise density (Hz)
    early_late_spacing::Float64 # chips
    coherent_integration_time::Float64 # s, per coherent dump (sets the DLL/FLL noise)
    # Broadcast offsets toward `CANDIDATE_HUB_SYSTEMS`, in that order.
    time_offsets::NTuple{3,BroadcastTimeOffset}
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

# Longest NCO delay (cycles) `code_phase_advance` covers; longer ones are rejected.
const MAX_LANDING_LEAD_CYCLES = 2.5

# Mean DLL discriminator (chips), advanced to the epoch by `code_phase_advance`; negated
# because `dll_disc`'s sense is reversed. Zero when nothing was accumulated.
function accumulated_code_discriminator(state::SatVectorPLLAndDLL, T)
    mean = _mean_code_discriminator(state)
    isnothing(mean) ? 0.0 : -mean + code_phase_advance(state, T)
end

# Mean FLL discriminator (Hz), `f_incoming − f_replica`; the rate measurement is
# `λ·(carrier_doppler + mean_fll)`.
#
# Deliberately NOT advanced by half the NCO correction, unlike the code: a loop lagging a
# Doppler ramp by `τ` has a final replica low by `Ḋ·τ` and a mean residual high by `Ḋ·τ`,
# so the sum already lands on the end-of-cycle Doppler (verified to <0.001 m/s at
# 5 m/s²; a `T/2` term would add a·T/2 = 0.25 m/s of error). NCO motion not backed by
# the signal (the filter correcting itself) leaves a non-accumulating ≈1% residue.
function accumulated_carrier_discriminator(state::SatVectorPLLAndDLL)
    mean = _mean_carrier_discriminator(state)
    isnothing(mean) ? 0.0 : ustrip(Hz, mean)
end

# Whether a member accumulated a discriminator this cycle (no full dump, e.g. admitted at
# the cycle's end). The zero the `accumulated_*` helpers substitute would be a confidently
# zero residual pulling the state toward the NCO, so such members are withheld from the
# measurements (still NCO-corrected). A record without a previous prompt has no FLL
# reading, so a member may lack only its rate row.
has_accumulated_code_discriminator(state::SatVectorPLLAndDLL) =
    !isnothing(_mean_code_discriminator(state))
has_accumulated_carrier_discriminator(state::SatVectorPLLAndDLL) =
    !isnothing(_mean_carrier_discriminator(state))

# Linear C/N₀ (Hz) from dB-Hz, floored to 1 against a degenerate weight; a NaN estimate
# (empty correlator) maps to 1 so it cannot corrupt the Kalman update.
function linear_cn0_floor(cn0_dbhz)
    cn0_linear = 10^(cn0_dbhz / 10)
    isnan(cn0_linear) ? 1.0 : max(cn0_linear, 1.0)
end

# Pseudorange (m) from receive and transmit times of week, folded across a week
# rollover.
pseudorange_from_tows(receive_tow, transmit_tow) =
    fold_week_crossover(receive_tow - transmit_tow) * SPEED_OF_LIGHT

# ─────────────────────────────────────────────────────────────────────────────
# Measurement prediction

# Clock/IFB columns of the members at `indices` as PositionVelocityTime's `BiasColumns`,
# so its `calc_ρ_hat!` / `calc_H!` model the pseudoranges.
function vt_bias_columns!(columns::BiasColumns, members, indices)
    resize!(columns.clock_bias_indices, length(indices))
    resize!(columns.ifb_indices, length(indices))
    for (k, j) in enumerate(indices)
        columns.clock_bias_indices[k] = members[j].clock_bias_index
        columns.ifb_indices[k] = members[j].ifb_index
    end
    columns
end

vt_bias_columns(members, layout::NavFilterLayout, indices = eachindex(members)) =
    vt_bias_columns!(
        BiasColumns(Int[], num_clock_biases(layout), Int[], num_ifb(layout)),
        members,
        indices,
    )

# The same columns restricted to the bias states these members occupy, renumbered
# densely, plus `primary_clock_index` in that numbering (1 if absent). For DOP: an
# all-zero column of a missing constellation makes `HᵀH` singular. `clock_used` /
# `ifb_used` are scratch of the layout's clock and IFB counts.
function dense_bias_columns!(
    clock_indices::Vector{Int},
    ifb_indices::Vector{Int},
    clock_used::Vector{Int},
    ifb_used::Vector{Int},
    members,
    indices,
    primary_clock_index::Int,
)
    fill!(clock_used, 0)
    fill!(ifb_used, 0)
    for j in indices
        clock_used[members[j].clock_bias_index] = 1
        members[j].ifb_index != 0 && (ifb_used[members[j].ifb_index] = 1)
    end
    num_clocks = 0
    for i in eachindex(clock_used)
        clock_used[i] == 0 && continue
        num_clocks += 1
        clock_used[i] = num_clocks
    end
    num_ifbs = 0
    for i in eachindex(ifb_used)
        ifb_used[i] == 0 && continue
        num_ifbs += 1
        ifb_used[i] = num_ifbs
    end
    resize!(clock_indices, length(indices))
    resize!(ifb_indices, length(indices))
    for (k, j) in enumerate(indices)
        clock_indices[k] = clock_used[members[j].clock_bias_index]
        ifb_indices[k] = members[j].ifb_index == 0 ? 0 : ifb_used[members[j].ifb_index]
    end
    primary =
        1 <= primary_clock_index <= length(clock_used) &&
            clock_used[primary_clock_index] != 0 ? clock_used[primary_clock_index] : 1
    BiasColumns(clock_indices, num_clocks, ifb_indices, num_ifbs), primary
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

# Predicted pseudorange rate (m/s), with the sign of the measured `λ · carrier_doppler`
# (positive while closing).
@inline function predict_pseudorange_rate(
    user_pos,
    user_vel,
    user_clock_drift,
    sat_pos,
    sat_vel,
    sat_clock_drift,
)
    # `e` points receiver→satellite, hence the user's velocity relative to the satellite.
    e = calc_line_of_sight(sat_pos, user_pos)
    dot(e, user_vel - sat_vel) + sat_clock_drift * SPEED_OF_LIGHT - user_clock_drift
end

# Measurement model `h!(y, x)`: candidates' pseudoranges, then the rates of `rate_rows`
# (positions among the candidates), then one row per hub constraint. A callable struct,
# not a closure, so the update is concrete and allocation-free.
struct VTMeasurementModel
    idxs::NavFilterIndices
    ξ::Vector{Float64}
    positions::Vector{SVector{3,Float64}}
    velocities::Vector{SVector{3,Float64}}
    clock_drifts::Vector{Float64}
    columns::BiasColumns
    rate_rows::Vector{Int}
    constraints::Vector{Tuple{Int,Int,Float64}}
end

function (model::VTMeasurementModel)(y, x)
    idxs = model.idxs
    num_sats = length(model.positions)
    position_and_bias_vector!(model.ξ, x, idxs)
    calc_ρ_hat!(y, model.positions, model.ξ, model.columns)
    offset = num_sats
    if !isempty(model.rate_rows)
        pos, vel, clock_drift = nav_filter_states(x, idxs)
        for (i, j) in enumerate(model.rate_rows)
            y[num_sats+i] = predict_pseudorange_rate(
                pos,
                vel,
                clock_drift,
                model.positions[j],
                model.velocities[j],
                model.clock_drifts[j],
            )
        end
        offset += length(model.rate_rows)
    end
    for (i, (state, hub_state, _)) in enumerate(model.constraints)
        y[offset+i] = x[idxs.clock_biases[state]] - x[idxs.clock_biases[hub_state]]
    end
    y
end

# Measurement-noise variances of one signal, from its linear C/N₀, coherent integration
# time `T_coh` (s), tap spacing `d` (chips) and the span `S = N·T_coh` (s) its `N`
# readings cover this cycle (shorter than the interval for a signal with gaps; see
# `_member_measurements`). Each is the per-dump discriminator variance propagated through
# the mean of the `N` dumps.
#
# Code: `dll_disc` is the noncoherent early-minus-late discriminator (Kaplan & Hegarty,
# "Understanding GPS: Principles and Applications", 2nd ed., Artech House 2006, §5.5.2;
# Betz & Kolodziejski, "Generalized Theory of Code Tracking with an Early-Late
# Discriminator, Part II", IEEE Trans. AES 45(4), 2009, pp. 1557-1564), with the
# open-loop `B_n = 1/(2·T_coh)`. Dump noise is white, so
#     var(mean) = d/(4·C/N0·S) · (1 + 2/((2 − d)·C/N0·T_coh))   [chips²]:
# the thermal term averages down over `S`, the squaring loss stays pinned to `T_coh`.
# Times `chip_length²` for metres. This is the BPSK model; for VEML `d` is the inner
# pair's spacing and the caller scales by `_dll_variance_factor`. Assumes `d < 2`.
# At a low `C/N₀ · T_coh` it overstates every signal's variance (the normalised
# discriminators saturate: about 20× at `C/N₀ · T_coh ≈ 1` for a 1 ms GPS L1 C/A dump);
# left as is, since the overstatement is common to the signals it weighs and errs
# pessimistic.

# VEML-to-BPSK-model DLL variance ratio for the BOC(1,1) family (default ±0.15/±0.6
# chip taps), a first-order correction: a sample-level simulation of `dll_disc` (25 MHz,
# zero code error) gave 0.36 for Galileo E1B as CBOC and 0.43 as BOC(1,1) at 40–45 dB-Hz,
# 0.33–0.45 down to 30 dB-Hz, and 0.94 for a GPS L1 C/A EPL dump. CBOC's BOC(6,1) part
# makes it depend on the front-end bandwidth, which the simulation did not limit.
const _VEML_DLL_VARIANCE_FACTOR = 0.4

_dll_variance_factor(::AbstractCorrelator) = 1.0
_dll_variance_factor(::VeryEarlyPromptLateCorrelator) = _VEML_DLL_VARIANCE_FACTOR

function _pseudorange_noise_variance(cn0, coherent_integration_time, d, chip_length, span)
    cn0_tcoh = cn0 * coherent_integration_time
    squaring_loss = 1 + 2 / ((2 - d) * cn0_tcoh)
    d / (4 * span * cn0) * squaring_loss * chip_length^2
end

# Rate: the ATAN FLL (`fll_disc`) phase jitter per dump is
#     σ_φ² = 1/(2·C/N0·T_coh)·(1 + 1/(2·C/N0·T_coh))   [rad²].
# Each dump is `(θ_k − θ_{k-1})/(2π·T_coh)`, so the mean telescopes to
# `(θ_N − θ_0)/(2π·S)`:
#     var(mean) = 2·σ_φ² / (2π·S)²   [Hz²],   times λ² for (m/s)².
# Consecutive cycles share a boundary phase (`previous_prompt` chains across them) with
# opposite signs: cov = −σ_φ²/(2π·S)² against var = 2·σ_φ²/(2π·S)², so their rate
# errors have lag-1 correlation −1/2, which the filter cannot represent. Being
# negative, it only makes the filter pessimistic about the rates, so `R` is deliberately
# not inflated; the ≈ −0.5 lag-1 autocorrelation of the rate residuals is by construction
# and must not be tuned against.
function _pseudorange_rate_noise_variance(cn0, coherent_integration_time, wavelength, span)
    cn0_tcoh = cn0 * coherent_integration_time
    sigma_phi2 = 1 / (2 * cn0_tcoh) * (1 + 1 / (2 * cn0_tcoh))
    wavelength^2 * sigma_phi2 / (2 * π^2 * span^2)
end

# ─────────────────────────────────────────────────────────────────────────────
# Bias observability

# Accuracy (m) credited to a broadcast GGTO / BGTO when it collapses a clock onto the
# hub's: the Galileo OS SDD (Issue 1.2, Nov 2021) commits the GGTO to < 20 ns (95 %),
# σ ≈ 10 ns ≈ 3 m; the BGTO is credited the same. Too coarse to use when the geometry
# observes the offset, hence the collapse only for unsupported independent layouts.
const HUB_OFFSET_STD = 3.0

# The bias-layout decision for one measurement set, mirroring `decide_bias_layout`.
#
# `num_unknowns`: 3 + the clock biases + the *observable* IFBs of the layout in force
# (merged under a collapse) — the measurement count needed. `num_distinct_sats_required`
# (3 + clock biases) is the independent second condition on distinct satellites: a
# second band of a satellite adds no line of sight (three satellites on two bands pass
# the count but not this). Neither condition is sufficient; residual degeneracy is
# reported via `VTStatus`'s `position_std`.
#
# A collapse leaves one `(state, hub_state, isb)` per collapsed system in the workspace's
# `hub_offset_constraints`: the pseudo-measurement `x[state] - x[hub_state] = isb`
# rather than a merged column, so the state dimension stays fixed and the clock stays
# available for epochs that observe it.
struct BiasObservability
    num_unknowns::Int
    num_distinct_sats_required::Int
    num_distinct_sats::Int
end

# The scratch of `assess_bias_observability!`, and where it leaves the hub
# constraints.
struct ObservabilityWorkspace
    time_systems::Vector{SupportedTimeSystem}
    merged_time_systems::Vector{SupportedTimeSystem}
    bands::Vector{Symbol}
    scratch::BandLayoutScratch{Symbol}
    ifb_indices::Vector{Int}
    extra_bands::Vector{Symbol}
    reference_bands::Vector{Symbol}
    collapsed::Vector{SupportedTimeSystem}
    hub_offset_constraints::Vector{Tuple{Int,Int,Float64}}
end

function ObservabilityWorkspace(max_measurements::Integer = 0)
    ws = ObservabilityWorkspace(
        SupportedTimeSystem[],
        SupportedTimeSystem[],
        Symbol[],
        BandLayoutScratch{Symbol}(),
        Int[],
        Symbol[],
        Symbol[],
        SupportedTimeSystem[],
        Tuple{Int,Int,Float64}[],
    )
    for v in (ws.time_systems, ws.merged_time_systems, ws.bands, ws.ifb_indices)
        sizehint!(v, max_measurements)
    end
    ws
end

# Whether a measurement set meets both conditions of `BiasObservability`.
is_epoch_solvable(obs::BiasObservability, num_measurements) =
    num_measurements >= obs.num_unknowns &&
    obs.num_distinct_sats >= obs.num_distinct_sats_required

# The number of distinct entries of `values`, without a set.
function _num_distinct(values)
    count = 0
    for j in eachindex(values)
        seen = false
        for i = firstindex(values):(j-1)
            if values[i] === values[j]
                seen = true
                break
            end
        end
        seen || (count += 1)
    end
    count
end

# `(num_unknowns, num_distinct_sats_required, num_components)` of one measurement set
# (see `BiasObservability`), read off the *epoch's own* (constellation × band) coverage
# graph, not the configured layout, so an IFB unobservable this epoch is not counted.
function epoch_bias_unknowns!(ws::ObservabilityWorkspace, time_systems, bands)
    num_components = band_ifb_layout!(
        ws.ifb_indices,
        ws.extra_bands,
        ws.reference_bands,
        ws.scratch,
        time_systems,
        bands,
    )
    num_distinct_sats_required = 3 + _num_distinct(time_systems)
    num_distinct_sats_required + length(ws.extra_bands),
    num_distinct_sats_required,
    num_components
end

# Decide this cycle's bias layout as `decide_bias_layout` does: independent biases when
# the coverage graph is connected and the epoch solvable, otherwise the broadcast clock
# collapse, which removes a clock unknown and reconnects a disjoint band split. Its
# constraints are left in `ws.hub_offset_constraints`.
function assess_bias_observability!(
    ws::ObservabilityWorkspace,
    layout::NavFilterLayout,
    members,
    candidate_indices,
)
    num_measurements = length(candidate_indices)
    constraints = empty!(ws.hub_offset_constraints)
    time_systems = resize!(ws.time_systems, num_measurements)
    bands = resize!(ws.bands, num_measurements)
    for (k, j) in enumerate(candidate_indices)
        time_systems[k] = layout.time_systems[members[j].clock_bias_index]
        bands[k] = layout.band_by_group[members[j].group]
    end
    # Distinct satellites by `(clock-bias state ≙ time system, PRN)`.
    num_distinct_sats = 0
    for (k, j) in enumerate(candidate_indices)
        seen = false
        for i = 1:(k-1)
            other = members[candidate_indices[i]]
            if other.clock_bias_index == members[j].clock_bias_index &&
               other.prn == members[j].prn
                seen = true
                break
            end
        end
        seen || (num_distinct_sats += 1)
    end

    num_unknowns, num_distinct_required, num_components =
        epoch_bias_unknowns!(ws, time_systems, bands)
    independent = BiasObservability(num_unknowns, num_distinct_required, num_distinct_sats)
    if num_components == 1 && is_epoch_solvable(independent, num_measurements)
        return independent
    end

    # Collapse onto the first hub (GPST, GST, BDT, as `decide_bias_layout`) that yields a
    # constraint, for systems whose clock and the hub's both have measurements.
    for hub_offset_index in eachindex(CANDIDATE_HUB_SYSTEMS)
        hub = _candidate_hub(hub_offset_index)
        hub_state = _time_system_index(layout.time_systems, hub)
        (hub_state == 0 || _time_system_index(time_systems, hub) == 0) && continue
        collapsed = empty!(ws.collapsed)
        for state in eachindex(layout.time_systems)
            time_system = layout.time_systems[state]
            (time_system === hub || _time_system_index(time_systems, time_system) == 0) &&
                continue
            # The offset is constellation-wide: the first decoded copy serves (as in
            # `calc_hub_range_offsets`).
            offset_member = 0
            for j in candidate_indices
                if members[j].clock_bias_index == state &&
                   members[j].time_offsets[hub_offset_index].available
                    offset_member = j
                    break
                end
            end
            offset_member == 0 && continue
            member = members[offset_member]
            # Δt = (system time) − (hub time), so the clock sits −c·Δt from the hub's,
            # the sign of `decide_bias_layout`'s `inter_system_biases`.
            isb =
                -SPEED_OF_LIGHT *
                calc_steering_offset(member.time_offsets[hub_offset_index], member.time)
            push!(constraints, (state, hub_state, isb))
            push!(collapsed, time_system)
        end
        isempty(constraints) && continue
        # Recount on the merged graph: reconnected components turn former reference
        # bands into observable IFBs again.
        merged = resize!(ws.merged_time_systems, num_measurements)
        for k in eachindex(time_systems)
            merged[k] =
                _time_system_index(collapsed, time_systems[k]) != 0 ? hub : time_systems[k]
        end
        merged_unknowns, merged_distinct_required, _ =
            epoch_bias_unknowns!(ws, merged, bands)
        # Returned even if still unsolvable: the merge only lowers both requirements, so
        # solvability agrees with `decide_bias_layout`, and the constraints are free
        # information.
        return BiasObservability(
            merged_unknowns,
            merged_distinct_required,
            num_distinct_sats,
        )
    end

    independent
end

# `CANDIDATE_HUB_SYSTEMS[i]`, spelled out so the type is the closed union rather than
# whatever indexing the heterogeneous tuple infers.
_candidate_hub(i)::SupportedTimeSystem = i == 1 ? GPST() : i == 2 ? GST() : BDT()

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

# NCO corrections of one member: the predicted minus the replica's pseudorange (rate),
# as code / carrier frequency offsets removing it over the next interval `T`; the carrier
# one feeds the FLL branch. Predict from the *updated* state only: adding the update's
# projected state correction too would double-count it (loop gain 2, oscillation).
nco_code_correction(predicted_pseudorange, measured_pseudorange, code_frequency, T) =
    -(predicted_pseudorange - measured_pseudorange) * code_frequency / (T * SPEED_OF_LIGHT)

nco_carrier_correction(predicted_pseudorange_rate, measured_pseudorange_rate, wavelength) =
    (predicted_pseudorange_rate - measured_pseudorange_rate) / wavelength

# Post-fit residuals of one member, from predictions at the *updated* state:
# `(pseudorange in m, range rate in m/s)`, observed − computed as `calc_pvt` (and RTKLIB)
# report them, sign included. Hence opposite subtraction orders: the rate residual is
# `h(x) - z` because this loop measures `+λ · carrier_doppler` (positive while closing)
# while `calc_pvt` residuates the geometric range rate (positive while receding).
#
# A well-tracked member's residuals reduce to its own discriminators; one the solution
# predicts poorly (diverging, or out of lock) keeps a large residual, so members outside
# the update are reported too. Under VDLL the rate residual tests the solution against
# rates it never fused.
function vt_post_fit_residuals(
    member::VTMember,
    measured_pseudorange,
    predicted_pseudorange,
    predicted_pseudorange_rate,
)
    (
        measured_pseudorange + member.code_discriminator * member.chip_length -
        predicted_pseudorange
    ) * m,
    (
        predicted_pseudorange_rate - member.pseudorange_rate -
        member.carrier_discriminator * member.wavelength
    ) * (m / s)
end

# ─────────────────────────────────────────────────────────────────────────────
# PVT solution from the navigation filter

# The clock-bias state to report against: kept while its system has an included
# measurement (no flicker), else re-picked as `PositionVelocityTime` does — GPST if
# present, else the system with most measurements (ties by state order). Picking only
# present systems keeps a week number, and so a timestamp, available.
function report_primary_clock_index(
    layout::NavFilterLayout,
    members,
    included_indices,
    current_index,
)
    isempty(included_indices) && return current_index
    for j in included_indices
        members[j].clock_bias_index == current_index && return current_index
    end
    gpst_index = _time_system_index(layout.time_systems, GPST())
    if gpst_index != 0
        for j in included_indices
            members[j].clock_bias_index == gpst_index && return gpst_index
        end
    end
    best_index = 0
    best_count = 0
    for index in eachindex(layout.time_systems)
        n = 0
        for j in included_indices
            members[j].clock_bias_index == index && (n += 1)
        end
        if n > best_count
            best_index, best_count = index, n
        end
    end
    best_index
end

# The epoch offset for the next cycle, advanced by a week when `reference_time` has just
# wrapped (else every later `pvt.time` is a week early). Only an already cached offset
# is advanced; a freshly resolved one comes from a decoder that has rolled over too.
rolled_over_time_epoch_offset(resolved, cached, week_rollover) =
    week_rollover && !isnothing(cached) ? cached + SECONDS_PER_WEEK : resolved

# The constant part of the solution's epoch from a primary-system measurement row: week
# in seconds + start epoch + the scale offset to the GPS Time count `reference_time`
# runs on (`VTMember.time_gpst_count`). Resolved once and cached, so a solution cannot
# lose its timestamp; the cache survives a change of primary clock, since all systems'
# offsets then agree to within their broadcast steering (ns).
time_epoch_offset(row::SatelliteMeasurement) =
    row.week * SECONDS_PER_WEEK +
    row.system_start_epoch.second +
    round(Int, row.count_offset_to_gpst)

# Absolute solution epoch: clock-corrected `reference_time` plus the cached offset, or
# `nothing` without one.
function vt_time(time_epoch_offset, reference_time, primary_clock_bias)
    isnothing(time_epoch_offset) && return nothing
    corrected_reference_time =
        ustrip(s, reference_time) - primary_clock_bias / SPEED_OF_LIGHT
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
function initial_nav_state!(
    x,
    P,
    layout::NavFilterLayout,
    idxs::NavFilterIndices,
    pvt::PVTSolution,
)
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
    primary_clock_index =
        isnothing(reference_system) ? 0 :
        _time_system_index(layout.time_systems, reference_system)
    primary_clock_index == 0 && (primary_clock_index = 1)
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
            x[state_index] =
                ustrip(m, pvt.inter_frequency_biases[layout.extra_bands[index]].value)
            P[state_index, state_index] = init_std_seeded_ifb^2
        else
            P[state_index, state_index] = init_std_unseeded_ifb^2
        end
    end
    primary_clock_index
end

# Whether the fix estimated clock-bias state `index` (primary, or via an inter-system bias).
_fix_seeds_clock(pvt::PVTSolution, layout::NavFilterLayout, primary_clock_index, index) =
    index == primary_clock_index ||
    haskey(pvt.inter_system_biases, layout.time_systems[index])

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
# columns (state maps `clock_used`, `ifb_used`). A geometry-blind variance would trust a
# GDOP-40 fix like a GDOP-2 one (JuliaGNSS/TrackingLoops.jl#16), and the full block keeps
# the position–clock correlations. Unseeded bias states keep their variance.
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
    clock_used,
    ifb_used,
)
    mul!(normal_matrix, H', H)
    factorization = cholesky!(Symmetric(normal_matrix); check = false)
    _has_full_rank(factorization) || return false
    covariance = LinearAlgebra.inv!(factorization)
    σ² = FIX_PSEUDORANGE_STD^2
    num_columns = size(covariance, 1)
    for column = 1:num_columns, row = 1:num_columns
        i = _fix_state_index(
            idxs,
            layout,
            pvt,
            primary_clock_index,
            clock_used,
            ifb_used,
            row,
        )
        j = _fix_state_index(
            idxs,
            layout,
            pvt,
            primary_clock_index,
            clock_used,
            ifb_used,
            column,
        )
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
function _fix_state_index(
    idxs,
    layout,
    pvt,
    primary_clock_index,
    clock_used,
    ifb_used,
    column,
)
    column <= 3 && return idxs.pos[column]
    for (index, dense) in enumerate(clock_used)
        dense == column - 3 || continue
        return _fix_seeds_clock(pvt, layout, primary_clock_index, index) ?
               idxs.clock_biases[index] : 0
    end
    num_clocks = count(!iszero, clock_used)
    for (index, dense) in enumerate(ifb_used)
        dense == column - 3 - num_clocks || continue
        return _fix_seeds_ifb(pvt, layout, index) ? idxs.ifb[index] : 0
    end
    0
end
