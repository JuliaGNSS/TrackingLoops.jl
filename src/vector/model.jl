# ─────────────────────────────────────────────────────────────────────────────
# Vector tracking (VDLL / VDFLL): the navigation filter's model
#
# In scalar tracking every satellite closes its own code/carrier loops from its
# own discriminators. In vector tracking a central navigation Kalman filter
# closes all loops at once: each satellite's accumulated DLL (and FLL)
# discriminator outputs become pseudorange (and pseudorange-rate) measurements,
# the filter fuses them into a position/velocity/clock state, and the predicted
# line-of-sight dynamics are fed back as per-satellite NCO corrections — a
# vector delay lock loop (VDLL), or a vector delay/frequency lock loop (VDFLL)
# when the pseudorange rates are measured too (`use_pseudorange_rates`). Weak
# or briefly obscured satellites are carried through outages by the collective
# solution instead of losing lock on their own.
#
# The receiver may track several constellations and frequency bands at once;
# the filter follows `PositionVelocityTime`'s bias model: one receiver clock
# bias per GNSS time system (all driven by the one oscillator's clock drift)
# and one receiver inter-frequency bias per band beyond a reference band. The
# measured pseudoranges are corrected for the broadcast ionospheric model and
# the Saastamoinen tropospheric delay, as in the scalar PVT solve. Which of
# those biases a given epoch can actually determine is decided per cycle, the
# way `decide_bias_layout` decides it for the scalar solve — including the
# broadcast-offset collapse of clocks onto a hub system's (through the GGTO /
# BGTO) when the measurements do not support them independently (see
# `assess_bias_observability!`).
#
# The loop split: the per-record `VectorPLLAndDLL` accumulates each satellite's
# discriminators and applies the NCO corrections; everything here — the
# navigation filter, measurement construction and the loop-closure maths — and
# the cycle in `vector/tracking.jl` sit above it. This file holds the model:
# plain functions of the configuration, the layout and the per-cycle members,
# none of which allocates once its buffers exist.
# ─────────────────────────────────────────────────────────────────────────────

"""
    VectorTracking(; use_pseudorange_rates = true,
                   motion_model_order = 2, clock_model_order = 2,
                   acceleration_noise_std = 5.0m/s^2,
                   h0 = 2e-19, hm2 = 2e-20,
                   ifb_noise_density = 0.01m/sqrt(1.0s),
                   insufficient_meas_timeout = 10.0s)

Configuration of the vector-tracking navigation filter, for a
[`VectorPLLAndDLL`](@ref): the platform and the receiver's oscillator. The
defaults suit a vehicle carrying a consumer-grade front end.

# Fields
- `use_pseudorange_rates`: with `true` (VDFLL) the navigation filter measures
  both the pseudoranges (DLL discriminators) and the pseudorange rates (FLL
  discriminators on the carrier Doppler); with `false` (VDLL) only the
  pseudoranges enter the filter — the carrier NCO corrections are then derived
  purely from the filter's velocity/clock-drift prediction.
- `motion_model_order`: order of the per-axis motion model — `1` position only,
  `2` position + velocity, `3` position + velocity + acceleration.
- `clock_model_order`: order of the receiver clock model — `1` biases only,
  `2` biases + a common drift. One clock bias is modelled per GNSS time
  system, all integrating the single oscillator drift.
- `acceleration_noise_std`: white-acceleration process-noise standard deviation
  driving the motion model — how hard the platform manoeuvres. The `5 m/s²`
  default suits automotive / vehicular dynamics (≈0.5 g of manoeuvring); use
  ~`1 m/s²` for pedestrian or ship dynamics and tens of `m/s²` for aircraft or
  launch vehicles. It means the same thing at every `motion_model_order`: the
  process noise always models the platform's first *unmodelled* derivative, so
  the orders that do not model the acceleration (1 and 3) derive their velocity
  and jerk figures from this one through the manoeuvre time constant
  `MANOEUVRE_TIME` — see `motion_noise_model`, which is also where to change that
  constant for a platform whose manoeuvres are much shorter or longer than the
  couple of seconds a road vehicle takes.
- `h0`, `hm2`: Allan-variance coefficients ``h_0`` (seconds) and ``h_{-2}``
  (1/seconds) of the receiver oscillator, sizing the clock process noise —
  smaller coefficients for a more stable oscillator. They follow from the
  oscillator's Allan deviation: ``σ_y²(τ) = h_0 / 2τ`` at short averaging times
  gives ``h_0 = 2τσ_y²(τ)``, and ``σ_y²(τ) = (2π²/3)·h_{-2}·τ`` at long ones
  gives ``h_{-2} = 3σ_y²(τ)/(2π²τ)``. The defaults model a TCXO-grade clock (a
  typical consumer/automotive receiver oscillator); an OCXO is orders of magnitude
  tighter, a bare crystal looser.
- `ifb_noise_density`: random-walk process-noise density of the inter-frequency
  biases, in `m/√s`. Per-band RF-chain delays are nearly constant, so this only
  has to cover thermal drift of the front end: the default lets a bias wander
  about `0.01 m` in a second of elapsed time. Raise it for a front end whose
  bands are less thermally coupled, lower it for one that is stable or
  externally calibrated.
- `insufficient_meas_timeout`: how long the navigation filter may coast on
  epochs it cannot solve before vector tracking is abandoned and the satellites
  fall back to scalar tracking. An epoch counts as unsolvable when it has fewer
  measurements than the bias layout in force has unknowns. How certain the filter
  is of the solution it did produce is reported rather than policed — see
  [`VTStatus`](@ref)'s `position_std`.
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

# Which clock-bias and inter-frequency-bias state each signal group maps to.
# Established once from the *configured* signals (not from the currently
# tracked satellites), so the filter's state dimension is fixed for the whole
# run; a constellation without current measurements simply coasts on its
# process noise. The inter-frequency-bias columns follow `band_ifb_layout`,
# which creates a column only where the bias is observable (a band stranded
# alone on its constellation folds its delay into that constellation's clock
# instead). Groups are addressed by their position in the signal tuple.
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

# Indices of the navigation-filter state vector
# `[x, ẋ, (ẍ), y, ẏ, (ÿ), z, ż, (z̈), clock_bias_1..M, (clock_drift), ifb_1..B]`
# — positions and clock/inter-frequency biases in metres, derivatives in m/s
# (m/s²), so the clock terms are directly commensurable with the pseudorange
# measurements.
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

# Read (position, velocity, common clock drift) out of a state vector,
# substituting zeros for unmodelled derivatives. The per-system clock biases
# and per-band IFBs are indexed directly via `idxs.clock_biases` / `idxs.ifb`.
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

# Discrete-time process model F: a constant-velocity (or -position /
# -acceleration) block per axis; every clock bias integrates the single
# oscillator drift; the inter-frequency biases are constant.
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

# The per-axis process-noise gain vector `Γ` of one motion model order, and the standard
# deviation of the scalar per-interval noise driving it: the axis block of `Q` is `Γ Γᵀ σ²`,
# one unmodelled derivative acting over the interval and propagated into every modelled
# state (the discrete white-noise model of the order above the highest state — Bar-Shalom,
# Li & Kirubarajan, "Estimation with Applications to Tracking and Navigation", Wiley 2001,
# §6.3.2).
#
# Which derivative is unmodelled is what the order changes, and with it what `σ` physically
# is:
#
#   order 1 (p)     → the platform's *velocity* is unmodelled: Γ = [T],             σ = σ_v
#   order 2 (p,v)   → its *acceleration* is:                   Γ = [T²/2, T],       σ = σ_a
#   order 3 (p,v,a) → its *jerk* is:                           Γ = [T³/6, T²/2, T], σ = σ_j
#
# The configuration carries one dynamics figure, `acceleration_noise_std`, so the other two
# are derived from it through the manoeuvre time constant `MANOEUVRE_TIME` (τ) below. That
# keeps `acceleration_noise_std` meaning the same thing ("how hard the platform manoeuvres")
# at every order — and, unlike a fixed rescaling factor, it stays right when the filter
# interval changes, because each order's `Γ` already carries the `T` dependence its own
# derivative implies. A constant factor is only correct at one `T`: matching a fixed
# velocity or jerk with the order-2 gain vector needs a factor going as `1/T`.
#
# How long one manoeuvre lasts. A manoeuvre of this duration changes the velocity by `σ_a·τ`
# — what order 1, modelling no velocity, has to absorb — and is built out of a jerk of
# `σ_a/τ`, what order 3, modelling the acceleration, has to absorb. Two seconds is a road
# vehicle's: a lane change, or a 0.5 g stop from 36 km/h, which is `acceleration_noise_std`'s
# own default read as a manoeuvre. Deliberately a constant and not a keyword: it is inert at
# the default `motion_model_order = 2`, where the acceleration itself is the unmodelled
# derivative, so as a keyword it would sit on everyone's constructor to serve only the two
# rarely-chosen orders. A platform whose manoeuvres are much shorter or longer changes it
# here.
const MANOEUVRE_TIME = 2.0s

# `Γ` comes back padded with zeros to three entries; only the first
# `motion_model_order` are the gain vector.
function motion_noise_model(config::VectorTracking, T)
    acc_std = ustrip(m / s^2, config.acceleration_noise_std)
    τ = ustrip(s, MANOEUVRE_TIME)
    config.motion_model_order == 1 && return SVector(T, 0.0, 0.0), acc_std * τ
    config.motion_model_order == 2 && return SVector(T^2 / 2, T, 0.0), acc_std
    SVector(T^3 / 6, T^2 / 2, T), acc_std / τ
end

# Process noise Q: the motion noise of `motion_noise_model` above, a clock model
# from the oscillator's Allan-variance coefficients, and a small random walk on
# the inter-frequency biases. All clock biases ride the *same* oscillator, so the
# drift random walk (Sg) is fully correlated across them; only the white
# frequency noise (Sf) is applied per bias.
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

    # Allan variance to clock process noise (biases in m, drift in m/s):
    # Sf [m²/s] from white frequency noise h0, Sg [m²/s³] from the frequency
    # random walk h-2.
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

    # The inter-frequency biases are near-constant RF-chain delays; their random walk only
    # has to cover the front end's thermal drift (`ifb_noise_density`, in m/√s).
    ifb_density = ustrip(m / sqrt(s), config.ifb_noise_density)
    for ifb in idxs.ifb
        Q[ifb, ifb] = ifb_density^2 * T
    end
    Q
end

"""
    NavFilterModel

The navigation filter's linear process model for one integration interval: `F`
and `Q`, rebuilt in place whenever the measured interval changes (see
`ensure_nav_filter_integration_time!`).
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

function rebuild_nav_filter_model!(model::NavFilterModel, config::VectorTracking, integration_time)
    T = ustrip(s, integration_time)
    nav_filter_process_model!(model.F, config, model.idxs, T)
    nav_filter_process_noise_covariance!(model.Q, config, model.idxs, T)
    model.integration_time = integration_time
    model
end

# The navigation filter's update interval is measured each cycle, and the process model
# must propagate the state by exactly that interval: `reference_time` advances by it, so a
# model kept at the nominal interval mispredicts every cycle by the difference — the
# clock bias by `drift·ΔT`, the position by `v·ΔT`. The normal deviation is the
# chunk-size quantisation (a cycle runs on the first chunk boundary at or past the
# nominal interval, so it overshoots by up to one chunk, e.g. 4 ms on 100 ms), and with a
# TCXO's hundreds of m/s of clock drift that alone is metres against a clock process noise
# of centimetres. So `F`/`Q` are rebuilt whenever the interval changes at all; the rebuild
# is cheap and allocates nothing.
function ensure_nav_filter_integration_time!(
    model::NavFilterModel,
    config::VectorTracking,
    integration_time,
)
    integration_time == model.integration_time && return model
    rebuild_nav_filter_model!(model, config, integration_time)
end

# The state propagated kinematically by `τ` seconds, into `x_τ`: position
# `+ v·τ (+ a·τ²/2)`, velocity `+ a·τ`, each clock bias `+ drift·τ`, IFBs unchanged.
# With `τ = 0` it is a plain copy (`x + v·0.0 == x`). This is what the NCO
# corrections are sized with when they land `τ` after the cycle epoch.
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

# Everything the navigation filter needs to know about one vector-loop member
# for one cycle. Gathered once per cycle; all per-satellite buffers downstream
# (measurements, predictions, noise) are aligned with the member vector. Plain
# data only, so a member vector is stored inline and refilled without
# allocating.
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
    time::Float64          # corrected transmit time (own system's time of week, s)
    # The same instant expressed on the GPS Time count: `time` minus the system's
    # defined scale offset (0 for GPST/GST, −14 s for BDT — a BDT second-of-week reads
    # 14 s below the GPS time of week for the same instant). Everything that
    # *differences* times across members — `reference_time` and the pseudoranges —
    # uses this field, mirroring `calc_pvt`'s `calc_time_scale_offsets`; everything
    # that evaluates a broadcast polynomial (ephemeris, clock, `calc_steering_offset`)
    # keeps the own-scale `time`.
    time_gpst_count::Float64
    sat_position::SVector{3,Float64}
    sat_velocity::SVector{3,Float64}
    sat_clock_drift::Float64 # s/s
    pseudorange_rate::Float64 # measured λ·doppler (m/s)
    code_discriminator::Float64    # accumulated DLL output, moved to the epoch (chips)
    carrier_discriminator::Float64 # accumulated FLL output (Hz)
    cn0::Float64             # linear carrier-to-noise density (Hz)
    early_late_spacing::Float64 # chips
    coherent_integration_time::Float64 # s, per coherent correlator dump (sets DLL/FLL noise)
    # The member's broadcast offsets toward `CANDIDATE_HUB_SYSTEMS`, in that order,
    # as its measurement row carries them.
    time_offsets::NTuple{3,BroadcastTimeOffset}
end

# The code phase (chips) the vector loop's code correction steered out over the second
# half of the cycle, `∫_{T/2}^{T} c(t) dt`: it moves the mean code discriminator, which
# measures the average delay error while the NCO was already steering it out and so sits
# at mid-cycle, to the epoch the code observable is read at. `c₀, c₁, c₂` are the code
# corrections of the last three cycles, newest first, and `L` how long after its cycle's
# epoch each reached the replica: `c₀` at `L` after the last epoch, `c₁` at `L − T` and
# `c₂` at `L − 2T`. Without an NCO delay `L = 0`, and this is exactly `c₀·T/2`
# (`x − y·0.0 == x`). With `T ≤ L ≤ 1.5T` the half-cycle ran on `c₁` until it landed,
# and with `1.5T ≤ L ≤ 2.5T` on `c₂` and then `c₁`. A longer delay would need a fourth
# correction; a satellite taking up a cycle rejects it (`MAX_LANDING_LEAD_CYCLES`).
function code_phase_advance(state::SatVectorPLLAndDLL, T)
    c₀, c₁, c₂ = map(c -> ustrip(Hz, c), state.code_freq_update_history)
    L = ustrip(s, state.code_update_landing_lead)
    c₀ * T / 2 - (c₀ - c₁) * clamp(L - T / 2, 0.0, T / 2) -
    (c₁ - c₂) * clamp(L - 3T / 2, 0.0, T / 2)
end

# The longest NCO delay, in navigation cycles, `code_phase_advance` covers with the three
# corrections it keeps.
const MAX_LANDING_LEAD_CYCLES = 2.5

# Mean DLL discriminator (chips) over the last filter interval, advanced to the epoch by
# `code_phase_advance`. `dll_disc`'s sense is reversed relative to its own observable,
# hence the negated mean. Zero when nothing was accumulated (see
# `has_accumulated_discriminators`).
function accumulated_code_discriminator(state::SatVectorPLLAndDLL, T)
    mean = _mean_code_discriminator(state)
    isnothing(mean) ? 0.0 : -mean + code_phase_advance(state, T)
end

# Mean FLL discriminator (Hz) over the last filter interval. `fll_disc` measures
# f_incoming − f_replica, so the true carrier Doppler is the replica Doppler plus this
# residual; the pseudorange-rate measurement is therefore `λ · carrier_doppler + λ ·
# mean_fll` — the discriminator enters with its own sign. (The code discriminator carries
# the opposite sign because `dll_disc`'s sense is reversed relative to its own observable.)
#
# Deliberately NOT corrected by half the NCO frequency correction, unlike
# `accumulated_code_discriminator` — the asymmetry is load-bearing, not an oversight.
# The code observable is a *delay* read at the end of the cycle combined with the *mean*
# delay error over it, so the two refer to epochs `T/2` apart and the mean has to be
# advanced to the end. The rate observable has no such gap: the carrier Doppler is the
# replica frequency at the end of the cycle, and a loop lagging a Doppler ramp by `τ` has
# both a final replica low by `Ḋ·τ` and a mean residual high by `Ḋ·τ` — the lag cancels
# exactly, so `carrier_doppler + mean_fll` already lands on the *end-of-cycle* incoming
# Doppler, which is the epoch the navigation filter's predicted state is referenced to.
# Verified: under a 5 m/s² line-of-sight acceleration the sum tracks the end-of-cycle
# Doppler to <0.001 m/s while sitting 0.25 m/s (= a·T/2) away from the mid-cycle value.
# Adding a `T/2` term here would introduce exactly that 0.25 m/s of error.
#
# The cancellation is first-order in the loop's lag, so it is exact only for NCO motion
# that tracks real Doppler motion. NCO motion the incoming signal does not back — the
# navigation filter correcting its own past error — leaves a residue of ≈1% of the applied
# correction, whose sign follows the carrier loop's transient rather than the correction, so
# it does not accumulate across cycles.
function accumulated_carrier_discriminator(state::SatVectorPLLAndDLL)
    mean = _mean_carrier_discriminator(state)
    isnothing(mean) ? 0.0 : ustrip(Hz, mean)
end

# Whether a member accumulated any discriminator at all this cycle. The means are
# `nothing` while their accumulator count is zero, which is what a cycle without a single
# fully integrated correlator dump for this satellite looks like (a member admitted at the
# very end of a cycle, or one whose samples were starved). The `accumulated_*` helpers
# substitute a zero there, and a zero discriminator is not a missing measurement to the
# navigation filter — it is a *confidently zero* residual carrying the full measurement
# weight of `R`, which would pull the state towards the current NCO instead of leaving it
# to coast. Members without an accumulation are therefore withheld from the measurement
# set; they stay in the vector loop and keep getting NCO corrections, exactly like a member
# out of code lock.
#
# Both accumulators are incremented in the same branch of `step_loop`, so the two counts
# never disagree; the carrier one is checked as well so this stays true if that changes.
has_accumulated_discriminators(state::SatVectorPLLAndDLL) =
    !isnothing(_mean_code_discriminator(state)) &&
    !isnothing(_mean_carrier_discriminator(state))

# Linear carrier-to-noise density (Hz) floored to 1, from a CN0 estimate in
# dB-Hz. The floor keeps a starved estimator from producing a degenerate
# measurement weight; the `isnan` guard keeps a NaN estimate (an empty
# correlator returns NaN dB-Hz) from putting a NaN into the measurement-noise
# covariance and corrupting the whole Kalman update.
function linear_cn0_floor(cn0_dbhz)
    cn0_linear = 10^(cn0_dbhz / 10)
    isnan(cn0_linear) ? 1.0 : max(cn0_linear, 1.0)
end

# Pseudorange (m) from the receive and transmit times-of-week. Both are
# seconds-of-week that wrap at 604800 s, so the small receive − transmit
# light-travel difference is folded modulo the week (`fold_week_crossover`
# maps a near-±week difference back to near zero) to stay correct when the two
# straddle a GNSS week rollover.
pseudorange_from_tows(receive_tow, transmit_tow) =
    fold_week_crossover(receive_tow - transmit_tow) * SPEED_OF_LIGHT

# ─────────────────────────────────────────────────────────────────────────────
# Measurement prediction

# Per-member clock/IFB column assignment in PositionVelocityTime's `BiasColumns`
# form, for the members at `indices`, written into the vectors of `columns` (which
# must have been built with room for them), so its `calc_ρ_hat!` / `calc_H!` do the
# pseudorange modelling (including the earth-rotation correction).
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

# The same assignment restricted to the bias states these members actually
# occupy, renumbered densely, into `columns`; returns `primary_clock_index`
# translated into the dense numbering (1 when that clock is absent). The
# navigation filter carries a fixed set of bias states, but a design matrix
# built for DOP must only carry the columns the measurement set can determine:
# an all-zero column for a constellation with no satellites this epoch makes
# `HᵀH` singular, and `calc_DOP!` then reports −1 for every epoch a
# constellation happens to be missing from. `clock_used` / `ifb_used` are
# scratch vectors of the layout's clock and IFB counts.
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
    # Dense column per state, in state order (the order `sort!(unique(…))` gives).
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

# Predicted pseudorange rate (m/s): line-of-sight closing speed plus satellite
# clock drift minus the (single, common) receiver clock drift — the same sign
# convention as the measured `λ · carrier_doppler`.
@inline function predict_pseudorange_rate(
    user_pos,
    user_vel,
    user_clock_drift,
    sat_pos,
    sat_vel,
    sat_clock_drift,
)
    # `calc_line_of_sight` points receiver→satellite, so the closing speed —
    # positive while the range shrinks, the Doppler sign — is its dot product
    # with the *user's* velocity relative to the satellite.
    e = calc_line_of_sight(sat_pos, user_pos)
    dot(e, user_vel - sat_vel) + sat_clock_drift * SPEED_OF_LIGHT - user_clock_drift
end

# The measurement model `h!(y, x)` of one update, over the candidates' buffers: their
# pseudoranges (`calc_ρ_hat!`), then their pseudorange rates when fused, then one row per
# hub constraint, the broadcast offset between two clock states. A callable struct rather
# than a closure, so the update compiles to one concrete method and allocates nothing.
struct VTMeasurementModel
    idxs::NavFilterIndices
    ξ::Vector{Float64}
    positions::Vector{SVector{3,Float64}}
    velocities::Vector{SVector{3,Float64}}
    clock_drifts::Vector{Float64}
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
        pos, vel, clock_drift = nav_filter_states(x, idxs)
        for j = 1:num_sats
            y[num_sats+j] = predict_pseudorange_rate(
                pos,
                vel,
                clock_drift,
                model.positions[j],
                model.velocities[j],
                model.clock_drifts[j],
            )
        end
        offset += num_sats
    end
    for (i, (state, hub_state, _)) in enumerate(model.constraints)
        y[offset+i] = x[idxs.clock_biases[state]] - x[idxs.clock_biases[hub_state]]
    end
    y
end

# The rate rows are the derived single-cycle variance, deliberately carrying no inflation
# factor. Consecutive cycles are not independent: the FLL's `previous_prompt` chains
# across cycles while the accumulators are reset every cycle, so cycle `i` measures
# `(θ_N − θ_0)/(2π·T)` and cycle `i+1` measures `(θ_2N − θ_N)/(2π·T)`. They share the
# boundary phase estimate with opposite signs, giving
#     cov = −σ_φ²/(2π·T)²,   var = 2·σ_φ²/(2π·T)²   ⇒   ρ(lag 1) = −1/2,
# which a Kalman update cannot represent. Two consequences, neither of them a reason to inflate
# `R`:
#
#  - The correlation is *negative*, so noise averages out faster across cycles than a
#    white-noise filter credits. The filter is therefore already pessimistic about the rate
#    channel, not overconfident — inflating `R` moves further in the direction it already errs.
#    What the correlation does cost is a pessimistic reported velocity / clock-drift
#    uncertainty, which only measurement differencing (Bryson-Henrikson) or carrying the
#    boundary phase as a state would fix.
#  - The rate residuals carry a ≈ −0.5 lag-1 autocorrelation *by construction*. That is the
#    telescoping, not a tracking fault, and it must not be tuned against.
#
# Note also that the derived variance is not conservative by accident: treating all `N`
# per-dump discriminators as independent would give `N` times this value, and it is exactly the
# −1/2 adjacency correlation making the interior phases telescope away that earns the tighter
# figure. It is the right variance for the estimator the mean FLL discriminator actually is.

# Measurement-noise variances: CN0-driven DLL thermal-noise variance for the pseudoranges,
# and the pseudorange-rate variance for the ATAN frequency-lock discriminator
# (`atan(cross/dot)/(2π·T_coh)`; see `fll_disc`).
#
# Both are built the same way: the per-dump discriminator variance for a coherent
# integration time `T_coh` (`member.coherent_integration_time`), propagated through the
# averaging of the `N = T/T_coh` dumps that the filter's measurement is a mean of.
#
# Code. `dll_disc` is the *noncoherent* envelope-normalized early-minus-late
# discriminator, whose per-dump jitter carries a squaring loss set by the coherent
# integration time (Kaplan & Hegarty, "Understanding GPS: Principles and Applications",
# 2nd ed., Artech House 2006, §5.5.2, noncoherent early-late DLL tracking jitter; derived in
# general form by Betz & Kolodziejski, "Generalized Theory of Code Tracking with an
# Early-Late Discriminator, Part II: Noncoherent Processing and Numerical Results", IEEE
# Trans. Aerospace and Electronic Systems 45(4), 2009, pp. 1557-1564):
#     σ_τ,dump² = d/(4·C/N0·T_coh) · (1 + 2/((2 − d)·C/N0·T_coh))   [chips²],
# the first factor being the coherent early-late variance (the reference formulas carry a
# loop noise bandwidth `B_n`; a single dump is the open-loop case `B_n = 1/(2·T_coh)`) and
# the bracket the squaring loss from multiplying two noisy envelopes. The filter's code
# measurement is the mean of the `N` per-dump discriminators, whose noise is white across
# dumps (successive dumps share no samples), so
#     var(mean) = σ_τ,dump²/N = d/(4·C/N0·T) · (1 + 2/((2 − d)·C/N0·T_coh))   [chips²],
# i.e. the thermal term averages down over the whole filter interval `T` while the squaring
# loss stays pinned to `T_coh` — a short coherent dump inflates the code variance no matter
# how long the filter interval is. The pseudorange variance is `chip_length²` times that.
# For the BOC VEML correlator `d` is the inner early-late pair's spacing, so the model is
# the EPL approximation of it: the extra very-early/very-late taps average a little more
# noise away, making the model mildly conservative there.
#
# The noise model assumes the early and late taps still sit inside the correlation
# triangle, i.e. `d < 2` chips — which every real correlator configuration is well under
# (the default is 0.5).
function pseudorange_noise_variance(member::VTMember, T)
    cn0_tcoh = member.cn0 * member.coherent_integration_time
    d = member.early_late_spacing
    squaring_loss = 1 + 2 / ((2 - d) * cn0_tcoh)
    d / (4 * T * member.cn0) * squaring_loss * member.chip_length^2
end

# Rate. A coherent dump of length `T_coh` estimates carrier phase with the ATAN
# discriminator jitter
#     σ_φ² = 1/(2·C/N0·T_coh)·(1 + 1/(2·C/N0·T_coh))   [rad²].
# The filter's rate measurement is the mean of the `N` per-dump discriminators. Each dump
# is a frequency — a phase difference over one `T_coh`, `(θ_k − θ_{k-1})/(2π·T_coh)` — so
# the mean reduces to `(θ_N − θ_0)/(2π·T)`: the interior phases cancel, leaving only the
# two endpoint phase estimates, giving
#     var(mean) = 2·σ_φ² / (2π·T)²   [Hz²],
# and the pseudorange-rate variance is λ² times that.
function pseudorange_rate_noise_variance(member::VTMember, T)
    cn0_tcoh = member.cn0 * member.coherent_integration_time
    sigma_phi2 = 1 / (2 * cn0_tcoh) * (1 + 1 / (2 * cn0_tcoh))
    member.wavelength^2 * sigma_phi2 / (2 * π^2 * T^2)
end

# ─────────────────────────────────────────────────────────────────────────────
# Bias observability

# Accuracy (m) credited to a broadcast inter-system clock offset — the Galileo GGTO or
# a BGTO variant — when it collapses a constellation's clock state onto the hub's. The
# Galileo OS SDD (Issue 1.2, Nov 2021) commits the broadcast GGTO to below 20 ns at the
# 95th percentile — a σ near 10 ns, so ~3 m of pseudorange; BeiDou publishes no
# comparable commitment for the BGTO, which is credited the same. Good enough to remove
# a clock unknown under satellite starvation, but far too coarse to constrain a solution
# whose geometry can observe the offset directly, which is why the collapse is only
# applied when the independent layout is not supported.
const HUB_OFFSET_STD = 3.0

# The bias-layout decision for one measurement set, mirroring
# `decide_bias_layout` for the navigation filter.
#
# `num_unknowns` is the number of parameters this cycle's pseudoranges have to
# determine — 3 position components plus the clock biases and the *observable*
# inter-frequency biases of the layout in force, the merged one where the clock
# collapse applies (see `epoch_bias_unknowns!`). It is the measurement count the
# epoch must reach to be solvable, and so accounts for the bias unknowns a
# multi-GNSS, multi-band configuration adds beyond the bare four.
#
# `num_distinct_sats_required` is the second, independent condition
# `decide_bias_layout` makes, and reaching `num_unknowns` measurements does not
# imply it: the measurement count may be made up of extra *bands* of satellites
# already counted, and a second band of an already-tracked satellite carries no
# new line of sight — it constrains the inter-frequency biases, not the position
# and clock unknowns. Those need `3 + clock biases` measurements from distinct
# satellites (`num_distinct_sats`, identified as `decide_bias_layout` identifies
# them). Three satellites on two bands is the case the plain count misses: six
# measurements against five unknowns passes, while the design's rows take only
# four distinct values outside the inter-frequency-bias columns.
#
# Both conditions are necessary and neither is sufficient — the surviving
# geometry can still be degenerate in a way no satellite count can see. The
# scalar solve leaves that to `calc_pvt`'s checks on the assembled design; here
# it is reported as `VTStatus`'s `position_std` rather than policed.
#
# The hub offset constraints live in the workspace (`hub_offset_constraints`),
# one `(state, hub_state, isb)` per collapsed time system: the collapse, expressed
# as the linear pseudo-measurement `x[state] - x[hub_state] = isb` rather than as a
# merged design-matrix column, so the filter keeps a fixed state dimension and that
# constellation's clock stays available (and keeps tracking the shared oscillator
# drift) for the epochs where the geometry does observe it. Several because a mixed
# epoch can collapse several systems onto the hub at once — Galileo through its GGTO
# and BeiDou through its BGTO, independently — which is what `decide_bias_layout`
# does on the scalar side. Empty when nothing is collapsed.
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

# Whether a measurement set can determine the navigation state under the layout it was
# assessed for: both of `decide_bias_layout`'s conditions, the second of which the
# assessment already holds the operands of.
is_epoch_solvable(obs::BiasObservability, num_measurements) =
    num_measurements >= obs.num_unknowns &&
    obs.num_distinct_sats >= obs.num_distinct_sats_required

# The number of distinct entries of `values`, without a set.
function _num_distinct(values)
    count = 0
    for j in eachindex(values)
        seen = false
        for i in firstindex(values):(j-1)
            if values[i] === values[j]
                seen = true
                break
            end
        end
        seen || (count += 1)
    end
    count
end

# The three quantities `decide_bias_layout` decides the scalar layout from, evaluated for
# one measurement set: how many parameters its pseudoranges have to determine, how many of
# those need a distinct satellite each (the position and clock unknowns — see
# `BiasObservability`), and how many connected components its (constellation × band)
# coverage graph has.
#
# Both are read off the *epoch's own* coverage graph rather than off the configured layout,
# exactly as `decide_bias_layout` reads them off the epoch's satellites. `band_ifb_layout!`
# creates an inter-frequency-bias column only where the bias is observable, so a band whose
# component reference carries no measurement this cycle folds its delay into the clock and
# is not an unknown of this epoch — counting the configured layout's columns instead would
# demand a measurement for a parameter this epoch cannot and need not determine.
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

# Decide the bias layout for this cycle's measurement set, following
# `decide_bias_layout`: estimate every bias independently when the
# (constellation × band) coverage graph is connected and there are enough
# measurements, and otherwise fall back to the broadcast clock collapse — which
# both removes a clock unknown (the scarce-satellite case) and reconnects a
# disjoint band split (the disconnected case, where a band's inter-frequency
# bias is collinear with the stranded constellation's clock). The constraints of
# a collapse are left in `ws.hub_offset_constraints`.
function assess_bias_observability!(
    ws::ObservabilityWorkspace,
    layout::NavFilterLayout,
    members,
    candidate_indices,
)
    num_measurements = length(candidate_indices)
    constraints = empty!(ws.hub_offset_constraints)
    # Only the constellations and bands some measurement touches are unknowns of this
    # epoch; a constellation or band without measurements simply coasts.
    time_systems = resize!(ws.time_systems, num_measurements)
    bands = resize!(ws.bands, num_measurements)
    for (k, j) in enumerate(candidate_indices)
        time_systems[k] = layout.time_systems[members[j].clock_bias_index]
        bands[k] = layout.band_by_group[members[j].group]
    end
    # Distinct physical satellites, identified by `(time system, PRN)` exactly as
    # `decide_bias_layout` identifies them — a PRN is only unique within its GNSS, and a
    # satellite tracked on several bands is one line of sight however many measurements it
    # contributes. The clock-bias state and the time system are in bijection here, so
    # keying on either identifies the same satellites.
    num_distinct_sats = 0
    for (k, j) in enumerate(candidate_indices)
        seen = false
        for i in 1:(k-1)
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

    # Connected-but-scarce or disconnected: collapse every clock this cycle has a
    # broadcast offset toward a hub system for onto that hub — the same hubs, in the
    # same fixed order, as `decide_bias_layout` (GPST, then GST, then BDT), so a
    # GPS-bearing cycle behaves exactly as it always did. Worth doing only for a
    # system whose clock and the hub's are both in play this cycle; otherwise it would
    # replace a well-observed clock state with the coarser broadcast value for
    # nothing. One hub per cycle, as on the scalar side: the first hub that yields any
    # constraint wins.
    for hub_offset_index in eachindex(CANDIDATE_HUB_SYSTEMS)
        hub = _candidate_hub(hub_offset_index)
        hub_state = _time_system_index(layout.time_systems, hub)
        (hub_state == 0 || _time_system_index(time_systems, hub) == 0) && continue
        collapsed = empty!(ws.collapsed)
        for state in eachindex(layout.time_systems)
            time_system = layout.time_systems[state]
            (time_system === hub || _time_system_index(time_systems, time_system) == 0) &&
                continue
            # The offset is one constellation-wide value whichever of the system's
            # satellites reports it, so the first decoded copy per system converts all of
            # that system's measurements — the rule `calc_hub_range_offsets` follows.
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
            # The broadcast offset is Δt_systems = (that system's time) − (the hub's),
            # and the clock states are in metres of pseudorange, so that system's clock
            # sits −c·Δt_systems from the hub's — the same sign convention
            # `decide_bias_layout` gives its `inter_system_biases`.
            isb =
                -SPEED_OF_LIGHT *
                calc_steering_offset(member.time_offsets[hub_offset_index], member.time)
            push!(constraints, (state, hub_state, isb))
            push!(collapsed, time_system)
        end
        isempty(constraints) && continue
        # The unknowns have to be recounted on the merged graph rather than simply
        # decremented: dropping a clock removes one unknown, but merging two
        # constellations can also reconnect two coverage components — which is the
        # point of the collapse in the disconnected case — and every band that stops
        # being a component reference then becomes an observable, and countable,
        # inter-frequency bias again. `decide_bias_layout` recounts for the same reason.
        merged = resize!(ws.merged_time_systems, num_measurements)
        for k in eachindex(time_systems)
            merged[k] =
                _time_system_index(collapsed, time_systems[k]) != 0 ? hub : time_systems[k]
        end
        merged_unknowns, merged_distinct_required, _ = epoch_bias_unknowns!(ws, merged, bands)
        # Reported even when the merged layout is still short of its conditions, where
        # `decide_bias_layout` would fall back to the independent one and call the epoch
        # unsolvable: the merge can only lower both requirements (it drops a clock unknown
        # per collapsed system and can add back at most one inter-frequency bias each), so
        # the two agree on solvability, and applying the constraints on a starved epoch is
        # free information rather than a decision.
        return BiasObservability(merged_unknowns, merged_distinct_required, num_distinct_sats)
    end

    # No collapse available: the layout stays independent, and the epoch is solvable only
    # if the measurements and the distinct satellites among them suffice on their own.
    independent
end

# `CANDIDATE_HUB_SYSTEMS[i]` as the `SupportedTimeSystem` it is, spelled out so the
# element type is the closed union rather than whatever indexing a heterogeneous
# tuple infers.
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

# The NCO corrections for one member: the prediction residual — the difference between
# the pseudorange (rate) the filter predicts for the satellite and the one its replica
# realises — converted into the code / carrier frequency offsets that remove it over the
# next interval `T`. The carrier corrections feed the FLL branch of each satellite's
# carrier loop.
#
# The predictions must be evaluated at the *updated* state, which alone makes them the
# complete correction. Adding the measurement update's state correction projected onto
# the line of sight on top would count it twice and command double the required slew — a
# loop gain of 2 that leaves the replica oscillating about the solution instead of
# settling.
nco_code_correction(predicted_pseudorange, measured_pseudorange, code_frequency, T) =
    -(predicted_pseudorange - measured_pseudorange) * code_frequency / (T * SPEED_OF_LIGHT)

nco_carrier_correction(predicted_pseudorange_rate, measured_pseudorange_rate, wavelength) =
    (predicted_pseudorange_rate - measured_pseudorange_rate) / wavelength

# Post-fit residuals of one member: evaluated from the predictions at the *updated* state,
# so these are post-fit residuals and not the pre-fit innovations. `(pseudorange residual
# in m, range-rate residual in m/s)`, both measured − modelled ("observed minus
# computed") — the orientation the scalar `calc_pvt` reports its own two residuals in,
# and RTKLIB before it, so a vector-tracking solution and a scalar one are directly
# comparable, sign included.
#
# Mind that the two are therefore written with opposite subtraction order here. The
# pseudorange residual is `z - h(x)`, straightforwardly. The rate residual is
# `h(x) - z` because the rate *observable* differs: this loop measures
# `+λ · carrier_doppler`, positive while the satellite closes, whereas `calc_pvt` (and
# RTKLIB's `resdop`) residuate the geometric range rate, positive while it recedes.
# Observed − computed of that quantity is `h(x) - z` of this one — verified against
# `calc_pvt`'s own numbers, not just its wording. Making both subtractions read alike
# would silently invert the reported rate residual against every scalar fix.
#
# Because the vector loop steers every replica onto the navigation solution, a
# well-tracked member's residuals reduce to its own discriminators — the code residual
# to `+code_discriminator · chip_length`, the rate residual to
# `-carrier_discriminator · wavelength` — while a member the solution predicts poorly
# (one diverging, or one out of lock and coasting) keeps a large residual. That is
# what makes them worth reporting for members outside the update too: such a member
# is monitored rather than dropped as a missing satellite.
#
# The rate residual is a least-squares post-fit residual proper only under VDFLL,
# where the rates entered the update. Under VDLL (`use_pseudorange_rates = false`)
# the rates are measured but never fused, so it tests the velocity/clock-drift
# solution against a measurement it never saw — the more searching check of the two.
# Either way it flags a satellite whose Doppler disagrees with the solution
# independently of its pseudorange.
function vt_post_fit_residuals(
    member::VTMember,
    measured_pseudorange,
    predicted_pseudorange,
    predicted_pseudorange_rate,
)
    (measured_pseudorange + member.code_discriminator * member.chip_length -
     predicted_pseudorange) * m,
    (predicted_pseudorange_rate - member.pseudorange_rate -
     member.carrier_discriminator * member.wavelength) * (m / s)
end

# ─────────────────────────────────────────────────────────────────────────────
# PVT solution from the navigation filter

# The clock-bias state to report the solution against. Kept while its time
# system still contributes an included measurement, so the reference does not
# flicker; re-picked on loss following `PositionVelocityTime`'s reference-system
# convention — GPST when GPS is present, otherwise the time system with the most
# measurements (ties broken by clock-state order). Re-picking only from time
# systems that are present this cycle is what keeps the solution able to read a
# week number from a decoded satellite of the primary system, so it always
# carries a timestamp.
function report_primary_clock_index(layout::NavFilterLayout, members, included_indices, current_index)
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

# The epoch offset to carry into the next cycle: the cached one (or a freshly resolved one),
# advanced by a week when `reference_time` has just wrapped at the 604800 s boundary. Without
# that advance a mid-run Saturday→Sunday rollover would leave every reported `pvt.time`
# exactly one week in the past for the rest of the run — the wrap takes a week off the time
# of week and the offset, being a constant of the run everywhere else, never puts it back.
# Only an offset that was *already* cached is advanced: one resolved on this very cycle was
# read from a decoder that has rolled its own week number over too, so it is already current.
rolled_over_time_epoch_offset(resolved, cached, week_rollover) =
    week_rollover && !isnothing(cached) ? cached + SECONDS_PER_WEEK : resolved

# The constant part of the solution's epoch, read off a measurement row of the primary
# system: its week count in seconds plus that system's start epoch. `reference_time` is a GPS-Time-count time of
# week, so the epoch that anchors it must absorb the system's scale offset: a BDT
# week·604800 + start epoch pairs with BDT seconds-of-week, which read 14 s below the GPST
# count.
#
# It is a constant of the run, so it is resolved once and cached — after which a solution
# can no longer lose its timestamp, and a timestamp is how every consumer tells a fix from
# a non-fix. Caching it survives a change of primary clock: `reference_time` runs on the
# GPS Time count regardless of which clock reports (`VTMember.time_gpst_count`), and the
# scale offset folded in here puts every system's `week·604800 + start epoch` onto that
# same count — BDT's, for instance, lands 14 s below GPST's, exactly compensating the 14 s
# its seconds-of-week read low. What remains between two systems' cached offsets is their
# broadcast steering — nanoseconds, against a quantity used to stamp a 100 ms cycle.
#
# A row is only built for a satellite that has finished decoding for positioning, and it
# carries the week with the GPS L1 C/A rollover resolved, so any member of the primary
# system dates the run.
time_epoch_offset(row::SatelliteMeasurement) =
    row.week * SECONDS_PER_WEEK +
    row.system_start_epoch.second +
    round(Int, row.count_offset_to_gpst)

# Absolute epoch of the vector-tracking solution: the clock-bias-corrected reference time
# (seconds of week on the GPS Time count), anchored by the cached epoch offset, or
# `nothing` while there is none.
function vt_time(time_epoch_offset, reference_time, primary_clock_bias)
    isnothing(time_epoch_offset) && return nothing
    corrected_reference_time = ustrip(s, reference_time) - primary_clock_bias / SPEED_OF_LIGHT
    TAITime(
        time_epoch_offset + floor(Int, corrected_reference_time),
        corrected_reference_time - floor(corrected_reference_time),
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Initialization

# Initial navigation state and covariance from a scalar PVT fix, written into `x` and
# `P`; returns the primary clock index. The primary system's clock bias starts at zero
# (the pseudorange reference epoch is already corrected by the fix's clock bias); the
# other systems' biases and the inter-frequency biases are seeded from the fix where it
# observed them, and start at zero with a generous variance where it did not — their
# first measurements then pull them in through the Kalman update. The fixed variances
# of the position and the seeded biases are a fallback: `seed_fix_covariance!` replaces
# them with the fix's own covariance wherever its geometry is known.
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
            x[state_index] = ustrip(m, pvt.inter_frequency_biases[layout.extra_bands[index]].value)
            P[state_index, state_index] = init_std_seeded_ifb^2
        else
            P[state_index, state_index] = init_std_unseeded_ifb^2
        end
    end
    primary_clock_index
end

# Whether the fix estimated the clock-bias state `index` (the primary one, or another
# system's through its inter-system bias), so that `initial_nav_state!` seeds it.
_fix_seeds_clock(pvt::PVTSolution, layout::NavFilterLayout, primary_clock_index, index) =
    index == primary_clock_index || haskey(pvt.inter_system_biases, layout.time_systems[index])

# Whether the fix seeds the inter-frequency-bias state `index`: only when it measured the
# bias against the same reference band as the filter's layout — otherwise it refers to a
# different quantity.
function _fix_seeds_ifb(pvt::PVTSolution, layout::NavFilterLayout, index)
    band = layout.extra_bands[index]
    haskey(pvt.inter_frequency_biases, band) &&
        pvt.inter_frequency_biases[band].reference == layout.reference_bands[index]
end

# The pseudorange error (m) a scalar fix is taken to carry — the user equivalent range
# error its covariance is scaled by in `seed_fix_covariance!`.
const FIX_PSEUDORANGE_STD = 1.0

# Overwrite the position and bias block of `P`, as `initial_nav_state!` seeded it, with
# the covariance of the least-squares fix it was seeded from: `σ² (HᵀH)⁻¹` for the
# design matrix `H` of the fix's satellites (`calc_H!`, over the dense bias columns of
# `dense_bias_columns!`, whose state-to-column maps `clock_used` and `ifb_used` are), and
# `σ = FIX_PSEUDORANGE_STD`. A fixed variance, whatever the geometry, takes a fix of a
# GDOP of 40 as accurate as one of 2: its error, many times the variance, is then held by
# the measurements that agree with it, while the covariance shrinks (JuliaGNSS/
# TrackingLoops.jl#16). The geometry also correlates the position with the clocks — an
# inter-system bias determined by one satellite is as wrong as the position along its
# line of sight — which the full block carries. A bias state the fix did not seed keeps
# its generous variance, uncorrelated, though the position's variance still counts it as
# an unknown. `normal_matrix` is a square scratch matrix of `H`'s column count. Returns
# whether the block was written: a rank-deficient `H` leaves `P` as seeded.
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
    issuccess(factorization) || return false
    covariance = LinearAlgebra.inv!(factorization)
    σ² = FIX_PSEUDORANGE_STD^2
    num_columns = size(covariance, 1)
    for column = 1:num_columns, row = 1:num_columns
        i = _fix_state_index(idxs, layout, pvt, primary_clock_index, clock_used, ifb_used, row)
        j = _fix_state_index(idxs, layout, pvt, primary_clock_index, clock_used, ifb_used, column)
        i == 0 || j == 0 || (P[i, j] = σ² * covariance[row, column])
    end
    true
end

# The state behind column `column` of the fix's dense design matrix, `0` for a bias
# state the fix did not seed.
function _fix_state_index(idxs, layout, pvt, primary_clock_index, clock_used, ifb_used, column)
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
