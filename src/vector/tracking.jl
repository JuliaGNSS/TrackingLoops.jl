# ─────────────────────────────────────────────────────────────────────────────
# Vector tracking: the navigation cycle that closes all loops
#
# Once per navigation cycle the caller hands `update_navigation!` one
# `VTSignalGroup` per ranging signal, each a preallocated vector of `VTSat`s it
# has filled from its own channels: the replica at the cycle epoch, the replica
# predicted where the command computed now will land, the C/N₀, the lock flags
# and each satellite's `SatVectorPLLAndDLL`. Before vector tracking runs, the
# cycle is a scalar PVT solve; its first fix seeds the navigation filter, and
# from then on each cycle is one filter iteration that also writes every
# member's NCO corrections back into its estimator state. Nothing here knows a
# correlator: the software receiver and a hardware loop process fill the same
# `VTSat`s.
# ─────────────────────────────────────────────────────────────────────────────

"""
    VTReleaseReason

Why [`update_navigation!`](@ref) handed a satellite back to its scalar loop
this cycle:

  - `VT_NOT_RELEASED`: it was not;
  - `VT_INELIGIBLE`: no longer tracked (`active == false`), decoded for
    positioning, or healthy;
  - `VT_BELOW_HORIZON`: below the horizon at the updated position. It is not
    admitted again until it stands a degree above the horizon, so a satellite
    at the horizon is not admitted and released every cycle;
  - `VT_FALLBACK`: vector tracking stopped (starvation timeout, or no member
    left).

A receiver that forces released satellites out of lock does so for the first
two, and not for a fallback.
"""
@enum VTReleaseReason VT_NOT_RELEASED VT_INELIGIBLE VT_BELOW_HORIZON VT_FALLBACK

"""
    VTSat(decoder, estimator_state; prn = decoder.prn, kwargs...)

One satellite slot of a [`VTSignalGroup`](@ref), filled by the caller every
navigation cycle and partly written back by [`update_navigation!`](@ref).
Every field is a keyword of the constructor.

Filled by the caller:

  - `prn`, `active` (the slot holds a tracked satellite), `decoder` and
    `estimator_state` (a [`SatVectorPLLAndDLL`](@ref));
  - at the cycle epoch, the replica *actually running* at that sample (under an
    NCO delay, read from the [`NCOTimeline`](@ref), not the last command):
    `code_phase` (chips), `carrier_phase`, `carrier_doppler`, `code_doppler`;
  - at the landing of the command computed now: `landing_lead` (how long after
    the epoch it lands, at most `2.5` navigation cycles), `code_phase_at_landing`,
    `carrier_doppler_at_landing` and `code_doppler_at_landing` (the replica
    predicted there under the words already committed). The code phase at
    landing counts on from `code_phase`: it is `code_phase` plus the chips the
    replica advances until the landing, not reduced to a code period, so it is
    read against the same decoded bit. Without an NCO delay these are `0.0s`,
    `code_phase`, `carrier_doppler` and `code_doppler`;
  - `cn0_dbhz`, `coherent_integration_time` (the last dump's) and
    `early_late_spacing` (chips, measured from the correlator);
  - the flags `in_lock` and `pvt_ready` (ready to enter the scalar PVT solve).

Written back: `estimator_state` (membership, corrections, emptied
accumulators) and `release_reason` (a [`VTReleaseReason`](@ref)). A released
satellite's inner loop is re-seeded from `carrier_doppler_at_landing` and
`code_doppler_at_landing`, the replica its scalar loop takes over from (see
[`release_from_vector_tracking`](@ref)).
"""
@kwdef mutable struct VTSat{D,E<:SatVectorPLLAndDLL}
    prn::Int
    active::Bool = false
    decoder::D
    estimator_state::E
    code_phase::Float64 = 0.0
    carrier_phase::Float64 = 0.0
    carrier_doppler::typeof(1.0Hz) = 0.0Hz
    code_doppler::typeof(1.0Hz) = 0.0Hz
    landing_lead::typeof(1.0s) = 0.0s
    code_phase_at_landing::Float64 = 0.0
    carrier_doppler_at_landing::typeof(1.0Hz) = 0.0Hz
    code_doppler_at_landing::typeof(1.0Hz) = 0.0Hz
    cn0_dbhz::Float64 = NaN
    coherent_integration_time::typeof(1.0s) = 0.001s
    early_late_spacing::Float64 = 0.5
    in_lock::Bool = false
    pvt_ready::Bool = false
    release_reason::VTReleaseReason = VT_NOT_RELEASED
end

VTSat(decoder, estimator_state::SatVectorPLLAndDLL; prn = decoder.prn, kwargs...) =
    VTSat(; prn, decoder, estimator_state, kwargs...)

"""
    VTSignalGroup(signal, sats::Vector{<:VTSat})

The satellites of one ranging signal: the signal, and a preallocated vector of
satellite slots whose length is the most the group can hold at once. The
`VTSignalGroup`s of a receiver are passed as a tuple, in a fixed order, to both
[`VectorTrackingState`](@ref) and [`update_navigation!`](@ref).
"""
struct VTSignalGroup{S<:AbstractGNSSSignal,D,E}
    signal::S
    sats::Vector{VTSat{D,E}}
end

"""
    VTStatus

What one [`update_navigation!`](@ref) call did.

  - `running`: vector tracking is running after this cycle;
  - `position_std`, `clock_std`: the filter's 1σ 3-D position and primary-clock
    uncertainties (`NaN` while not running). A degenerate geometry shows up
    here rather than in the starvation timer;
  - `time_with_insufficient_meas`: how long the filter has been coasting on
    epochs it could not solve, counted against `insufficient_meas_timeout`;
    nonzero means the loop is on its way out, paid back at half rate;
  - `num_members`: satellites in the vector loop after this cycle;
  - the events `enabled` (a fresh scalar fix seeded the filter), `fell_back`
    (vector tracking stopped) and `released` (some satellite was released; see
    each `VTSat`'s `release_reason`).

The per-member report of the latest update is the state's `member_sats`.
"""
struct VTStatus
    running::Bool
    position_std::typeof(1.0m)
    clock_std::typeof(1.0m)
    time_with_insufficient_meas::typeof(1.0s)
    num_members::Int
    enabled::Bool
    fell_back::Bool
    released::Bool
end

# The buffers of one measurement count: the unscented update's intermediate and
# the measurement vector and noise covariance it is handed.
struct MeasurementBuffers{I}
    intermediate::I
    z::Vector{Float64}
    R::Matrix{Float64}
end

const _UKFMU = typeof(UKFMUIntermediate(Float64, 1, 1))

MeasurementBuffers(num_states, num_measurements) = MeasurementBuffers{_UKFMU}(
    UKFMUIntermediate(Float64, num_states, num_measurements),
    zeros(num_measurements),
    zeros(num_measurements, num_measurements),
)

# Everything a cycle works in, sized at construction.
struct VTBuffers{SB<:Tuple}
    # Per group, the `SatelliteState`s whose rows are collected.
    states::SB
    rows::Vector{SatelliteMeasurement}
    members::Vector{VTMember}
    active::Vector{Bool}
    candidates::Vector{Int}
    delays::Vector{Float64}
    measured_pseudoranges::Vector{Float64}
    predicted_pseudoranges::Vector{Float64}
    predicted_pseudorange_rates::Vector{Float64}
    residuals::Vector{typeof(1.0m)}
    rate_residuals::Vector{typeof(1.0m / s)}
    positions::Vector{SVector{3,Float64}}
    member_columns::BiasColumns
    candidate_positions::Vector{SVector{3,Float64}}
    candidate_velocities::Vector{SVector{3,Float64}}
    candidate_clock_drifts::Vector{Float64}
    candidate_columns::BiasColumns
    ξ::Vector{Float64}
    ξ_model::Vector{Float64}
    ξ_landing::Vector{Float64}
    x_landing::Vector{Float64}
    landing_position::Vector{SVector{3,Float64}}
    landing_columns::BiasColumns
    landing_range::Vector{Float64}
    observability::ObservabilityWorkspace
    time_update::KFTUIntermediate{Float64}
    measurement_updates::Vector{Union{Nothing,MeasurementBuffers{_UKFMU}}}
    dense_clock_indices::Vector{Int}
    dense_ifb_indices::Vector{Int}
    clock_used::Vector{Int}
    ifb_used::Vector{Int}
    design_matrix::Matrix{Float64}
    normal_matrices::Vector{Matrix{Float64}}
end

_capacity_vector(::Type{T}, n) where {T} = sizehint!(T[], n)

function VTBuffers(states::Tuple, num_states, layout::NavFilterLayout, max_members)
    num_clocks = num_clock_biases(layout)
    num_ifbs = num_ifb(layout)
    num_lsq = 3 + num_clocks + num_ifbs
    max_measurements = 2 * max_members + num_clocks
    VTBuffers(
        states,
        _capacity_vector(SatelliteMeasurement, max_members),
        _capacity_vector(VTMember, max_members),
        _capacity_vector(Bool, max_members),
        _capacity_vector(Int, max_members),
        _capacity_vector(Float64, max_members),
        _capacity_vector(Float64, max_members),
        _capacity_vector(Float64, max_members),
        _capacity_vector(Float64, max_members),
        _capacity_vector(typeof(1.0m), max_members),
        _capacity_vector(typeof(1.0m / s), max_members),
        _capacity_vector(SVector{3,Float64}, max_members),
        BiasColumns(_capacity_vector(Int, max_members), num_clocks, _capacity_vector(Int, max_members), num_ifbs),
        _capacity_vector(SVector{3,Float64}, max_members),
        _capacity_vector(SVector{3,Float64}, max_members),
        _capacity_vector(Float64, max_members),
        BiasColumns(_capacity_vector(Int, max_members), num_clocks, _capacity_vector(Int, max_members), num_ifbs),
        zeros(num_lsq),
        zeros(num_lsq),
        zeros(num_lsq),
        zeros(num_states),
        [zero(SVector{3,Float64})],
        BiasColumns([1], num_clocks, [0], num_ifbs),
        [0.0],
        ObservabilityWorkspace(max_members),
        KFTUIntermediate(Float64, num_states),
        Union{Nothing,MeasurementBuffers{_UKFMU}}[nothing for _ = 1:max_measurements],
        _capacity_vector(Int, max_members),
        _capacity_vector(Int, max_members),
        zeros(Int, num_clocks),
        zeros(Int, num_ifbs),
        zeros(max(max_members, 1), num_lsq),
        [zeros(n, n) for n = 1:num_lsq],
    )
end

"""
    VectorTrackingState(config::Union{VectorTracking,Nothing}, groups;
                        approximate_year = year(now(UTC)),
                        enable_ionospheric_correction = true,
                        enable_tropospheric_correction = true,
                        integration_time = 100ms)

The vector-tracking loop of one receiver, built once for the tuple of
[`VTSignalGroup`](@ref)s it will be updated with: the bias layout of the
groups' signals, the navigation filter's process model, state and covariance,
and every buffer a cycle needs, sized to the groups' satellite slots.
`config = nothing` builds a state that only ever solves the scalar PVT, so a
receiver has one [`update_navigation!`](@ref) path either way.
`integration_time` is the nominal navigation cycle the process model starts
from; it follows the measured one.

It holds the latest solution (`pvt`, a `PVTSolution` whose containers every
cycle reuses) and the per-member report of the latest update (`member_sats`):
every member of the loop, measured *and* coasted, keyed exactly as `pvt.sats`
is, each with the satellite position, transmit time and post-fit residuals that
update produced. `pvt.sats` carries only the members the update measured, so
the difference between the two key sets is what the filter predicted through
an obscuration. Emptied while the scalar solve is in control.
"""
mutable struct VectorTrackingState{SB<:Tuple}
    const config::VectorTracking
    const enabled::Bool
    const layout::NavFilterLayout
    const model::NavFilterModel
    const x::Vector{Float64}
    const P::Matrix{Float64}
    running::Bool
    # The clock-bias state the solution is reported against.
    primary_clock_index::Int
    # The epoch the pseudoranges are referenced to — always on the GPS Time count (see
    # `VTMember.time_gpst_count`), so it does not move when the primary clock changes.
    reference_time::typeof(1.0s)
    time_with_insufficient_meas::typeof(1.0s)
    # Constant part of the reported epoch (seconds), cached once resolved — see
    # `time_epoch_offset`. `nothing` until a primary-system member has been measured.
    time_epoch_offset::Union{Nothing,Int}
    const member_sats::Dictionary{Tuple{Symbol,Int},SatInfo}
    pvt::PVTSolution
    const workspace::PVTWorkspace
    const approximate_year::Int
    const enable_ionospheric_correction::Bool
    const enable_tropospheric_correction::Bool
    const buffers::VTBuffers{SB}
end

function VectorTrackingState(
    config::Union{VectorTracking,Nothing},
    groups::Tuple{Vararg{VTSignalGroup}};
    approximate_year::Integer = year(now(UTC)),
    enable_ionospheric_correction::Bool = true,
    enable_tropospheric_correction::Bool = true,
    integration_time = 100.0ms,
)
    filter_config = something(config, VectorTracking())
    layout = NavFilterLayout(map(group -> group.signal, groups))
    model = NavFilterModel(filter_config, layout, uconvert(s, integration_time))
    n = num_nav_states(filter_config, layout)
    max_members = sum(group -> length(group.sats), groups; init = 0)
    states = map(_satellite_state_buffer, groups)
    VectorTrackingState(
        filter_config,
        !isnothing(config),
        layout,
        model,
        zeros(n),
        zeros(n, n),
        false,
        1,
        0.0s,
        0.0s,
        nothing,
        Dictionary{Tuple{Symbol,Int},SatInfo}(),
        PVTSolution(),
        PVTWorkspace(),
        Int(approximate_year),
        enable_ionospheric_correction,
        enable_tropospheric_correction,
        VTBuffers(states, n, layout, max_members),
    )
end

_satellite_state_buffer(group::VTSignalGroup{S,D}) where {S,D} =
    sizehint!(SatelliteState{Float64,D,S}[], max(length(group.sats), 1))

"""
    position_uncertainty(vt::VectorTrackingState)
    clock_uncertainty(vt::VectorTrackingState)

The filter's own 1σ uncertainties (m): the 3-D position and the clock bias the
solution is referenced to.
"""
position_uncertainty(vt::VectorTrackingState) = position_uncertainty(vt.P, vt.model.idxs)

function clock_uncertainty(vt::VectorTrackingState)
    index = vt.model.idxs.clock_biases[vt.primary_clock_index]
    sqrt(vt.P[index, index])
end

# ─────────────────────────────────────────────────────────────────────────────
# Walking the groups

# `f(accumulator, group, g, buffer, args...)` over the groups, in order, with the
# group's position `g` and its entry of the per-group `buffers`; returns the final
# accumulator. Recursion over the tuples keeps every group's call statically
# dispatched, however heterogeneous the groups are.
@inline _fold_groups(f::F, acc, ::Tuple{}, ::Tuple{}, g, args...) where {F} = acc
@inline _fold_groups(f::F, acc, groups::Tuple, buffers::Tuple, g, args...) where {F} =
    _fold_groups(
        f,
        f(acc, first(groups), g, first(buffers), args...),
        Base.tail(groups),
        Base.tail(buffers),
        g + 1,
        args...,
    )

# The per-group `SignalGroup`s over `buffers`, for the PVT collection pass.
_signal_groups(groups, buffers) =
    map((group, buffer) -> SignalGroup(group.signal, buffer), groups, buffers)

# ─────────────────────────────────────────────────────────────────────────────
# The scalar solve

function _collect_pvt_ready!(acc, group, g, buffer)
    empty!(buffer)
    for sat in group.sats
        sat.active && sat.pvt_ready || continue
        push!(buffer, _satellite_state(group.signal, sat))
    end
    acc
end

@inline _satellite_state(signal, sat::VTSat) = SatelliteState(;
    decoder = sat.decoder,
    system = signal,
    code_phase = sat.code_phase,
    carrier_doppler = sat.carrier_doppler,
    carrier_phase = sat.carrier_phase,
)

# The scalar PVT over the satellites marked `pvt_ready`, into the state's solution:
# a fresh fix is a new object, a failed epoch the solution itself.
function _solve_scalar_pvt!(vt::VectorTrackingState, groups)
    buffers = vt.buffers.states
    _fold_groups(_collect_pvt_ready!, nothing, groups, buffers, 1)
    vt.pvt = calc_pvt!(
        vt.pvt,
        vt.workspace,
        _signal_groups(groups, buffers),
        vt.pvt;
        approximate_year = vt.approximate_year,
        enable_ionospheric_correction = vt.enable_ionospheric_correction,
        enable_tropospheric_correction = vt.enable_tropospheric_correction,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Membership

function _reset_release_reasons!(acc, group, g, buffer)
    for sat in group.sats
        sat.release_reason = VT_NOT_RELEASED
    end
    acc
end

# Hand `sat` back to its scalar loop for `reason`. The words committed until the
# landing were the vector loop's, so the scalar loop takes over from the replica
# there; without an NCO delay that is the epoch's.
function _release!(sat::VTSat, reason::VTReleaseReason)
    sat.estimator_state = release_from_vector_tracking(
        sat.estimator_state,
        sat.carrier_doppler_at_landing,
        sat.code_doppler_at_landing,
    )
    sat.release_reason = reason
    nothing
end

_is_eligible(sat::VTSat) =
    sat.active &&
    is_decoding_completed_for_positioning(sat.decoder) &&
    is_sat_healthy(sat.decoder)

# The elevation a satellite has to reach before it is admitted, above the horizon at
# which a member is released: the hysteresis keeps a satellite near the horizon from
# being admitted and released, its scalar loop re-seeded, every cycle. One degree is a
# few minutes of a rising satellite.
const ADMISSION_ELEVATION = deg2rad(1.0)

# Whether a satellite stands at least `ADMISSION_ELEVATION` above the horizon of
# `enu_from_ecef`, at its transmit time.
function _is_above_admission_mask(signal, sat::VTSat, enu_from_ecef)
    orbit = calc_satellite_position_and_velocity(_satellite_state(signal, sat))
    get_sat_enu(enu_from_ecef, _ecef(get_sat_position(orbit))).ϕ >= ADMISSION_ELEVATION
end

# Admit the satellites that are eligible (tracked, decoded for positioning, healthy),
# in lock and above the admission mask at the filter's position (`enu_from_ecef`), and
# release the members that are no longer eligible. Returns whether anything was
# released.
#
# NOTE: admission is deliberately gated on `in_lock` and not on a stricter
# ranging-ready flag, even though admitting a satellite whose code phase is still tens
# of metres out into the navigation filter is a real question. One list serves two
# purposes at once — the admission and, through `in_lock`, the measurement eligibility
# of the members — so gating it would also withhold discriminators from *existing*
# members, which is wrong. The two uses need separating first.
#
# A member out of lock is not released: it stays in the loop, unmeasured, and keeps
# receiving corrections.
function _update_membership!(released, group, g, buffer, enu_from_ecef)
    for sat in group.sats
        eligible = _is_eligible(sat)
        state = sat.estimator_state
        if eligible && sat.in_lock
            if state.vt_on || _is_above_admission_mask(group.signal, sat, enu_from_ecef)
                sat.estimator_state = enable_vector_tracking(state)
            end
        elseif !eligible && state.vt_on
            _release!(sat, VT_INELIGIBLE)
            released = true
        end
    end
    released
end

# ─────────────────────────────────────────────────────────────────────────────
# Measurement gathering

# The members' satellite states at the epoch, in member order.
function _collect_member_states!(acc, group, g, buffer)
    empty!(buffer)
    for sat in group.sats
        sat.estimator_state.vt_on && push!(buffer, _satellite_state(group.signal, sat))
    end
    acc
end

# One member per satellite in the loop, from its row; returns the running member
# index.
function _collect_members!(j, group, g, buffer, vt, T)
    layout = vt.layout
    buffers = vt.buffers
    signal = group.signal
    code_frequency = ustrip(Hz, get_code_frequency(signal))
    chip_length = SPEED_OF_LIGHT / code_frequency
    wavelength = SPEED_OF_LIGHT / ustrip(Hz, get_center_frequency(signal))
    clock_bias_index = layout.clock_bias_index_by_group[g]
    ifb_index = layout.ifb_index_by_group[g]
    for (slot, sat) in enumerate(group.sats)
        state = sat.estimator_state
        state.vt_on || continue
        j += 1
        row = buffers.rows[j]
        push!(
            buffers.members,
            VTMember(
                g,
                slot,
                sat.prn,
                clock_bias_index,
                ifb_index,
                chip_length,
                wavelength,
                code_frequency,
                sat.in_lock && has_accumulated_discriminators(state),
                row.time,
                row.time - row.count_offset_to_gpst,
                row.position,
                row.velocity,
                row.clock_drift,
                wavelength * ustrip(Hz, sat.carrier_doppler),
                accumulated_code_discriminator(state, T),
                accumulated_carrier_discriminator(state),
                linear_cn0_floor(sat.cn0_dbhz),
                sat.early_late_spacing,
                ustrip(s, sat.coherent_integration_time),
                row.time_offsets,
            ),
        )
    end
    j
end

# Gather this cycle's members: the satellites in the loop, their rows at the epoch, and
# the per-member model pieces. Returns the ionospheric correction the rows select.
function _gather_members!(vt::VectorTrackingState, groups, T)
    buffers = vt.buffers
    _fold_groups(_collect_member_states!, nothing, groups, buffers.states, 1)
    ionospheric_correction = collect_measurement_rows!(
        buffers.rows,
        _signal_groups(groups, buffers.states);
        approximate_year = vt.approximate_year,
    )
    # Every member is decoded and healthy, which is exactly what the collection pass
    # keeps, so the rows are the members in order.
    num_rows = length(buffers.rows)
    num_members = mapreduce(length, +, buffers.states; init = 0)
    num_rows == num_members || throw(
        ArgumentError(
            "a vector-loop member was not decoded for positioning and healthy; " *
            "its measurement row is missing",
        ),
    )
    empty!(buffers.members)
    _fold_groups(_collect_members!, 0, groups, buffers.states, 1, vt, T)
    members = buffers.members
    resize!(buffers.positions, num_members)
    for j in eachindex(members)
        buffers.positions[j] = members[j].sat_position
    end
    vt_bias_columns!(buffers.member_columns, members, eachindex(members))
    ionospheric_correction
end

# The atmosphere-corrected pseudoranges of every member against `reference_tow`, with
# the delays predicted at the state `x`.
function _measure_pseudoranges!(vt::VectorTrackingState, ionospheric_correction, reference_tow)
    buffers = vt.buffers
    members = buffers.members
    rows = buffers.rows
    num_members = length(members)
    delays = resize!(buffers.delays, num_members)
    if num_members > 0 &&
       (vt.enable_ionospheric_correction || vt.enable_tropospheric_correction)
        # The Niell mapping's seasonal term takes the day of year, derived the way
        # `calc_pvt` derives it: from a decoded satellite's absolute week plus the time of
        # week. Any member dates the epoch — every member has finished decoding and the
        # time systems differ by at most their defined scale offset, 14 s for BDT, against
        # a one-year period — so the first one serves.
        doy = day_of_year(rows[1].system_start_time, rows[1].week, reference_tow)
        position_and_bias_vector!(buffers.ξ, vt.x, vt.model.idxs)
        predict_atmospheric_delays!(
            delays,
            buffers.ξ,
            rows,
            IonosphericModel(vt.enable_ionospheric_correction ? ionospheric_correction : nothing),
            reference_tow,
            doy,
            vt.enable_tropospheric_correction,
        )
    else
        fill!(delays, 0.0)
    end
    measured = resize!(buffers.measured_pseudoranges, num_members)
    for j in eachindex(members)
        measured[j] = pseudorange_from_tows(reference_tow, members[j].time_gpst_count) - delays[j]
    end
    measured
end

# ─────────────────────────────────────────────────────────────────────────────
# The measurement update

function _measurement_buffers!(buffers::VTBuffers, num_states, num_measurements)
    cache = buffers.measurement_updates
    if num_measurements > length(cache)
        resize!(cache, num_measurements)
        cache[num_measurements] = nothing
    end
    cached = cache[num_measurements]
    isnothing(cached) || return cached
    new_buffers = MeasurementBuffers(num_states, num_measurements)
    cache[num_measurements] = new_buffers
    new_buffers
end

# Fuse the candidates' measurements at the predicted state `vt.x`, leaving the posterior
# in `vt.x` and `vt.P`.
function _measurement_update!(vt::VectorTrackingState, T)
    buffers = vt.buffers
    members = buffers.members
    candidates = buffers.candidates
    measured = buffers.measured_pseudoranges
    constraints = buffers.observability.hub_offset_constraints
    use_rates = vt.config.use_pseudorange_rates
    num_sats = length(candidates)
    resize!(buffers.candidate_positions, num_sats)
    resize!(buffers.candidate_velocities, num_sats)
    resize!(buffers.candidate_clock_drifts, num_sats)
    for (k, j) in enumerate(candidates)
        buffers.candidate_positions[k] = members[j].sat_position
        buffers.candidate_velocities[k] = members[j].sat_velocity
        buffers.candidate_clock_drifts[k] = members[j].sat_clock_drift
    end
    vt_bias_columns!(buffers.candidate_columns, members, candidates)
    num_rate_rows = use_rates ? num_sats : 0
    num_measurements = num_sats + num_rate_rows + length(constraints)
    update = _measurement_buffers!(buffers, length(vt.x), num_measurements)
    z = update.z
    R = fill!(update.R, 0.0)
    for (k, j) in enumerate(candidates)
        member = members[j]
        z[k] = measured[j] + member.code_discriminator * member.chip_length
        R[k, k] = pseudorange_noise_variance(member, T)
        if use_rates
            z[num_sats+k] =
                member.pseudorange_rate + member.carrier_discriminator * member.wavelength
            R[num_sats+k, num_sats+k] = pseudorange_rate_noise_variance(member, T)
        end
    end
    # Each clock collapse rides along as one extra measurement row, appended after the
    # satellite rows.
    offset = num_sats + num_rate_rows
    for (i, (_, _, isb)) in enumerate(constraints)
        z[offset+i] = isb
        R[offset+i, offset+i] = HUB_OFFSET_STD^2
    end
    h! = VTMeasurementModel(
        vt.model.idxs,
        buffers.ξ_model,
        buffers.candidate_positions,
        buffers.candidate_velocities,
        buffers.candidate_clock_drifts,
        buffers.candidate_columns,
        use_rates,
        constraints,
    )
    measurement_update!(update.intermediate, vt.x, vt.P, z, h!, R)
    nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Loop closure

# The predictions of every member at the state `vt.x` — the updated one — and their
# post-fit residuals.
function _predict_members!(vt::VectorTrackingState)
    buffers = vt.buffers
    members = buffers.members
    num_members = length(members)
    idxs = vt.model.idxs
    position_and_bias_vector!(buffers.ξ, vt.x, idxs)
    predicted = resize!(buffers.predicted_pseudoranges, num_members)
    calc_ρ_hat!(predicted, buffers.positions, buffers.ξ, buffers.member_columns)
    rates = resize!(buffers.predicted_pseudorange_rates, num_members)
    residuals = resize!(buffers.residuals, num_members)
    rate_residuals = resize!(buffers.rate_residuals, num_members)
    user_pos, user_vel, user_clock_drift = nav_filter_states(vt.x, idxs)
    for (j, member) in enumerate(members)
        rates[j] = predict_pseudorange_rate(
            user_pos,
            user_vel,
            user_clock_drift,
            member.sat_position,
            member.sat_velocity,
            member.sat_clock_drift,
        )
        residuals[j], rate_residuals[j] = vt_post_fit_residuals(
            member,
            buffers.measured_pseudoranges[j],
            predicted[j],
            rates[j],
        )
    end
    nothing
end

# The NCO corrections of every active member, evaluated where each lands: `τ =
# landing_lead` after the epoch, at the state `vt.x` propagated by `τ` and against the
# range the replica will realise there — the transmit time from the code phase at
# landing (the epoch's corrected transmit time moved on by the chips the replica
# advances; the satellite clock correction changes by picoseconds in `τ`), the satellite
# at that time, the receive time `reference_tow + τ`, the epoch's atmospheric delay (it
# moves by millimetres in `τ`). The code correction removes the range error (the TOW-based,
# atmosphere-corrected range, no discriminator term), the carrier correction the rate
# error against the replica's own Doppler at landing (no FLL term). With no NCO delay
# every quantity is the epoch's and these are the corrections at the updated state.
function _close_loops!(acc, group, g, buffer, vt, reference_tow, T)
    buffers = vt.buffers
    members = buffers.members
    idxs = vt.model.idxs
    for (j, member) in enumerate(members)
        member.group == g && buffers.active[j] || continue
        sat = group.sats[member.slot]
        τ = ustrip(s, sat.landing_lead)
        landing_time =
            member.time + (sat.code_phase_at_landing - sat.code_phase) / member.code_frequency
        landing_orbit = calc_satellite_position_and_velocity(sat.decoder, landing_time)
        landing_position = get_sat_position(landing_orbit)
        landing_velocity = get_sat_velocity(landing_orbit)
        propagate_state!(buffers.x_landing, vt.x, idxs, τ)
        position_and_bias_vector!(buffers.ξ_landing, buffers.x_landing, idxs)
        buffers.landing_position[1] = landing_position
        buffers.landing_columns.clock_bias_indices[1] = member.clock_bias_index
        buffers.landing_columns.ifb_indices[1] = member.ifb_index
        calc_ρ_hat!(
            buffers.landing_range,
            buffers.landing_position,
            buffers.ξ_landing,
            buffers.landing_columns,
        )
        predicted_pseudorange = buffers.landing_range[1]
        user_pos, user_vel, user_clock_drift = nav_filter_states(buffers.x_landing, idxs)
        predicted_rate = predict_pseudorange_rate(
            user_pos,
            user_vel,
            user_clock_drift,
            landing_position,
            landing_velocity,
            calc_satellite_clock_drift(sat.decoder, landing_time),
        )
        measured_pseudorange =
            pseudorange_from_tows(
                reference_tow + τ,
                landing_time - (member.time - member.time_gpst_count),
            ) - buffers.delays[j]
        measured_rate = member.wavelength * ustrip(Hz, sat.carrier_doppler_at_landing)
        code_update = nco_code_correction(
            predicted_pseudorange,
            measured_pseudorange,
            member.code_frequency,
            T,
        )
        carrier_update = nco_carrier_correction(predicted_rate, measured_rate, member.wavelength)
        sat.estimator_state = set_vector_corrections(
            sat.estimator_state,
            code_update * Hz,
            carrier_update * Hz,
            sat.landing_lead,
        )
    end
    acc
end

# Release the members of this group whose `active` flag equals `which`, for `reason`.
# Returns whether any was released.
function _release_members!(released, group, g, buffer, vt, which::Bool, reason::VTReleaseReason)
    buffers = vt.buffers
    for (j, member) in enumerate(buffers.members)
        member.group == g && buffers.active[j] == which || continue
        _release!(group.sats[member.slot], reason)
        released = true
    end
    released
end

function _reset_accumulators!(acc, group, g, buffer)
    for sat in group.sats
        state = sat.estimator_state
        state.vt_on && (sat.estimator_state = reset_discriminator_accumulators(state))
    end
    acc
end

function _count_members(n, group, g, buffer)
    for sat in group.sats
        sat.estimator_state.vt_on && (n += 1)
    end
    n
end

# ─────────────────────────────────────────────────────────────────────────────
# Seeding from a scalar fix

function _enable_fix_satellites!(num_enabled, group, g, buffer, vt)
    signal_id = vt.layout.signal_id_by_group[g]
    sats = vt.pvt.sats
    for sat in group.sats
        sat.active && haskey(sats, (signal_id, sat.prn)) || continue
        sat.estimator_state = enable_vector_tracking(sat.estimator_state)
        num_enabled += 1
    end
    num_enabled
end

function _count_fix_satellites(n, group, g, buffer, vt)
    signal_id = vt.layout.signal_id_by_group[g]
    for sat in group.sats
        sat.active && haskey(vt.pvt.sats, (signal_id, sat.prn)) && (n += 1)
    end
    n
end

# Switch from scalar to vector tracking off the fresh scalar fix in `vt.pvt`: promote the
# fix's satellites into the vector loop, seed the navigation filter from the fix, and close
# the loops a first time so the NCOs already steer toward the navigation solution — with
# no measurement update and no accumulator reset. Returns `false`, touching nothing, when
# the fix has kept no tracked satellite.
function _seed!(vt::VectorTrackingState, groups, cycle_time)
    buffers = vt.buffers
    # Need at least one still-tracked fix satellite to seed the loop from: the guard sits
    # before anything is put in the loop and keeps the reference epoch below defined. The
    # filter and its per-cycle observability watchdog take over the geometry check from
    # there.
    _fold_groups(_count_fix_satellites, 0, groups, buffers.states, 1, vt) == 0 && return false
    model = vt.model
    ensure_nav_filter_integration_time!(model, vt.config, cycle_time)
    T = ustrip(s, model.integration_time)
    pvt = vt.pvt
    vt.primary_clock_index = initial_nav_state!(vt.x, vt.P, vt.layout, model.idxs, pvt)
    _fold_groups(_enable_fix_satellites!, 0, groups, buffers.states, 1, vt)
    ionospheric_correction = _gather_members!(vt, groups, T)
    members = buffers.members
    # The pseudorange reference epoch is the latest transmit time corrected by the fix's
    # receiver clock bias, matching how `calc_pvt` timestamps the fix.
    latest = -Inf
    for member in members
        latest = max(latest, member.time_gpst_count)
    end
    reference_tow = latest - ustrip(m, pvt.time_correction) / SPEED_OF_LIGHT
    _measure_pseudoranges!(vt, ionospheric_correction, reference_tow)
    # First loop closure: the filter is seeded exactly at the fix, so the NCO corrections
    # are the prediction residuals at the seeded state.
    resize!(buffers.active, length(members))
    fill!(buffers.active, true)
    _fold_groups(_close_loops!, 0, groups, buffers.states, 1, vt, reference_tow, T)
    vt.running = true
    vt.reference_time = reference_tow * s
    vt.time_with_insufficient_meas = 0.0s
    true
end

# ─────────────────────────────────────────────────────────────────────────────
# One cycle

# One navigation-filter cycle: predict, fuse the accumulated discriminators as
# pseudorange (and, for VDFLL, pseudorange-rate) measurements, close every member's loops
# with fresh NCO corrections, and manage the membership (admission, availability,
# release, fallback to scalar tracking). Returns `(released, fell_back)`.
function _run_cycle!(vt::VectorTrackingState, groups, cycle_time)
    config = vt.config
    buffers = vt.buffers
    model = vt.model
    idxs = model.idxs
    T = ustrip(s, cycle_time)
    ensure_nav_filter_integration_time!(model, config, cycle_time)
    # Advance the receive time-of-week, wrapping at the 604800 s week boundary so it
    # stays a valid seconds-of-week for the pseudorange differencing, the atmospheric
    # time-of-week and the solution's week/second split. The wrap is remembered: the
    # week it drops has to be added to the cached epoch offset
    # (`rolled_over_time_epoch_offset`), which is the only other place the run's
    # absolute epoch is held.
    advanced_reference_time = vt.reference_time + cycle_time
    week_rollover = advanced_reference_time >= SECONDS_PER_WEEK * s
    reference_time = mod(advanced_reference_time, SECONDS_PER_WEEK * s)
    reference_tow = ustrip(s, reference_time)

    # Admission is judged at the solution the filter starts the cycle from.
    admission_frame = ENUfromECEF(_ecef(first(nav_filter_states(vt.x, idxs))), wgs84)
    released =
        _fold_groups(_update_membership!, false, groups, buffers.states, 1, admission_frame)
    ionospheric_correction = _gather_members!(vt, groups, T)
    members = buffers.members
    num_members = length(members)

    # Navigation filter prediction, then the measurements at the predicted state.
    time_update!(buffers.time_update, vt.x, vt.P, model.F, model.Q)
    _measure_pseudoranges!(vt, ionospheric_correction, reference_tow)

    # Measurement candidates: members whose signal is currently available (the
    # discriminators of an obscured satellite carry no information).
    candidates = empty!(buffers.candidates)
    for (j, member) in enumerate(members)
        member.available && push!(candidates, j)
    end
    # What this epoch must supply to determine the state, decided below from the bias
    # layout actually in force. Until a measurement set is gathered it is the bare floor —
    # 3 position components and one clock, from four distinct satellites none of which are
    # there — so an epoch with no candidates at all counts as unsolvable.
    observability = BiasObservability(4, 4, 0)
    # The innovation gate GNSSReceiver once had is gone: the observability watchdog below
    # catches a diverging solution, and a gate could release a healthy satellite whose
    # large-but-explained innovation the Kalman update would have absorbed (e.g. the first
    # measurements of a not-yet-observed constellation).
    if !isempty(candidates)
        observability =
            assess_bias_observability!(buffers.observability, vt.layout, members, candidates)
        _measurement_update!(vt, T)
    end
    num_included = length(candidates)

    # The measurement model at the *updated* state, for every member: the post-fit
    # residuals, and — unless the loop falls back — the NCO corrections.
    _predict_members!(vt)

    # Starvation watchdog: grow the timer whenever the epoch could not determine the
    # navigation state — too few measurements, or too few distinct satellites among them,
    # for the bias layout in force — and pay it back down (at half rate) otherwise. How
    # certain the filter is of the solution it did produce is not policed here; it is
    # reported as `VTStatus`'s `position_std`.
    time_with_insufficient_meas =
        is_epoch_solvable(observability, num_included) ?
        max(0.0s, vt.time_with_insufficient_meas - cycle_time / 2) :
        vt.time_with_insufficient_meas + cycle_time

    # Elevation mask: release members that dropped below the horizon, at the updated
    # position — before the corrections, so every remaining member gets one.
    active = resize!(buffers.active, num_members)
    fill!(active, true)
    user_pos, _, _ = nav_filter_states(vt.x, idxs)
    enu_from_ecef = ENUfromECEF(_ecef(user_pos), wgs84)
    num_active = 0
    for (j, member) in enumerate(members)
        active[j] = get_sat_enu(enu_from_ecef, _ecef(member.sat_position)).ϕ >= 0
        active[j] && (num_active += 1)
    end
    released = _fold_groups(
        _release_members!,
        released,
        groups,
        buffers.states,
        1,
        vt,
        false,
        VT_BELOW_HORIZON,
    )

    # Fall back to scalar tracking when the filter has coasted too long or no members
    # remain; otherwise close every member's loops with fresh corrections toward the
    # updated solution.
    fell_back = time_with_insufficient_meas > config.insufficient_meas_timeout || num_active == 0
    if fell_back
        released = _fold_groups(
            _release_members!,
            released,
            groups,
            buffers.states,
            1,
            vt,
            true,
            VT_FALLBACK,
        )
        vt.running = false
    else
        _fold_groups(_close_loops!, 0, groups, buffers.states, 1, vt, reference_tow, T)
    end

    # The navigation filter consumed this interval's discriminators.
    _fold_groups(_reset_accumulators!, nothing, groups, buffers.states, 1)

    vt.primary_clock_index =
        report_primary_clock_index(vt.layout, members, candidates, vt.primary_clock_index)
    vt.time_epoch_offset = rolled_over_time_epoch_offset(
        _resolve_time_epoch_offset(vt),
        vt.time_epoch_offset,
        week_rollover,
    )
    vt.reference_time = reference_time
    vt.time_with_insufficient_meas = time_with_insufficient_meas
    _write_solution!(vt, groups)
    released, fell_back
end

_ecef(v) = ECEF(v[1], v[2], v[3])

# The epoch offset from a measured member of the primary system, or the cached one.
function _resolve_time_epoch_offset(vt::VectorTrackingState)
    isnothing(vt.time_epoch_offset) || return vt.time_epoch_offset
    members = vt.buffers.members
    for j in vt.buffers.candidates
        members[j].clock_bias_index == vt.primary_clock_index &&
            return time_epoch_offset(vt.buffers.rows[j])
    end
    nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# The solution

function _write_solution!(vt::VectorTrackingState, groups)
    buffers = vt.buffers
    members = buffers.members
    layout = vt.layout
    idxs = vt.model.idxs
    x = vt.x
    user_pos, user_vel, user_clock_drift = nav_filter_states(x, idxs)
    position = _ecef(user_pos)
    velocity = _ecef(user_vel)
    primary_clock_bias = x[idxs.clock_biases[vt.primary_clock_index]]
    included = buffers.candidates

    # Geometry of the satellites this update measured, or `nothing` when there is none to
    # report: no measurements at all, or a rank-deficient design for which `calc_DOP!`
    # returns its all-`-1` sentinel. That sentinel stays inside: `calc_pvt` never emits
    # one either (it rejects the epoch instead), so a solution's `dop` is always either a
    # real geometry or absent.
    dop = nothing
    if !isempty(included)
        columns, primary_column = dense_bias_columns!(
            buffers.dense_clock_indices,
            buffers.dense_ifb_indices,
            buffers.clock_used,
            buffers.ifb_used,
            members,
            included,
            vt.primary_clock_index,
        )
        num_columns = 3 + columns.num_clock_biases + columns.num_ifb
        H = view(buffers.design_matrix, 1:length(included), 1:num_columns)
        position_and_bias_vector!(buffers.ξ, x, idxs)
        calc_H!(H, buffers.candidate_positions, buffers.ξ, columns)
        candidate = calc_DOP!(
            buffers.normal_matrices[num_columns],
            H,
            position,
            primary_column,
        )
        candidate.GDOP < 0 || (dop = candidate)
    end

    # The solution reports the satellites that determined it — the members this update
    # measured — which is what `calc_pvt` reports for a scalar solve, so a consumer reads
    # `pvt.sats` the same way under either tracking mode and `pvt.dop` above describes
    # exactly this set. The coasted members are reported, with their own post-fit
    # residuals, in `member_sats`.
    solution = empty_keeping_capacity!(vt.pvt)
    sats = solution.sats
    for j in included
        set!(sats, _member_key(layout, members[j]), _member_info(buffers, j))
    end
    member_sats = empty_keeping_capacity!(vt.member_sats)
    for j in eachindex(members)
        set!(member_sats, _member_key(layout, members[j]), _member_info(buffers, j))
    end

    inter_system_biases = solution.inter_system_biases
    for (index, time_system) in enumerate(layout.time_systems)
        index == vt.primary_clock_index && continue
        inter_system_biases[time_system] = (x[idxs.clock_biases[index]] - primary_clock_bias) * m
    end
    inter_frequency_biases = solution.inter_frequency_biases
    for (index, band) in enumerate(layout.extra_bands)
        inter_frequency_biases[band] =
            InterFrequencyBias(x[idxs.ifb[index]] * m, layout.reference_bands[index])
    end

    vt.pvt = PVTSolution(
        position,
        velocity,
        calc_course_over_ground(position, velocity),
        primary_clock_bias * m,
        vt_time(vt.time_epoch_offset, vt.reference_time, primary_clock_bias),
        user_clock_drift / SPEED_OF_LIGHT,
        dop,
        sats,
        layout.time_systems[vt.primary_clock_index],
        inter_system_biases,
        inter_frequency_biases,
    )
    nothing
end

_member_key(layout::NavFilterLayout, member::VTMember) =
    (layout.signal_id_by_group[member.group], member.prn)

_member_info(buffers::VTBuffers, j) = SatInfo(
    buffers.members[j].sat_position,
    buffers.members[j].time,
    buffers.residuals[j],
    buffers.rate_residuals[j],
)

# ─────────────────────────────────────────────────────────────────────────────
# The entry point

"""
    update_navigation!(vt::VectorTrackingState, groups, cycle_time) -> (pvt, status::VTStatus)

One navigation cycle over the `groups` (the tuple of [`VTSignalGroup`](@ref)s
`vt` was built for), `cycle_time` after the previous one — the *measured*
interval, not the nominal one.

  - **Vector tracking not running:** the scalar PVT over the satellites marked
    `pvt_ready`. A fresh fix seeds the filter from it, puts the fix's
    satellites that are still tracked into the vector loop and closes their
    loops a first time at the seeded state.
  - **Running:** one filter cycle: the members are admitted and released, their
    accumulated discriminators measured and fused, and every member's NCO
    corrections written into its `estimator_state`, sized for the moment each
    lands (`landing_lead`). Members out of lock stay in the loop, unmeasured.
    After `insufficient_meas_timeout` of unsolvable epochs, or with no member
    left, every member is released and the next cycle solves the scalar PVT
    again.
  - **Built with `config = nothing`:** the scalar PVT only.

Returns the solution (the state's `pvt`) and a [`VTStatus`](@ref). Each
`VTSat`'s `release_reason` says whether and why it was released this cycle.
Nothing is logged; the caller reports the events.

Throws an `ArgumentError` for a `cycle_time` that is not positive and finite, and for
an active satellite whose `landing_lead` is negative or longer than `2.5 ·
cycle_time` (the code measurement keeps the corrections of three cycles to cover
the delay).
"""
function update_navigation!(vt::VectorTrackingState, groups::Tuple, cycle_time)
    buffers = vt.buffers
    0.0s < cycle_time < Inf * s ||
        throw(ArgumentError("the navigation cycle time must be positive and finite"))
    _fold_groups(_check_landing_lead, nothing, groups, buffers.states, 1, uconvert(s, cycle_time))
    _fold_groups(_reset_release_reasons!, nothing, groups, buffers.states, 1)
    enabled = released = fell_back = false
    if vt.enabled && vt.running
        released, fell_back = _run_cycle!(vt, groups, uconvert(s, cycle_time))
    else
        # This cycle's solution comes from the scalar solve, so the filter has no
        # per-member report to make.
        empty_keeping_capacity!(vt.member_sats)
        previous_pvt = vt.pvt
        pvt = _solve_scalar_pvt!(vt, groups)
        # `calc_pvt!` returns the very solution it was handed on an epoch it cannot
        # solve, so identity is exactly the freshness test.
        if vt.enabled && pvt !== previous_pvt
            enabled = _seed!(vt, groups, uconvert(s, cycle_time))
        end
    end
    num_members = _fold_groups(_count_members, 0, groups, buffers.states, 1)
    status = VTStatus(
        vt.running,
        vt.running || fell_back ? position_uncertainty(vt) * m : NaN * m,
        vt.running || fell_back ? clock_uncertainty(vt) * m : NaN * m,
        vt.time_with_insufficient_meas,
        num_members,
        enabled,
        fell_back,
        released,
    )
    vt.pvt, status
end

function _check_landing_lead(acc, group, g, buffer, cycle_time)
    for sat in group.sats
        sat.active || continue
        0.0s <= sat.landing_lead <= MAX_LANDING_LEAD_CYCLES * cycle_time || throw(
            ArgumentError(
                "a satellite's landing lead must lie between zero and 2.5 navigation " *
                "cycles",
            ),
        )
    end
    acc
end

# ─────────────────────────────────────────────────────────────────────────────
# Decoding

"""
    decode_soft_bits!(decoder, state::SignalLoopState) -> decoder

Decode the soft bits `state`'s bit buffer has completed since the last call, and
empty them. The decoder is immutable, so — as with `GNSSDecoder.decode!` — keep
the returned one: it shares the old one's buffers, which this overwrites.
"""
function decode_soft_bits!(decoder, state::SignalLoopState)
    soft_bits = get_soft_bits(state)
    isempty(soft_bits) && return decoder
    decoder = decode!(decoder, soft_bits, length(soft_bits))
    empty!(soft_bits)
    decoder
end
