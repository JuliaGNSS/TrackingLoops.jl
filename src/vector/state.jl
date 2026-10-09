# ─────────────────────────────────────────────────────────────────────────────
# The navigation engine's state: its satellites' slots, buffers and status, and what
# a host reads off it.
# ─────────────────────────────────────────────────────────────────────────────

"""
    VTReleaseReason

Why the latest navigation cycle handed a satellite back to its scalar loop
(see [`release_reason`](@ref)):

  - `VT_NOT_RELEASED`: it was not;
  - `VT_INELIGIBLE`: no longer tracked (no record for two navigation cycles),
    decoded for positioning, or healthy;
  - `VT_BELOW_HORIZON`: below the horizon at the updated position; re-admitted
    only one degree above it (hysteresis);
  - `VT_FALLBACK`: vector tracking stopped (starvation timeout, or no member
    left).

A receiver that forces released satellites out of lock does so for the first
two, not for a fallback.
"""
@enum VTReleaseReason VT_NOT_RELEASED VT_INELIGIBLE VT_BELOW_HORIZON VT_FALLBACK

"""
    SatelliteReport

What a [`VectorPLLAndDLL`](@ref) knows of one satellite (see
[`satellite_report`](@ref)), so a consumer need not decode or estimate C/N₀ again:

  - `prn`, and `tracked`: whether it is stepped now (`false` after two cycles
    without a record; the rest then describes it as it was);
  - `decoder`: its navigation-message decoder up to the last record. It shares
    buffers with the running decoder, so `copy` it to keep it past the next record;
  - `bit_synced`: whether its bit clock has found the bit edges;
  - at its latest snapshot (`epoch`, on the records' time grid, `nothing` before
    the first): `cn0_dbhz`, `in_lock` (synced and C/N₀ above the lock threshold)
    and `pvt_ready` (in lock, decoded for positioning and healthy);
  - `in_vector_loop` and `release_reason`: the latest cycle's decision.

The report is the estimator's own object, refreshed in place by every
`satellite_report` call (no allocation): copy out what is needed beyond the next
call.
"""
mutable struct SatelliteReport{D}
    prn::Int
    tracked::Bool
    decoder::D
    bit_synced::Bool
    epoch::Union{Nothing,typeof(1.0s)}
    cn0_dbhz::Float64
    in_lock::Bool
    pvt_ready::Bool
    in_vector_loop::Bool
    release_reason::VTReleaseReason
end

SatelliteReport(decoder) = SatelliteReport(
    0,
    false,
    decoder,
    false,
    nothing,
    NaN,
    false,
    false,
    false,
    VT_NOT_RELEASED,
)

# One passenger signal of a satellite: what its readings' variances are built from
# (C/N₀, coherent integration time, tap spacing in chips). The readings themselves are
# in the satellite's state (`PassengerReadings`).
struct VTPassenger
    cn0_estimator::MomentsCN0Estimator
    # The host's C/N₀ (dB-Hz) with the last record, read instead of `cn0_estimator`'s;
    # `NaN` if none.
    host_cn0_dbhz::Float64
    # As `VTSlot.held_cn0_dbhz`, for the passenger.
    held_cn0_dbhz::Float64
    integration_time::typeof(1.0s)
    early_late_spacing::Float64
    # Its correlator's DLL variance factor against the BPSK model (`_dll_variance_factor`).
    dll_variance_factor::Float64
end

VTPassenger(cn0_estimator::MomentsCN0Estimator) =
    VTPassenger(cn0_estimator, NaN, NaN, 0.001s, 0.5, 1.0)

_passenger_cn0_dbhz(p::VTPassenger) =
    _cn0_dbhz(p.host_cn0_dbhz, p.held_cn0_dbhz, p.cn0_estimator, p.integration_time)

# One satellite of the navigation engine. A slot is never deleted: a dropped satellite
# leaves it, with its storage, for the next one to reuse.
#
# Bit clock and decoder run on the group's data signal (the driver, or for a dataless
# pilot driver its data passenger); the `data_*` fields follow that signal's records,
# everything else the driver's.
mutable struct VTSlot{D,B<:Unsigned,E<:SatVectorPLLAndDLL}
    prn::Int
    occupied::Bool
    # The registration holding the slot; a state of an older one registers again.
    registration::Int
    # Whether the slot snapshotted the running cycle's epoch.
    active::Bool
    bit_buffer::BitBuffer{B}
    running_decoder::D
    cn0_estimator::MomentsCN0Estimator
    # As `VTPassenger.host_cn0_dbhz`, for the driver.
    host_cn0_dbhz::Float64
    # The engine's own estimate from before the last change of record length, read while
    # `cn0_estimator` refills (see `_restart_cn0`); `NaN` without one.
    held_cn0_dbhz::Float64
    # The fold the bit sync was found in: its later records were correlated
    # before the sync.
    sync_fold_end::Int
    last_end_sample::Int
    last_end_time::Float64 # s
    data_last_end_sample::Int
    data_last_end_time::Float64 # s
    data_last_code_phase_fraction::Float64 # chips past the nearest code-block boundary
    last_integration_time::typeof(1.0s)
    last_early_late_spacing::Float64 # chips
    last_dll_variance_factor::Float64
    # Replica chips from the epoch of the latest snapshot to the last record's end.
    chips_since_epoch::Float64
    # The first epoch the slot can snapshot, and the epoch of its latest snapshot.
    first_epoch::Int
    snapshot_epoch::Int
    # The snapshot the cycle reads and partly writes: the decoder at the last record
    # end before the epoch, the replica moved on to the epoch, and the state there
    # with the discriminators accumulated up to it, carrying the cycle's decisions.
    decoder::D
    estimator_state::E
    code_phase::Float64 # chips since the last decoded data symbol
    carrier_phase::Float64
    carrier_doppler::typeof(1.0Hz)
    code_doppler::typeof(1.0Hz)
    cn0_dbhz::Float64
    coherent_integration_time::typeof(1.0s)
    early_late_spacing::Float64
    dll_variance_factor::Float64
    in_lock::Bool
    pvt_ready::Bool
    # What the latest cycle decided, for the satellite to take up.
    release_reason::VTReleaseReason
    restart_cycle::Int
    correction_cycle::Int
    member_index::Int
    # One per passenger signal of the group, in its order.
    const passengers::Vector{VTPassenger}
    # What `satellite_report` hands out, refreshed in place.
    const report::SatelliteReport{D}
end

function VTSlot(
    data_signal::AbstractGNSSSignal,
    num_passengers::Int,
    prn::Integer,
    state::SatVectorPLLAndDLL,
    num_prompts_for_cn0_estimation,
)
    decoder = GNSSDecoderState(data_signal, prn)
    VTSlot(
        Int(prn),
        false,
        0,
        false,
        SignalLoopState(data_signal).bit_buffer,
        decoder,
        MomentsCN0Estimator(num_prompts_for_cn0_estimation),
        NaN,
        NaN,
        typemin(Int),
        0,
        0.0,
        0,
        0.0,
        0.0,
        0.001s,
        0.5,
        1.0,
        0.0,
        typemax(Int),
        typemin(Int),
        decoder,
        state,
        0.0,
        0.0,
        0.0Hz,
        0.0Hz,
        NaN,
        0.001s,
        0.5,
        1.0,
        false,
        false,
        VT_NOT_RELEASED,
        -1,
        -1,
        0,
        [
            VTPassenger(MomentsCN0Estimator(num_prompts_for_cn0_estimation)) for
            _ = 1:num_passengers
        ],
        SatelliteReport(decoder),
    )
end

# The slots of one driver signal with its passengers, its `data_signal` (see `VTSlot`)
# and the `prototype` state a new slot starts from.
struct VTSlotGroup{
    S<:AbstractGNSSSignal,
    P<:Tuple,
    C<:AbstractGNSSSignal,
    V<:VTSlot,
    E<:SatVectorPLLAndDLL,
}
    signal::S
    passengers::P
    data_signal::C
    slots::Vector{V}
    prototype::E
    num_prompts_for_cn0_estimation::Int
end

function VTSlotGroup(signals::Tuple, prototype::SatVectorPLLAndDLL, capacity, num_prompts)
    driver, passengers = first(signals), Base.tail(signals)
    data_signal = signals[findfirst(signal -> !iszero(get_data_frequency(signal)), signals)]
    slots = [
        VTSlot(data_signal, length(passengers), 1, prototype, num_prompts) for
        _ = 1:capacity
    ]
    VTSlotGroup(
        driver,
        passengers,
        data_signal,
        sizehint!(slots, 2 * capacity),
        prototype,
        num_prompts,
    )
end

_new_slot(group::VTSlotGroup, prn, state) = VTSlot(
    group.data_signal,
    length(group.passengers),
    prn,
    state,
    group.num_prompts_for_cn0_estimation,
)

"""
    VTStatus

What the latest navigation cycle of a [`VectorPLLAndDLL`](@ref) did (see
[`navigation_status`](@ref)).

  - `running`: vector tracking is running after this cycle;
  - `position_std`, `clock_std`: the filter's 1σ 3-D position and primary-clock
    uncertainties (`NaN` while not running); a degenerate geometry shows here,
    not in the starvation timer;
  - `time_with_insufficient_meas`: the starvation timer, grown by unsolvable
    epochs, paid back at half rate otherwise, and checked against
    `insufficient_meas_timeout`;
  - `num_members`: satellites in the vector loop after this cycle;
  - the events `enabled` (a fresh scalar fix seeded the filter), `fell_back`
    (vector tracking stopped) and `released` (some satellite was released; see
    [`release_reason`](@ref)).

The per-member report of the latest cycle is [`member_sats`](@ref).
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

VTStatus() = VTStatus(false, NaN * m, NaN * m, 0.0s, 0, false, false, false)

# The unscented update's buffers (intermediate, `z`, `R`) for one measurement count.
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

# A cycle's scratch, sized for the preallocated slots; vectors grow past that with
# `push!` / `resize!`, the design matrix by replacement.
mutable struct VTBuffers{SB<:Tuple}
    # Per group, the `SatelliteState`s whose rows are collected.
    states::SB
    rows::Vector{SatelliteMeasurement}
    members::Vector{VTMember}
    active::Vector{Bool}
    candidates::Vector{Int}
    # Indices into `candidates` whose rate is fused.
    rate_rows::Vector{Int}
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
        _capacity_vector(Int, max_members),
        _capacity_vector(Float64, max_members),
        _capacity_vector(Float64, max_members),
        _capacity_vector(Float64, max_members),
        _capacity_vector(Float64, max_members),
        _capacity_vector(typeof(1.0m), max_members),
        _capacity_vector(typeof(1.0m / s), max_members),
        _capacity_vector(SVector{3,Float64}, max_members),
        BiasColumns(
            _capacity_vector(Int, max_members),
            num_clocks,
            _capacity_vector(Int, max_members),
            num_ifbs,
        ),
        _capacity_vector(SVector{3,Float64}, max_members),
        _capacity_vector(SVector{3,Float64}, max_members),
        _capacity_vector(Float64, max_members),
        BiasColumns(
            _capacity_vector(Int, max_members),
            num_clocks,
            _capacity_vector(Int, max_members),
            num_ifbs,
        ),
        zeros(num_lsq),
        zeros(num_lsq),
        zeros(num_lsq),
        zeros(num_states),
        [zero(SVector{3,Float64})],
        BiasColumns([1], num_clocks, [0], num_ifbs),
        [0.0],
        ObservabilityWorkspace(max_members),
        KFTUIntermediate(Float64, num_states),
        # One per possible measurement count, so a membership change never builds one.
        Union{Nothing,MeasurementBuffers{_UKFMU}}[
            MeasurementBuffers(num_states, k) for k = 1:max_measurements
        ],
        _capacity_vector(Int, max_members),
        _capacity_vector(Int, max_members),
        zeros(Int, num_clocks),
        zeros(Int, num_ifbs),
        zeros(max(max_members, 1), num_lsq),
        [zeros(n, n) for n = 1:num_lsq],
    )
end

"""
    VectorNavigation

The navigation engine of a [`VectorPLLAndDLL`](@ref), shared by every satellite
it steps: the slots, the bias layout, the navigation filter's model, state and
covariance, and every buffer a cycle needs.

It holds the latest solution (`pvt`, whose containers every cycle reuses) and
`member_sats` (see [`member_sats`](@ref)). `pvt.sats` carries only the members
the update measured, so the keys missing from it are the members the filter
coasted through an obscuration.
"""
mutable struct VectorNavigation{G<:Tuple,SB<:Tuple}
    const config::VectorTracking
    const enabled::Bool
    const layout::NavFilterLayout
    const model::NavFilterModel
    const x::Vector{Float64}
    const P::Matrix{Float64}
    running::Bool
    # The clock-bias state the solution is reported against.
    primary_clock_index::Int
    # The pseudoranges' reference epoch, on the GPS Time count (see
    # `VTMember.time_gpst_count`) so it does not move when the primary clock changes.
    reference_time::typeof(1.0s)
    time_with_insufficient_meas::typeof(1.0s)
    # Constant part (s) of the reported epoch (see `time_epoch_offset`); `nothing`
    # until a primary-system member has been measured.
    time_epoch_offset::Union{Nothing,Int}
    const member_sats::Dictionary{Tuple{Symbol,Int},SatInfo}
    pvt::PVTSolution
    status::VTStatus
    const workspace::PVTWorkspace
    const approximate_year::Int
    const enable_ionospheric_correction::Bool
    const enable_tropospheric_correction::Bool
    const buffers::VTBuffers{SB}
    const groups::G
    const cycle_time::typeof(1.0s)
    const lock_cn0_threshold::Float64 # dB-Hz
    # The epoch (in `cycle_time`s on the records' time grid) the slots snapshot next,
    # and how many have.
    pending_epoch::Int
    num_snapshots::Int
    # The latest cycle's id, epoch and the integration time its corrections assume.
    cycle_id::Int
    cycle_epoch::Int
    cycle_integration_time::Float64 # s
    registrations::Int
end

function VectorNavigation(
    config::Union{VectorTracking,Nothing},
    signal_groups::Tuple{Vararg{Tuple}},
    inner::AbstractDopplerEstimator;
    cycle_time,
    lock_cn0_threshold::Float64,
    max_satellites_per_signal::Int,
    num_prompts_for_cn0_estimation::Int,
    approximate_year::Integer,
    enable_ionospheric_correction::Bool,
    enable_tropospheric_correction::Bool,
)
    filter_config = isnothing(config) ? VectorTracking() : config
    # The filter ranges on the drivers.
    signals = map(first, signal_groups)
    layout = NavFilterLayout(signals)
    model = NavFilterModel(filter_config, layout, uconvert(s, cycle_time))
    n = num_nav_states(filter_config, layout)
    groups = map(signal_groups) do group_signals
        inner_state = init_estimator_state(inner, first(group_signals), 0.0Hz, 0.0Hz)
        prototype = SatVectorPLLAndDLL(inner_state, false, Base.tail(group_signals))
        VTSlotGroup(
            group_signals,
            prototype,
            max_satellites_per_signal,
            num_prompts_for_cn0_estimation,
        )
    end
    max_members = max_satellites_per_signal * length(signals)
    states = map(_satellite_state_buffer, groups)
    VectorNavigation(
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
        VTStatus(),
        PVTWorkspace(),
        Int(approximate_year),
        enable_ionospheric_correction,
        enable_tropospheric_correction,
        VTBuffers(states, n, layout, max_members),
        groups,
        uconvert(s, cycle_time),
        lock_cn0_threshold,
        typemin(Int),
        0,
        0,
        typemin(Int),
        ustrip(s, cycle_time),
        0,
    )
end

_satellite_state_buffer(group::VTSlotGroup{S,P,C,<:VTSlot{D}}) where {S,P,C,D} =
    sizehint!(SatelliteState{Float64,D,S}[], 2 * max(length(group.slots), 1))

"""
    position_uncertainty(estimator::VectorPLLAndDLL)

The navigation filter's 1σ 3-D position uncertainty (m), once seeded.
"""
position_uncertainty(estimator::VectorPLLAndDLL) =
    position_uncertainty(estimator.navigation)

"""
    clock_uncertainty(estimator::VectorPLLAndDLL)

The navigation filter's 1σ uncertainty (m) of the solution's reference clock
bias, once seeded.
"""
clock_uncertainty(estimator::VectorPLLAndDLL) = clock_uncertainty(estimator.navigation)

position_uncertainty(vt::VectorNavigation) = position_uncertainty(vt.P, vt.model.idxs)

function clock_uncertainty(vt::VectorNavigation)
    index = vt.model.idxs.clock_biases[vt.primary_clock_index]
    sqrt(vt.P[index, index])
end

"""
    navigation_solution(estimator::VectorPLLAndDLL) -> PVTSolution

The latest navigation solution: the scalar PVT's until the filter is seeded, the
filter's after. The next cycle reuses its containers; copy out what is needed.
"""
navigation_solution(estimator::VectorPLLAndDLL) = estimator.navigation.pvt

"""
    navigation_status(estimator::VectorPLLAndDLL) -> VTStatus

What the latest navigation cycle did (see [`VTStatus`](@ref)).
"""
navigation_status(estimator::VectorPLLAndDLL) = estimator.navigation.status

navigation_cycle(estimator::VectorPLLAndDLL) = estimator.navigation.cycle_id

function navigation_epoch(estimator::VectorPLLAndDLL)
    nav = estimator.navigation
    nav.cycle_epoch == typemin(Int) ? nothing : nav.cycle_epoch * nav.cycle_time
end

function satellite_report(
    estimator::VectorPLLAndDLL,
    signal::AbstractGNSSSignal,
    prn::Integer,
)
    nav = estimator.navigation
    _satellite_report(nav, nav.groups, signal, Int(prn))
end

_satellite_report(nav, ::Tuple{}, signal, prn) = nothing
function _satellite_report(nav, groups::Tuple, signal::S, prn) where {S}
    group = first(groups)
    group.signal isa S || return _satellite_report(nav, Base.tail(groups), signal, prn)
    for slot in group.slots
        slot.prn == prn && slot.registration > 0 || continue
        report = slot.report
        report.prn = slot.prn
        report.tracked = slot.occupied
        report.decoder = slot.running_decoder
        report.bit_synced = slot.bit_buffer.found
        report.epoch =
            slot.snapshot_epoch == typemin(Int) ? nothing :
            slot.snapshot_epoch * nav.cycle_time
        report.cn0_dbhz = slot.cn0_dbhz
        report.in_lock = slot.in_lock
        report.pvt_ready = slot.pvt_ready
        report.in_vector_loop = slot.estimator_state.vt_on
        report.release_reason = slot.release_reason
        return report
    end
    nothing
end

"""
    member_sats(estimator::VectorPLLAndDLL)

The latest cycle's per-member report: every member of the vector loop, measured
*and* coasted, keyed `(signal_id, prn)` like the solution's `sats`, each with the
satellite position, transmit time and post-fit residuals. Empty while the scalar
solve is in control.
"""
member_sats(estimator::VectorPLLAndDLL) = estimator.navigation.member_sats

"""
    release_reason(estimator::VectorPLLAndDLL, signal, prn) -> VTReleaseReason

Whether and why the latest navigation cycle released satellite `prn` of
`signal` from the vector loop.
"""
function release_reason(
    estimator::VectorPLLAndDLL,
    signal::AbstractGNSSSignal,
    prn::Integer,
)
    _release_reason(estimator.navigation.groups, signal, Int(prn))
end

_release_reason(::Tuple{}, signal, prn) = VT_NOT_RELEASED
function _release_reason(groups::Tuple, signal::S, prn) where {S}
    group = first(groups)
    group.signal isa S || return _release_reason(Base.tail(groups), signal, prn)
    for slot in group.slots
        slot.prn == prn && slot.registration > 0 && return slot.release_reason
    end
    VT_NOT_RELEASED
end
