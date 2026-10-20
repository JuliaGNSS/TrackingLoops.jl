# ─────────────────────────────────────────────────────────────────────────────
# Vector tracking: the navigation engine that closes all loops
#
# One `VectorNavigation` is shared by every satellite a `VectorPLLAndDLL` steps.
# It keeps one slot per satellite, which the per-record step fills from the
# records themselves: the decoder, from the soft bits of the satellite's decoding
# signal, and at every navigation epoch a snapshot of the driver's replica, its
# C/N₀ and the accumulated discriminators, with the decoding signal's symbol
# count. Once every satellite has completed an epoch's snapshot, the cycle runs:
# before vector tracking it is a scalar PVT solve, whose first fix seeds the
# navigation filter; from then on it is one filter iteration. The cycle leaves
# its decisions in the slots — admission, release, the commands — and each
# satellite takes them up on its next record. Nothing here knows a correlator:
# the software receiver and a hardware loop process step the same estimator.
# ─────────────────────────────────────────────────────────────────────────────

"""
    VTReleaseReason

Why the latest navigation cycle handed a satellite back to its scalar loop
(see [`release_reason`](@ref)):

  - `VT_NOT_RELEASED`: it was not;
  - `VT_INELIGIBLE`: no longer tracked (no record for two navigation cycles),
    decoded for positioning, or healthy;
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
    SatelliteReport

What a [`VectorPLLAndDLL`](@ref) knows of one satellite (see
[`satellite_report`](@ref)), so a consumer need not decode its bits or estimate
its C/N₀ again:

  - `prn`, and `tracked`: whether a satellite is stepped on it now (`false` once
    it went two cycles without a record; the rest then describes it as it was);
  - `decoder`: its navigation-message decoder, up to the last record of its
    decoding signal — the ephemeris, health and time of week. It shares its
    buffers with the one the estimator keeps decoding into, so copy
    (`copy(decoder)`) what is needed after the next record. It is the
    estimator's, to read only: never reset or decode into it (e.g.
    `reset_decoder_state!`, `decode!`), as that changes the estimator's own
    decoder in place. Once the satellite's slot is freed, the slot's buffers
    are reused for whichever satellite registers on it next, so a decoder kept
    past that point shares them with another satellite's: a consumer that needs
    a decoder of its own for a satellite builds a fresh one;
  - `bit_synced`: whether its decoding signal's bit clock held the bit sync at
    that signal's last record;
  - at the latest epoch it was snapshotted at (`epoch`, on the records' time
    grid, `nothing` before the first): `cn0_dbhz`, the driver's, from the C/N₀
    estimator the host configured for it, `in_lock` (the decoding signal
    synced and the driver's C/N₀ above the lock threshold) and `pvt_ready` (in
    lock, decoded for positioning and healthy);
  - `in_vector_loop`: whether the latest cycle has it in the vector loop, and
    `release_reason` whether and why that cycle released it.

The report is the estimator's own object, one per satellite slot, refreshed by
every `satellite_report` call, which therefore allocates nothing: copy out what
is needed beyond the next call.
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

SatelliteReport(decoder) =
    SatelliteReport(0, false, decoder, false, nothing, NaN, false, false, false, VT_NOT_RELEASED)

# One satellite of the navigation engine. A slot is never deleted: a satellite that is
# dropped leaves it free with all its storage, for the next satellite to reuse.
mutable struct VTSlot{D,E<:SatVectorPLLAndDLL}
    prn::Int
    occupied::Bool
    # The registration that holds the slot; a state of an older one registers again.
    registration::Int
    # Whether the slot snapshotted the epoch of the running cycle.
    in_cycle::Bool
    # The decoder, advanced by every record of the decoding signal.
    running_decoder::D
    # The driver's last record end.
    driver_last_end_sample::Int
    driver_last_end_time::Float64 # s
    last_code_phase::Float64 # chips, as the record reported it; `NaN` if it did not
    last_integration_time::typeof(1.0s)
    last_early_late_spacing::Float64 # chips
    # Replica chips from the epoch of the latest snapshot to the driver's last record end.
    chips_since_epoch::Float64
    # The decoding signal's last record end, and its bit clock there.
    decoding_last_end_sample::Int
    decoding_last_end_time::Float64 # s
    decoding_bit_synced::Bool
    decoding_blocks_into_symbol::Int
    decoding_code_phase_fraction::Float64 # chips past the nearest code-block boundary
    # The fold the decoding signal's bit sync was found in: its later records were
    # correlated before the sync.
    sync_fold_end::Int
    # The data symbol in the making while the filter steers the carrier: the decoding
    # signal's prompts rotated onto the incoming carrier by the residual PLL, whether
    # they cover the symbol from its edge on, and the soft bit it ends in.
    symbol_accumulator::ComplexF64
    symbol_from_edge::Bool
    const soft_bit::Vector{Float32}
    # The residual PLL's latest prompts, for a pair's decoding signal (`PromptHistory`).
    const prompt_history::PromptHistory
    # The first epoch the slot can snapshot; the epochs its driver's and its decoding
    # signal's halves of a snapshot were taken at, and the epoch of its latest complete
    # snapshot.
    first_epoch::Int
    driver_epoch::Int
    decoding_epoch::Int
    snapshot_epoch::Int
    # The decoding signal's half of the snapshot in the making: the decoder at its last
    # record end before the epoch, whether its bit clock held the sync there, and the
    # code phase at the epoch from the decoder's last symbol edge (chips).
    epoch_decoder::D
    epoch_bit_synced::Bool
    epoch_code_phase::Float64
    # The driver's replica code phase at the epoch (chips; `NaN` if not reported).
    epoch_driver_code_phase::Float64
    # The snapshot at the epoch: what the cycle reads, and partly writes. The decoder
    # is the one at the record end before the epoch, the replica moved on to the
    # epoch; `estimator_state` is the satellite's state there, with the
    # discriminators accumulated up to it, and carries the cycle's decisions.
    decoder::D
    estimator_state::E
    code_phase::Float64 # chips since the last decoded data symbol
    carrier_phase::Float64
    carrier_doppler::typeof(1.0Hz)
    code_doppler::typeof(1.0Hz)
    cn0_dbhz::Float64 # the driver's
    coherent_integration_time::typeof(1.0s)
    early_late_spacing::Float64
    in_lock::Bool
    pvt_ready::Bool
    # What the latest cycle decided, for the satellite to take up.
    release_reason::VTReleaseReason
    command_cycle::Int
    member_index::Int
    # What `satellite_report` hands out, refreshed in place.
    const report::SatelliteReport{D}
end

function VTSlot(
    decoding_signal::AbstractGNSSSignal,
    prn::Integer,
    state::SatVectorPLLAndDLL,
    prompt_history_length::Int,
)
    decoder = GNSSDecoderState(decoding_signal, prn)
    VTSlot(
        Int(prn),
        false,
        0,
        false,
        decoder,
        0,
        0.0,
        NaN,
        0.001s,
        0.5,
        0.0,
        0,
        0.0,
        false,
        0,
        NaN,
        typemin(Int),
        zero(ComplexF64),
        false,
        zeros(Float32, 1),
        PromptHistory(prompt_history_length),
        typemax(Int),
        typemin(Int),
        typemin(Int),
        typemin(Int),
        decoder,
        false,
        0.0,
        NaN,
        decoder,
        state,
        0.0,
        0.0,
        0.0Hz,
        0.0Hz,
        NaN,
        0.001s,
        0.5,
        false,
        false,
        VT_NOT_RELEASED,
        -1,
        0,
        SatelliteReport(decoder),
    )
end

# The slots of one ranging signal: the driver the satellites range on, the signal
# whose navigation data they decode (the driver itself for a plain signal), a fresh
# satellite state to fill a new slot with, and how many prompts a slot's history holds
# (`_prompt_history_length`).
struct VTSlotGroup{S<:AbstractGNSSSignal,DS<:AbstractGNSSSignal,V<:VTSlot,E<:SatVectorPLLAndDLL}
    driver_signal::S
    decoding_signal::DS
    slots::Vector{V}
    prototype::E
    prompt_history_length::Int
end

function VTSlotGroup(
    driver_signal::AbstractGNSSSignal,
    decoding_signal::AbstractGNSSSignal,
    prototype::SatVectorPLLAndDLL,
    capacity,
    cycle_time,
)
    prompt_history_length = _prompt_history_length(driver_signal, decoding_signal, cycle_time)
    slots = [VTSlot(decoding_signal, 1, prototype, prompt_history_length) for _ = 1:capacity]
    VTSlotGroup(driver_signal, decoding_signal, sizehint!(slots, 2 * capacity), prototype, prompt_history_length)
end

"""
    VTStatus

What the latest navigation cycle of a [`VectorPLLAndDLL`](@ref) did (see
[`navigation_status`](@ref)).

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

# The buffers of one measurement count: the unscented update's intermediate and
# the measurement vector and noise covariance it is handed.
struct MeasurementBuffers{I}
    intermediate::I
    z::Vector{Float64}
    R::Matrix{Float64}
end

const _UKFIntermediate = typeof(UKFMUIntermediate(Float64, 1, 1))

MeasurementBuffers(num_states, num_measurements) = MeasurementBuffers{_UKFIntermediate}(
    UKFMUIntermediate(Float64, num_states, num_measurements),
    zeros(num_measurements),
    zeros(num_measurements, num_measurements),
)

# Everything a cycle works in, sized at construction for the preallocated slots. The
# vectors grow past that with `push!` / `resize!`, the design matrix by replacement.
mutable struct VTBuffers{SB<:Tuple}
    # Per group, the `SatelliteState`s whose rows are collected.
    sat_states::SB
    rows::Vector{SatelliteMeasurement}
    members::Vector{VTMember}
    # Per member, whether it stayed above the horizon this cycle.
    above_horizon::Vector{Bool}
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
    measurement_updates::Vector{Union{Nothing,MeasurementBuffers{_UKFIntermediate}}}
    dense_clock_indices::Vector{Int}
    dense_ifb_indices::Vector{Int}
    clock_columns::Vector{Int}
    ifb_columns::Vector{Int}
    design_matrix::Matrix{Float64}
    normal_matrices::Vector{Matrix{Float64}}
end

_capacity_vector(::Type{T}, n) where {T} = sizehint!(T[], n)

function VTBuffers(sat_states::Tuple, num_states, layout::NavFilterLayout, max_members)
    num_clocks = num_clock_biases(layout)
    num_ifbs = num_ifb(layout)
    num_lsq = 3 + num_clocks + num_ifbs
    max_measurements = 2 * max_members + num_clocks
    VTBuffers(
        sat_states,
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
        # One per measurement count the slots can produce, so a change of membership
        # never builds one mid-run.
        Union{Nothing,MeasurementBuffers{_UKFIntermediate}}[
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
it steps: the slots, the bias layout of its signals, the navigation filter's
process model, state and covariance, and every buffer a cycle needs.

It holds the latest solution (`pvt`, a `PVTSolution` whose containers every
cycle reuses) and the per-member report of the latest cycle (`member_sats`):
every member of the loop, measured *and* coasted, keyed exactly as `pvt.sats`
is, each with the satellite position, transmit time and post-fit residuals that
update produced. `pvt.sats` carries only the members the update measured, so
the difference between the two key sets is what the filter predicted through
an obscuration. Emptied while the scalar solve is in control.
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
    # The epoch the pseudoranges are referenced to — always on the GPS Time count (see
    # `VTMember.time_gpst_count`), so it does not move when the primary clock changes.
    reference_time::typeof(1.0s)
    time_with_insufficient_meas::typeof(1.0s)
    # Constant part of the reported epoch (seconds), cached once resolved — see
    # `time_epoch_offset`. `nothing` until a primary-system member has been measured.
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
    const lock_cn0_threshold_dbhz::Float64
    # The epoch (a multiple of `cycle_time` on the records' time grid) the slots
    # snapshot next, and how many have.
    pending_epoch::Int
    num_snapshots::Int
    # The latest cycle: its id and its epoch.
    cycle_id::Int
    cycle_epoch::Int
    registrations::Int
end

function VectorNavigation(
    config::Union{VectorTracking,Nothing},
    pairs::Tuple,
    inner::AbstractDopplerEstimator;
    cycle_time,
    lock_cn0_threshold_dbhz::Float64,
    max_satellites_per_signal::Int,
    approximate_year::Integer,
    enable_ionospheric_correction::Bool,
    enable_tropospheric_correction::Bool,
)
    filter_config = something(config, VectorTracking())
    signals = map(first, pairs)
    layout = NavFilterLayout(signals)
    model = NavFilterModel(filter_config, layout, uconvert(s, cycle_time))
    n = num_nav_states(filter_config, layout)
    groups = map(pairs) do (signal, decoding_signal)
        prototype = SatVectorPLLAndDLL(init_estimator_state(inner, signal, 0.0Hz, 0.0Hz), false)
        VTSlotGroup(signal, decoding_signal, prototype, max_satellites_per_signal, cycle_time)
    end
    max_members = max_satellites_per_signal * length(signals)
    sat_states = map(_satellite_state_buffer, groups)
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
        VTBuffers(sat_states, n, layout, max_members),
        groups,
        uconvert(s, cycle_time),
        lock_cn0_threshold_dbhz,
        typemin(Int),
        0,
        0,
        typemin(Int),
        0,
    )
end

_satellite_state_buffer(group::VTSlotGroup{S,DS,<:VTSlot{D}}) where {S,DS,D} =
    sizehint!(SatelliteState{Float64,D,S}[], 2 * max(length(group.slots), 1))

"""
    position_uncertainty(estimator::VectorPLLAndDLL)

The navigation filter's own 1σ uncertainty (m) of the 3-D position. Meaningful
once the filter has been seeded.
"""
position_uncertainty(estimator::VectorPLLAndDLL) = position_uncertainty(estimator.navigation)

"""
    clock_uncertainty(estimator::VectorPLLAndDLL)

The navigation filter's own 1σ uncertainty (m) of the clock bias the solution is
referenced to. Meaningful once the filter has been seeded.
"""
clock_uncertainty(estimator::VectorPLLAndDLL) = clock_uncertainty(estimator.navigation)

position_uncertainty(nav::VectorNavigation) = position_uncertainty(nav.P, nav.model.idxs)

function clock_uncertainty(nav::VectorNavigation)
    index = nav.model.idxs.clock_biases[nav.primary_clock_index]
    sqrt(nav.P[index, index])
end

"""
    navigation_solution(estimator::VectorPLLAndDLL) -> PVTSolution

The latest navigation solution: the scalar PVT's until the filter is seeded,
the filter's while vector tracking runs. Its containers are reused by the next
cycle, so copy out what is needed later.
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

function satellite_report(estimator::VectorPLLAndDLL, signal::AbstractGNSSSignal, prn::Integer)
    nav = estimator.navigation
    _satellite_report(nav, nav.groups, signal, Int(prn))
end

_satellite_report(nav, ::Tuple{}, signal, prn) = nothing
function _satellite_report(nav, groups::Tuple, signal::S, prn) where {S}
    group = first(groups)
    group.driver_signal isa S || return _satellite_report(nav, Base.tail(groups), signal, prn)
    for slot in group.slots
        slot.prn == prn && slot.registration > 0 || continue
        report = slot.report
        report.prn = slot.prn
        report.tracked = slot.occupied
        report.decoder = slot.running_decoder
        report.bit_synced = slot.decoding_bit_synced
        report.epoch =
            slot.snapshot_epoch == typemin(Int) ? nothing : slot.snapshot_epoch * nav.cycle_time
        report.cn0_dbhz = slot.cn0_dbhz
        report.in_lock = slot.in_lock
        report.pvt_ready = slot.pvt_ready
        report.in_vector_loop = slot.estimator_state.in_vector_loop
        report.release_reason = slot.release_reason
        return report
    end
    nothing
end

"""
    member_sats(estimator::VectorPLLAndDLL)

The per-member report of the latest cycle: every member of the vector loop,
measured *and* coasted, keyed `(signal_id, prn)` as the solution's `sats` are,
each with the satellite position, transmit time and post-fit residuals. Empty
while the scalar solve is in control.
"""
member_sats(estimator::VectorPLLAndDLL) = estimator.navigation.member_sats

"""
    release_reason(estimator::VectorPLLAndDLL, signal, prn) -> VTReleaseReason

Whether and why the latest navigation cycle released satellite `prn` of
`signal` from the vector loop.
"""
function release_reason(estimator::VectorPLLAndDLL, signal::AbstractGNSSSignal, prn::Integer)
    _release_reason(estimator.navigation.groups, signal, Int(prn))
end

_release_reason(::Tuple{}, signal, prn) = VT_NOT_RELEASED
function _release_reason(groups::Tuple, signal::S, prn) where {S}
    group = first(groups)
    group.driver_signal isa S || return _release_reason(Base.tail(groups), signal, prn)
    for slot in group.slots
        slot.prn == prn && slot.registration > 0 && return slot.release_reason
    end
    VT_NOT_RELEASED
end

# ─────────────────────────────────────────────────────────────────────────────
# Walking the groups

# `f(accumulator, group, g, buffer, args...)` over the groups, in order, with the
# group's position `g` and its entry of the per-group `buffers`; returns the final
# accumulator. Recursion over the tuples keeps every group's call statically
# dispatched, however heterogeneous the groups are.
@inline _fold_groups_with_buffers(f::F, acc, ::Tuple{}, ::Tuple{}, g, args...) where {F} = acc
@inline _fold_groups_with_buffers(f::F, acc, groups::Tuple, buffers::Tuple, g, args...) where {F} =
    _fold_groups_with_buffers(
        f,
        f(acc, first(groups), g, first(buffers), args...),
        Base.tail(groups),
        Base.tail(buffers),
        g + 1,
        args...,
    )

# The per-group `SignalGroup`s over `buffers`, for the PVT collection pass.
_signal_groups(groups, buffers) =
    map((group, buffer) -> SignalGroup(group.driver_signal, buffer), groups, buffers)

# ─────────────────────────────────────────────────────────────────────────────
# The scalar solve

function _collect_pvt_ready!(acc, group, g, buffer)
    empty!(buffer)
    for slot in group.slots
        slot.in_cycle && slot.pvt_ready || continue
        push!(buffer, _satellite_state(group.driver_signal, slot))
    end
    acc
end

@inline _satellite_state(signal, slot::VTSlot) = SatelliteState(;
    decoder = slot.decoder,
    system = signal,
    code_phase = slot.code_phase,
    carrier_doppler = slot.carrier_doppler,
    carrier_phase = slot.carrier_phase,
)

# The scalar PVT over the satellites marked `pvt_ready`, into the state's solution:
# a fresh fix is a new object, a failed epoch the solution itself.
function _solve_scalar_pvt!(nav::VectorNavigation, groups)
    buffers = nav.buffers.sat_states
    _fold_groups_with_buffers(_collect_pvt_ready!, nothing, groups, buffers, 1)
    nav.pvt = calc_pvt!(
        nav.pvt,
        nav.workspace,
        _signal_groups(groups, buffers),
        nav.pvt;
        approximate_year = nav.approximate_year,
        enable_ionospheric_correction = nav.enable_ionospheric_correction,
        enable_tropospheric_correction = nav.enable_tropospheric_correction,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Membership

# Hand `slot` back to its scalar loop for `reason`. The satellite takes it up on its
# next record, re-seeding its scalar loop from the replica where that lands
# (`_take_up_cycle`).
function _mark_release!(slot::VTSlot, reason::VTReleaseReason)
    slot.estimator_state = _release_from_vector_loop(slot.estimator_state)
    slot.release_reason = reason
    nothing
end

# Whether the slot is a member of the vector loop this cycle: it snapshotted the
# epoch, and its state at the epoch is in the loop. A member that missed the
# epoch sits the cycle out.
_is_member(slot::VTSlot) = slot.in_cycle && slot.estimator_state.in_vector_loop

_is_eligible(slot::VTSlot) =
    slot.in_cycle &&
    is_decoding_completed_for_positioning(slot.decoder) &&
    is_sat_healthy(slot.decoder)

# The elevation a satellite has to reach before it is admitted, above the horizon at
# which a member is released: the hysteresis keeps a satellite near the horizon from
# being admitted and released, its scalar loop re-seeded, every cycle. One degree is a
# few minutes of a rising satellite.
const ADMISSION_ELEVATION = deg2rad(1.0)

# Whether a satellite stands at least `ADMISSION_ELEVATION` above the horizon of
# `enu_from_ecef`, at its transmit time.
function _is_above_admission_mask(signal, slot::VTSlot, enu_from_ecef)
    orbit = calc_satellite_position_and_velocity(_satellite_state(signal, slot))
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
# receiving commands.
function _update_membership!(released, group, g, buffer, enu_from_ecef)
    for slot in group.slots
        slot.in_cycle || continue
        eligible = _is_eligible(slot)
        state = slot.estimator_state
        if eligible && slot.in_lock
            if state.in_vector_loop || _is_above_admission_mask(group.driver_signal, slot, enu_from_ecef)
                slot.estimator_state = _add_to_vector_loop(state)
            end
        elseif !eligible && state.in_vector_loop
            _mark_release!(slot, VT_INELIGIBLE)
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
    for slot in group.slots
        _is_member(slot) && push!(buffer, _satellite_state(group.driver_signal, slot))
    end
    acc
end

# One member per satellite in the loop, from its row; returns the running member
# index.
function _collect_members!(j, group, g, buffer, nav, T)
    layout = nav.layout
    buffers = nav.buffers
    signal = group.driver_signal
    code_frequency = ustrip(Hz, get_code_frequency(signal))
    chip_length = SPEED_OF_LIGHT / code_frequency
    wavelength = SPEED_OF_LIGHT / ustrip(Hz, get_center_frequency(signal))
    clock_bias_index = layout.clock_bias_index_by_group[g]
    ifb_index = layout.ifb_index_by_group[g]
    for (slot_index, slot) in enumerate(group.slots)
        _is_member(slot) || continue
        state = slot.estimator_state
        j += 1
        row = buffers.rows[j]
        push!(
            buffers.members,
            VTMember(
                g,
                slot_index,
                slot.prn,
                clock_bias_index,
                ifb_index,
                chip_length,
                wavelength,
                code_frequency,
                slot.in_lock && has_disc_sums(state),
                row.time,
                row.time - row.count_offset_to_gpst,
                row.position,
                row.velocity,
                row.clock_drift,
                mean_code_disc_at_epoch(state, T),
                wavelength *
                (ustrip(Hz, slot.carrier_doppler) + mean_fll_disc(state)),
                linear_cn0_floor(slot.cn0_dbhz),
                slot.early_late_spacing,
                ustrip(s, slot.coherent_integration_time),
                row.time_offsets,
            ),
        )
    end
    j
end

# Gather this cycle's members: the satellites in the loop, their rows at the epoch, and
# the per-member model pieces. Returns the ionospheric correction the rows select.
function _gather_members!(nav::VectorNavigation, groups, T)
    buffers = nav.buffers
    _fold_groups_with_buffers(_collect_member_states!, nothing, groups, buffers.sat_states, 1)
    ionospheric_correction = collect_measurement_rows!(
        buffers.rows,
        _signal_groups(groups, buffers.sat_states);
        approximate_year = nav.approximate_year,
    )
    # Every member is decoded and healthy, which is exactly what the collection pass
    # keeps, so the rows are the members in order.
    num_rows = length(buffers.rows)
    num_members = mapreduce(length, +, buffers.sat_states; init = 0)
    num_rows == num_members || throw(
        ArgumentError(
            "a vector-loop member was not decoded for positioning and healthy; " *
            "its measurement row is missing",
        ),
    )
    empty!(buffers.members)
    _fold_groups_with_buffers(_collect_members!, 0, groups, buffers.sat_states, 1, nav, T)
    members = buffers.members
    resize!(buffers.positions, num_members)
    for j in eachindex(members)
        buffers.positions[j] = members[j].sat_position
    end
    member_bias_columns!(buffers.member_columns, members, eachindex(members))
    ionospheric_correction
end

# The atmosphere-corrected pseudoranges of every member against `reference_tow`, with
# the delays predicted at the state `x`.
function _measure_pseudoranges!(nav::VectorNavigation, ionospheric_correction, reference_tow)
    buffers = nav.buffers
    members = buffers.members
    rows = buffers.rows
    num_members = length(members)
    delays = resize!(buffers.delays, num_members)
    if num_members > 0 &&
       (nav.enable_ionospheric_correction || nav.enable_tropospheric_correction)
        # The Niell mapping's seasonal term takes the day of year, derived the way
        # `calc_pvt` derives it: from a decoded satellite's absolute week plus the time of
        # week. Any member dates the epoch — every member has finished decoding and the
        # time systems differ by at most their defined scale offset, 14 s for BDT, against
        # a one-year period — so the first one serves.
        doy = day_of_year(rows[1].system_start_time, rows[1].week, reference_tow)
        position_and_bias_vector!(buffers.ξ, nav.x, nav.model.idxs)
        predict_atmospheric_delays!(
            delays,
            buffers.ξ,
            rows,
            IonosphericModel(nav.enable_ionospheric_correction ? ionospheric_correction : nothing),
            reference_tow,
            doy,
            nav.enable_tropospheric_correction,
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
        # The element type is not isbits: every new slot must be filled, or it is
        # `#undef` for a later, smaller count.
        num_cached = length(cache)
        resize!(cache, num_measurements)
        fill!(view(cache, (num_cached+1):num_measurements), nothing)
    end
    cached = cache[num_measurements]
    isnothing(cached) || return cached
    new_buffers = MeasurementBuffers(num_states, num_measurements)
    cache[num_measurements] = new_buffers
    new_buffers
end

# Fuse the candidates' measurements at the predicted state `nav.x`, leaving the posterior
# in `nav.x` and `nav.P`.
function _measurement_update!(nav::VectorNavigation, T)
    buffers = nav.buffers
    members = buffers.members
    candidates = buffers.candidates
    measured = buffers.measured_pseudoranges
    constraints = buffers.observability.hub_offset_constraints
    use_rates = nav.config.use_pseudorange_rates
    num_sats = length(candidates)
    resize!(buffers.candidate_positions, num_sats)
    for (k, j) in enumerate(candidates)
        buffers.candidate_positions[k] = members[j].sat_position
    end
    member_bias_columns!(buffers.candidate_columns, members, candidates)
    num_rate_rows = use_rates ? num_sats : 0
    num_measurements = num_sats + num_rate_rows + length(constraints)
    update = _measurement_buffers!(buffers, length(nav.x), num_measurements)
    z = update.z
    R = fill!(update.R, 0.0)
    for (k, j) in enumerate(candidates)
        member = members[j]
        z[k] = measured[j] + member.code_discriminator * member.chip_length
        R[k, k] = pseudorange_noise_variance(member, T)
        if use_rates
            z[num_sats+k] = member.pseudorange_rate
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
        nav.model.idxs,
        buffers.ξ_model,
        buffers.candidate_positions,
        members,
        candidates,
        buffers.candidate_columns,
        use_rates,
        constraints,
    )
    measurement_update!(update.intermediate, nav.x, nav.P, z, h!, R)
    nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Loop closure

# The predictions of every member at the state `nav.x` — the updated one — and their
# post-fit residuals.
function _predict_members!(nav::VectorNavigation)
    buffers = nav.buffers
    members = buffers.members
    num_members = length(members)
    idxs = nav.model.idxs
    position_and_bias_vector!(buffers.ξ, nav.x, idxs)
    predicted = resize!(buffers.predicted_pseudoranges, num_members)
    calc_ρ_hat!(predicted, buffers.positions, buffers.ξ, buffers.member_columns)
    rates = resize!(buffers.predicted_pseudorange_rates, num_members)
    residuals = resize!(buffers.residuals, num_members)
    rate_residuals = resize!(buffers.rate_residuals, num_members)
    for (j, member) in enumerate(members)
        rates[j] = predict_pseudorange_rate(nav.x, idxs, member)
        residuals[j], rate_residuals[j] = post_fit_residuals(
            member,
            buffers.measured_pseudoranges[j],
            predicted[j],
            rates[j],
        )
    end
    nothing
end

# Assign every member above the horizon this cycle's commands: the satellite evaluates
# them where its own command lands, on its next record (`_commands_at_landing`),
# against this cycle's posterior and member buffers.
function _assign_commands!(acc, group, g, buffer, nav)
    buffers = nav.buffers
    for (j, member) in enumerate(buffers.members)
        member.group == g && buffers.above_horizon[j] || continue
        slot = group.slots[member.slot_index]
        slot.command_cycle = nav.cycle_id
        slot.member_index = j
    end
    acc
end

# The NCO commands of member `j` (of slot `slot`), `τ` after the epoch: where its command
# lands, and one cycle later for the carrier ramp's end. Evaluated at the state `nav.x`
# propagated by `τ` and against the range the replica will realise there — the transmit
# time from the code phase there (the epoch's corrected transmit time moved on by the
# `chips` the replica advances from the epoch; the satellite clock correction changes
# by picoseconds in `τ`), the satellite at that time, the receive time
# `reference_tow + τ`, the epoch's atmospheric delay (it moves by millimetres in `τ`).
# The code correction removes the range error (the TOW-based, atmosphere-corrected
# range, no discriminator term); the carrier is the predicted Doppler itself. Returns
# both in Hz.
function _member_commands(nav::VectorNavigation, slot::VTSlot, j, τ, chips)
    member = nav.buffers.members[j]
    predicted_pseudorange, predicted_rate, measured_pseudorange =
        _predict_after_epoch(nav, slot, member, j, ustrip(s, nav.reference_time), τ, chips)
    code_update = nco_code_correction(
        predicted_pseudorange,
        measured_pseudorange,
        member.code_frequency,
        ustrip(s, nav.model.cycle_time),
    )
    code_update, predicted_carrier_doppler(predicted_rate, member.wavelength)
end

# The predicted pseudorange and rate of member `j` `τ` after the epoch and `chips` on
# from the epoch's code phase, at the state propagated there, and the pseudorange its
# replica realises there.
function _predict_after_epoch(nav::VectorNavigation, slot::VTSlot, member::VTMember, j, reference_tow, τ, chips)
    buffers = nav.buffers
    idxs = nav.model.idxs
    landing_time = member.time + chips / member.code_frequency
    landing_orbit = calc_satellite_position_and_velocity(slot.decoder, landing_time)
    landing_position = get_sat_position(landing_orbit)
    landing_velocity = get_sat_velocity(landing_orbit)
    propagate_state!(buffers.x_landing, nav.x, idxs, τ)
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
    # Doppler-signed, as `predict_pseudorange_rate`.
    predicted_rate = -calc_pseudorange_rate(
        user_pos,
        user_vel,
        user_clock_drift,
        landing_position,
        landing_velocity,
        calc_satellite_clock_drift(slot.decoder, landing_time),
    )
    measured_pseudorange =
        pseudorange_from_tows(
            reference_tow + τ,
            landing_time - (member.time - member.time_gpst_count),
        ) - buffers.delays[j]
    predicted_pseudorange, predicted_rate, measured_pseudorange
end

# Release the members of this group that dropped below the horizon.
# Returns whether any was released.
function _release_below_horizon!(released, group, g, buffer, nav)
    buffers = nav.buffers
    for (j, member) in enumerate(buffers.members)
        member.group == g && !buffers.above_horizon[j] || continue
        _mark_release!(group.slots[member.slot_index], VT_BELOW_HORIZON)
        released = true
    end
    released
end

# Release every satellite of this group still in the loop, for `reason`: the members,
# and those that sat the cycle out, having missed its epoch. Returns whether any was
# released.
function _release_all!(released, group, g, buffer, reason::VTReleaseReason)
    for slot in group.slots
        slot.estimator_state.in_vector_loop || continue
        _mark_release!(slot, reason)
        released = true
    end
    released
end

function _count_members(n, group, g, buffer)
    for slot in group.slots
        _is_member(slot) && (n += 1)
    end
    n
end

# ─────────────────────────────────────────────────────────────────────────────
# Seeding from a scalar fix

function _add_fix_satellites!(num_added, group, g, buffer, nav)
    signal_id = nav.layout.signal_id_by_group[g]
    sats = nav.pvt.sats
    for slot in group.slots
        slot.in_cycle && haskey(sats, (signal_id, slot.prn)) || continue
        slot.estimator_state = _add_to_vector_loop(slot.estimator_state)
        num_added += 1
    end
    num_added
end

# Scale the seeded covariance by the geometry of the fix (`seed_fix_covariance!`): the
# design matrix of the members the fix solved with, at the seeded position. The
# candidates are rebuilt by every cycle, so they serve as scratch here.
function _seed_fix_covariance!(nav::VectorNavigation)
    buffers = nav.buffers
    members = buffers.members
    layout = nav.layout
    idxs = nav.model.idxs
    included = empty!(buffers.candidates)
    for j in eachindex(members)
        haskey(nav.pvt.sats, _member_key(layout, members[j])) && push!(included, j)
    end
    isempty(included) && return false
    resize!(buffers.candidate_positions, length(included))
    for (k, j) in enumerate(included)
        buffers.candidate_positions[k] = members[j].sat_position
    end
    columns, _ = dense_bias_columns!(
        buffers.dense_clock_indices,
        buffers.dense_ifb_indices,
        buffers.clock_columns,
        buffers.ifb_columns,
        members,
        included,
        nav.primary_clock_index,
    )
    num_columns = 3 + columns.num_clock_biases + columns.num_ifb
    length(included) < num_columns && return false
    if size(buffers.design_matrix, 1) < length(included)
        buffers.design_matrix = zeros(2 * length(included), size(buffers.design_matrix, 2))
    end
    H = view(buffers.design_matrix, 1:length(included), 1:num_columns)
    position_and_bias_vector!(buffers.ξ, nav.x, idxs)
    calc_H!(H, buffers.candidate_positions, buffers.ξ, columns)
    seed_fix_covariance!(
        nav.P,
        idxs,
        layout,
        nav.pvt,
        nav.primary_clock_index,
        H,
        buffers.normal_matrices[num_columns],
        buffers.clock_columns,
        buffers.ifb_columns,
    )
end

# Switch from scalar to vector tracking off the fresh scalar fix in `nav.pvt`: promote the
# fix's satellites into the vector loop, seed the navigation filter from the fix, and close
# the loops a first time so the NCOs already steer toward the navigation solution — with
# no measurement update and no accumulator reset. A fresh fix always has satellites, and
# every one of them is a slot in the cycle (only those enter the scalar solve), so there is
# always a member to seed from. No satellite is in the loop then: registration puts a
# satellite out of it, and the fallback releases every satellite in it.
function _seed!(nav::VectorNavigation, groups, cycle_time)
    buffers = nav.buffers
    model = nav.model
    ensure_nav_filter_cycle_time!(model, nav.config, cycle_time)
    T = ustrip(s, model.cycle_time)
    pvt = nav.pvt
    nav.primary_clock_index = initial_nav_state!(nav.x, nav.P, nav.layout, model.idxs, pvt)
    _fold_groups_with_buffers(_add_fix_satellites!, 0, groups, buffers.sat_states, 1, nav)
    ionospheric_correction = _gather_members!(nav, groups, T)
    _seed_fix_covariance!(nav)
    members = buffers.members
    # The pseudorange reference epoch is the latest transmit time corrected by the fix's
    # receiver clock bias, matching how `calc_pvt` timestamps the fix.
    latest = -Inf
    for member in members
        latest = max(latest, member.time_gpst_count)
    end
    reference_tow = latest - ustrip(m, pvt.time_correction) / SPEED_OF_LIGHT
    _measure_pseudoranges!(nav, ionospheric_correction, reference_tow)
    # First loop closure: the filter is seeded exactly at the fix, so the commands
    # are the prediction residuals at the seeded state.
    _predict_members!(nav)
    resize!(buffers.above_horizon, length(members))
    fill!(buffers.above_horizon, true)
    _fold_groups_with_buffers(_assign_commands!, 0, groups, buffers.sat_states, 1, nav)
    nav.running = true
    nav.reference_time = reference_tow * s
    nav.time_with_insufficient_meas = 0.0s
    # The epoch offset is re-read with the reference epoch it anchors: one cached by an
    # earlier run would be a week stale if the scalar solve was in control across a
    # week rollover, which no running cycle saw.
    nav.time_epoch_offset = _primary_time_epoch_offset(nav, eachindex(members))
    nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# One cycle

# One navigation-filter cycle: predict, fuse the accumulated discriminators as
# pseudorange (and, for VDFLL, pseudorange-rate) measurements, assign every member fresh
# NCO commands, and manage the membership (admission, availability,
# release, fallback to scalar tracking). Returns `(released, fell_back)`.
function _run_cycle!(nav::VectorNavigation, groups, cycle_time)
    config = nav.config
    buffers = nav.buffers
    model = nav.model
    idxs = model.idxs
    T = ustrip(s, cycle_time)
    ensure_nav_filter_cycle_time!(model, config, cycle_time)
    # Advance the receive time-of-week, wrapping at the 604800 s week boundary so it
    # stays a valid seconds-of-week for the pseudorange differencing, the atmospheric
    # time-of-week and the solution's week/second split. The wrap is remembered: the
    # week it drops has to be added to the cached epoch offset
    # (`rolled_over_time_epoch_offset`), which is the only other place the run's
    # absolute epoch is held.
    advanced_reference_time = nav.reference_time + cycle_time
    week_rollover = advanced_reference_time >= SECONDS_PER_WEEK * s
    reference_time = mod(advanced_reference_time, SECONDS_PER_WEEK * s)
    reference_tow = ustrip(s, reference_time)

    # Admission is judged at the solution the filter starts the cycle from.
    enu_from_ecef = ENUfromECEF(_ecef(first(nav_filter_states(nav.x, idxs))), wgs84)
    released =
        _fold_groups_with_buffers(_update_membership!, false, groups, buffers.sat_states, 1, enu_from_ecef)
    ionospheric_correction = _gather_members!(nav, groups, T)
    members = buffers.members
    num_members = length(members)

    # Navigation filter prediction, then the measurements at the predicted state.
    time_update!(buffers.time_update, nav.x, nav.P, model.F, model.Q)
    _measure_pseudoranges!(nav, ionospheric_correction, reference_tow)

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
            assess_bias_observability!(buffers.observability, nav.layout, members, candidates)
        _measurement_update!(nav, T)
    end
    num_candidates = length(candidates)

    # The measurement model at the *updated* state, for every member: the post-fit
    # residuals, and — unless the loop falls back — the commands.
    _predict_members!(nav)

    # Starvation watchdog: grow the timer whenever the epoch could not determine the
    # navigation state — too few measurements, or too few distinct satellites among them,
    # for the bias layout in force — and pay it back down (at half rate) otherwise. How
    # certain the filter is of the solution it did produce is not policed here; it is
    # reported as `VTStatus`'s `position_std`.
    time_with_insufficient_meas =
        is_epoch_solvable(observability, num_candidates) ?
        max(0.0s, nav.time_with_insufficient_meas - cycle_time / 2) :
        nav.time_with_insufficient_meas + cycle_time

    # Elevation mask: release members that dropped below the horizon, at the updated
    # position — before the corrections, so every remaining member gets one.
    above_horizon = resize!(buffers.above_horizon, num_members)
    fill!(above_horizon, true)
    user_pos, _, _ = nav_filter_states(nav.x, idxs)
    enu_from_ecef = ENUfromECEF(_ecef(user_pos), wgs84)
    num_above_horizon = 0
    for (j, member) in enumerate(members)
        above_horizon[j] = get_sat_enu(enu_from_ecef, _ecef(member.sat_position)).ϕ >= 0
        above_horizon[j] && (num_above_horizon += 1)
    end
    released = _fold_groups_with_buffers(_release_below_horizon!, released, groups, buffers.sat_states, 1, nav)

    # Fall back to scalar tracking when the filter has coasted too long or no members
    # remain; otherwise assign every member fresh commands toward the updated solution.
    fell_back = time_with_insufficient_meas > config.insufficient_meas_timeout || num_above_horizon == 0
    if fell_back
        released = _fold_groups_with_buffers(_release_all!, released, groups, buffers.sat_states, 1, VT_FALLBACK)
        nav.running = false
    else
        _fold_groups_with_buffers(_assign_commands!, 0, groups, buffers.sat_states, 1, nav)
    end

    nav.primary_clock_index =
        select_primary_clock_index(nav.layout, members, candidates, nav.primary_clock_index)
    nav.time_epoch_offset = rolled_over_time_epoch_offset(
        _resolve_time_epoch_offset(nav),
        nav.time_epoch_offset,
        week_rollover,
    )
    nav.reference_time = reference_time
    nav.time_with_insufficient_meas = time_with_insufficient_meas
    _write_solution!(nav, groups)
    released, fell_back
end

_ecef(v) = ECEF(v[1], v[2], v[3])

# The epoch offset from a measured member of the primary system, or the cached one.
_resolve_time_epoch_offset(nav::VectorNavigation) =
    isnothing(nav.time_epoch_offset) ?
    _primary_time_epoch_offset(nav, nav.buffers.candidates) : nav.time_epoch_offset

# The epoch offset read off the first of the members at `indices` of the primary system,
# `nothing` if there is none.
function _primary_time_epoch_offset(nav::VectorNavigation, indices)
    members = nav.buffers.members
    for j in indices
        members[j].clock_bias_index == nav.primary_clock_index &&
            return time_epoch_offset(nav.buffers.rows[j])
    end
    nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# The solution

function _write_solution!(nav::VectorNavigation, groups)
    buffers = nav.buffers
    members = buffers.members
    layout = nav.layout
    idxs = nav.model.idxs
    x = nav.x
    user_pos, user_vel, user_clock_drift = nav_filter_states(x, idxs)
    position = _ecef(user_pos)
    velocity = _ecef(user_vel)
    primary_clock_bias = x[idxs.clock_biases[nav.primary_clock_index]]
    candidates = buffers.candidates

    # Geometry of the satellites this update measured, or `nothing` when there is none to
    # report: no measurements at all, or a rank-deficient design for which `calc_DOP!`
    # returns its all-`-1` sentinel. That sentinel stays inside: `calc_pvt` never emits
    # one either (it rejects the epoch instead), so a solution's `dop` is always either a
    # real geometry or absent.
    dop = nothing
    if !isempty(candidates)
        columns, primary_column = dense_bias_columns!(
            buffers.dense_clock_indices,
            buffers.dense_ifb_indices,
            buffers.clock_columns,
            buffers.ifb_columns,
            members,
            candidates,
            nav.primary_clock_index,
        )
        num_columns = 3 + columns.num_clock_biases + columns.num_ifb
        if size(buffers.design_matrix, 1) < length(candidates)
            buffers.design_matrix = zeros(2 * length(candidates), size(buffers.design_matrix, 2))
        end
        H = view(buffers.design_matrix, 1:length(candidates), 1:num_columns)
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
    solution = empty_keeping_capacity!(nav.pvt)
    sats = solution.sats
    for j in candidates
        set!(sats, _member_key(layout, members[j]), _member_info(buffers, j))
    end
    member_sats = empty_keeping_capacity!(nav.member_sats)
    for j in eachindex(members)
        set!(member_sats, _member_key(layout, members[j]), _member_info(buffers, j))
    end

    # Only the biases this update measured, as `calc_pvt` reports only the ones it
    # estimated: a time system or band without a measurement coasts on its process noise
    # (or was never seeded at all), and a consumer could not tell its value from a
    # measured one.
    inter_system_biases = solution.inter_system_biases
    for (index, time_system) in enumerate(layout.time_systems)
        index == nav.primary_clock_index && continue
        _is_measured(member -> member.clock_bias_index == index, members, candidates) ||
            continue
        inter_system_biases[time_system] = (x[idxs.clock_biases[index]] - primary_clock_bias) * m
    end
    inter_frequency_biases = solution.inter_frequency_biases
    for (index, band) in enumerate(layout.extra_bands)
        _is_measured(member -> member.ifb_index == index, members, candidates) || continue
        inter_frequency_biases[band] =
            InterFrequencyBias(x[idxs.ifb[index]] * m, layout.reference_bands[index])
    end

    nav.pvt = PVTSolution(
        position,
        velocity,
        calc_course_over_ground(position, velocity),
        primary_clock_bias * m,
        solution_time(nav.time_epoch_offset, nav.reference_time, primary_clock_bias),
        user_clock_drift / SPEED_OF_LIGHT,
        dop,
        sats,
        layout.time_systems[nav.primary_clock_index],
        inter_system_biases,
        inter_frequency_biases,
    )
    nothing
end

# Whether one of the members at `candidates` satisfies `predicate`.
function _is_measured(predicate::F, members, candidates) where {F}
    for j in candidates
        predicate(members[j]) && return true
    end
    false
end
_member_key(layout::NavFilterLayout, member::VTMember) =
    (layout.signal_id_by_group[member.group], member.prn)

_member_info(buffers::VTBuffers, j) = SatInfo(
    buffers.members[j].sat_position,
    buffers.members[j].time,
    buffers.residuals[j],
    buffers.rate_residuals[j],
)

