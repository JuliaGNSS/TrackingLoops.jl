# ─────────────────────────────────────────────────────────────────────────────
# Vector tracking: the navigation processor that closes all loops
#
# One `NavigationProcessor` is shared by every satellite a `VectorTrackingLoop` steps. Each
# satellite's slot is filled by its integrations: the decoder from the soft bits of its
# decoding signal and, at every navigation epoch, a snapshot of the driver's replica,
# C/N₀ and accumulated discriminators, with the decoding signal's bit count. Once
# every satellite has completed the epoch's snapshot the cycle runs: a scalar PVT solve
# until its first fix seeds the filter, one filter iteration after. The cycle leaves its
# decisions (admission, release, commands) in the slots for each satellite to take up
# on its next integration. Nothing here knows a correlator, so software and hardware loops
# step the same loop.
# ─────────────────────────────────────────────────────────────────────────────

"""
    VectorTrackingReleaseReason

Why the latest navigation cycle handed a satellite back to its scalar loop
(a [`get_satellite_report`](@ref)'s `release_reason`):

  - `VT_NOT_RELEASED`: it was not;
  - `VT_INELIGIBLE`: no longer tracked (no integration for two navigation cycles),
    decoded for positioning, or healthy;
  - `VT_BELOW_HORIZON`: below the horizon at the solution the cycle starts
    from, so not measured in that cycle; re-admitted only `ADMISSION_ELEVATION`
    (1°) above it (hysteresis);
  - `VT_FALLBACK`: vector tracking stopped (starvation timeout, or no member
    left).

A receiver that forces released satellites out of lock does so for the first
two, not for a fallback.
"""
@enum VectorTrackingReleaseReason VT_NOT_RELEASED VT_INELIGIBLE VT_BELOW_HORIZON VT_FALLBACK

"""
    SatelliteReport

What a [`VectorTrackingLoop`](@ref) knows of one member, a satellite on one ranging
signal (see
[`get_satellite_report`](@ref)), so a consumer need not decode or estimate C/N₀ again:

  - `prn`, and `tracked`: whether it is stepped now (`false` after two cycles
    without an integration; the rest then describes it as it was);
  - `bit_or_secondary_code_synced`: whether the decoding signal's bit clock held the bit sync at
    that signal's last integration;
  - at its latest snapshot (`snapshot_time`, on the navigation-epoch grid, a multiple of
    `navigation_filter_update_interval`, `nothing` before
    the first): `cn0_dbhz` (the driver's, from the host's C/N₀ estimator),
    `measurement_ready` (decoding signal synced and the driver's C/N₀ at least the lock
    threshold) and `positioning_ready` (in lock, decoded for positioning and healthy);
  - `vector_controlled` and `release_reason`: the latest cycle's decision.

The report is an immutable snapshot whose values remain valid after later integrations
and report calls. Access the separately borrowed decoder with [`get_decoder`](@ref).
"""
struct SatelliteReport
    prn::Int
    tracked::Bool
    bit_or_secondary_code_synced::Bool
    snapshot_time::Union{Nothing,typeof(1.0s)}
    cn0_dbhz::Float64
    measurement_ready::Bool
    positioning_ready::Bool
    vector_controlled::Bool
    release_reason::VectorTrackingReleaseReason
end

# A satellite's driver role in the navigation processor: where the driver signal's last
# integration ended, and what the driver's half of a snapshot reads off it. A role, not a
# signal: a signal that ranges and carries the data itself (GPS L1 C/A alone) plays both
# roles and fills this part and `DecodingState`.
mutable struct DriverState
    last_end_sample_index::Int
    # Hz, of the signal's band: the end is at `last_end_sample_index / sampling_frequency`
    # on the integrations' time grid (`_last_end_time`).
    sampling_frequency::Float64
    last_code_phase::Float64 # chips, as the integration reported it; `NaN` if it did not
    last_coherent_integration_time::typeof(1.0s)
    last_early_late_spacing::Float64 # chips
    # Replica chips from the epoch of the latest snapshot to the last integration end.
    chips_since_epoch::Float64
end

# A satellite's decoding role: the decoder every integration of the decoding signal
# advances, where its last integration ended and its bit clock there (see `DriverState`
# for the roles).
mutable struct DecodingState{D}
    decoder::D
    last_end_sample_index::Int
    sampling_frequency::Float64 # Hz, of the signal's band (see `DriverState`)
    bit_synced::Bool
    num_code_periods_into_bit::Int
    code_phase_fraction::Float64 # chips past the nearest code-period boundary
    # The `command_issue_sample_index` of the batch the bit sync was found in
    # (`typemin(Int)` for none): its later integrations were correlated before the sync.
    sync_command_issue_sample_index::Int
end

# The data bit in the making while the filter steers the carrier: the sum of the
# decoding signal's prompts rotated onto the incoming carrier by the residual PLL, whether
# it covers the bit from its edge on, the soft bit it ends in, and the residual PLL's
# phases at the driver's latest prompts it rotates a pair's decoding signal by
# (`PromptHistory`).
mutable struct BitAccumulator
    prompt_sum::ComplexF64
    from_edge::Bool
    const soft_bit::Vector{Float32}
    const prompt_history::PromptHistory
end

# A satellite at a navigation epoch. A slot builds the pending epoch's from its two
# signals' halves (`pending_snapshot`) and, once both are in, keeps it as the snapshot
# the cycle reads (`snapshot`). The cycle does not write it: its decisions are the slot's.
struct SatelliteSnapshot{D,E}
    # The driver's half: the state at the epoch with the discriminators accumulated up to
    # it, the replica's code phase there
    # (chips; `NaN` if not reported), the Dopplers commanded there, the driver's C/N₀,
    # coherent integration time and early-late spacing.
    loop_state::E
    driver_code_phase::Float64
    carrier_doppler::typeof(1.0Hz)
    code_doppler::typeof(1.0Hz)
    cn0_dbhz::Float64
    coherent_integration_time::typeof(1.0s)
    early_late_spacing::Float64
    # The decoding signal's half: the decoder at its last integration end before the
    # epoch, whether its bit clock held the sync there, and the code phase at the epoch
    # from the decoder's last bit edge (chips). In `snapshot`, the decoder and code phase
    # are moved onto the bit the driver's replica is in (`_complete_snapshot!`).
    decoder::D
    bit_synced::Bool
    code_phase::Float64
    # Set when the snapshot completes.
    measurement_ready::Bool
    positioning_ready::Bool
end

SatelliteSnapshot(
    snapshot::SatelliteSnapshot;
    loop_state = snapshot.loop_state,
    driver_code_phase = snapshot.driver_code_phase,
    carrier_doppler = snapshot.carrier_doppler,
    code_doppler = snapshot.code_doppler,
    cn0_dbhz = snapshot.cn0_dbhz,
    coherent_integration_time = snapshot.coherent_integration_time,
    early_late_spacing = snapshot.early_late_spacing,
    decoder = snapshot.decoder,
    bit_synced = snapshot.bit_synced,
    code_phase = snapshot.code_phase,
    measurement_ready = snapshot.measurement_ready,
    positioning_ready = snapshot.positioning_ready,
) = typeof(snapshot)(
    loop_state,
    driver_code_phase,
    carrier_doppler,
    code_doppler,
    cn0_dbhz,
    coherent_integration_time,
    early_late_spacing,
    decoder,
    bit_synced,
    code_phase,
    measurement_ready,
    positioning_ready,
)

# One satellite of the navigation processor. A slot is never deleted: a dropped satellite
# leaves it, with its storage, for the next one to reuse.
mutable struct VectorTrackingSlot{D,E<:VectorLoopState}
    prn::Int
    occupied::Bool
    # The slot_generation holding the slot; a state of an older one registers again.
    slot_generation::Int
    # Whether the slot snapshotted the running cycle's epoch.
    in_cycle::Bool
    const driver::DriverState
    const decoding::DecodingState{D}
    const bit_accumulator::BitAccumulator
    # The first epoch the slot can snapshot, the epochs of the driver's and decoding
    # signal's halves of the pending snapshot, and that of its latest complete snapshot.
    first_epoch_index::Int
    driver_epoch_index::Int
    decoding_epoch_index::Int
    snapshot_epoch_index::Int
    pending_snapshot::SatelliteSnapshot{D,E}
    snapshot::SatelliteSnapshot{D,E}
    # What the latest cycle decided, for the satellite to take up: whether it is in the
    # vector loop (from its snapshot's state until a cycle admits or releases it), why it
    # was released, and the cycle and member index of its commands.
    in_vector_loop::Bool
    release_reason::VectorTrackingReleaseReason
    command_cycle_index::Int
    member_index::Int
end

function VectorTrackingSlot(
    decoding_signal::AbstractGNSSSignal,
    prn::Integer,
    loop_state::VectorLoopState,
    prompt_history_length::Int,
)
    decoder = GNSSDecoderState(decoding_signal, prn)
    snapshot = SatelliteSnapshot(loop_state, NaN, 0.0Hz, 0.0Hz, NaN, 0.001s, 0.5, decoder, false, 0.0, false, false)
    VectorTrackingSlot(
        Int(prn),
        false,
        0,
        false,
        DriverState(0, 1.0, NaN, 0.001s, 0.5, 0.0),
        DecodingState(decoder, 0, 1.0, false, 0, NaN, typemin(Int)),
        BitAccumulator(zero(ComplexF64), false, zeros(Float32, 1), PromptHistory(prompt_history_length)),
        typemax(Int),
        typemin(Int),
        typemin(Int),
        typemin(Int),
        snapshot,
        snapshot,
        loop_state.vector_controlled,
        VT_NOT_RELEASED,
        -1,
        0,
    )
end

# The slots of one ranging signal: the driver, the signal whose navigation data they
# decode (the driver itself for a plain signal), the `prototype` state a new slot starts
# from and the length of a slot's prompt history (`_prompt_history_length`).
struct VectorTrackingSlotGroup{S<:AbstractGNSSSignal,DS<:AbstractGNSSSignal,V<:VectorTrackingSlot,E<:VectorLoopState}
    driver_signal::S
    decoding_signal::DS
    slots::Vector{V}
    prototype::E
    prompt_history_length::Int
end

function VectorTrackingSlotGroup(
    driver_signal::AbstractGNSSSignal,
    decoding_signal::AbstractGNSSSignal,
    prototype::VectorLoopState,
    capacity,
    navigation_filter_update_interval,
)
    prompt_history_length = _prompt_history_length(driver_signal, decoding_signal, navigation_filter_update_interval)
    # Inactive slots need a valid decoder prototype; the receiver validates tracked PRNs.
    prototype_prn = first(get_supported_prns(decoding_signal))
    slots = [VectorTrackingSlot(decoding_signal, prototype_prn, prototype, prompt_history_length) for _ = 1:capacity]
    VectorTrackingSlotGroup(driver_signal, decoding_signal, sizehint!(slots, 2 * capacity), prototype, prompt_history_length)
end

"""
    VectorTrackingStatus

What the latest navigation cycle of a [`VectorTrackingLoop`](@ref) did (see
[`get_navigation_status`](@ref)).

  - `vector_tracking_active`: vector tracking is active after this cycle;
  - `position_std`, `clock_bias_std`: the filter's 1σ 3-D position and reference-clock
    uncertainties (`NaN` unless the filter ran or fell back this cycle); a
    degenerate geometry shows here, not in the starvation timer;
  - `measurement_deficit_duration`: how long the filter has coasted on epochs it
    could not solve, paid back at half rate otherwise, and checked against
    `measurement_deficit_timeout`; nonzero means the loop is on its way out;
  - `num_vector_tracking_members`: members of the vector loop after this cycle, one
    per satellite and ranging signal (see [`get_vector_tracking_members`](@ref));
  - the events `seeded` (a fresh scalar fix seeded the filter), `fell_back`
    (vector tracking stopped) and `released` (some satellite was released; see
    the reports' `release_reason`);
  - `solution_from_filter`: [`get_navigation_solution`](@ref) is the navigation filter's
    this cycle, not a scalar fix. It differs from `vector_tracking_active` on the
    transitions: the cycle that seeds the filter reports the scalar fix it seeded
    from, and the cycle that falls back still reports the filter's last update.

The per-member report of the latest cycle is [`get_vector_tracking_members`](@ref).
"""
struct VectorTrackingStatus
    vector_tracking_active::Bool
    position_std::typeof(1.0m)
    clock_bias_std::typeof(1.0m)
    measurement_deficit_duration::typeof(1.0s)
    num_vector_tracking_members::Int
    seeded::Bool
    fell_back::Bool
    released::Bool
    solution_from_filter::Bool
end

VectorTrackingStatus() = VectorTrackingStatus(false, NaN * m, NaN * m, 0.0s, 0, false, false, false, false)

# The unscented update's buffers (intermediate, `z`, `R`) for one measurement count.
struct MeasurementUpdateWorkspace{I}
    intermediate::I
    z::Vector{Float64}
    R::Matrix{Float64}
end

const _UKFIntermediate = typeof(UKFMUIntermediate(Float64, 1, 1))

MeasurementUpdateWorkspace(num_states, num_measurements) = MeasurementUpdateWorkspace{_UKFIntermediate}(
    UKFMUIntermediate(Float64, num_states, num_measurements),
    zeros(num_measurements),
    zeros(num_measurements, num_measurements),
)

# A cycle's scratch, sized for the preallocated slots; vectors grow past that with
# `push!` / `resize!`, the design matrix by replacement.
mutable struct VectorTrackingWorkspace{SB<:Tuple}
    # Per group, the `SignalObservation`s whose measurements are collected.
    signal_observations::SB
    measurements::Vector{PreparedObservation}
    members::Vector{VectorTrackingObservation}
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
    ξ_application::Vector{Float64}
    x_application::Vector{Float64}
    application_position::Vector{SVector{3,Float64}}
    application_columns::BiasColumns
    application_range::Vector{Float64}
    # The cycle's bias-layout decision, and its constraints over the filter's clocks.
    bias_layout::BiasLayout
    clock_constraints::Vector{Tuple{Int,Int,Float64}}
    time_update::KFTUIntermediate{Float64}
    measurement_updates::Vector{Union{Nothing,MeasurementUpdateWorkspace{_UKFIntermediate}}}
    design_matrix::Matrix{Float64}
    normal_matrices::Vector{Matrix{Float64}}
end

_capacity_vector(::Type{T}, n) where {T} = sizehint!(T[], n)

function VectorTrackingWorkspace(signal_observations::Tuple, num_states, layout::NavFilterLayout, max_members)
    num_clocks = num_clock_biases(layout)
    num_ifb_states = num_ifbs(layout)
    num_lsq = 3 + num_clocks + num_ifb_states
    max_measurements = 2 * max_members + num_clocks
    VectorTrackingWorkspace(
        signal_observations,
        _capacity_vector(PreparedObservation, max_members),
        _capacity_vector(VectorTrackingObservation, max_members),
        _capacity_vector(Bool, max_members),
        _capacity_vector(Int, max_members),
        _capacity_vector(Float64, max_members),
        _capacity_vector(Float64, max_members),
        _capacity_vector(Float64, max_members),
        _capacity_vector(Float64, max_members),
        _capacity_vector(typeof(1.0m), max_members),
        _capacity_vector(typeof(1.0m / s), max_members),
        _capacity_vector(SVector{3,Float64}, max_members),
        BiasColumns(_capacity_vector(Int, max_members), num_clocks, _capacity_vector(Int, max_members), num_ifb_states),
        _capacity_vector(SVector{3,Float64}, max_members),
        BiasColumns(_capacity_vector(Int, max_members), num_clocks, _capacity_vector(Int, max_members), num_ifb_states),
        zeros(num_lsq),
        zeros(num_lsq),
        zeros(num_lsq),
        zeros(num_states),
        [zero(SVector{3,Float64})],
        BiasColumns([1], num_clocks, [0], num_ifb_states),
        [0.0],
        BiasLayout(),
        _capacity_vector(Tuple{Int,Int,Float64}, num_clocks),
        KFTUIntermediate(Float64, num_states),
        # One per possible measurement count, so a membership change never builds one.
        Union{Nothing,MeasurementUpdateWorkspace{_UKFIntermediate}}[
            MeasurementUpdateWorkspace(num_states, k) for k = 1:max_measurements
        ],
        zeros(max(max_members, 1), num_lsq),
        [zeros(n, n) for n = 1:num_lsq],
    )
end

"""
    NavFilter

The navigation filter, a Kalman filter with a linear time update and an unscented
measurement update: its configuration, the state layout and process model, the state
`x` and its covariance `P`, the clock-bias state the solution is reported against and
the receive epoch the pseudoranges are referenced to.
"""
mutable struct NavFilter
    const config::NavigationFilterConfig
    const layout::NavFilterLayout
    const model::NavFilterModel
    const x::Vector{Float64}
    const P::Matrix{Float64}
    # The clock-bias state the solution is reported against.
    reference_clock_index::Int
    # The receive epoch the pseudoranges are referenced to, uncorrected for the receiver
    # clock: the fix's epoch at the seed, advanced by the cycle time every cycle. They
    # difference against its time of week on the reference system's count (see
    # `_measure_pseudoranges!`), so it does not move when the reference clock changes.
    reference_epoch::TAITime
end

"""
    NavigationProcessor

The navigation processor of a [`VectorTrackingLoop`](@ref), shared by every satellite
it steps: the navigation filter (`nav_filter`), the slots, the scalar PVT it falls back
to, and every buffer a cycle needs.

It holds the latest solution (`pvt`, whose containers every cycle reuses) and
the member results (`vector_tracking_members`, read with
[`get_vector_tracking_members`](@ref)). `pvt.observations` carries only
the members the update measured, so the keys missing from it are the members the
filter coasted through an obscuration.
"""
mutable struct NavigationProcessor{G<:Tuple,SB<:Tuple}
    const nav_filter::NavFilter
    const enabled::Bool
    vector_tracking_active::Bool
    measurement_deficit_duration::typeof(1.0s)
    const vector_tracking_members::Dictionary{Tuple{Symbol,Int},ObservationResiduals}
    pvt::PVTSolution
    status::VectorTrackingStatus
    const workspace::PVTWorkspace
    const approximate_year::Int
    const enable_ionospheric_correction::Bool
    const enable_tropospheric_correction::Bool
    const buffers::VectorTrackingWorkspace{SB}
    const groups::G
    const navigation_filter_update_interval::typeof(1.0s)
    const min_tracking_cn0_dbhz::Float64
    # The epoch (in `navigation_filter_update_interval`s on the integrations' time grid) the slots snapshot
    # next, and how many have.
    pending_epoch_index::Int
    num_snapshots::Int
    # The latest cycle's id and epoch.
    cycle_index::Int
    cycle_epoch_index::Int
    next_slot_generation::Int
end

function NavigationProcessor(
    filter_config::Union{NavigationFilterConfig,Nothing},
    pairs::Tuple,
    scalar_loop::AbstractTrackingLoop;
    navigation_filter_update_interval,
    min_tracking_cn0_dbhz::Float64,
    satellite_capacity_per_signal::Int,
    approximate_year::Integer,
    enable_ionospheric_correction::Bool,
    enable_tropospheric_correction::Bool,
)
    config = something(filter_config, NavigationFilterConfig())
    signals = map(first, pairs)
    layout = NavFilterLayout(signals)
    model = NavFilterModel(config, layout, uconvert(s, navigation_filter_update_interval))
    n = num_nav_states(config, layout)
    groups = map(pairs) do (signal, decoding_signal)
        prototype = VectorLoopState(init_loop_state(scalar_loop, signal, 0.0Hz, 0.0Hz), false)
        VectorTrackingSlotGroup(signal, decoding_signal, prototype, satellite_capacity_per_signal, navigation_filter_update_interval)
    end
    max_members = satellite_capacity_per_signal * length(signals)
    signal_observations = map(_signal_observation_buffer, groups)
    NavigationProcessor(
        NavFilter(config, layout, model, zeros(n), zeros(n, n), 1, TAITime(0, 0.0)),
        !isnothing(filter_config),
        false,
        0.0s,
        Dictionary{Tuple{Symbol,Int},ObservationResiduals}(),
        PVTSolution(),
        VectorTrackingStatus(),
        PVTWorkspace(),
        Int(approximate_year),
        enable_ionospheric_correction,
        enable_tropospheric_correction,
        VectorTrackingWorkspace(signal_observations, n, layout, max_members),
        groups,
        uconvert(s, navigation_filter_update_interval),
        min_tracking_cn0_dbhz,
        typemin(Int),
        0,
        0,
        typemin(Int),
        0,
    )
end

_signal_observation_buffer(group::VectorTrackingSlotGroup{S,DS,<:VectorTrackingSlot{D}}) where {S,DS,D} =
    sizehint!(SignalObservation{Float64,D,S}[], 2 * max(length(group.slots), 1))

"""
    calc_position_std(tracking_loop::VectorTrackingLoop)

The navigation filter's 1σ 3-D position uncertainty (m), once seeded.
"""
calc_position_std(tracking_loop::VectorTrackingLoop) = calc_position_std(tracking_loop.navigation_processor)

"""
    calc_clock_bias_std(tracking_loop::VectorTrackingLoop)

The navigation filter's 1σ uncertainty (m) of the solution's reference clock
bias, once seeded.
"""
calc_clock_bias_std(tracking_loop::VectorTrackingLoop) = calc_clock_bias_std(tracking_loop.navigation_processor)

calc_position_std(nav_processor::NavigationProcessor) = calc_position_std(nav_processor.nav_filter.P, nav_processor.nav_filter.model.idxs)

function calc_clock_bias_std(nav_processor::NavigationProcessor)
    nav_filter = nav_processor.nav_filter
    index = nav_filter.model.idxs.clock_biases[nav_filter.reference_clock_index]
    sqrt(nav_filter.P[index, index])
end

"""
    get_navigation_solution(tracking_loop::VectorTrackingLoop) -> PVTSolution

The latest navigation solution: the scalar PVT's until the filter is seeded, the
filter's after. The next cycle reuses its containers; copy out what is needed.
"""
get_navigation_solution(tracking_loop::VectorTrackingLoop) = tracking_loop.navigation_processor.pvt

"""
    get_navigation_status(tracking_loop::VectorTrackingLoop) -> VectorTrackingStatus

What the latest navigation cycle did (see [`VectorTrackingStatus`](@ref)).
"""
get_navigation_status(tracking_loop::VectorTrackingLoop) = tracking_loop.navigation_processor.status

get_navigation_cycle_index(tracking_loop::VectorTrackingLoop) = tracking_loop.navigation_processor.cycle_index

function get_navigation_elapsed_time(tracking_loop::VectorTrackingLoop)
    nav_processor = tracking_loop.navigation_processor
    nav_processor.cycle_epoch_index == typemin(Int) ? nothing : nav_processor.cycle_epoch_index * nav_processor.navigation_filter_update_interval
end

function get_satellite_report(tracking_loop::VectorTrackingLoop, signal::AbstractGNSSSignal, prn::Integer)
    nav_processor = tracking_loop.navigation_processor
    _satellite_report(nav_processor, nav_processor.groups, signal, Int(prn))
end

_satellite_report(nav_processor, ::Tuple{}, signal, prn) = nothing
function _satellite_report(nav_processor, groups::Tuple, signal::S, prn) where {S}
    group = first(groups)
    group.driver_signal isa S || return _satellite_report(nav_processor, Base.tail(groups), signal, prn)
    for slot in group.slots
        slot.prn == prn && slot.slot_generation > 0 || continue
        snapshot_time = slot.snapshot_epoch_index == typemin(Int) ? nothing :
            slot.snapshot_epoch_index * nav_processor.navigation_filter_update_interval
        return SatelliteReport(
            slot.prn,
            slot.occupied,
            slot.decoding.bit_synced,
            snapshot_time,
            slot.snapshot.cn0_dbhz,
            slot.snapshot.measurement_ready,
            slot.snapshot.positioning_ready,
            slot.in_vector_loop,
            slot.release_reason,
        )
    end
    nothing
end

"""
    get_decoder(tracking_loop::VectorTrackingLoop, signal, prn)

Borrow the satellite's navigation decoder through its last decoding integration, or
`nothing` if the satellite has no slot. Treat the decoder as read-only: `decode!`
and `reset_decoder_state!` would change the navigation processor's state. Its shared buffers may
change on the next integration, and a reused slot gives them to another satellite.
Copy the decoder to retain its current contents; construct a fresh decoder for
independent decoding.
"""
get_decoder(tracking_loop::VectorTrackingLoop, signal::AbstractGNSSSignal, prn::Integer) =
    _get_decoder(tracking_loop.navigation_processor.groups, signal, Int(prn))

"""
    get_signal_pairs(tracking_loop::VectorTrackingLoop) -> Tuple

The signals `tracking_loop` was built with, one `driver_signal => decoding_signal` pair per
signal: a pilot ranged on and the data component decoded with it, or a data signal
paired with itself.
"""
get_signal_pairs(tracking_loop::VectorTrackingLoop) =
    map(group -> group.driver_signal => group.decoding_signal, tracking_loop.navigation_processor.groups)

"""
    get_scalar_loop(tracking_loop::VectorTrackingLoop) -> AbstractTrackingLoop

The scalar tracking loop each satellite runs outside vector control (the `scalar_loop`
`tracking_loop` was built with).
"""
get_scalar_loop(tracking_loop::VectorTrackingLoop) = tracking_loop.scalar_loop

"""
    is_vector_tracking_enabled(tracking_loop::AbstractTrackingLoop) -> Bool

Whether `tracking_loop` closes the satellites' loops through a navigation filter: `false` for a
scalar loop and for a `VectorTrackingLoop` built with `filter_config = nothing`, which
only solves a scalar PVT.
"""
is_vector_tracking_enabled(::AbstractTrackingLoop) = false
is_vector_tracking_enabled(tracking_loop::VectorTrackingLoop) = tracking_loop.navigation_processor.enabled

_get_decoder(::Tuple{}, signal, prn) = nothing
function _get_decoder(groups::Tuple, signal::S, prn) where {S}
    group = first(groups)
    group.driver_signal isa S || return _get_decoder(Base.tail(groups), signal, prn)
    for slot in group.slots
        slot.prn == prn && slot.slot_generation > 0 && return slot.decoding.decoder
    end
    nothing
end

"""
    get_vector_tracking_members(tracking_loop::VectorTrackingLoop)

The latest cycle's per-member results: every member of the vector loop, measured
*and* coasted, keyed `(driver_signal_id, prn)`, each with the satellite position,
transmit time and post-fit residuals. One physical satellite can have several
members, one per ranging signal. The solution's `observations` contains only the
measured members. Empty while the scalar solve is in control.

The returned dictionary is borrowed storage, reused by the next navigation cycle;
copy it to retain the current results.
"""
get_vector_tracking_members(tracking_loop::VectorTrackingLoop) = tracking_loop.navigation_processor.vector_tracking_members

# ─────────────────────────────────────────────────────────────────────────────
# Walking the groups

# Fold `f(acc, group, group_index, buffer, args...)` over the groups in order, `group_index` the group's
# index and `buffer` its entry of `buffers`. Recursing over the tuples keeps every
# call statically dispatched, however heterogeneous the groups.
@inline _fold_groups_with_buffers(f::F, acc, ::Tuple{}, ::Tuple{}, group_index, args...) where {F} = acc
@inline _fold_groups_with_buffers(f::F, acc, groups::Tuple, buffers::Tuple, group_index, args...) where {F} =
    _fold_groups_with_buffers(
        f,
        f(acc, first(groups), group_index, first(buffers), args...),
        Base.tail(groups),
        Base.tail(buffers),
        group_index + 1,
        args...,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Scalar observation collection

function _collect_pvt_ready!(acc, group, group_index, buffer)
    empty!(buffer)
    for slot in group.slots
        slot.in_cycle && slot.snapshot.positioning_ready || continue
        push!(buffer, _signal_observation(group.driver_signal, slot))
    end
    acc
end

@inline _signal_observation(signal, slot::VectorTrackingSlot) = SignalObservation(;
    decoder = slot.snapshot.decoder,
    signal = signal,
    code_phase = slot.snapshot.code_phase,
    carrier_doppler = slot.snapshot.carrier_doppler,
)

# ─────────────────────────────────────────────────────────────────────────────
# Membership

# Hand `slot` back to its scalar loop; taken up on its next integration (`_take_up_cycle`).
function _mark_release!(slot::VectorTrackingSlot, reason::VectorTrackingReleaseReason)
    slot.in_vector_loop = false
    slot.release_reason = reason
    nothing
end

# A member that missed the epoch sits the cycle out.
_is_member(slot::VectorTrackingSlot) = slot.in_cycle && slot.in_vector_loop

_is_eligible(slot::VectorTrackingSlot) =
    slot.in_cycle &&
    has_positioning_data(slot.snapshot.decoder) &&
    is_sat_healthy(slot.snapshot.decoder)

# Admission elevation, above the horizon that releases a member: the hysteresis keeps a
# satellite near the horizon from being admitted and released every cycle. One degree
# is a few minutes of a rising satellite.
const ADMISSION_ELEVATION = deg2rad(1.0)

# Evaluated at the satellite's transmit time.
function _is_above_admission_mask(signal, slot::VectorTrackingSlot, enu_from_ecef)
    orbit = calc_satellite_position_and_velocity(_signal_observation(signal, slot))
    first(calc_elevation_and_azimuth(enu_from_ecef, orbit.position)) >= ADMISSION_ELEVATION
end

# Admit eligible satellites that are in lock and above the admission mask at the
# filter's position, and release members no longer eligible. Returns whether any was
# released. A member out of lock stays in the loop, unmeasured, and keeps receiving
# commands.
#
# NOTE: admission is gated on `measurement_ready`, not a stricter ranging-ready flag, although a
# satellite whose code phase is still tens of metres out may then enter the filter.
# `measurement_ready` also decides the members' measurement eligibility, so a stricter gate would
# withhold discriminators from existing members; the two uses need separating first.
function _update_membership!(released, group, group_index, buffer, enu_from_ecef)
    for slot in group.slots
        slot.in_cycle || continue
        eligible = _is_eligible(slot)
        if eligible && slot.snapshot.measurement_ready
            if slot.in_vector_loop || _is_above_admission_mask(group.driver_signal, slot, enu_from_ecef)
                slot.in_vector_loop = true
            end
        elseif !eligible && slot.in_vector_loop
            _mark_release!(slot, VT_INELIGIBLE)
            released = true
        end
    end
    released
end

# ─────────────────────────────────────────────────────────────────────────────
# Measurement gathering

function _collect_member_states!(acc, group, group_index, buffer)
    empty!(buffer)
    for slot in group.slots
        _is_member(slot) && push!(buffer, _signal_observation(group.driver_signal, slot))
    end
    acc
end

# A member's loop state at the epoch, as the vector loop sees it: its snapshot's, or, for a
# satellite this cycle admitted, that state with the vector interface empty (no
# discriminators accumulated yet), as `_take_up_cycle` will make it.
_member_loop_state(slot::VectorTrackingSlot) = _add_to_vector_loop(slot.snapshot.loop_state)

# One `VectorTrackingObservation` per satellite in the loop, from its measurement; returns the running index.
function _collect_members!(j, group, group_index, buffer, nav_processor, T)
    layout = nav_processor.nav_filter.layout
    buffers = nav_processor.buffers
    signal = group.driver_signal
    code_frequency = ustrip(Hz, get_chip_rate(signal))
    chip_length = SPEED_OF_LIGHT / code_frequency
    wavelength = SPEED_OF_LIGHT / ustrip(Hz, get_center_frequency(signal))
    clock_bias_index = layout.clock_bias_index_by_group[group_index]
    ifb_index = layout.ifb_index_by_group[group_index]
    for (slot_index, slot) in enumerate(group.slots)
        _is_member(slot) || continue
        loop_state = _member_loop_state(slot)
        j += 1
        measurement = buffers.measurements[j]
        push!(
            buffers.members,
            VectorTrackingObservation(
                group_index,
                slot_index,
                slot.prn,
                clock_bias_index,
                ifb_index,
                chip_length,
                wavelength,
                code_frequency,
                slot.snapshot.measurement_ready && has_disc_sums(loop_state),
                measurement.transmit_time,
                measurement.position,
                measurement.velocity,
                measurement.satellite_clock_drift,
                mean_code_disc_at_epoch(loop_state, T),
                -wavelength *
                (ustrip(Hz, slot.snapshot.carrier_doppler) + mean_fll_disc(loop_state)),
                linear_cn0_floor(slot.snapshot.cn0_dbhz),
                slot.snapshot.early_late_spacing,
                ustrip(s, slot.snapshot.coherent_integration_time),
            ),
        )
    end
    j
end

# Gather this cycle's members, their measurements at the epoch and per-member model pieces.
# Returns the ionospheric model and the broadcast time offsets prepared from the same
# observations, as `prepare_observations!` does.
function _gather_members!(nav_processor::NavigationProcessor, groups, T)
    buffers = nav_processor.buffers
    _fold_groups_with_buffers(_collect_member_states!, nothing, groups, buffers.signal_observations, 1)
    prepared = prepare_observations!(
        buffers.measurements,
        buffers.signal_observations;
        approximate_year = nav_processor.approximate_year,
    )
    # The collection pass keeps exactly the decoded, healthy satellites, i.e. every
    # member, so measurements and members align.
    num_rows = length(buffers.measurements)
    num_members = mapreduce(length, +, buffers.signal_observations; init = 0)
    num_rows == num_members || throw(
        ArgumentError(
            "a vector-loop member was not decoded for positioning and healthy; " *
            "its measurement is missing",
        ),
    )
    empty!(buffers.members)
    _fold_groups_with_buffers(_collect_members!, 0, groups, buffers.signal_observations, 1, nav_processor, T)
    members = buffers.members
    resize!(buffers.positions, num_members)
    for j in eachindex(members)
        buffers.positions[j] = members[j].sat_pos
    end
    member_bias_columns!(buffers.member_columns, members, eachindex(members))
    prepared
end

# The time system of the clock the solution is reported against, whose count the
# pseudoranges are measured on.
_reference_system(nav_filter::NavFilter) = nav_filter.layout.time_systems[nav_filter.reference_clock_index]

# Every member's atmosphere-corrected pseudorange against the receive epoch, the delays
# predicted at the filter's `x` for that epoch. Measured as `calc_pvt` measures them: on the
# reference system's count, onto which `calc_pseudorange` moves each transmit time by
# GNSSSignals' `get_time_system_offset`. The common count cancels into the receiver clock
# biases, so the pseudoranges do not depend on which system is the reference.
function _measure_pseudoranges!(nav_processor::NavigationProcessor, ionospheric_model, reference_epoch)
    nav_filter = nav_processor.nav_filter
    reference_system = _reference_system(nav_filter)
    reference_tow = get_time_of_week(reference_system, reference_epoch)
    buffers = nav_processor.buffers
    members = buffers.members
    measurements = buffers.measurements
    num_members = length(members)
    delays = resize!(buffers.delays, num_members)
    if num_members > 0 &&
       (nav_processor.enable_ionospheric_correction || nav_processor.enable_tropospheric_correction)
        position_and_bias_vector!(buffers.ξ, nav_filter.x, nav_filter.model.idxs)
        predict_atmospheric_delays!(
            delays,
            buffers.ξ,
            measurements,
            nav_processor.enable_ionospheric_correction ? ionospheric_model : IonosphericModel(nothing),
            reference_epoch,
            nav_processor.enable_tropospheric_correction,
        )
    else
        fill!(delays, 0.0)
    end
    measured = resize!(buffers.measured_pseudoranges, num_members)
    for j in eachindex(members)
        correction = get_time_system_offset(measurements[j].time_system, reference_system)
        measured[j] = calc_pseudorange(reference_tow, members[j].transmit_tow, correction) - delays[j]
    end
    measured
end

# ─────────────────────────────────────────────────────────────────────────────
# The measurement update

function _measurement_buffers!(buffers::VectorTrackingWorkspace, num_states, num_measurements)
    cache = buffers.measurement_updates
    if num_measurements > length(cache)
        # Not isbits: fill the new entries, or they stay `#undef`.
        num_cached = length(cache)
        resize!(cache, num_measurements)
        fill!(view(cache, (num_cached+1):num_measurements), nothing)
    end
    cached = cache[num_measurements]
    isnothing(cached) || return cached
    new_buffers = MeasurementUpdateWorkspace(num_states, num_measurements)
    cache[num_measurements] = new_buffers
    new_buffers
end

# Fuse the candidates' measurements at the filter's predicted `x`; the posterior into its
# `x` and `P`.
function _measurement_update!(nav_processor::NavigationProcessor, T)
    nav_filter = nav_processor.nav_filter
    buffers = nav_processor.buffers
    members = buffers.members
    candidates = buffers.candidates
    measured = buffers.measured_pseudoranges
    constraints = buffers.clock_constraints
    use_rates = nav_filter.config.use_pseudorange_rates
    num_observations = length(candidates)
    resize!(buffers.candidate_positions, num_observations)
    for (k, j) in enumerate(candidates)
        buffers.candidate_positions[k] = members[j].sat_pos
    end
    member_bias_columns!(buffers.candidate_columns, members, candidates)
    num_rate_rows = use_rates ? num_observations : 0
    num_measurements = num_observations + num_rate_rows + length(constraints)
    update = _measurement_buffers!(buffers, length(nav_filter.x), num_measurements)
    z = update.z
    R = fill!(update.R, 0.0)
    for (k, j) in enumerate(candidates)
        member = members[j]
        z[k] = measured[j] + member.code_discriminator * member.chip_length
        R[k, k] = pseudorange_noise_variance(member, T)
        if use_rates
            z[num_observations+k] = member.pseudorange_rate
            R[num_observations+k, num_observations+k] = pseudorange_rate_noise_variance(member, T)
        end
    end
    # Each clock constraint (a broadcast inter-system bias) is one extra row after the
    # pseudorange and rate rows.
    offset = num_observations + num_rate_rows
    for (i, (_, _, isb)) in enumerate(constraints)
        z[offset+i] = isb
        R[offset+i, offset+i] = BROADCAST_INTER_SYSTEM_BIAS_STD^2
    end
    h! = NavFilterMeasurementModel(
        nav_filter.model.idxs,
        buffers.ξ_model,
        buffers.candidate_positions,
        members,
        candidates,
        buffers.candidate_columns,
        use_rates,
        constraints,
    )
    measurement_update!(update.intermediate, nav_filter.x, nav_filter.P, z, h!, R)
    nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Loop closure

# Every member's pseudorange and rate predicted at the filter's updated `x`, which its NCO
# commands follow (`_assign_commands!`), and post-fit residuals.
function _predict_nco_commands!(nav_processor::NavigationProcessor)
    nav_filter = nav_processor.nav_filter
    buffers = nav_processor.buffers
    members = buffers.members
    num_members = length(members)
    idxs = nav_filter.model.idxs
    position_and_bias_vector!(buffers.ξ, nav_filter.x, idxs)
    predicted = resize!(buffers.predicted_pseudoranges, num_members)
    predict_pseudoranges!(predicted, buffers.positions, buffers.ξ, buffers.member_columns)
    rates = resize!(buffers.predicted_pseudorange_rates, num_members)
    residuals = resize!(buffers.residuals, num_members)
    rate_residuals = resize!(buffers.rate_residuals, num_members)
    for (j, member) in enumerate(members)
        rates[j] = predict_pseudorange_rate(nav_filter.x, idxs, member)
        residuals[j], rate_residuals[j] = post_fit_residuals(
            member,
            buffers.measured_pseudoranges[j],
            predicted[j],
            rates[j],
        )
    end
    nothing
end

# Point every member above the horizon at this cycle's commands; the satellite
# evaluates them where its command takes effect, on its next integration (`_commands_at_application`).
function _assign_commands!(acc, group, group_index, buffer, nav_processor)
    buffers = nav_processor.buffers
    for (j, member) in enumerate(buffers.members)
        member.group == group_index && buffers.above_horizon[j] || continue
        slot = group.slots[member.slot_index]
        slot.command_cycle_index = nav_processor.cycle_index
        slot.member_index = j
    end
    acc
end

# The NCO commands (Hz) of member `j`, `τ` after the epoch: where its command takes effect,
# and one cycle later for the carrier ramp's end (see `_predict_after_epoch`). The code
# correction removes the range error (TOW-based, atmosphere-corrected, no
# discriminator term); the carrier is the predicted Doppler itself.
function _member_commands(nav_processor::NavigationProcessor, slot::VectorTrackingSlot, j, τ, chips)
    nav_filter = nav_processor.nav_filter
    member = nav_processor.buffers.members[j]
    predicted_pseudorange, predicted_rate, measured_pseudorange =
        _predict_after_epoch(nav_processor, slot, member, j, _reference_system(nav_filter), τ, chips)
    code_update = nco_code_correction(
        predicted_pseudorange,
        measured_pseudorange,
        member.code_frequency,
        ustrip(s, nav_filter.model.cycle_time),
    )
    code_update, predicted_carrier_doppler(predicted_rate, member.wavelength)
end

# Member `j`'s predicted pseudorange and rate `τ` after the epoch, from the filter's `x`
# propagated by `τ`, and the pseudorange its replica realises there: transmit time
# moved on by the `chips` the replica advances (the satellite clock correction changes
# by picoseconds in `τ`), receive time `τ` after the receive epoch on the count of
# `reference_system` (see `_measure_pseudoranges!`), and the epoch's atmospheric delay
# (it moves by millimetres in `τ`).
function _predict_after_epoch(nav_processor::NavigationProcessor, slot::VectorTrackingSlot, member::VectorTrackingObservation, j, reference_system, τ, chips)
    nav_filter = nav_processor.nav_filter
    buffers = nav_processor.buffers
    idxs = nav_filter.model.idxs
    application_transmit_tow = member.transmit_tow + chips / member.code_frequency
    ephemeris = something(get_broadcast_ephemeris(slot.snapshot.decoder))
    application_orbit = calc_satellite_position_and_velocity(ephemeris, application_transmit_tow)
    application_position = application_orbit.position
    application_velocity = application_orbit.velocity
    propagate_state!(buffers.x_application, nav_filter.x, idxs, τ)
    position_and_bias_vector!(buffers.ξ_application, buffers.x_application, idxs)
    buffers.application_position[1] = application_position
    buffers.application_columns.clock_columns[1] = member.clock_bias_index
    buffers.application_columns.ifb_columns[1] = member.ifb_index
    predict_pseudoranges!(
        buffers.application_range,
        buffers.application_position,
        buffers.ξ_application,
        buffers.application_columns,
    )
    predicted_pseudorange = buffers.application_range[1]
    user_pos, user_vel, user_clock_drift = nav_filter_states(buffers.x_application, idxs)
    # Positive receding, as `predict_pseudorange_rate`.
    predicted_rate = calc_pseudorange_rate(
        user_pos,
        user_vel,
        user_clock_drift,
        application_position,
        application_velocity,
        calc_satellite_clock_drift(ephemeris, application_transmit_tow),
    )
    measured_pseudorange =
        calc_pseudorange(
            get_time_of_week(reference_system, nav_filter.reference_epoch) + τ,
            application_transmit_tow,
            get_time_system_offset(buffers.measurements[j].time_system, reference_system),
        ) -
        buffers.delays[j]
    predicted_pseudorange, predicted_rate, measured_pseudorange
end

# Release the group's members below the horizon; returns whether any was.
function _release_below_horizon!(released, group, group_index, buffer, nav_processor)
    buffers = nav_processor.buffers
    for (j, member) in enumerate(buffers.members)
        member.group == group_index && !buffers.above_horizon[j] || continue
        _mark_release!(group.slots[member.slot_index], VT_BELOW_HORIZON)
        released = true
    end
    released
end

# Release every satellite of the group still in the loop, members and those that
# missed the epoch alike, for `reason`; returns whether any was.
function _release_all!(released, group, group_index, buffer, reason::VectorTrackingReleaseReason)
    for slot in group.slots
        slot.in_vector_loop || continue
        _mark_release!(slot, reason)
        released = true
    end
    released
end

function _count_members(n, group, group_index, buffer)
    for slot in group.slots
        _is_member(slot) && (n += 1)
    end
    n
end

# ─────────────────────────────────────────────────────────────────────────────
# Seeding from a scalar fix

function _add_fix_satellites!(num_added, group, group_index, buffer, nav_processor)
    signal_id = nav_processor.nav_filter.layout.signal_id_by_group[group_index]
    observations = nav_processor.pvt.observations
    for slot in group.slots
        slot.in_cycle && haskey(observations, (signal_id, slot.prn)) || continue
        slot.in_vector_loop = true
        num_added += 1
    end
    num_added
end

# Scale the seeded covariance by the fix's geometry (`seed_fix_covariance!`) at the
# seeded position, over the bias layout `calc_pvt` decided for the fix with the epoch's
# broadcast `inter_system_time_offsets`. `candidates` and the bias-layout workspace serve as scratch:
# every cycle decides its layout before anything reads it.
function _seed_fix_covariance!(nav_processor::NavigationProcessor, inter_system_time_offsets)
    nav_filter = nav_processor.nav_filter
    buffers = nav_processor.buffers
    members = buffers.members
    layout = nav_filter.layout
    idxs = nav_filter.model.idxs
    included = empty!(buffers.candidates)
    for j in eachindex(members)
        haskey(nav_processor.pvt.observations, _member_key(layout, members[j])) && push!(included, j)
    end
    isempty(included) && return false
    resize!(buffers.candidate_positions, length(included))
    for (k, j) in enumerate(included)
        buffers.candidate_positions[k] = members[j].sat_pos
    end
    fix_bias_layout = buffers.bias_layout
    decide_bias_layout!(fix_bias_layout, view(buffers.measurements, included), inter_system_time_offsets) || return false
    H = _design_matrix!(buffers, nav_filter.x, idxs, fit_bias_columns(fix_bias_layout), length(included))
    # A broadcast offset is credited `BROADCAST_INTER_SYSTEM_BIAS_STD`, as in the filter's update, not the
    # pseudoranges' `FIX_PSEUDORANGE_STD`: its rows follow the satellites'. A loop, as
    # `H[rows, :] .*= w` would copy the right-hand slice.
    for row in (length(included)+1):size(H, 1), column in axes(H, 2)
        H[row, column] *= FIX_PSEUDORANGE_STD / BROADCAST_INTER_SYSTEM_BIAS_STD
    end
    seed_fix_covariance!(
        nav_filter.P,
        idxs,
        layout,
        nav_processor.pvt,
        nav_filter.reference_clock_index,
        H,
        buffers.normal_matrices[size(H, 2)],
        fix_bias_layout,
    )
end

# The design matrix of the first `num_observations` of `buffers.candidate_positions` over
# `bias_columns`, linearised at `x`, a row per satellite and then per constraint: a view
# of `buffers.design_matrix`, grown if short.
function _design_matrix!(buffers, x, idxs, bias_columns::BiasColumns, num_observations)
    num_rows = num_observations + length(bias_columns.constraints)
    if size(buffers.design_matrix, 1) < num_rows
        buffers.design_matrix = zeros(2 * num_rows, size(buffers.design_matrix, 2))
    end
    H = view(buffers.design_matrix, 1:num_rows, 1:num_parameters(bias_columns))
    position_and_bias_vector!(buffers.ξ, x, idxs)
    calc_design_matrix!(H, buffers.candidate_positions, buffers.ξ, bias_columns)
end

# Switch to vector tracking off the fresh fix in `nav_processor.pvt`: add its satellites, seed the
# filter from it and assign first commands (no measurement update, no accumulator
# reset). A fresh fix has satellites, all in the cycle (only those enter the scalar
# solve), so there is always a member to seed from. None is in the loop yet:
# slot_generation puts a satellite out of it, and the fallback releases every one in it.
function _seed!(nav_processor::NavigationProcessor, groups, cycle_time)
    nav_filter = nav_processor.nav_filter
    buffers = nav_processor.buffers
    model = nav_filter.model
    ensure_nav_filter_cycle_time!(model, nav_filter.config, cycle_time)
    T = ustrip(s, model.cycle_time)
    pvt = nav_processor.pvt
    nav_filter.reference_clock_index = initial_nav_state!(nav_filter.x, nav_filter.P, nav_filter.layout, model.idxs, pvt)
    _fold_groups_with_buffers(_add_fix_satellites!, 0, groups, buffers.signal_observations, 1, nav_processor)
    prepared = _gather_members!(nav_processor, groups, T)
    _seed_fix_covariance!(nav_processor, prepared.inter_system_time_offsets)
    members = buffers.members
    # The fix's epoch: the receive time the filter's clock biases count from, the
    # reference one starting at zero, taken from `pvt.time`.
    reference_epoch = something(pvt.time)
    _measure_pseudoranges!(nav_processor, prepared.ionospheric_model, reference_epoch)
    _predict_nco_commands!(nav_processor)
    resize!(buffers.above_horizon, length(members))
    fill!(buffers.above_horizon, true)
    _fold_groups_with_buffers(_assign_commands!, 0, groups, buffers.signal_observations, 1, nav_processor)
    nav_processor.vector_tracking_active = true
    nav_filter.reference_epoch = reference_epoch
    nav_processor.measurement_deficit_duration = 0.0s
    nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# One cycle

# One cycle of the navigation filter, a linear Kalman time update and an unscented
# measurement update: manage membership, predict (time update), fuse the accumulated
# discriminators as pseudorange (and, for VDFLL, rate) measurements (measurement update),
# and assign the NCO commands or fall back to scalar tracking. Returns
# `(released, fell_back)`.
function _nav_filter_cycle!(nav_processor::NavigationProcessor, groups, cycle_time)
    nav_filter = nav_processor.nav_filter
    config = nav_filter.config
    buffers = nav_processor.buffers
    model = nav_filter.model
    idxs = model.idxs
    T = ustrip(s, cycle_time)
    ensure_nav_filter_cycle_time!(model, config, cycle_time)
    # Advance the receive epoch: its time of week crosses into the next week by itself.
    reference_epoch = nav_filter.reference_epoch + T

    # 1. Membership: admission and the elevation mask, judged at the cycle's starting
    #    solution.
    enu_from_ecef = ENUfromECEF(_ecef(first(nav_filter_states(nav_filter.x, idxs))), wgs84)
    released =
        _fold_groups_with_buffers(_update_membership!, false, groups, buffers.signal_observations, 1, enu_from_ecef)
    prepared = _gather_members!(nav_processor, groups, T)
    members = buffers.members
    num_members = length(members)

    # A member below the horizon is released, neither measured nor commanded.
    above_horizon = resize!(buffers.above_horizon, num_members)
    num_above_horizon = 0
    for (j, member) in enumerate(members)
        above_horizon[j] = first(calc_elevation_and_azimuth(enu_from_ecef, member.sat_pos)) >= 0
        above_horizon[j] && (num_above_horizon += 1)
    end
    released = _fold_groups_with_buffers(_release_below_horizon!, released, groups, buffers.signal_observations, 1, nav_processor)

    # 2. Time update (prediction) to the cycle's epoch: x⁻ = F·x⁺, P⁻ = F·P⁺·Fᵀ + Q.
    time_update!(buffers.time_update, nav_filter.x, nav_filter.P, model.F, model.Q)

    # 3. Measurements: each member's pseudorange from its snapshot, corrected by its
    #    accumulated code discriminator in the update below.
    _measure_pseudoranges!(nav_processor, prepared.ionospheric_model, reference_epoch)

    # 4. Candidates: members above the horizon with an available signal (an obscured one
    #    carries no information), and the clock and bias states they determine.
    candidates = empty!(buffers.candidates)
    for (j, member) in enumerate(members)
        member.available && above_horizon[j] && push!(candidates, j)
    end
    # Decided every cycle, without candidates too (unsolvable, no constraint), so nothing
    # below reads an earlier cycle's decision.
    candidate_observations = view(buffers.measurements, candidates)
    solvable = decide_bias_layout!(buffers.bias_layout, candidate_observations, prepared.inter_system_time_offsets)
    clock_constraints!(buffers.clock_constraints, nav_filter.layout, buffers.bias_layout)
    # 5. Measurement update (unscented): the predicted measurements from sigma points
    #    through h, then x⁺ = x⁻ + K·(z − ẑ) and P⁺.
    #    No innovation gate: the starvation watchdog catches divergence, and a gate could
    #    release a healthy satellite whose large but explained innovation the update would
    #    absorb (e.g. a constellation's first measurements).
    isempty(candidates) || _measurement_update!(nav_processor, T)

    # 6. Loop closure: every member's pseudorange and rate predicted at the updated state,
    #    which the NCO commands follow.
    _predict_nco_commands!(nav_processor)

    # 7. Starvation watchdog (see `VectorTrackingStatus`): solvability only; the solution's
    # certainty is reported as `position_std`, not policed.
    measurement_deficit_duration =
        solvable ?
        max(0.0s, nav_processor.measurement_deficit_duration - cycle_time / 2) :
        nav_processor.measurement_deficit_duration + cycle_time

    # 8. The NCO commands, or the fallback to scalar tracking.
    fell_back = measurement_deficit_duration > config.measurement_deficit_timeout || num_above_horizon == 0
    if fell_back
        released = _fold_groups_with_buffers(_release_all!, released, groups, buffers.signal_observations, 1, VT_FALLBACK)
        nav_processor.vector_tracking_active = false
    else
        _fold_groups_with_buffers(_assign_commands!, 0, groups, buffers.signal_observations, 1, nav_processor)
    end

    # 9. The solution, dated by the reference system's clock.
    nav_filter.reference_clock_index =
        select_reference_clock_index(nav_filter.layout, buffers.bias_layout, nav_filter.reference_clock_index)
    nav_filter.reference_epoch = reference_epoch
    nav_processor.measurement_deficit_duration = measurement_deficit_duration
    _write_solution!(nav_processor, groups)
    released, fell_back
end

_ecef(v) = ECEF(v[1], v[2], v[3])

# ─────────────────────────────────────────────────────────────────────────────
# The solution

function _write_solution!(nav_processor::NavigationProcessor, groups)
    nav_filter = nav_processor.nav_filter
    buffers = nav_processor.buffers
    members = buffers.members
    layout = nav_filter.layout
    idxs = nav_filter.model.idxs
    x = nav_filter.x
    user_pos, user_vel, user_clock_drift = nav_filter_states(x, idxs)
    position = _ecef(user_pos)
    velocity = _ecef(user_vel)
    reference_clock_bias = x[idxs.clock_biases[nav_filter.reference_clock_index]]
    candidates = buffers.candidates

    # DOP of the measured satellites over the cycle's bias layout, as `calc_pvt` computes
    # it, or `nothing` without measurements or for a rank-deficient design (`calc_dop!`'s
    # all-`NaN` sentinel, which `calc_pvt` never emits either).
    dop = nothing
    if !isempty(candidates)
        cycle_layout = buffers.bias_layout
        H = _design_matrix!(buffers, x, idxs, dop_bias_columns(cycle_layout), length(candidates))
        reference_column = clock_column(
            cycle_layout,
            view(buffers.measurements, candidates),
            layout.time_systems[nav_filter.reference_clock_index],
        )
        candidate = calc_dop!(buffers.normal_matrices[size(H, 2)], H, position, reference_column)
        isnan(candidate.GDOP) || (dop = candidate)
    end

    # `pvt.observations` holds the measured members only, as `calc_pvt` does, so it matches
    # `pvt.dop`; coasted members are in `get_vector_tracking_members`.
    solution = empty_keeping_capacity!(nav_processor.pvt)
    observations = solution.observations
    for j in candidates
        set!(observations, _member_key(layout, members[j]), _member_info(buffers, j))
    end
    satellite_reports = empty_keeping_capacity!(nav_processor.vector_tracking_members)
    for j in eachindex(members)
        set!(satellite_reports, _member_key(layout, members[j]), _member_info(buffers, j))
    end

    # Only measured biases, as in `calc_pvt`: an unmeasured one coasts on process noise
    # (or was never seeded) and would be indistinguishable from a measured one.
    inter_system_biases = solution.inter_system_biases
    for (index, time_system) in enumerate(layout.time_systems)
        index == nav_filter.reference_clock_index && continue
        _is_measured(member -> member.clock_bias_index == index, members, candidates) ||
            continue
        set!(inter_system_biases, time_system, (x[idxs.clock_biases[index]] - reference_clock_bias) * m)
    end
    inter_frequency_biases = solution.inter_frequency_biases
    for (index, band) in enumerate(layout.extra_bands)
        _is_measured(member -> member.ifb_index == index, members, candidates) || continue
        set!(
            inter_frequency_biases,
            band,
            InterFrequencyBias(x[idxs.ifb[index]] * m, layout.reference_bands[index]),
        )
    end

    # The broadcast offsets the cycle's update took as pseudo-measurements, as `calc_pvt`
    # reports the ones it takes as given; here they hold to `BROADCAST_INTER_SYSTEM_BIAS_STD` rather than
    # exactly. Without candidates there was no update.
    applied_biases = solution.applied_broadcast_inter_system_biases
    isempty(candidates) || applied_broadcast_inter_system_biases!(applied_biases, buffers.bias_layout)

    nav_processor.pvt = PVTSolution(;
        position,
        velocity,
        course_over_ground = calc_course_over_ground(position, velocity),
        time = nav_filter.reference_epoch - reference_clock_bias / SPEED_OF_LIGHT,
        clock_drift = user_clock_drift * (m / s),
        reference_system = layout.time_systems[nav_filter.reference_clock_index],
        inter_system_biases,
        applied_broadcast_inter_system_biases = applied_biases,
        inter_frequency_biases,
        dop,
        observations = observations,
    )
    nothing
end

function _is_measured(predicate::F, members, candidates) where {F}
    for j in candidates
        predicate(members[j]) && return true
    end
    false
end
_member_key(layout::NavFilterLayout, member::VectorTrackingObservation) =
    (layout.signal_id_by_group[member.group], member.prn)

_member_info(buffers::VectorTrackingWorkspace, j) = ObservationResiduals(
    buffers.members[j].sat_pos,
    buffers.members[j].transmit_tow * s,
    buffers.residuals[j],
    buffers.rate_residuals[j],
)

