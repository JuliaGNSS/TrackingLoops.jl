# ─────────────────────────────────────────────────────────────────────────────
# Vector tracking: the navigation processor that closes all loops
#
# One `NavigationProcessor` is shared by every satellite a `VectorTrackingLoop` steps. Each
# satellite's slot is filled by its integrations: the decoder from the soft bits of its
# decoding signal and, at every navigation epoch, a snapshot of the driver's replica,
# C/N₀ and accumulated discriminators, with the decoding signal's bit count. Once
# every satellite has completed the epoch's snapshot the update runs: a scalar PVT solve
# until its first fix seeds the filter, one filter iteration after. The update leaves its
# decisions (admission, release, commands) in the slots for each satellite to apply
# on its next integration. Nothing here knows a correlator, so software and hardware loops
# step the same loop.
# ─────────────────────────────────────────────────────────────────────────────

"""
    VectorTrackingReleaseReason

Why the latest navigation update handed a satellite back to its scalar loop
(a [`get_satellite_report`](@ref)'s `release_reason`):

  - `VT_NOT_RELEASED`: it was not;
  - `VT_INELIGIBLE`: no longer tracked (no integration for two navigation filter
    updates), no longer decoded for positioning, or no longer healthy;
  - `VT_BELOW_HORIZON`: below the horizon at the solution the update starts
    from, so not measured in that update; re-admitted only `ADMISSION_ELEVATION`
    (1°) above it (hysteresis);
  - `VT_FALLBACK`: vector tracking stopped (measurement deficit timeout, or no member
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

  - `prn`; a satellite dropped after two update intervals without an integration is
    still reported, as it was, with `release_reason` `VT_INELIGIBLE`;
  - `bit_or_secondary_code_synced`: whether the decoding signal's bit clock held the bit sync at
    that signal's last integration;
  - at its latest snapshot (`snapshot_time`, on the navigation-epoch grid, a multiple of
    `navigation_update_interval`, `nothing` before
    the first): `cn0` (the driver's, from the host's C/N₀ estimator),
    `measurement_ready` (decoding signal synced and the driver's C/N₀ at least the lock
    threshold) and `positioning_ready` (in lock, decoded for positioning and healthy);
  - `vector_controlled` and `release_reason`: the latest update's decision.

The report is an immutable snapshot whose values remain valid after later integrations
and report calls. Access the separately borrowed decoder with [`get_decoder`](@ref).
"""
struct SatelliteReport
    prn::Int
    bit_or_secondary_code_synced::Bool
    snapshot_time::Union{Nothing,typeof(1.0s)}
    cn0::typeof(1.0dBHz)
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
    # Of the signal's band: the end is at `last_end_sample_index / sampling_frequency`
    # on the integrations' time grid (`_last_end_time`).
    sampling_frequency::typeof(1.0Hz)
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
    sampling_frequency::typeof(1.0Hz) # of the signal's band (see `DriverState`)
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
# the update reads (`snapshot`). The update does not write it: its decisions are the slot's.
struct SatelliteSnapshot{D,E}
    # The driver's half: the state at the epoch with the discriminators accumulated up to
    # it, the replica's code phase there
    # (chips; `NaN` if not reported), the Dopplers commanded there, the driver's C/N₀,
    # coherent integration time and early-late spacing.
    loop_state::E
    driver_code_phase::Float64
    carrier_doppler::typeof(1.0Hz)
    code_doppler::typeof(1.0Hz)
    cn0::typeof(1.0dBHz)
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
    cn0 = snapshot.cn0,
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
    cn0,
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
    # Whether the slot snapshotted the running update's epoch.
    in_update::Bool
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
    # What the latest update decided, for the satellite to apply: whether it is in the
    # vector loop (from its snapshot's state until an update admits or releases it), why it
    # was released, and the update and member index of its commands.
    vector_controlled::Bool
    release_reason::VectorTrackingReleaseReason
    commanded_update_index::Int
    member_index::Int
end

function VectorTrackingSlot(
    decoding_signal::AbstractGNSSSignal,
    prn::Integer,
    loop_state::VectorLoopState,
    prompt_history_length::Int,
)
    decoder = GNSSDecoderState(decoding_signal, prn)
    snapshot = SatelliteSnapshot(loop_state, NaN, 0.0Hz, 0.0Hz, -Inf * dBHz, 0.001s, 0.5, decoder, false, 0.0, false, false)
    VectorTrackingSlot(
        Int(prn),
        false,
        0,
        false,
        DriverState(0, 1.0Hz, NaN, 0.001s, 0.5, 0.0),
        DecodingState(decoder, 0, 1.0Hz, false, 0, NaN, typemin(Int)),
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
    navigation_update_interval,
)
    prompt_history_length = _prompt_history_length(driver_signal, decoding_signal, navigation_update_interval)
    # Inactive slots need a valid decoder prototype; the receiver validates tracked PRNs.
    prototype_prn = first(get_supported_prns(decoding_signal))
    slots = [VectorTrackingSlot(decoding_signal, prototype_prn, prototype, prompt_history_length) for _ = 1:capacity]
    VectorTrackingSlotGroup(driver_signal, decoding_signal, sizehint!(slots, 2 * capacity), prototype, prompt_history_length)
end

"""
    VectorTrackingStatus

What the latest navigation update of a [`VectorTrackingLoop`](@ref) did (see
[`get_navigation_status`](@ref)).

  - `vector_tracking_active`: vector tracking is active after this update;
  - `position_std`, `clock_bias_std`: the filter's 1σ 3-D position and reference-clock
    uncertainties (`NaN` unless the filter ran or fell back this update); a
    degenerate geometry shows here, not in the measurement deficit timer;
  - `measurement_deficit_duration`: how long the filter has coasted on epochs it
    could not solve, paid back at half rate otherwise, and checked against
    `measurement_deficit_timeout`; nonzero means the loop is on its way out;
  - `num_vector_tracking_members`: members of the vector loop after this update, one
    per satellite and ranging signal (see [`get_vector_tracking_members`](@ref));
  - the events `seeded` (a fresh scalar fix seeded the filter), `fell_back`
    (vector tracking stopped) and `released` (some satellite was released; see
    the reports' `release_reason`);
  - `solution_from_nav_filter`: [`get_navigation_solution`](@ref) is the navigation filter's
    this update, not a scalar fix. It differs from `vector_tracking_active` on the
    transitions: the update that seeds the filter reports the scalar fix it seeded
    from, and the update that falls back still reports the filter's last update.

The per-member report of the latest update is [`get_vector_tracking_members`](@ref).
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
    solution_from_nav_filter::Bool
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

# An update's scratch, sized for the preallocated slots; vectors grow past that with
# `push!` / `resize!`, the design matrix by replacement.
mutable struct VectorTrackingWorkspace{SB<:Tuple}
    # Per group, the `SignalObservation`s whose measurements are collected.
    signal_observations::SB
    measurements::Vector{PreparedObservation}
    members::Vector{VectorTrackingObservation}
    # Per member, whether it stayed above the horizon this update.
    above_horizon::Vector{Bool}
    used_member_indices::Vector{Int}
    atmospheric_delays::Vector{Float64}
    measured_pseudoranges::Vector{Float64}
    predicted_pseudoranges::Vector{Float64}
    predicted_pseudorange_rates::Vector{Float64}
    residuals::Vector{typeof(1.0m)}
    rate_residuals::Vector{typeof(1.0m / s)}
    member_satellite_positions::Vector{SVector{3,Float64}}
    member_bias_columns::BiasColumns
    used_satellite_positions::Vector{SVector{3,Float64}}
    used_bias_columns::BiasColumns
    ξ::Vector{Float64}
    ξ_model::Vector{Float64}
    ξ_at_application::Vector{Float64}
    x_at_application::Vector{Float64}
    satellite_position_at_application::Vector{SVector{3,Float64}}
    bias_columns_at_application::BiasColumns
    predicted_pseudorange_at_application::Vector{Float64}
    # The update's bias-layout decision, and its constraints over the filter's clocks.
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
    const config::NavFilterConfig
    const layout::NavFilterLayout
    const model::NavFilterModel
    const x::Vector{Float64}
    const P::Matrix{Float64}
    # The clock-bias state the solution is reported against.
    reference_clock_index::Int
    # The receive epoch the pseudoranges are referenced to, uncorrected for the receiver
    # clock: the fix's epoch at the seed, advanced by the update interval every update. They
    # difference against its time of week on the reference system's count (see
    # `_measure_pseudoranges!`), so it does not move when the reference clock changes.
    reference_epoch::TAITime
end

"""
    NavigationProcessor

The navigation processor of a [`VectorTrackingLoop`](@ref), shared by every satellite
it steps: the navigation filter (`nav_filter`), the slots, the scalar PVT it falls back
to, and every buffer an update needs.

It holds the latest solution (`pvt`, whose containers every update reuses) and
the member results (`vector_tracking_members`, read with
[`get_vector_tracking_members`](@ref)). `pvt.observations` carries only
the members the update measured, so the keys missing from it are the members the
filter coasted through an obscuration. `vector_tracking_active` says whether the
navigation filter runs: `false` until a scalar fix seeds it and again after a fallback,
while the update solves the scalar PVT.

A satellite passes these checks on its way into an update's measurements, each on
top of the one before:

  - `occupied`: its slot holds it (it is stepped and not stale);
  - `in_update`: it latched the update's measurement epoch;
  - `measurement_ready`: its decoding signal is synced and the driver's C/N₀ is at least
    the lock threshold;
  - available: measurement-ready with discriminator sums accumulated since the last
    update (`has_disc_sums`);
  - used: available and above the horizon at the update's starting solution.

Admission into the vector loop takes `positioning_ready` (in lock, decoded for
positioning and healthy) and, on top, the elevation mask (`ADMISSION_ELEVATION`).

The phases the residual PLL predicted at the driver's latest prompts live in the slot
(`prompt_history`), not in the loop state that holds the residual PLL: a data
component's integrations reach the processor before or after the driver's, and the
history lets either be wiped with the phase at its own prompt.
"""
mutable struct NavigationProcessor{G<:Tuple,SB<:Tuple}
    const nav_filter::NavFilter
    vector_tracking_active::Bool
    measurement_deficit_duration::typeof(1.0s)
    const vector_tracking_members::Dictionary{Tuple{Symbol,Int},ObservationResiduals}
    pvt::PVTSolution
    status::VectorTrackingStatus
    const pvt_workspace::PVTWorkspace
    const approximate_year::Int
    const enable_ionospheric_correction::Bool
    const enable_tropospheric_correction::Bool
    const workspace::VectorTrackingWorkspace{SB}
    const groups::G
    const update_interval::typeof(1.0s)
    const min_tracking_cn0::typeof(1.0dBHz)
    # The epoch (in `navigation_update_interval`s on the integrations' time grid) the slots snapshot
    # next, and how many have.
    pending_epoch_index::Int
    num_snapshots::Int
    # The latest update's id and epoch.
    update_index::Int
    last_update_epoch_index::Int
    next_slot_generation::Int
end

function NavigationProcessor(
    config::NavFilterConfig,
    pairs::Tuple,
    scalar_loop::AbstractTrackingLoop;
    navigation_update_interval,
    min_tracking_cn0::typeof(1.0dBHz),
    satellite_capacity_per_signal::Int,
    approximate_year::Integer,
    enable_ionospheric_correction::Bool,
    enable_tropospheric_correction::Bool,
)
    signals = map(first, pairs)
    layout = NavFilterLayout(signals)
    model = NavFilterModel(config, layout, uconvert(s, navigation_update_interval))
    n = num_nav_states(config, layout)
    groups = map(pairs) do (signal, decoding_signal)
        prototype = VectorLoopState(init_loop_state(scalar_loop, signal, 0.0Hz, 0.0Hz), false)
        VectorTrackingSlotGroup(signal, decoding_signal, prototype, satellite_capacity_per_signal, navigation_update_interval)
    end
    max_members = satellite_capacity_per_signal * length(signals)
    signal_observations = map(_signal_observation_buffer, groups)
    NavigationProcessor(
        NavFilter(config, layout, model, zeros(n), zeros(n, n), 1, TAITime(0, 0.0)),
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
        uconvert(s, navigation_update_interval),
        min_tracking_cn0,
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
filter's after. The next update reuses its containers; copy out what is needed.
"""
get_navigation_solution(tracking_loop::VectorTrackingLoop) = tracking_loop.navigation_processor.pvt

"""
    get_navigation_status(tracking_loop::VectorTrackingLoop) -> VectorTrackingStatus

What the latest navigation update did (see [`VectorTrackingStatus`](@ref)).
"""
get_navigation_status(tracking_loop::VectorTrackingLoop) = tracking_loop.navigation_processor.status

get_navigation_update_index(tracking_loop::VectorTrackingLoop) = tracking_loop.navigation_processor.update_index

function get_navigation_elapsed_time(tracking_loop::VectorTrackingLoop)
    nav_processor = tracking_loop.navigation_processor
    nav_processor.last_update_epoch_index == typemin(Int) ? nothing : nav_processor.last_update_epoch_index * nav_processor.update_interval
end

function get_satellite_report(tracking_loop::VectorTrackingLoop, signal::AbstractGNSSSignal, prn::Integer)
    nav_processor = tracking_loop.navigation_processor
    slot = _driver_slot(nav_processor.groups, signal, Int(prn))
    isnothing(slot) ? nothing : _satellite_report(nav_processor, slot)
end

# The slot that holds or held `prn`, or `nothing`.
function _held_slot(group, prn)
    index = _find_slot(group.slots, prn)
    index == 0 && return nothing
    slot = group.slots[index]
    slot.prn == prn && slot.slot_generation > 0 ? slot : nothing
end

# The slot that holds or held `prn`, else any free one; 0 when none is free.
function _find_slot(slots, prn)
    free = 0
    for (index, slot) in enumerate(slots)
        if slot.prn == prn && slot.slot_generation > 0
            return index
        end
        !slot.occupied && free == 0 && (free = index)
    end
    free
end

# The slot of the group `signal` drives that holds or held `prn`, or `nothing`; searched by
# type, so the search folds at compile time.
_driver_slot(::Tuple{}, signal, prn) = nothing
function _driver_slot(groups::Tuple, signal::S, prn) where {S}
    group = first(groups)
    group.driver_signal isa S ? _held_slot(group, prn) : _driver_slot(Base.tail(groups), signal, prn)
end

function _satellite_report(nav_processor, slot::VectorTrackingSlot)
    snapshot_time = slot.snapshot_epoch_index == typemin(Int) ? nothing :
        slot.snapshot_epoch_index * nav_processor.update_interval
    SatelliteReport(
        slot.prn,
        slot.decoding.bit_synced,
        snapshot_time,
        slot.snapshot.cn0,
        slot.snapshot.measurement_ready,
        slot.snapshot.positioning_ready,
        slot.vector_controlled,
        slot.release_reason,
    )
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
function get_decoder(tracking_loop::VectorTrackingLoop, signal::AbstractGNSSSignal, prn::Integer)
    slot = _driver_slot(tracking_loop.navigation_processor.groups, signal, Int(prn))
    isnothing(slot) ? nothing : slot.decoding.decoder
end

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
    get_vector_tracking_members(tracking_loop::VectorTrackingLoop)

The latest update's per-member results: every member of the vector loop, measured
*and* coasted, keyed `(driver_signal_id, prn)`, each with the satellite position,
transmit time and post-fit residuals. One physical satellite can have several
members, one per ranging signal. The solution's `observations` contains only the
measured members. Empty while the scalar solve is in control.

The returned dictionary is borrowed storage, reused by the next navigation update;
copy it to retain the current results.
"""
get_vector_tracking_members(tracking_loop::VectorTrackingLoop) = tracking_loop.navigation_processor.vector_tracking_members

# ─────────────────────────────────────────────────────────────────────────────
# Time on the integrations' grid

# The navigation epochs are compared with the integrations in samples, on each signal's own
# grid, not in seconds since the origin: `sample_index / fs` and `k · Δt_nav` round apart as
# the session grows, by nanoseconds after a year, while sample indices stay exact.

_update_interval_seconds(nav_processor::NavigationProcessor) = ustrip(s, nav_processor.update_interval)
# Where a signal's last integration ended (s) on the integrations' time grid.
_last_end_time(role::Union{DriverState,DecodingState}) = role.last_end_sample_index / ustrip(Hz, role.sampling_frequency)

# The samples an update interval spans at `fs` (Hz), snapped to the whole number it is
# meant to be when it comes out within a millionth of a sample of one (70 ms at 5 MHz is a
# hair over 350000 in binary).
function _samples_per_update(nav_processor::NavigationProcessor, fs)
    num_samples = _update_interval_seconds(nav_processor) * fs
    whole = round(num_samples)
    abs(num_samples - whole) < 1e-6 ? whole : num_samples
end

# Epoch `epoch_index` as a sample index on the grid of a signal sampled at `fs` (Hz); exact
# where an update interval spans whole samples.
_epoch_sample_index(nav_processor::NavigationProcessor, epoch_index::Int, fs) =
    epoch_index * _samples_per_update(nav_processor, fs)

# The first epoch not before sample index `sample_index` of a signal sampled at `fs` (Hz)
# (`sample_index / samples_per_update` may round to either side), or not before a role's
# last integration end.
function _epoch_at_or_after(nav_processor::NavigationProcessor, sample_index, fs)
    num_samples = _samples_per_update(nav_processor, fs)
    epoch_index = ceil(Int, sample_index / num_samples)
    (epoch_index - 1) * num_samples >= sample_index ? epoch_index - 1 :
    epoch_index * num_samples < sample_index ? epoch_index + 1 : epoch_index
end
_epoch_at_or_after(nav_processor::NavigationProcessor, role::Union{DriverState,DecodingState}) =
    _epoch_at_or_after(nav_processor, role.last_end_sample_index, ustrip(Hz, role.sampling_frequency))

# Whether a role's last integration ended by epoch `epoch_index`.
_ended_by_epoch(nav_processor::NavigationProcessor, role::Union{DriverState,DecodingState}, epoch_index::Int) =
    role.last_end_sample_index <= _epoch_sample_index(nav_processor, epoch_index, ustrip(Hz, role.sampling_frequency))

# ─────────────────────────────────────────────────────────────────────────────
# Walking the groups

# `f(acc, group, args...)` over the slot groups, in order, statically dispatched; a group
# given with its index or its observation buffer (`_indexed`, `_with_signal_observations`) is handed to
# `f` as `group, group_index` or `group, buffer`.
@inline _fold_groups(f::F, acc, ::Tuple{}, args...) where {F} = acc
@inline _fold_groups(f::F, acc, groups::Tuple, args...) where {F} =
    _fold_groups(f, f(acc, _group_args(first(groups))..., args...), Base.tail(groups), args...)
@inline _group_args(group) = (group,)
@inline _group_args(item::Tuple) = item

# The groups with their indices, or with their observation buffers, for `_fold_groups` to
# hand `f` as `group, group_index` or `group, buffer`.
_indexed(groups::Tuple) = map(tuple, groups, ntuple(identity, length(groups)))
_with_signal_observations(groups::Tuple, signal_observations::Tuple) = map(tuple, groups, signal_observations)

# ─────────────────────────────────────────────────────────────────────────────
# Scalar observation collection

# The observations of the group's slots that `is_collected` into its buffer.
function _collect_signal_observations!(acc, group, buffer, is_collected::F) where {F}
    empty!(buffer)
    for slot in group.slots
        is_collected(slot) && push!(buffer, _signal_observation(group.driver_signal, slot))
    end
    acc
end

_is_positioning_ready(slot::VectorTrackingSlot) = slot.in_update && slot.snapshot.positioning_ready

@inline _signal_observation(signal, slot::VectorTrackingSlot) = SignalObservation(;
    decoder = slot.snapshot.decoder,
    signal = signal,
    code_phase = slot.snapshot.code_phase,
    carrier_doppler = slot.snapshot.carrier_doppler,
)

# ─────────────────────────────────────────────────────────────────────────────
# Membership

# Hand `slot` back to its scalar loop; applied on its next integration (`_apply_nav_filter_feedback`).
function _mark_release!(slot::VectorTrackingSlot, reason::VectorTrackingReleaseReason)
    slot.vector_controlled = false
    slot.release_reason = reason
    nothing
end

# A member that missed the epoch sits the update out.
_is_member(slot::VectorTrackingSlot) = slot.in_update && slot.vector_controlled

_is_eligible(slot::VectorTrackingSlot) =
    slot.in_update &&
    has_positioning_data(slot.snapshot.decoder) &&
    is_sat_healthy(slot.snapshot.decoder)

# Admission elevation, above the horizon that releases a member: the hysteresis keeps a
# satellite near the horizon from being admitted and released every update. One degree
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
function _update_membership!(released, group, enu_from_ecef)
    for slot in group.slots
        slot.in_update || continue
        eligible = _is_eligible(slot)
        if eligible && slot.snapshot.measurement_ready
            if slot.vector_controlled || _is_above_admission_mask(group.driver_signal, slot, enu_from_ecef)
                slot.vector_controlled = true
            end
        elseif !eligible && slot.vector_controlled
            _mark_release!(slot, VT_INELIGIBLE)
            released = true
        end
    end
    released
end

# ─────────────────────────────────────────────────────────────────────────────
# Measurement gathering

# One `VectorTrackingObservation` per satellite in the loop, from its measurement; returns the running index.
function _collect_members!(j, group, group_index, nav_processor, Δt_nav)
    layout = nav_processor.nav_filter.layout
    workspace = nav_processor.workspace
    signal = group.driver_signal
    chip_rate = ustrip(Hz, get_chip_rate(signal))
    chip_length = SPEED_OF_LIGHT / chip_rate
    wavelength = SPEED_OF_LIGHT / ustrip(Hz, get_center_frequency(signal))
    clock_bias_index = layout.clock_bias_index_by_group[group_index]
    ifb_index = layout.ifb_index_by_group[group_index]
    for (slot_index, slot) in enumerate(group.slots)
        _is_member(slot) || continue
        # Its loop state at the epoch, as the vector loop sees it: its snapshot's, or, for a
        # satellite this update admitted, that state with the vector interface empty (no
        # discriminators accumulated yet), as `_apply_nav_filter_feedback` will make it.
        loop_state = _add_to_vector_loop(slot.snapshot.loop_state)
        j += 1
        measurement = workspace.measurements[j]
        push!(
            workspace.members,
            VectorTrackingObservation(
                group_index,
                slot_index,
                slot.prn,
                clock_bias_index,
                ifb_index,
                chip_length,
                wavelength,
                chip_rate,
                slot.snapshot.measurement_ready && has_disc_sums(loop_state),
                measurement.transmit_time,
                measurement.position,
                measurement.velocity,
                measurement.satellite_clock_drift,
                calc_mean_code_disc_at_epoch(loop_state, Δt_nav),
                -wavelength *
                (ustrip(Hz, slot.snapshot.carrier_doppler) + calc_mean_fll_disc(loop_state)),
                calc_linear_cn0_floor(slot.snapshot.cn0),
                slot.snapshot.early_late_spacing,
                ustrip(s, slot.snapshot.coherent_integration_time),
            ),
        )
    end
    j
end

# Gather this update's members, their measurements at the epoch and per-member model pieces.
# Returns the ionospheric model and the broadcast time offsets prepared from the same
# observations, as `prepare_observations!` does.
function _gather_members!(nav_processor::NavigationProcessor, groups, Δt_nav)
    workspace = nav_processor.workspace
    _fold_groups(_collect_signal_observations!, nothing, _with_signal_observations(groups, workspace.signal_observations), _is_member)
    prepared = prepare_observations!(
        workspace.measurements,
        workspace.signal_observations;
        approximate_year = nav_processor.approximate_year,
    )
    # The collection pass keeps exactly the decoded, healthy satellites, i.e. every
    # member, so measurements and members align.
    num_rows = length(workspace.measurements)
    num_members = mapreduce(length, +, workspace.signal_observations; init = 0)
    num_rows == num_members || throw(
        ArgumentError(
            "a vector-loop member was not decoded for positioning and healthy; " *
            "its measurement is missing",
        ),
    )
    empty!(workspace.members)
    _fold_groups(_collect_members!, 0, _indexed(groups), nav_processor, Δt_nav)
    members = workspace.members
    resize!(workspace.member_satellite_positions, num_members)
    for j in eachindex(members)
        workspace.member_satellite_positions[j] = members[j].sat_pos
    end
    member_bias_columns!(workspace.member_bias_columns, members, eachindex(members))
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
    workspace = nav_processor.workspace
    members = workspace.members
    measurements = workspace.measurements
    num_members = length(members)
    delays = resize!(workspace.atmospheric_delays, num_members)
    if num_members > 0 &&
       (nav_processor.enable_ionospheric_correction || nav_processor.enable_tropospheric_correction)
        position_and_bias_vector!(workspace.ξ, nav_filter.x, nav_filter.model.idxs)
        predict_atmospheric_delays!(
            delays,
            workspace.ξ,
            measurements,
            nav_processor.enable_ionospheric_correction ? ionospheric_model : IonosphericModel(nothing),
            reference_epoch,
            nav_processor.enable_tropospheric_correction,
        )
    else
        fill!(delays, 0.0)
    end
    measured = resize!(workspace.measured_pseudoranges, num_members)
    for j in eachindex(members)
        correction = get_time_system_offset(measurements[j].time_system, reference_system)
        measured[j] = calc_pseudorange(reference_tow, members[j].transmit_tow, correction) - delays[j]
    end
    measured
end

# ─────────────────────────────────────────────────────────────────────────────
# The measurement update

function _measurement_workspace!(workspace::VectorTrackingWorkspace, num_states, num_measurements)
    cache = workspace.measurement_updates
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

# Fuse the used members' measurements at the filter's predicted `x`; the posterior into its
# `x` and `P`.
function _measurement_update!(nav_processor::NavigationProcessor, Δt_nav)
    nav_filter = nav_processor.nav_filter
    workspace = nav_processor.workspace
    members = workspace.members
    used = workspace.used_member_indices
    measured = workspace.measured_pseudoranges
    constraints = workspace.clock_constraints
    use_rates = nav_filter.config.use_pseudorange_rates
    num_observations = length(used)
    resize!(workspace.used_satellite_positions, num_observations)
    for (k, j) in enumerate(used)
        workspace.used_satellite_positions[k] = members[j].sat_pos
    end
    member_bias_columns!(workspace.used_bias_columns, members, used)
    num_rate_rows = use_rates ? num_observations : 0
    num_measurements = num_observations + num_rate_rows + length(constraints)
    update = _measurement_workspace!(workspace, length(nav_filter.x), num_measurements)
    z = update.z
    R = fill!(update.R, 0.0)
    for (k, j) in enumerate(used)
        member = members[j]
        z[k] = measured[j] + member.code_discriminator * member.chip_length
        R[k, k] = pseudorange_noise_variance(member, Δt_nav)
        if use_rates
            z[num_observations+k] = member.pseudorange_rate
            R[num_observations+k, num_observations+k] = pseudorange_rate_noise_variance(member, Δt_nav)
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
        workspace.ξ_model,
        workspace.used_satellite_positions,
        members,
        used,
        workspace.used_bias_columns,
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
function _predict_measurements!(nav_processor::NavigationProcessor)
    nav_filter = nav_processor.nav_filter
    workspace = nav_processor.workspace
    members = workspace.members
    num_members = length(members)
    idxs = nav_filter.model.idxs
    position_and_bias_vector!(workspace.ξ, nav_filter.x, idxs)
    predicted = resize!(workspace.predicted_pseudoranges, num_members)
    predict_pseudoranges!(predicted, workspace.member_satellite_positions, workspace.ξ, workspace.member_bias_columns)
    rates = resize!(workspace.predicted_pseudorange_rates, num_members)
    residuals = resize!(workspace.residuals, num_members)
    rate_residuals = resize!(workspace.rate_residuals, num_members)
    for (j, member) in enumerate(members)
        rates[j] = predict_pseudorange_rate(nav_filter.x, idxs, member)
        residuals[j], rate_residuals[j] = calc_post_fit_residuals(
            member,
            workspace.measured_pseudoranges[j],
            predicted[j],
            rates[j],
        )
    end
    nothing
end

# Point every member above the horizon at this update's commands; the satellite
# evaluates them where its command takes effect, on its next integration (`_commands_at_application`).
function _assign_commands!(acc, group, group_index, nav_processor)
    workspace = nav_processor.workspace
    for (j, member) in enumerate(workspace.members)
        member.group == group_index && workspace.above_horizon[j] || continue
        slot = group.slots[member.slot_index]
        slot.commanded_update_index = nav_processor.update_index
        slot.member_index = j
    end
    acc
end

# The slot's commands applied at sample `application_sample_index` of a device sampling at `fs`
# (Hz): the code correction, the carrier Doppler ramp from the predicted Dopplers there and one
# update interval later (with the replica's code moved on at its NCO command), and the
# feedback delay (s) after the measurement epoch.
function _commands_at_application(
    nav_processor::NavigationProcessor,
    group::VectorTrackingSlotGroup,
    slot::VectorTrackingSlot,
    fs,
    nco_history,
    application_sample_index::Int64,
)
    # A command applied on the epoch itself may read a rounding error early.
    feedback_delay = (application_sample_index - _epoch_sample_index(nav_processor, nav_processor.last_update_epoch_index, fs)) / fs
    -1 / fs <= feedback_delay <= MAX_FEEDBACK_DELAY_UPDATE_INTERVALS * _update_interval_seconds(nav_processor) || throw(
        ArgumentError(
            "a command must take effect between zero and $(MAX_FEEDBACK_DELAY_UPDATE_INTERVALS) navigation " *
            "update intervals after the epoch whose commands it carries",
        ),
    )
    feedback_delay = max(feedback_delay, 0.0)
    Δt_nav = _update_interval_seconds(nav_processor)
    chip_rate = ustrip(Hz, get_chip_rate(group.driver_signal))
    code_command = ustrip(Hz, calc_mean_nco_command(nco_history, slot.driver.last_end_sample_index, application_sample_index).code_doppler)
    chips = slot.driver.chips_since_epoch +
        _replica_chips(nco_history, slot.driver.last_end_sample_index, application_sample_index, fs * Hz, chip_rate * Hz)
    member = nav_processor.workspace.members[slot.member_index]
    # The code correction removes the range error where the command takes effect (TOW-based,
    # atmosphere-corrected, no discriminator term); the carrier is the predicted Doppler itself,
    # there and one update interval later for the ramp's end.
    predicted_pseudorange, predicted_rate, replica_pseudorange = _predict_after_epoch(nav_processor, slot, feedback_delay, chips)
    code_frequency_correction = calc_code_frequency_correction(
        predicted_pseudorange,
        replica_pseudorange,
        member.chip_rate,
        ustrip(s, nav_processor.nav_filter.model.propagation_interval),
    )
    doppler = predict_carrier_doppler(predicted_rate, member.wavelength)
    _, next_rate, _ = _predict_after_epoch(nav_processor, slot, feedback_delay + Δt_nav, chips + Δt_nav * (chip_rate + code_command))
    next_doppler = predict_carrier_doppler(next_rate, member.wavelength)
    code_frequency_correction, CarrierDopplerRamp(doppler * Hz, (next_doppler - doppler) / Δt_nav * Hz / s, application_sample_index / fs * s), feedback_delay
end

# The slot's member's predicted pseudorange and rate `τ` after the epoch, from the filter's `x`
# propagated by `τ`, and the pseudorange its replica realises there: transmit time
# moved on by the `chips` the replica advances (the satellite clock correction changes
# by picoseconds in `τ`), receive time `τ` after the receive epoch's time of week on the
# count of the reference system (see `_measure_pseudoranges!`), and the epoch's
# atmospheric delay (it moves by millimetres in `τ`).
function _predict_after_epoch(nav_processor::NavigationProcessor, slot::VectorTrackingSlot, τ, chips)
    nav_filter = nav_processor.nav_filter
    workspace = nav_processor.workspace
    j = slot.member_index
    member = workspace.members[j]
    reference_system = _reference_system(nav_filter)
    reference_tow = get_time_of_week(reference_system, nav_filter.reference_epoch)
    idxs = nav_filter.model.idxs
    application_transmit_tow = member.transmit_tow + chips / member.chip_rate
    ephemeris = something(get_broadcast_ephemeris(slot.snapshot.decoder))
    application_orbit = calc_satellite_position_and_velocity(ephemeris, application_transmit_tow)
    application_position = application_orbit.position
    application_velocity = application_orbit.velocity
    propagate_state!(workspace.x_at_application, nav_filter.x, idxs, τ)
    position_and_bias_vector!(workspace.ξ_at_application, workspace.x_at_application, idxs)
    workspace.satellite_position_at_application[1] = application_position
    workspace.bias_columns_at_application.clock_columns[1] = member.clock_bias_index
    workspace.bias_columns_at_application.ifb_columns[1] = member.ifb_index
    predict_pseudoranges!(
        workspace.predicted_pseudorange_at_application,
        workspace.satellite_position_at_application,
        workspace.ξ_at_application,
        workspace.bias_columns_at_application,
    )
    predicted_pseudorange = workspace.predicted_pseudorange_at_application[1]
    user_pos, user_vel, user_clock_drift = nav_filter_states(workspace.x_at_application, idxs)
    # Positive receding, as `predict_pseudorange_rate`.
    predicted_rate = calc_pseudorange_rate(
        user_pos,
        user_vel,
        user_clock_drift,
        application_position,
        application_velocity,
        calc_satellite_clock_drift(ephemeris, application_transmit_tow),
    )
    replica_pseudorange =
        calc_pseudorange(
            reference_tow + τ,
            application_transmit_tow,
            get_time_system_offset(workspace.measurements[j].time_system, reference_system),
        ) -
        workspace.atmospheric_delays[j]
    predicted_pseudorange, predicted_rate, replica_pseudorange
end

# Release the group's members below the horizon; returns whether any was.
function _release_below_horizon!(released, group, group_index, nav_processor)
    workspace = nav_processor.workspace
    for (j, member) in enumerate(workspace.members)
        member.group == group_index && !workspace.above_horizon[j] || continue
        _mark_release!(group.slots[member.slot_index], VT_BELOW_HORIZON)
        released = true
    end
    released
end

# Release every satellite of the group still in the loop, members and those that
# missed the epoch alike, for `reason`; returns whether any was.
function _release_all!(released, group, reason::VectorTrackingReleaseReason)
    for slot in group.slots
        slot.vector_controlled || continue
        _mark_release!(slot, reason)
        released = true
    end
    released
end

function _count_members(n, group)
    for slot in group.slots
        _is_member(slot) && (n += 1)
    end
    n
end

# ─────────────────────────────────────────────────────────────────────────────
# Seeding from a scalar fix

function _add_fix_satellites!(num_added, group, group_index, nav_processor)
    signal_id = nav_processor.nav_filter.layout.signal_id_by_group[group_index]
    observations = nav_processor.pvt.observations
    for slot in group.slots
        slot.in_update && haskey(observations, (signal_id, slot.prn)) || continue
        slot.vector_controlled = true
        num_added += 1
    end
    num_added
end

# Scale the seeded covariance by the fix's geometry (`seed_fix_covariance!`) at the
# seeded position, over the bias layout `calc_pvt` decided for the fix with the epoch's
# broadcast `inter_system_time_offsets`. `used_member_indices` and the bias layout serve as scratch:
# every update decides its layout before anything reads it.
function _seed_fix_covariance!(nav_processor::NavigationProcessor, inter_system_time_offsets)
    nav_filter = nav_processor.nav_filter
    workspace = nav_processor.workspace
    members = workspace.members
    layout = nav_filter.layout
    idxs = nav_filter.model.idxs
    included = empty!(workspace.used_member_indices)
    for j in eachindex(members)
        haskey(nav_processor.pvt.observations, _member_key(layout, members[j])) && push!(included, j)
    end
    isempty(included) && return false
    resize!(workspace.used_satellite_positions, length(included))
    for (k, j) in enumerate(included)
        workspace.used_satellite_positions[k] = members[j].sat_pos
    end
    fix_bias_layout = workspace.bias_layout
    decide_bias_layout!(fix_bias_layout, view(workspace.measurements, included), inter_system_time_offsets) || return false
    H = _design_matrix!(workspace, nav_filter.x, idxs, fit_bias_columns(fix_bias_layout), length(included))
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
        workspace.normal_matrices[size(H, 2)],
        fix_bias_layout,
    )
end

# The design matrix of the first `num_observations` of `workspace.used_satellite_positions` over
# `bias_columns`, linearised at `x`, a row per satellite and then per constraint: a view
# of `workspace.design_matrix`, grown if short.
function _design_matrix!(workspace, x, idxs, bias_columns::BiasColumns, num_observations)
    num_rows = num_observations + length(bias_columns.constraints)
    if size(workspace.design_matrix, 1) < num_rows
        workspace.design_matrix = zeros(2 * num_rows, size(workspace.design_matrix, 2))
    end
    H = view(workspace.design_matrix, 1:num_rows, 1:num_parameters(bias_columns))
    position_and_bias_vector!(workspace.ξ, x, idxs)
    calc_design_matrix!(H, workspace.used_satellite_positions, workspace.ξ, bias_columns)
end

# Switch to vector tracking off the fresh fix in `nav_processor.pvt`: add its satellites, seed the
# filter from it and assign first commands (no measurement update, no accumulator
# reset). A fresh fix has satellites, all in the update (only those enter the scalar
# solve), so there is always a member to seed from. None is in the loop yet:
# slot_generation puts a satellite out of it, and the fallback releases every one in it.
function _seed_nav_filter!(nav_processor::NavigationProcessor, groups, update_interval)
    nav_filter = nav_processor.nav_filter
    workspace = nav_processor.workspace
    model = nav_filter.model
    ensure_nav_filter_propagation_interval!(model, nav_filter.config, update_interval)
    Δt_nav = ustrip(s, model.propagation_interval)
    pvt = nav_processor.pvt
    nav_filter.reference_clock_index = initial_nav_state!(nav_filter.x, nav_filter.P, nav_filter.layout, model.idxs, pvt)
    _fold_groups(_add_fix_satellites!, 0, _indexed(groups), nav_processor)
    prepared = _gather_members!(nav_processor, groups, Δt_nav)
    _seed_fix_covariance!(nav_processor, prepared.inter_system_time_offsets)
    members = workspace.members
    # The fix's epoch: the receive time the filter's clock biases count from, the
    # reference one starting at zero, taken from `pvt.time`.
    reference_epoch = something(pvt.time)
    _measure_pseudoranges!(nav_processor, prepared.ionospheric_model, reference_epoch)
    _predict_measurements!(nav_processor)
    resize!(workspace.above_horizon, length(members))
    fill!(workspace.above_horizon, true)
    _fold_groups(_assign_commands!, 0, _indexed(groups), nav_processor)
    nav_processor.vector_tracking_active = true
    nav_filter.reference_epoch = reference_epoch
    nav_processor.measurement_deficit_duration = 0.0s
    nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# One navigation update

# The latest of `latest` and the first epoch not before any occupied slot's last integration
# end, driver or decoding.
function _latest_epoch_at_or_after(latest, group::VectorTrackingSlotGroup, nav_processor::NavigationProcessor)
    for slot in group.slots
        slot.occupied || continue
        latest = max(latest, _epoch_at_or_after(nav_processor, slot.driver), _epoch_at_or_after(nav_processor, slot.decoding))
    end
    latest
end

# Stale: no integration of the driver, or of the decoding signal, for two update intervals.
_is_stale(nav_processor::NavigationProcessor, slot::VectorTrackingSlot, time) =
    min(_last_end_time(slot.driver), _last_end_time(slot.decoding)) <
    time - 2 * _update_interval_seconds(nav_processor)

# Before an update: mark the slots it reads, clear release reasons, and free stale slots
# (storage kept, members released). Returns whether a member was dropped.
function _prepare_slots!(released, group::VectorTrackingSlotGroup, nav_processor::NavigationProcessor, time)
    for slot in group.slots
        slot.release_reason = VT_NOT_RELEASED
        slot.in_update = slot.occupied && slot.snapshot_epoch_index == nav_processor.pending_epoch_index
        if slot.occupied && !slot.in_update && _is_stale(nav_processor, slot, time)
            slot.occupied = false
            if slot.vector_controlled
                _mark_release!(slot, VT_INELIGIBLE)
                released = true
            end
        end
    end
    released
end

# One navigation update at the pending epoch.
#
#   - Not running: the scalar PVT. A fresh fix seeds the filter, puts its satellites
#     into the vector loop and closes their loops a first time at the seeded state.
#   - Running: one filter update (`_update_nav_filter!`) — admission and release, the members'
#     accumulated discriminators fused, commands for every member. Admitted are
#     satellites decoded, healthy, in lock and `ADMISSION_ELEVATION` above the
#     horizon; members out of lock stay in the loop unmeasured. After
#     `measurement_deficit_timeout` of unsolvable epochs, or with no member left, every
#     satellite in the loop is released (those that missed the epoch too) and the next
#     update solves the scalar PVT.
function _run_navigation_update!(nav_processor::NavigationProcessor, time)
    groups = nav_processor.groups
    workspace = nav_processor.workspace
    epoch_index = nav_processor.pending_epoch_index
    update_interval =
        nav_processor.last_update_epoch_index == typemin(Int) ? nav_processor.update_interval :
        (epoch_index - nav_processor.last_update_epoch_index) * nav_processor.update_interval
    nav_processor.update_index += 1
    dropped = _fold_groups(_prepare_slots!, false, groups, nav_processor, time)
    seeded = released = fell_back = false
    if nav_processor.vector_tracking_active
        released, fell_back = _update_nav_filter!(nav_processor, groups, update_interval)
    else
        # Scalar solve: no per-member report.
        empty_keeping_capacity!(nav_processor.vector_tracking_members)
        _fold_groups(_collect_signal_observations!, nothing, _with_signal_observations(groups, workspace.signal_observations), _is_positioning_ready)
        result = calc_pvt!(
            nav_processor.pvt,
            nav_processor.pvt_workspace,
            workspace.signal_observations,
            nav_processor.pvt;
            approximate_year = nav_processor.approximate_year,
            enable_ionospheric_correction = nav_processor.enable_ionospheric_correction,
            enable_tropospheric_correction = nav_processor.enable_tropospheric_correction,
        )
        if result.status == PositionVelocityTime.solved
            nav_processor.pvt = result.solution
            seeded = true
            _seed_nav_filter!(nav_processor, groups, update_interval)
        end
    end
    num_members = _fold_groups(_count_members, 0, groups)
    nav_processor.status = VectorTrackingStatus(
        nav_processor.vector_tracking_active,
        nav_processor.vector_tracking_active || fell_back ? calc_position_std(nav_processor) * m : NaN * m,
        nav_processor.vector_tracking_active || fell_back ? calc_clock_bias_std(nav_processor) * m : NaN * m,
        nav_processor.measurement_deficit_duration,
        num_members,
        seeded,
        fell_back,
        released || dropped,
        nav_processor.vector_tracking_active && !seeded || fell_back,
    )
    nav_processor.last_update_epoch_index = epoch_index
    # The epoch to snapshot next: the next one, or, if a satellite is past it already (see
    # `_run_navigation_update_if_due!`), the first no satellite is past. The update after skipped
    # epochs is longer, as after a gap. The host steps every satellite through each chunk, which
    # is shorter than an update interval, so in step no satellite is past the next epoch.
    nav_processor.pending_epoch_index =
        _fold_groups(_latest_epoch_at_or_after, epoch_index + 1, nav_processor.groups, nav_processor)
    nav_processor.num_snapshots = 0
    nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# One update

# One update of the navigation filter, a linear Kalman time update and an unscented
# measurement update: manage membership, predict (time update), fuse the accumulated
# discriminators as pseudorange (and, for VDFLL, rate) measurements (measurement update),
# and assign the NCO commands or fall back to scalar tracking. Returns
# `(released, fell_back)`.
function _update_nav_filter!(nav_processor::NavigationProcessor, groups, update_interval)
    nav_filter = nav_processor.nav_filter
    config = nav_filter.config
    workspace = nav_processor.workspace
    model = nav_filter.model
    idxs = model.idxs
    Δt_nav = ustrip(s, update_interval)
    ensure_nav_filter_propagation_interval!(model, config, update_interval)
    # Advance the receive epoch: its time of week crosses into the next week by itself.
    reference_epoch = nav_filter.reference_epoch + Δt_nav

    # 1. Membership: admission and the elevation mask, judged at the update's starting
    #    solution.
    enu_from_ecef = ENUfromECEF(_ecef(first(nav_filter_states(nav_filter.x, idxs))), wgs84)
    released =
        _fold_groups(_update_membership!, false, groups, enu_from_ecef)
    prepared = _gather_members!(nav_processor, groups, Δt_nav)
    members = workspace.members
    num_members = length(members)

    # A member below the horizon is released, neither measured nor commanded.
    above_horizon = resize!(workspace.above_horizon, num_members)
    num_above_horizon = 0
    for (j, member) in enumerate(members)
        above_horizon[j] = first(calc_elevation_and_azimuth(enu_from_ecef, member.sat_pos)) >= 0
        above_horizon[j] && (num_above_horizon += 1)
    end
    released = _fold_groups(_release_below_horizon!, released, _indexed(groups), nav_processor)

    # 2. Time update (prediction) to the measurement epoch: x⁻ = F·x⁺, P⁻ = F·P⁺·Fᵀ + Q.
    time_update!(workspace.time_update, nav_filter.x, nav_filter.P, model.F, model.Q)

    # 3. Measurements: each member's pseudorange from its snapshot, corrected by its
    #    accumulated code discriminator in the update below.
    _measure_pseudoranges!(nav_processor, prepared.ionospheric_model, reference_epoch)

    # 4. Used measurements: members above the horizon with an available signal (an
    #    obscured one carries no information), and the clock and bias states they determine.
    #    No measurement gate selects among them yet: planned filtering (an innovation or
    #    NIS gate, RAIM/FDE, C/N₀ or lock thresholds, health and URA flags, multipath and
    #    NLOS detection, satellite subset selection) would take these as its input,
    #    `candidates`, and keep `used` for its output.
    used = empty!(workspace.used_member_indices)
    for (j, member) in enumerate(members)
        member.available && above_horizon[j] && push!(used, j)
    end
    # Decided every update, without used members too (unsolvable, no constraint), so nothing
    # below reads an earlier update's decision.
    used_observations = view(workspace.measurements, used)
    solvable = decide_bias_layout!(workspace.bias_layout, used_observations, prepared.inter_system_time_offsets)
    clock_constraints!(workspace.clock_constraints, nav_filter.layout, workspace.bias_layout)
    # 5. Measurement update (unscented): the predicted measurements from sigma points
    #    through h, then x⁺ = x⁻ + K·(z − ẑ) and P⁺.
    #    No innovation gate: the measurement deficit watchdog catches divergence, and a gate could
    #    release a healthy satellite whose large but explained innovation the update would
    #    absorb (e.g. a constellation's first measurements).
    isempty(used) || _measurement_update!(nav_processor, Δt_nav)

    # 6. Loop closure: every member's pseudorange and rate predicted at the updated state,
    #    which the NCO commands follow.
    _predict_measurements!(nav_processor)

    # 7. Measurement deficit watchdog (see `VectorTrackingStatus`): solvability only; the solution's
    # certainty is reported as `position_std`, not policed.
    measurement_deficit_duration =
        solvable ?
        max(0.0s, nav_processor.measurement_deficit_duration - update_interval / 2) :
        nav_processor.measurement_deficit_duration + update_interval

    # 8. The NCO commands, or the fallback to scalar tracking.
    fell_back = measurement_deficit_duration > config.measurement_deficit_timeout || num_above_horizon == 0
    if fell_back
        released = _fold_groups(_release_all!, released, groups, VT_FALLBACK)
        nav_processor.vector_tracking_active = false
    else
        _fold_groups(_assign_commands!, 0, _indexed(groups), nav_processor)
    end

    # 9. The solution, dated by the reference system's clock.
    nav_filter.reference_clock_index =
        select_reference_clock_index(nav_filter.layout, workspace.bias_layout, nav_filter.reference_clock_index)
    nav_filter.reference_epoch = reference_epoch
    nav_processor.measurement_deficit_duration = measurement_deficit_duration
    _write_solution!(nav_processor)
    released, fell_back
end

_ecef(v) = ECEF(v[1], v[2], v[3])

# ─────────────────────────────────────────────────────────────────────────────
# The solution

function _write_solution!(nav_processor::NavigationProcessor)
    nav_filter = nav_processor.nav_filter
    workspace = nav_processor.workspace
    members = workspace.members
    layout = nav_filter.layout
    idxs = nav_filter.model.idxs
    x = nav_filter.x
    user_pos, user_vel, user_clock_drift = nav_filter_states(x, idxs)
    position = _ecef(user_pos)
    velocity = _ecef(user_vel)
    reference_clock_bias = x[idxs.clock_biases[nav_filter.reference_clock_index]]
    used = workspace.used_member_indices

    # DOP of the measured satellites over the update's bias layout, as `calc_pvt` computes
    # it, or `nothing` without measurements or for a rank-deficient design (`calc_dop!`'s
    # all-`NaN` sentinel, which `calc_pvt` never emits either).
    dop = nothing
    if !isempty(used)
        update_layout = workspace.bias_layout
        H = _design_matrix!(workspace, x, idxs, dop_bias_columns(update_layout), length(used))
        reference_column = clock_column(
            update_layout,
            view(workspace.measurements, used),
            layout.time_systems[nav_filter.reference_clock_index],
        )
        candidate = calc_dop!(workspace.normal_matrices[size(H, 2)], H, position, reference_column)
        isnan(candidate.GDOP) || (dop = candidate)
    end

    # `pvt.observations` holds the measured members only, as `calc_pvt` does, so it matches
    # `pvt.dop`; coasted members are in `get_vector_tracking_members`.
    solution = empty_keeping_capacity!(nav_processor.pvt)
    observations = solution.observations
    for j in used
        set!(observations, _member_key(layout, members[j]), _member_info(workspace, j))
    end
    satellite_reports = empty_keeping_capacity!(nav_processor.vector_tracking_members)
    for j in eachindex(members)
        set!(satellite_reports, _member_key(layout, members[j]), _member_info(workspace, j))
    end

    # Only measured biases, as in `calc_pvt`: an unmeasured one coasts on process noise
    # (or was never seeded) and would be indistinguishable from a measured one.
    inter_system_biases = solution.inter_system_biases
    for (index, time_system) in enumerate(layout.time_systems)
        index == nav_filter.reference_clock_index && continue
        _is_measured(member -> member.clock_bias_index == index, members, used) ||
            continue
        set!(inter_system_biases, time_system, (x[idxs.clock_biases[index]] - reference_clock_bias) * m)
    end
    inter_frequency_biases = solution.inter_frequency_biases
    for (index, band) in enumerate(layout.extra_bands)
        _is_measured(member -> member.ifb_index == index, members, used) || continue
        set!(
            inter_frequency_biases,
            band,
            InterFrequencyBias(x[idxs.ifb[index]] * m, layout.reference_bands[index]),
        )
    end

    # The broadcast offsets the measurement update took as pseudo-measurements, as `calc_pvt`
    # reports the ones it takes as given; here they hold to `BROADCAST_INTER_SYSTEM_BIAS_STD` rather than
    # exactly. Without used members there was no update.
    applied_biases = solution.applied_broadcast_inter_system_biases
    isempty(used) || applied_broadcast_inter_system_biases!(applied_biases, workspace.bias_layout)

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

function _is_measured(predicate::F, members, used) where {F}
    for j in used
        predicate(members[j]) && return true
    end
    false
end
_member_key(layout::NavFilterLayout, member::VectorTrackingObservation) =
    (layout.signal_id_by_group[member.group], member.prn)

_member_info(workspace::VectorTrackingWorkspace, j) = ObservationResiduals(
    workspace.members[j].sat_pos,
    workspace.members[j].transmit_tow * s,
    workspace.residuals[j],
    workspace.rate_residuals[j],
)

