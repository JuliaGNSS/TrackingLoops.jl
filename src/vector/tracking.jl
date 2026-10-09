# ─────────────────────────────────────────────────────────────────────────────
# Vector tracking: the navigation engine that closes all loops
#
# One `VectorNavigation` is shared by every satellite a `VectorPLLAndDLL` steps. Each
# satellite's slot is filled by its records (bit clock, decoder, C/N₀) and, at every
# navigation epoch, by a snapshot of the replica and the accumulated discriminators.
# Once every satellite has reached the epoch the cycle runs: a scalar PVT solve until
# its first fix seeds the filter, one filter iteration after. The cycle leaves its
# decisions (admission, release, corrections) in the slots for each satellite to take
# up on its next record. Nothing here knows a correlator, so software and hardware
# loops step the same estimator.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Walking the groups

# Fold `f(acc, group, g, buffer, args...)` over the groups in order, `g` the group's
# index and `buffer` its entry of `buffers`. Recursing over the tuples keeps every
# call statically dispatched, however heterogeneous the groups.
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

_signal_groups(groups, buffers) =
    map((group, buffer) -> SignalGroup(group.signal, buffer), groups, buffers)

# ─────────────────────────────────────────────────────────────────────────────
# The scalar solve

function _collect_pvt_ready!(acc, group, g, buffer)
    empty!(buffer)
    for sat in group.slots
        sat.active && sat.pvt_ready || continue
        push!(buffer, _satellite_state(group.signal, sat))
    end
    acc
end

@inline _satellite_state(signal, sat::VTSlot) = SatelliteState(;
    decoder = sat.decoder,
    system = signal,
    code_phase = sat.code_phase,
    carrier_doppler = sat.carrier_doppler,
    carrier_phase = sat.carrier_phase,
)

# The scalar PVT over the `pvt_ready` satellites: a fresh fix is a new object, a
# failed epoch returns the old solution.
function _solve_scalar_pvt!(vt::VectorNavigation, groups)
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
    for sat in group.slots
        sat.release_reason = VT_NOT_RELEASED
    end
    acc
end

# Hand `sat` back to its scalar loop; taken up on its next record (`_take_up_cycle`).
function _release!(sat::VTSlot, reason::VTReleaseReason)
    sat.estimator_state = _disable_vector_tracking(sat.estimator_state)
    sat.release_reason = reason
    nothing
end

# A member that missed the epoch sits the cycle out.
_is_member(sat::VTSlot) = sat.active && sat.estimator_state.vt_on

_is_eligible(sat::VTSlot) =
    sat.active &&
    is_decoding_completed_for_positioning(sat.decoder) &&
    is_sat_healthy(sat.decoder)

# Admission elevation, above the horizon that releases a member: the hysteresis keeps a
# satellite near the horizon from being admitted and released every cycle. One degree
# is a few minutes of a rising satellite.
const ADMISSION_ELEVATION = deg2rad(1.0)

# Evaluated at the satellite's transmit time.
function _is_above_admission_mask(signal, sat::VTSlot, enu_from_ecef)
    orbit = calc_satellite_position_and_velocity(_satellite_state(signal, sat))
    get_sat_enu(enu_from_ecef, _ecef(get_sat_position(orbit))).ϕ >= ADMISSION_ELEVATION
end

# Admit eligible satellites that are in lock and above the admission mask at the
# filter's position, and release members no longer eligible. Returns whether any was
# released. A member out of lock stays in the loop, unmeasured, and keeps receiving
# corrections.
#
# NOTE: admission is gated on `in_lock`, not a stricter ranging-ready flag, although a
# satellite whose code phase is still tens of metres out may then enter the filter.
# `in_lock` also decides the members' measurement eligibility, so a stricter gate would
# withhold discriminators from existing members; the two uses need separating first.
function _update_membership!(released, group, g, buffer, enu_from_ecef)
    for sat in group.slots
        sat.active || continue
        eligible = _is_eligible(sat)
        state = sat.estimator_state
        if eligible && sat.in_lock
            if state.vt_on || _is_above_admission_mask(group.signal, sat, enu_from_ecef)
                sat.estimator_state = _enable_vector_tracking(state)
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

function _collect_member_states!(acc, group, g, buffer)
    empty!(buffer)
    for sat in group.slots
        _is_member(sat) && push!(buffer, _satellite_state(group.signal, sat))
    end
    acc
end

# One `VTMember` per satellite in the loop, from its row; returns the running index.
function _collect_members!(j, group, g, buffer, vt, T)
    layout = vt.layout
    buffers = vt.buffers
    signal = group.signal
    code_frequency = ustrip(Hz, get_code_frequency(signal))
    chip_length = SPEED_OF_LIGHT / code_frequency
    wavelength = SPEED_OF_LIGHT / ustrip(Hz, get_center_frequency(signal))
    clock_bias_index = layout.clock_bias_index_by_group[g]
    ifb_index = layout.ifb_index_by_group[g]
    for (slot, sat) in enumerate(group.slots)
        _is_member(sat) || continue
        state = sat.estimator_state
        j += 1
        row = buffers.rows[j]
        code_discriminator,
        carrier_discriminator,
        code_variance,
        rate_variance,
        has_code,
        has_rate = _member_measurements(sat, state, T, chip_length, wavelength)
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
                sat.in_lock && has_code,
                has_rate,
                row.time,
                row.time - row.count_offset_to_gpst,
                row.position,
                row.velocity,
                row.clock_drift,
                wavelength * ustrip(Hz, sat.carrier_doppler),
                code_discriminator,
                carrier_discriminator,
                code_variance,
                rate_variance,
                linear_cn0_floor(sat.cn0_dbhz),
                sat.early_late_spacing,
                ustrip(s, sat.coherent_integration_time),
                row.time_offsets,
            ),
        )
    end
    j
end

# This cycle's fused code (chips) and rate (Hz) measurements of slot `sat`, their
# variances (m², m²/s²) and whether each exists:
# `(code, rate, code_variance, rate_variance, has_code, has_rate)`.
#
# Each signal's mean reading is weighted by its inverse variance, from its own C/N₀,
# coherent integration time, tap spacing and the span `n·T_coh` its `n` readings cover,
# so a signal read for part of a cycle only (just admitted, group delay just set)
# weighs that much less. Every reading counts at the signal's latest coherent
# integration time: in the cycle where its records change length (at bit sync, say)
# that cycle's variance is off by up to the ratio of the two lengths. The fused
# variance is the inverse of the summed weights. Passenger DLL readings are already
# referred to the driver's code phase and moved to the epoch like the driver's: one
# replica steers them all. Only thermal noise averages down; orbit, clock and
# atmosphere are common to all signals.
#
# A signal without a reading is left out, not entered as zero. No code reading of any
# signal withholds the member from the update, no rate reading only its rate row.
# Without passenger readings the result is the driver's own, bit for bit.
function _member_measurements(
    sat::VTSlot,
    state::SatVectorPLLAndDLL,
    T,
    chip_length,
    wavelength,
)
    cn0 = linear_cn0_floor(sat.cn0_dbhz)
    coherent_integration_time = ustrip(s, sat.coherent_integration_time)
    code_variance =
        sat.dll_variance_factor * _code_variance(
            cn0,
            coherent_integration_time,
            sat.early_late_spacing,
            chip_length,
            first(state.code_discr_acc),
        )
    rate_variance = _rate_variance(
        cn0,
        coherent_integration_time,
        wavelength,
        first(state.carrier_discr_acc),
    )
    has_code = has_accumulated_code_discriminator(state)
    has_rate = has_accumulated_carrier_discriminator(state)
    code = accumulated_code_discriminator(state, T)
    rate = accumulated_carrier_discriminator(state)
    readings = state.passenger_readings
    any(r -> first(r.code) > 0 || first(r.carrier) > 0, readings) ||
        return code, rate, code_variance, rate_variance, has_code, has_rate
    code_weight = has_code ? inv(code_variance) : 0.0
    rate_weight = has_rate ? inv(rate_variance) : 0.0
    code_sum = code_weight * code
    rate_sum = rate_weight * rate
    advance = code_phase_advance(state, T)
    for (p, r) in zip(sat.passengers, readings)
        p_cn0 = linear_cn0_floor(_passenger_cn0_dbhz(p))
        p_coherent_integration_time = ustrip(s, p.integration_time)
        count, discr_sum = r.code
        if count > 0
            weight = inv(
                p.dll_variance_factor * _code_variance(
                    p_cn0,
                    p_coherent_integration_time,
                    p.early_late_spacing,
                    chip_length,
                    count,
                ),
            )
            code_weight += weight
            code_sum += weight * (-discr_sum / count + advance)
        end
        count, frequency_sum = r.carrier
        if count > 0
            weight =
                inv(_rate_variance(p_cn0, p_coherent_integration_time, wavelength, count))
            rate_weight += weight
            rate_sum += weight * ustrip(Hz, frequency_sum) / count
        end
    end
    has_code = code_weight > 0
    has_rate = rate_weight > 0
    (
        has_code ? code_sum / code_weight : 0.0,
        has_rate ? rate_sum / rate_weight : 0.0,
        has_code ? inv(code_weight) : Inf,
        has_rate ? inv(rate_weight) : Inf,
        has_code,
        has_rate,
    )
end

# Variance of `count` readings of `coherent_integration_time` each; `Inf` for none.
function _code_variance(cn0, coherent_integration_time, d, chip_length, count)
    count > 0 || return Inf
    span = count * coherent_integration_time
    _pseudorange_noise_variance(cn0, coherent_integration_time, d, chip_length, span)
end
function _rate_variance(cn0, coherent_integration_time, wavelength, count)
    count > 0 || return Inf
    span = count * coherent_integration_time
    _pseudorange_rate_noise_variance(cn0, coherent_integration_time, wavelength, span)
end

# Gather this cycle's members, their rows at the epoch and per-member model pieces.
# Returns the ionospheric correction the rows select.
function _gather_members!(vt::VectorNavigation, groups, T)
    buffers = vt.buffers
    _fold_groups(_collect_member_states!, nothing, groups, buffers.states, 1)
    ionospheric_correction = collect_measurement_rows!(
        buffers.rows,
        _signal_groups(groups, buffers.states);
        approximate_year = vt.approximate_year,
    )
    # The collection pass keeps exactly the decoded, healthy satellites, i.e. every
    # member, so rows and members align.
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

# Every member's atmosphere-corrected pseudorange against `reference_tow`, the delays
# predicted at `vt.x`.
function _measure_pseudoranges!(vt::VectorNavigation, ionospheric_correction, reference_tow)
    buffers = vt.buffers
    members = buffers.members
    rows = buffers.rows
    num_members = length(members)
    delays = resize!(buffers.delays, num_members)
    if num_members > 0 &&
       (vt.enable_ionospheric_correction || vt.enable_tropospheric_correction)
        # Day of year for the Niell mapping, as `calc_pvt` derives it. Any member dates
        # the epoch (time systems differ by at most 14 s, BDT, against a one-year
        # period), so the first serves.
        doy = day_of_year(rows[1].system_start_time, rows[1].week, reference_tow)
        position_and_bias_vector!(buffers.ξ, vt.x, vt.model.idxs)
        predict_atmospheric_delays!(
            delays,
            buffers.ξ,
            rows,
            IonosphericModel(
                vt.enable_ionospheric_correction ? ionospheric_correction : nothing,
            ),
            reference_tow,
            doy,
            vt.enable_tropospheric_correction,
        )
    else
        fill!(delays, 0.0)
    end
    measured = resize!(buffers.measured_pseudoranges, num_members)
    for j in eachindex(members)
        measured[j] =
            pseudorange_from_tows(reference_tow, members[j].time_gpst_count) - delays[j]
    end
    measured
end

# ─────────────────────────────────────────────────────────────────────────────
# The measurement update

function _measurement_buffers!(buffers::VTBuffers, num_states, num_measurements)
    cache = buffers.measurement_updates
    if num_measurements > length(cache)
        # Not isbits: fill the new entries, or they stay `#undef`.
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

# Fuse the candidates' measurements at the predicted `vt.x`; posterior into `vt.x`,
# `vt.P`.
function _measurement_update!(vt::VectorNavigation, T)
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
    rate_rows = empty!(buffers.rate_rows)
    if use_rates
        for (k, j) in enumerate(candidates)
            members[j].rate_available && push!(rate_rows, k)
        end
    end
    num_rate_rows = length(rate_rows)
    num_measurements = num_sats + num_rate_rows + length(constraints)
    update = _measurement_buffers!(buffers, length(vt.x), num_measurements)
    z = update.z
    R = fill!(update.R, 0.0)
    for (k, j) in enumerate(candidates)
        member = members[j]
        z[k] = measured[j] + member.code_discriminator * member.chip_length
        R[k, k] = member.code_variance
    end
    for (i, k) in enumerate(rate_rows)
        member = members[candidates[k]]
        z[num_sats+i] =
            member.pseudorange_rate + member.carrier_discriminator * member.wavelength
        R[num_sats+i, num_sats+i] = member.rate_variance
    end
    # Each clock collapse is one extra measurement row after the satellite rows.
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
        rate_rows,
        constraints,
    )
    measurement_update!(update.intermediate, vt.x, vt.P, z, h!, R)
    nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Loop closure

# Every member's predictions at the updated `vt.x`, and post-fit residuals.
function _predict_members!(vt::VectorNavigation)
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

# Point every active member at this cycle's corrections; the satellite evaluates them
# where its command lands, on its next record (`_correction_at_landing`).
function _close_loops!(acc, group, g, buffer, vt)
    buffers = vt.buffers
    for (j, member) in enumerate(buffers.members)
        member.group == g && buffers.active[j] || continue
        sat = group.slots[member.slot]
        sat.correction_cycle = vt.cycle_id
        sat.member_index = j
    end
    acc
end

# The NCO corrections (Hz) of member `j`, evaluated where its command lands, `τ` after
# the epoch (see `_predict_at_landing`). The code correction removes the range error
# (TOW-based, atmosphere-corrected, no discriminator term), the carrier correction the
# rate error against the replica's Doppler at landing, `carrier_doppler` (Hz; no FLL
# term).
function _member_corrections(
    vt::VectorNavigation,
    sat::VTSlot,
    j,
    τ,
    chips,
    carrier_doppler,
)
    member = vt.buffers.members[j]
    predicted_pseudorange, predicted_rate, measured_pseudorange =
        _predict_at_landing(vt, sat, member, j, ustrip(s, vt.reference_time), τ, chips)
    measured_rate = member.wavelength * carrier_doppler
    code_update = nco_code_correction(
        predicted_pseudorange,
        measured_pseudorange,
        member.code_frequency,
        vt.cycle_integration_time,
    )
    carrier_update =
        nco_carrier_correction(predicted_rate, measured_rate, member.wavelength)
    code_update, carrier_update
end

# Member `j`'s predicted pseudorange and rate at landing, `τ` after the epoch, from
# `vt.x` propagated by `τ`, and the pseudorange its replica realises there: transmit
# time moved on by the `chips` the replica advances (the satellite clock correction
# changes by picoseconds in `τ`), receive time `reference_tow + τ`, and the epoch's
# atmospheric delay (it moves by millimetres in `τ`).
function _predict_at_landing(
    vt::VectorNavigation,
    sat::VTSlot,
    member::VTMember,
    j,
    reference_tow,
    τ,
    chips,
)
    buffers = vt.buffers
    idxs = vt.model.idxs
    landing_time = member.time + chips / member.code_frequency
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
    predicted_pseudorange, predicted_rate, measured_pseudorange
end

# Release the group's members whose `active` flag equals `which`; returns whether any
# was.
function _release_members!(
    released,
    group,
    g,
    buffer,
    vt,
    which::Bool,
    reason::VTReleaseReason,
)
    buffers = vt.buffers
    for (j, member) in enumerate(buffers.members)
        member.group == g && buffers.active[j] == which || continue
        _release!(group.slots[member.slot], reason)
        released = true
    end
    released
end

function _count_members(n, group, g, buffer)
    for sat in group.slots
        _is_member(sat) && (n += 1)
    end
    n
end

# ─────────────────────────────────────────────────────────────────────────────
# Seeding from a scalar fix

function _enable_fix_satellites!(num_enabled, group, g, buffer, vt)
    signal_id = vt.layout.signal_id_by_group[g]
    sats = vt.pvt.sats
    for sat in group.slots
        sat.active && haskey(sats, (signal_id, sat.prn)) || continue
        sat.estimator_state = _enable_vector_tracking(sat.estimator_state)
        num_enabled += 1
    end
    num_enabled
end

# Members kept by a host across a fallback carry the old filter's accumulators and
# corrections when a fix seeds the loop: release the ineligible ones (no measurement
# row), restart the others as if joining. Returns whether any was released.
function _restart_stale_members!(released, group, g, buffer, vt)
    for sat in group.slots
        _is_member(sat) || continue
        state = sat.estimator_state
        if _is_eligible(sat)
            sat.estimator_state = _enable_vector_tracking(_disable_vector_tracking(state))
            sat.restart_cycle = vt.cycle_id
        else
            _release!(sat, VT_INELIGIBLE)
            released = true
        end
    end
    released
end

# Scale the seeded covariance by the fix's geometry (`seed_fix_covariance!`) at the
# seeded position. `candidates` serves as scratch; every cycle rebuilds it.
function _seed_fix_covariance!(vt::VectorNavigation)
    buffers = vt.buffers
    members = buffers.members
    layout = vt.layout
    idxs = vt.model.idxs
    included = empty!(buffers.candidates)
    for j in eachindex(members)
        haskey(vt.pvt.sats, _member_key(layout, members[j])) && push!(included, j)
    end
    isempty(included) && return false
    resize!(buffers.candidate_positions, length(included))
    for (k, j) in enumerate(included)
        buffers.candidate_positions[k] = members[j].sat_position
    end
    columns, _ = dense_bias_columns!(
        buffers.dense_clock_indices,
        buffers.dense_ifb_indices,
        buffers.clock_used,
        buffers.ifb_used,
        members,
        included,
        vt.primary_clock_index,
    )
    num_columns = 3 + columns.num_clock_biases + columns.num_ifb
    length(included) < num_columns && return false
    if size(buffers.design_matrix, 1) < length(included)
        buffers.design_matrix = zeros(2 * length(included), size(buffers.design_matrix, 2))
    end
    H = view(buffers.design_matrix, 1:length(included), 1:num_columns)
    position_and_bias_vector!(buffers.ξ, vt.x, idxs)
    calc_H!(H, buffers.candidate_positions, buffers.ξ, columns)
    seed_fix_covariance!(
        vt.P,
        idxs,
        layout,
        vt.pvt,
        vt.primary_clock_index,
        H,
        buffers.normal_matrices[num_columns],
        buffers.clock_used,
        buffers.ifb_used,
    )
end

# Switch to vector tracking off the fresh fix in `vt.pvt`: admit its satellites, seed the
# filter from it and close the loops once (no measurement update, no accumulator reset).
# The fix's satellites are all active slots, so there is always a member to seed from.
# Returns whether a stale member was released.
function _seed!(vt::VectorNavigation, groups, cycle_time)
    buffers = vt.buffers
    model = vt.model
    ensure_nav_filter_integration_time!(model, vt.config, cycle_time)
    T = ustrip(s, model.integration_time)
    pvt = vt.pvt
    vt.primary_clock_index = initial_nav_state!(vt.x, vt.P, vt.layout, model.idxs, pvt)
    released = _fold_groups(_restart_stale_members!, false, groups, buffers.states, 1, vt)
    _fold_groups(_enable_fix_satellites!, 0, groups, buffers.states, 1, vt)
    ionospheric_correction = _gather_members!(vt, groups, T)
    _seed_fix_covariance!(vt)
    members = buffers.members
    # The latest transmit time corrected by the fix's clock bias, as `calc_pvt` stamps it.
    latest = -Inf
    for member in members
        latest = max(latest, member.time_gpst_count)
    end
    reference_tow = latest - ustrip(m, pvt.time_correction) / SPEED_OF_LIGHT
    _measure_pseudoranges!(vt, ionospheric_correction, reference_tow)
    _predict_members!(vt)
    resize!(buffers.active, length(members))
    fill!(buffers.active, true)
    vt.cycle_integration_time = T
    _fold_groups(_close_loops!, 0, groups, buffers.states, 1, vt)
    vt.running = true
    vt.reference_time = reference_tow * s
    vt.time_with_insufficient_meas = 0.0s
    # Re-read: a cached offset would be a week stale after a rollover under the scalar
    # solve, which no running cycle saw.
    vt.time_epoch_offset = _primary_time_epoch_offset(vt, eachindex(members))
    released
end

# ─────────────────────────────────────────────────────────────────────────────
# One cycle

# One filter cycle: predict, fuse the accumulated discriminators as pseudorange (and,
# for VDFLL, rate) measurements, close the loops and manage membership. Returns
# `(released, fell_back)`.
function _run_cycle!(vt::VectorNavigation, groups, cycle_time)
    config = vt.config
    buffers = vt.buffers
    model = vt.model
    idxs = model.idxs
    T = ustrip(s, cycle_time)
    ensure_nav_filter_integration_time!(model, config, cycle_time)
    # Advance the receive TOW, wrapping at the week so it stays a valid seconds-of-week.
    # The dropped week goes into the cached epoch offset (`rolled_over_time_epoch_offset`),
    # the only other place the absolute epoch is held.
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

    time_update!(buffers.time_update, vt.x, vt.P, model.F, model.Q)
    _measure_pseudoranges!(vt, ionospheric_correction, reference_tow)

    # Candidates: members with an available signal (an obscured one carries no
    # information).
    candidates = empty!(buffers.candidates)
    for (j, member) in enumerate(members)
        member.available && push!(candidates, j)
    end
    # The bare floor (position and one clock from four satellites), so an epoch without
    # candidates counts as unsolvable.
    observability = BiasObservability(4, 4, 0)
    # No innovation gate: the observability watchdog catches divergence, and a gate
    # could release a healthy satellite whose large but explained innovation the update
    # would absorb (e.g. a constellation's first measurements).
    if !isempty(candidates)
        observability = assess_bias_observability!(
            buffers.observability,
            vt.layout,
            members,
            candidates,
        )
        _measurement_update!(vt, T)
    end
    num_included = length(candidates)

    _predict_members!(vt)

    # Starvation watchdog (see `VTStatus`): solvability only; the solution's certainty is
    # reported as `position_std`, not policed.
    time_with_insufficient_meas =
        is_epoch_solvable(observability, num_included) ?
        max(0.0s, vt.time_with_insufficient_meas - cycle_time / 2) :
        vt.time_with_insufficient_meas + cycle_time

    # Elevation mask at the updated position, before the corrections so every remaining
    # member gets one.
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

    fell_back =
        time_with_insufficient_meas > config.insufficient_meas_timeout || num_active == 0
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
        vt.cycle_integration_time = T
        _fold_groups(_close_loops!, 0, groups, buffers.states, 1, vt)
    end

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

_resolve_time_epoch_offset(vt::VectorNavigation) =
    isnothing(vt.time_epoch_offset) ?
    _primary_time_epoch_offset(vt, vt.buffers.candidates) : vt.time_epoch_offset

# From the first primary-system member among `indices`; `nothing` if none.
function _primary_time_epoch_offset(vt::VectorNavigation, indices)
    members = vt.buffers.members
    for j in indices
        members[j].clock_bias_index == vt.primary_clock_index &&
            return time_epoch_offset(vt.buffers.rows[j])
    end
    nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# The solution

function _write_solution!(vt::VectorNavigation, groups)
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

    # DOP of the measured satellites, or `nothing` without measurements or for a
    # rank-deficient design (`calc_DOP!`'s all-`-1` sentinel, which `calc_pvt` never
    # emits either).
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
        if size(buffers.design_matrix, 1) < length(included)
            buffers.design_matrix =
                zeros(2 * length(included), size(buffers.design_matrix, 2))
        end
        H = view(buffers.design_matrix, 1:length(included), 1:num_columns)
        position_and_bias_vector!(buffers.ξ, x, idxs)
        calc_H!(H, buffers.candidate_positions, buffers.ξ, columns)
        candidate =
            calc_DOP!(buffers.normal_matrices[num_columns], H, position, primary_column)
        candidate.GDOP < 0 || (dop = candidate)
    end

    # `pvt.sats` holds the measured members only, as `calc_pvt` does, so it matches
    # `pvt.dop`; coasted members are in `member_sats`.
    solution = empty_keeping_capacity!(vt.pvt)
    sats = solution.sats
    for j in included
        set!(sats, _member_key(layout, members[j]), _member_info(buffers, j))
    end
    member_sats = empty_keeping_capacity!(vt.member_sats)
    for j in eachindex(members)
        set!(member_sats, _member_key(layout, members[j]), _member_info(buffers, j))
    end

    # Only measured biases, as in `calc_pvt`: an unmeasured one coasts on process noise
    # (or was never seeded) and would be indistinguishable from a measured one.
    inter_system_biases = solution.inter_system_biases
    for (index, time_system) in enumerate(layout.time_systems)
        index == vt.primary_clock_index && continue
        _is_measured(member -> member.clock_bias_index == index, members, included) ||
            continue
        inter_system_biases[time_system] =
            (x[idxs.clock_biases[index]] - primary_clock_bias) * m
    end
    inter_frequency_biases = solution.inter_frequency_biases
    for (index, band) in enumerate(layout.extra_bands)
        _is_measured(member -> member.ifb_index == index, members, included) || continue
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

function _is_measured(predicate::F, members, included) where {F}
    for j in included
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
