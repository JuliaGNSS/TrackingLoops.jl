# ─────────────────────────────────────────────────────────────────────────────
# The per-record vector loop: a scalar FLL-assisted PLL/DLL of the user's
# choice, which a vector-tracking filter can take over.
#
# In vector tracking a central navigation filter closes every satellite's loops
# at once (see `NavigationFilterConfig`). Per record, the satellite's DLL and FLL
# outputs are only *accumulated* for the filter, and the filter sets the
# replica: the carrier NCO to its predicted
# Doppler, the code NCO by its code correction, carrier-aided. Before the
# filter takes a satellite over (and after it lets it go) the satellite runs
# the scalar loop it wraps, bit for bit.
# ─────────────────────────────────────────────────────────────────────────────

"""
    VectorPLLAndDLL(signals...; inner = ConventionalAssistedPLLAndDLL(),
                    filter_config = NavigationFilterConfig(), vt_update_interval = 100ms,
                    lock_cn0_threshold = 30dBHz,
                    residual_pll_loop_filter_bandwidth = default_residual_pll_loop_filter_bandwidth(),
                    max_satellites_per_signal = 16,
                    approximate_year = year(now(UTC)),
                    enable_ionospheric_correction = true,
                    enable_tropospheric_correction = true)

Vector-tracking Doppler estimator for the ranging `signals`. Each is a signal
that carries navigation data, ranged on and decoded alike (`GPSL1CA()`), or a
pilot + data pair, `driver => decoding_signal` (`GalileoE1C() => GalileoE1B()`,
`GPSL5Q() => GPSL5I()`): the satellite ranges on the pilot, whose prompt has no
data-bit transitions, and decodes the data component. A decoding signal must
carry data and have its driver's chip rate; the code lengths may differ (GPS
`GPSL2CL() => GPSL2CM()`). No signal may be given twice.

It does the whole pipeline inside [`step_loop`](@ref), from what the records
carry: every satellite stepped with this estimator shares one navigation
engine, which decodes the navigation data, solves the scalar PVT, seeds the
navigation filter from its first fix and from then on runs one filter cycle
every `vt_update_interval`, taking satellites over and handing them back. A host needs
no vector-specific code: it builds a satellite's state with
[`init_estimator_state`](@ref) for its driver and steps it with every record of
the satellite, like any other loop's. Read the results with
[`navigation_solution`](@ref), [`navigation_status`](@ref),
[`release_reason`](@ref), [`member_sats`](@ref), [`position_uncertainty`](@ref)
and [`clock_uncertainty`](@ref), all keyed by the driver.

The engine runs no bit clock and no C/N₀ estimator of its own: it reads the
host's, from records built from each signal's [`SignalLoopState`](@ref) (see
[`LoopRecord`](@ref)). The decoding signal's records carry where its bit clock
stands and the soft bits it formed, which the engine decodes while the satellite
runs its scalar loop; the driver's carry the C/N₀ estimator the host configured,
read at each navigation epoch for the lock and the measurement weights (capped
at 80 dB-Hz). A satellite is in lock when the
driver's C/N₀ is at least `lock_cn0_threshold` and the decoding signal holds
its bit sync. A noise-referenced estimator reads `-Inf` dB-Hz until the host
has a noise density, so until then the satellite is out of lock.

The records must also identify their satellite and replica: `prn` and
`code_phase` on the [`LoopRecord`](@ref), and `sample_index /
sampling_frequency` on a time grid shared by all satellites. A record without a
PRN, or of a driver not among `signals`, throws an `ArgumentError`; a record of
another passenger is ignored — also when it is the decoding signal of a pair
that does not drive this satellite. A satellite's snapshot of a navigation epoch
needs a record of its driver and one of its decoding signal across the epoch,
in either order. The host should step every satellite at least once per
`vt_update_interval / 2`: a cycle runs once every satellite has completed its epoch's
snapshot, and a satellite of which one signal has not stepped for
`2 · vt_update_interval` is dropped.

Per satellite, every driver record runs `inner` — the scalar loop the
satellite uses until the filter takes it over, and again once the filter
releases it. While a satellite is in the vector loop (see the vector-tracking
page of the manual):

  - its carrier NCO is set by the navigation filter alone: each record is
    commanded the filter's predicted Doppler at the middle of the record the
    command lands on, from a ramp the latest cycle's state gives (velocity,
    acceleration where modelled, clock drift, the satellite's motion). No loop
    filter and no phase discriminator enter it;
  - a residual PLL (`TrackingLoops.ResidualPLL`), a third-order PLL at
    `residual_pll_loop_filter_bandwidth`, follows the incoming carrier on the
    prompts the NCO leaves, from the words the NCO ran, Costas or
    four-quadrant by the record's `carrier_phase_polarity`. It never steers the
    NCO. The scalar loop, its staging included, stands still;
    [`phase_lock_indicator`](@ref) is the residual PLL's, on the prompts it rotated
    onto the incoming carrier, and [`carrier_loop_stage`](@ref) reports
    `VECTOR_LOOP`;
  - its code filter is frozen, and the filter's code correction replaces its
    output: `code_doppler = init_code_doppler + code_freq_update +
    (carrier_doppler - init_carrier_doppler) · code_center_frequency_ratio`;
  - the DLL output (chips) and the FLL discriminator (Hz) are accumulated for
    the filter to read. The replica runs on the filter's command alone, so the
    FLL measures the signal against it undisturbed;
  - the engine forms the soft bits it decodes itself: it rotates the decoding
    signal's prompts onto the incoming carrier by the residual PLL's phase and
    accumulates them over the data symbols the host's bit clock counts. The
    host's own soft bits are not decoded, so the host's prompts need no
    rotation; its C/N₀ estimator should read power only (the default
    noise-referenced one, or the moment-based one): the prompts rotate at the
    filter's Doppler error, which a coherent estimator such as
    [`NWPRCN0Estimator`](@ref) reads as a small loss.

The satellite is added to the vector loop at its scalar loop's Doppler. It is
released at its residual PLL's estimate of the incoming Doppler and its rate (out
of lock, at the filter's), which the scalar loop's carrier filter starts from, into a
restarted staging, FLL-assisted again, so the scalar loop pulls in afresh.

`residual_pll_loop_filter_bandwidth` is the residual PLL's noise bandwidth,
capped like the scalar PLL's against the record's integration time
([`effective_carrier_loop_filter_bandwidth`](@ref)). Under the default
`motion_model_order = 2` the filter's prediction does not carry the user's
acceleration, so the residual PLL follows a manoeuvre by itself: a third-order
loop's error to a jerk `j` is `j / (λ ω₀³)` cycles (`ω₀ = 1.2 B_L`, the scalar
PLL's loop filter's), 5.4° at 10 Hz for the 2.5 m/s³ of a 0.5 g manoeuvre built
up over two seconds, 43° at 5 Hz. Narrower suits a gentler platform or
`motion_model_order = 3`.

Each satellite picks up the commands of the latest cycle on its next driver
record, sized for where they land: at `landing_sample`, or at the record's end
under `NO_LANDING_SAMPLE`. A record of any other signal returns the command in
force there.

`inner` must have an FLL-assisted carrier filter, which pulls a released
satellite in again: a `ConventionalPLLAndDLL{ThirdOrderAssistedBilinearLF}`
(what [`ConventionalAssistedPLLAndDLL`](@ref) builds) or an
[`NCOReferencedPLLAndDLL`](@ref). Anything else throws an `ArgumentError`, as does a dataless signal given alone (a pilot such
as GPS L1C-P or Galileo E1C must be paired with its data component).
`filter_config = nothing` only ever solves the scalar PVT.

Storage is allocated at construction for `max_satellites_per_signal`
satellites per ranging signal — a `pilot => data` pair is one — and only grows
past that: a satellite that is dropped leaves its storage to the next one.
"""
struct VectorPLLAndDLL{E<:AbstractDopplerEstimator,N} <: AbstractDopplerEstimator
    inner::E
    navigation::N
    # The residual PLL's bandwidth, which each satellite's state takes on at
    # `init_estimator_state` and keeps.
    residual_pll_loop_filter_bandwidth::typeof(1.0Hz)
end

# Whether an estimator's carrier filter has an FLL path, as `_has_fll_path` of a
# carrier filter type: the FLL-assisted conventional loop and the NCO-referenced
# loop.
_has_fll_path(::AbstractDopplerEstimator) = false
_has_fll_path(::ConventionalPLLAndDLL{CA}) where {CA} = _has_fll_path(CA)
_has_fll_path(::NCOReferencedPLLAndDLL) = true

"""
    SatVectorPLLAndDLL

Per-satellite state of a [`VectorPLLAndDLL`](@ref), produced by
[`init_estimator_state`](@ref) and advanced by [`step_loop`](@ref). Its fields
are internal and may change in any release. Use [`phase_lock_indicator`](@ref)
and [`carrier_loop_stage`](@ref) to inspect it, and [`satellite_report`](@ref)
for what the navigation engine knows of the satellite.
"""
struct SatVectorPLLAndDLL{S}
    # The inner scalar loop's state and, on top, the interface to the vector-tracking
    # filter:
    #
    #   - `in_vector_loop`: whether the filter controls this satellite's NCOs. While `false`
    #     the satellite runs the inner loop and nothing is accumulated.
    #   - `code_disc_sum`: `(count, sum)` of the DLL output (chips) since the
    #     filter last read it.
    #   - `fll_disc_sum`: `(count, sum)` of the FLL discriminator (Hz) since
    #     the filter last read it, over the records with a previous prompt.
    #   - `previous_replica_half`: `(cycles, duration)` the replica's carrier advanced
    #     by over the second half of the driver's last record, and that half's length,
    #     `NaN` before the first record. The residual PLL propagates its phase
    #     from one prompt to the next by it, across a hand-over between the loops
    #     too.
    #   - `code_freq_update`: the code correction the per-record step applies, set
    #     from the filter's latest cycle.
    #   - `code_freq_update_history` and `code_update_landing_lead`: the last three
    #     code corrections the filter set, newest first, and how long after the
    #     filter's cycle epoch the newest reached the replica. The filter's next
    #     code measurement needs them to move the mid-cycle mean discriminator to
    #     the epoch. They record what the replica was steered by, so a
    #     `reset_estimator_state`, which folds the newest correction into the inner
    #     loop's Dopplers, keeps them.
    #   - `ramp` and `residual_pll`: the carrier Doppler ramp the filter commands, from its
    #     latest cycle, and the satellite's residual PLL (`ResidualPLL`). Both are set
    #     on the filter's first command, which a satellite is added to the vector loop with,
    #     and cleared out of it (`_has_carrier_ramp`).
    #   - `slot_index`, `registration` and `cycle_id`: where the navigation engine keeps
    #     this satellite (`0` until its first record), and the latest cycle it has
    #     taken up.
    inner::S
    in_vector_loop::Bool
    code_disc_sum::Tuple{Int,Float64}
    fll_disc_sum::Tuple{Int,typeof(1.0Hz)}
    previous_replica_half::Tuple{Float64,typeof(1.0s)}
    code_freq_update::typeof(1.0Hz)
    code_freq_update_history::NTuple{3,typeof(1.0Hz)}
    code_update_landing_lead::typeof(1.0s)
    ramp::CarrierRamp
    residual_pll::ResidualPLL
    slot_index::Int
    registration::Int
    cycle_id::Int
end

_is_driver_record(state::SatVectorPLLAndDLL, record::LoopRecord) =
    _is_driver_record(state.inner, record)
_is_driven_by(state::SatVectorPLLAndDLL, signal::AbstractGNSSSignal) =
    _is_driven_by(state.inner, signal)

# A satellite with the vector interface empty — out of the loop or just added —
# on the slot of `registration`. The replica's last half record is not the vector
# interface's but the replica's, so it is kept; so is the residual PLL's
# bandwidth, which `residual_pll_loop_filter_bandwidth` gives.
SatVectorPLLAndDLL(
    inner,
    in_vector_loop::Bool,
    slot_index::Int = 0,
    registration::Int = 0,
    cycle_id::Int = -1;
    previous_replica_half::Tuple{Float64,typeof(1.0s)} = (NaN, NaN * s),
    residual_pll_loop_filter_bandwidth::typeof(1.0Hz) = default_residual_pll_loop_filter_bandwidth(),
) =
    SatVectorPLLAndDLL(
        inner,
        in_vector_loop,
        (0, 0.0),
        (0, 0.0Hz),
        previous_replica_half,
        0.0Hz,
        (0.0Hz, 0.0Hz, 0.0Hz),
        0.0s,
        CarrierRamp(),
        ResidualPLL(residual_pll_loop_filter_bandwidth),
        slot_index,
        registration,
        cycle_id,
    )

function SatVectorPLLAndDLL(
    state::SatVectorPLLAndDLL{S};
    inner::Maybe{S} = nothing,
    in_vector_loop::Maybe{Bool} = nothing,
    code_disc_sum::Maybe{Tuple{Int,Float64}} = nothing,
    fll_disc_sum::Maybe{Tuple{Int,typeof(1.0Hz)}} = nothing,
    previous_replica_half::Maybe{Tuple{Float64,typeof(1.0s)}} = nothing,
    code_freq_update::Maybe{typeof(1.0Hz)} = nothing,
    code_freq_update_history::Maybe{NTuple{3,typeof(1.0Hz)}} = nothing,
    code_update_landing_lead::Maybe{typeof(1.0s)} = nothing,
    ramp::Maybe{CarrierRamp} = nothing,
    residual_pll::Maybe{ResidualPLL} = nothing,
    slot_index::Maybe{Int} = nothing,
    registration::Maybe{Int} = nothing,
    cycle_id::Maybe{Int} = nothing,
) where {S}
    SatVectorPLLAndDLL{S}(
        isnothing(inner) ? state.inner : inner,
        isnothing(in_vector_loop) ? state.in_vector_loop : in_vector_loop,
        isnothing(code_disc_sum) ? state.code_disc_sum : code_disc_sum,
        isnothing(fll_disc_sum) ? state.fll_disc_sum : fll_disc_sum,
        isnothing(previous_replica_half) ? state.previous_replica_half :
        previous_replica_half,
        isnothing(code_freq_update) ? state.code_freq_update : code_freq_update,
        isnothing(code_freq_update_history) ? state.code_freq_update_history :
        code_freq_update_history,
        isnothing(code_update_landing_lead) ? state.code_update_landing_lead :
        code_update_landing_lead,
        isnothing(ramp) ? state.ramp : ramp,
        isnothing(residual_pll) ? state.residual_pll : residual_pll,
        isnothing(slot_index) ? state.slot_index : slot_index,
        isnothing(registration) ? state.registration : registration,
        isnothing(cycle_id) ? state.cycle_id : cycle_id,
    )
end

# The state on the fresh slot `slot_index` of `registration`, owing the pick-up of the
# latest cycle.
_registered(state::SatVectorPLLAndDLL, slot_index::Int, registration::Int) =
    SatVectorPLLAndDLL(state; slot_index, registration, cycle_id = -1)

"""
    init_estimator_state(estimator::VectorPLLAndDLL, driver_signal, carrier_doppler, code_doppler)

The inner loop's state with the vector interface empty: out of the vector
loop, nothing accumulated, no commands, and the residual PLL at the
estimator's `residual_pll_loop_filter_bandwidth`. The satellite registers with
the navigation engine on its first record.
"""
init_estimator_state(
    estimator::VectorPLLAndDLL,
    driver_signal::AbstractGNSSSignal,
    carrier_doppler,
    code_doppler,
) = SatVectorPLLAndDLL(
    init_estimator_state(estimator.inner, driver_signal, carrier_doppler, code_doppler),
    false;
    residual_pll_loop_filter_bandwidth = estimator.residual_pll_loop_filter_bandwidth,
)

"""
    reset_estimator_state(estimator::VectorPLLAndDLL, state, carrier_doppler, code_doppler)

Re-seed the inner loop from the converged Dopplers, zero both accumulators and
stop applying the code correction, keeping `in_vector_loop`, the carrier ramp, the
residual PLL and the satellite's place in the navigation engine. The code correction
must leave the NCO words with the re-seed: the converged Dopplers already
contain it, so keeping it would apply it twice. The replica is still steered by
it, though, so the correction history and its landing lead, which the next code
measurement is moved to the epoch with, are kept. The carrier ramp, the filter's
command, stays: it sets the carrier NCO to the filter's predicted Doppler in place
of the inner loop's word, so the converged Dopplers do not contain it.
"""
reset_estimator_state(
    estimator::VectorPLLAndDLL,
    state::SatVectorPLLAndDLL,
    carrier_doppler,
    code_doppler,
) = SatVectorPLLAndDLL(
    state;
    inner = reset_estimator_state(estimator.inner, state.inner, carrier_doppler, code_doppler),
    code_disc_sum = (0, 0.0),
    fll_disc_sum = (0, 0.0Hz),
    code_freq_update = 0.0Hz,
)

# In the vector loop the residual PLL follows the carrier, whatever stage the inner
# loop's staging had reached: the stage is the residual PLL, and the phase-lock
# indicator is the residual PLL's; out of it, the inner loop's.
carrier_loop_stage(state::SatVectorPLLAndDLL) =
    state.in_vector_loop ? VECTOR_LOOP : carrier_loop_stage(state.inner)
phase_lock_indicator(state::SatVectorPLLAndDLL) =
    state.in_vector_loop ? state.residual_pll.lock.lock_detector.indicator :
    phase_lock_indicator(state.inner)


# ─────────────────────────────────────────────────────────────────────────────
# What the navigation engine does to a satellite's state. Each keeps the
# satellite's place in the engine.

# Put the satellite into the vector loop with its vector interface cleared: the
# accumulators and commands zeroed, so that it never runs on stale values, and the inner
# loop as it stands. One already in the loop is returned unchanged.
_add_to_vector_loop(state::SatVectorPLLAndDLL) =
    state.in_vector_loop ? state :
    SatVectorPLLAndDLL(
        state.inner,
        true,
        state.slot_index,
        state.registration,
        state.cycle_id;
        state.previous_replica_half,
        residual_pll_loop_filter_bandwidth = state.residual_pll.bandwidth,
    )

# Take the satellite out of the vector loop with its vector interface cleared, as
# `_add_to_vector_loop` puts it in, and its inner loop as it stands. Unlike
# `_hand_over_to_scalar_loop` it does not re-seed the inner loop: it marks the cycle's
# release on the slot's copy of the state (`_mark_release!`), which the satellite takes up
# on its own state by handing over to its scalar loop (`_take_up_cycle`), and it registers
# a satellite out of the loop.
_release_from_vector_loop(state::SatVectorPLLAndDLL) =
    SatVectorPLLAndDLL(
        state.inner,
        false,
        state.slot_index,
        state.registration,
        state.cycle_id;
        state.previous_replica_half,
        residual_pll_loop_filter_bandwidth = state.residual_pll.bandwidth,
    )

# `_release_from_vector_loop` with the inner loop re-seeded from the vector loop's
# latest estimates at the sample `landing` (of the records' grid at `fs`, Hz), where the
# release lands, so the scalar loop takes over from where the vector loop left the
# carrier: the Doppler its residual PLL estimates there — the filter's ramp plus the
# residual PLL's residual — and their Doppler rate, which a third-order carrier filter
# starts from. A residual PLL out of lock (or one that never stepped) knows no better than
# the filter, so the Doppler is then the replica's carrier word there, from `words`,
# and the rate the ramp's. The code starts from the replica's code word there. Unlike
# a reset, the release restarts the carrier loop's staging: the scalar loop pulls in
# again, FLL-assisted at the wide bandwidth, which keeps a satellite the vector loop
# let go.
function _hand_over_to_scalar_loop(state::SatVectorPLLAndDLL, words, landing, fs)
    carrier_word, code_word = mean_nco_word(words, landing, landing)
    carrier_doppler = carrier_word * Hz
    code_doppler = code_word * Hz
    time = landing / fs
    ramp, residual_pll = state.ramp, state.residual_pll
    rate = ramp.rate
    if residual_pll.lock.locked && !isnan(residual_pll.prompt_center)
        dt = time - residual_pll.prompt_center
        carrier_doppler =
            (_doppler_at(ramp, time) + _residual_doppler(residual_pll) + _residual_doppler_rate(residual_pll) * dt) * Hz
        rate += _residual_doppler_rate(residual_pll)
    end
    SatVectorPLLAndDLL(
        _release_from_vector_loop(state);
        inner = _reseed_inner(state.inner, carrier_doppler, code_doppler, rate),
    )
end

# The inner reset reads the per-satellite state only, never the estimator's
# configuration, so a default-configured estimator stands in for it. The carrier
# filter starts at the Doppler rate `rate` (Hz/s).
_reseed_inner(state::SatConventionalPLLAndDLL, carrier_doppler, code_doppler, rate) =
    SatConventionalPLLAndDLL(
        reset_estimator_state(ConventionalPLLAndDLL(), state, carrier_doppler, code_doppler);
        carrier_loop_filter = _carrier_filter_at_rate(state.carrier_loop_filter, rate),
        staging = CarrierLoopStaging(typeof(state.carrier_loop_filter)),
    )
_reseed_inner(state::SatNCOReferencedPLLAndDLL, carrier_doppler, code_doppler, rate) =
    SatNCOReferencedPLLAndDLL(
        reset_estimator_state(NCOReferencedPLLAndDLL(), state, carrier_doppler, code_doppler);
        carrier_loop_filter = _carrier_filter_at_rate(state.carrier_loop_filter, rate),
        staging = CarrierLoopStaging(typeof(state.carrier_loop_filter)),
    )

# A fresh carrier filter of `filter`'s type at the Doppler rate `rate` (Hz/s): a
# third-order one keeps the rate as its second state, `x2`; one of lower order starts
# at zero.
_carrier_filter_at_rate(filter, rate) =
    hasfield(typeof(filter), :x2) ? _constructorof(typeof(filter))(zero(filter.x1), rate * Hz^2) :
    _constructorof(typeof(filter))()

# The filter's command from its latest cycle, taking over at `landing_lead` after
# the cycle's epoch: the code correction, which joins the front of
# `code_freq_update_history`, and the carrier Doppler `ramp`. A satellite added to
# the loop starts its residual PLL from the scalar loop's carrier word in `words` at
# the sample `landing` the command lands on, which the replica ran on until then; one
# in the loop keeps its residual PLL's estimate of the incoming Doppler across the new
# ramp.
function _set_vector_commands(
    state::SatVectorPLLAndDLL,
    code_freq_update,
    ramp::CarrierRamp,
    landing_lead,
    words,
    landing,
)
    newest, previous, _ = state.code_freq_update_history
    residual_pll = if _has_carrier_ramp(state)
        _residual_pll_on_new_ramp(state.residual_pll, state.ramp, ramp)
    else
        scalar_doppler, _ = mean_nco_word(words, landing, landing)
        _initialized_residual_pll(state.residual_pll, ramp, scalar_doppler, ramp.time, state.inner.staging)
    end
    SatVectorPLLAndDLL(
        state;
        code_freq_update,
        code_freq_update_history = (code_freq_update, newest, previous),
        code_update_landing_lead = landing_lead,
        ramp,
        residual_pll,
    )
end

# Whether the satellite has a carrier ramp yet: from the filter's first command, which
# its residual PLL takes over from the scalar loop at, until it leaves the vector loop. A
# member has one on every record, as it is added with its first command; only the command
# it takes up reads this, to tell the first from the later ones (`_set_vector_commands`).
_has_carrier_ramp(state::SatVectorPLLAndDLL) = !isnan(state.ramp.time)

# Empty both discriminator accumulators, once the filter has read them.
_reset_discriminator_accumulators(state::SatVectorPLLAndDLL) =
    SatVectorPLLAndDLL(state; code_disc_sum = (0, 0.0), fll_disc_sum = (0, 0.0Hz))

# The mean of a discriminator sum `(count, sum)` — the DLL's (chips) or the FLL's (Hz)
# since the last reset — or `nothing` if nothing has been accumulated.
_disc_mean((count, disc_sum)) = count == 0 ? nothing : disc_sum / count

# The carrier phase (cycles) the replica advanced by over the first and the second
# half of `record`, each with the half's length, as `(cycles, duration)`, from the
# words it ran on (Hz).
@inline function _replica_halves(record::LoopRecord, words)
    fs = record.sampling_frequency
    record_end = record.sample_index
    half_samples = record.integrated_samples / 2
    center = record_end - half_samples
    first_word, _ = mean_nco_word(words, record_end - record.integrated_samples, center)
    second_word, _ = mean_nco_word(words, center, record_end)
    half_duration = convert(typeof(1.0s), half_samples / fs)
    first_cycles = Float64(uconvert(NoUnits, first_word * Hz * half_duration))
    second_cycles = Float64(uconvert(NoUnits, second_word * Hz * half_duration))
    (first = (first_cycles, half_duration), second = (second_cycles, half_duration))
end

# The FLL accumulator with `record`'s discriminator added. A record without a
# previous prompt (the first one, or one whose carrier phase polarity changed) has
# no frequency measurement and is left out rather than taken as 0 Hz.
@inline _accumulate_fll((count, disc_sum), fll_discriminator, record::LoopRecord) =
    iszero(record.previous_prompt) ? (count, disc_sum) :
    (count + 1, disc_sum + fll_discriminator)

# One record through the satellite's loop: the inner loop out of the vector loop,
# the vector step in it.
@inline function _step_satellite(
    estimator::VectorPLLAndDLL,
    state::SatVectorPLLAndDLL,
    record::LoopRecord,
    words,
    landing_sample::Int64,
)
    replica_halves = _replica_halves(record, words)
    if !state.in_vector_loop
        inner, carrier_doppler, code_doppler =
            step_loop(estimator.inner, state.inner, record, words, landing_sample)
        return SatVectorPLLAndDLL(state; inner, previous_replica_half = replica_halves.second),
        carrier_doppler,
        code_doppler
    end
    _step_vector_loop(state, record, words, landing_sample, replica_halves, estimator.navigation.lock_cn0_threshold_dbhz)
end

# The DLL output (chips) of a record, normalised with the code word the replica ran
# on, as the inner loops measure it.
@inline function _code_discriminator(record::LoopRecord, words)
    record_start = record.sample_index - record.integrated_samples
    _, applied_code = mean_nco_word(words, record_start, record.sample_index)
    dll_disc(record.signal, record.filtered_correlator, applied_code * Hz, record.sampling_frequency)
end

# One record of a member: the DLL measured and accumulated, the carrier measured
# (`_measure_carrier`), and the next command: the filter's ramp at the middle of the
# record it lands on for the carrier and, carrier-aided from it, the filter's code
# correction for the code. A satellite is added to the vector loop with its first command
# (`_take_up_cycle`), so a member always has a ramp. The inner loop stands still, its staging too; a
# release re-seeds it.
@inline function _step_vector_loop(
    state::SatVectorPLLAndDLL,
    record::LoopRecord,
    words,
    landing_sample::Int64,
    replica_halves,
    lock_cn0_threshold_dbhz::Float64,
)
    inner = state.inner
    landing = landing_sample == NO_LANDING_SAMPLE ? record.sample_index : landing_sample
    residual_pll, fll_disc_sum = _measure_carrier(state, record, words, replica_halves, lock_cn0_threshold_dbhz)
    # The command lands on a record taken to be this one's length.
    carrier = _doppler_at(state.ramp, (landing + record.integrated_samples / 2) / _sampling_frequency_hz(record))
    carrier_doppler, code_doppler = aid_dopplers(
        record.signal,
        inner.init_carrier_doppler,
        inner.init_code_doppler,
        carrier * Hz - inner.init_carrier_doppler,
        state.code_freq_update,
    )
    SatVectorPLLAndDLL(
        state;
        code_disc_sum = state.code_disc_sum .+ (1, _code_discriminator(record, words)),
        fll_disc_sum,
        previous_replica_half = replica_halves.second,
        residual_pll,
    ),
    carrier_doppler,
    code_doppler
end

# The carrier of a member's record: the FLL measured and accumulated, the residual PLL
# stepped on the prompt and its lock tested on the prompt it rotated (`_with_lock_test`).
# Returns the residual PLL and the FLL accumulator.
#
# Both span the previous prompt's centre to this one's: the second half of the
# previous record, any gap a host left between the records — the replica ran on
# through it, on the words in force there — and the first half of this one.
@inline function _measure_carrier(
    state::SatVectorPLLAndDLL,
    record::LoopRecord,
    words,
    replica_halves,
    lock_cn0_threshold_dbhz::Float64,
)
    fs = _sampling_frequency_hz(record)
    integration_time = record.integrated_samples / fs
    previous_cycles, previous_duration = state.previous_replica_half
    first_cycles, first_duration = replica_halves.first
    record_start = record.sample_index - record.integrated_samples
    # The previous record's end, known once the residual PLL has stepped on its prompt.
    previous_end = state.residual_pll.prompt_center + ustrip(s, previous_duration)
    gap = isnan(previous_end) ? 0.0 : record_start / fs - previous_end
    gap_cycles = 0.0
    if gap > 0.5 / fs
        gap_word, _ = mean_nco_word(words, previous_end * fs, record_start)
        gap_cycles = gap_word * gap
    else
        gap = 0.0
    end
    span = ustrip(s, previous_duration) + gap + ustrip(s, first_duration)
    # The incoming carrier against the replica between the two prompts' centres; with
    # the replica on the filter's command alone, nothing but the signal moves it.
    fll_discriminator = fll_disc(
        record.signal,
        record.filtered_correlator,
        record.previous_prompt,
        (isnan(span) ? integration_time : span) * s;
        polarity = record.carrier_phase_polarity,
    )
    residual_pll, wiped_prompt =
        _step_residual_pll(state.residual_pll, state.ramp, record, previous_cycles + gap_cycles + first_cycles)
    host_cn0_dbhz = Float64(ustrip(estimate_cn0(record.cn0_estimator, integration_time * s)))
    residual_pll = _with_lock_test(
        residual_pll,
        wiped_prompt,
        integration_time,
        _phase_lock_time_constant(record.signal, record.integrated_code_blocks),
        lock_cn0_threshold_dbhz,
        host_cn0_dbhz >= lock_cn0_threshold_dbhz,
    )
    residual_pll, _accumulate_fll(state.fll_disc_sum, fll_discriminator, record)
end

