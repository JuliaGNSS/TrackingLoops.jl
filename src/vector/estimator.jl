# ─────────────────────────────────────────────────────────────────────────────
# The per-record vector loop: a scalar FLL-assisted PLL/DLL of the user's
# choice, which a vector-tracking filter can take over.
#
# In vector tracking a central navigation filter closes every satellite's loops
# at once (see `VectorTracking`). Per record, the satellite's own loop still
# runs its PLL, but its DLL and FLL outputs are only *accumulated* for the
# filter, and the filter's per-satellite NCO corrections steer the replica:
# the code correction replaces the code filter's output, the carrier
# correction drives the FLL branch of the carrier filter. Before the filter
# takes a satellite over (and after it lets it go) the satellite runs the
# scalar loop it wraps, bit for bit.
# ─────────────────────────────────────────────────────────────────────────────

"""
    VectorPLLAndDLL(signals...; inner = ConventionalAssistedPLLAndDLL(),
                    config = VectorTracking(), cycle_time = 100ms,
                    lock_cn0_threshold = 30dBHz, max_satellites_per_signal = 16,
                    approximate_year = year(now(UTC)),
                    enable_ionospheric_correction = true,
                    enable_tropospheric_correction = true,
                    vector_fll_loop_filter_bandwidth = nothing)

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
every `cycle_time`, taking satellites over and handing them back. A host needs
no vector-specific code: it builds a satellite's state with
[`init_estimator_state`](@ref) for its driver and steps it with every record of
the satellite, like any other loop's. Read the results with
[`navigation_solution`](@ref), [`navigation_status`](@ref),
[`release_reason`](@ref), [`member_sats`](@ref), [`position_uncertainty`](@ref)
and [`clock_uncertainty`](@ref), all keyed by the driver.

The engine runs no bit clock and no C/N₀ estimator of its own: it reads the
host's, from records built from each signal's [`SignalLoopState`](@ref) (see
[`LoopRecord`](@ref)). The decoding signal's records carry the soft bits the
engine decodes and where its bit clock stands; the driver's carry the C/N₀
estimator the host configured, read at each navigation epoch for the lock and
the measurement weights (capped at 80 dB-Hz). A satellite is in lock when the
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
`cycle_time / 2`: a cycle runs once every satellite has completed its epoch's
snapshot, and a satellite of which one signal has not stepped for
`2 · cycle_time` is dropped.

Per satellite, every driver record runs `inner` — the scalar loop the
satellite uses until the filter takes it over, and again once the filter
releases it. While a satellite is in the vector loop:

  - its carrier filter always runs, with the PLL branch fed by the satellite's
    own phase discriminator at the narrow PLL bandwidth, whatever
    [`CarrierLoopStage`](@ref) its scalar loop had reached, which
    [`carrier_loop_stage`](@ref) reports as `NARROW_PLL`. The FLL branch is fed
    the filter's carrier correction, which it integrates as a frequency error
    (the correction is not added to the Doppler directly). It runs at its own
    bandwidth, `vector_fll_loop_filter_bandwidth`, 1 Hz by default
    ([`default_vector_fll_loop_filter_bandwidth`](@ref)) and capped against
    `cycle_time` ([`effective_vector_fll_loop_filter_bandwidth`](@ref)). The
    correction is set once per cycle and held for the cycle, so the FLL branch
    sits well below the 5 Hz Nyquist frequency of the default 10 Hz cycle and
    well below the PLL. The satellite is released into a restarted staging,
    FLL-assisted again, so the scalar loop pulls in afresh;
  - its code filter is frozen, and the filter's code correction replaces its
    output: `code_doppler = init_code_doppler + code_freq_update +
    carrier_filter_output · code_center_frequency_ratio`;
  - the DLL output (chips) and the carrier phase the incoming signal advanced
    by between the prompts — the replica's own advance plus the change of the
    PLL phase error, unwrapped — are accumulated for the filter to read.

Each satellite picks up the corrections of the latest cycle on its next driver
record, sized for where they land: at `landing_sample`, or at the record's end
under `NO_LANDING_SAMPLE`. A record of any other signal returns the command in
force there.

`inner` must have an FLL-assisted carrier filter, since that is the only input
path the carrier correction has into the loop: a
`ConventionalPLLAndDLL{ThirdOrderAssistedBilinearLF}` (what
[`ConventionalAssistedPLLAndDLL`](@ref) builds) or an
[`NCOReferencedPLLAndDLL`](@ref). With the latter the phase discriminator keeps
its prediction to the landing sample in the vector loop too. Anything else
throws an `ArgumentError`, as does a dataless signal given alone (a pilot such
as GPS L1C-P or Galileo E1C must be paired with its data component).
`config = nothing` only ever solves the scalar PVT.

Storage is allocated at construction for `max_satellites_per_signal`
satellites per ranging signal and only grows past that: a satellite that is
dropped leaves its storage to the next one.
"""
struct VectorPLLAndDLL{E<:AbstractDopplerEstimator,N} <: AbstractDopplerEstimator
    inner::E
    navigation::N
    # The FLL branch's bandwidth in the vector loop; `nothing` takes the driver
    # signal's default, resolved into each satellite's `LoopBandwidths`.
    vector_fll_loop_filter_bandwidth::Maybe{typeof(1.0Hz)}
end

# Whether an estimator's carrier filter has an FLL path, as `_has_fll_path` of a
# carrier filter type: the FLL-assisted conventional loop and the NCO-referenced
# loop.
_has_fll_path(::AbstractDopplerEstimator) = false
_has_fll_path(::ConventionalPLLAndDLL{CA}) where {CA} = _has_fll_path(CA)
_has_fll_path(::NCOReferencedPLLAndDLL) = true

"""
    CarrierPhaseAccumulator

What a satellite in the vector loop accumulates for the navigation filter's
pseudorange-rate measurement: per pair of consecutive prompts of its driver,
the carrier phase the incoming signal advanced by between their centres — the
replica's own advance over that span plus the change of the PLL phase error
(`pll_disc`) from the first prompt to the second, the incoming signal's advance
against the replica — and the span. Over a navigation cycle the spans tile the
time between the centres of the cycle's last record and the previous cycle's,
so consecutive cycles share their boundary prompt and the sum of the cycles'
measurements is the incoming carrier phase, at the noise of two prompt phases.

The change of the phase error is not wrapped: summed over the spans it
telescopes to the last prompt's phase error minus the first's, each within the
discriminator's range. Prompt noise that turns one prompt by more than that
range from the one before (a quarter cycle for the Costas discriminator, half a
cycle for the four-quadrant one) is undone by the next span instead of leaving
half a cycle or a whole one in the sum, as the FLL's rotation, wrapped to that
range, did. A slip of the PLL itself still shows, as one cycle's outlier: the
replica's phase really moved by half a cycle or a whole one against the
signal's.

  - `count`: the spans accumulated;
  - `cycles`: the incoming carrier phase over them (cycles);
  - `duration`: their total length;
  - `time_moment`: their lengths times their midpoints, summed, on the
    records' time grid (`sample_index / sampling_frequency`), so
    `time_moment / duration` is the time the mean Doppler
    `cycles / duration` refers to.
"""
struct CarrierPhaseAccumulator
    count::Int
    cycles::Float64
    duration::typeof(1.0s)
    time_moment::typeof(1.0s^2)
end

CarrierPhaseAccumulator() = CarrierPhaseAccumulator(0, 0.0, 0.0s, 0.0s^2)

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
    #   - `vt_on`: whether the filter controls this satellite's NCOs. While `false`
    #     the satellite runs the inner loop and nothing is accumulated.
    #   - `code_discr_acc`: `(count, sum)` of the DLL output (chips) since the
    #     filter last read it.
    #   - `carrier_phase_acc`: the incoming carrier phase between consecutive
    #     prompts since the filter last read it, a `CarrierPhaseAccumulator`.
    #   - `previous_replica_half`: `(cycles, duration)` the replica's carrier advanced
    #     by over the second half of the driver's last record, and that half's length,
    #     `NaN` before the first record. It carries the accumulation across the
    #     filter's reads and across a hand-over between the loops.
    #   - `previous_phase_error`: `(cycles, polarity)`, the PLL phase error of the
    #     driver's last prompt against the replica that produced it and the carrier
    #     phase polarity it was read with, `NaN` before the first record. It carries
    #     the accumulation as `previous_replica_half` does.
    #   - `code_freq_update` / `carrier_freq_update`: the corrections the per-record
    #     step applies, set from the filter's latest cycle.
    #   - `code_freq_update_history` and `code_update_landing_lead`: the last three
    #     code corrections the filter set, newest first, and how long after the
    #     filter's cycle epoch the newest reached the replica. The filter's next
    #     code measurement needs them to move the mid-cycle mean discriminator to
    #     the epoch. They record what the replica was steered by, so a
    #     `reset_estimator_state`, which folds the newest correction into the inner
    #     loop's Dopplers, keeps them.
    #   - `slot`, `registration` and `cycle_id`: where the navigation engine keeps
    #     this satellite (`0` until its first record), and the latest cycle it has
    #     taken up.
    inner::S
    vt_on::Bool
    code_discr_acc::Tuple{Int,Float64}
    carrier_phase_acc::CarrierPhaseAccumulator
    previous_replica_half::Tuple{Float64,typeof(1.0s)}
    previous_phase_error::Tuple{Float64,Int8}
    code_freq_update::typeof(1.0Hz)
    carrier_freq_update::typeof(1.0Hz)
    code_freq_update_history::NTuple{3,typeof(1.0Hz)}
    code_update_landing_lead::typeof(1.0s)
    slot::Int
    registration::Int
    cycle_id::Int
end

_is_driver_record(state::SatVectorPLLAndDLL, record::LoopRecord) =
    _is_driver_record(state.inner, record)
_is_driven_by(state::SatVectorPLLAndDLL, signal::AbstractGNSSSignal) =
    _is_driven_by(state.inner, signal)

# A satellite with the vector interface empty — out of the loop or just joined —
# on the slot of `registration`. The replica's last half record and its last
# prompt's phase error are not the vector interface's but the replica's, so they
# are kept.
SatVectorPLLAndDLL(
    inner,
    vt_on::Bool,
    slot::Int = 0,
    registration::Int = 0,
    cycle_id::Int = -1;
    previous_replica_half::Tuple{Float64,typeof(1.0s)} = (NaN, NaN * s),
    previous_phase_error::Tuple{Float64,Int8} = (NaN, Int8(0)),
) =
    SatVectorPLLAndDLL(
        inner,
        vt_on,
        (0, 0.0),
        CarrierPhaseAccumulator(),
        previous_replica_half,
        previous_phase_error,
        0.0Hz,
        0.0Hz,
        (0.0Hz, 0.0Hz, 0.0Hz),
        0.0s,
        slot,
        registration,
        cycle_id,
    )

function SatVectorPLLAndDLL(
    state::SatVectorPLLAndDLL{S};
    inner::Maybe{S} = nothing,
    code_discr_acc::Maybe{Tuple{Int,Float64}} = nothing,
    carrier_phase_acc::Maybe{CarrierPhaseAccumulator} = nothing,
    previous_replica_half::Maybe{Tuple{Float64,typeof(1.0s)}} = nothing,
    previous_phase_error::Maybe{Tuple{Float64,Int8}} = nothing,
    cycle_id::Maybe{Int} = nothing,
) where {S}
    SatVectorPLLAndDLL{S}(
        something(inner, state.inner),
        state.vt_on,
        something(code_discr_acc, state.code_discr_acc),
        something(carrier_phase_acc, state.carrier_phase_acc),
        something(previous_replica_half, state.previous_replica_half),
        something(previous_phase_error, state.previous_phase_error),
        state.code_freq_update,
        state.carrier_freq_update,
        state.code_freq_update_history,
        state.code_update_landing_lead,
        state.slot,
        state.registration,
        something(cycle_id, state.cycle_id),
    )
end

# The state on a fresh slot `slot` of `registration`, owing the pick-up of the
# latest cycle.
_registered(state::SatVectorPLLAndDLL{S}, slot::Int, registration::Int) where {S} =
    SatVectorPLLAndDLL{S}(
        state.inner,
        state.vt_on,
        state.code_discr_acc,
        state.carrier_phase_acc,
        state.previous_replica_half,
        state.previous_phase_error,
        state.code_freq_update,
        state.carrier_freq_update,
        state.code_freq_update_history,
        state.code_update_landing_lead,
        slot,
        registration,
        -1,
    )

"""
    init_estimator_state(estimator::VectorPLLAndDLL, driver_signal, carrier_doppler, code_doppler)

The inner loop's state with the vector interface empty: out of the vector
loop, nothing accumulated, no corrections. The estimator's
`vector_fll_loop_filter_bandwidth`, or the driver signal's default where it is
`nothing`, becomes the inner state's `bandwidths.vector_fll`. The satellite joins the navigation
engine on its first record.
"""
init_estimator_state(
    estimator::VectorPLLAndDLL,
    driver_signal::AbstractGNSSSignal,
    carrier_doppler,
    code_doppler,
) = SatVectorPLLAndDLL(
    _with_vector_fll_bandwidth(
        init_estimator_state(estimator.inner, driver_signal, carrier_doppler, code_doppler),
        something(
            estimator.vector_fll_loop_filter_bandwidth,
            default_vector_fll_loop_filter_bandwidth(driver_signal),
        ),
    ),
    false,
)

# The inner state with the vector FLL bandwidth resolved into its bandwidths.
@inline function _with_vector_fll_bandwidth(inner, vector_fll)
    b = inner.bandwidths
    bandwidths = LoopBandwidths(b.wide_carrier, b.narrow_carrier, b.fll_assist, b.code, vector_fll)
    _with_bandwidths(inner, bandwidths)
end
_with_bandwidths(state::SatConventionalPLLAndDLL, bandwidths) =
    SatConventionalPLLAndDLL(state; bandwidths)
_with_bandwidths(state::SatNCOReferencedPLLAndDLL, bandwidths) =
    SatNCOReferencedPLLAndDLL(state; bandwidths)

"""
    reset_estimator_state(estimator::VectorPLLAndDLL, state, carrier_doppler, code_doppler)

Re-seed the inner loop from the converged Dopplers, zero both accumulators and
stop applying the corrections, keeping `vt_on` and the satellite's place in the
navigation engine. The corrections must leave the NCO words with the re-seed:
the converged Dopplers already contain the last correction, so keeping it
would apply it twice. The replica is still steered by it, though, so the
correction history and its landing lead, which the next code measurement is
moved to the epoch with, are kept.
"""
reset_estimator_state(
    estimator::VectorPLLAndDLL,
    state::SatVectorPLLAndDLL{S},
    carrier_doppler,
    code_doppler,
) where {S} = SatVectorPLLAndDLL{S}(
    reset_estimator_state(estimator.inner, state.inner, carrier_doppler, code_doppler),
    state.vt_on,
    (0, 0.0),
    CarrierPhaseAccumulator(),
    state.previous_replica_half,
    state.previous_phase_error,
    0.0Hz,
    0.0Hz,
    state.code_freq_update_history,
    state.code_update_landing_lead,
    state.slot,
    state.registration,
    state.cycle_id,
)

# In the vector loop the carrier loop runs at the narrow PLL, whatever stage the
# inner loop's staging has reached; out of it, the inner loop's stage.
carrier_loop_stage(state::SatVectorPLLAndDLL) =
    state.vt_on ? NARROW_PLL : carrier_loop_stage(state.inner)
phase_lock_indicator(state::SatVectorPLLAndDLL) = phase_lock_indicator(state.inner)

# ─────────────────────────────────────────────────────────────────────────────
# What the navigation engine does to a satellite's state. Each keeps the
# satellite's place in the engine.

# Put the satellite into the vector loop. A satellite joining starts with empty
# accumulators and no corrections; one already in the loop is returned unchanged.
_enable_vector_tracking(state::SatVectorPLLAndDLL) =
    state.vt_on ? state :
    SatVectorPLLAndDLL(
        state.inner,
        true,
        state.slot,
        state.registration,
        state.cycle_id;
        state.previous_replica_half,
        state.previous_phase_error,
    )

# Hand the satellite back to its inner scalar loop, with the accumulators and
# corrections zeroed so a later re-enable never runs on stale values.
_disable_vector_tracking(state::SatVectorPLLAndDLL) =
    SatVectorPLLAndDLL(
        state.inner,
        false,
        state.slot,
        state.registration,
        state.cycle_id;
        state.previous_replica_half,
        state.previous_phase_error,
    )

# `_disable_vector_tracking` with the inner loop re-seeded from the Dopplers the
# replica runs at — what `reset_estimator_state` does — so the scalar loop takes over
# from where the vector loop steered the replica, without a transient. Unlike a reset,
# the release restarts the carrier loop's staging: the scalar loop pulls in again,
# FLL-assisted at the wide bandwidth, which keeps a satellite the vector loop let go.
_release_from_vector_tracking(state::SatVectorPLLAndDLL, carrier_doppler, code_doppler) =
    SatVectorPLLAndDLL(
        _reseed_inner(state.inner, carrier_doppler, code_doppler),
        false,
        state.slot,
        state.registration,
        state.cycle_id;
        state.previous_replica_half,
        state.previous_phase_error,
    )

# The inner reset reads the per-satellite state only, never the estimator's
# configuration, so a default-configured estimator stands in for it.
_reseed_inner(state::SatConventionalPLLAndDLL, carrier_doppler, code_doppler) =
    SatConventionalPLLAndDLL(
        reset_estimator_state(ConventionalPLLAndDLL(), state, carrier_doppler, code_doppler);
        staging = CarrierLoopStaging(typeof(state.carrier_loop_filter)),
    )
_reseed_inner(state::SatNCOReferencedPLLAndDLL, carrier_doppler, code_doppler) =
    SatNCOReferencedPLLAndDLL(
        reset_estimator_state(NCOReferencedPLLAndDLL(), state, carrier_doppler, code_doppler);
        staging = CarrierLoopStaging(typeof(state.carrier_loop_filter)),
    )

# The filter's NCO corrections for this satellite, taking over at `landing_lead`
# after the filter's cycle epoch. The code correction joins the front of
# `code_freq_update_history`.
function _set_vector_corrections(
    state::SatVectorPLLAndDLL{S},
    code_freq_update,
    carrier_freq_update,
    landing_lead = 0.0s,
) where {S}
    newest, previous, _ = state.code_freq_update_history
    SatVectorPLLAndDLL{S}(
        state.inner,
        state.vt_on,
        state.code_discr_acc,
        state.carrier_phase_acc,
        state.previous_replica_half,
        state.previous_phase_error,
        code_freq_update,
        carrier_freq_update,
        (code_freq_update, newest, previous),
        landing_lead,
        state.slot,
        state.registration,
        state.cycle_id,
    )
end

# Empty both accumulators, once the filter has read them. The replica's last half
# record and its last prompt's phase error stay: the next cycle's first span starts
# from them.
_reset_code_and_carrier_accumulators(state::SatVectorPLLAndDLL) = SatVectorPLLAndDLL(
    state;
    code_discr_acc = (0, 0.0),
    carrier_phase_acc = CarrierPhaseAccumulator(),
)

# The mean DLL output (chips) accumulated since the last reset, or `nothing` if
# nothing has been accumulated.
function _mean_code_discriminator(state::SatVectorPLLAndDLL)
    count, discr_sum = state.code_discr_acc
    count == 0 ? nothing : discr_sum / count
end

# The mean incoming carrier Doppler over the spans accumulated since the last reset
# and the time it refers to (on the records' grid), or `nothing` if none was.
function _mean_carrier_doppler(state::SatVectorPLLAndDLL)
    acc = state.carrier_phase_acc
    acc.count == 0 && return nothing
    uconvert(Hz, acc.cycles / acc.duration), acc.time_moment / acc.duration
end

# The carrier phase (cycles) the replica advanced by over the first and the second
# half of `record`, each with the half's length, as `(cycles, duration)`, and the
# record's centre, from the words it ran on (Hz).
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
    (
        first = (first_cycles, half_duration),
        second = (second_cycles, half_duration),
        center = convert(typeof(1.0s), center / fs),
    )
end

# The raw PLL phase error (cycles) of `record`'s prompt against the replica that
# produced it, and the carrier phase polarity it was read with: the Costas or the
# four-quadrant discriminator the inner loop reads it with, but never the
# NCO-referenced loop's prediction to the landing sample, which refers to another
# replica.
@inline _prompt_phase_error(record::LoopRecord) = (
    Float64(pll_disc(record.signal, record.filtered_correlator; polarity = record.carrier_phase_polarity)),
    record.carrier_phase_polarity,
)

# The accumulator with the span from the previous record's prompt to this one's
# added: the replica's advance over it — the second half of the previous record and
# the first of this one — plus the change of the phase error from the previous
# prompt to this one, the incoming signal's advance against the replica over the
# same span. The difference is taken as it is, not wrapped: over consecutive spans
# it telescopes to the last phase error minus the first, each within the
# discriminator's range, so a difference beyond that range, which prompt noise
# makes, is undone by the next span, where the FLL's wrapped rotation left a half
# or a whole cycle. A record without a previous prompt (the first one, or one whose
# carrier phase polarity changed), one whose phase error was read with another
# polarity than the previous prompt's, as the two refer to phases half a cycle
# apart, and one without a previous half record: the span is left out rather than
# taken as a zero.
@inline function _accumulate_carrier_phase(
    acc::CarrierPhaseAccumulator,
    previous_replica_half,
    previous_phase_error,
    replica_halves,
    phase_error,
    record::LoopRecord,
)
    previous_cycles, previous_duration = previous_replica_half
    first_cycles, first_duration = replica_halves.first
    previous_error, previous_polarity = previous_phase_error
    current_error, polarity = phase_error
    (
        iszero(record.previous_prompt) ||
        isnan(previous_cycles) ||
        isnan(previous_error) ||
        polarity != previous_polarity
    ) && return acc
    duration = previous_duration + first_duration
    midpoint = replica_halves.center - duration / 2
    CarrierPhaseAccumulator(
        acc.count + 1,
        acc.cycles + previous_cycles + first_cycles + (current_error - previous_error),
        acc.duration + duration,
        acc.time_moment + duration * midpoint,
    )
end

# One record through the satellite's loop: the inner loop out of the vector loop,
# the vector step with the discriminators accumulated in it.
@inline function _step_satellite(
    estimator::VectorPLLAndDLL,
    state::SatVectorPLLAndDLL,
    record::LoopRecord,
    words,
    landing_sample::Int64,
)
    replica_halves = _replica_halves(record, words)
    previous_replica_half = replica_halves.second
    previous_phase_error = _prompt_phase_error(record)
    if !state.vt_on
        inner, carrier_doppler, code_doppler =
            step_loop(estimator.inner, state.inner, record, words, landing_sample)
        return SatVectorPLLAndDLL(state; inner, previous_replica_half, previous_phase_error),
        carrier_doppler,
        code_doppler
    end
    inner, carrier_doppler, code_doppler, dll_discriminator = _step_vector_loop(
        estimator.inner,
        state.inner,
        record,
        words,
        landing_sample,
        state.code_freq_update,
        state.carrier_freq_update,
        estimator.navigation.cycle_time,
    )
    SatVectorPLLAndDLL(
        state;
        inner,
        code_discr_acc = state.code_discr_acc .+ (1, dll_discriminator),
        carrier_phase_acc = _accumulate_carrier_phase(
            state.carrier_phase_acc,
            state.previous_replica_half,
            state.previous_phase_error,
            replica_halves,
            previous_phase_error,
            record,
        ),
        previous_replica_half,
        previous_phase_error,
    ),
    carrier_doppler,
    code_doppler
end

# The inner loop's record, in the vector loop: the discriminators exactly as the
# inner loop measures them (`_record_discriminators`; the NCO-referenced loop's
# phase error keeps its prediction to the landing sample), the carrier filter
# stepped with the filter's carrier correction in its FLL slot at the vector FLL
# bandwidth, capped against the navigation `cycle_time`, the code filter frozen
# and the code correction in place of its output. Returns the inner state, both
# Dopplers and the DLL discriminator to accumulate. The FLL is not formed: its
# slot takes the filter's correction.
@inline function _step_vector_loop(
    estimator,
    state,
    record::LoopRecord,
    words,
    landing_sample::Int64,
    code_freq_update,
    carrier_freq_update,
    cycle_time,
)
    discriminators = _record_discriminators(estimator, state, record, words, landing_sample; form_fll = false)
    integration_time = discriminators.integration_time
    # At the narrow PLL bandwidth, whatever the inner loop's stage, with the FLL
    # branch at the vector FLL's own, capped against the cycle that sets its input.
    carrier_filter_output, carrier_loop_filter = filter_loop(
        state.carrier_loop_filter,
        (discriminators.phase_error, carrier_freq_update),
        integration_time,
        (
            effective_carrier_loop_filter_bandwidth(state.bandwidths.narrow_carrier, integration_time),
            effective_vector_fll_loop_filter_bandwidth(state.bandwidths.vector_fll, cycle_time),
        ),
    )
    carrier_doppler, code_doppler = aid_dopplers(
        record.signal,
        state.init_carrier_doppler,
        state.init_code_doppler,
        carrier_filter_output,
        code_freq_update,
    )
    # The staging keeps advancing only so that `phase_lock_indicator` stays live:
    # in the vector loop the carrier filter runs at the narrow bandwidth whatever
    # the stage, and a release restarts the staging.
    _stepped_state(
        state,
        carrier_loop_filter,
        state.code_loop_filter,
        discriminators.center,
        _advance_staging(state.staging, record),
    ),
    carrier_doppler,
    code_doppler,
    discriminators.code_error
end
