# Running sums over a `CorrelatorNoiseEstimator`'s window so neither appending nor
# reading walks it: `span` drives the trim, `weighted_density / looks` is
# `get_noise_density`.
#
# `stale` counts appends since the last exact recomputation. Incremental `Float64`
# add/subtract drifts, and a producer may mix e.g. a 0.2 s pre-averaged entry with
# 1 ms ones, so the totals are rebuilt once per window's worth of appends (O(1)
# amortised).
#
# Immutable and isbits, held in the estimator's single `Ref`: an update is one write
# to one cache line, and the estimator itself is never rebuilt. It is a cache derived
# exactly from `buffered`, not independent state.
struct NoiseWindowTotals{D,T}
    span::T
    weighted_density::D
    looks::Int
    stale::Int
end

NoiseWindowTotals{D,T}() where {D,T} = NoiseWindowTotals{D,T}(zero(T), zero(D), 0, 0)

"""
$(SIGNATURES)

The signal's noise reference, measured by **despreading an untracked PRN**. The
only shipped [`AbstractNoiseEstimator`](@ref); on a hardware path fill it with
[`append_noise_observation!`](@ref) instead of [`update_noise!`](@ref).

# Why a despread and not a power meter

The reference runs the same quantise → downconvert → despread → accumulate path as
the prompt, with the same kernel and (keyed by signal) the same code. So the
measured `N₀` already contains quantisation loss, the quantiser's operating point
under load, AGC scaling and a CBOC replica's code amplitude, with no per-backend
model; and the despread evaluates the signal's own `∫ S_I(f)·|G(f)|² df` (see
[Noise estimation](@ref)). A `Σ|x|²` power meter gets both wrong.

# Open loop

No feedback of any kind: the reference runs at the band's nominal IF plus a random
dither, at a random code phase, rotating over **all** PRNs of the family. It reads
nothing from satellite state, so it works with zero satellites tracked, and on an
FPGA a noise channel is a subset of a tracking channel.

# Why code phase and Doppler are randomised

At a fixed code phase and zero Doppler, the relative phase against an untracked but
present signal (spoofed, or visible and not acquired) is frozen, since it drifts
only at the code Doppler `f_d/1540`. Such a signal has a `3 × 1.5 / 1023 ≈ 0.44 %`
chance per PRN of sitting in one of the three taps and then adds
`T·(C/N₀)/3 ≈ 10.5·N₀` (45 dB-Hz) to every observation on that PRN indefinitely
(≈ +1.5 dB on `N̂₀` after the rotation's dilution). Fresh draws per sub-integration
turn that into independent per-observation trials: ≈ 0.07 dB residual for a full
sky at 45 dB-Hz. Hitting a tracked satellite's peak (≈ 0.6 %, ≈ 0.045 dB) is an
order of magnitude below that, hence no untracked-PRN restriction; a constant
32-PRN pool also keeps the dilution from worsening as more satellites are acquired.

Neither draw biases the measurement: `N̂₀ = |B|²/(N·A_c²·f_s)` is unbiased for any
phase, and a ±5 kHz dither smears the spectral weighting by 0.25 % of a 2 MHz main
lobe. On an FPGA a free-running code generator gives an arbitrary phase for free.

# Fields / configuration

  - `window_duration`: how far back the sliding window reaches (1 s). Longer means
    lower variance but more smear across AGC changes. The default holds
    `K_n ≈ 1000` looks per tap at one sub-integration per code period, ≤ 0.08 dB
    from a variance-free reference at every C/N₀.
  - `tap_code_shift`: tap spacing in chips (1.5). Taps help only if independent:
    `corr(|Bᵢ|²,|Bⱼ|²) = |ρᵢⱼ|²`, so at ±0.5 chip three taps are worth 2.25 looks,
    at ≥ 1 chip 2.98. 1.5 chips sits in the autocorrelation null, clear of the 1-
    and 2-chip sidelobes.
  - `carrier_dither`: half-width of the uniform offset added to the nominal IF per
    sub-integration (5 kHz, the terrestrial Doppler spread). Zero is useful only to
    isolate the code-phase draw in a test.
  - `rng`: source of both draws, seeded `Xoshiro(0)` by default and advanced in
    place. Scattering the draws is the whole requirement (an attacker cannot
    observe the chunk grid), so a seeded, repeatable default costs nothing. Pass
    `Random.default_rng()` for a task-local stream (e.g. across threads).
    Repeatable on one Julia version only: do not tune a tolerance to a particular
    draw; design against the `1/√(3K)` spread of `N̂₀` over `K` observations.
  - `buffered`: the sliding window, a length-managed FIFO written in place, so the
    struct is never rebuilt and per-signal state can live in an immutable
    `Tracking.TrackState`.
  - `totals`: running sums of the window (see `NoiseWindowTotals`), keeping append
    and read **O(1)**; an O(K) scan per call would make each chunk's cost grow with
    the `K` (≈ 2500 at 1 s and 0.4 ms chunks) the accuracy is bought with.

The sub-integration length is **not** a field: it is the signal's primary code
period. Coherent integration buys nothing for noise power (`Var(|B|²) = (E|B|²)²`),
so only the number of looks matters, yet shorter dumps would cost SIMD efficiency
and a new kernel (losing bit-identical arithmetic), lose the full-period Gold code's
DC balance (an ADC offset would bias the despread), and break cadence parity with a
hardware correlator dumping on the code epoch. A long window buys the variance back
cheaply (~1000 `Float64` per signal per second).
"""
struct CorrelatorNoiseEstimator{D,T,R<:AbstractRNG} <: AbstractNoiseEstimator
    window_duration::typeof(1.0s)
    tap_code_shift::Float64
    carrier_dither::typeof(1.0Hz)
    buffered::Vector{NoiseObservation{D,T}}
    totals::Base.RefValue{NoiseWindowTotals{D,T}}
    rng::R
end

"""
$(SIGNATURES)

Construct a [`CorrelatorNoiseEstimator`](@ref); see the type for the parameters.

The window is `sizehint!`-ed to 4× the code-period observations it expects: with
less headroom Julia periodically shifts the FIFO's front offset back and
reallocates, so `push!`/`popfirst!` would not be allocation-free.

`num_ants` must match the antenna count of the signal group. Above `NumAnts(1)` the
window holds the `M×M` noise covariance `R̂`, which each satellite reduces to its
own floor `wᴴR̂w` through its beamforming weights (see [`update_noise!`](@ref),
[`AbstractPostCorrFilter`](@ref)). `TrackState` provisions the count when
`noise_estimators` is `nothing` and rejects a mismatch otherwise.
"""
function CorrelatorNoiseEstimator(;
    window_duration = 1.0s,
    tap_code_shift = 1.5,
    carrier_dither = 5000.0Hz,
    num_ants::NumAnts = NumAnts(1),
    rng::AbstractRNG = Xoshiro(0),
)
    window_duration > zero(window_duration) ||
        throw(ArgumentError("window_duration must be positive, got $window_duration"))
    tap_code_shift > 0 ||
        throw(ArgumentError("tap_code_shift must be positive, got $tap_code_shift"))
    carrier_dither >= zero(carrier_dither) ||
        throw(ArgumentError("carrier_dither must not be negative, got $carrier_dither"))
    D = _density_type_for_num_ants(num_ants)
    buffered = NoiseObservation{D,typeof(1.0s)}[]
    # 4× the 1 ms sub-integration count; see the docstring.
    sizehint!(buffered, 4 * max(1, round(Int, window_duration / 1.0ms)) + 1)
    CorrelatorNoiseEstimator(
        uconvert(s, float(window_duration)),
        Float64(tap_code_shift),
        uconvert(Hz, float(carrier_dither)),
        buffered,
        Ref(NoiseWindowTotals{D,typeof(1.0s)}()),
        rng,
    )
end

# The antenna count, derived from `D` so window and correlator cannot disagree.
@inline _num_ants(::CorrelatorNoiseEstimator{D}) where {D} = _num_ants_of_density_type(D)

"""
$(SIGNATURES)

Append `observation` to the signal's sliding window, dropping entries off the
front while the remainder still spans `window_duration`. Returns `estimator`.
Amortised **O(1)**.

The window is bounded in **time**, not count, so one producer may report 16-chip
accumulations and another a single pre-averaged 0.2 s figure under the same
configuration.

Any [`NoiseObservation`](@ref) is converted to the window's field types (free for
builder output), so a hand-assembled or `Float32` one is not silently dropped by
falling through to the abstract no-op.
"""
function append_noise_observation!(
    estimator::CorrelatorNoiseEstimator{D,T},
    observation::NoiseObservation,
) where {D,T}
    observation = convert(NoiseObservation{D,T}, observation)
    buffered = estimator.buffered
    totals = estimator.totals
    push!(buffered, observation)
    _add_observation!(totals, observation)
    _trim_noise_window!(buffered, totals, estimator.window_duration)
    _refresh_totals_if_stale!(buffered, totals)
    estimator
end

@inline function _add_observation!(totals::Base.RefValue{<:NoiseWindowTotals}, observation)
    t = totals[]
    totals[] = typeof(t)(
        t.span + observation.duration,
        t.weighted_density + observation.num_sub_integrations * observation.noise_density,
        t.looks + observation.num_sub_integrations,
        t.stale,
    )
    nothing
end

@inline function _drop_observation!(totals::Base.RefValue{<:NoiseWindowTotals}, observation)
    t = totals[]
    totals[] = typeof(t)(
        t.span - observation.duration,
        t.weighted_density - observation.num_sub_integrations * observation.noise_density,
        t.looks - observation.num_sub_integrations,
        t.stale,
    )
    nothing
end

# Keep the window minimal but never shorter than `window_duration`, and never empty
# (a single observation longer than the window still counts).
@inline function _trim_noise_window!(
    buffered::Vector{<:NoiseObservation},
    totals::Base.RefValue{<:NoiseWindowTotals},
    window_duration,
)
    @inbounds while length(buffered) > 1 &&
                    totals[].span - buffered[1].duration >= window_duration
        _drop_observation!(totals, buffered[1])
        popfirst!(buffered)
    end
    nothing
end

# Exact recomputation once per window's worth of appends; see `NoiseWindowTotals`.
@inline function _refresh_totals_if_stale!(
    buffered::Vector{<:NoiseObservation},
    totals::Base.RefValue{<:NoiseWindowTotals},
)
    t = totals[]
    totals[] = typeof(t)(t.span, t.weighted_density, t.looks, t.stale + 1)
    t.stale + 1 < length(buffered) && return nothing
    _refresh_totals!(buffered, totals)
end

function _refresh_totals!(
    buffered::Vector{<:NoiseObservation},
    totals::Base.RefValue{NoiseWindowTotals{D,T}},
) where {D,T}
    span = zero(T)
    weighted_density = zero(D)
    looks = 0
    @inbounds for i in eachindex(buffered)
        observation = buffered[i]
        span += observation.duration
        weighted_density += observation.num_sub_integrations * observation.noise_density
        looks += observation.num_sub_integrations
    end
    totals[] = NoiseWindowTotals{D,T}(span, weighted_density, looks, 0)
    nothing
end

"""
$(SIGNATURES)

The window's mean density weighted by `num_sub_integrations` (`M`), or `nothing`
while it is empty. **O(1)**, read off the running totals.

An entry's relative variance is `1/M` (its independent looks), so entries of
different dump counts combine correctly only weighted by `M`.
"""
function get_noise_density(estimator::CorrelatorNoiseEstimator)
    totals = estimator.totals[]
    totals.looks == 0 && return nothing
    totals.weighted_density / totals.looks
end

noise_density_type(::CorrelatorNoiseEstimator{D}) where {D} = D

# `looks` is the running sum of `num_sub_integrations`, the independent-look count.
noise_window_looks(estimator::CorrelatorNoiseEstimator) = estimator.totals[].looks

"""
$(SIGNATURES)

Number of observations in the window. Diagnostic only: it varies with the
producer's dump cadence.
"""
Base.length(estimator::CorrelatorNoiseEstimator) = length(estimator.buffered)

"""
$(SIGNATURES)

Measure this signal's noise over samples `first_sample:last_sample` of
`measurement`, append the observations and return `estimator`. Forwards to
[`despread_noise!`](@ref) on `context.downconvert_and_correlator`; a loop process
uses [`append_noise_observation!`](@ref) instead.
"""
update_noise!(
    estimator::CorrelatorNoiseEstimator,
    measurement,
    first_sample::Integer,
    last_sample::Integer,
    context::NoiseUpdateContext,
) = despread_noise!(
    context.downconvert_and_correlator,
    estimator,
    measurement,
    first_sample,
    last_sample,
    context,
)

"""
    despread_noise!(backend, estimator, measurement, first_sample, last_sample, context)

The software fill path of a [`CorrelatorNoiseEstimator`](@ref): despread an
untracked PRN over the slice with `backend`'s kernel and append the observations.
Implemented by Tracking.jl for its backends.
"""
function despread_noise! end

# Pool the taps of one despread (independent looks at ≥ 1 chip spacing): the scalar
# power for one antenna, the spatial covariance for an array.
@inline function _pool_taps(accumulators, ::NumAnts{1})
    power = 0.0
    for tap in accumulators
        power += abs2(tap)
    end
    power
end

@inline function _pool_taps(accumulators, ::NumAnts{M}) where {M}
    covariance = zero(SMatrix{M,M,ComplexF64,M * M})
    for tap in accumulators
        covariance += tap * tap'
    end
    covariance
end

# Next PRN of the rotation, carried in the newest observation (1 when empty).
@inline function _next_noise_prn(
    estimator::CorrelatorNoiseEstimator,
    signal_type::AbstractGNSSSignal,
)
    num_prns = size(get_codes(signal_type), 2)
    previous = isempty(estimator.buffered) ? 0 : Int(last(estimator.buffered).prn)
    mod(previous, num_prns) + 1
end
