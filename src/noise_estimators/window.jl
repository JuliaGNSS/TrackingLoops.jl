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

The signal's noise reference, measured by **despreading a PRN at a random code
phase and Doppler**. The only shipped [`AbstractNoiseEstimator`](@ref). A host
fills it with the despreads [`noise_despreads`](@ref) plans, a software one through
[`update_noise!`](@ref), or appends the observations it builds itself with
[`append_noise_observation!`](@ref).

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
turn that into independent per-observation trials (a hit needs both draws to land,
so the two randomisations multiply): ≈ 0.07 dB residual for a full sky at
45 dB-Hz. Landing within ±1 chip of a tracked satellite's peak (≈ 0.6 %,
≈ 0.045 dB for the one observation it touches) is an order of magnitude under the
whole-sky leakage already carried uncorrected, hence no untracked-PRN restriction;
a constant 32-PRN pool also keeps the dilution from worsening as more satellites
are acquired.

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

The window is `sizehint!`-ed to 4× the 1 ms observations in `window_duration`: with
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
fills the window through [`noise_despreads`](@ref) instead.
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

The software fill path of a [`CorrelatorNoiseEstimator`](@ref): despread each of the
slice's [`noise_despreads`](@ref) with [`noise_correlator`](@ref)'s taps on `backend`'s
kernel, the one the prompt goes through, and append it with
[`append_despread_noise!`](@ref). Implemented by Tracking.jl for its backends, which
read the sampling and intermediate frequencies off `measurement`.
"""
function despread_noise! end

"""
    NoiseDespread

One sub-integration of a [`CorrelatorNoiseEstimator`](@ref)'s reference, as
[`noise_despreads`](@ref) plans it: despread samples
`first_sample:first_sample + num_samples - 1` with `signal`'s code for `prn`, starting
at `code_phase` chips, at the nominal `code_frequency` and at `carrier_frequency`, the
band's intermediate frequency plus the dither, with no carrier phase. A software host
runs it on its kernel; a hardware host programs a noise channel with it. Either hands
the taps to [`append_despread_noise!`](@ref).
"""
struct NoiseDespread{S<:AbstractGNSSSignal,K,C,F}
    signal::S
    prn::Int
    first_sample::Int
    num_samples::Int
    code_phase::Float64
    code_frequency::K
    carrier_frequency::C
    sampling_frequency::F
end

# The sub-integrations of one slice, drawn as they are iterated.
struct NoiseDespreads{S<:AbstractGNSSSignal,R<:AbstractRNG,F,I}
    signal::S
    prn::Int
    first_sample::Int
    num_samples::Int
    num_sub::Int
    sampling_frequency::F
    intermediate_frequency::I
    carrier_dither::typeof(1.0Hz)
    rng::R
end

"""
    noise_despreads(estimator::CorrelatorNoiseEstimator, signal::AbstractGNSSSignal,
                    first_sample, last_sample, sampling_frequency, intermediate_frequency)

The sub-integrations a host despreads to measure `signal`'s noise over samples
`first_sample:last_sample`, sampled at `sampling_frequency` on a band at
`intermediate_frequency`: an iterator of [`NoiseDespread`](@ref)s, empty for an empty
slice. Iterate it once, in order, and append each despread with
[`append_despread_noise!`](@ref).

The slice is split into `num_sub` **equal** sub-integrations,

```julia
num_sub = max(1, round(Int, slice_duration / code_period))
```

with `code_period` the primary code period of `signal` (see
[`CorrelatorNoiseEstimator`](@ref) for why it is derived, not configured), and at most
one per sample. Equal slices waste no remainder; the `max(1, …)` keeps a code longer
than the chunk (10 ms code, 1 ms chunks) from never yielding an observation. Each
sub-integration is its own observation, so the window's time span does not follow the
Doppler update rate.

All of them despread the next PRN of the rotation over `signal`'s whole code family,
tracked or not, carried in the newest observation of the window. Each draws a fresh
random code phase and a fresh carrier within ±`carrier_dither` of the intermediate
frequency from the estimator's `rng` as it is iterated, so every window entry is an
independent trial. A despread follows no code-period boundary: with a wrong PRN at a
wrong phase there is nothing to accumulate coherently, so `N̂₀ = |B|²/(N·A_c²·f_s)` is
unbiased for any `N` and start phase, and no partial accumulator is carried across
calls. Allocates nothing.
"""
function noise_despreads(
    estimator::CorrelatorNoiseEstimator,
    signal::AbstractGNSSSignal,
    first_sample::Integer,
    last_sample::Integer,
    sampling_frequency,
    intermediate_frequency,
)
    num_samples = max(0, Int(last_sample) - Int(first_sample) + 1)
    code_period = get_code_length(signal) / get_code_frequency(signal)
    slice_duration = num_samples / sampling_frequency
    num_code_periods = round(Int, uconvert(NoUnits, slice_duration / code_period))
    num_sub = num_samples == 0 ? 0 : min(num_samples, max(1, num_code_periods))
    NoiseDespreads(
        signal,
        _next_noise_prn(estimator, signal),
        Int(first_sample),
        num_samples,
        num_sub,
        sampling_frequency,
        intermediate_frequency,
        estimator.carrier_dither,
        estimator.rng,
    )
end

Base.length(despreads::NoiseDespreads) = despreads.num_sub

@inline function Base.iterate(despreads::NoiseDespreads, sub = 1)
    sub > despreads.num_sub && return nothing
    (; signal, first_sample, num_samples, num_sub, rng) = despreads
    slice_start = first_sample + div((sub - 1) * num_samples, num_sub)
    slice_stop = first_sample + div(sub * num_samples, num_sub) - 1
    # Code phase first, then the carrier: one draw each per sub-integration.
    code_phase = rand(rng) * get_code_length(signal)
    carrier_frequency =
        despreads.intermediate_frequency + (2 * rand(rng) - 1) * despreads.carrier_dither
    despread = NoiseDespread(
        signal,
        despreads.prn,
        slice_start,
        slice_stop - slice_start + 1,
        code_phase,
        get_code_frequency(signal),
        carrier_frequency,
        despreads.sampling_frequency,
    )
    despread, sub + 1
end

"""
    noise_correlator(estimator::CorrelatorNoiseEstimator) -> EarlyPromptLateCorrelator

The correlator a [`NoiseDespread`](@ref) runs through: three taps `tap_code_shift` chips
apart, one column per antenna. At that spacing the taps are independent looks, which
[`append_despread_noise!`](@ref) pools.
"""
noise_correlator(estimator::CorrelatorNoiseEstimator) = EarlyPromptLateCorrelator(;
    num_ants = NumAnts(estimator),
    preferred_early_late_to_prompt_code_shift = estimator.tap_code_shift,
)

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

"""
    append_despread_noise!(estimator::CorrelatorNoiseEstimator, despread::NoiseDespread,
                           accumulators) -> estimator

Append the observation of one of [`noise_despreads`](@ref): `accumulators` are the taps
of [`noise_correlator`](@ref) after running `despread`. The taps are independent looks,
so they pool into one power sum, or into the spatial covariance `Σ b·bᴴ` for an array,
which each satellite reduces to its own floor `wᴴR̂w` at C/N₀ time; the observation
spans the despread's duration, which every tap integrated, under its PRN. Allocates
nothing.
"""
@inline function append_despread_noise!(
    estimator::CorrelatorNoiseEstimator,
    despread::NoiseDespread,
    accumulators,
)
    num_taps = length(accumulators)
    (; num_samples, sampling_frequency) = despread
    append_noise_observation!(
        estimator,
        _noise_observation(
            _pool_taps(accumulators, NumAnts(estimator)),
            num_taps,
            num_taps * num_samples,
            sampling_frequency,
            get_code_amplitude(despread.signal),
            despread.prn,
            num_samples / sampling_frequency,
        ),
    )
end
