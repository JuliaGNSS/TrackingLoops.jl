"""
$(SIGNATURES)

C/N₀ against a **measured noise reference**: per integration,

```
Ĉ/N₀ = ⟨|P|²⟩ / N̂₀ − 1/T
```

with `N̂₀` the signal's noise density from its [`AbstractNoiseDensityEstimator`](@ref) and
`T` that integration's own integration time. The ring averages the per-integration terms
and [`estimate_cn0`](@ref) converts the mean once.

Unlike [`NWPRCN0Estimator`](@ref) this is **non-coherent**: no bit sync, no
window, no `M`, no fallback, no saturation ceiling, and immune to residual
carrier-phase error. It therefore works uniformly on **every** signal, including
GPS L1C-D, Galileo E1B and secondary-coded signals before sync (JuliaGNSS/Tracking.jl#217).

# What it needs

A noise density for its **own signal** (see [Noise estimation](@ref) for why per
signal). On the sample-driven path `Tracking.TrackState` provisions a
[`CorrelatorNoiseEstimator`](@ref) automatically (see
[`requires_noise_density`](@ref)); on a correlator-ingest path fill the same type
with [`append_noise_observation!`](@ref). With **no** source configured, `update!`
throws: a silent substitution would hide a backend that never populates the
reference. While a configured window is still **empty**, post-correlation processing skips the
update (Tracking.jl warns once per signal) and [`estimate_cn0`](@ref) reports
`-Inf dB-Hz`.

# Deliberate properties

  - **No `fallback`.** One integration plus a density is a valid, if noisy, estimate
    on every signal.
  - **Only the average is floored, never the per-integration term.** Negative terms
    at low C/N₀ keep the mean unbiased; clamping them would reintroduce a floor
    like [`MomentsCN0Estimator`](@ref)'s.

# The one bias it carries

The reference despreads with a *wrong* PRN, so it also collects the tracked
satellite's own power, `ε_self/N₀ = C/f_chip`, which NWPR cancels in its ratio:
C/N₀ reads low by 0.013 dB at 35 dB-Hz, 0.042 at 40, 0.13 at 45 and 0.40 at 50
on GPS L1 C/A (≈10× less on the 10.23 Mcps signals). It is not corrected, since
`N̂₀ ← N̂₀ − Σᵢ Ĉᵢ/f_chip` would couple every satellite's C/N₀ to every other's.
Below 40 dB-Hz, where lock and loss are decided, it sits under NWPR's own
+0.05 dB bias.

# Fields / configuration

  - `num_integrations` — how many integrations the estimate averages over, i.e. the memory
    of the estimator (100 by default, ~100 ms at GPS L1 C/A).
  - `buffered_cn0`, `write_index`, `num_filled` — the ring of per-integration
    C/N₀ terms in **linear Hz**, written in place, plus its position and fill.
"""
struct NoiseReferencedCN0Estimator <: AbstractCN0Estimator
    num_integrations::Int
    buffered_cn0::Vector{Float64}
    write_index::Int
    num_filled::Int
end

"""
$(SIGNATURES)

Construct a fresh [`NoiseReferencedCN0Estimator`](@ref) averaging over the last
`num_integrations` integrations.
"""
function NoiseReferencedCN0Estimator(; num_integrations::Int = 100)
    num_integrations >= 1 ||
        throw(ArgumentError("num_integrations must be at least 1, got $num_integrations"))
    NoiseReferencedCN0Estimator(num_integrations, zeros(Float64, num_integrations), 0, 0)
end

length(estimator::NoiseReferencedCN0Estimator) = estimator.num_filled
get_write_index(estimator::NoiseReferencedCN0Estimator) = estimator.write_index

"""
$(SIGNATURES)

`true`: a signal carrying this estimator is provisioned a
[`CorrelatorNoiseEstimator`](@ref); see [`requires_noise_density`](@ref).
"""
requires_noise_density(::Type{NoiseReferencedCN0Estimator}) = true

"""
$(SIGNATURES)

Add one integration's `prompt` to the ring: `|P|²/N̂₀ − 1/T` in linear Hz, with the
density and the integration's own integration time taken from `context`. The term is
**not** clamped (see [`NoiseReferencedCN0Estimator`](@ref)).
"""
function update!(estimator::NoiseReferencedCN0Estimator, prompt, context::CN0UpdateContext)
    cn0 = ustrip(
        Hz,
        uconvert(Hz, abs2(prompt) / context.noise_density - 1 / context.coherent_integration_time),
    )
    buffer_length = Base.length(estimator.buffered_cn0)
    next_index = mod(estimator.write_index, buffer_length) + 1
    estimator.buffered_cn0[next_index] = cn0
    NoiseReferencedCN0Estimator(
        estimator.num_integrations,
        estimator.buffered_cn0,
        next_index,
        min(estimator.num_filled + 1, buffer_length),
    )
end

# A missing density or integration time is `Nothing` in the context's type, so
# both are caught by dispatch on the first integration. The third method resolves the
# ambiguity of `{…,Nothing,Nothing}` and reports the missing source, the root
# cause.
@noinline update!(
    ::NoiseReferencedCN0Estimator,
    prompt,
    ::CN0UpdateContext{<:AbstractGNSSSignal,<:Unsigned,Nothing},
) = _throw_noise_ref_needs_density()
@noinline update!(
    ::NoiseReferencedCN0Estimator,
    prompt,
    ::CN0UpdateContext{<:AbstractGNSSSignal,<:Unsigned,Nothing,Nothing},
) = _throw_noise_ref_needs_density()
@noinline update!(
    ::NoiseReferencedCN0Estimator,
    prompt,
    ::CN0UpdateContext{<:AbstractGNSSSignal,<:Unsigned,<:Any,Nothing},
) = _throw_noise_ref_needs_coherent_integration_time()

@noinline _throw_noise_ref_needs_density() = throw(
    ArgumentError(
        "NoiseReferencedCN0Estimator needs a noise density for its signal, but no " *
        "AbstractNoiseDensityEstimator is configured for it. Pass " *
        "`cn0_estimator = NWPRCN0Estimator()` if you are feeding externally " *
        "supplied correlator outputs without a noise observation, or give " *
        "`TrackState` a `noise_estimators` entry for this signal and fill it " *
        "with `append_noise_observation!`.",
    ),
)

@noinline _throw_noise_ref_needs_coherent_integration_time() = throw(
    ArgumentError(
        "NoiseReferencedCN0Estimator has this signal's noise density but no integration " *
        "time: it applies `T` per integration (`|P|²/N̂₀ - 1/T`), so it cannot add one " *
        "without it. Build the context with both — " *
        "`CN0UpdateContext(signal, bit_synchronizer, num_code_periods; noise_density, " *
        "coherent_integration_time)` — which is what the tracking loop does; " *
        "`coherent_integration_time` is that integration's own " *
        "`num_integrated_samples / sampling_frequency`.",
    ),
)

"""
$(SIGNATURES)

A bare prompt stream carries no noise density and no integration time, so this
estimator has no two-argument form — use the three-argument
[`update!(::AbstractCN0Estimator, ::Any, ::CN0UpdateContext)`](@ref), which is
what the tracking loop calls, or [`MomentsCN0Estimator`](@ref) /
[`NWPRCN0Estimator`](@ref) for folding a captured prompt stream by hand.
"""
@noinline update!(::NoiseReferencedCN0Estimator, prompt) = throw(
    ArgumentError(
        "NoiseReferencedCN0Estimator cannot be updated from a bare prompt: it needs the " *
        "signal's noise density and the predetection integration's integration time, both of " *
        "which live in the CN0UpdateContext. Call the three-argument `update!`.",
    ),
)

"""
$(SIGNATURES)

Mean of the buffered per-integration terms, converted once with `dBHz`.

`coherent_integration_time` is **ignored**: `T` was applied per integration in `update!`, so an
integration lengthened by `Tracking.set_preferred_num_code_periods_to_integrate!` is
handled correctly beside shorter ones. The argument stays for interface
uniformity.

An empty ring, a mean that has not cleared zero, and a **non-finite** mean all
report `-Inf dB-Hz`, never `NaN` (see [`NoCN0Estimator`](@ref) for why). The
explicit `isfinite` catches a `NaN` prompt, which passes `mean_cn0 <= 0` (a zero
floor is already kept out of the ring by [`noise_density_and_ready`](@ref)).
"""
function estimate_cn0(estimator::NoiseReferencedCN0Estimator, coherent_integration_time)
    filled = length(estimator)
    filled == 0 && return dBHz(0.0Hz)
    total = 0.0
    @inbounds for i = 1:filled
        total += estimator.buffered_cn0[i]
    end
    mean_cn0 = total / filled
    isfinite(mean_cn0) && mean_cn0 > 0 || return dBHz(0.0Hz)
    dBHz(mean_cn0 * Hz)
end
