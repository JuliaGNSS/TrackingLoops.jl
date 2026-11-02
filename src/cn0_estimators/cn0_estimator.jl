"""
$(SIGNATURES)

Abstract supertype for CN0 (carrier-to-noise-density ratio) estimators.
Each `Tracking.TrackedSignal` holds one estimator instance, stored in a type
parameter — pass any subtype instance as the `cn0_estimator` keyword of
`Tracking.TrackedSignal` / `Tracking.TrackedSat` to replace the default
[`NoiseReferencedCN0Estimator`](@ref); see [`default_cn0_estimator`](@ref) for which to
pick when. Custom estimators subtype this and implement `TrackingLoops.update!` and
[`estimate_cn0`](@ref), plus [`requires_noise_density`](@ref) if they read a
measured noise floor.
"""
abstract type AbstractCN0Estimator end

"""
$(SIGNATURES)

Per-integration side information handed to `update!` alongside the prompt:
what the tracking loop knows about the integration that an estimator cannot recover
from the prompt stream alone — chiefly the **navigation-bit grid**, which lets an
estimator sum prompts coherently without straddling a bit flip
([`NWPRCN0Estimator`](@ref) does).

Fields:

  - `signal` — the signal the integration belongs to.
  - `num_code_periods` — primary-code periods this integration spanned. One for the
    default configuration; more when the correlate step was lengthened by
    `Tracking.set_preferred_num_code_periods_to_integrate!` or an external
    producer handed over longer integrations.
  - `num_code_periods_per_bit` — code periods that form one navigation bit of
    `signal`; `20` for GPS L1 C/A, `1` for GPS L1C-D / Galileo E1B, and `0` for a
    pilot, which carries no data and therefore has **no** bit grid — post-sync
    its prompts stay coherent for as long as the loops hold them.
  - `bit_code_period_index` — this integration's offset inside the open navigation
    bit (`0` = starts a fresh bit). `-1` when the prompt cannot be summed
    coherently against the grid: before bit / secondary-code sync, and, for a
    secondary-coded signal, for an integration correlated without the overlay wipe-off
    a sync in the same batch just established (see `_process_correlator_bank_output!`).
    Such an integration's code periods still count towards the grid.
  - `bit_synchronizer` — the signal's `BitSynchronizer` as of *before* this integration:
    the soft bits decoded so far, the open coherent bit accumulator, the lock
    polarity and the secondary-code phase.
  - `noise_density` — **this signal's** measured noise density `N₀` (dimension
    `1/Hz`), read by [`NoiseReferencedCN0Estimator`](@ref). `Nothing` (in the type, not
    as a sentinel) means no [`AbstractNoiseDensityEstimator`](@ref) is configured for
    the signal, so call sites monomorphise. A configured source whose window is
    still empty never reaches here — post-correlation processing skips the update — so otherwise
    this is always a plain scalar.
  - `coherent_integration_time` — this integration's own `T`, so integrations of different lengths
    in one ring are each handled correctly. `nothing` on the bare-prompt-stream
    path.
"""
struct CN0UpdateContext{S<:AbstractGNSSSignal,B<:Unsigned,N,T}
    signal::S
    num_code_periods::Int
    num_code_periods_per_bit::Int
    bit_code_period_index::Int
    bit_synchronizer::BitSynchronizer{B}
    noise_density::N
    coherent_integration_time::T
end

# Positional convenience for the first five fields, defaulting the two only a
# noise-referenced estimator reads.
@inline CN0UpdateContext(
    signal::AbstractGNSSSignal,
    num_code_periods::Integer,
    num_code_periods_per_bit::Integer,
    bit_code_period_index::Integer,
    bit_synchronizer::BitSynchronizer,
) = CN0UpdateContext(
    signal,
    Int(num_code_periods),
    Int(num_code_periods_per_bit),
    Int(bit_code_period_index),
    bit_synchronizer,
    nothing,
    nothing,
)

# Build the context from the post-correlation state. `bit_synchronizer` must be the
# one from *before* this integration was processed, so `bit_code_period_index` is this
# integration's offset. `bit_sync_usable = false` forces the "no bit grid" marker (see
# `drop_prompt` in `_process_correlator_bank_output!`).
#
# The keyword form is for callers building a context by hand; post-correlation processing
# calls the positional core below, because a keyword call allocates per call on
# Julia 1.10 (still supported) and this is the hottest construction site.
@inline CN0UpdateContext(
    signal::AbstractGNSSSignal,
    bit_synchronizer::BitSynchronizer,
    num_code_periods::Integer,
    bit_sync_usable::Bool = true;
    noise_density = nothing,
    coherent_integration_time = nothing,
) = CN0UpdateContext(
    signal,
    bit_synchronizer,
    num_code_periods,
    bit_sync_usable,
    noise_density,
    coherent_integration_time,
)

@inline function CN0UpdateContext(
    signal::AbstractGNSSSignal,
    bit_synchronizer::BitSynchronizer,
    num_code_periods::Integer,
    bit_sync_usable::Bool,
    noise_density,
    coherent_integration_time,
)
    usable = bit_sync_usable && is_bit_or_secondary_code_synced(bit_synchronizer)
    CN0UpdateContext(
        signal,
        Int(num_code_periods),
        get_code_periods_per_bit(signal),
        usable ? bit_synchronizer.num_code_periods_into_bit : -1,
        bit_synchronizer,
        noise_density,
        coherent_integration_time,
    )
end

"""
$(SIGNATURES)

Does `estimator` read its signal's noise density out of its
[`CN0UpdateContext`](@ref)? `false` for every estimator that infers its noise
floor from the prompt stream ([`MomentsCN0Estimator`](@ref),
[`NWPRCN0Estimator`](@ref), [`NoCN0Estimator`](@ref)); `true` only for
[`NoiseReferencedCN0Estimator`](@ref).

A trait, so a custom estimator can opt in. Two things key off it:

  - **Provisioning.** `Tracking.TrackState` gives a signal a
    [`CorrelatorNoiseEstimator`](@ref) only where its estimator returns `true`;
    any other signal runs no despread at all.
  - **The warm-up skip.** While a configured source's window is still empty, the
    post-correlation processing skips the C/N₀ update for the requiring signal *only*. Skipping its
    neighbours too would make an `NWPRCN0Estimator` beside it drop its open
    window on every skipped integration (see `_update_nwpr`).

The trait's home is the **type** and the instance method forwards to it, so
provisioning folds out of `Tracking.TrackState`'s type parameters. A custom
estimator should define the type form to keep that.
"""
requires_noise_density(::Type{<:AbstractCN0Estimator}) = false
requires_noise_density(estimator::AbstractCN0Estimator) =
    requires_noise_density(typeof(estimator))

"""
$(SIGNATURES)

Add one completed integration's `prompt` to `estimator` and return the updated
estimator, which replaces `estimator`. This three-argument form is the one the tracking
loop calls and the **extension point** for custom
[`AbstractCN0Estimator`](@ref)s that profit from the navigation-bit information
in `context` (see [`CN0UpdateContext`](@ref)).

The default implementation drops `context` and calls the two-argument
`TrackingLoops.update!(estimator, prompt)`, so an estimator that only needs the
prompt stream — like [`MomentsCN0Estimator`](@ref) — implements just that one.
"""
update!(estimator::AbstractCN0Estimator, prompt, ::CN0UpdateContext) =
    update!(estimator, prompt)

"""
$(SIGNATURES)

The default CN0 estimator for `signal`: a [`NoiseReferencedCN0Estimator`](@ref)
averaging over `num_integrations` integrations, against that signal's
own measured noise density.

It reads a density, so `Tracking.TrackState` provisions the signal a
[`CorrelatorNoiseEstimator`](@ref) automatically (see
[`requires_noise_density`](@ref)) and `track!` fills it from the samples — the
sample-driven path needs no configuration at all.

# Why this and not NWPR

[`NWPRCN0Estimator`](@ref) needs a coherent narrowband window, so it does not
apply uniformly. Measured through `track!` on a data-modulated GPS L1 C/A signal,
1200 code periods, median over 9 seeds, fraction of runs reporting `-Inf dB-Hz`
in brackets:

| true | NWPR, 1-period integrations | NoiseRef    | NWPR, 20-period integrations | NoiseRef    |
|:---- | ---------------------:| -----------:| ----------------------:| -----------:|
| 25   | 11.2 (56 %)           | 23.4 (0)    | —                      | —           |
| 30   | 29.3 ± 1.40           | 29.6 ± 0.62 | —                      | —           |
| 40   | 39.7 ± 0.35           | 39.9 ± 0.19 | 27.1 ± 5.79            | 39.8 ± 0.37 |
| 45   | 44.7 ± 0.30           | 44.9 ± 0.13 | 32.9 ± 2.00            | 44.7 ± 0.30 |

At **25 dB-Hz**, where lock decisions are made, NWPR's ratio leaves
`1 < μ̂ < M` in over half the runs. At **long coherent integrations** an integration as long
as its window has `NBP ≡ WBP` and NWPR falls back, while the non-coherent
reference is immune to phase noise. On GPS L1C-D, Galileo E1B and secondary-coded
signals before sync NWPR never gets a window at all (JuliaGNSS/Tracking.jl#217).
NWPR is better only at the top of the range, where the reference carries a
self-leakage bias (see [`NoiseReferencedCN0Estimator`](@ref)).

# When to pass something else

Configure `NWPRCN0Estimator` explicitly for **externally supplied correlator
outputs without a noise observation**. Everywhere else, prefer appending a
[`NoiseObservation`](@ref) per signal with [`append_noise_observation!`](@ref).
"""
function default_cn0_estimator(
    signal::AbstractGNSSSignal,
    num_integrations::Int,
)
    NoiseReferencedCN0Estimator(; num_integrations = num_integrations)
end
