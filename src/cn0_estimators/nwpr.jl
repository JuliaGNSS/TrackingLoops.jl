"""
$(SIGNATURES)

Van Dierendonck's **narrowband/wideband power ratio** (NWPR) CN0 estimator.

The estimator for a **correlator-ingest path with no noise observation** (see
[`default_cn0_estimator`](@ref));
[`NWPRCN0Estimator(::AbstractGNSSSignal)`](@ref) sizes the window for the signal.
Over a window of `M` consecutive integrations it forms

```
NBP = |Σ_M prompt|²          (coherent — narrowband)
WBP =  Σ_M |prompt|²         (incoherent — wideband)
```

and over the windows that fit in `num_coherent_integrations` integrations reports

```
μ̂    = Σ_K NBP_k / Σ_K WBP_k
Ĉ/N₀ = (1 / T) · (μ̂ − 1) / (M − μ̂)
```

with `T` the coherent integration time. Reference:
A. J. Van Dierendonck,
"GPS Receivers", ch. 8 in *Global Positioning System: Theory and Applications*,
Vol. I, ed. B. W. Parkinson & J. J. Spilker Jr.; the formulas above are reproduced on
[ESA Navipedia](https://gssc.esa.int/navipedia/index.php/Lock_Detectors).

# Why the ratio of the sums, and not the mean of the ratios

The reference uses the mean of the per-window ratios, but the inversion assumes
`μ = E[NBP] / E[WBP]`, which only the ratio of the sums estimates consistently
(`E[NBP/WBP] < E[NBP]/E[WBP]` at finite `M`). Bias measured with `K → ∞`:

| true C/N₀ | mean of ratios, `M = 2` | `M = 5` | `M = 20` | ratio of sums, any `M` |
|:--------- | -----------------------:| -------:| --------:| ----------------------:|
| 20 dB-Hz  | 18.4                    | 19.3    | 19.8     | 20.0                   |
| 25 dB-Hz  | 23.6                    | 24.4    | 24.9     | 25.0                   |
| 30 dB-Hz  | 28.9                    | 29.6    | 29.9     | 30.0                   |

The ratio of sums is consistent at every `M`, so the estimate does not depend on
the window choice. It shifts the estimate without changing its spread, and at equal
false-alarm rate is the slightly better detector (at `M = 2` and 0.1 %, it detects a
true 25 dB-Hz in 22.5 % of updates against 15.0 %).

# Against the moments method

Unlike [`MomentsCN0Estimator`](@ref), whose median on pure noise is ~27.6 dB-Hz
at its default 100-prompt window (so a lock threshold below ~30 dB-Hz never
trips and one at 30 dB-Hz false-alarms on ~19 % of updates), NWPR reports "no
signal" on noise (Falletti, Pini & Lo Presti, *IEEE T-AES* 47(1):420–437, 2011;
JuliaGNSS/Tracking.jl#217).

# The coherence constraint

`NBP` is **coherent**, so a window must not straddle a bit flip (~7 dB loss) and
must stay short against residual Doppler (`M·T ≪ 1/(2·Δf)`; 25 Hz over 20 ms is
half a cycle, ~9 dB). The window follows the
navigation-bit grid in [`CN0UpdateContext`](@ref):

| signal state                           | narrowband window                                 |
|:-------------------------------------- |:------------------------------------------------- |
| synced, data-bearing                   | `num_narrowband_code_periods`, tiling the bit     |
| synced, pilot (no bit grid)            | `num_narrowband_code_periods`, free-running       |
| unsynced, data, no secondary code      | `num_presync_narrowband_code_periods`, unaligned  |
| unsynced, secondary code               | none: the unknown overlay flips every code period |
| one bit per code period (L1C-D, E1B)   | none: no coherent window beyond one integration   |
| integration at least as long as window | none: one integration has `NBP == WBP`            |

The unaligned pre-sync window exists because bit sync takes seconds and fails
below ~30 dB-Hz; it straddles a flip with probability `(M−1)/L` (~0.6 dB below
30 dB-Hz at five code periods of a 20-period bit, up to ~6 dB at a strong signal until sync).

# The window is capped by the loop's coherence time, not by the bit period

A longer window buys little spread (the same integrations are only partitioned
differently), while residual phase noise costs a *bias* averaging cannot remove.
GPS L1 C/A at 1 ms integrations, locked at 45 dB-Hz and faded to a true 25 dB-Hz, median /
10th percentile over 96 runs:

| PLL                        | 2 integrations   | 5           | 10          | 20 (one bit) |
|:-------------------------- | -----------:| -----------:| -----------:| ------------:|
| default, narrowed to 18 Hz | 24.8 / 21.1 | 24.7 / 22.7 | 24.7 / 22.8 | 24.7 / 22.6  |
| 40 Hz                      | 24.8 / 18.3 | 24.0 / 19.0 | 22.7 / -Inf | 18.2 / -Inf  |

The default loop (see [`CarrierLoopStage`](@ref)) has narrowed by the fade and holds
phase over a whole bit: past five integrations the window changes neither the bias nor
the spread (standard deviation 1.6 dB at 5 integrations, 1.4 at 20). A wider PLL does
not, and the whole-bit window reads `-Inf` ("no signal") in 30 of the 96 runs. The
default (~5 ms, see [`default_cn0_estimator`](@ref)) therefore stays short, for a
loop that has not narrowed or runs wider; raise it where the PLL is narrow, for a
pilot or for a signal that is never weak.

Where no window is admissible the `fallback` is reported (by default
[`MomentsCN0Estimator`](@ref), with its noise floor), e.g. for integrations spanning
a whole navigation bit.

# Fields / configuration

  - `num_coherent_integrations`: integrations the estimate averages over (100, ~100 ms at GPS L1 C/A);
    memory is independent of `M`.
  - `num_narrowband_code_periods`: window length in primary-code periods, the cap on
    the coherent sum (the length itself without a bit grid: a pilot, or the
    two-argument `TrackingLoops.update!`).
  - `num_presync_narrowband_code_periods`: window length before the bit grid is
    known; `0` reports the `fallback` until sync.
  - `buffered_narrowband_powers`, `buffered_wideband_powers`,
    `newest_index`, `num_buffered`, `num_coherent_integrations_per_ratio`,
    `ratios_are_bit_aligned`: rings of completed windows' `NBP` and `WBP`, the `M`
    they were formed with and whether they followed the bit grid; a change of
    either restarts the rings.
  - `narrowband_sum`, `wideband_power`, `num_accumulated_integrations`,
    `num_accumulated_code_periods`: the open window.
  - `fallback`: reported until a window completes, or always where none is
    admissible; fed every prompt.
"""
struct NWPRCN0Estimator{F<:AbstractCN0Estimator} <: AbstractCN0Estimator
    num_coherent_integrations::Int
    num_narrowband_code_periods::Int
    num_presync_narrowband_code_periods::Int
    buffered_narrowband_powers::Vector{Float64}
    buffered_wideband_powers::Vector{Float64}
    newest_index::Int
    num_buffered::Int
    num_coherent_integrations_per_ratio::Int
    ratios_are_bit_aligned::Bool
    narrowband_sum::ComplexF64
    wideband_power::Float64
    num_accumulated_integrations::Int
    num_accumulated_code_periods::Int
    fallback::F
end

"""
$(SIGNATURES)

Construct a fresh [`NWPRCN0Estimator`](@ref) averaging over the last `num_coherent_integrations`
integrations; see there for the parameters.
"""
function NWPRCN0Estimator(;
    num_coherent_integrations::Int = 100,
    num_narrowband_code_periods::Int = 5,
    num_presync_narrowband_code_periods::Int = 5,
    fallback::AbstractCN0Estimator = MomentsCN0Estimator(; num_coherent_integrations = num_coherent_integrations),
)
    num_coherent_integrations >= 2 ||
        throw(ArgumentError("num_coherent_integrations must be at least 2, got $num_coherent_integrations"))
    num_narrowband_code_periods >= 1 || throw(
        ArgumentError(
            "num_narrowband_code_periods must be at least 1, got " *
            "$num_narrowband_code_periods",
        ),
    )
    num_presync_narrowband_code_periods >= 0 || throw(
        ArgumentError(
            "num_presync_narrowband_code_periods must not be negative, got " *
            "$num_presync_narrowband_code_periods",
        ),
    )
    # A window holds at least two integrations, so at most `num_coherent_integrations ÷ 2` windows.
    NWPRCN0Estimator(
        num_coherent_integrations,
        num_narrowband_code_periods,
        num_presync_narrowband_code_periods,
        zeros(Float64, div(num_coherent_integrations, 2)),
        zeros(Float64, div(num_coherent_integrations, 2)),
        0,
        0,
        0,
        false,
        complex(0.0, 0.0),
        0.0,
        0,
        0,
        fallback,
    )
end

"""
$(SIGNATURES)

Construct an [`NWPRCN0Estimator`](@ref) whose coherent window covers about 5 ms of
`signal`'s code periods, at least two (5 for a 1 ms code, 2 for GPS L1C-P's 10 ms);
see [`NWPRCN0Estimator`](@ref) for why 5 ms. (A fixed period count would scale the
window with the code period.) On a correlator-ingest path with no noise observation:

```julia
TrackedSat(
    GPSL1C_P(),
    prn,
    code_phase,
    doppler;
    cn0_estimator = NWPRCN0Estimator(GPSL1C_P()),
)
```

Every other keyword is forwarded unchanged.
"""
function NWPRCN0Estimator(
    signal::AbstractGNSSSignal;
    num_coherent_integrations::Int = 100,
    num_narrowband_code_periods::Int = _default_narrowband_code_periods(signal),
    kwargs...,
)
    NWPRCN0Estimator(; num_coherent_integrations, num_narrowband_code_periods, kwargs...)
end

# Whole code periods covering ~5 ms, at least two.
@inline function _default_narrowband_code_periods(signal::AbstractGNSSSignal)
    code_period = get_code_length(signal) / get_chip_rate(signal)
    max(2, round(Int, 5ms / code_period))
end

length(estimator::NWPRCN0Estimator) = estimator.num_buffered
get_buffered_narrowband_powers(estimator::NWPRCN0Estimator) =
    estimator.buffered_narrowband_powers
get_buffered_wideband_powers(estimator::NWPRCN0Estimator) =
    estimator.buffered_wideband_powers
get_newest_index(estimator::NWPRCN0Estimator) = estimator.newest_index

# Ring slots in use at the current `M`, capped by the buffer.
@inline function _num_ratios(estimator::NWPRCN0Estimator)
    capacity = Base.length(estimator.buffered_narrowband_powers)
    estimator.num_coherent_integrations_per_ratio < 1 && return capacity
    clamp(div(estimator.num_coherent_integrations, estimator.num_coherent_integrations_per_ratio), 1, capacity)
end

# `(window_code_periods, window_code_period_index)`: the window length (`0` = none,
# report the fallback) and the integration's offset in a grid-following window (`-1` =
# free-running). Cases: see the table in `NWPRCN0Estimator`.
@inline function _narrowband_window(estimator::NWPRCN0Estimator, context::CN0UpdateContext)
    num_code_periods_per_bit = context.num_code_periods_per_bit
    if context.bit_code_period_index < 0
        # Bit grid unknown: only a data signal without secondary code is coherent in a bit.
        num_code_periods_per_bit > 1 && get_secondary_code_length(context.signal) == 1 ||
            return (0, -1)
        return (
            min(estimator.num_presync_narrowband_code_periods, num_code_periods_per_bit),
            -1,
        )
    end
    # One bit per code period.
    num_code_periods_per_bit == 1 && return (0, -1)
    # Pilot: secondary code wiped off post-sync, window free-running.
    num_code_periods_per_bit == 0 && return (estimator.num_narrowband_code_periods, -1)
    # Data-bearing and synced: tile the bit from its start. The trailing remainder
    # is skipped: a shorter window has another `M` and would restart the ring.
    window_code_periods = min(estimator.num_narrowband_code_periods, num_code_periods_per_bit)
    window_start =
        div(context.bit_code_period_index, window_code_periods) * window_code_periods
    window_start + window_code_periods > num_code_periods_per_bit && return (0, -1)
    (window_code_periods, context.bit_code_period_index - window_start)
end

"""
$(SIGNATURES)

Accumulate one integration's `prompt` into the open window, with length and alignment
from the bit grid in `context` (see [`NWPRCN0Estimator`](@ref)). An open window is
dropped once no longer admissible (e.g. the grid moved under it at sync); a window
closing on one integration (`NBP == WBP`) empties the rings, handing over to the
`fallback`.
"""
function update!(estimator::NWPRCN0Estimator, prompt, context::CN0UpdateContext)
    window_code_periods, window_code_period_index = _narrowband_window(estimator, context)
    _update_nwpr(
        estimator,
        prompt,
        update!(estimator.fallback, prompt, context),
        context.num_code_periods,
        window_code_periods,
        window_code_period_index,
    )
end

"""
$(SIGNATURES)

Advance the estimator on a bare prompt stream without bit context: each prompt is a
one-period integration and windows run back to back at `num_narrowband_code_periods`.
"""
update!(estimator::NWPRCN0Estimator, prompt) = _update_nwpr(
    estimator,
    prompt,
    update!(estimator.fallback, prompt),
    1,
    estimator.num_narrowband_code_periods,
    -1,
)

# Shared accumulation core. `fallback` is already advanced; the window arguments
# are as returned by `_narrowband_window`.
@inline function _update_nwpr(
    estimator::NWPRCN0Estimator,
    prompt,
    fallback::AbstractCN0Estimator,
    num_code_periods::Int,
    window_code_periods::Int,
    window_code_period_index::Int,
)
    # No admissible window: drop whatever was open.
    window_code_periods < 1 && return _with_window_state(estimator, fallback)
    # A fractional-period integration (after a sync phase snap) would break the
    # inversion's `M` equal integrations: skip it, window untouched.
    num_code_periods < 1 && return _with_open_window(estimator, fallback)
    # A grid-following window holding `k` code periods must sit `k` code periods in; otherwise
    # the grid moved (sync just found): drop it, and only a window start reopens.
    if window_code_period_index >= 0 &&
       window_code_period_index != estimator.num_accumulated_code_periods
        window_code_period_index == 0 || return _with_window_state(estimator, fallback)
        estimator = _with_window_state(estimator, fallback)
    end
    narrowband_sum = estimator.narrowband_sum + prompt
    wideband_power = estimator.wideband_power + abs2(prompt)
    num_coherent_integrations = estimator.num_accumulated_integrations + 1
    num_code_periods_accumulated = estimator.num_accumulated_code_periods + num_code_periods
    num_code_periods_accumulated < window_code_periods && return _with_window_state(
        estimator,
        fallback;
        narrowband_sum,
        wideband_power,
        num_accumulated_integrations = num_coherent_integrations,
        num_accumulated_code_periods = num_code_periods_accumulated,
    )
    # Window complete. With one integration (integrations grew to the window length, a
    # lasting change) also empty the ring, or the estimate would freeze on old
    # windows: `estimate_cn0` consults the `fallback` only while it is empty.
    num_coherent_integrations < 2 && return _with_window_state(
        estimator,
        fallback;
        newest_index = 0,
        num_buffered = 0,
        num_coherent_integrations_per_ratio = 0,
    )
    # All-zero window: division guard.
    iszero(wideband_power) && return _with_window_state(estimator, fallback)
    narrowband_powers = estimator.buffered_narrowband_powers
    wideband_powers = estimator.buffered_wideband_powers
    narrowband_power = abs2(narrowband_sum)
    bit_aligned = window_code_period_index >= 0
    if num_coherent_integrations == estimator.num_coherent_integrations_per_ratio &&
       bit_aligned == estimator.ratios_are_bit_aligned
        num_ratios = _num_ratios(estimator)
        newest_index = mod(estimator.newest_index, num_ratios) + 1
        narrowband_powers[newest_index] = narrowband_power
        wideband_powers[newest_index] = wideband_power
        return _with_window_state(
            estimator,
            fallback;
            newest_index,
            num_buffered = min(estimator.num_buffered + 1, num_ratios),
            num_coherent_integrations_per_ratio = num_coherent_integrations,
            ratios_are_bit_aligned = bit_aligned,
        )
    end
    # First window at this `M` or after sync: re-zero the rings (`estimate_cn0`
    # sums them whole). Pre-sync windows go even at unchanged `M`, as some
    # straddled a bit flip.
    fill!(narrowband_powers, 0.0)
    fill!(wideband_powers, 0.0)
    narrowband_powers[1] = narrowband_power
    wideband_powers[1] = wideband_power
    _with_window_state(
        estimator,
        fallback;
        newest_index = 1,
        num_buffered = 1,
        num_coherent_integrations_per_ratio = num_coherent_integrations,
        ratios_are_bit_aligned = bit_aligned,
    )
end

# Rebuild with new ring / open-window state, reusing the vectors. Defaults: ring
# unchanged, window closed.
@inline _with_window_state(
    estimator::NWPRCN0Estimator,
    fallback::AbstractCN0Estimator;
    newest_index::Int = estimator.newest_index,
    num_buffered::Int = estimator.num_buffered,
    num_coherent_integrations_per_ratio::Int = estimator.num_coherent_integrations_per_ratio,
    ratios_are_bit_aligned::Bool = estimator.ratios_are_bit_aligned,
    narrowband_sum::ComplexF64 = complex(0.0, 0.0),
    wideband_power::Float64 = 0.0,
    num_accumulated_integrations::Int = 0,
    num_accumulated_code_periods::Int = 0,
) = NWPRCN0Estimator(
    estimator.num_coherent_integrations,
    estimator.num_narrowband_code_periods,
    estimator.num_presync_narrowband_code_periods,
    estimator.buffered_narrowband_powers,
    estimator.buffered_wideband_powers,
    newest_index,
    num_buffered,
    num_coherent_integrations_per_ratio,
    ratios_are_bit_aligned,
    narrowband_sum,
    wideband_power,
    num_accumulated_integrations,
    num_accumulated_code_periods,
    fallback,
)

# Advance only the fallback, keeping the open window.
@inline _with_open_window(estimator::NWPRCN0Estimator, fallback::AbstractCN0Estimator) =
    _with_window_state(
        estimator,
        fallback;
        narrowband_sum = estimator.narrowband_sum,
        wideband_power = estimator.wideband_power,
        num_accumulated_integrations = estimator.num_accumulated_integrations,
        num_accumulated_code_periods = estimator.num_accumulated_code_periods,
    )

"""
$(SIGNATURES)

Estimate the C/N₀ from the buffered powers (see [`NWPRCN0Estimator`](@ref));
`coherent_integration_time` is the integration's, the `T` of the formula. Returns the
`fallback`'s value while no window has completed, and always where none is admissible.

`μ̂ ≤ 1` (no detectable signal) yields `-Inf dB-Hz`, `μ̂ ≥ M` yields `Inf dB-Hz`:
the limits of the expression, not clamped. Thresholding needs no special case;
averaging the estimate does.
"""
function estimate_cn0(estimator::NWPRCN0Estimator, coherent_integration_time)
    length(estimator) == 0 && return estimate_cn0(estimator.fallback, coherent_integration_time)
    num_coherent_integrations = estimator.num_coherent_integrations_per_ratio
    total_wideband_power = sum(get_buffered_wideband_powers(estimator))
    iszero(total_wideband_power) &&
        return estimate_cn0(estimator.fallback, coherent_integration_time)
    mean_ratio = sum(get_buffered_narrowband_powers(estimator)) / total_wideband_power
    mean_ratio <= 1 && return dBHz(0.0 / coherent_integration_time)
    mean_ratio >= num_coherent_integrations && return dBHz(Inf / coherent_integration_time)
    SNR = (mean_ratio - 1) / (num_coherent_integrations - mean_ratio)
    dBHz(SNR / coherent_integration_time)
end
