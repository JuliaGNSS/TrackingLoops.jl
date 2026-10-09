"""
$(SIGNATURES)

Van Dierendonck's **narrowband/wideband power ratio** (NWPR) CN0 estimator.

For a **correlator-ingest path with no noise observation** (see
[`default_cn0_estimator`](@ref)); [`NWPRCN0Estimator(::AbstractGNSSSignal)`](@ref)
sizes the window for the signal. Over a window of `M` consecutive records it forms

```
NBP = |Σ_M prompt|²          (coherent — narrowband)
WBP =  Σ_M |prompt|²         (incoherent — wideband)
```

and over the windows that fit in `num_records` records reports

```
μ̂    = Σ_K NBP_k / Σ_K WBP_k
Ĉ/N₀ = (1 / T) · (μ̂ − 1) / (M − μ̂)
```

with `T` the record's integration time. Reference: A. J. Van Dierendonck, "GPS Receivers",
ch. 8 in *Global Positioning System: Theory and Applications*, Vol. I, ed.
B. W. Parkinson & J. J. Spilker Jr.; the formulas above are reproduced on
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

The ratio of sums shifts the estimate without changing its spread. Unlike
[`MomentsCN0Estimator`](@ref), NWPR reports "no signal" on pure noise rather than a
~27.6 dB-Hz floor (JuliaGNSS/Tracking.jl#217).

# The coherence constraint

`NBP` is **coherent**, so a window must not straddle a bit flip (~7 dB loss) and
must stay short against residual Doppler (`M·T ≪ 1/(2·Δf)`). The window follows the
navigation-bit grid in [`CN0UpdateContext`](@ref):

  - bit / secondary sync found, data-bearing signal: `num_narrowband_code_blocks`,
    tiling the navigation bit from its start;
  - bit / secondary sync found, pilot (no data): `num_narrowband_code_blocks` (no
    bit grid to respect);
  - sync not found yet, data-bearing without secondary code:
    `num_presync_narrowband_code_blocks`, unaligned;
  - sync not found yet, signal with a secondary code: none — the unknown overlay
    flips sign every code block;
  - one symbol per code block (GPS L1C-D, Galileo E1B): none — no coherent window
    longer than one record exists;
  - record at least as long as its own window: none — a one-record window has
    `NBP == WBP` by construction.

The unaligned pre-sync window exists because bit sync takes seconds and fails
below ~30 dB-Hz; it straddles a flip with probability `(M−1)/L` (~0.6 dB at the
defaults).

# The window is capped by the loop's coherence time, not by the bit period

A longer window buys little spread (the same records are only partitioned
differently), while residual phase noise costs a *bias* averaging cannot remove.
GPS L1 C/A at 1 ms records, locked at 45 dB-Hz and faded to a true 25 dB-Hz, median /
10th percentile over 96 runs:

| PLL                        | 2 records   | 5           | 10          | 20 (one bit) |
|:-------------------------- | -----------:| -----------:| -----------:| ------------:|
| default, narrowed to 18 Hz | 24.8 / 21.1 | 24.7 / 22.7 | 24.7 / 22.8 | 24.7 / 22.6  |
| 40 Hz                      | 24.8 / 18.3 | 24.0 / 19.0 | 22.7 / -Inf | 18.2 / -Inf  |

The default loop has narrowed by the fade and holds phase over a whole bit: past
five records the window changes neither the bias nor the spread (standard deviation
1.6 dB at 5 records, 1.4 at 20). A wider PLL does not, and the whole-bit window
reads `-Inf` ("no signal") in 30 of the 96 runs. The default (~5 ms, see
[`default_cn0_estimator`](@ref)) therefore stays short, for a loop that has not
narrowed or runs wider; raise it where the PLL is narrow, for a pilot or for a
signal that is never weak.

Where no window is admissible the `fallback` is reported (by default
[`MomentsCN0Estimator`](@ref), with its noise floor), e.g. for records integrated
over a whole navigation bit.

# Fields / configuration

  - `num_records`: records the estimate averages over (100, ~100 ms at GPS L1 C/A);
    memory is independent of `M`.
  - `num_narrowband_code_blocks`: window length in primary-code blocks, the cap on
    the coherent sum (the length itself without a bit grid: a pilot, or the
    two-argument `TrackingLoops.update`).
  - `num_presync_narrowband_code_blocks`: window length before the bit grid is
    known; `0` reports the `fallback` until sync.
  - `buffered_narrowband_powers`, `buffered_wideband_powers`,
    `ratio_current_index`, `filled_ratio_length`, `num_records_per_ratio`,
    `ratios_are_bit_aligned`: rings of completed windows' `NBP` and `WBP`, the `M`
    they were formed with and whether they followed the bit grid; a change of
    either restarts the rings.
  - `narrowband_sum`, `wideband_power`, `num_accumulated_records`,
    `num_accumulated_code_blocks`: the open window.
  - `fallback`: reported until a window completes, or always where none is
    admissible; fed every prompt.
"""
struct NWPRCN0Estimator{F<:AbstractCN0Estimator} <: AbstractCN0Estimator
    num_records::Int
    num_narrowband_code_blocks::Int
    num_presync_narrowband_code_blocks::Int
    buffered_narrowband_powers::Vector{Float64}
    buffered_wideband_powers::Vector{Float64}
    ratio_current_index::Int
    filled_ratio_length::Int
    num_records_per_ratio::Int
    ratios_are_bit_aligned::Bool
    narrowband_sum::ComplexF64
    wideband_power::Float64
    num_accumulated_records::Int
    num_accumulated_code_blocks::Int
    fallback::F
end

"""
$(SIGNATURES)

Construct a fresh [`NWPRCN0Estimator`](@ref) averaging over the last `num_records`
records; see there for the parameters.
"""
function NWPRCN0Estimator(;
    num_records::Int = 100,
    num_narrowband_code_blocks::Int = 5,
    num_presync_narrowband_code_blocks::Int = 5,
    fallback::AbstractCN0Estimator = MomentsCN0Estimator(num_records),
)
    num_records >= 2 ||
        throw(ArgumentError("num_records must be at least 2, got $num_records"))
    num_narrowband_code_blocks >= 1 || throw(
        ArgumentError(
            "num_narrowband_code_blocks must be at least 1, got " *
            "$num_narrowband_code_blocks",
        ),
    )
    num_presync_narrowband_code_blocks >= 0 || throw(
        ArgumentError(
            "num_presync_narrowband_code_blocks must not be negative, got " *
            "$num_presync_narrowband_code_blocks",
        ),
    )
    # A window holds at least two records, so at most `num_records ÷ 2` windows.
    NWPRCN0Estimator(
        num_records,
        num_narrowband_code_blocks,
        num_presync_narrowband_code_blocks,
        zeros(Float64, div(num_records, 2)),
        zeros(Float64, div(num_records, 2)),
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
`signal`'s code blocks, at least two (5 for a 1 ms code, 2 for GPS L1C-P's 10 ms).
Use this form rather than counting blocks blindly:

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
    num_records::Int = 100,
    num_narrowband_code_blocks::Int = _default_narrowband_code_blocks(signal),
    kwargs...,
)
    NWPRCN0Estimator(; num_records, num_narrowband_code_blocks, kwargs...)
end

# Whole code blocks covering ~5 ms, at least two.
@inline function _default_narrowband_code_blocks(signal::AbstractGNSSSignal)
    code_period = get_code_length(signal) / get_code_frequency(signal)
    max(2, round(Int, 5ms / code_period))
end

length(estimator::NWPRCN0Estimator) = estimator.filled_ratio_length
get_buffered_narrowband_powers(estimator::NWPRCN0Estimator) =
    estimator.buffered_narrowband_powers
get_buffered_wideband_powers(estimator::NWPRCN0Estimator) =
    estimator.buffered_wideband_powers
get_current_index(estimator::NWPRCN0Estimator) = estimator.ratio_current_index
get_fallback_cn0_estimator(estimator::NWPRCN0Estimator) = estimator.fallback

# Ring slots in use at the current `M`, capped by the buffer.
@inline function _num_ratios(estimator::NWPRCN0Estimator)
    capacity = Base.length(estimator.buffered_narrowband_powers)
    estimator.num_records_per_ratio < 1 && return capacity
    clamp(div(estimator.num_records, estimator.num_records_per_ratio), 1, capacity)
end

# `(window_code_blocks, window_block_index)`: the window length (`0` = none,
# report the fallback) and the record's offset in a grid-following window (`-1` =
# free-running). Cases: see the table in `NWPRCN0Estimator`.
@inline function _narrowband_window(estimator::NWPRCN0Estimator, context::CN0UpdateContext)
    num_code_blocks_per_bit = context.num_code_blocks_per_bit
    if context.bit_code_block_index < 0
        # Bit grid unknown.
        num_code_blocks_per_bit > 1 && get_secondary_code_length(context.signal) == 1 ||
            return (0, -1)
        return (
            min(estimator.num_presync_narrowband_code_blocks, num_code_blocks_per_bit),
            -1,
        )
    end
    # One symbol per code block.
    num_code_blocks_per_bit == 1 && return (0, -1)
    # Pilot: secondary code wiped off post-sync, window free-running.
    num_code_blocks_per_bit == 0 && return (estimator.num_narrowband_code_blocks, -1)
    # Data-bearing and synced: tile the bit from its start. The trailing remainder
    # is skipped: a shorter window has another `M` and would restart the ring.
    window_code_blocks = min(estimator.num_narrowband_code_blocks, num_code_blocks_per_bit)
    window_start =
        div(context.bit_code_block_index, window_code_blocks) * window_code_blocks
    window_start + window_code_blocks > num_code_blocks_per_bit && return (0, -1)
    (window_code_blocks, context.bit_code_block_index - window_start)
end

"""
$(SIGNATURES)

Accumulate one record's `prompt` into the open window, with length and alignment
from the bit grid in `context` (see [`NWPRCN0Estimator`](@ref)). An open window is
dropped once no longer admissible (e.g. the grid moved under it at sync); a
one-record window (`NBP == WBP`) is not buffered.
"""
function update(estimator::NWPRCN0Estimator, prompt, context::CN0UpdateContext)
    window_code_blocks, window_block_index = _narrowband_window(estimator, context)
    _update_nwpr(
        estimator,
        prompt,
        update(estimator.fallback, prompt, context),
        context.num_code_blocks,
        window_code_blocks,
        window_block_index,
    )
end

"""
$(SIGNATURES)

Advance the estimator on a bare prompt stream without bit context: each prompt is a
one-block record and windows run back to back at `num_narrowband_code_blocks`.
"""
update(estimator::NWPRCN0Estimator, prompt) = _update_nwpr(
    estimator,
    prompt,
    update(estimator.fallback, prompt),
    1,
    estimator.num_narrowband_code_blocks,
    -1,
)

# Shared accumulation core. `fallback` is already advanced; the window arguments
# are as returned by `_narrowband_window`.
@inline function _update_nwpr(
    estimator::NWPRCN0Estimator,
    prompt,
    fallback::AbstractCN0Estimator,
    num_code_blocks::Int,
    window_code_blocks::Int,
    window_block_index::Int,
)
    # No admissible window: drop whatever was open.
    window_code_blocks < 1 && return _with_window_state(estimator, fallback)
    # A fractional-block record (after a sync phase snap) would break the
    # inversion's `M` equal records: skip it, window untouched.
    num_code_blocks < 1 && return _with_open_window(estimator, fallback)
    # A grid-following window holding `k` blocks must sit `k` blocks in; otherwise
    # the grid moved (sync just found): drop it, and only a window start reopens.
    if window_block_index >= 0 &&
       window_block_index != estimator.num_accumulated_code_blocks
        window_block_index == 0 || return _with_window_state(estimator, fallback)
        estimator = _with_window_state(estimator, fallback)
    end
    narrowband_sum = estimator.narrowband_sum + prompt
    wideband_power = estimator.wideband_power + abs2(prompt)
    num_records = estimator.num_accumulated_records + 1
    num_code_blocks_accumulated = estimator.num_accumulated_code_blocks + num_code_blocks
    num_code_blocks_accumulated < window_code_blocks && return _with_window_state(
        estimator,
        fallback;
        narrowband_sum,
        wideband_power,
        num_accumulated_records = num_records,
        num_accumulated_code_blocks = num_code_blocks_accumulated,
    )
    # Window complete. With one record (records grew to the window length, a
    # lasting change) also empty the ring, or the estimate would freeze on old
    # windows: `estimate_cn0` consults the `fallback` only while it is empty.
    num_records < 2 && return _with_window_state(
        estimator,
        fallback;
        ratio_current_index = 0,
        filled_ratio_length = 0,
        num_records_per_ratio = 0,
    )
    # All-zero window: division guard.
    iszero(wideband_power) && return _with_window_state(estimator, fallback)
    narrowband_powers = estimator.buffered_narrowband_powers
    wideband_powers = estimator.buffered_wideband_powers
    narrowband_power = abs2(narrowband_sum)
    bit_aligned = window_block_index >= 0
    if num_records == estimator.num_records_per_ratio &&
       bit_aligned == estimator.ratios_are_bit_aligned
        num_ratios = _num_ratios(estimator)
        ratio_current_index = mod(estimator.ratio_current_index, num_ratios) + 1
        narrowband_powers[ratio_current_index] = narrowband_power
        wideband_powers[ratio_current_index] = wideband_power
        return _with_window_state(
            estimator,
            fallback;
            ratio_current_index,
            filled_ratio_length = min(estimator.filled_ratio_length + 1, num_ratios),
            num_records_per_ratio = num_records,
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
        ratio_current_index = 1,
        filled_ratio_length = 1,
        num_records_per_ratio = num_records,
        ratios_are_bit_aligned = bit_aligned,
    )
end

# Rebuild with new ring / open-window state, reusing the vectors. Defaults: ring
# unchanged, window closed.
@inline _with_window_state(
    estimator::NWPRCN0Estimator,
    fallback::AbstractCN0Estimator;
    ratio_current_index::Int = estimator.ratio_current_index,
    filled_ratio_length::Int = estimator.filled_ratio_length,
    num_records_per_ratio::Int = estimator.num_records_per_ratio,
    ratios_are_bit_aligned::Bool = estimator.ratios_are_bit_aligned,
    narrowband_sum::ComplexF64 = complex(0.0, 0.0),
    wideband_power::Float64 = 0.0,
    num_accumulated_records::Int = 0,
    num_accumulated_code_blocks::Int = 0,
) = NWPRCN0Estimator(
    estimator.num_records,
    estimator.num_narrowband_code_blocks,
    estimator.num_presync_narrowband_code_blocks,
    estimator.buffered_narrowband_powers,
    estimator.buffered_wideband_powers,
    ratio_current_index,
    filled_ratio_length,
    num_records_per_ratio,
    ratios_are_bit_aligned,
    narrowband_sum,
    wideband_power,
    num_accumulated_records,
    num_accumulated_code_blocks,
    fallback,
)

# Advance only the fallback, keeping the open window.
@inline _with_open_window(estimator::NWPRCN0Estimator, fallback::AbstractCN0Estimator) =
    _with_window_state(
        estimator,
        fallback;
        narrowband_sum = estimator.narrowband_sum,
        wideband_power = estimator.wideband_power,
        num_accumulated_records = estimator.num_accumulated_records,
        num_accumulated_code_blocks = estimator.num_accumulated_code_blocks,
    )

"""
$(SIGNATURES)

Estimate the C/N₀ from the buffered powers (see [`NWPRCN0Estimator`](@ref));
`integration_time` is the record's, the `T` of the formula. Returns the
`fallback`'s value while no window has completed.

`μ̂ ≤ 1` (no detectable signal) yields `-Inf dB-Hz`, `μ̂ ≥ M` yields `Inf dB-Hz`:
the limits of the expression, not clamped. Thresholding needs no special case;
averaging the estimate does.
"""
function estimate_cn0(estimator::NWPRCN0Estimator, integration_time)
    length(estimator) == 0 && return estimate_cn0(estimator.fallback, integration_time)
    num_records = estimator.num_records_per_ratio
    total_wideband_power = sum(get_buffered_wideband_powers(estimator))
    iszero(total_wideband_power) &&
        return estimate_cn0(estimator.fallback, integration_time)
    mean_ratio = sum(get_buffered_narrowband_powers(estimator)) / total_wideband_power
    mean_ratio <= 1 && return dBHz(0.0 / integration_time)
    mean_ratio >= num_records && return dBHz(Inf / integration_time)
    SNR = (mean_ratio - 1) / (num_records - mean_ratio)
    dBHz(SNR / integration_time)
end
