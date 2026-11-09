"""
    SyncResult

Outcome of a per-signal bit-sync / secondary-code-sync detector call.

Fields:

  - `found::Bool`: whether the detector locked on this update.
  - `phase::Int`: the secondary-code chip the *upcoming* integration aligns to
    (`0:N-1`, from the hard `_secondary_code_search`); zero for the soft CFAR
    detectors and signals without a secondary code, which fire on a boundary.
  - `polarity::Int8`: `±1`, the locked match orientation, applied to the decoded
    bits so a negative-polarity lock does not invert them.
"""
struct SyncResult
    found::Bool
    phase::Int
    polarity::Int8
end

"""
$(SIGNATURES)

Standard-normal quantile `Φ⁻¹(probability) = √2 · erfinv(2·probability − 1)`, the
`dof → ∞` anchor of `_t_quantile`. `±Inf` at `probability = 1` / `0`.
"""
@inline _norm_quantile(probability::Float64) = sqrt(2.0) * erfinv(2 * probability - 1)

"""
$(SIGNATURES)

`probability`-quantile of a Student-t distribution with `dof` degrees of freedom.
`_cfar_decide` uses it as a **small-sample penalty** on its threshold: its z-score
is not exactly Student-t (χ² bin energies, correlated hypotheses), so the nominal
`dof` (`_bin_energy_variance_dof`) is a heuristic. What matters is the shape: steep as
`dof → 1`, the normal quantile as `dof → ∞`.

Hill's algorithm (Hill, G. W. (1970), *Algorithm 396: Student's t-quantiles*,
Comm. ACM 13(10), 619–620) on the two-tailed probability: closed forms at
`dof = 1, 2`, otherwise a series in the normal deviate or (deep tail) in the
probability. Accuracy ≲ 2e-4 relative (worst at `dof ≈ 3–10`). Used instead of
`SpecialFunctions.beta_inc_inv`, which allocates over part of the `dof` range and
is 10–170× slower: it runs every pre-sync code period, which must not allocate (checked
in `test/bit_synchronizer.jl`).
"""
function _t_quantile(probability::Float64, dof::Real)
    probability == 0.5 && return 0.0
    probability < 0.5 && return -_t_quantile(1 - probability, dof)
    two_tailed_probability = 2 * (1 - probability)
    # `dof = 1` is the Cauchy, `cot(π·two_tailed/2)` (accurate in the far tail,
    # where `tan(π(probability − ½))` cancels); `dof = 2` inverts
    # `½·(1 + t/√(2 + t²))`. The series is weakest exactly there.
    dof == 1 &&
        return cos(two_tailed_probability * pi / 2) / sin(two_tailed_probability * pi / 2)
    dof == 2 && return sqrt(2 / (two_tailed_probability * (2 - two_tailed_probability)) - 2)
    a = 1 / (dof - 0.5)
    b = 48 / a^2
    c = ((20700 * a / b - 98) * a - 16) * a + 96.36
    d = ((94.5 / (b + c) - 3) / b + 1) * sqrt(a * pi / 2) * dof
    y = (d * two_tailed_probability)^(2 / dof)
    if y > 0.05 + a
        # Near-normal branch: asymptotic series in `1/dof` on the normal deviate.
        x = _norm_quantile(probability)
        y = x^2
        dof < 5 && (c += 0.3 * (dof - 4.5) * (x + 0.6))
        c = (((0.05 * d * x - 5) * x - 7) * x - 2) * x + b + c
        y = (((((0.4 * y + 6.3) * y + 36) * y + 94.5) / c - y - 3) / b + 1) * x
        y = expm1(a * y^2)
    else
        # Deep-tail branch: series in the two-tailed probability.
        y =
            (
                (
                    1 / (((dof + 6) / (dof * y) - 0.089 * d - 0.822) * (dof + 2) * 3) +
                    0.5 / (dof + 4)
                ) * y - 1
            ) * (dof + 1) / (dof + 2) + 1 / y
    end
    sqrt(dof * y)
end

"""
$(SIGNATURES)

Per-hypothesis bin statistics for the soft CFAR sync detectors, one entry per
timing hypothesis: bit-edge phase with one-bit bins (`_detect_bit_edge_cfar`,
`_update_phase_accumulators!`) or overlay rotation with one overlay-wiped
secondary period per bin (`_detect_secondary_code_cfar`,
`_update_secondary_accumulators!`). O(hypotheses) per code period, no growing history.
`period` is `code_periods_per_bit` or the secondary-code length `N`.

The vectors are mutated in place across [`BitSynchronizer`](@ref) rebuilds: an immutable
form would be copied (~660 B) or boxed every code period, allocating pre-sync.

Fields (all length `period` once seeded; empty before the first code period):

  - `open_bin_sum`: coherent sum of the open bin (overlay-wiped for secondary).
  - `mean_bin_energy` / `bin_energy_sum_of_squared_deviations`: running mean and
    `M₂ = Σ(energyᵢ − mean)²` of the completed-bin energies (Welford; `M₂` forgets
    beyond `BIN_ENERGY_VARIANCE_MEMORY` bins). The bin count is
    `div(num_code_periods - hypothesis, period)`.
  - `last_bin_polarity`: sign of the last completed bin's real part (`0` before).
"""
struct SyncHypothesisAccumulators
    open_bin_sum::Vector{ComplexF64}
    mean_bin_energy::Vector{Float64}
    bin_energy_sum_of_squared_deviations::Vector{Float64}
    last_bin_polarity::Vector{Int8}
end

SyncHypothesisAccumulators() = SyncHypothesisAccumulators(ComplexF64[], Float64[], Float64[], Int8[])

# Are the accumulators seeded for a `code_periods_per_bit`-phase search yet?
@inline _is_seeded(accumulators::SyncHypothesisAccumulators, code_periods_per_bit::Int) =
    length(accumulators.mean_bin_energy) == code_periods_per_bit

# Size the accumulator vectors to `code_periods_per_bit` phases and zero them.
function _seed_phase_accumulators!(accumulators::SyncHypothesisAccumulators, code_periods_per_bit::Int)
    resize!(accumulators.open_bin_sum, code_periods_per_bit)
    resize!(accumulators.mean_bin_energy, code_periods_per_bit)
    resize!(accumulators.bin_energy_sum_of_squared_deviations, code_periods_per_bit)
    resize!(accumulators.last_bin_polarity, code_periods_per_bit)
    _reset_phase_accumulators!(accumulators)
end

# Completed bins the bin-energy variance remembers: one second of L1 C/A bits. The
# mean energy (the timing statistic) averages every bin, but its variance (the CFAR
# noise scale) is the sample variance up to this many bins and an exponentially
# weighted one beyond. A transient during the search (carrier pull-in, blockage,
# fade) swings every hypothesis's energy alike: it barely moves the gap between them
# but would inflate a never-forgetting variance and hold the lock off for tens of
# seconds.
const BIN_ENERGY_VARIANCE_MEMORY = 50

# Fold a hypothesis's `bin_count`-th completed bin energy into its running mean and
# sum of squared deviations, kept such that `M₂ / (bin_count − 1)` is the variance
# `σ²ₙ = (1 − α) σ²ₙ₋₁ + α δ δ′` with
# `α = max(1 / (bin_count − 1), 1 / BIN_ENERGY_VARIANCE_MEMORY)`: exactly Welford's
# update for the first `BIN_ENERGY_VARIANCE_MEMORY + 1` bins.
@inline function _fold_bin_energy!(
    accumulators::SyncHypothesisAccumulators,
    i::Int,
    bin_energy,
    bin_count::Int,
)
    @inbounds begin
        energy_delta = bin_energy - accumulators.mean_bin_energy[i]
        accumulators.mean_bin_energy[i] += energy_delta / bin_count
        squared_deviation = energy_delta * (bin_energy - accumulators.mean_bin_energy[i])
        if bin_count <= BIN_ENERGY_VARIANCE_MEMORY + 1
            accumulators.bin_energy_sum_of_squared_deviations[i] += squared_deviation
        else
            α = 1 / BIN_ENERGY_VARIANCE_MEMORY
            accumulators.bin_energy_sum_of_squared_deviations[i] =
                (bin_count - 1) * (
                    (1 - α) * accumulators.bin_energy_sum_of_squared_deviations[i] /
                    (bin_count - 2) + α * squared_deviation
                )
        end
    end
    nothing
end

# Degrees of freedom of the forgetting variance: the sample variance's, capped at
# the exponential weights' (effective sample size `2 · BIN_ENERGY_VARIANCE_MEMORY − 1`).
@inline _bin_energy_variance_dof(bin_count::Int) =
    min(bin_count - 1, 2 * BIN_ENERGY_VARIANCE_MEMORY - 2)

# Zero in place, keeping the length: on resync (`_resync_bit_synchronizer`) `_is_seeded` is
# still true, so without this the next code period would fold into the old statistics.
# A no-op on the hard path, whose vectors stay empty.
function _reset_phase_accumulators!(accumulators::SyncHypothesisAccumulators)
    fill!(accumulators.open_bin_sum, zero(ComplexF64))
    fill!(accumulators.mean_bin_energy, 0.0)
    fill!(accumulators.bin_energy_sum_of_squared_deviations, 0.0)
    fill!(accumulators.last_bin_polarity, Int8(0))
    accumulators
end

"""
$(SIGNATURES)

Fold the `prompt` of the code period at 0-based `code_period_index` into the
[`SyncHypothesisAccumulators`](@ref) of a `code_periods_per_bit`-phase bit-edge search. Phase
`phase`'s bins start at code period `phase`; the prompt joins every open bin, and the one
phase whose bin ends here folds `|bin_sum|²` into its Welford statistics and
records its polarity.
"""
function _update_phase_accumulators!(
    accumulators::SyncHypothesisAccumulators,
    prompt::ComplexF64,
    code_period_index::Int,
    code_periods_per_bit::Int,
)
    @inbounds for phase = 0:(code_periods_per_bit-1)
        code_period_index < phase && continue
        accumulators.open_bin_sum[phase+1] += prompt
        if (code_period_index - phase) % code_periods_per_bit == code_periods_per_bit - 1
            bin_sum = accumulators.open_bin_sum[phase+1]
            _fold_bin_energy!(
                accumulators,
                phase + 1,
                abs2(bin_sum),
                div(code_period_index + 1 - phase, code_periods_per_bit),
            )
            accumulators.last_bin_polarity[phase+1] = real(bin_sum) < 0 ? Int8(-1) : Int8(1)
            accumulators.open_bin_sum[phase+1] = zero(ComplexF64)
        end
    end
    accumulators
end

"""
$(SIGNATURES)

Shared CFAR (constant-false-alarm-rate) decision of the soft sync detectors:
find the maximum-energy hypothesis and its closest competitor in the
[`SyncHypothesisAccumulators`](@ref) statistics and decide whether the peak is significant.
`period` is both the code periods per bin and the number of hypotheses; `confidence` is
the target `1 − P(false lock)`.

Returns `(accepted, peak_index, peak_bin_count)`: whether the gap is significant
(the caller still gates on the bin boundary), the 0-based winner (`-1` if none)
and its completed-bin count.

The true hypothesis keeps full coherent gain on every bin while wrong ones lose
energy (straddling a bit transition, or failing to wipe the overlay), so the mean
bin energy is the maximum-likelihood timing statistic. The noise scale is the
winner's *bin-to-bin* energy variance, which captures thermal noise and slow
drift (a within-bin residual would miss the drift and accept a small systematic
asymmetry between hypotheses). The peak is accepted only when

    z_score = energy_gap / standard_error  ≥  t⁻¹(1 - (1 - confidence) / (period - 1); ν)

with the standard error over the peak and runner-up bin counts, the false-alarm
probability Bonferroni-split over the competitors, and `_t_quantile` as a
small-sample penalty at `ν = _bin_energy_variance_dof(peak_bin_count)`. A real peak's
`z_score` grows like √bins, so the detector self-paces with C/N₀; a drift-only
asymmetry stays bounded and never locks.
"""
@inline function _cfar_decide(
    mean_bin_energy::AbstractVector{Float64},
    bin_energy_sum_of_squared_deviations::AbstractVector{Float64},
    num_code_periods::Int,
    period::Int,
    confidence::Float64,
)
    # Need two complete bins on some hypothesis.
    num_code_periods < 2 * period && return (false, -1, 0)

    # One pass: the peak (highest energy with ≥ 2 bins) and the two highest
    # overall; the runner-up is the higher of those that isn't the peak.
    peak_index = -1
    peak_energy = -1.0
    peak_bin_count = 0          # best, ≥ 2 bins
    best_index = -1
    best_energy = -1.0
    best_bin_count = 0          # highest, ≥ 1 bin
    second_best_energy = -1.0
    second_best_bin_count = 0             # 2nd highest, ≥ 1 bin
    @inbounds for h = 0:(period-1)
        bin_count = div(num_code_periods - h, period)
        bin_count < 1 && continue
        energy = mean_bin_energy[h+1]
        if energy > best_energy
            second_best_energy = best_energy
            second_best_bin_count = best_bin_count
            best_energy = energy
            best_index = h
            best_bin_count = bin_count
        elseif energy > second_best_energy
            second_best_energy = energy
            second_best_bin_count = bin_count
        end
        if bin_count >= 2 && energy > peak_energy
            peak_energy = energy
            peak_index = h
            peak_bin_count = bin_count
        end
    end
    # Unreachable: hypothesis 0 has ≥ 2 bins here.
    peak_index < 0 && return (false, -1, 0)

    runner_up_energy, runner_up_bin_count =
        best_index == peak_index ? (second_best_energy, second_best_bin_count) :
        (best_energy, best_bin_count)
    runner_up_bin_count < 1 && return (false, peak_index, peak_bin_count)

    energy_gap = peak_energy - runner_up_energy
    energy_gap <= 0 && return (false, peak_index, peak_bin_count)

    # The runner-up is assumed to share the peak's variance (its straddling bins
    # can only make it larger): the fastest-locking choice that rejects drift.
    @inbounds bin_energy_variance =
        bin_energy_sum_of_squared_deviations[peak_index+1] / (peak_bin_count - 1)
    standard_error =
        sqrt(bin_energy_variance * (1 / peak_bin_count + 1 / runner_up_bin_count))
    z_score =
        standard_error > 0 ? energy_gap / standard_error : (energy_gap > 0 ? Inf : 0.0)

    # Clamp to (0, 1): `confidence = 1.0` would give a NaN threshold, which the
    # gate below would pass, locking immediately.
    false_alarm_probability = 1 - confidence
    quantile_argument =
        clamp(1 - false_alarm_probability / (period - 1), nextfloat(0.0), prevfloat(1.0))
    # Student-t, not normal: at 1–2 bins the normal threshold (~3.9) is a ~10%
    # per-hypothesis false alarm at 1 d.o.f., not 1e-4, which let a variance-collapsed
    # 2-bin fluctuation lock. The t quantile demands z ≈ 6000 at 1 d.o.f. and relaxes
    # to normal as bins accumulate (which also lets the carrier settle).
    z_threshold = _t_quantile(quantile_argument, _bin_energy_variance_dof(peak_bin_count))
    z_score < z_threshold && return (false, peak_index, peak_bin_count)

    (true, peak_index, peak_bin_count)
end

"""
$(SIGNATURES)

Soft CFAR bit-edge detector (see [`uses_soft_bit_edge_detection`](@ref); GPS L1 C/A)
running `_cfar_decide` over the edge phases `0:code_periods_per_bit-1`.

Fires only when the latest code period *ends* the winning phase's bit, so the upcoming
integration starts a fresh bit (rules out the off-by-one lock of
JuliaGNSS/Tracking.jl#124).
`polarity` is the sign of the last completed bin's sum.
"""
function _detect_bit_edge_cfar(
    accumulators::SyncHypothesisAccumulators,
    code_periods_per_bit::Int,
    confidence::Float64,
    num_code_periods::Int,
)
    accepted, peak_phase, _ = _cfar_decide(
        accumulators.mean_bin_energy,
        accumulators.bin_energy_sum_of_squared_deviations,
        num_code_periods,
        code_periods_per_bit,
        confidence,
    )
    accepted || return SyncResult(false, 0, Int8(0))

    num_code_periods % code_periods_per_bit != peak_phase && return SyncResult(false, 0, Int8(0))

    @inbounds polarity =
        accumulators.last_bin_polarity[peak_phase+1] < 0 ? Int8(-1) : Int8(+1)
    SyncResult(true, 0, polarity)
end

"""
$(SIGNATURES)

The secondary-code analog of `_update_phase_accumulators!`: rotation `d ∈ 0:N-1`
has `N`-period bins starting at code period `d`, and code period `i` is multiplied by secondary
chip `mod(i - d, N)` (the overlay the post-sync replica applies) before summing,
so the correct rotation adds coherently. Chip 0 is thus the physical secondary chip
0, the rotation the hard `_secondary_code_search` locks (up to a polarity the energy
ignores).
"""
function _update_secondary_accumulators!(
    accumulators::SyncHypothesisAccumulators,
    prompt::ComplexF64,
    code_period_index::Int,
    secondary_code_length::Int,
    signal::AbstractGNSSSignal,
    prn::Integer,
)
    N = secondary_code_length
    secondary_code = get_secondary_code(signal)
    @inbounds for d = 0:(N-1)
        code_period_index < d && continue
        chip = (code_period_index - d) % N
        overlay = GNSSSignals.secondary_value(secondary_code, prn, chip)
        accumulators.open_bin_sum[d+1] += prompt * overlay
        if chip == N - 1
            bin_sum = accumulators.open_bin_sum[d+1]
            _fold_bin_energy!(accumulators, d + 1, abs2(bin_sum), div(code_period_index + 1 - d, N))
            accumulators.last_bin_polarity[d+1] = real(bin_sum) < 0 ? Int8(-1) : Int8(1)
            accumulators.open_bin_sum[d+1] = zero(ComplexF64)
        end
    end
    accumulators
end

"""
$(SIGNATURES)

Soft CFAR secondary-code detector, the analog of `_detect_bit_edge_cfar` for short
overlays (see [`uses_soft_secondary_code_detection`](@ref)). Unlike the hard
`_secondary_code_search` it rejects the noise-driven false locks a short template
match is prone to.

Fires only when the latest code period ends the winning rotation's period, so the
upcoming integration starts at secondary chip 0 and `SyncResult.phase` is `0`.
`polarity` is the sign of the winning period's overlay-wiped sum.
"""
function _detect_secondary_code_cfar(
    accumulators::SyncHypothesisAccumulators,
    secondary_code_length::Int,
    confidence::Float64,
    num_code_periods::Int,
)
    accepted, peak_rotation, _ = _cfar_decide(
        accumulators.mean_bin_energy,
        accumulators.bin_energy_sum_of_squared_deviations,
        num_code_periods,
        secondary_code_length,
        confidence,
    )
    accepted || return SyncResult(false, 0, Int8(0))

    num_code_periods % secondary_code_length != peak_rotation &&
        return SyncResult(false, 0, Int8(0))

    @inbounds polarity =
        accumulators.last_bin_polarity[peak_rotation+1] < 0 ? Int8(-1) : Int8(+1)
    SyncResult(true, 0, polarity)
end

"""
$(SIGNATURES)

**Hard-decision** secondary-code rotation search (currently the 1800-chip overlay
pilots GPS L1C-P and BeiDou B1C-P); locks within one secondary period.

`received` is the prompt-sign window, newest code period in bit 0; `reference` is packed
newest-first (bit `i` = chip `N - 1 - i`). Rotating the low `N` bits left by `d`
finds the best Hamming match of either polarity; the upcoming integration's chip is
`mod(N - d, N)` (Tracking.jl's `_snap_code_phase_from_synced_signal` anchor). No
lock if the best distance exceeds `max_errors`.
"""
@inline function _secondary_code_search(
    received::B,
    reference::B,
    secondary_code_length::Int,
    max_errors::Int,
) where {B<:Unsigned}
    N = secondary_code_length
    # `one(B) << N` is undefined for an exact-width history (UInt1800).
    mask = N == 8 * sizeof(B) ? ~zero(B) : (one(B) << N) - one(B)
    masked = received & mask
    best_d = 0
    best_dist = N + 1
    best_pol = Int8(0)
    @inbounds for d = 0:(N-1)
        # Rotate left within N bits; `masked >> N` is undefined at exact width.
        shifted = d == 0 ? masked : ((masked << d) | (masked >> (N - d))) & mask
        dist_pos = count_ones(shifted ⊻ reference)
        dist_neg = N - dist_pos
        if dist_pos < best_dist
            best_dist = dist_pos
            best_d = d
            best_pol = Int8(+1)
        end
        if dist_neg < best_dist
            best_dist = dist_neg
            best_d = d
            best_pol = Int8(-1)
        end
    end
    best_dist > max_errors && return SyncResult(false, 0, Int8(0))
    SyncResult(true, mod(N - best_d, N), best_pol)
end

"""
$(SIGNATURES)

Detector for signals with no sub-period boundary to find: `found = true` from the
first integration. Covers one bit per code period (GPS L1C-D, L2CM, Galileo
E1B / E6-B, BeiDou B2b-I / B1C-D; GNSSDecoder.jl resolves the ±1 polarity from
the preamble) and Galileo E5a-QP (no data, no overlay; the lock only gates
whole-code-cycle integration).
"""
@inline function _detect_bit_is_code_period_sync(
    ::AbstractGNSSSignal,
    ::Integer,           # PRN — ignored
    ::Unsigned,
    ::Integer,
)
    SyncResult(true, 0, Int8(+1))
end

"""
$(SIGNATURES)

Hard-path detector for secondary-coded signals: once `N` code periods are buffered, run
`_secondary_code_search` against `_packed_secondary_code` allowing
`floor(tolerance × N)` errors ([`get_secondary_code_error_tolerance`](@ref));
at the default, an exact match for a code shorter than 40 chips.

A new secondary-coded signal needs `get_secondary_code`, a
`prompt_sign_history_type` of at least `N` bits and a
`detect_sync` method delegating here (a soft-path signal
reaches it only with its soft trait overridden).
"""
@inline function _detect_secondary_code_sync(
    signal::AbstractGNSSSignal,
    prn::Integer,
    code_period_bits::B,
    num_code_periods::Integer,
) where {B<:Unsigned}
    secondary_code_length = get_secondary_code_length(signal)
    num_code_periods < secondary_code_length && return SyncResult(false, 0, Int8(0))
    max_errors =
        floor(Int, get_secondary_code_error_tolerance(signal) * secondary_code_length)
    _secondary_code_search(
        code_period_bits,
        _packed_secondary_code(B, signal, prn),
        secondary_code_length,
        max_errors,
    )
end

"""
$(SIGNATURES)

The secondary code for `prn` packed into `B` in `_secondary_code_search`'s order
(bit `i` holds chip `N - 1 - i`). Specialize only for an overlay not reachable
through `get_secondary_code`.
"""
function _packed_secondary_code end

# Bit `N - 1 - k` set iff chip `k` is positive; one convention for all signals, so
# `polarity = +1` always means the prompt signs follow `get_secondary_code`.
@inline function _packed_secondary_code(
    ::Type{B},
    signal::AbstractGNSSSignal,
    prn::Integer,
) where {B<:Unsigned}
    secondary_code = get_secondary_code(signal)
    N = get_secondary_code_length(signal)
    packed = zero(B)
    @inbounds for k = 0:(N-1)
        if GNSSSignals.secondary_value(secondary_code, prn, k) > 0
            packed |= one(B) << (N - 1 - k)
        end
    end
    packed
end

"""
$(SIGNATURES)

Bit sync state and demodulated soft bits of one signal.

`prompt_sign_history` is the pre-sync window of prompt signs, `B` bits wide
([`prompt_sign_history_type`](@ref)), unused after sync. `soft_bits` collects
the demodulated soft bits (see [`get_soft_bits`](@ref)), unbounded between resets.
`phase_accumulators` holds the soft CFAR detector's [`SyncHypothesisAccumulators`](@ref), empty on
the hard path.
"""
struct BitSynchronizer{B<:Unsigned}
    prompt_sign_history::B
    num_prompt_signs::Int
    bit_or_secondary_code_synced::Bool
    secondary_code_phase::Int      # 0 until found; secondary-chip offset post-sync
    bit_polarity::Int8            # +1 or -1 once found; 0 before sync
    bit_prompt_sum::ComplexF64
    num_code_periods_into_bit::Int
    soft_bits::Vector{Float32}
    phase_accumulators::SyncHypothesisAccumulators
end

# Untyped default: a `UInt128` prompt-sign history.
function BitSynchronizer()
    BitSynchronizer{UInt128}(
        zero(UInt128),
        0,
        false,
        0,
        Int8(0),
        complex(0.0, 0.0),
        0,
        Float32[],
        SyncHypothesisAccumulators(),
    )
end

function BitSynchronizer{B}() where {B<:Unsigned}
    BitSynchronizer{B}(
        zero(B),
        0,
        false,
        0,
        Int8(0),
        complex(0.0, 0.0),
        0,
        Float32[],
        SyncHypothesisAccumulators(),
    )
end

# For tests and benchmarks: zero phase / polarity, empty accumulators; `soft_bits`
# is aliased, not copied.
function BitSynchronizer(
    prompt_sign_history::B,
    num_prompt_signs::Integer,
    found::Bool,
    bit_prompt_sum::Complex,
    num_code_periods_into_bit::Integer,
    soft_bits::Vector{Float32} = Float32[],
) where {B<:Unsigned}
    BitSynchronizer{B}(
        prompt_sign_history,
        Int(num_prompt_signs),
        found,
        0,
        Int8(0),
        ComplexF64(bit_prompt_sum),
        Int(num_code_periods_into_bit),
        soft_bits,
        SyncHypothesisAccumulators(),
    )
end

@inline length(bit_synchronizer::BitSynchronizer) = Base.length(bit_synchronizer.soft_bits)
"""
    is_bit_or_secondary_code_synced(bit_synchronizer::BitSynchronizer)
    is_bit_or_secondary_code_synced(state::PostCorrelationState)

Whether the navigation-bit boundary or secondary-code phase has been found; from
then on the signal may integrate over several code periods and bits are collected
for [`get_soft_bits`](@ref).
"""
@inline is_bit_or_secondary_code_synced(bit_synchronizer::BitSynchronizer) = bit_synchronizer.bit_or_secondary_code_synced

"""
    get_soft_bits(bit_synchronizer::BitSynchronizer)
    get_soft_bits(state::PostCorrelationState)

The soft bits collected so far: per navigation bit, the polarity-corrected real part
of the summed de-rotated prompts. The sign is the hard decision, the magnitude the
reliability, ready for soft-decision decoding (e.g. Viterbi for Galileo E1B, LDPC
for GPS L1C-D). Bits recovered from the pre-sync window are sign votes scaled by the
sync-time prompt magnitude, to stay comparable.

Outside GNSSDecoder, a bit is a bit as tracked: a channel symbol before
error-correction decoding. On a signal with forward error correction (GPS L2C, L5 and
L1C-D, Galileo, the BeiDou B-CNAV family) the decoder recovers the message bits from
them; elsewhere they are the message bits.

The bit synchronizer's own vector, not a copy: drain it (`empty!`) after reading. A
[`PostCorrelationState`](@ref)'s holds 64 bits before it allocates.
"""
@inline get_soft_bits(bit_synchronizer::BitSynchronizer) = bit_synchronizer.soft_bits

"""
$(SIGNATURES)

Concrete `Unsigned` width `B` of `BitSynchronizer.prompt_sign_history` for `signal`
(default `UInt64`), a type parameter so construction is type-stable.

Hard-path signals need at least `N` bits (the 1800-chip overlay pilots use
`UInt1800`); soft secondary-code signals keep one period so their hard path stays
available; GPS L1 C/A's holds the pre-sync bits recovered at sync; signals with
nothing to search use `UInt8`. Per-signal values: [Prompt-sign history widths](@ref).
"""
@inline prompt_sign_history_type(::AbstractGNSSSignal) = UInt64

"""
$(SIGNATURES)

Hamming tolerance of the **hard-decision** `_secondary_code_search`: the largest
fraction of chip errors accepted, `max_errors = floor(Int, tolerance × N)`. Default
`0.025` (45 of 1800 chips). Only hard-path signals (GPS L1C-P, BeiDou B1C-P) read
it; the soft detectors use [`get_sync_detection_confidence`](@ref).

# Overriding

E.g. to loosen it for low-C/N₀ work:

```julia
TrackingLoops.get_secondary_code_error_tolerance(::GPSL1C_P) = 0.05
```

Takes effect at the next detector call.
"""
@inline get_secondary_code_error_tolerance(::AbstractGNSSSignal) = 0.025

"""
$(SIGNATURES)

Whether `signal`'s bit edge is found by the soft CFAR detector
`_detect_bit_edge_cfar` instead of the hard `detect_sync`.

By default true when a bit spans **more than one** code period **and** there is no
secondary code (currently only GPS L1 C/A). BeiDou B1I/B3I do not qualify even on
GEO PRNs, since their signal type reports a 20-chip secondary code (see
[BeiDou GEO satellites](@ref)). Override per signal type, e.g.:

```julia
TrackingLoops.uses_soft_bit_edge_detection(::SomeSignal) = false
```

Constant-folded per signal type.
"""
@inline uses_soft_bit_edge_detection(signal::AbstractGNSSSignal) =
    get_code_periods_per_bit(signal) > 1 &&
    get_secondary_code_length(signal) == 1

"""
$(SIGNATURES)

Whether `signal`'s secondary code is found by the soft CFAR detector
`_detect_secondary_code_cfar` instead of the hard `_secondary_code_search`.

Each bin integrates one secondary period coherently, so the default enables it
only for short codes, `1 < N ≤ 100`; the 1800-chip overlays (18 s) are too long to
integrate and too long to false-lock, and keep the hard sweep. Mutually exclusive
with [`uses_soft_bit_edge_detection`](@ref). Override per signal type, e.g.:

```julia
TrackingLoops.uses_soft_secondary_code_detection(::GPSL5I) = false
```

Constant-folded per signal type.
"""
@inline uses_soft_secondary_code_detection(signal::AbstractGNSSSignal) =
    1 < get_secondary_code_length(signal) <= 100

"""
$(SIGNATURES)

Target confidence (`1 − P(false lock)`) of the soft CFAR sync detectors, default
`0.999`: they integrate until the best hypothesis beats its competitor with this
confidence (two bins for a clean signal, longer in noise). Lower it to lock faster
at the cost of more false locks.

# Overriding

```julia
TrackingLoops.get_sync_detection_confidence(::GPSL1CA) = 0.9999
```

Takes effect at the next detector call.
"""
@inline get_sync_detection_confidence(::AbstractGNSSSignal) = 0.999


"""
$(SIGNATURES)

Advance the bit synchronizer by one integration of `num_code_periods` code periods.

Post-sync, one soft bit is emitted each time the period count reaches a bit. An
integration carrying the count *past* the boundary (only from a non-bit-aligned external
producer) drops sync instead (JuliaGNSS/Tracking.jl#238): its energy belongs to two
bits, and the equality test would never be met again. The detector then re-runs
from clean accumulators and the caller falls back to single-period integrations; the
decoded `soft_bits` are kept. Silent; [`process_correlator_bank_output!`](@ref) reports it as
the state's `last_sync_change`, `SYNC_LOST`.
"""
function update_bit_sync!(
    signal::AbstractGNSSSignal,
    prn::Integer,
    bit_synchronizer::BitSynchronizer{B},
    num_code_periods,
    prompt,
) where {B<:Unsigned}
    num_code_periods_per_bit = get_code_periods_per_bit(signal)

    if (bit_synchronizer.bit_or_secondary_code_synced == false)
        return _update_sync_search(
            signal,
            prn,
            bit_synchronizer,
            num_code_periods_per_bit,
            num_code_periods,
            prompt,
        )
    end

    # Pilots: nothing to decode; the lock still anchors the code phase.
    num_code_periods_per_bit == 0 && return bit_synchronizer

    bit_prompt_sum = bit_synchronizer.bit_prompt_sum + prompt
    num_code_periods_into_bit =
        bit_synchronizer.num_code_periods_into_bit + num_code_periods

    if num_code_periods_into_bit > num_code_periods_per_bit
        # Overshoot (e.g. 18 + a 3-period integration for L1 C/A): resync rather than
        # emit at `>=`, see the docstring. Must not allocate or format a string.
        return _resync_bit_synchronizer(bit_synchronizer)
    elseif num_code_periods_into_bit == num_code_periods_per_bit
        bit_acc =
            bit_synchronizer.bit_polarity < 0 ? -real(bit_prompt_sum) : real(bit_prompt_sum)
        push!(bit_synchronizer.soft_bits, Float32(bit_acc))
        return BitSynchronizer{B}(
            bit_synchronizer.prompt_sign_history,
            bit_synchronizer.num_prompt_signs,
            true,
            bit_synchronizer.secondary_code_phase,
            bit_synchronizer.bit_polarity,
            zero(bit_prompt_sum),
            0,
            bit_synchronizer.soft_bits,
            bit_synchronizer.phase_accumulators,
        )
    else
        return BitSynchronizer{B}(
            bit_synchronizer.prompt_sign_history,
            bit_synchronizer.num_prompt_signs,
            true,
            bit_synchronizer.secondary_code_phase,
            bit_synchronizer.bit_polarity,
            bit_prompt_sum,
            num_code_periods_into_bit,
            bit_synchronizer.soft_bits,
            bit_synchronizer.phase_accumulators,
        )
    end
end

# Back to pre-sync after an overshoot; `soft_bits` is kept (same vector).
@inline function _resync_bit_synchronizer(bit_synchronizer::BitSynchronizer{B}) where {B<:Unsigned}
    BitSynchronizer{B}(
        zero(B),
        0,
        false,
        0,
        Int8(0),
        complex(0.0, 0.0),
        0,
        bit_synchronizer.soft_bits,
        _reset_phase_accumulators!(bit_synchronizer.phase_accumulators),
    )
end

function _update_sync_search(
    signal,
    prn::Integer,
    bit_synchronizer::BitSynchronizer{B},
    num_code_periods_per_bit,
    num_code_periods,
    prompt,
) where {B<:Unsigned}
    if (num_code_periods != 1)
        error(
            "The number of code periods must be equal to 1 if bit or secondary code hasn't been found yet.",
        )
    end
    # 1. The prompt's sign, onto the history the hard detectors search.
    prompt_sign_history = (bit_synchronizer.prompt_sign_history << 1) + B(real(prompt) > 0)
    num_prompt_signs = bit_synchronizer.num_prompt_signs + 1

    # 2. The detection of the bit edge or the secondary-code phase. Soft detectors
    #    share `phase_accumulators`; the rest take the hard path. Folds at compile time.
    phase_accumulators = bit_synchronizer.phase_accumulators
    if uses_soft_bit_edge_detection(signal)
        code_periods_per_bit = num_code_periods_per_bit
        _is_seeded(phase_accumulators, code_periods_per_bit) ||
            _seed_phase_accumulators!(phase_accumulators, code_periods_per_bit)
        _update_phase_accumulators!(
            phase_accumulators,
            ComplexF64(prompt),
            num_prompt_signs - 1,
            code_periods_per_bit,
        )
        sync = _detect_bit_edge_cfar(
            phase_accumulators,
            code_periods_per_bit,
            get_sync_detection_confidence(signal),
            num_prompt_signs,
        )
    elseif uses_soft_secondary_code_detection(signal)
        secondary_code_length = get_secondary_code_length(signal)
        _is_seeded(phase_accumulators, secondary_code_length) ||
            _seed_phase_accumulators!(phase_accumulators, secondary_code_length)
        _update_secondary_accumulators!(
            phase_accumulators,
            ComplexF64(prompt),
            num_prompt_signs - 1,
            secondary_code_length,
            signal,
            prn,
        )
        sync = _detect_secondary_code_cfar(
            phase_accumulators,
            secondary_code_length,
            get_sync_detection_confidence(signal),
            num_prompt_signs,
        )
    else
        sync = detect_sync(
            signal,
            prn,
            prompt_sign_history,
            num_prompt_signs,
        )
    end
    # 3. Not found yet: keep searching.
    if !sync.found
        return BitSynchronizer{B}(
            prompt_sign_history,
            num_prompt_signs,
            false,
            0,
            Int8(0),
            complex(0.0, 0.0),
            0,
            bit_synchronizer.soft_bits,
            phase_accumulators,
        )
    end
    # 4. Found: synced from here on, with the bits the history already holds recovered.
    if get_secondary_code_length(signal) > 1
        # Pre-sync signs carry the overlay, not data: no bits to recover. The hard
        # search can lock `sync.phase` code periods into a bit, so seed the period count
        # with it and the first bit ends on the boundary (JuliaGNSS/Tracking.jl#125).
        return BitSynchronizer{B}(
            prompt_sign_history,
            num_prompt_signs,
            true,
            sync.phase,
            sync.polarity,
            complex(0.0, 0.0),
            sync.phase,
            bit_synchronizer.soft_bits,
            phase_accumulators,
        )
    end
    if num_code_periods_per_bit == 0
        # Dataless, no overlay (Galileo E5a-QP): avoid dividing by zero below.
        return BitSynchronizer{B}(
            prompt_sign_history,
            num_prompt_signs,
            true,
            0,
            sync.polarity,
            complex(0.0, 0.0),
            0,
            bit_synchronizer.soft_bits,
            phase_accumulators,
        )
    end
    num_bits = min(
        div(num_prompt_signs, num_code_periods_per_bit),
        div(sizeof(prompt_sign_history) * 8, num_code_periods_per_bit),
    )
    # Hoisted: capturing `sync` (assigned in several branches) would box it, ~80 B per
    # pre-sync code period.
    sync_polarity = Int(sync.polarity)
    for bit_index = num_bits:-1:1     # oldest recovered bit first
        # Lock polarity applies here too (JuliaGNSS/Tracking.jl#127).
        bit_sum =
            sum(0:(num_code_periods_per_bit-1)) do code_period_index
                buffer_code_period_index =
                    (bit_index - 1) * num_code_periods_per_bit + code_period_index
                ((prompt_sign_history & (one(B) << buffer_code_period_index)) > 0) * 2 - 1
            end * sync_polarity
        # Sign votes scaled to soft-bit units (see `get_soft_bits`).
        push!(bit_synchronizer.soft_bits, Float32(bit_sum * abs(prompt)))
    end
    return BitSynchronizer{B}(
        prompt_sign_history,
        num_prompt_signs,
        true,
        sync.phase,
        sync.polarity,
        complex(0, 0),
        0,
        bit_synchronizer.soft_bits,
        phase_accumulators,
    )
end

"""
$(SIGNATURES)

Advance a just-synced `bit_synchronizer`'s `secondary_code_phase` by `num_code_periods`
(mod `N`); no-op without a secondary code.

The detector reports the phase for the code period after the syncing integration, but
Tracking.jl's code-phase snap reads it only after the whole chunk, so every later
integration of the chunk (not `secondary_code_wiped`, see `process_correlator_bank_output!`) must move it along, or
the replica applies the wrong overlay chip (cf. JuliaGNSS/Tracking.jl#219).
"""
@inline function _advance_secondary_phase(
    signal::AbstractGNSSSignal,
    bit_synchronizer::BitSynchronizer{B},
    num_code_periods::Integer,
) where {B<:Unsigned}
    secondary_code_length = get_secondary_code_length(signal)
    secondary_code_length > 1 || return bit_synchronizer
    BitSynchronizer{B}(
        bit_synchronizer.prompt_sign_history,
        bit_synchronizer.num_prompt_signs,
        bit_synchronizer.bit_or_secondary_code_synced,
        mod(bit_synchronizer.secondary_code_phase + Int(num_code_periods), secondary_code_length),
        bit_synchronizer.bit_polarity,
        bit_synchronizer.bit_prompt_sum,
        bit_synchronizer.num_code_periods_into_bit,
        bit_synchronizer.soft_bits,
        bit_synchronizer.phase_accumulators,
    )
end

"""
    drain_soft_bits!(state::BitSynchronizer) -> BitSynchronizer

Empty the buffered soft bits in place, preserving synchronization and accumulation
history. Previously returned integration bit views must be consumed before this call.
"""
function drain_soft_bits!(state::BitSynchronizer)
    empty!(state.soft_bits)
    state
end
