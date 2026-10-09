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
`dof = peak_bin_count − 1` is a heuristic. What matters is the shape: steep as
`dof → 1`, the normal quantile as `dof → ∞`.

Hill's algorithm (Hill, G. W. (1970), *Algorithm 396: Student's t-quantiles*,
Comm. ACM 13(10), 619–620) on the two-tailed probability: closed forms at
`dof = 1, 2`, otherwise a series in the normal deviate or (deep tail) in the
probability. Accuracy ≲ 2e-4 relative (worst at `dof ≈ 3–10`). Used instead of
`SpecialFunctions.beta_inc_inv`, which allocates over part of the `dof` range
(Tracking.jl's `test/track_in_place.jl` requires a zero-allocation pre-sync path)
and is 10–170× slower.
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
`_update_secondary_accumulators!`). O(hypotheses) per block, no growing history.
`period` is `blocks_per_bit` or the secondary-code length `N`.

The vectors are mutated in place across [`BitBuffer`](@ref) rebuilds: an immutable
form would be copied (~660 B) or boxed every block, allocating pre-sync.

Fields (all length `period` once seeded; empty before the first block):

  - `open_bin_sum`: coherent sum of the open bin (overlay-wiped for secondary).
  - `mean_bin_energy` / `bin_energy_sum_of_squared_deviations`: Welford mean and
    `M₂ = Σ(energyᵢ − mean)²` of the completed-bin energies; the bin count is
    `div(num_blocks - hypothesis, period)`.
  - `last_bin_polarity`: sign of the last completed bin's real part (`0` before).
"""
struct PhaseAccumulators
    open_bin_sum::Vector{ComplexF64}
    mean_bin_energy::Vector{Float64}
    bin_energy_sum_of_squared_deviations::Vector{Float64}
    last_bin_polarity::Vector{Int8}
end

PhaseAccumulators() = PhaseAccumulators(ComplexF64[], Float64[], Float64[], Int8[])

# Are the accumulators seeded for a `blocks_per_bit`-phase search yet?
@inline _is_seeded(accumulators::PhaseAccumulators, blocks_per_bit::Int) =
    length(accumulators.mean_bin_energy) == blocks_per_bit

# Size the accumulator vectors to `blocks_per_bit` phases and zero them.
function _seed_phase_accumulators!(accumulators::PhaseAccumulators, blocks_per_bit::Int)
    for vector in (
        accumulators.open_bin_sum,
        accumulators.mean_bin_energy,
        accumulators.bin_energy_sum_of_squared_deviations,
        accumulators.last_bin_polarity,
    )
        resize!(vector, blocks_per_bit)
    end
    _reset_phase_accumulators!(accumulators)
end

# Zero in place, keeping the length: on resync `_is_seeded` is still true, so
# without this the next block would fold into the old statistics.
function _reset_phase_accumulators!(accumulators::PhaseAccumulators)
    fill!(accumulators.open_bin_sum, zero(ComplexF64))
    fill!(accumulators.mean_bin_energy, 0.0)
    fill!(accumulators.bin_energy_sum_of_squared_deviations, 0.0)
    fill!(accumulators.last_bin_polarity, Int8(0))
    accumulators
end

"""
$(SIGNATURES)

Fold the `prompt` of the block at 0-based `block_index` into the
[`PhaseAccumulators`](@ref) of a `blocks_per_bit`-phase bit-edge search. Phase
`phase`'s bins start at block `phase`; the prompt joins every open bin, and the one
phase whose bin ends here folds `|bin_sum|²` into its Welford statistics and
records its polarity.
"""
function _update_phase_accumulators!(
    accumulators::PhaseAccumulators,
    prompt::ComplexF64,
    block_index::Int,
    blocks_per_bit::Int,
)
    @inbounds for phase = 0:(blocks_per_bit-1)
        block_index < phase && continue
        accumulators.open_bin_sum[phase+1] += prompt
        if (block_index - phase) % blocks_per_bit == blocks_per_bit - 1
            bin_sum = accumulators.open_bin_sum[phase+1]
            bin_energy = abs2(bin_sum)
            # Welford update.
            completed_bin_count = div(block_index + 1 - phase, blocks_per_bit)
            energy_delta = bin_energy - accumulators.mean_bin_energy[phase+1]
            accumulators.mean_bin_energy[phase+1] += energy_delta / completed_bin_count
            accumulators.bin_energy_sum_of_squared_deviations[phase+1] +=
                energy_delta * (bin_energy - accumulators.mean_bin_energy[phase+1])
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
[`PhaseAccumulators`](@ref) statistics and decide whether the peak is significant.
`period` is both the blocks per bin and the number of hypotheses; `confidence` is
the target `1 − P(false lock)`.

Returns `(accepted, peak_index, peak_bin_count)`: whether the gap is significant
(the caller still gates on the bin boundary), the 0-based winner (`-1` if none)
and its completed-bin count.

The true hypothesis keeps full coherent gain on every bin while wrong ones lose
energy (straddling a bit transition, or failing to wipe the overlay), so the mean
bin energy is the maximum-likelihood timing statistic. The noise scale is the
winner's *bin-to-bin* energy variance, which captures thermal noise and slow
drift. The peak is accepted only when

    z_score = energy_gap / standard_error
            ≥ t⁻¹(1 - false_alarm_probability/(period - 1);  ν = peak_bin_count − 1)

with the standard error over the peak and runner-up bin counts, the false-alarm
probability Bonferroni-split over the competitors, and `_t_quantile` as a
small-sample penalty. A real peak's `z_score` grows like √bins, so the detector
self-paces with C/N₀; a drift-only asymmetry stays bounded and never locks.
"""
@inline function _cfar_decide(
    mean_bin_energy::AbstractVector{Float64},
    bin_energy_sum_of_squared_deviations::AbstractVector{Float64},
    num_blocks::Int,
    period::Int,
    confidence::Float64,
)
    # Need two complete bins on some hypothesis.
    num_blocks < 2 * period && return (false, -1, 0)

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
        bin_count = div(num_blocks - h, period)
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
    z_threshold = _t_quantile(quantile_argument, peak_bin_count - 1)
    z_score < z_threshold && return (false, peak_index, peak_bin_count)

    (true, peak_index, peak_bin_count)
end

"""
$(SIGNATURES)

Soft CFAR bit-edge detector (see [`uses_soft_bit_edge_detection`](@ref); GPS L1 C/A)
running `_cfar_decide` over the edge phases `0:blocks_per_bit-1`.

Fires only when the latest block *ends* the winning phase's bit, so the upcoming
integration starts a fresh bit (rules out the off-by-one lock of JuliaGNSS/Tracking.jl#124).
`polarity` is the sign of the last completed bin's sum.
"""
function _detect_bit_edge_cfar(
    accumulators::PhaseAccumulators,
    blocks_per_bit::Int,
    confidence::Float64,
    num_blocks::Int,
)
    accepted, peak_phase, _ = _cfar_decide(
        accumulators.mean_bin_energy,
        accumulators.bin_energy_sum_of_squared_deviations,
        num_blocks,
        blocks_per_bit,
        confidence,
    )
    accepted || return SyncResult(false, 0, Int8(0))

    num_blocks % blocks_per_bit != peak_phase && return SyncResult(false, 0, Int8(0))

    @inbounds polarity =
        accumulators.last_bin_polarity[peak_phase+1] < 0 ? Int8(-1) : Int8(+1)
    SyncResult(true, 0, polarity)
end

"""
$(SIGNATURES)

The secondary-code analog of `_update_phase_accumulators!`: rotation `d ∈ 0:N-1`
has `N`-block bins starting at block `d`, and block `i` is multiplied by secondary
chip `mod(i - d, N)` (the overlay the post-sync replica applies) before summing,
so the correct rotation adds coherently.
"""
function _update_secondary_accumulators!(
    accumulators::PhaseAccumulators,
    prompt::ComplexF64,
    block_index::Int,
    secondary_code_length::Int,
    signal::AbstractGNSSSignal,
    prn::Integer,
)
    N = secondary_code_length
    secondary_code = get_secondary_code(signal)
    @inbounds for d = 0:(N-1)
        block_index < d && continue
        chip = (block_index - d) % N
        overlay = GNSSSignals.secondary_value(secondary_code, prn, chip)
        accumulators.open_bin_sum[d+1] += prompt * overlay
        if chip == N - 1
            bin_sum = accumulators.open_bin_sum[d+1]
            bin_energy = abs2(bin_sum)
            # Welford update.
            completed_bin_count = div(block_index + 1 - d, N)
            energy_delta = bin_energy - accumulators.mean_bin_energy[d+1]
            accumulators.mean_bin_energy[d+1] += energy_delta / completed_bin_count
            accumulators.bin_energy_sum_of_squared_deviations[d+1] +=
                energy_delta * (bin_energy - accumulators.mean_bin_energy[d+1])
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

Fires only when the latest block ends the winning rotation's period, so the
upcoming integration starts at secondary chip 0 and `SyncResult.phase` is `0`.
"""
function _detect_secondary_code_cfar(
    accumulators::PhaseAccumulators,
    secondary_code_length::Int,
    confidence::Float64,
    num_blocks::Int,
)
    accepted, peak_rotation, _ = _cfar_decide(
        accumulators.mean_bin_energy,
        accumulators.bin_energy_sum_of_squared_deviations,
        num_blocks,
        secondary_code_length,
        confidence,
    )
    accepted || return SyncResult(false, 0, Int8(0))

    num_blocks % secondary_code_length != peak_rotation &&
        return SyncResult(false, 0, Int8(0))

    @inbounds polarity =
        accumulators.last_bin_polarity[peak_rotation+1] < 0 ? Int8(-1) : Int8(+1)
    SyncResult(true, 0, polarity)
end

"""
$(SIGNATURES)

**Hard-decision** secondary-code rotation search (currently the 1800-chip overlay
pilots GPS L1C-P and BeiDou B1C-P); locks within one secondary period.

`received` is the prompt-sign window, newest block in bit 0; `reference` is packed
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
    # `one(B) << N` is undefined for an exact-width buffer (UInt1800).
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

Detector for signals with no sub-block boundary to find: `found = true` from the
first integration. Covers one symbol per code period (GPS L1C-D, L2CM, Galileo
E1B / E6-B, BeiDou B2b-I / B1C-D; GNSSDecoder.jl resolves the ±1 polarity from
the preamble) and Galileo E5a-QP (no data, no overlay; the lock only gates
whole-code-cycle integration).
"""
@inline function _detect_symbol_is_code_block_sync(
    ::AbstractGNSSSignal,
    ::Integer,           # PRN — ignored
    ::Unsigned,
    ::Integer,
)
    SyncResult(true, 0, Int8(+1))
end

"""
$(SIGNATURES)

Hard-path detector for secondary-coded signals: once `N` blocks are buffered, run
`_secondary_code_search` against `_packed_secondary_code` allowing
`floor(tolerance × N)` errors ([`get_bit_edge_or_secondary_code_tolerance`](@ref)).

A new secondary-coded signal needs `get_secondary_code`, a
`get_code_block_buffer_type` of at least `N` bits and a
`detect_bit_or_secondary_code_sync` method delegating here (reached by soft-path
signals only if their trait is overridden).
"""
@inline function _detect_secondary_code_sync(
    signal::AbstractGNSSSignal,
    prn::Integer,
    code_block_bits::B,
    num_code_blocks::Integer,
) where {B<:Unsigned}
    secondary_code_length = get_secondary_code_length(signal)
    num_code_blocks < secondary_code_length && return SyncResult(false, 0, Int8(0))
    max_errors =
        floor(Int, get_bit_edge_or_secondary_code_tolerance(signal) * secondary_code_length)
    _secondary_code_search(
        code_block_bits,
        _packed_secondary_code(B, signal, prn),
        secondary_code_length,
        max_errors,
    )
end

"""
$(SIGNATURES)

The secondary code for `prn` packed into `B` in `_secondary_code_search`'s order.
Specialize only for an overlay not reachable through `get_secondary_code`.
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

Bit sync state and decoded soft bits of one signal.

`code_block_buffer` is the pre-sync window of prompt signs, `B` bits wide
([`get_code_block_buffer_type`](@ref)), unused after sync. `soft_bits` collects
the decoded bits (see [`get_soft_bits`](@ref)), unbounded between resets.
`phase_acc` holds the soft CFAR detector's [`PhaseAccumulators`](@ref), empty on
the hard path.
"""
struct BitBuffer{B<:Unsigned}
    code_block_buffer::B
    code_block_buffer_length::Int
    found::Bool
    secondary_phase::Int      # 0 until found; secondary-chip offset post-sync
    polarity::Int8            # +1 or -1 once found; 0 before sync
    prompt_accumulator::ComplexF64
    prompt_accumulator_integrated_code_blocks::Int
    soft_bits::Vector{Float32}
    phase_acc::PhaseAccumulators
end

# Untyped default: a `UInt128` search buffer.
function BitBuffer()
    BitBuffer{UInt128}(
        zero(UInt128),
        0,
        false,
        0,
        Int8(0),
        complex(0.0, 0.0),
        0,
        Float32[],
        PhaseAccumulators(),
    )
end

function BitBuffer{B}() where {B<:Unsigned}
    BitBuffer{B}(
        zero(B),
        0,
        false,
        0,
        Int8(0),
        complex(0.0, 0.0),
        0,
        Float32[],
        PhaseAccumulators(),
    )
end

# For tests and benchmarks: zero phase / polarity, empty accumulators; `soft_bits`
# is aliased, not copied.
function BitBuffer(
    code_block_buffer::B,
    code_block_buffer_length::Integer,
    found::Bool,
    prompt_accumulator::Complex,
    prompt_accumulator_integrated_code_blocks::Integer,
    soft_bits::Vector{Float32} = Float32[],
) where {B<:Unsigned}
    BitBuffer{B}(
        code_block_buffer,
        Int(code_block_buffer_length),
        found,
        0,
        Int8(0),
        ComplexF64(prompt_accumulator),
        Int(prompt_accumulator_integrated_code_blocks),
        soft_bits,
        PhaseAccumulators(),
    )
end

@inline length(bit_buffer::BitBuffer) = Base.length(bit_buffer.soft_bits)
"""
    has_bit_or_secondary_code_been_found(bit_buffer::BitBuffer)
    has_bit_or_secondary_code_been_found(state::SignalLoopState)

Whether the navigation-bit boundary or secondary-code phase has been found; from
then on the signal may integrate over several code periods and bits are collected
for [`get_soft_bits`](@ref).
"""
@inline has_bit_or_secondary_code_been_found(bit_buffer::BitBuffer) = bit_buffer.found

"""
    get_soft_bits(bit_buffer::BitBuffer)
    get_soft_bits(state::SignalLoopState)

The soft bits decoded so far: per navigation bit, the polarity-corrected real part
of the summed de-rotated prompts. The sign is the hard decision, the magnitude the
reliability. Bits recovered from the pre-sync window are sign votes scaled by the
sync-time prompt magnitude, to stay comparable.

The buffer's own vector, not a copy: drain it (`empty!`) after reading; it holds
64 bits before it allocates.
"""
@inline get_soft_bits(bit_buffer::BitBuffer) = bit_buffer.soft_bits

"""
$(SIGNATURES)

Concrete `Unsigned` width `B` of `BitBuffer.code_block_buffer` for `signal`
(default `UInt64`), a type parameter so construction is type-stable.

Hard-path signals need at least `N` bits (the 1800-chip overlay pilots use
`UInt1800`); soft-detector signals keep a width holding their horizon so the hard
path stays available; signals with nothing to search use `UInt8`. Per-signal
values: [Code-block buffer widths](@ref).
"""
@inline get_code_block_buffer_type(::AbstractGNSSSignal) = UInt64

"""
$(SIGNATURES)

Hamming tolerance of the **hard-decision** `_secondary_code_search`: the largest
fraction of chip errors accepted, `max_errors = floor(Int, tolerance × N)`. Default
`0.025` (45 of 1800 chips). Only hard-path signals (GPS L1C-P, BeiDou B1C-P) read
it; the soft detectors use [`get_bit_edge_detection_confidence`](@ref).

# Overriding

E.g. to loosen it for low-C/N₀ work:

```julia
TrackingLoops.get_bit_edge_or_secondary_code_tolerance(::GPSL1C_P) = 0.05
```

Takes effect at the next detector call.
"""
@inline get_bit_edge_or_secondary_code_tolerance(::AbstractGNSSSignal) = 0.025

"""
$(SIGNATURES)

Whether `signal`'s bit edge is found by the soft CFAR detector
`_detect_bit_edge_cfar` instead of the hard `detect_bit_or_secondary_code_sync`.

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
    _calc_num_code_blocks_that_form_a_bit(signal) > 1 &&
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
TrackingLoops.get_bit_edge_detection_confidence(::GPSL1CA) = 0.9999
```

Takes effect at the next detector call.
"""
@inline get_bit_edge_detection_confidence(::AbstractGNSSSignal) = 0.999

# Primary-code blocks per navigation bit; 0 for pilots, which callers must guard.
@inline function _calc_num_code_blocks_that_form_a_bit(signal::AbstractGNSSSignal)
    data_freq = get_data_frequency(signal)
    iszero(data_freq) && return 0
    Int(get_code_frequency(signal) / (get_code_length(signal) * data_freq))
end

"""
$(SIGNATURES)

Advance the bit buffer by one record of `integrated_code_blocks` blocks.

Post-sync, one soft bit is emitted each time the block count reaches a bit. A
record carrying the count *past* the boundary (only from a non-bit-aligned external
producer) drops sync instead (JuliaGNSS/Tracking.jl#238): its energy belongs to two
bits. Silent; [`apply_record`](@ref) reports it as `overshoot`.
"""
function buffer(
    signal::AbstractGNSSSignal,
    prn::Integer,
    bit_buffer::BitBuffer{B},
    integrated_code_blocks,
    prompt,
) where {B<:Unsigned}
    num_code_blocks_that_form_a_bit = _calc_num_code_blocks_that_form_a_bit(signal)

    if (bit_buffer.found == false)
        return _buffer_find_bit(
            signal,
            prn,
            bit_buffer,
            num_code_blocks_that_form_a_bit,
            integrated_code_blocks,
            prompt,
        )
    end

    # Pilots: nothing to decode; the lock still anchors the code phase.
    num_code_blocks_that_form_a_bit == 0 && return bit_buffer

    prompt_accumulator = bit_buffer.prompt_accumulator + prompt
    prompt_accumulator_integrated_code_blocks =
        bit_buffer.prompt_accumulator_integrated_code_blocks + integrated_code_blocks

    if prompt_accumulator_integrated_code_blocks > num_code_blocks_that_form_a_bit
        # Overshoot (e.g. 18 + a 3-block record for L1 C/A): resync rather than
        # emit at `>=`, see the docstring. Must not allocate or format a string.
        return _resync_bit_buffer(bit_buffer)
    elseif prompt_accumulator_integrated_code_blocks == num_code_blocks_that_form_a_bit
        bit_acc =
            bit_buffer.polarity < 0 ? -real(prompt_accumulator) : real(prompt_accumulator)
        push!(bit_buffer.soft_bits, Float32(bit_acc))
        return BitBuffer{B}(
            bit_buffer.code_block_buffer,
            bit_buffer.code_block_buffer_length,
            true,
            bit_buffer.secondary_phase,
            bit_buffer.polarity,
            zero(prompt_accumulator),
            0,
            bit_buffer.soft_bits,
            bit_buffer.phase_acc,
        )
    else
        return BitBuffer{B}(
            bit_buffer.code_block_buffer,
            bit_buffer.code_block_buffer_length,
            true,
            bit_buffer.secondary_phase,
            bit_buffer.polarity,
            prompt_accumulator,
            prompt_accumulator_integrated_code_blocks,
            bit_buffer.soft_bits,
            bit_buffer.phase_acc,
        )
    end
end

# Back to pre-sync after an overshoot; `soft_bits` is kept (same vector).
@inline function _resync_bit_buffer(bit_buffer::BitBuffer{B}) where {B<:Unsigned}
    BitBuffer{B}(
        zero(B),
        0,
        false,
        0,
        Int8(0),
        complex(0.0, 0.0),
        0,
        bit_buffer.soft_bits,
        _reset_phase_accumulators!(bit_buffer.phase_acc),
    )
end

function _buffer_find_bit(
    signal,
    prn::Integer,
    bit_buffer::BitBuffer{B},
    num_code_blocks_that_form_a_bit,
    integrated_code_blocks,
    prompt,
) where {B<:Unsigned}
    if (integrated_code_blocks != 1)
        error(
            "The number code blocks must be equal to 1 if bit or secondary code " *
            "hasn't been found yet.",
        )
    end
    code_block_buffer = (bit_buffer.code_block_buffer << 1) + B(real(prompt) > 0)
    code_block_buffer_length = bit_buffer.code_block_buffer_length + 1

    # Soft detectors share `phase_acc`; the rest take the hard path. Folds at
    # compile time.
    phase_acc = bit_buffer.phase_acc
    if uses_soft_bit_edge_detection(signal)
        blocks_per_bit = num_code_blocks_that_form_a_bit
        _is_seeded(phase_acc, blocks_per_bit) ||
            _seed_phase_accumulators!(phase_acc, blocks_per_bit)
        _update_phase_accumulators!(
            phase_acc,
            ComplexF64(prompt),
            code_block_buffer_length - 1,
            blocks_per_bit,
        )
        sync = _detect_bit_edge_cfar(
            phase_acc,
            blocks_per_bit,
            get_bit_edge_detection_confidence(signal),
            code_block_buffer_length,
        )
    elseif uses_soft_secondary_code_detection(signal)
        secondary_code_length = get_secondary_code_length(signal)
        _is_seeded(phase_acc, secondary_code_length) ||
            _seed_phase_accumulators!(phase_acc, secondary_code_length)
        _update_secondary_accumulators!(
            phase_acc,
            ComplexF64(prompt),
            code_block_buffer_length - 1,
            secondary_code_length,
            signal,
            prn,
        )
        sync = _detect_secondary_code_cfar(
            phase_acc,
            secondary_code_length,
            get_bit_edge_detection_confidence(signal),
            code_block_buffer_length,
        )
    else
        sync = detect_bit_or_secondary_code_sync(
            signal,
            prn,
            code_block_buffer,
            code_block_buffer_length,
        )
    end
    if !sync.found
        return BitBuffer{B}(
            code_block_buffer,
            code_block_buffer_length,
            false,
            0,
            Int8(0),
            complex(0.0, 0.0),
            0,
            bit_buffer.soft_bits,
            phase_acc,
        )
    end
    if get_secondary_code_length(signal) > 1
        # Pre-sync signs carry the overlay, not data: no bits to recover. Seed the
        # block count with `sync.phase` so the first bit ends on the boundary
        # (JuliaGNSS/Tracking.jl#125).
        return BitBuffer{B}(
            code_block_buffer,
            code_block_buffer_length,
            true,
            sync.phase,
            sync.polarity,
            complex(0.0, 0.0),
            sync.phase,
            bit_buffer.soft_bits,
            phase_acc,
        )
    end
    if num_code_blocks_that_form_a_bit == 0
        # Dataless, no overlay (Galileo E5a-QP): avoid dividing by zero below.
        return BitBuffer{B}(
            code_block_buffer,
            code_block_buffer_length,
            true,
            0,
            sync.polarity,
            complex(0.0, 0.0),
            0,
            bit_buffer.soft_bits,
            phase_acc,
        )
    end
    num_bits = min(
        div(code_block_buffer_length, num_code_blocks_that_form_a_bit),
        div(sizeof(code_block_buffer) * 8, num_code_blocks_that_form_a_bit),
    )
    # Hoisted: capturing `sync` (assigned in several branches) would box it.
    lock_polarity = Int(sync.polarity)
    for bit_index = num_bits:-1:1     # oldest recovered bit first
        # Lock polarity applies here too (JuliaGNSS/Tracking.jl#127).
        bit_sum =
            sum(0:(num_code_blocks_that_form_a_bit-1)) do code_block_index
                buffer_code_block_index =
                    (bit_index - 1) * num_code_blocks_that_form_a_bit + code_block_index
                ((code_block_buffer & (one(B) << buffer_code_block_index)) > 0) * 2 - 1
            end * lock_polarity
        # Sign votes scaled to soft-bit units (see `get_soft_bits`).
        push!(bit_buffer.soft_bits, Float32(bit_sum * abs(prompt)))
    end
    return BitBuffer{B}(
        code_block_buffer,
        code_block_buffer_length,
        true,
        sync.phase,
        sync.polarity,
        complex(0, 0),
        0,
        bit_buffer.soft_bits,
        phase_acc,
    )
end

"""
$(SIGNATURES)

Advance a just-synced `bit_buffer`'s `secondary_phase` by `num_code_blocks`
(mod `N`); no-op without a secondary code.

The detector reports the phase for the block after the syncing record, but
Tracking.jl reads it only after the whole chunk, so every later record of the
chunk (`correlated_pre_sync`) must move it along, or the replica applies the wrong
overlay chip (cf. JuliaGNSS/Tracking.jl#219).
"""
@inline function _advance_secondary_phase(
    signal::AbstractGNSSSignal,
    bit_buffer::BitBuffer{B},
    num_code_blocks::Integer,
) where {B<:Unsigned}
    secondary_code_length = get_secondary_code_length(signal)
    secondary_code_length > 1 || return bit_buffer
    BitBuffer{B}(
        bit_buffer.code_block_buffer,
        bit_buffer.code_block_buffer_length,
        bit_buffer.found,
        mod(bit_buffer.secondary_phase + Int(num_code_blocks), secondary_code_length),
        bit_buffer.polarity,
        bit_buffer.prompt_accumulator,
        bit_buffer.prompt_accumulator_integrated_code_blocks,
        bit_buffer.soft_bits,
        bit_buffer.phase_acc,
    )
end

function reset(bit_buffer::BitBuffer{B}) where {B<:Unsigned}
    empty!(bit_buffer.soft_bits)
    BitBuffer{B}(
        bit_buffer.code_block_buffer,
        bit_buffer.code_block_buffer_length,
        bit_buffer.found,
        bit_buffer.secondary_phase,
        bit_buffer.polarity,
        bit_buffer.prompt_accumulator,
        bit_buffer.prompt_accumulator_integrated_code_blocks,
        bit_buffer.soft_bits,
        bit_buffer.phase_acc,
    )
end

"""
    get_sync_polarity(signal, bit_buffer, prn) -> Int8

A pilot's prompt polarity from its secondary-code sync, or `0`. Nonzero, the replica
wipes every sign modulation off the prompt, so both discriminators are four-quadrant:
the PLL reads the prompt with this sign (`pll_disc(...; polarity)`), the FLL
`atan(cross, dot)`. `0` keeps both two-quadrant: data signals, pilots before sync and
pilots without a secondary code (GPS L2 CL, Galileo E5a-QP). Pass the bit buffer as
it was when the record was correlated; the result is a [`LoopRecord`](@ref)'s
`polarity`.

The pre-sync replica carries secondary chip 0 on every block, so the post-sync
prompt has the sync polarity times chip 0. A half-cycle Costas slip before the
switch is pulled over once: the start of the resolved phase, not a slip within it.
"""
@inline function get_sync_polarity(
    signal::AbstractGNSSSignal,
    bit_buffer::BitBuffer,
    prn::Integer,
)
    iszero(get_data_frequency(signal)) &&
    get_secondary_code_length(signal) > 1 &&
    has_bit_or_secondary_code_been_found(bit_buffer) || return Int8(0)
    chip0 = GNSSSignals.secondary_value(get_secondary_code(signal), prn, 0)
    Int8(bit_buffer.polarity * sign(chip0))
end
