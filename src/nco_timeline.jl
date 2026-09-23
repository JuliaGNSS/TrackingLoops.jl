# ─────────────────────────────────────────────────────────────────────────────
# What the device NCO ran: per-channel word timelines (moved here from
# GNSSReceiver's hardware_correlator.jl, unchanged in arithmetic).
# ─────────────────────────────────────────────────────────────────────────────

"\"No landing sample\": the command acts at each record's end, i.e. no delay."
const NO_LANDING_SAMPLE = typemin(Int64)

# One NCO word the host has scheduled at a named device sample. Plain `Float64`
# Hz.
struct ScheduledNCOWord
    sample::Int64
    carrier_doppler::Float64
    code_doppler::Float64
end

"""
    NCOTimeline(; capacity = 64)

The carrier and code words one hardware channel's NCOs ran and will run: the
word in effect now and the words already scheduled at named device samples.

A hardware loop is closed through a device that holds each word until the next
one lands, milliseconds after the record that motivated it ended. The loop
filter therefore cannot assume that the word it last computed is the one a
record was integrated under, and a correction computed against the wrong word
restates an error the device is already about to remove. The timeline is the
record of what the NCO actually did, so the estimator can attribute every
record to the word that really ran ([`mean_nco_word`](@ref)) and size its
correction for the moment it will land ([`NCOReferencedPLLAndDLL`](@ref)).

The scheduled words live in a fixed-size vector of `capacity` entries with a
count, so scheduling and promoting words never touch the allocator. Should more
words than that ever be in flight, the oldest is taken as landed. Read them
with [`scheduled_words`](@ref).
"""
mutable struct NCOTimeline
    applied_carrier_doppler::Float64
    applied_code_doppler::Float64
    # `words[1:count]`, ascending in `sample`; every entry lands strictly after
    # the applied word.
    const words::Vector{ScheduledNCOWord}
    count::Int
end

NCOTimeline(; capacity::Integer = 64) = NCOTimeline(
    0.0,
    0.0,
    [ScheduledNCOWord(0, 0.0, 0.0) for _ = 1:capacity],
    0,
)

"The words scheduled and not yet landed, oldest first (a view)."
scheduled_words(timeline::NCOTimeline) = view(timeline.words, 1:timeline.count)
Base.isempty(timeline::NCOTimeline) = timeline.count == 0

"A handover: the device starts on these words and nothing is in flight."
function reset_timeline!(timeline::NCOTimeline, carrier_doppler_hz, code_doppler_hz)
    timeline.applied_carrier_doppler = Float64(carrier_doppler_hz)
    timeline.applied_code_doppler = Float64(code_doppler_hz)
    timeline.count = 0
    timeline
end

"""
    schedule_word!(timeline, sample, carrier_doppler_hz, code_doppler_hz)

Record a word the device has accepted for `sample`. A device keeps the newest
command for a given sample, and a later command is never scheduled for an
earlier sample, so anything queued at or past `sample` is superseded.
"""
function schedule_word!(timeline::NCOTimeline, sample, carrier_doppler_hz, code_doppler_hz)
    words = timeline.words
    count = timeline.count
    @inbounds while count > 0 && words[count].sample >= sample
        count -= 1
    end
    if count >= length(words)
        # Never grow: the oldest word has certainly landed by the time this many
        # are in flight.
        @inbounds oldest = words[1]
        timeline.applied_carrier_doppler = oldest.carrier_doppler
        timeline.applied_code_doppler = oldest.code_doppler
        @inbounds for i = 1:(count-1)
            words[i] = words[i+1]
        end
        count -= 1
    end
    @inbounds words[count+1] =
        ScheduledNCOWord(Int64(sample), Float64(carrier_doppler_hz), Float64(code_doppler_hz))
    timeline.count = count + 1
    timeline
end

"""
    reschedule_word!(timeline, from_sample, to_sample)

The word scheduled at `from_sample` really landed at `to_sample`: move it.
Anything scheduled at or past `to_sample` is superseded, as for
[`schedule_word!`](@ref). A no-op when nothing is scheduled at `from_sample`.
"""
function reschedule_word!(timeline::NCOTimeline, from_sample, to_sample)
    words = timeline.words
    count = timeline.count
    index = 0
    @inbounds for i = count:-1:1
        if words[i].sample == from_sample
            index = i
            break
        end
    end
    index == 0 && return timeline
    @inbounds word = words[index]
    @inbounds for i = index:(count-1)
        words[i] = words[i+1]
    end
    timeline.count = count - 1
    schedule_word!(timeline, to_sample, word.carrier_doppler, word.code_doppler)
end

"""
    promote_words!(timeline, sample)

Everything scheduled at or before `sample` has landed: fold it into the applied
word. Only call this once no query will start before `sample` again.
"""
function promote_words!(timeline::NCOTimeline, sample)
    words = timeline.words
    count = timeline.count
    n = 0
    @inbounds for i = 1:count
        word = words[i]
        word.sample <= sample || break
        timeline.applied_carrier_doppler = word.carrier_doppler
        timeline.applied_code_doppler = word.code_doppler
        n += 1
    end
    if n > 0
        @inbounds for i = 1:(count-n)
            words[i] = words[i+n]
        end
        timeline.count = count - n
    end
    timeline
end

"Whether a scheduled word takes effect in `(lo, hi]`, i.e. whether records ending at `lo` and starting at `hi` ran on different words."
function word_changes_within(timeline::NCOTimeline, lo, hi)
    @inbounds for i = 1:timeline.count
        lo < timeline.words[i].sample <= hi && return true
    end
    false
end

"The word in effect at device sample `sample`, as `(carrier_hz, code_hz)`."
function nco_word_at(timeline::NCOTimeline, sample)
    carrier, code = timeline.applied_carrier_doppler, timeline.applied_code_doppler
    @inbounds for i = 1:timeline.count
        word = timeline.words[i]
        word.sample <= sample || break
        carrier, code = word.carrier_doppler, word.code_doppler
    end
    carrier, code
end

"""
    mean_nco_word(words, a, b) -> (carrier_doppler_hz, code_doppler_hz)

Time-weighted mean of the carrier and code words in effect over the device
samples `[a, b)` — the replica frequencies a record integrated over that span
was really correlated with. For `b <= a` the word in effect at `a`.

`words` is an [`NCOTimeline`](@ref) for a hardware channel, or a
[`FixedNCOWord`](@ref) where the replica ran on one known word (the software
receiver regenerates its replicas from the satellite's Doppler every chunk).
"""
function mean_nco_word(timeline::NCOTimeline, a::Real, b::Real)
    total = b - a
    total > 0 || return nco_word_at(timeline, a)
    carrier, code = timeline.applied_carrier_doppler, timeline.applied_code_doppler
    carrier_sum = 0.0
    code_sum = 0.0
    t = a
    @inbounds for i = 1:timeline.count
        word = timeline.words[i]
        word.sample >= b && break
        if word.sample > t
            carrier_sum += carrier * (word.sample - t)
            code_sum += code * (word.sample - t)
            t = word.sample
        end
        carrier, code = word.carrier_doppler, word.code_doppler
    end
    carrier_sum += carrier * (b - t)
    code_sum += code * (b - t)
    carrier_sum / total, code_sum / total
end

"""
    FixedNCOWord(carrier_doppler_hz, code_doppler_hz)

A replica that ran on one known word for every span asked about — what the
software receiver's replicas do within a chunk. See [`mean_nco_word`](@ref).
"""
struct FixedNCOWord
    carrier_doppler::Float64
    code_doppler::Float64
end

mean_nco_word(word::FixedNCOWord, a::Real, b::Real) = word.carrier_doppler, word.code_doppler
nco_word_at(word::FixedNCOWord, sample) = word.carrier_doppler, word.code_doppler
