@testset "An NCO timeline averages the words that ran over a span" begin
    tl = NCOTimeline()
    reset_timeline!(tl, 100.0, 0.1)
    @test mean_nco_word(tl, 0, 4000) == (100.0, 0.1)
    @test nco_word_at(tl, 12345) == (100.0, 0.1)
    @test !word_changes_within(tl, 0, 10_000)

    schedule_word!(tl, 1000, 110.0, 0.11)
    schedule_word!(tl, 2000, 120.0, 0.12)
    # Time-weighted: half the span on each word.
    @test mean_nco_word(tl, 500, 1500) == (105.0, 0.105)
    @test mean_nco_word(tl, 0, 4000) == (112.5, 0.1125)
    # An empty span is the word in effect at its start.
    @test mean_nco_word(tl, 1000, 1000) == (110.0, 0.11)
    @test mean_nco_word(tl, 999, 999) == (100.0, 0.1)
    @test nco_word_at(tl, 2000) == (120.0, 0.12)
    @test all(mean_nco_word(tl, 999.5, 1000.5) .≈ (105.0, 0.105))
    @test word_changes_within(tl, 0, 1000)
    @test !word_changes_within(tl, 1001, 2000 - 1)
    @test word_changes_within(tl, 1999, 2000)
    # Back-to-back records meeting where a word lands ran on different words,
    # exactly as `mean_nco_word` attributes them.
    @test word_changes_within(tl, 1000, 1000)
    @test mean_nco_word(tl, 0, 1000) != mean_nco_word(tl, 1000, 2000)
    @test !word_changes_within(tl, 1500, 1500)

    # A newer command for the same or an earlier sample supersedes.
    schedule_word!(tl, 2000, 130.0, 0.13)
    @test mean_nco_word(tl, 2000, 3000) == (130.0, 0.13)
    @test length(scheduled_words(tl)) == 2
    schedule_word!(tl, 1500, 140.0, 0.14)
    @test [w.sample for w in scheduled_words(tl)] == [1000, 1500]

    # Promotion folds landed words into the applied one and forgets them.
    promote_words!(tl, 1200)
    @test tl.applied_carrier_doppler == 110.0
    @test [w.sample for w in scheduled_words(tl)] == [1500]
    @test mean_nco_word(tl, 1000, 2000) == (125.0, 0.125)
    promote_words!(tl, 10_000)
    @test isempty(scheduled_words(tl))
    @test mean_nco_word(tl, 0, 1) == (140.0, 0.14)

    reset_timeline!(tl, 7.0, 0.007)
    @test isempty(scheduled_words(tl))
    @test mean_nco_word(tl, 0, 1) == (7.0, 0.007)

    @test mean_nco_word(FixedNCOWord(3.0, 0.3), 0, 100) == (3.0, 0.3)
end

@testset "A timeline never grows past its capacity" begin
    @test_throws ArgumentError NCOTimeline(; capacity = 0)
    tl1 = NCOTimeline(; capacity = 1)
    schedule_word!(tl1, 1000, 1.0, 0.0)
    schedule_word!(tl1, 2000, 2.0, 0.0)
    @test [w.sample for w in scheduled_words(tl1)] == [2000]
    @test tl1.applied_carrier_doppler == 1.0
    tl = NCOTimeline(; capacity = 4)
    reset_timeline!(tl, 1.0, 0.1)
    for k = 1:10
        schedule_word!(tl, 1000k, Float64(k), 0.0)
    end
    @test length(scheduled_words(tl)) == 4
    @test [w.sample for w in scheduled_words(tl)] == [7000, 8000, 9000, 10_000]
    # The words that fell off the front were taken as landed.
    @test tl.applied_carrier_doppler == 6.0
end

@testset "A word is rescheduled to where it really landed" begin
    tl = NCOTimeline()
    reset_timeline!(tl, 100.0, 0.1)
    schedule_word!(tl, 1000, 110.0, 0.11)
    schedule_word!(tl, 2000, 120.0, 0.12)
    reschedule_word!(tl, 2000, 2500)
    @test [w.sample for w in scheduled_words(tl)] == [1000, 2500]
    @test nco_word_at(tl, 2200) == (110.0, 0.11)
    @test nco_word_at(tl, 2500) == (120.0, 0.12)
    # Landed before an earlier word: that word is superseded, as the device
    # keeps the newest command it was given.
    reschedule_word!(tl, 2500, 900)
    @test [w.sample for w in scheduled_words(tl)] == [900]
    @test nco_word_at(tl, 950) == (120.0, 0.12)
    # Nothing scheduled there: a no-op.
    reschedule_word!(tl, 4242, 5000)
    @test [w.sample for w in scheduled_words(tl)] == [900]
end
