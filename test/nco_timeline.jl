# The samples the commands are scheduled at, oldest first.
scheduled_samples(timeline) =
    [c.sample_index for c in view(timeline.scheduled_commands, 1:timeline.num_scheduled_commands)]

@testset "An NCO timeline averages the commands that ran over a span" begin
    tl = NCOTimeline()
    reset_timeline!(tl, 100.0Hz, 0.1Hz)
    @test calc_mean_nco_command(tl, 0, 4000) == NCOCommand(100.0Hz, 0.1Hz)
    @test get_nco_command(tl, 12345) == NCOCommand(100.0Hz, 0.1Hz)

    schedule_command!(tl, 1000, 110.0Hz, 0.11Hz)
    schedule_command!(tl, 2000, 120.0Hz, 0.12Hz)
    # Time-weighted: half the span on each command.
    @test calc_mean_nco_command(tl, 500, 1500) == NCOCommand(105.0Hz, 0.105Hz)
    @test calc_mean_nco_command(tl, 0, 4000) == NCOCommand(112.5Hz, 0.1125Hz)
    # An empty span is the command in effect at its start.
    @test calc_mean_nco_command(tl, 1000, 1000) == NCOCommand(110.0Hz, 0.11Hz)
    @test calc_mean_nco_command(tl, 999, 999) == NCOCommand(100.0Hz, 0.1Hz)
    @test get_nco_command(tl, 2000) == NCOCommand(120.0Hz, 0.12Hz)
    straddling = calc_mean_nco_command(tl, 999.5, 1000.5)
    @test straddling.carrier_doppler ≈ 105.0Hz
    @test straddling.code_doppler ≈ 0.105Hz
    # Back-to-back integrations meeting where a command takes effect ran on different commands.
    @test calc_mean_nco_command(tl, 0, 1000) != calc_mean_nco_command(tl, 1000, 2000)

    # A newer command for the same or an earlier sample supersedes.
    schedule_command!(tl, 2000, 130.0Hz, 0.13Hz)
    @test calc_mean_nco_command(tl, 2000, 3000) == NCOCommand(130.0Hz, 0.13Hz)
    @test length(scheduled_samples(tl)) == 2
    schedule_command!(tl, 1500, 140.0Hz, 0.14Hz)
    @test scheduled_samples(tl) == [1000, 1500]

    # Advancing folds the applied commands into the one in effect and forgets them.
    advance_timeline!(tl, 1200)
    @test tl.applied_command.carrier_doppler == 110.0Hz
    @test scheduled_samples(tl) == [1500]
    @test calc_mean_nco_command(tl, 1000, 2000) == NCOCommand(125.0Hz, 0.125Hz)
    advance_timeline!(tl, 10_000)
    @test isempty(scheduled_samples(tl))
    @test calc_mean_nco_command(tl, 0, 1) == NCOCommand(140.0Hz, 0.14Hz)

    reset_timeline!(tl, 7.0Hz, 0.007Hz)
    @test isempty(scheduled_samples(tl))
    @test calc_mean_nco_command(tl, 0, 1) == NCOCommand(7.0Hz, 0.007Hz)

    @test calc_mean_nco_command(ConstantNCOCommand(3.0Hz, 0.3Hz), 0, 100) == NCOCommand(3.0Hz, 0.3Hz)
end

@testset "A full timeline refuses a command and keeps its history" begin
    @test_throws ArgumentError NCOTimeline(; capacity = 0)
    tl = NCOTimeline(; capacity = 2)
    reset_timeline!(tl, 10.0Hz, 0.01Hz)
    @test schedule_command!(tl, 100, 1.0Hz, 0.0Hz)
    @test schedule_command!(tl, 200, 2.0Hz, 0.0Hz)
    @test !schedule_command!(tl, 300, 3.0Hz, 0.0Hz)
    # Refused, nothing moved: before the first command the initial one is in effect.
    @test scheduled_samples(tl) == [100, 200]
    @test get_nco_command(tl, 0) == NCOCommand(10.0Hz, 0.01Hz)
    @test get_nco_command(tl, 300) == NCOCommand(2.0Hz, 0.0Hz)
    # A command superseding the suffix needs no room.
    @test schedule_command!(tl, 150, 1.5Hz, 0.0Hz)
    @test scheduled_samples(tl) == [100, 150]
    # Advancing makes room.
    advance_timeline!(tl, 100)
    @test schedule_command!(tl, 300, 3.0Hz, 0.0Hz)
    @test scheduled_samples(tl) == [150, 300]
    @test tl.applied_command == NCOCommand(1.0Hz, 0.0Hz)
end

@testset "Hardware feedback preserves unrelated pending commands" begin
    tl = NCOTimeline(; capacity = 3)
    reset_timeline!(tl, 10.0Hz, 0.01Hz)
    schedule_command!(tl, 1000, 11.0Hz, 0.011Hz)
    schedule_command!(tl, 2000, 12.0Hz, 0.012Hz)
    schedule_command!(tl, 3000, 13.0Hz, 0.013Hz)
    # A delayed first command crosses another pending command without deleting it.
    confirm_command!(tl, 1000, 2500)
    @test scheduled_samples(tl) == [2000, 2500, 3000]
    @test get_nco_command(tl, 2200) == NCOCommand(12.0Hz, 0.012Hz)
    @test get_nco_command(tl, 2600) == NCOCommand(11.0Hz, 0.011Hz)
    # Early confirmation also preserves the commands it crosses.
    confirm_command!(tl, 3000, 1500)
    @test scheduled_samples(tl) == [1500, 2000, 2500]
    # Feedback for a command not pending under that name is a no-op: none was
    # scheduled for 999, and the one at 1500 was scheduled for 3000 and is confirmed.
    confirm_command!(tl, 999, 2200)
    cancel_command!(tl, 999)
    confirm_command!(tl, 1500, 1600)
    cancel_command!(tl, 1500)
    @test scheduled_samples(tl) == [1500, 2000, 2500]
    cancel_command!(tl, 2000)
    @test scheduled_samples(tl) == [1500, 2500]
    @test get_nco_command(tl, 2200) == NCOCommand(13.0Hz, 0.013Hz)
    advance_timeline!(tl, 2500)
    confirm_command!(tl, 1000, 2600)
    @test get_nco_command(tl, 3000) == NCOCommand(11.0Hz, 0.011Hz)
end

@testset "A command confirmed onto another keeps both identities" begin
    # A scheduled for 100 takes effect at 200, where B is still pending: both stay, A in effect.
    tl = NCOTimeline()
    reset_timeline!(tl, 10.0Hz, 0.01Hz)
    schedule_command!(tl, 100, 1.0Hz, 0.0Hz)
    schedule_command!(tl, 200, 2.0Hz, 0.0Hz)
    confirm_command!(tl, 100, 200)
    @test scheduled_samples(tl) == [200, 200]
    @test get_nco_command(tl, 200) == NCOCommand(1.0Hz, 0.0Hz)
    # B's own acknowledgement still finds B, not A.
    confirm_command!(tl, 200, 300)
    @test scheduled_samples(tl) == [200, 300]
    @test get_nco_command(tl, 250) == NCOCommand(1.0Hz, 0.0Hz)
    @test get_nco_command(tl, 300) == NCOCommand(2.0Hz, 0.0Hz)
    # Two writes taking effect at one sample: the later confirmed is in effect.
    schedule_command!(tl, 400, 4.0Hz, 0.0Hz)
    schedule_command!(tl, 500, 5.0Hz, 0.0Hz)
    confirm_command!(tl, 400, 600)
    confirm_command!(tl, 500, 600)
    @test get_nco_command(tl, 600) == NCOCommand(5.0Hz, 0.0Hz)
    @test calc_mean_nco_command(tl, 500, 700) == NCOCommand(3.5Hz, 0.0Hz)
end

@testset "Physical commands are immutable Hz offsets, destructured by name" begin
    command = NCOCommand(12Hz, 0.25f0Hz)
    @test isbitstype(NCOCommand)
    @test command.carrier_doppler isa typeof(1.0Hz)
    @test command.code_doppler isa typeof(1.0Hz)
    (; carrier_doppler, code_doppler) = command
    @test carrier_doppler == 12.0Hz
    @test code_doppler == 0.25Hz
    # Any frequency unit, stored in Hz; a plain number is not a frequency.
    @test NCOCommand(0.1u"kHz", 0.0Hz).carrier_doppler == 100.0Hz
    @test_throws Unitful.DimensionError NCOCommand(100.0, 0.0)
    @test_throws MethodError iterate(command)
    fixed = ConstantNCOCommand(command)
    @test get_nco_command(fixed, 100) === command
    @test calc_mean_nco_command(fixed, 0, 100) === command
end
