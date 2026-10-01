@testset "Maximum and default number of code blocks to integrate" begin
    # One data bit for data-bearing signals ...
    @test @inferred(max_num_code_blocks_to_integrate(GPSL1CA())) == 20
    @test @inferred(max_num_code_blocks_to_integrate(GPSL5I())) == 10
    # ... one secondary-code period for pilots (`data_frequency == 0`).
    @test @inferred(max_num_code_blocks_to_integrate(GPSL1C_P())) == 1800
    @test @inferred(max_num_code_blocks_to_integrate(GPSL5Q())) == 20

    @test @inferred(default_num_code_blocks_to_integrate(GPSL1CA())) == 1
    @test @inferred(default_num_code_blocks_to_integrate(GPSL1C_P())) == 1
end

@testset "Calculate number of code blocks to integrate" begin
    gpsl1 = GPSL1CA()

    @test @inferred(calc_num_code_blocks_to_integrate(gpsl1, 1, false)) == 1
    @test @inferred(calc_num_code_blocks_to_integrate(gpsl1, 2, false)) == 1
    @test @inferred(calc_num_code_blocks_to_integrate(gpsl1, 2, true)) == 2
    @test @inferred(calc_num_code_blocks_to_integrate(gpsl1, 20, true)) == 20
    @test @inferred(calc_num_code_blocks_to_integrate(gpsl1, 21, true)) == 20
    @test @inferred(calc_num_code_blocks_to_integrate(gpsl1, 0, true)) == 1

    # A preferred value that doesn't divide the 20 blocks per bit is clamped to
    # the largest divisor below it, so an integration never straddles a bit.
    @test @inferred(calc_num_code_blocks_to_integrate(gpsl1, 3, true)) == 2
    @test @inferred(calc_num_code_blocks_to_integrate(gpsl1, 7, true)) == 5
    @test @inferred(calc_num_code_blocks_to_integrate(gpsl1, 19, true)) == 10

    # Pilots are capped by their secondary-code period once it has been found.
    gpsl1c_p = GPSL1C_P()
    @test @inferred(calc_num_code_blocks_to_integrate(gpsl1c_p, 10, false)) == 1
    @test @inferred(calc_num_code_blocks_to_integrate(gpsl1c_p, 10, true)) == 10
    @test @inferred(calc_num_code_blocks_to_integrate(gpsl1c_p, 1800, true)) == 1800
    @test @inferred(calc_num_code_blocks_to_integrate(gpsl1c_p, 2000, true)) == 1800
    # Data-bearing signals stay capped by the bit period.
    @test @inferred(calc_num_code_blocks_to_integrate(GPSL5I(), 11, true)) == 10
end

@testset "Calculate number of code blocks for the bit buffer" begin
    gpsl1 = GPSL1CA()
    fs = 4e6Hz
    # Before sync the detectors shift exactly one prompt per call.
    @test @inferred(calc_num_code_blocks_for_bit_buffer(gpsl1, 20 * 4000, fs, false)) == 1
    # Afterwards the whole blocks the record covered, from its sample count.
    @test @inferred(calc_num_code_blocks_for_bit_buffer(gpsl1, 4000, fs, true)) == 1
    @test @inferred(calc_num_code_blocks_for_bit_buffer(gpsl1, 20 * 4000, fs, true)) == 20
    # A record a sample short or long of a whole block still rounds to it.
    @test calc_num_code_blocks_for_bit_buffer(gpsl1, 5 * 4000 - 1, fs, true) == 5
    @test calc_num_code_blocks_for_bit_buffer(gpsl1, 5 * 4000 + 1, fs, true) == 5
    # GPS L5I: 10 230 chips at 10.23 MHz, i.e. 1 ms blocks.
    @test calc_num_code_blocks_for_bit_buffer(GPSL5I(), 10 * 25_000, 25e6Hz, true) == 10
end
