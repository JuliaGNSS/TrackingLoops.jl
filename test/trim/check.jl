# Builds `app.jl` into a `juliac --trim=safe` executable and checks that it prints
# what the same app prints in a regular Julia session. Needs Julia 1.12+ and
# the JuliaC app (`pkg> app add JuliaC`):
#
#     julia --project=test/trim -e 'using Pkg; Pkg.instantiate()'
#     julia --project=test/trim test/trim/check.jl
#
# The trim verifier rejects any call it cannot resolve statically, so a build failure
# here names the dynamic dispatch that crept into the loop.
#
# The two outputs agree in every count and flag, and in every float to within a
# rounding error the closed loops have amplified: the executable and the session are
# compiled separately, for the system image's CPU targets and the session's own, so
# their floating point may round differently (fused multiply-adds, vector widths). A
# loop that runs differently, not just rounds differently, misses that by far.
juliac = something(
    Sys.which("juliac"),
    joinpath(first(DEPOT_PATH), "bin", Sys.iswindows() ? "juliac.bat" : "juliac"),
)
isfile(juliac) || error("juliac not found; install it with `pkg> app add JuliaC`")

app = joinpath(@__DIR__, "app.jl")
build = mktempdir()
run(`$juliac --output-exe vector_tracking_app --project $(@__DIR__) --bundle $build
    --trim=safe --experimental $app`)

executable = joinpath(build, "bin", Sys.iswindows() ? "vector_tracking_app.exe" : "vector_tracking_app")
trimmed = read(`$executable`, String)
regular = read(`$(Base.julia_cmd()) --project=$(@__DIR__) $app`, String)

# Counts and flags exactly; floats within 5 cm (or m/s) and a part in 10⁹, NaN
# matching NaN.
function same_value(a, b)
    a == b && return true
    (tryparse(Int, a) === nothing && tryparse(Bool, a) === nothing) || return false
    x = tryparse(Float64, a)
    y = tryparse(Float64, b)
    (x === nothing || y === nothing) && return false
    isnan(x) && isnan(y) || isapprox(x, y; atol = 0.05, rtol = 1e-9)
end
same_line(a, b) = length(split(a)) == length(split(b)) && all(same_value.(split(a), split(b)))
trimmed_lines = split(trimmed, '\n')
regular_lines = split(regular, '\n')

print(trimmed)
if length(trimmed_lines) != length(regular_lines) ||
   !all(same_line.(trimmed_lines, regular_lines))
    println("\nThe regular session printed instead:\n", regular)
    error("the trimmed executable's output differs from the regular session's")
end
println("Trimmed executable ($(filesize(executable) ÷ 1024) KiB) matches the regular ",
    "session on all $(count(==('\n'), trimmed)) reported cycles.")
