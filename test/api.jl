# The API is what the topic pages' `@docs` blocks list: an export bare, any other name
# qualified, `TrackingLoops.name`, a method by its signature. Returns the bare names and
# the qualified ones.
function listed_names(pages)
    bare, qualified = Symbol[], Symbol[]
    for page in pages
        in_docs = false
        for line in eachline(joinpath(@__DIR__, "..", "docs", "src", page))
            line == "```@docs" && (in_docs = true; continue)
            line == "```" && (in_docs = false; continue)
            in_docs || continue
            m = match(r"^(TrackingLoops\.)?([^(\s]+)", line)
            push!(isnothing(m[1]) ? bare : qualified, Symbol(m[2]))
        end
    end
    unique(bare), unique(qualified)
end

# An enum's values are documented with their type.
documented(name) =
    haskey(Base.Docs.meta(TrackingLoops), Base.Docs.Binding(TrackingLoops, name)) ||
    (value = getfield(TrackingLoops, name); value isa Enum && documented(nameof(typeof(value))))

@testset "API" begin
    pages = filter(!=("internals.md"), readdir(joinpath(@__DIR__, "..", "docs", "src")))
    bare, qualified = listed_names(pages)
    exported = names(TrackingLoops)
    # Every export is listed bare, or is an enum's value listed with its type, and no
    # other name is listed bare.
    @test isempty(setdiff(bare, exported))
    for name in exported
        value = getfield(TrackingLoops, name)
        @test name in bare || (value isa Enum && nameof(typeof(value)) in [bare; qualified])
    end
    for name in bare
        name === :TrackingLoops && continue
        @test documented(name)
    end
    for name in qualified
        @test isdefined(TrackingLoops, name)
        @test name ∉ exported
        @test documented(name)
    end
end
