using Documenter, TrackingLoops, GNSSSignals

# The API is what the topic pages' `@docs` blocks list: an export bare, any other name
# qualified, `TrackingLoops.name`, a method by its signature. Returns the bare names and
# the qualified ones.
function listed_names(pages)
    bare, qualified = Symbol[], Symbol[]
    for page in pages
        in_docs = false
        for line in eachline(joinpath(@__DIR__, "src", page))
            line == "```@docs" && (in_docs = true; continue)
            line == "```" && (in_docs = false; continue)
            in_docs || continue
            m = match(r"^(TrackingLoops\.)?([^(\s]+)", line)
            push!(isnothing(m[1]) ? bare : qualified, Symbol(m[2]))
        end
    end
    unique(bare), unique(qualified)
end

const PAGES = [
    "index.md",
    "correlators.md",
    "doppler_estimators.md",
    "nco_timeline.md",
    "bit_sync.md",
    "cn0_estimators.md",
    "noise_estimators.md",
    "vector_tracking.md",
]

# The internals page documents what the topic pages do not list.
const LISTED = [getfield(TrackingLoops, name) for name in vcat(listed_names(PAGES)...)]
listed(x) = any(y -> y === x, LISTED)

DocMeta.setdocmeta!(
    TrackingLoops,
    :DocTestSetup,
    :(using TrackingLoops, GNSSSignals; using Unitful: Hz);
    recursive = true,
)

makedocs(
    sitename = "TrackingLoops.jl",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    modules = [TrackingLoops],
    doctest = true,
    # Every export must appear in the manual, and every `@ref` must resolve: both are
    # build errors, not warnings.
    checkdocs = :exports,
    pages = [PAGES; "internals.md"],
)

deploydocs(repo = "github.com/JuliaGNSS/TrackingLoops.jl.git", push_preview = true)
