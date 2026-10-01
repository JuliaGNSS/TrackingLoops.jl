using Documenter, TrackingLoops, GNSSSignals

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
    # Every exported symbol must appear in the manual, and every `@ref` must
    # resolve: both are build errors, not warnings.
    checkdocs = :exports,
    pages = [
        "index.md",
        "correlators.md",
        "doppler_estimators.md",
        "nco_timeline.md",
        "bit_sync.md",
        "cn0_estimators.md",
        "noise_estimators.md",
        "internals.md",
    ],
)

deploydocs(repo = "github.com/JuliaGNSS/TrackingLoops.jl.git", push_preview = true)
