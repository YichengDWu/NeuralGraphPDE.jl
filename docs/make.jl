using NeuralGraphPDE
using Documenter

DocMeta.setdocmeta!(NeuralGraphPDE, :DocTestSetup, :(using NeuralGraphPDE); recursive=true)

makedocs(;
    modules=[NeuralGraphPDE],
    authors="MilkshakeForReal <yicheng.wu@ucalgary.ca> and contributors",
    repo="https://github.com/MilkshakeForReal/NeuralGraphPDE.jl/blob/{commit}{path}#{line}",
    sitename="NeuralGraphPDE.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://MilkshakeForReal.github.io/NeuralGraphPDE.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API Reference" => 
            [
                "Message Passing" => "api/messagepassing.md",
            ]
    ],
)

deploydocs(;
    repo="github.com/MilkshakeForReal/NeuralGraphPDE.jl",
    devbranch="main",
)
