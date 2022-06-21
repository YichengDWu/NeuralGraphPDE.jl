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
    
    doctest = false,
    strict = [
        :doctest,
        :linkcheck,
        :parse_error,
        :example_block,
        # Other available options are
        # :autodocs_block, :cross_references, :docs_block, :eval_block, :example_block, :footnote, :meta_block, :missing_docs, :setup_block
    ],

    pages=[
        "Home" => "index.md",
        "Tutorials" => 
                [
                    "Neural Graph Ordinary Differential Equations" => "tutorials/graph_node.md",
                ],
        "API Reference" => 
            [
                "Layers" => "api/layers.md",
                "Message Passing" => "api/messagepassing.md",
                "Utilities" => "api/utilities.md"
            ],
    ],
)

deploydocs(;
    repo="github.com/MilkshakeForReal/NeuralGraphPDE.jl",
    devbranch="main",
)
