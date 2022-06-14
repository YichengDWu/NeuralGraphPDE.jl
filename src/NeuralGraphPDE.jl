module NeuralGraphPDE

using GraphNeuralNetworks, Lux

include("gnn.jl")
include("msgpass.jl")

export apply_edges, propagate
end
