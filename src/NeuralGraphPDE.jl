module NeuralGraphPDE

using Lux, GraphNeuralNetworks
using Lux: AbstractExplicitContainerLayer, AbstractExplicitLayer
import GraphNeuralNetworks: propagate, apply_edges
using Statistics: mean

include("utils.jl")
include("msgpass.jl")
include("layers.jl")

end
