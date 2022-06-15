module NeuralGraphPDE

using Lux, GraphNeuralNetworks
using Lux: AbstractExplicitContainerLayer, AbstractExplicitLayer
import GraphNeuralNetworks: propagate, apply_edges

include("layers.jl")
include("msgpass.jl")

end
