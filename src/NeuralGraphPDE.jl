module NeuralGraphPDE

using Lux, GraphNeuralNetworks, NNlib
using Lux: AbstractExplicitContainerLayer, AbstractExplicitLayer, glorot_normal, glorot_uniform,
           ones32, zeros32

import GraphNeuralNetworks: propagate, apply_edges, ADJMAT_T
import Lux: initialparameters
using Random
using Statistics: mean

include("utils.jl")
include("msgpass.jl")
include("layers.jl")

export ExplicitEdgeConv, ExplicitGCNConv
end