module NeuralGraphPDE

using Lux, GraphNeuralNetworks, NNlib
using Lux: AbstractExplicitContainerLayer, AbstractExplicitLayer, glorot_normal, glorot_uniform,
           ones32, zeros32, AbstractRNG
using GraphNeuralNetworks: ADJMAT_T
using Statistics: mean

import GraphNeuralNetworks: propagate, apply_edges
import Lux: initialparameters, parameterlength, statelength, Chain, applychain
include("utils.jl")
include("msgpass.jl")
include("layers.jl")

export ExplicitEdgeConv, ExplicitGCNConv, WithStaticGraph
end