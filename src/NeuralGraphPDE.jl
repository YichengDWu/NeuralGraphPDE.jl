module NeuralGraphPDE

using Lux, GraphNeuralNetworks, NNlib, NNlibCUDA
using Lux: AbstractExplicitContainerLayer, AbstractExplicitLayer, glorot_normal,
           glorot_uniform, ones32, zeros32, AbstractRNG, applyactivation, elementwise_add
using GraphNeuralNetworks: ADJMAT_T
using Statistics: mean

import GraphNeuralNetworks: propagate, apply_edges
import Lux: initialparameters, parameterlength, statelength, Chain, applychain,
            initialstates
include("utils.jl")
include("layers.jl")

export AbstractGNNLayer, AbstractGNNContainerLayer
export ExplicitEdgeConv, ExplicitGCNConv, VMHConv, MPPDEConv, GNOConv
export updategraph

end
