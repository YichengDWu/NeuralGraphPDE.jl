module NeuralGraphPDE

using Reexport
@reexport using GraphNeuralNetworks.GNNGraphs
@reexport import GraphNeuralNetworks: reduce_nodes,
                                        reduce_edges,
                                        softmax_nodes,
                                        softmax_edges,
                                        broadcast_nodes,
                                        broadcast_edges,
                                        softmax_edge_neighbors,

                                        # msgpass
                                        apply_edges,
                                        aggregate_neighbors,
                                        propagate,
                                        copy_xj,
                                        copy_xi,
                                        xi_dot_xj,
                                        e_mul_xj,
                                        w_mul_xj,

                                        ADJMAT_T


using Lux, NNlib, NNlibCUDA
using Lux: AbstractExplicitContainerLayer, AbstractExplicitLayer, glorot_normal,
           glorot_uniform, ones32, zeros32, AbstractRNG, applyactivation, elementwise_add
using Graphs
using Statistics: mean
using Functors: fmap

import Lux: initialparameters, parameterlength, statelength, Chain, applychain,
            initialstates

include("utils.jl")
include("layers.jl")

export AbstractGNNLayer, AbstractGNNContainerLayer

#layers
export ExplicitEdgeConv, GCNConv, VMHConv, MPPDEConv, GNOConv, SpectralConv

#utils
export updategraph
end
