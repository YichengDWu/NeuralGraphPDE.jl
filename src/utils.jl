@inline drop(nt::NamedTuple, key::Symbol) = Base.structdiff(nt, NamedTuple{(key,)})
"""
    copg(g::GNNGraph)

Make a shollow copy of the input graph `g`.
"""
Base.copy(g::GNNGraph) = GNNGraph(g.graph, g.num_nodes,g.num_edges,g.num_graphs,g.graph_indicator,g.ndata,g.edata,g.gdata)


"""
    updategraph(st, g) -> st
Replace the value of `st.graph` with a shallow copy of `g`.
"""
updategraph(st::NamedTuple,g::GNNGraph) = merge(st,(graph=copy(g)))
