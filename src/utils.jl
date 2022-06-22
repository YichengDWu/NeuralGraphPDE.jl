import Base: copy

@inline drop(nt::NamedTuple, key::Symbol) = Base.structdiff(nt, NamedTuple{(key,)})

"""
    copy(g::GNNGraph)

Create a shollow copy of the input graph `g`. This is equivalent to `GNNGraph(g)`.
"""
copy(g::GNNGraph) = GNNGraph(g)

"""
    updategraph(st, g) -> st
Replace the value of `st.graph` with a shallow copy of `g`.
"""
updategraph(st::NamedTuple,g::GNNGraph) = merge(st,(graph=copy(g),))
