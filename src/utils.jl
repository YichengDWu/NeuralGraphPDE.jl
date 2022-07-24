import Base: copy

@inline drop(nt::NamedTuple, key::Symbol) = Base.structdiff(nt, NamedTuple{(key,)})

"""
    copy(g::GNNGraph, kwarg...)

Create a shollow copy of the input graph `g`. This is equivalent to `GNNGraph(g)`.
"""
copy(g::GNNGraph, kwarg...) = GNNGraph(g, kwarg...)

@doc raw"""
    wrapgraph(g::GNNGraph) = () -> copy(g)
    wrapgraph(f::Function) = f

Creater a function wrapper of the input graph.
"""
wrapgraph(g::GNNGraph) = () -> copy(g)
wrapgraph(f::Function) = f

@doc raw"""
    updategraph(st, g) -> st
Recursively replace the value of `graph` with a shallow copy of `g`.
"""
function updategraph(st::NamedTuple, g::GNNGraph)
    isempty(st) && return st
    st = fmap(Base.Fix2(_updategraph, copy(g)), st; exclude=x -> x isa GNNGraph)
    return st
end

_updategraph(og, ng::GNNGraph) = og isa GNNGraph ? ng : og
