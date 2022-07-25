@inline drop(nt::NamedTuple, key::Symbol) = Base.structdiff(nt, NamedTuple{(key,)})

"""
    copy(g::GNNGraph; kwargs...)

Create a shollow copy of the input graph `g`. This is equivalent to `GNNGraph(g)`.
"""
Base.copy(g::GNNGraph; kwargs...) = GNNGraph(g; kwargs...)

@doc raw"""
    wrapgraph(g::GNNGraph) = () -> copy(g)
    wrapgraph(f::Function) = f

Creater a function wrapper of the input graph.
"""
wrapgraph(g::GNNGraph) = () -> copy(g)
wrapgraph(f::Function) = f

@doc raw"""
    updategraph(st, g; kwargs...) -> st
Recursively replace the value of `graph` with a shallow copy of `g`. If `g` is nothing, then
only update the old graph with the data given in `kwargs`.
"""
function updategraph(st::NamedTuple, g=nothing; kwargs...)
    isempty(st) && return st
    st = fmap(og -> _updategraph(og, g; kwargs...), st; exclude=x -> x isa GNNGraph)
    return st
end

_updategraph(og, ng::GNNGraph; kwargs...) = og isa GNNGraph ? copy(ng) : og
_updategraph(og, ng::Nothing; kwargs...) = og isa GNNGraph ? copy(og; kwargs...) : og
