import Base: copy

@inline drop(nt::NamedTuple, key::Symbol) = Base.structdiff(nt, NamedTuple{(key,)})

"""
    copy(g::GNNGraph)

Create a shollow copy of the input graph `g`. This is equivalent to `GNNGraph(g)`.
"""
copy(g::GNNGraph) = GNNGraph(g)

@doc doc"""
    wrapgraph(g::GNNGraph) = () -> copy(g)
    wrapgraph(f::Function) = f

Creater a function wrapper of the input function. 
"""
wrapgraph(g::GNNGraph) = () -> copy(g)
wrapgraph(f::Function) = f

"""
    updategraph(st, g) -> st
Recursively replace the value of `graph` with a shallow copy of `g`.
"""
function updategraph(st::NamedTuple, g::GNNGraph)
    isempty(st) && return st
    for (key, val) in pairs(st)
        if key == :graph
            st = merge(st, (graph = copy(g),))
        elseif val isa NamedTuple
            st = merge(st, (key => updategraph(val, g),))
        end
    end
    return st
end
