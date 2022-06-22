import Base: copy

@inline drop(nt::NamedTuple, key::Symbol) = Base.structdiff(nt, NamedTuple{(key,)})

"""
    copy(g::GNNGraph)

Create a shollow copy of the input graph `g`. This is equivalent to `GNNGraph(g)`.
"""
copy(g::GNNGraph) = GNNGraph(g)

"""
    updategraph(st, g) -> st
Recursively replace the value of `graph` with a shallow copy of `g`.
"""
function updategraph(st::NamedTuple, g::GNNGraph)
    st == (;) || return

    for (key, value) in pairs(st)
        if key == :graph
            st = merge(st, (key=copy(g),))
        elseif val isa NamedTuple
            st = merge(st, (key=updategraph(value, g),))
        end
    end
    return st
end
