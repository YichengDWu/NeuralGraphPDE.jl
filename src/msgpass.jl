"""
    apply_edges(f, g, xi, xj, e, ps, st)
    apply_edges(f, g, ps, st; [xi, xj, e])

# Arguments

- `ps`: Parameters of the neural network inside the function `f`
- `st`: State of the neural network inside the function `f`
"""
apply_edges(f, g::GNNGraph, ps::NamedTuple, st::NamedTuple; xi=nothing, xj=nothing, e=nothing) = 
    apply_edges(f, g, xi, xj, e, ps, st)

function apply_edges(f, g::GNNGraph, xi, xj, e, ps, st)
    check_num_nodes(g, xi)
    check_num_nodes(g, xj)
    check_num_edges(g, e)
    s, t = edge_index(g)
    xi = GNNGraphs._gather(xi, t)   # size: (D, num_nodes) -> (D, num_edges)
    xj = GNNGraphs._gather(xj, s)
    m, st = f(xi, xj, e, ps, st)
    return m, st
end

"""
    propagate(f, g, aggr, ps, st; xi, xj, e)  ->  m̄
"""
propagate(f, g::GNNGraph, aggr, ps::NamedTuple, st::NamedTuple; xi=nothing, xj=nothing, e=nothing) = 
    propagate(f, g, aggr, xi, xj, e, ps, st)

function propagate(f, g::GNNGraph, aggr, xi, xj, e, ps, st)
    m, st = apply_edges(f, g, xi, xj, e, ps, st) 
    m̄ = aggregate_neighbors(g, aggr, m)
    return m̄, st
end