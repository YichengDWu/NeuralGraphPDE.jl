"""
    ExplicitEdgeConv(ϕ; aggr=max)
```math
\mathbf{x}_i' = \square_{j \in N(i)}\, \phi([\mathbf{x}_i; \mathbf{x}_j - \mathbf{x}_i])
```
# Arguments
- `ϕ`: A neural network. 
- `aggr`: Aggregation operator for the incoming messages (e.g. `+`, `*`, `max`, `min`, and `mean`).

# Input 
- Case1: 
    - `ndata`: NamedTuple `(u=u,...,x=x)` where `u` is the node embedding and `x` is the spatial coordinate.
- Case2: 
    - `ndata`: NamedTuple or Array.
    - `edata`: Array of spatial differences.
"""
struct ExplicitEdgeConv{M<:AbstractExplicitLayer} <:
        AbstractExplicitContainerLayer{(:ϕ,)}
    ϕ::M
    aggr
end

ExplicitEdgeConv(ϕ; aggr = mean) = ExplicitEdgeConv(ϕ, aggr)

function (l::ExplicitEdgeConv)(g:: GNNGraph,
                               ndata::AbstractArray, edata::AbstractArray,
                               ps, st::NamedTuple) 
    function message(xi, xj, e, ps, st)
        return l.ϕ(cat(xi, xj, e, dims = 1), ps, st) 
    end    
    return propagate(message, g, l.aggr, ps, st, xi = ndata, xj = ndata, e = edata)
end

function (l::ExplicitEdgeConv)(g:: GNNGraph,
                               ndata::NamedTuple, edata::AbstractArray,
                               ps, st::NamedTuple) 
    function message(xi,xj, e, ps, st)
        return l.ϕ(cat(values(xi)..., values(xj)..., e, dims = 1), ps, st) 
    end    
    return propagate(message, g, l.aggr, ps, st, xi = ndata, xj = ndata, e = edata)
end

function (l::ExplicitEdgeConv)(g:: GNNGraph,
                               ndata::NamedTuple, 
                               ps, st::NamedTuple) 
    function message(ndatai,ndataj, e, ps, st)
        xi,xj = ndatai.x, ndataj.x
        hi,hj = drop(ndatai, :x), drop(ndataj, :x)
        return l.ϕ(cat(values(hi)..., values(hj)..., xj-xi, dims = 1), ps, st) 
    end    
    return propagate(message, g, l.aggr, ps, st, xi = ndata, xj = ndata)
end


