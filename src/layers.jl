struct ExplicitEdgeConv{M<:AbstractExplicitLayer} <:
        AbstractExplicitContainerLayer{(:ϕ,)}
    ϕ::M
    aggr
end

ExplicitEdgeConv(ϕ; aggr = mean) = ExplicitEdgeConv(ϕ, aggr)

function (l::ExplicitEdgeConv)(g:: GNNGraph,
                               ndata::AbstractArray, edata::AbstractArray,
                               ps::NamedTuple,st::NamedTuple) 
    function message(xi, xj, e, ps, st)
        return l.ϕ(cat(xi, xj, e, dims = 1), ps, st) 
    end    
    return propagate(message, g, l.aggr, ps, st, xi = ndata, xj = ndata, e = edata)
end

function (l::ExplicitEdgeConv)(g:: GNNGraph,
                               ndata::NamedTuple, edata::AbstractArray,
                               ps::NamedTuple,st::NamedTuple) 
    function message(xi,xj, e, ps, st)
        return l.ϕ(cat(values(xi)..., values(xj)..., e, dims = 1), ps, st) 
    end    
    return propagate(message, g, l.aggr, ps, st, xi = ndata, xj = ndata, e = edata)
end
