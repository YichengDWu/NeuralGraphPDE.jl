struct GNNLayer <:
        AbstractExplicitContainerLayer{(:ϕ,:ψ)}
    ϕ::AbstractExplicitLayer
    ψ::AbstractExplicitLayer
end

function (gnn::GNNLayer)(g:: NTuple{2,AbstractVector{T}},
                         u::AbstractArray, x::AbstractArray,
                         ps::NamedTuple,st::NamedTuple) where {T<:Integer}
    function message(u,x,ps,st)
        return gnn.ϕ(cat(u,x),ps,st) # from j to i
    end    
end
