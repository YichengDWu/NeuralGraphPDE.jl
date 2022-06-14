using Lux
using Lux: AbstractExplicitContainerLayer, AbstractExplicitLayer
using NNlib: gather, scatter
import NNlibCUDA: NNlib.gather!, NNlib.scatter!

struct GNNLayer <:
        AbstractExplicitContainerLayer{(:ϕ,:ψ)}
    ϕ::AbstractExplicitLayer
    ψ::AbstractExplicitLayer
end

function (gnn::GNNLayer)(edge_index:: NTuple{2,AbstractVector{T}},
                         u::AbstractArray, x::AbstractArray,
                         ps::NamedTuple,st::NamedTuple) where {T<:Integer}
    s, t = edge_index
    n = length(s)

    uj, ui = gather(u, s), gather(u, t)
    xj, xi = gather(x, s), gather(x, t)

    m = gnn.ϕ(cat(uj, xj-xi; dims = 1), ps.ϕ, st.ϕ)
    
    dstsize =  (size(m)[1:end-1]..., n)
    m = scatter(mean, src, idx; dstsize = dstsize)

    function message(u,x)
        return gnn.ϕ(cat(u,x)) # from j to i
    end    
end
