abstract type AbstractGNNLayer <: AbstractExplicitLayer end
"""
    AbstractGNNContainerLayer{layers} <: AbstractExplicitContainerLayer{layers}

This is an abstract type of GNN layers that contains other layers.
"""
abstract type AbstractGNNContainerLayer{layers} <: AbstractExplicitContainerLayer{layers} end

function initialgraph end

initialstates(rng::AbstractRNG, l::AbstractGNNLayer) = (graph= l.initialgraph(),)
statelength(l::AbstractGNNLayer) = 1 #default

function initialstates(rng::AbstractRNG,
    l::AbstractGNNContainerLayer{layers}) where {layers}
    return merge(NamedTuple{layers}(initialstates.(rng, getfield.((l,), layers))), (graph = l.initialgraph(),))
end

function statelength(l::AbstractGNNContainerLayer{layers}) where {layers}
    return sum(statelength, getfield.((l,), layers))+1
end

wrapgraph(g::GNNGraph) = () -> copy(g)
wrapgraph(f::Function) = f

"""
    ExplicitEdgeConv(ϕ; aggr=max)
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
struct ExplicitEdgeConv{F,M<:AbstractExplicitLayer} <:
        AbstractGNNContainerLayer{(:ϕ,)}
    initialgraph::F
    ϕ::M
    aggr
end

ExplicitEdgeConv(ϕ; initialgraph=initialgraph, aggr = mean) = ExplicitEdgeConv(wrapgraph(initialgraph), ϕ, aggr)

function (l::ExplicitEdgeConv)(ndata::AbstractArray, edata::AbstractArray,
                               ps, st::NamedTuple) 
    g = st.graph
    function message(xi, xj, e, ps, st)
        return l.ϕ(cat(xi, xj, e, dims = 1), ps, st) 
    end    
    return propagate(message, g, l.aggr, ps, st.ϕ, xi = ndata, xj = ndata, e = edata)
end

function (l::ExplicitEdgeConv)((ndata, edata)::NTuple{2,NamedTuple},
                               ps, st::NamedTuple) 
    g = st.graph
    function message(xi,xj, e, ps, st)
        return l.ϕ(cat(values(xi)..., values(xj)..., e, dims = 1), ps, st) 
    end    
    return propagate(message, g, l.aggr, ps, st.ϕ, xi = ndata, xj = ndata, e = edata)
end

function (l::ExplicitEdgeConv)(ndata::NamedTuple, 
                               ps, st::NamedTuple) 
    g = st.graph
    function message(ndatai,ndataj, e, ps, st)
        xi, xj = ndatai.x, ndataj.x
        hi, hj = drop(ndatai, :x), drop(ndataj, :x)
        return l.ϕ(cat(values(hi)..., values(hj)..., xj-xi, dims = 1), ps, st) 
    end    
    return propagate(message, g, l.aggr, ps, st.ϕ, xi = ndata, xj = ndata)
end


"""
    ExplicitGCNConv()

Same as the one in GraphNeuralNetworks.jl but with exiplicit paramters

# Arguments
    
- `in_chs`: 
- `out_chs`:
- `activation`:
- `add_self_loops`: 
- `use_edge_weight`:
    
# Examples

```julia
# create data
s = [1,1,2,3]
t = [2,3,1,1]
g = GNNGraph(s, t)
x = randn(3, g.num_nodes)

# create layer
l = ExplicitGCNConv(3 => 5, initialgraph = g) 

# setup layer
rng = Random.default_rng()
Random.seed!(rng, 0)

ps, st = Lux.setup(rng, l)

# forward pass
y = l(x, ps, st)       # size:  5 × num_nodes
```
"""
struct ExplicitGCNConv{bias,F1,F2,F3,F4} <: AbstractGNNLayer
    initialgraph:: F1
    in_chs::Int
    out_chs::Int
    activation::F2
    init_weight::F3
    init_bias::F4
    add_self_loops::Bool
    use_edge_weight::Bool
end

function Base.show(io::IO, l::ExplicitGCNConv)
    print(io, "ExplicitGCNConv($(l.in_chs) => $(l.out_chs)")
    (l.activation == identity) || print(io, ", ", l.activation)
    print(io, ")")
end

function initialparameters(rng::AbstractRNG, d::ExplicitGCNConv{bias}) where {bias}
    if bias
        return (weight=d.init_weight(rng, d.out_chs, d.in_chs),
                bias=d.init_bias(rng, d.out_chs, 1))
    else
        return (weight=d.init_weight(rng, d.out_chs, d.in_chs),)
    end
end

function parameterlength(d::ExplicitGCNConv{bias}) where {bias}
    return bias ? d.out_chs * (d.in_chs + 1) : d.out_chs * d.in_chs
end

function ExplicitGCNConv(in_chs::Int, out_chs::Int, activation = identity;
                         initialgraph = initialgraph, init_weight=glorot_normal, init_bias=zeros32,
                         bias::Bool=true, add_self_loops::Bool=true, use_edge_weight::Bool=false) 
    activation = NNlib.fast_act(activation)
    initialgraph = wrapgraph(initialgraph)
    return ExplicitGCNConv{bias, typeof(initialgraph), typeof(activation), typeof(init_weight), typeof(init_bias)}(initialgraph, in_chs, out_chs, activation, 
                                                                                                                   init_weight, init_bias, 
                                                                                                                   add_self_loops, use_edge_weight)
end

function ExplicitGCNConv(ch::Pair{Int,Int}, activation=identity;
                         initialgraph = initialgraph, init_weight=glorot_uniform, init_bias = zeros32,
                         bias::Bool=true, add_self_loops=true, use_edge_weight=false)
    return ExplicitGCNConv(first(ch), last(ch), activation, 
                           initialgraph = initialgraph, init_weight = init_weight, init_bias = init_bias,
                           bias = bias, add_self_loops = add_self_loops, use_edge_weight=use_edge_weight)
end

function (l::ExplicitGCNConv)(x::AbstractMatrix{T}, ps, st:: NamedTuple, edge_weight::EW=nothing) where 
    {T, EW<:Union{Nothing,AbstractVector}}
    g = st.graph
    @assert !(g isa GNNGraph{<:ADJMAT_T} && edge_weight !== nothing) "Providing external edge_weight is not yet supported for adjacency matrix graphs"

    if edge_weight !== nothing
        @assert length(edge_weight) == g.num_edges "Wrong number of edge weights (expected $(g.num_edges) but given $(length(edge_weight)))" 
    end

    if l.add_self_loops
        g = add_self_loops(g)
        if edge_weight !== nothing
            # Pad weights with ones
            # TODO for ADJMAT_T the new edges are not generally at the end
            edge_weight = [edge_weight; fill!(similar(edge_weight, g.num_nodes), 1)]
            @assert length(edge_weight) == g.num_edges
        end
    end
    Dout, Din = l.out_chs, l.in_chs 
    if Dout < Din
        # multiply before convolution if it is more convenient, otherwise multiply after
        x = ps.weight * x
    end
    d = degree(g, T; dir=:in, edge_weight)
    c = 1 ./ sqrt.(d)
    x = x .* c'
    if edge_weight !== nothing
        x = propagate(e_mul_xj, g, +, xj=x, e=edge_weight)
    elseif l.use_edge_weight        
        x = propagate(w_mul_xj, g, +, xj=x)
    else
        x = propagate(copy_xj, g, +, xj=x)
    end
    x = x .* c'
    if Dout >= Din
        x = ps.weight * x
    end
    return l.activation.(x .+ ps.bias), st
end