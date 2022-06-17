"""
    WithStaticGraph(model,g)

A wrapper for `model`, assuming the graph is static and nontrainable.

# Arguments

- `model`: A function that takes a graph.
- `g`: A graph.

# Examples
```
s = [1,1,2,3]
t = [2,3,1,1]
g = GNNGraph(s, t)
model = ExplicitGCNConv(3 => 5)
wg = WithStaticGraph(model, g)
```
"""
struct WithStaticGraph{M<:AbstractExplicitLayer,G<:GNNGraph} <:
        AbstractExplicitContainerLayer{(:model,)}
    model: M
    g: G
end

(w::WithStaticGraph)(g::GNNGraph, x...;kws...) = w.model(g, x...;kws...)
(w::WithStaticGraph)(x...;kws...) = w.model(w.g, x...;kws...)
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
l = ExplicitGCNConv(3 => 5) 

# setup layer
rng = Random.default_rng()
Random.seed!(rng, 0)

ps, st = Lux.setup(rng, l)

# forward pass
y = l(g, x, ps, st)       # size:  5 × num_nodes

# convolution with edge weights
w = [1.1, 0.1, 2.3, 0.5]
y = l(g, x, ps, st, w)

# Edge weights can also be embedded in the graph.
g = GNNGraph(s, t, w)
l = ExplicitGCNConv(3 => 5, use_edge_weight=true) 
y = l(g, x, ps, st) # same as l(g, x, ps, st, w) 
```
"""
struct ExplicitGCNConv{bias,F1,F2,F3} <: AbstractExplicitLayer
    in_chs::Int
    out_chs::Int
    activation::F1
    init_weight::F2
    init_bias::F3
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

statelength(d::ExplicitGCNConv) = 0

function ExplicitGCNConv(in_chs::Int, out_chs::Int, activation = identity;
                         init_weight=glorot_normal, init_bias=zeros32,
                         bias::Bool=true, add_self_loops::Bool=true, use_edge_weight::Bool=false) 
    activation = NNlib.fast_act(activation)
    return ExplicitGCNConv{bias, typeof(activation), typeof(init_weight), typeof(init_bias)}(in_chs, out_chs, activation, 
                                                                                             init_weight, init_bias, 
                                                                                             add_self_loops, use_edge_weight)
end

function ExplicitGCNConv(ch::Pair{Int,Int}, activation=identity;
                         init_weight=glorot_uniform, init_bias = zeros32,
                         bias::Bool=true, add_self_loops=true, use_edge_weight=false)
    return ExplicitGCNConv(first(ch), last(ch), activation, 
                           init_weight = init_weight, init_bias = init_bias,
                           bias = bias, add_self_loops = add_self_loops, use_edge_weight=use_edge_weight)
end

function (l::ExplicitGCNConv)(g::GNNGraph, x::AbstractMatrix{T}, ps, st:: NamedTuple, edge_weight::EW=nothing) where 
    {T, EW<:Union{Nothing,AbstractVector}}
    
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

function (l::ExplicitGCNConv)(g::GNNGraph{<:ADJMAT_T}, x::AbstractMatrix, ps, st::NamedTuple, edge_weight::AbstractVector)
    g = GNNGraph(edge_index(g)...; g.num_nodes)  # convert to COO
    return l(g, x, ps, st, edge_weight)
end