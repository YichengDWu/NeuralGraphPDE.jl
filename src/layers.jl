@doc raw"""
    AbstractGNNLayer <: AbstractExplicitLayer
An abstract type of graph neural networks. See also [`AbstractGNNContainerLayer`](@ref)
"""
abstract type AbstractGNNLayer <: AbstractExplicitLayer end

@doc raw"""
    AbstractGNNContainerLayer <: AbstractExplicitContainerLayer

This is an abstract type of GNN layers that contains other layers.
"""
abstract type AbstractGNNContainerLayer{layers} <: AbstractExplicitContainerLayer{layers} end

const EMPTYGRAPH = rand_graph(0, 0)

"""
    initialgraph() = EMPTYGRAPH
THe default graph initializer for a GNN layer. Return an empty graph.
"""
initialgraph() = EMPTYGRAPH

initialstates(rng::AbstractRNG, l::AbstractGNNLayer) = (graph = l.initialgraph(),)
statelength(l::AbstractGNNLayer) = 1 #default

function initialstates(rng::AbstractRNG,
                       l::AbstractGNNContainerLayer{layers}) where {layers}
    return merge(NamedTuple{layers}(initialstates.(rng, getfield.((l,), layers))),
                 (graph = l.initialgraph(),))
end

function statelength(l::AbstractGNNContainerLayer{layers}) where {layers}
    return sum(statelength, getfield.((l,), layers)) + 1
end

@doc raw"""
    ExplicitEdgeConv(ϕ; initialgraph = initialgraph, aggr = mean)

Edge convolutional layer.

```math
\mathbf{h}_i' = \square_{j \in N(i)}\, \phi([\mathbf{h}_i, \mathbf{h}_j; \mathbf{x}_j - \mathbf{x}_i])
```

# Arguments

- `ϕ`: A neural network. 
- `initialgraph`: `GNNGraph` or a function that returns a `GNNGraph`
- `aggr`: Aggregation operator for the incoming messages (e.g. `+`, `*`, `max`, `min`, and `mean`).

# Inputs

- `h`: Trainable node embeddings, `NamedTuple` or `Array`.

# Returns

- `NamedTuple` or `Array` that has the same struct with `x` with different a size of channels.

# Parameters

- Parameters of `ϕ`.

# States

- `graph`: `GNNGraph` where `graph.ndata.x` represents the spatial coordinates of nodes. You can also put other nontrainable node features in `graph.ndata` with arbitrary keys. They will be concatenated like `u`.

# Examples
```julia

s = [1, 1, 2, 3]
t = [2, 3, 1, 1]
g = GNNGraph(s, t)

u = randn(4, g.num_nodes)
g = GNNGraph(g, ndata = (; x = rand(3, g.num_nodes)))
nn = Dense(4 + 4 + 3 => 5)
l = ExplicitEdgeConv(nn, initialgraph=g)

rng = Random.default_rng()
ps, st = Lux.setup(rng, l)

```

"""
struct ExplicitEdgeConv{F, M} <:
       AbstractGNNContainerLayer{(:ϕ,)}
    initialgraph::F
    ϕ::M
    aggr::Function
end

function ExplicitEdgeConv(ϕ::AbstractExplicitLayer; initialgraph = initialgraph,
                          aggr = mean)
    ExplicitEdgeConv(wrapgraph(initialgraph), ϕ, aggr)
end

function (l::ExplicitEdgeConv)(x::AbstractArray, ps, st::NamedTuple)
    return l((; preservedname = x), ps, st)
end

function (l::ExplicitEdgeConv)(x::NamedTuple, ps, st::NamedTuple)
    # the spatial coordinate x should be in st
    g = st.graph
    s = g.ndata  #nontrainable node data

    function message(xi, xj, e)
        posi, posj = xi.x, xj.x
        hi, hj = drop(xi, :x), drop(xj, :x)
        m, st_ϕ = l.ϕ(vcat(values(hi)..., values(hj)..., posj .- posi), ps, st.ϕ)
        st = merge(st, (ϕ = st_ϕ,))
        return m
    end
    xs = merge(x, s)
    return propagate(message, g, l.aggr, xi = xs, xj = xs), st
end

@doc raw"""
    ExplicitGCNConv(in_chs::Int, out_chs::Int, activation = identity;
                    initialgraph = initialgraph, init_weight = glorot_normal,
                    init_bias = zeros32)

Same as the one in GraphNeuralNetworks.jl but with exiplicit paramters.

# Arguments
    
- `initialgraph`: `GNNGraph` or a function that returns a `GNNGraph`
    
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
struct ExplicitGCNConv{bias, F1, F2, F3, F4} <: AbstractGNNLayer
    initialgraph::F1
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
    return print(io, ")")
end

function initialparameters(rng::AbstractRNG, d::ExplicitGCNConv{bias}) where {bias}
    if bias
        return (weight = d.init_weight(rng, d.out_chs, d.in_chs),
                bias = d.init_bias(rng, d.out_chs, 1))
    else
        return (weight = d.init_weight(rng, d.out_chs, d.in_chs),)
    end
end

function parameterlength(d::ExplicitGCNConv{bias}) where {bias}
    return bias ? d.out_chs * (d.in_chs + 1) : d.out_chs * d.in_chs
end

function ExplicitGCNConv(in_chs::Int, out_chs::Int, activation = identity;
                         initialgraph = initialgraph, init_weight = glorot_normal,
                         init_bias = zeros32,
                         bias::Bool = true, add_self_loops::Bool = true,
                         use_edge_weight::Bool = false)
    activation = NNlib.fast_act(activation)
    initialgraph = wrapgraph(initialgraph)
    return ExplicitGCNConv{bias, typeof(initialgraph), typeof(activation),
                           typeof(init_weight), typeof(init_bias)
                           }(initialgraph, in_chs, out_chs, activation,
                             init_weight, init_bias,
                             add_self_loops, use_edge_weight)
end

function ExplicitGCNConv(ch::Pair{Int, Int}, activation = identity;
                         initialgraph = initialgraph, init_weight = glorot_uniform,
                         init_bias = zeros32,
                         bias::Bool = true, add_self_loops = true, use_edge_weight = false)
    return ExplicitGCNConv(first(ch), last(ch), activation,
                           initialgraph = initialgraph, init_weight = init_weight,
                           init_bias = init_bias,
                           bias = bias, add_self_loops = add_self_loops,
                           use_edge_weight = use_edge_weight)
end

function (l::ExplicitGCNConv)(x::AbstractMatrix{T}, ps, st::NamedTuple,
                              edge_weight::EW = nothing) where
    {T, EW <: Union{Nothing, AbstractVector}}
    g = st.graph
    @assert !(g isa GNNGraph{<:ADJMAT_T} && edge_weight !== nothing) "Providing external edge_weight is not yet supported for adjacency matrix graphs"

    if edge_weight !== nothing
        @assert length(edge_weight)==g.num_edges "Wrong number of edge weights (expected $(g.num_edges) but given $(length(edge_weight)))"
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
    d = degree(g, T; dir = :in, edge_weight)
    c = 1 ./ sqrt.(d)
    x = x .* c'
    if edge_weight !== nothing
        x = propagate(e_mul_xj, g, +, xj = x, e = edge_weight)
    elseif l.use_edge_weight
        x = propagate(w_mul_xj, g, +, xj = x)
    else
        x = propagate(copy_xj, g, +, xj = x)
    end
    x = x .* c'
    if Dout >= Din
        x = ps.weight * x
    end
    return l.activation.(x .+ ps.bias), st
end

@doc raw"""
    VMHConv(ϕ, γ; initialgraph = initialgraph, aggr = mean)

Convolutional layer from [Learning continuous-time PDEs from sparse data with graph neural networks](https://arxiv.org/abs/2006.08956).
```math
\begin{aligned}
\mathbf{m}_i &= \square_{j \in N(i)}\, \phi(\mathbf{h}_i, \mathbf{h}_j - \mathbf{h}_i; \mathbf{x}_j - \mathbf{x}_i)\\
\mathbf{h}_i' &= \gamma(\mathbf{h}_i ,\mathbf{m}_i)
\end{aligned}
```

# Arguments

- `ϕ`: The neural network for the message function. 
- `γ`: The neural network for the update function.
- `initialgraph`: `GNNGraph` or a function that returns a `GNNGraph`
- `aggr`: Aggregation operator for the incoming messages (e.g. `+`, `*`, `max`, `min`, and `mean`).

# Inputs

- `h`: Trainable node embeddings, `NamedTuple` or `Array`.

# Returns

- `NamedTuple` or `Array` that has the same struct with `x` with different a size of channels.

# Parameters

- Parameters of `ϕ`.
- Parameters of `γ`.

# States

- `graph`: `GNNGraph` where `graph.ndata.x` represents the spatial coordinates of nodes.

# Examples
```julia

s = [1, 1, 2, 3]
t = [2, 3, 1, 1]
g = GNNGraph(s, t)

u = randn(4, g.num_nodes)
g = GNNGraph(g, ndata = (; x = rand(3, g.num_nodes)))
ϕ = Dense(4 + 4 + 3 => 5)
γ = Dense(5 + 4 => 7)
l = VMHConv(ϕ, γ, initialgraph = g)

rng = Random.default_rng()
ps, st = Lux.setup(rng, l)

y, st = l(u, ps, st)
```
                    
"""
struct VMHConv{F, M1, M2, A} <: AbstractGNNContainerLayer{(:ϕ, :γ)}
    initialgraph::F
    ϕ::M1
    γ::M2
    aggr::A
end

function VMHConv(ϕ::AbstractExplicitLayer, γ::AbstractExplicitLayer;
                 initialgraph = initialgraph, aggr = mean)
    initialgraph = wrapgraph(initialgraph)
    VMHConv{typeof(initialgraph), typeof(ϕ), typeof(γ), typeof(aggr)}(initialgraph, ϕ, γ,
                                                                      aggr)
end

function (l::VMHConv)(x::AbstractArray, ps, st::NamedTuple)
    return l((; preservedname = x), ps, st)
end

function (l::VMHConv)(x::NamedTuple, ps, st::NamedTuple)
    function message(xi, xj, e)
        posi, posj = xi.x, xj.x
        hi, hj = values(drop(xi, :x)), values(drop(xj, :x))
        m, st_ϕ = l.ϕ(vcat(hi..., (hj .- hi)..., posj .- posi), ps.ϕ, st.ϕ)
        st = merge(st, (; ϕ = st_ϕ))
        return m
    end

    g = st.graph
    s = g.ndata

    xs = merge(x, s)

    m = propagate(message, g, l.aggr, xi = xs, xj = xs)

    y, st_γ = l.γ(vcat(values(x)..., m), ps.γ, st.γ)
    st = merge(st, (; γ = st_γ))

    return y, st
end

@doc raw"""
    MPPDEConv(ϕ, ψ; initialgraph = initialgraph, aggr = sum, local_features = (:u, :x))

Convolutional layer from [Message Passing Neural PDE Solvers](https://arxiv.org/abs/2202.03376), without the temporal bulking trick.
```math
\begin{aligned}
	\mathbf{m}_i&=\Box _{j\in N(i)}\,\phi (\mathbf{h}_i,\mathbf{h}_j;\mathbf{u}_i-\mathbf{u}_j;\mathbf{x}_i-\mathbf{x}_j;\theta )\\
	\mathbf{h}_i'&=\psi (\mathbf{h}_i,\mathbf{m}_i,\theta )\\
\end{aligned}
```

# Arguments

- `ϕ`: The neural network for the message function. 
- `ψ`: The neural network for the update function.
- `initialgraph`: `GNNGraph` or a function that returns a `GNNGraph`
- `aggr`: Aggregation operator for the incoming messages (e.g. `+`, `*`, `max`, `min`, and `mean`).
- `local_features`: The features that will be differentiated in the message function. 

# Inputs

- `h`: Trainable node embeddings, `Array`.

# Returns

- `NamedTuple` or `Array` that has the same struct with `x` with different a size of channels.

# Parameters

- Parameters of `ϕ`.
- Parameters of `ψ`.

# States

- `graph`: `GNNGraph` where `graph.ndata.x` represents the spatial coordinates of nodes, `graph.ndata.u` represents the initial condition, and `graph.gdata.θ` represents the graph level features of the underlying PDE. `θ` should be a vector. 

# Examples
```julia
g = rand_graph(10, 6)

g = GNNGraph(g, ndata = (; u = rand(2, 10), x = rand(3, 10)), gdata = (; θ = rand(4)))
h = randn(5, 10)
ϕ = Dense(5 + 5 + 2 + 3 + 4 => 5)
ψ = Dense(5 + 5 + 4 => 7)
l = MPPDEConv(ϕ, ψ, initialgraph = g)

rng = Random.default_rng()
ps, st = Lux.setup(rng, l)

y, st = l(h, ps, st)
```

"""
struct MPPDEConv{F, L, M1, M2, A} <: AbstractGNNContainerLayer{(:ϕ, :ψ)}
    initialgraph::F
    local_features::L
    ϕ::M1
    ψ::M2
    aggr::A
end

function MPPDEConv(ϕ::AbstractExplicitLayer, ψ::AbstractExplicitLayer;
                   aggr = +,
                   initialgraph = initialgraph,
                   local_features = (:u, :x))
    initialgraph = wrapgraph(initialgraph)
    MPPDEConv{typeof(initialgraph), typeof(local_features), typeof(ϕ), typeof(ψ),
              typeof(aggr)}(initialgraph, local_features, ϕ, ψ, aggr)
end

function (l::MPPDEConv)(x::AbstractArray, ps, st::NamedTuple)
    num_nodes = st.graph.num_nodes
    num_edges = st.graph.num_edges
    θ = reduce(vcat, values(st.graph.gdata), init = eltype(x)[])

    function message(xi, xj, e)
        di, dj = reduce(vcat, values(xi[l.local_features])),
                 reduce(vcat, values(xj[l.local_features]))
        hi, hj = xi.h, xj.h
        m, st_ϕ = l.ϕ(vcat(hi, hj, di .- dj, repeat(θ, 1, num_edges)), ps.ϕ, st.ϕ)
        st = merge(st, (; ϕ = st_ϕ))
        return m
    end

    g = st.graph
    s = g.ndata

    xs = merge((; h = x), s)
    m = propagate(message, g, l.aggr, xi = xs, xj = xs)

    y, st_ψ = l.ψ(vcat(x, m, repeat(θ, 1, num_nodes)), ps.ψ, st.ψ)
    st = merge(st, (; ψ = st_ψ))

    return y, st
end
