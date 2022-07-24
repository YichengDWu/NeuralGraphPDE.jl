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

initialstates(rng::AbstractRNG, l::AbstractGNNLayer) = (graph=l.initialgraph(),)
statelength(l::AbstractGNNLayer) = 1 #default

function initialstates(rng::AbstractRNG,
                       l::AbstractGNNContainerLayer{layers}) where {layers}
    return merge(NamedTuple{layers}(initialstates.(rng, getfield.((l,), layers))),
                 (graph=l.initialgraph(),))
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
struct ExplicitEdgeConv{M} <: AbstractGNNContainerLayer{(:ϕ,)}
    initialgraph::Function
    ϕ::M
    aggr::Function
end

function ExplicitEdgeConv(ϕ::AbstractExplicitLayer; initialgraph=initialgraph, aggr=mean)
    return ExplicitEdgeConv{typeof(ϕ)}(wrapgraph(initialgraph), ϕ, aggr)
end

function (l::ExplicitEdgeConv)(x::AbstractArray, ps, st::NamedTuple)
    return l((; preservedname=x), ps, st)
end

function (l::ExplicitEdgeConv)(x::NamedTuple, ps, st::NamedTuple)
    # the spatial coordinate x should be in st
    g = st.graph
    s = g.ndata  #nontrainable node data

    function message(xi, xj, e)
        posi, posj = xi.x, xj.x
        hi, hj = drop(xi, :x), drop(xj, :x)
        m, st_ϕ = l.ϕ(vcat(values(hi)..., values(hj)..., posj .- posi), ps, st.ϕ)
        st = merge(st, (ϕ=st_ϕ,))
        return m
    end
    xs = merge(x, s)
    return propagate(message, g, l.aggr; xi=xs, xj=xs), st
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
struct ExplicitGCNConv{bias, F1, F2, F3} <: AbstractGNNLayer
    initialgraph::Function
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
    return print(io, ")")
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

function ExplicitGCNConv(in_chs::Int, out_chs::Int, activation=identity;
                         initialgraph=initialgraph, init_weight=glorot_normal,
                         init_bias=zeros32, bias::Bool=true, add_self_loops::Bool=true,
                         use_edge_weight::Bool=false)
    activation = NNlib.fast_act(activation)
    initialgraph = wrapgraph(initialgraph)
    return ExplicitGCNConv{bias, typeof(activation), typeof(init_weight), typeof(init_bias)
                           }(initialgraph, in_chs, out_chs, activation, init_weight,
                             init_bias, add_self_loops, use_edge_weight)
end

function ExplicitGCNConv(ch::Pair{Int, Int}, activation=identity; initialgraph=initialgraph,
                         init_weight=glorot_uniform, init_bias=zeros32, bias::Bool=true,
                         add_self_loops=true, use_edge_weight=false)
    return ExplicitGCNConv(first(ch), last(ch), activation; initialgraph=initialgraph,
                           init_weight=init_weight, init_bias=init_bias, bias=bias,
                           add_self_loops=add_self_loops, use_edge_weight=use_edge_weight)
end

function (l::ExplicitGCNConv)(x::AbstractMatrix{T}, ps, st::NamedTuple,
                              edge_weight::EW=nothing) where {T,
                                                              EW <:
                                                              Union{Nothing, AbstractVector
                                                                    }}
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
    d = degree(g, T; dir=:in, edge_weight)
    c = 1 ./ sqrt.(d)
    x = x .* c'
    if edge_weight !== nothing
        x = propagate(e_mul_xj, g, +; xj=x, e=edge_weight)
    elseif l.use_edge_weight
        x = propagate(w_mul_xj, g, +; xj=x)
    else
        x = propagate(copy_xj, g, +; xj=x)
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
struct VMHConv{M1, M2, A} <: AbstractGNNContainerLayer{(:ϕ, :γ)}
    initialgraph::Function
    ϕ::M1
    γ::M2
    aggr::A
end

function VMHConv(ϕ::AbstractExplicitLayer, γ::AbstractExplicitLayer;
                 initialgraph=initialgraph, aggr=mean)
    initialgraph = wrapgraph(initialgraph)
    return VMHConv{typeof(ϕ), typeof(γ), typeof(aggr)}(initialgraph, ϕ, γ, aggr)
end

function (l::VMHConv)(x::AbstractArray, ps, st::NamedTuple)
    return l((; preservedname=x), ps, st)
end

function (l::VMHConv)(x::NamedTuple, ps, st::NamedTuple)
    function message(xi, xj, e)
        posi, posj = xi.x, xj.x
        hi, hj = values(drop(xi, :x)), values(drop(xj, :x))
        m, st_ϕ = l.ϕ(vcat(hi..., (hj .- hi)..., posj .- posi), ps.ϕ, st.ϕ)
        st = merge(st, (; ϕ=st_ϕ))
        return m
    end

    g = st.graph
    s = g.ndata

    xs = merge(x, s)

    m = propagate(message, g, l.aggr; xi=xs, xj=xs)

    y, st_γ = l.γ(vcat(values(x)..., m), ps.γ, st.γ)
    st = merge(st, (; γ=st_γ))

    return y, st
end

@doc raw"""
    MPPDEConv(ϕ, ψ; initialgraph = initialgraph, aggr = mean, local_features = (:u, :x))
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

# Inputs
- `h`: Trainable node embeddings, `Array`.
# Returns
- `NamedTuple` or `Array` that has the same struct with `x` with different a size of channels.
# Parameters
- Parameters of `ϕ`.
- Parameters of `ψ`.
																
# States
- `graph`: `GNNGraph` for which `graph.gdata` represents the graph level features of the underlying PDE. All features in `graph.gdata`
    should be a matrices of the size `(num_feats, num_graphs)`. You can store `u`(`x`) in `graph.ndata` or `u_j-u_i`(`x_jx_i`) in `graph.edata`.
    If `g` is a batched graph, then currently all graphs need to have the same structure. Note that `t` is included in `graph.gdata`
    in the original paper.

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
struct MPPDEConv{A, M1, M2} <: AbstractGNNContainerLayer{(:ϕ, :ψ)}
    initialgraph::Function
    ϕ::M1
    ψ::M2
    aggr::A
end

function MPPDEConv(ϕ::AbstractExplicitLayer, ψ::AbstractExplicitLayer; aggr=mean,
                   initialgraph=initialgraph)
    initialgraph = wrapgraph(initialgraph)
    return MPPDEConv{typeof(aggr), typeof(ϕ), typeof(ψ)}(initialgraph, ϕ, ψ, aggr)
end

function (l::MPPDEConv)(x::AbstractArray, ps, st::NamedTuple)
    g = st.graph
    num_nodes = g.num_nodes
    num_edges = g.num_edges
    num_graphs = g.num_graphs
    s = g.ndata
    e = g.edata
    θ = reduce(vcat, values(st.graph.gdata); init=similar(x, 0, num_graphs))

    nkeys = keys(s)
    initarray = similar(x, 0, num_edges)

    function message(xi, xj, e)
        di, dj = xi[nkeys], xj[nkeys]
        di, dj = reduce(vcat, values(di); init=initarray),
                 reduce(vcat, values(dj); init=initarray)

        e = reduce(vcat, values(e); init=initarray)
        hi, hj = xi.preservedname, xj.preservedname
        m, st_ϕ = l.ϕ(vcat(hi, hj, di .- dj, e,
                           repeat(θ; inner=(1, num_edges ÷ num_graphs))), ps.ϕ, st.ϕ)
        st = merge(st, (; ϕ=st_ϕ))
        return m
    end

    xs = merge((; preservedname=x), s)
    m = propagate(message, g, l.aggr; xi=xs, xj=xs, e=e)

    y, st_ψ = l.ψ(vcat(x, m, repeat(θ; inner=(1, num_nodes ÷ num_graphs))), ps.ψ, st.ψ)
    st = merge(st, (; ψ=st_ψ))

    return y, st
end

@doc raw"""
    GNOConv(in_chs => out_chs, ϕ; initialgraph = initialgraph, aggr = mean, bias = true)

Convolutional layer from [Neural Operator: Graph Kernel Network for Partial Differential Equations](https://openreview.net/forum?id=5fbUEUTZEn7).
```math
\begin{aligned}
	\mathbf{m}_i&=\Box _{j\in N(i)}\,\phi (\mathbf{a}_i,\mathbf{a}_j,\mathbf{x}_i,\mathbf{x}_j)\mathbf{h}_j\\
	\mathbf{h}_i'&=\,\,\sigma \left( \mathbf{Wh}_i+\mathbf{m}_i+\mathbf{b} \right)\\
\end{aligned}
```

# Arguments

- `in_chs`: Number of input channels.
- `out_chs`: Number of output channels.
- `ϕ`: Neural network for the message function. The output size of `ϕ` should be `in_chs * out_chs`.
- `initialgraph`: `GNNGraph` or a function that returns a `GNNGraph`
- `aggr`: Aggregation operator for the incoming messages (e.g. `+`, `*`, `max`, `min`, and `mean`).
- `bias`: Whether to add bias to the output.

# Inputs

- `h`: `Array` of the size `(in_chs, num_nodes)`.

# Returns

- `Array` of the size `(out_chs, num_nodes)`.

# Parameters

- Parameters of `ϕ`.
- `W`.
- `b`.

# States

- `graph`: `GNNGraph`. All features are stored in either `graph.ndata` or `graph.edata`. They will be concatenated and then fed into `ϕ`.

# Examples
```julia
g = rand_graph(10, 6)

g = GNNGraph(g, ndata = (; a = rand(2, 10), x = rand(3, 10)))
in_chs, out_chs = 5, 7
h = randn(in_chs, 10)
ϕ = Dense(2 + 2 + 3 + 3 => in_chs * out_chs)
l = GNOConv(5 => 7, ϕ, initialgraph = g)

rng = Random.default_rng()
ps, st = Lux.setup(rng, l)

y, st = l(h, ps, st)

#edge features
e = rand(2 + 2 + 3 + 3, 6)
g = GNNGraph(g, edata = e)
st = updategraph(st, g)
y, st = l(h, ps, st)
```

"""
struct GNOConv{bias, A} <: AbstractGNNContainerLayer{(:linear, :ϕ)}
    in_chs::Int
    out_chs::Int
    initialgraph::Function
    aggr::A
    linear::Dense
    ϕ::AbstractExplicitLayer
end

function GNOConv(in_chs::Int, out_chs::Int, ϕ::AbstractExplicitLayer, activation=identity;
                 initialgraph=initialgraph, init_weight=glorot_uniform, init_bias=zeros32,
                 aggr=mean, bias::Bool=true)
    return GNOConv(in_chs => out_chs, ϕ, activation; initialgraph=initialgraph,
                   init_weight=init_weight, init_bias=init_bias, aggr=aggr, bias=bias)
end

function GNOConv(ch::Pair{Int, Int}, ϕ::AbstractExplicitLayer, activation=identity;
                 initialgraph=initialgraph, init_weight=glorot_uniform, init_bias=zeros32,
                 aggr=mean, bias::Bool=true)
    initialgraph = wrapgraph(initialgraph)
    linear = Dense(ch, activation; init_weight=init_weight, init_bias=init_bias, bias=bias)
    return GNOConv{bias, typeof(aggr)}(first(ch), last(ch), initialgraph, aggr, linear, ϕ)
end

function (l::GNOConv{bias})(x::AbstractArray, ps, st::NamedTuple) where {bias}
    g = st.graph
    s = g.ndata
    nkeys = keys(s)
    num_edges = g.num_edges
    initarray = similar(x, 0, num_edges)

    function message(xi, xj, e)
        si, sj = xi[nkeys], xj[nkeys]
        si, sj = reduce(vcat, values(si); init=initarray),
                 reduce(vcat, values(sj); init=initarray)

        e = reduce(vcat, values(e); init=initarray)

        W, st_ϕ = l.ϕ(vcat(si, sj, e), ps.ϕ, st.ϕ)
        st = merge(st, (; ϕ=st_ϕ))

        hj = xj.h_
        W = reshape(W, :, l.in_chs, num_edges)
        hj = reshape(hj, l.in_chs, 1, num_edges)
        m = NNlib.batched_mul(W, hj)
        return reshape(m, :, num_edges)
    end

    xs = merge((; h_=x), s)
    m = propagate(message, g, l.aggr; xi=xs, xj=xs, e=g.edata)

    y = applyactivation(l.linear.activation, _linearmap(x, m, ps.linear, Val(bias)))

    return y, st
end

@inline function _linearmap(x::AbstractMatrix, m::AbstractMatrix, ps, ::Val{true})
    return elementwise_add(elementwise_add(ps.weight * x, m), ps.bias)
end

@inline function _linearmap(x::AbstractMatrix, m::AbstractMatrix, ps, ::Val{false})
    return elementwise_add(ps.weight * x, m)
end
