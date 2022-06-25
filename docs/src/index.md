```@meta
CurrentModule = NeuralGraphPDE
```

# NeuralGraphPDE

Documentation for [NeuralGraphPDE](https://github.com/MilkshakeForReal/NeuralGraphPDE.jl).

## Features

  - Layers and graphs are coupled and decoupled at the same time: You can bind a graph to a layer at initialization, but the graph
    is stored in `st`, not in the layer. They are decoupled in the sense that you can easily update or change the graph by changing `st`:

```@example demo
using NeuralGraphPDE, GraphNeuralNetworks, Random, Lux
g = rand_graph(5, 4, bidirected = false)
x = randn(3, g.num_nodes)

# create layer
l = ExplicitGCNConv(3 => 5, initialgraph = g)

# setup layer
rng = Random.default_rng()
Random.seed!(rng, 0)

ps, st = Lux.setup(rng, l)

# forward pass
y, st = l(x, ps, st)    # you don't need to feed a graph explicitly

#change the graph
new_g = rand_graph(5, 7, bidirected = false)
st = updategraph(st, new_g)

y, st = l(x, ps, st)
```

  - You can omit the key argument `initalgraph` at initialization, and then call `updategraph` on `st`:

```@example demo
g = rand_graph(5, 4, bidirected = false)
x = randn(3, g.num_nodes)

model = Chain(ExplicitGCNConv(3 => 5),
              ExplicitGCNConv(5 => 3))  # you don't need to use `g` for initalization
# setup layer
rng = Random.default_rng()
Random.seed!(rng, 0)

ps, st = Lux.setup(rng, model) # st.graph is empty
st = updategraph(st, g) # put the graph in there

# forward pass
y, st = model(x, ps, st)
```

  - An unified interface for graph level tasks. As pointed out [here](https://discourse.julialang.org/t/using-a-variable-graph-structure-with-neuralode-and-gcnconv/78881), GNNs are difficult to work well with other neural networks when the input graph is changing. This will not be an issue here. You have an unified interface `y, st = model(x, ps, st)`. In `GraphNeuralNetwork.jl`, you can use `Chain(GNNChain(...), Dense(...))` for graph levels tasks but you will not be able to feed a graph to `Chain(Dense(...), GNNChain(...))`.

  - Having node embeddings and other nontrainable features such as spaital coordinates? Thanks to [Lux](http://lux.csail.mit.edu/dev/manual/migrate_from_flux/#implementing-custom-layers), trainable parameters and nonntrainable parameters are seperately stored in `x` and `st`. We will not have to unpack and merge them over and over again.

## Implementing custom layers

`NeuralGraphPDE` basically share the same interface with `Lux.jl`. You may want to take a look at its [doc](http://lux.csail.mit.edu/dev/manual/migrate_from_flux/#implementing-custom-layers) first. Based on that, `NeuralGraphPDE` provides two abstract types, `AbstractGNNLayer` and `AbstractGNNContainerLayer`, they are subtypes of `AbstractExplicitLayer` and `AbstractExplicitContainerLayer`, respectively. You should subtype your custom layers to them.

### AbstractGNNLayer

You can define a custom layer with the following steps:

Step 1. Define your type of the layer and add `initialgraph` as a field.

```
struct MyGNNLayer{F} <: AbstractGNNLayer
    initialgraph::F
    ...
end
```

Step 2. Define `initialparameters` as in `Lux`. The default `initialstates` returns `(graph = GNNGraph(...))`, so this is optional. If you want to put more things in `st` then you need to overload `initialstates` as well.

```julia
function initialstates(rng::AbstractRNG, l::AbstractGNNLayer)
    (graph = l.initialgraph(), otherstates)
end
```

In this case, it is recommended to also overload `statelength`, it should be like

```julia
statelength(l::AbstractGNNLayer) = 1 + length(otherstates) # 1 for the graph
```

Step 3. Define the constructor(s) that has the keyword argument `initialgraph=initialgraph`.

```
function MyGNNLayer(...; initialgraph=initialgraph)
  initalgraph = wrapgraph(initialgraph) # always wrap initialgraph so the input can be a graph or a function
  MyGNNLayer{typeof(initialgraph), ...}(initialgraph,...)
end
```

Step 4. Define the forward pass. Keep in mind that the graph is stored in `st`. It is recommended to store nontrainable node features in the graph.

```
function (l::MyGNNLayer)(x,ps,st)
    g = st.graph
    s = g.ndata # nontrainable node features, if there is any
    function message(xi, xj, e)
        ...
        return m
    end
    xs = merge(x, s) # assuming x is a named tuple
    return propagte(message, g, l.aggr, xi = xs, xj = xs), st
end
```

### AbstractExplicitContainerLayer

You should only subtype your layer to `AbstractExplicitContainerLayer` then

 1. you need to write a custom message function, and
 2. the layer contains other layers.

For the most part it will look identical to defining `AbstractGNNLayer`. You just need to treat the message function more carefully.

```
function message(xi, xj, e)
        ...
        m, st.nn = nn(..., st.nn)
        st = merge(st, (nn = st_nn,))
        return m
end
```

Note that if you have only one neural layer insider a `AbstractExplicitContainerLayer`, then the parameters will be reduced but not the states.

```julia
julia> l = ExplicitEdgeConv(nn, initialgraph = g)


julia> rng = Random.default_rng()

julia> ps, st = Lux.setup(rng, l)

julia> ps
(weight = Float32[0.22180015 -0.09448394 … -0.41880473 -0.49083555; -0.23709725 0.05150031 … 0.48641983 0.14893274; … ; 0.42824164 0.5589718 … -0.5763395 0.18395355; 0.25994122 0.22801241 … 0.59201854 0.3832495], bias = Float32[0.0; 0.0; … ; 0.0; 0.0;;])

julia> st
(ϕ = NamedTuple(), graph = GNNGraph(3, 4))
```
