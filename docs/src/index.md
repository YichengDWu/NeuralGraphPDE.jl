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

  - You can omit the key argument `initalgraph` at initialization, and then call `updategraph` on `st` to put the graph in it. All gnn layer can work smoothly with other layers. 

```@example demo
g = rand_graph(5, 4, bidirected = false)
x = randn(3, g.num_nodes)

model = Chain(Dense(3 => 5),
              ExplicitGCNConv(5 => 5),
              ExplicitGCNConv(5 => 3))  # you don't need to use `g` for initalization
# setup layer
rng = Random.default_rng()
Random.seed!(rng, 0)

ps, st = Lux.setup(rng, model) # the default graph is empty
st = updategraph(st, g) # put the graph in st

# forward pass
y, st = model(x, ps, st)
```

  - An unified interface for graph level tasks. As pointed out [here](https://discourse.julialang.org/t/using-a-variable-graph-structure-with-neuralode-and-gcnconv/78881), GNNs are difficult to work well with other neural networks when the input graph is changing. This will not be an issue here. You have an unified interface `y, st = model(x, ps, st)`. In `GraphNeuralNetwork.jl`, you can use `Chain(GNNChain(...), Dense(...))` for graph levels tasks but you will not be able to feed a graph to `Chain(Dense(...), GNNChain(...))`.

  - Having node embeddings and other nontrainable features such as spaital coordinates? Thanks to [Lux](http://lux.csail.mit.edu/dev/manual/migrate_from_flux/#implementing-custom-layers), trainable parameters and nonntrainable parameters are seperately stored in `x` and `st`. We will not have to unpack and merge them over and over again.
