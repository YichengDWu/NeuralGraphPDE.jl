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
g = rand_graph(5, 4, bidirected=false)
x = randn(3, g.num_nodes)

# create layer
l = ExplicitGCNConv(3 => 5, initialgraph = g) 

# setup layer
rng = Random.default_rng()
Random.seed!(rng, 0)

ps, st = Lux.setup(rng, l)

# forward pass
y, st = l(x, ps, st)    # you don't need to feed graph in the forward pass

#change the graph
new_g = rand_graph(5, 7, bidirected=false)
st = updategraph(new_g)

y, st = l(x, ps, st)
```

- For node level problems, you can define the graph only once and forget it. The way to do it is to overload `initalgraph`:
  
```@example demo
import NeuralGraphPDE: initialgraph
g = rand_graph(5, 4, bidirected=false)
x = randn(3, g.num_nodes)

initialgraph() = copy(g) 

model = Chain(ExplicitGCNConv(3 => 5),
              ExplicitGCNConv(5 => 3))  # you don't need to use `g` for initalization anymore
# setup layer
rng = Random.default_rng()
Random.seed!(rng, 0)

ps, st = Lux.setup(rng, model)

# forward pass
y, st = model(x, ps, st)
```