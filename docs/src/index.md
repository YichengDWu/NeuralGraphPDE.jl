```@meta
CurrentModule = NeuralGraphPDE
```

# NeuralGraphPDE

Documentation for [NeuralGraphPDE](https://github.com/MilkshakeForReal/NeuralGraphPDE.jl).


## Why `Lux`?

- Layers and graphs are coupled and decoupled at the same time: You can bind the a graph to a layer at initialization, but the graph
  is stored in `st`. They are decoupled in the sense that you can easily update or change the graph by change `st`. 
  ```julia
    g = rand_graph(5, 4, bidirected=false)
    x = randn(3, g.num_nodes)

    # create layer
    l = ExplicitGCNConv(3 => 5, initialgraph = g) 

    # setup layer
    rng = Random.default_rng()
    Random.seed!(rng, 0)

    ps, st = Lux.setup(rng, l)

    # forward pass
    y = l(x, ps, st)    # you don't need to feed graph in the forward pass
    
    #change graph
    new_g = rand_graph(5, 4, bidirected=false)
    merge(st, (graph = new_g,))

  ```