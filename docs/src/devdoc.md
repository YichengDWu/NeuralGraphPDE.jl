# Implementing custom layers

`NeuralGraphPDE` basically share the same interface with `Lux.jl`. You may want to take a look at its [doc](http://lux.csail.mit.edu/dev/manual/migrate_from_flux/#implementing-custom-layers) first. Based on that, `NeuralGraphPDE` provides two abstract types, `AbstractGNNLayer` and `AbstractGNNContainerLayer`, they are subtypes of `AbstractExplicitLayer` and `AbstractExplicitContainerLayer`, respectively. You should subtype your custom layers to them.

## AbstractGNNLayer

You can define a custom layer with the following steps:

Step 1. Define your type of the layer and add `initialgraph` as a field.

```
struct MyGNNLayer <: AbstractGNNLayer
    initialgraph::Function
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
  MyGNNLayer{...}(initialgraph,...)
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

## AbstractGNNContainerLayer

You should only subtype your layer to `AbstractGNNContainerLayer` when

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

Note that if you have only one neural layer insider a `AbstractGNNContainerLayer`, then the parameters will be reduced but not the states.

```julia
julia> l = ExplicitEdgeConv(nn, initialgraph = g)

julia> rng = Random.default_rng()

julia> ps, st = Lux.setup(rng, l)

julia> ps
(weight = Float32[0.22180015 -0.09448394 … -0.41880473 -0.49083555; -0.23709725 0.05150031 … 0.48641983 0.14893274; … ; 0.42824164 0.5589718 … -0.5763395 0.18395355; 0.25994122 0.22801241 … 0.59201854 0.3832495], bias = Float32[0.0; 0.0; … ; 0.0; 0.0;;])

julia> st
(ϕ = NamedTuple(), graph = GNNGraph(3, 4))
```
