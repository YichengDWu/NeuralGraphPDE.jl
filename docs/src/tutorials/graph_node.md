# Neural Graph Ordinary Differential Equations

This tutorial is adapted from [SciMLSensitivity](https://sensitivity.sciml.ai/dev/neural_ode/neural_gde/), [GraphNeuralNetworks](https://github.com/CarloLucibello/GraphNeuralNetworks.jl/blob/master/examples/neural_ode_cora.jl), and [Lux](http://lux.csail.mit.edu/dev/examples/generated/intermediate/NeuralODE/main/).

## Load the packages

```@example gnode
using GraphNeuralNetworks, NeuralGraphPDE, DifferentialEquations
using Lux, NNlib, Optimisers, Zygote, Random
using ComponentArrays, OneHotArrays
using SciMLSensitivity
using Statistics: mean
using MLDatasets: Cora
using CUDA
CUDA.allowscalar(false)
device = CUDA.functional() ? gpu : cpu
```

## Load data

```@example gnode
dataset = Cora();
classes = dataset.metadata["classes"]
g = device(mldataset2gnngraph(dataset))
X = g.ndata.features
y = onehotbatch(g.ndata.targets, classes) # a dense matrix is not the optimal
(; train_mask, val_mask, test_mask) = g.ndata
ytrain = y[:, train_mask]
```

## Model and data configuration

```@example gnode
nin = size(X, 1)
nhidden = 16
nout = length(classes)
epochs = 40
```

## Define Neural ODE

```@example gnode
struct NeuralODE{M <: Lux.AbstractExplicitLayer, So, Se, T, K} <:
       Lux.AbstractExplicitContainerLayer{(:model,)}
    model::M
    solver::So
    sensealg::Se
    tspan::T
    kwargs::K
end

function NeuralODE(model::Lux.AbstractExplicitLayer;
                   solver = Tsit5(),
                   sensealg = InterpolatingAdjoint(; autojacvec = ZygoteVJP()),
                   tspan = (0.0f0, 1.0f0),
                   kwargs...)
    return NeuralODE(model, solver, sensealg, tspan, kwargs)
end

function (n::NeuralODE)(x, ps, st)
    function dudt(u, p, t)
        u_, st = n.model(u, p, st)
        return u_
    end
    prob = ODEProblem{false}(ODEFunction{false}(dudt), x, n.tspan, ps)
    return solve(prob, n.solver; sensealg = n.sensealg, n.kwargs...), st
end

function diffeqsol_to_array(x::ODESolution{T, N, <:AbstractVector{<:CuArray}}) where {T, N}
    return dropdims(gpu(x); dims = 3)
end
diffeqsol_to_array(x::ODESolution) = dropdims(Array(x); dims = 3)
```

## Create and initialize the Neural Graph ODE layer

```@example gnode
function create_model()
    node_chain = Chain(ExplicitGCNConv(nhidden => nhidden, relu),
                       ExplicitGCNConv(nhidden => nhidden, relu))

    node = NeuralODE(node_chain,
                     save_everystep = false,
                     reltol = 1e-3, abstol = 1e-3, save_start = false)

    model = Chain(ExplicitGCNConv(nin => nhidden, relu),
                  node,
                  diffeqsol_to_array,
                  Dense(nhidden, nout))

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    ps, st = Lux.setup(rng, model)
    ps = ComponentArray(ps) |> device
    st = updategraph(st, g) |> device

    return model, ps, st
end
```

## Define the loss function

```@example gnode
logitcrossentropy(ŷ, y) = mean(-sum(y .* logsoftmax(ŷ); dims = 1))

function loss(x, y, mask, model, ps, st)
    ŷ, st = model(x, ps, st)
    return logitcrossentropy(ŷ[:, mask], y), st
end

function eval_loss_accuracy(X, y, mask, model, ps, st)
    ŷ, _ = model(X, ps, st)
    l = logitcrossentropy(ŷ[:, mask], y[:, mask])
    acc = mean(onecold(ŷ[:, mask]) .== onecold(y[:, mask]))
    return (loss = round(l, digits = 4), acc = round(acc * 100, digits = 2))
end
```

## Train the model

```@example gnode
function train()
    model, ps, st = create_model()

    # Optimizer
    opt = Optimisers.Adam(0.01f0)
    st_opt = Optimisers.setup(opt, ps)

    # Training Loop
    for epoch in 1:epochs
        (l, st), back = pullback(p -> loss(X, ytrain, train_mask, model, p, st), ps)
        gs = back((one(l), nothing))[1]
        st_opt, ps = Optimisers.update(st_opt, ps, gs)
        @show eval_loss_accuracy(X, y, val_mask, model, ps, st)
    end
end

train()
```
