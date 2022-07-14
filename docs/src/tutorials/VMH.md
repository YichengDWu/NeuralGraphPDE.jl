# Neural Graph Partial Differential Equations

This tutorial is adapted from the paper [LEARNING CONTINUOUS-TIME PDES FROM SPARSE DATA WITH GRAPH NEURAL NETWORKS](https://github.com/yakovlev31/graphpdes_experiments/blob/master/convdiff/train.py).
We will use [`VMHConv`](@ref) to learn the dynamics of the convection-diffusion equation defined as

```math
\frac{\partial u(x, y, t)}{\partial t}=0.25 \nabla^{2} u(x, y, t)-\mathbf{v} \cdot \nabla u(x, y, t).
```

Specifically, we will learn the operator from the inital condition to the solution on the given temporal and spatial domain.

## Load the packages

```@example ngpde
using DataDeps, MLUtils, GraphNeuralNetworks, Fetch
using NeuralGraphPDE, Lux, Optimisers, Random
using CUDA, JLD2
using SciMLSensitivity, DifferentialEquations
using Zygote
using Flux.Losses: mse
import Lux: initialparameters, initialstates
using NNlib
using DiffEqFlux: NeuralODE
```

## Load data

```@example ngpde
function register_convdiff()
    register(DataDep("Convection_Diffusion_Equation",
                     """
                     Convection-Diffusion equation dataset from
                     [Learning continuous-time PDEs from sparse data with graph neural networks](https://github.com/yakovlev31/graphpdes_experiments)
                     """,
                     "https://drive.google.com/file/d/1oyatNeLizoO5co2ZVXIwZmWjJ046E9j6/view?usp=sharing",
                     fetch_method = gdownload))
end

register_convdiff()

function get_data()
    data = load(joinpath(datadep"Convection_Diffusion_Equation", "convdiff_n3000.jld2"))

    train_data = (data["gs_train"], data["u_train"])
    test_data = (data["gs_test"], data["u_test"])
    return train_data, test_data, data["dt_train"], data["dt_test"], data["tspan_train"],
           data["tspan_test"]
end

train_data, test_data, dt_train, dt_test, tspan_train, tspan_test = get_data()
```

The training data contrains 24 simulations on the time interval ``[0,0.2]``. Simulations are obeserved on different 2D grids with 3000 points.
Neighbors for each node were selected by applying Delaunay triangulation to the measurement positions. Two nodes were considered to be
neighbors if they lie on the same edge of at least one triangle.

## Utilities function

```@example ngpde
function diffeqsol_to_array(x::ODESolution{T, N, <:AbstractVector{<:CuArray}}) where {T, N}
    return gpu(x)
end

diffeqsol_to_array(x::ODESolution) = Array(x)
```

## Model

We will use only one message passing layer. The layer will have the following structure:

```@example ngpde
initialparameters(rng::AbstractRNG, node::NeuralODE) = initialparameters(rng, node.model)
initialstates(rng::AbstractRNG, node::NeuralODE) = initialstates(rng, node.model)

act = tanh
nhidden = 60
nout = 40

ϕ = Chain(Dense(4 => nhidden, act),
          Dense(nhidden => nhidden, act),
          Dense(nhidden => nhidden, act),
          Dense(nhidden => nout))

γ = Chain(Dense(nout + 1 => nhidden, act),
          Dense(nhidden => nhidden, act),
          Dense(nhidden => nhidden, act),
          Dense(nhidden => 1))

gnn = VMHConv(ϕ, γ)

node = NeuralODE(gnn, tspan_train, Tsit5(), saveat = dt_train, reltol = 1e-9, abstol = 1e-3)

model = Chain(node, diffeqsol_to_array)
```

## Optimiser

Since we only have 24 samples, we will use the `Rprop` optimiser.

```@example ngpde
using Optimisers: @.., @lazy, AbstractRule, onevalue
import Optimisers: init, apply!

struct Rprop{T} <: AbstractRule
    eta::T
    ell::Tuple{T, T}
    gamma::Tuple{T, T}
end

Rprop(η = 1.0f-3, ℓ = (5.0f-1, 1.2f0), Γ = (1.0f-6, 50.0f0)) = Rprop{typeof(η)}(η, ℓ, Γ)

init(o::Rprop, x::AbstractArray) = (zero(x), onevalue(o.eta, x))

function apply!(o::Rprop, state, x, dx)
    ℓ, Γ = o.ell, o.gamma
    g, η = state

    η = broadcast(g, η, dx) do g, η, dx
        g * dx > 0 ? min(η * ℓ[2], Γ[2]) : g * dx < 0 ? max(η * ℓ[1], Γ[1]) : η
    end

    g = broadcast(g, dx) do g, dx
        g * dx < 0 ? zero(dx) : dx
    end

    dx′ = @lazy η * sign(g)

    return (g, η), dx′
end

opt = Rprop(1.0f-6, (5.0f-1, 1.2f0), (1.0f-8, 10.0f0))
```

## Loss function

We will use the `mse` loss function.

```@example ngpde
function loss(x, y, ps, st)
    ŷ, st = model(x, ps, st)
    l = mse(ŷ, y)
    return l
end
```

## Train the model

The solution data has the shape `(space_points , time_points, num_samples)`. We will first permute the last two dimensions, resulting in the shape `(space_points , num_samples, time_points)`.
Then we flatten the first two dimensions, `(1, space_points * num_samples, time_points)`, and use the initial condition as the input to the model.
The output of the model will be of size `(1, space_points * time_points, num_samples)`.

```julia
mydevice = CUDA.functional() ? gpu : cpu
train_loader = DataLoader(train_data, batchsize = 24, shuffle = true)

rng = Random.default_rng()
Random.seed!(rng, 0)

function train()
    ps, st = Lux.setup(rng, model)
    ps = Lux.ComponentArray(ps) |> mydevice
    st = st |> mydevice
    st_opt = Optimisers.setup(opt, ps)

    for i in 1:200
        for (g, u) in train_loader
            g = g |> mydevice
            st = updategraph(st, g)
            u = u |> mydevice
            u0 = reshape(u[:, 1, :], 1, :)
            ut = permutedims(u, (1, 3, 2))
            ut = reshape(ut, 1, g.num_nodes, :)

            l, back = pullback(p -> loss(u0, ut, p, st), ps)
            ((i - 1) % 10 == 0) && @info "epoch $i | train loss = $l"
            gs = back(one(l))[1]
            st_opt, ps = Optimisers.update(st_opt, ps, gs)
        end
    end
end

train()
```

## Expected output

```
[ Info: epoch 10 | train loss = 0.02720912251427  0.53685   0.425613  0.71604
[ Info: epoch 20 | train loss = 0.026874812
[ Info: epoch 30 | train loss = 0.025392009
[ Info: epoch 40 | train loss = 0.023239506
[ Info: epoch 50 | train loss = 0.010599495
[ Info: epoch 60 | train loss = 0.010421633
[ Info: epoch 70 | train loss = 0.0098072495
[ Info: epoch 80 | train loss = 0.008936066
[ Info: epoch 90 | train loss = 0.0063929264
[ Info: epoch 100 | train loss = 0.004207685
[ Info: epoch 110 | train loss = 0.0026181203
[ Info: epoch 120 | train loss = 0.0023022622
[ Info: epoch 130 | train loss = 0.0019534715
[ Info: epoch 140 | train loss = 0.0017379699
[ Info: epoch 150 | train loss = 0.0015728847
[ Info: epoch 160 | train loss = 0.0013444767
[ Info: epoch 170 | train loss = 0.0012353633
[ Info: epoch 180 | train loss = 0.0011409305
[ Info: epoch 190 | train loss = 0.0010424983
[ Info: epoch 200 | train loss = 0.0009809926
```
