var documenterSearchIndex = {"docs":
[{"location":"api/messagepassing/","page":"Message Passing","title":"Message Passing","text":"CurrentModule = NeuralGraphPDE","category":"page"},{"location":"api/messagepassing/#Message-Passing","page":"Message Passing","title":"Message Passing","text":"","category":"section"},{"location":"api/messagepassing/#Index","page":"Message Passing","title":"Index","text":"","category":"section"},{"location":"api/messagepassing/","page":"Message Passing","title":"Message Passing","text":"Order = [:type, :function]\nPages   = [\"messagepassing.md\"]","category":"page"},{"location":"api/messagepassing/#Docs","page":"Message Passing","title":"Docs","text":"","category":"section"},{"location":"api/messagepassing/","page":"Message Passing","title":"Message Passing","text":"apply_edges\npropagate","category":"page"},{"location":"api/messagepassing/#GraphNeuralNetworks.apply_edges","page":"Message Passing","title":"GraphNeuralNetworks.apply_edges","text":"apply_edges(f, g, xi, xj, e, ps, st)\napply_edges(f, g, ps, st; [xi, xj, e])\n\nArguments\n\nps: Parameters of the neural network inside the function f\nst: State of the neural network inside the function f\n\n\n\n\n\n","category":"function"},{"location":"api/messagepassing/#GraphNeuralNetworks.propagate","page":"Message Passing","title":"GraphNeuralNetworks.propagate","text":"propagate(f, g, aggr, ps, st; xi, xj, e)  ->  m̄\n\n\n\n\n\n","category":"function"},{"location":"api/layers/","page":"layers","title":"layers","text":"CurrentModule = NeuralGraphPDE","category":"page"},{"location":"api/layers/#Layers","page":"layers","title":"Layers","text":"","category":"section"},{"location":"api/layers/#Index","page":"layers","title":"Index","text":"","category":"section"},{"location":"api/layers/","page":"layers","title":"layers","text":"Order = [:type, :function]\nModules = [NeuralGraphPDE]\nPages = [\"layers.md\"]","category":"page"},{"location":"api/layers/#Docs","page":"layers","title":"Docs","text":"","category":"section"},{"location":"api/layers/","page":"layers","title":"layers","text":"Modules = [NeuralGraphPDE]\nPages   = [\"layers.jl\"]\nPrivate = false","category":"page"},{"location":"api/layers/#NeuralGraphPDE.ExplicitEdgeConv","page":"layers","title":"NeuralGraphPDE.ExplicitEdgeConv","text":"ExplicitEdgeConv(ϕ; aggr=max)\n\nArguments\n\nϕ: A neural network. \naggr: Aggregation operator for the incoming messages (e.g. +, *, max, min, and mean).\n\nInput\n\nCase1: \nndata: NamedTuple (u=u,...,x=x) where u is the node embedding and x is the spatial coordinate.\nCase2: \nndata: NamedTuple or Array.\nedata: Array of spatial differences.\n\n\n\n\n\n","category":"type"},{"location":"api/layers/#NeuralGraphPDE.ExplicitGCNConv","page":"layers","title":"NeuralGraphPDE.ExplicitGCNConv","text":"ExplicitGCNConv()\n\nSame as the one in GraphNeuralNetworks.jl but with exiplicit paramters\n\nArguments\n\nin_chs: \nout_chs:\nactivation:\nadd_self_loops: \nuse_edge_weight:\n\nExamples\n\n# create data\ns = [1,1,2,3]\nt = [2,3,1,1]\ng = GNNGraph(s, t)\nx = randn(3, g.num_nodes)\n\n# create layer\nl = ExplicitGCNConv(3 => 5, initialgraph = g) \n\n# setup layer\nrng = Random.default_rng()\nRandom.seed!(rng, 0)\n\nps, st = Lux.setup(rng, l)\n\n# forward pass\ny = l(x, ps, st)       # size:  5 × num_nodes\n\n\n\n\n\n","category":"type"},{"location":"tutorials/graph_node/#Neural-Graph-Ordinary-Differential-Equations","page":"Neural Graph Ordinary Differential Equations","title":"Neural Graph Ordinary Differential Equations","text":"","category":"section"},{"location":"tutorials/graph_node/","page":"Neural Graph Ordinary Differential Equations","title":"Neural Graph Ordinary Differential Equations","text":"This tutorial is adapted from SciMLSensitivity, GraphNeuralNetworks, and Lux.","category":"page"},{"location":"tutorials/graph_node/#Load-the-packages","page":"Neural Graph Ordinary Differential Equations","title":"Load the packages","text":"","category":"section"},{"location":"tutorials/graph_node/","page":"Neural Graph Ordinary Differential Equations","title":"Neural Graph Ordinary Differential Equations","text":"using GraphNeuralNetworks, NeuralGraphPDE, DifferentialEquations\nimport NeuralGraphPDE: initialgraph\nusing Lux, NNlib, Optimisers, Zygote, Random, ComponentArrays\nusing DiffEqSensitivity\nusing Statistics: mean\nusing MLDatasets: Cora\nusing CUDA\nCUDA.allowscalar(false)\ndevice = CUDA.functional() ? gpu : cpu","category":"page"},{"location":"tutorials/graph_node/#Load-data","page":"Neural Graph Ordinary Differential Equations","title":"Load data","text":"","category":"section"},{"location":"tutorials/graph_node/","page":"Neural Graph Ordinary Differential Equations","title":"Neural Graph Ordinary Differential Equations","text":"onehotbatch(data::CuArray,labels)= cu(labels).==reshape(data, 1,size(data)...)\nonecold(y) =  map(argmax,eachcol(y))\n\ndataset = Cora()\nclasses = dataset.metadata[\"classes\"]\ng = mldataset2gnngraph(dataset) |> device\nX = g.ndata.features\ny = onehotbatch(g.ndata.targets, classes) # a dense matrix is not the optimal\n(; train_mask, val_mask, test_mask) = g.ndata\nytrain = y[:,train_mask]","category":"page"},{"location":"tutorials/graph_node/#Model-and-data-configuration","page":"Neural Graph Ordinary Differential Equations","title":"Model and data configuration","text":"","category":"section"},{"location":"tutorials/graph_node/","page":"Neural Graph Ordinary Differential Equations","title":"Neural Graph Ordinary Differential Equations","text":"nin = size(X, 1)\nnhidden = 16\nnout = length(classes)\nepochs = 40","category":"page"},{"location":"tutorials/graph_node/#Define-Neural-ODE","page":"Neural Graph Ordinary Differential Equations","title":"Define Neural ODE","text":"","category":"section"},{"location":"tutorials/graph_node/","page":"Neural Graph Ordinary Differential Equations","title":"Neural Graph Ordinary Differential Equations","text":"struct NeuralODE{M <: Lux.AbstractExplicitLayer, So, Se, T, K} <:\n       Lux.AbstractExplicitContainerLayer{(:model,)}\n    model::M\n    solver::So\n    sensealg::Se\n    tspan::T\n    kwargs::K\nend\n\nfunction NeuralODE(model::Lux.AbstractExplicitLayer;\n                   solver=Tsit5(),\n                   sensealg=InterpolatingAdjoint(; autojacvec=ZygoteVJP()),\n                   tspan=(0.0f0, 1.0f0),\n                   kwargs...)\n    return NeuralODE(model, solver, sensealg, tspan, kwargs)\nend\n\nfunction (n::NeuralODE)(x, ps, st)\n    function dudt(u, p, t)\n        u_, st = n.model(u, p, st)\n        return u_\n    end\n    prob = ODEProblem{false}(ODEFunction{false}(dudt), x, n.tspan, ps)\n    return solve(prob, n.solver; sensealg=n.sensealg, n.kwargs...), st\nend\n\nfunction diffeqsol_to_array(x::ODESolution{T, N, <:AbstractVector{<:CuArray}}) where {T, N}\n    return dropdims(gpu(x); dims=3)\nend\ndiffeqsol_to_array(x::ODESolution) = dropdims(Array(x); dims=3)","category":"page"},{"location":"tutorials/graph_node/#Create-and-initialize-the-Neural-Graph-ODE-layer","page":"Neural Graph Ordinary Differential Equations","title":"Create and initialize the Neural Graph ODE layer","text":"","category":"section"},{"location":"tutorials/graph_node/","page":"Neural Graph Ordinary Differential Equations","title":"Neural Graph Ordinary Differential Equations","text":"initialgraph() = copy(g)\nfunction create_model()\n    node_chain = Chain(ExplicitGCNConv(nhidden => nhidden, relu),\n                       ExplicitGCNConv(nhidden => nhidden, relu))\n\n    node = NeuralODE(node_chain,\n                     save_everystep = false,\n                     reltol = 1e-3, abstol = 1e-3, save_start = false)\n\n    model = Chain(ExplicitGCNConv(nin => nhidden, relu),\n                  node,\n                  diffeqsol_to_array,\n                  Dense(nhidden, nout))\n\n    rng = Random.default_rng()\n    Random.seed!(rng, 0)\n\n    ps, st = Lux.setup(rng, model)\n    ps = ComponentArray(ps) |> gpu\n    st = st |> gpu\n\n    return model, ps, st\nend","category":"page"},{"location":"tutorials/graph_node/#Define-the-loss-function","page":"Neural Graph Ordinary Differential Equations","title":"Define the loss function","text":"","category":"section"},{"location":"tutorials/graph_node/","page":"Neural Graph Ordinary Differential Equations","title":"Neural Graph Ordinary Differential Equations","text":"logitcrossentropy(ŷ, y) = mean(-sum(y .* logsoftmax(ŷ); dims=1))\n\nfunction loss(x, y, mask, model, ps, st)\n    ŷ, st = model(x, ps, st)\n    return logitcrossentropy(ŷ[:,mask], y), st\nend\n\nfunction eval_loss_accuracy(X, y, mask, model, ps, st)\n    ŷ, _ = model(X, ps, st)\n    l = logitcrossentropy(ŷ[:,mask], y[:,mask])\n    acc = mean(onecold(ŷ[:,mask]) .== onecold(y[:,mask]))\n    return (loss = round(l, digits=4), acc = round(acc*100, digits=2))\nend","category":"page"},{"location":"tutorials/graph_node/#Train-the-model","page":"Neural Graph Ordinary Differential Equations","title":"Train the model","text":"","category":"section"},{"location":"tutorials/graph_node/","page":"Neural Graph Ordinary Differential Equations","title":"Neural Graph Ordinary Differential Equations","text":"function train()\n    model, ps, st = create_model()\n\n    # Optimizer\n    opt = Optimisers.Adam(0.01f0)\n    st_opt = Optimisers.setup(opt,ps)\n\n    # Training Loop\n    for epoch in 1:epochs\n        (l,st), back = pullback(p->loss(X, ytrain, train_mask, model, p, st),ps)\n        gs = back((one(l), nothing))[1]\n        st_opt, ps = Optimisers.update(st_opt, ps, gs)\n        @show eval_loss_accuracy(X, y, val_mask, model, ps, st)\n    end\nend\n\ntrain()","category":"page"},{"location":"tutorials/graph_node/#Expected-Output","page":"Neural Graph Ordinary Differential Equations","title":"Expected Output","text":"","category":"section"},{"location":"tutorials/graph_node/","page":"Neural Graph Ordinary Differential Equations","title":"Neural Graph Ordinary Differential Equations","text":"eval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.8719f0, acc = 38.0)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.7944f0, acc = 39.4)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.6933f0, acc = 41.6)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.5726f0, acc = 48.4)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.447f0, acc = 57.6)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.3408f0, acc = 62.8)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.2522f0, acc = 68.8)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.1682f0, acc = 70.0)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.0832f0, acc = 72.8)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.0095f0, acc = 74.8)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 0.9572f0, acc = 75.2)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 0.9304f0, acc = 75.2)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 0.9238f0, acc = 75.2)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 0.9119f0, acc = 74.6)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 0.8986f0, acc = 75.2)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 0.9082f0, acc = 75.8)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 0.9463f0, acc = 75.2)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 0.9791f0, acc = 75.6)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 0.9932f0, acc = 75.6)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.0197f0, acc = 75.6)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.0741f0, acc = 75.6)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.1411f0, acc = 75.4)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.1969f0, acc = 75.2)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.2325f0, acc = 75.2)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.2646f0, acc = 75.2)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.3119f0, acc = 74.6)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.3799f0, acc = 74.4)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.4597f0, acc = 74.4)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.5367f0, acc = 74.2)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.5992f0, acc = 74.0)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.6423f0, acc = 73.8)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.6703f0, acc = 74.2)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.694f0, acc = 74.4)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.7245f0, acc = 74.6)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.769f0, acc = 74.8)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.8259f0, acc = 74.4)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.8883f0, acc = 74.4)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.9483f0, acc = 74.6)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.9996f0, acc = 74.2)\neval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 2.0371f0, acc = 74.2)","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = NeuralGraphPDE","category":"page"},{"location":"#NeuralGraphPDE","page":"Home","title":"NeuralGraphPDE","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for NeuralGraphPDE.","category":"page"},{"location":"#Why-Lux?","page":"Home","title":"Why Lux?","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Layers and graphs are coupled and decoupled at the same time: You can bind the a graph to a layer at initialization, but the graph is stored in st. They are decoupled in the sense that you can easily update or change the graph by change st. \n  g = rand_graph(5, 4, bidirected=false)\n  x = randn(3, g.num_nodes)\n\n  # create layer\n  l = ExplicitGCNConv(3 => 5, initialgraph = g) \n\n  # setup layer\n  rng = Random.default_rng()\n  Random.seed!(rng, 0)\n\n  ps, st = Lux.setup(rng, l)\n\n  # forward pass\n  y = l(x, ps, st)    # you don't need to feed graph in the forward pass\n  \n  #change graph\n  new_g = rand_graph(5, 4, bidirected=false)\n  merge(st, (graph = new_g,))\n","category":"page"}]
}
