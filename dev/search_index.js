var documenterSearchIndex = {"docs":
[{"location":"devdoc/#Implementing-custom-layers","page":"Developer Documentation","title":"Implementing custom layers","text":"","category":"section"},{"location":"devdoc/","page":"Developer Documentation","title":"Developer Documentation","text":"NeuralGraphPDE basically share the same interface with Lux.jl. You may want to take a look at its doc first. Based on that, NeuralGraphPDE provides two abstract types, AbstractGNNLayer and AbstractGNNContainerLayer, they are subtypes of AbstractExplicitLayer and AbstractExplicitContainerLayer, respectively. You should subtype your custom layers to them.","category":"page"},{"location":"devdoc/#AbstractGNNLayer","page":"Developer Documentation","title":"AbstractGNNLayer","text":"","category":"section"},{"location":"devdoc/","page":"Developer Documentation","title":"Developer Documentation","text":"You can define a custom layer with the following steps:","category":"page"},{"location":"devdoc/","page":"Developer Documentation","title":"Developer Documentation","text":"Step 1. Define your type of the layer and add initialgraph as a field.","category":"page"},{"location":"devdoc/","page":"Developer Documentation","title":"Developer Documentation","text":"struct MyGNNLayer <: AbstractGNNLayer\n    initialgraph::Function\n    ...\nend","category":"page"},{"location":"devdoc/","page":"Developer Documentation","title":"Developer Documentation","text":"Step 2. Define initialparameters as in Lux. The default initialstates returns (graph = GNNGraph(...)), so this is optional. If you want to put more things in st then you need to overload initialstates as well.","category":"page"},{"location":"devdoc/","page":"Developer Documentation","title":"Developer Documentation","text":"function initialstates(rng::AbstractRNG, l::AbstractGNNLayer)\n    return (graph=l.initialgraph(), otherstates)\nend","category":"page"},{"location":"devdoc/","page":"Developer Documentation","title":"Developer Documentation","text":"In this case, it is recommended to also overload statelength, it should be like","category":"page"},{"location":"devdoc/","page":"Developer Documentation","title":"Developer Documentation","text":"statelength(l::AbstractGNNLayer) = 1 + length(otherstates) # 1 for the graph","category":"page"},{"location":"devdoc/","page":"Developer Documentation","title":"Developer Documentation","text":"Step 3. Define the constructor(s) that has the keyword argument initialgraph=initialgraph.","category":"page"},{"location":"devdoc/","page":"Developer Documentation","title":"Developer Documentation","text":"function MyGNNLayer(...; initialgraph=initialgraph)\n  initalgraph = wrapgraph(initialgraph) # always wrap initialgraph so the input can be a graph or a function\n  MyGNNLayer{...}(initialgraph,...)\nend","category":"page"},{"location":"devdoc/","page":"Developer Documentation","title":"Developer Documentation","text":"Step 4. Define the forward pass. Keep in mind that the graph is stored in st. It is recommended to store nontrainable node features in the graph.","category":"page"},{"location":"devdoc/","page":"Developer Documentation","title":"Developer Documentation","text":"function (l::MyGNNLayer)(x,ps,st)\n    g = st.graph\n    s = g.ndata # nontrainable node features, if there is any\n    function message(xi, xj, e)\n        ...\n        return m\n    end\n    xs = merge(x, s) # assuming x is a named tuple\n    return propagte(message, g, l.aggr, xi = xs, xj = xs), st\nend","category":"page"},{"location":"devdoc/#AbstractGNNContainerLayer","page":"Developer Documentation","title":"AbstractGNNContainerLayer","text":"","category":"section"},{"location":"devdoc/","page":"Developer Documentation","title":"Developer Documentation","text":"You should only subtype your layer to AbstractGNNContainerLayer when","category":"page"},{"location":"devdoc/","page":"Developer Documentation","title":"Developer Documentation","text":"you need to write a custom message function, and\nthe layer contains other layers.","category":"page"},{"location":"devdoc/","page":"Developer Documentation","title":"Developer Documentation","text":"For the most part it will look identical to defining AbstractGNNLayer. You just need to treat the message function more carefully.","category":"page"},{"location":"devdoc/","page":"Developer Documentation","title":"Developer Documentation","text":"function message(xi, xj, e)\n        ...\n        m, st.nn = nn(..., st.nn)\n        st = merge(st, (nn = st_nn,))\n        return m\nend","category":"page"},{"location":"devdoc/","page":"Developer Documentation","title":"Developer Documentation","text":"Note that if you have only one neural layer insider a AbstractGNNContainerLayer, then the parameters will be reduced but not the states.","category":"page"},{"location":"devdoc/","page":"Developer Documentation","title":"Developer Documentation","text":"julia> l = ExplicitEdgeConv(nn; initialgraph=g)\n\njulia> rng = Random.default_rng()\n\njulia> ps, st = Lux.setup(rng, l)\n\njulia> ps\n(weight = Float32[0.22180015 -0.09448394 … -0.41880473 -0.49083555; -0.23709725 0.05150031 … 0.48641983 0.14893274; … ; 0.42824164 0.5589718 … -0.5763395 0.18395355; 0.25994122 0.22801241 … 0.59201854 0.3832495], bias = Float32[0.0; 0.0; … ; 0.0; 0.0;;])\n\njulia> st\n(ϕ = NamedTuple(), graph = GNNGraph(3, 4))","category":"page"},{"location":"api/messagepassing/","page":"Message Passing","title":"Message Passing","text":"CurrentModule = NeuralGraphPDE","category":"page"},{"location":"api/messagepassing/#Message-Passing","page":"Message Passing","title":"Message Passing","text":"","category":"section"},{"location":"api/messagepassing/#Index","page":"Message Passing","title":"Index","text":"","category":"section"},{"location":"api/messagepassing/","page":"Message Passing","title":"Message Passing","text":"Order = [:type, :function]\nPages   = [\"messagepassing.md\"]","category":"page"},{"location":"api/messagepassing/#Docs","page":"Message Passing","title":"Docs","text":"","category":"section"},{"location":"api/messagepassing/","page":"Message Passing","title":"Message Passing","text":"apply_edges\npropagate","category":"page"},{"location":"api/utilities/","page":"Utilities","title":"Utilities","text":"CurrentModule = NeuralGraphPDE","category":"page"},{"location":"api/utilities/#Layers","page":"Utilities","title":"Layers","text":"","category":"section"},{"location":"api/utilities/#Index","page":"Utilities","title":"Index","text":"","category":"section"},{"location":"api/utilities/","page":"Utilities","title":"Utilities","text":"Order = [:type, :function]\nModules = [NeuralGraphPDE]\nPages = [\"utilities.md\"]","category":"page"},{"location":"api/utilities/#Docs","page":"Utilities","title":"Docs","text":"","category":"section"},{"location":"api/utilities/","page":"Utilities","title":"Utilities","text":"Modules = [NeuralGraphPDE]\nPages   = [\"utils.jl\"]\nPrivate = false","category":"page"},{"location":"api/utilities/#NeuralGraphPDE.updategraph","page":"Utilities","title":"NeuralGraphPDE.updategraph","text":"updategraph(st, g; kwargs...) -> st\n\nRecursively replace the value of graph with a shallow copy of g. If g is nothing, then only update the old graph with the data given in kwargs.\n\n\n\n\n\n","category":"function"},{"location":"api/utilities/","page":"Utilities","title":"Utilities","text":"copy","category":"page"},{"location":"api/utilities/#Base.copy","page":"Utilities","title":"Base.copy","text":"copy(g::GNNGraph; kwargs...)\n\nCreate a shollow copy of the input graph g. This is equivalent to GNNGraph(g).\n\n\n\n\n\n","category":"function"},{"location":"tutorials/VMH/#Neural-Graph-Partial-Differential-Equations","page":"Neural Graph Partial Differential Equations","title":"Neural Graph Partial Differential Equations","text":"","category":"section"},{"location":"tutorials/VMH/","page":"Neural Graph Partial Differential Equations","title":"Neural Graph Partial Differential Equations","text":"This tutorial is adapted from the paper LEARNING CONTINUOUS-TIME PDES FROM SPARSE DATA WITH GRAPH NEURAL NETWORKS. We will use VMHConv to learn the dynamics of the convection-diffusion equation defined as","category":"page"},{"location":"tutorials/VMH/","page":"Neural Graph Partial Differential Equations","title":"Neural Graph Partial Differential Equations","text":"fracpartial u(x y t)partial t=025 nabla^2 u(x y t)-mathbfv cdot nabla u(x y t)","category":"page"},{"location":"tutorials/VMH/","page":"Neural Graph Partial Differential Equations","title":"Neural Graph Partial Differential Equations","text":"Specifically, we will learn the operator from the inital condition to the solution on the given temporal and spatial domain.","category":"page"},{"location":"tutorials/VMH/#Load-the-packages","page":"Neural Graph Partial Differential Equations","title":"Load the packages","text":"","category":"section"},{"location":"tutorials/VMH/","page":"Neural Graph Partial Differential Equations","title":"Neural Graph Partial Differential Equations","text":"using DataDeps, MLUtils, GraphNeuralNetworks, Fetch\nusing NeuralGraphPDE, Lux, Optimisers, Random\nusing CUDA, JLD2\nusing SciMLSensitivity, DifferentialEquations\nusing Zygote\nusing Flux.Losses: mse\nimport Lux: initialparameters, initialstates\nusing NNlib\nusing DiffEqFlux: NeuralODE","category":"page"},{"location":"tutorials/VMH/#Load-data","page":"Neural Graph Partial Differential Equations","title":"Load data","text":"","category":"section"},{"location":"tutorials/VMH/","page":"Neural Graph Partial Differential Equations","title":"Neural Graph Partial Differential Equations","text":"function register_convdiff()\n    return register(DataDep(\"Convection_Diffusion_Equation\",\n                            \"\"\"\n                            Convection-Diffusion equation dataset from\n                            [Learning continuous-time PDEs from sparse data with graph neural networks](https://github.com/yakovlev31/graphpdes_experiments)\n                            \"\"\",\n                            \"https://drive.google.com/file/d/1oyatNeLizoO5co2ZVXIwZmWjJ046E9j6/view?usp=sharing\";\n                            fetch_method=gdownload))\nend\n\nregister_convdiff()\n\nfunction get_data()\n    data = load(joinpath(datadep\"Convection_Diffusion_Equation\", \"convdiff_n3000.jld2\"))\n\n    train_data = (data[\"gs_train\"], data[\"u_train\"])\n    test_data = (data[\"gs_test\"], data[\"u_test\"])\n    return train_data, test_data, data[\"dt_train\"], data[\"dt_test\"], data[\"tspan_train\"],\n           data[\"tspan_test\"]\nend\n\ntrain_data, test_data, dt_train, dt_test, tspan_train, tspan_test = get_data()","category":"page"},{"location":"tutorials/VMH/","page":"Neural Graph Partial Differential Equations","title":"Neural Graph Partial Differential Equations","text":"The training data contrains 24 simulations on the time interval 002. Simulations are obeserved on different 2D grids with 3000 points. Neighbors for each node were selected by applying Delaunay triangulation to the measurement positions. Two nodes were considered to be neighbors if they lie on the same edge of at least one triangle.","category":"page"},{"location":"tutorials/VMH/#Utilities-function","page":"Neural Graph Partial Differential Equations","title":"Utilities function","text":"","category":"section"},{"location":"tutorials/VMH/","page":"Neural Graph Partial Differential Equations","title":"Neural Graph Partial Differential Equations","text":"function diffeqsol_to_array(x::ODESolution{T, N, <:AbstractVector{<:CuArray}}) where {T, N}\n    return gpu(x)\nend\n\ndiffeqsol_to_array(x::ODESolution) = Array(x)","category":"page"},{"location":"tutorials/VMH/#Model","page":"Neural Graph Partial Differential Equations","title":"Model","text":"","category":"section"},{"location":"tutorials/VMH/","page":"Neural Graph Partial Differential Equations","title":"Neural Graph Partial Differential Equations","text":"We will use only one message passing layer. The layer will have the following structure:","category":"page"},{"location":"tutorials/VMH/","page":"Neural Graph Partial Differential Equations","title":"Neural Graph Partial Differential Equations","text":"initialparameters(rng::AbstractRNG, node::NeuralODE) = initialparameters(rng, node.model)\ninitialstates(rng::AbstractRNG, node::NeuralODE) = initialstates(rng, node.model)\n\nact = tanh\nnhidden = 60\nnout = 40\n\nϕ = Chain(Dense(4 => nhidden, act), Dense(nhidden => nhidden, act),\n          Dense(nhidden => nhidden, act), Dense(nhidden => nout))\n\nγ = Chain(Dense(nout + 1 => nhidden, act), Dense(nhidden => nhidden, act),\n          Dense(nhidden => nhidden, act), Dense(nhidden => 1))\n\ngnn = VMHConv(ϕ, γ)\n\nnode = NeuralODE(gnn, tspan_train, Tsit5(); saveat=dt_train, reltol=1e-9, abstol=1e-3)\n\nmodel = Chain(node, diffeqsol_to_array)","category":"page"},{"location":"tutorials/VMH/#Optimiser","page":"Neural Graph Partial Differential Equations","title":"Optimiser","text":"","category":"section"},{"location":"tutorials/VMH/","page":"Neural Graph Partial Differential Equations","title":"Neural Graph Partial Differential Equations","text":"Since we only have 24 samples, we will use the Rprop optimiser.","category":"page"},{"location":"tutorials/VMH/","page":"Neural Graph Partial Differential Equations","title":"Neural Graph Partial Differential Equations","text":"opt = Rprop(1.0f-6, (5.0f-1, 1.2f0), (1.0f-8, 10.0f0))","category":"page"},{"location":"tutorials/VMH/#Loss-function","page":"Neural Graph Partial Differential Equations","title":"Loss function","text":"","category":"section"},{"location":"tutorials/VMH/","page":"Neural Graph Partial Differential Equations","title":"Neural Graph Partial Differential Equations","text":"We will use the mse loss function.","category":"page"},{"location":"tutorials/VMH/","page":"Neural Graph Partial Differential Equations","title":"Neural Graph Partial Differential Equations","text":"function loss(x, y, ps, st)\n    ŷ, st = model(x, ps, st)\n    l = mse(ŷ, y)\n    return l\nend","category":"page"},{"location":"tutorials/VMH/#Train-the-model","page":"Neural Graph Partial Differential Equations","title":"Train the model","text":"","category":"section"},{"location":"tutorials/VMH/","page":"Neural Graph Partial Differential Equations","title":"Neural Graph Partial Differential Equations","text":"The solution data has the shape (space_points , time_points, num_samples). We will first permute the last two dimensions, resulting in the shape (space_points , num_samples, time_points). Then we flatten the first two dimensions, (1, space_points * num_samples, time_points), and use the initial condition as the input to the model. The output of the model will be of size (1, space_points * time_points, num_samples).","category":"page"},{"location":"tutorials/VMH/","page":"Neural Graph Partial Differential Equations","title":"Neural Graph Partial Differential Equations","text":"mydevice = CUDA.functional() ? gpu : cpu\ntrain_loader = DataLoader(train_data; batchsize=24, shuffle=true)\n\nrng = Random.default_rng()\nRandom.seed!(rng, 0)\n\nfunction train()\n    ps, st = Lux.setup(rng, model)\n    ps = Lux.ComponentArray(ps) |> mydevice\n    st = st |> mydevice\n    st_opt = Optimisers.setup(opt, ps)\n\n    for i in 1:200\n        for (g, u) in train_loader\n            g = g |> mydevice\n            st = updategraph(st, g)\n            u = u |> mydevice\n            u0 = reshape(u[:, 1, :], 1, :)\n            ut = permutedims(u, (1, 3, 2))\n            ut = reshape(ut, 1, g.num_nodes, :)\n\n            l, back = pullback(p -> loss(u0, ut, p, st), ps)\n            ((i - 1) % 10 == 0) && @info \"epoch $i | train loss = $l\"\n            gs = back(one(l))[1]\n            st_opt, ps = Optimisers.update(st_opt, ps, gs)\n        end\n    end\nend\n\ntrain()","category":"page"},{"location":"tutorials/VMH/#Expected-output","page":"Neural Graph Partial Differential Equations","title":"Expected output","text":"","category":"section"},{"location":"tutorials/VMH/","page":"Neural Graph Partial Differential Equations","title":"Neural Graph Partial Differential Equations","text":"[ Info: epoch 10 | train loss = 0.02720912251427  0.53685   0.425613  0.71604\n[ Info: epoch 20 | train loss = 0.026874812\n[ Info: epoch 30 | train loss = 0.025392009\n[ Info: epoch 40 | train loss = 0.023239506\n[ Info: epoch 50 | train loss = 0.010599495\n[ Info: epoch 60 | train loss = 0.010421633\n[ Info: epoch 70 | train loss = 0.0098072495\n[ Info: epoch 80 | train loss = 0.008936066\n[ Info: epoch 90 | train loss = 0.0063929264\n[ Info: epoch 100 | train loss = 0.004207685\n[ Info: epoch 110 | train loss = 0.0026181203\n[ Info: epoch 120 | train loss = 0.0023022622\n[ Info: epoch 130 | train loss = 0.0019534715\n[ Info: epoch 140 | train loss = 0.0017379699\n[ Info: epoch 150 | train loss = 0.0015728847\n[ Info: epoch 160 | train loss = 0.0013444767\n[ Info: epoch 170 | train loss = 0.0012353633\n[ Info: epoch 180 | train loss = 0.0011409305\n[ Info: epoch 190 | train loss = 0.0010424983\n[ Info: epoch 200 | train loss = 0.0009809926","category":"page"},{"location":"api/layers/","page":"Layers","title":"Layers","text":"CurrentModule = NeuralGraphPDE","category":"page"},{"location":"api/layers/#Layers","page":"Layers","title":"Layers","text":"","category":"section"},{"location":"api/layers/#Index","page":"Layers","title":"Index","text":"","category":"section"},{"location":"api/layers/","page":"Layers","title":"Layers","text":"Order = [:type, :function]\nModules = [NeuralGraphPDE]\nPages = [\"layers.md\"]","category":"page"},{"location":"api/layers/#Docs","page":"Layers","title":"Docs","text":"","category":"section"},{"location":"api/layers/","page":"Layers","title":"Layers","text":"Modules = [NeuralGraphPDE]\nPages   = [\"layers.jl\"]\nPrivate = false","category":"page"},{"location":"api/layers/#NeuralGraphPDE.AbstractGNNContainerLayer","page":"Layers","title":"NeuralGraphPDE.AbstractGNNContainerLayer","text":"AbstractGNNContainerLayer <: AbstractExplicitContainerLayer\n\nThis is an abstract type of GNN layers that contains other layers.\n\n\n\n\n\n","category":"type"},{"location":"api/layers/#NeuralGraphPDE.AbstractGNNLayer","page":"Layers","title":"NeuralGraphPDE.AbstractGNNLayer","text":"AbstractGNNLayer <: AbstractExplicitLayer\n\nAn abstract type of graph neural networks. See also AbstractGNNContainerLayer\n\n\n\n\n\n","category":"type"},{"location":"api/layers/#NeuralGraphPDE.ExplicitEdgeConv","page":"Layers","title":"NeuralGraphPDE.ExplicitEdgeConv","text":"ExplicitEdgeConv(ϕ; initialgraph = initialgraph, aggr = mean)\n\nEdge convolutional layer.\n\nmathbfh_i = square_j in N(i) phi(mathbfh_i mathbfh_j mathbfx_j - mathbfx_i)\n\nArguments\n\nϕ: A neural network.\ninitialgraph: GNNGraph or a function that returns a GNNGraph\naggr: Aggregation operator for the incoming messages (e.g. +, *, max, min, and mean).\n\nInputs\n\nh: Trainable node embeddings, NamedTuple or Array.\n\nReturns\n\nNamedTuple or Array that has the same struct with x with different a size of channels.\n\nParameters\n\nParameters of ϕ.\n\nStates\n\ngraph: GNNGraph where graph.ndata.x represents the spatial coordinates of nodes. You can also put other nontrainable node features in graph.ndata with arbitrary keys. They will be concatenated like u.\n\nExamples\n\ns = [1, 1, 2, 3]\nt = [2, 3, 1, 1]\ng = GNNGraph(s, t)\n\nu = randn(4, g.num_nodes)\ng = GNNGraph(g, ndata = (; x = rand(3, g.num_nodes)))\nnn = Dense(4 + 4 + 3 => 5)\nl = ExplicitEdgeConv(nn, initialgraph=g)\n\nrng = Random.default_rng()\nps, st = Lux.setup(rng, l)\n\n\n\n\n\n\n","category":"type"},{"location":"api/layers/#NeuralGraphPDE.ExplicitGCNConv","page":"Layers","title":"NeuralGraphPDE.ExplicitGCNConv","text":"ExplicitGCNConv(in_chs::Int, out_chs::Int, activation = identity;\n                initialgraph = initialgraph, init_weight = glorot_normal,\n                init_bias = zeros32)\n\nSame as the one in GraphNeuralNetworks.jl but with exiplicit paramters.\n\nArguments\n\ninitialgraph: GNNGraph or a function that returns a GNNGraph\n\nExamples\n\n# create data\ns = [1,1,2,3]\nt = [2,3,1,1]\ng = GNNGraph(s, t)\nx = randn(3, g.num_nodes)\n\n# create layer\nl = ExplicitGCNConv(3 => 5, initialgraph = g)\n\n# setup layer\nrng = Random.default_rng()\nRandom.seed!(rng, 0)\n\nps, st = Lux.setup(rng, l)\n\n# forward pass\ny = l(x, ps, st)       # size:  5 × num_nodes\n\n\n\n\n\n","category":"type"},{"location":"api/layers/#NeuralGraphPDE.GNOConv","page":"Layers","title":"NeuralGraphPDE.GNOConv","text":"GNOConv(in_chs => out_chs, ϕ; initialgraph = initialgraph, aggr = mean, bias = true)\n\nConvolutional layer from Neural Operator: Graph Kernel Network for Partial Differential Equations.\n\nbeginaligned\n\tmathbfm_i=Box _jin N(i)phi (mathbfa_imathbfa_jmathbfx_imathbfx_j)mathbfh_j\n\tmathbfh_i=sigma left( mathbfWh_i+mathbfm_i+mathbfb right)\nendaligned\n\nArguments\n\nin_chs: Number of input channels.\nout_chs: Number of output channels.\nϕ: Neural network for the message function. The output size of ϕ should be in_chs * out_chs.\ninitialgraph: GNNGraph or a function that returns a GNNGraph\naggr: Aggregation operator for the incoming messages (e.g. +, *, max, min, and mean).\nbias: Whether to add bias to the output.\n\nInputs\n\nh: Array of the size (in_chs, num_nodes).\n\nReturns\n\nArray of the size (out_chs, num_nodes).\n\nParameters\n\nParameters of ϕ.\nW.\nb.\n\nStates\n\ngraph: GNNGraph. All features are stored in either graph.ndata or graph.edata. They will be concatenated and then fed into ϕ.\n\nExamples\n\ng = rand_graph(10, 6)\n\ng = GNNGraph(g, ndata = (; a = rand(2, 10), x = rand(3, 10)))\nin_chs, out_chs = 5, 7\nh = randn(in_chs, 10)\nϕ = Dense(2 + 2 + 3 + 3 => in_chs * out_chs)\nl = GNOConv(5 => 7, ϕ, initialgraph = g)\n\nrng = Random.default_rng()\nps, st = Lux.setup(rng, l)\n\ny, st = l(h, ps, st)\n\n#edge features\ne = rand(2 + 2 + 3 + 3, 6)\ng = GNNGraph(g, edata = e)\nst = updategraph(st, g)\ny, st = l(h, ps, st)\n\n\n\n\n\n","category":"type"},{"location":"api/layers/#NeuralGraphPDE.MPPDEConv","page":"Layers","title":"NeuralGraphPDE.MPPDEConv","text":"MPPDEConv(ϕ, ψ; initialgraph = initialgraph, aggr = mean, local_features = (:u, :x))\n\nConvolutional layer from Message Passing Neural PDE Solvers, without the temporal bulking trick.\n\nbeginaligned\n\tmathbfm_i=Box _jin N(i)phi (mathbfh_imathbfh_jmathbfu_i-mathbfu_jmathbfx_i-mathbfx_jtheta )\n\tmathbfh_i=psi (mathbfh_imathbfm_itheta )\nendaligned\n\nArguments\n\nϕ: The neural network for the message function.\nψ: The neural network for the update function.\ninitialgraph: GNNGraph or a function that returns a GNNGraph\naggr: Aggregation operator for the incoming messages (e.g. +, *, max, min, and mean).\n\nInputs\n\nh: Trainable node embeddings, Array.\n\nReturns\n\nNamedTuple or Array that has the same struct with x with different a size of channels.\n\nParameters\n\nParameters of ϕ.\nParameters of ψ.\n\nStates\n\ngraph: GNNGraph for which graph.gdata represents the graph level features of the underlying PDE. All features in graph.gdata   should be a matrices of the size (num_feats, num_graphs). You can store u(x) in graph.ndata or u_j-u_i(x_jx_i) in graph.edata.   If g is a batched graph, then currently all graphs need to have the same structure. Note that t is included in graph.gdata   in the original paper.\n\nExamples\n\ng = rand_graph(10, 6)\ng = GNNGraph(g, ndata = (; u = rand(2, 10), x = rand(3, 10)), gdata = (; θ = rand(4)))\nh = randn(5, 10)\nϕ = Dense(5 + 5 + 2 + 3 + 4 => 5)\nψ = Dense(5 + 5 + 4 => 7)\nl = MPPDEConv(ϕ, ψ, initialgraph = g)\nrng = Random.default_rng()\nps, st = Lux.setup(rng, l)\ny, st = l(h, ps, st)\n\n\n\n\n\n","category":"type"},{"location":"api/layers/#NeuralGraphPDE.SpectralConv","page":"Layers","title":"NeuralGraphPDE.SpectralConv","text":"SpectralConv(n::Int)\n\nCompute the Fourier differentiation of a 1D periodic function evenly sampled on [0,2π]. This is only a toy function.\n\n    u_i =frac12 cos left(fracleft(x_i-x_jright) n2right) cot left(fracx_i-x_j2right) u_j\n\nArguments\n\nn: The number of sampled points.\n\nInputs\n\nu: Discret function values on 2jπn, for j=12n.\n\nReturns\n\nThe derivative of u.\n\nParameters\n\nNone.\n\nStates\n\ngraph: A comple graph g of the type GNNGraph, where g.edata.e is x_i-x_j.\n\nExamples\n\njulia> using Lux, Random\n\njulia> s = SpectralConv(100);\n\njulia> rng = Random.default_rng();\njulia> ps, st = Lux.setup(rng, s);\n\njulia> x = LinRange(0, 2π, 101)[2:end];\njulia> s(sin.(x), ps, st)[1] .- cos.(x)\n100-element Vector{Float64}:\n -2.9976021664879227e-15\n  4.440892098500626e-15\n -3.885780586188048e-15\n  4.9960036108132044e-15\n -1.1102230246251565e-15\n -6.328271240363392e-15\n  6.994405055138486e-15\n  5.551115123125783e-16\n  0.0\n  ⋮\n -1.892930256985892e-13\n  1.8640644583456378e-13\n -1.2012613126444194e-13\n  8.526512829121202e-14\n -6.405986852087153e-14\n  4.451994328746878e-14\n -2.631228568361621e-14\n  1.509903313490213e-14\n\njulia> s(cos.(x), ps, st)[1] .+ sin.(x)\n100-element Vector{Float64}:\n  1.9442780718748054e-14\n -3.552713678800501e-14\n  4.246603069191224e-15\n -8.715250743307479e-15\n  1.1934897514720433e-14\n -2.7533531010703882e-14\n  2.6867397195928788e-14\n -1.176836406102666e-14\n  6.5503158452884236e-15\n  ⋮\n  4.048983370807946e-13\n -4.0362158060247566e-13\n  2.742805982336449e-13\n -2.53408405370692e-13\n  2.479405569744131e-13\n -2.366440376988521e-13\n  2.0448920334814602e-13\n -6.064106189943799e-14\n\n\n\n\n\n","category":"type"},{"location":"api/layers/#NeuralGraphPDE.VMHConv","page":"Layers","title":"NeuralGraphPDE.VMHConv","text":"VMHConv(ϕ, γ; initialgraph = initialgraph, aggr = mean)\n\nConvolutional layer from Learning continuous-time PDEs from sparse data with graph neural networks.\n\nbeginaligned\nmathbfm_i = square_j in N(i) phi(mathbfh_i mathbfh_j - mathbfh_i mathbfx_j - mathbfx_i)\nmathbfh_i = gamma(mathbfh_i mathbfm_i)\nendaligned\n\nArguments\n\nϕ: The neural network for the message function.\nγ: The neural network for the update function.\ninitialgraph: GNNGraph or a function that returns a GNNGraph\naggr: Aggregation operator for the incoming messages (e.g. +, *, max, min, and mean).\n\nInputs\n\nh: Trainable node embeddings, NamedTuple or Array.\n\nReturns\n\nNamedTuple or Array that has the same struct with x with different a size of channels.\n\nParameters\n\nParameters of ϕ.\nParameters of γ.\n\nStates\n\ngraph: GNNGraph where graph.ndata.x represents the spatial coordinates of nodes.\n\nExamples\n\ns = [1, 1, 2, 3]\nt = [2, 3, 1, 1]\ng = GNNGraph(s, t)\n\nu = randn(4, g.num_nodes)\ng = GNNGraph(g, ndata = (; x = rand(3, g.num_nodes)))\nϕ = Dense(4 + 4 + 3 => 5)\nγ = Dense(5 + 4 => 7)\nl = VMHConv(ϕ, γ, initialgraph = g)\n\nrng = Random.default_rng()\nps, st = Lux.setup(rng, l)\n\ny, st = l(u, ps, st)\n\n\n\n\n\n","category":"type"},{"location":"tutorials/graph_node/#Neural-Graph-Ordinary-Differential-Equations","page":"Neural Graph Ordinary Differential Equations","title":"Neural Graph Ordinary Differential Equations","text":"","category":"section"},{"location":"tutorials/graph_node/","page":"Neural Graph Ordinary Differential Equations","title":"Neural Graph Ordinary Differential Equations","text":"This tutorial is adapted from SciMLSensitivity, GraphNeuralNetworks, and Lux.","category":"page"},{"location":"tutorials/graph_node/#Load-the-packages","page":"Neural Graph Ordinary Differential Equations","title":"Load the packages","text":"","category":"section"},{"location":"tutorials/graph_node/","page":"Neural Graph Ordinary Differential Equations","title":"Neural Graph Ordinary Differential Equations","text":"using GraphNeuralNetworks, NeuralGraphPDE, DifferentialEquations\nusing Lux, NNlib, Optimisers, Zygote, Random\nusing ComponentArrays, OneHotArrays\nusing SciMLSensitivity\nusing Statistics: mean\nusing MLDatasets: Cora\nusing CUDA\nCUDA.allowscalar(false)\ndevice = CUDA.functional() ? gpu : cpu","category":"page"},{"location":"tutorials/graph_node/#Load-data","page":"Neural Graph Ordinary Differential Equations","title":"Load data","text":"","category":"section"},{"location":"tutorials/graph_node/","page":"Neural Graph Ordinary Differential Equations","title":"Neural Graph Ordinary Differential Equations","text":"dataset = Cora();\nclasses = dataset.metadata[\"classes\"]\ng = device(mldataset2gnngraph(dataset))\nX = g.ndata.features\ny = onehotbatch(g.ndata.targets, classes) # a dense matrix is not the optimal\n(; train_mask, val_mask, test_mask) = g.ndata\nytrain = y[:, train_mask]","category":"page"},{"location":"tutorials/graph_node/#Model-and-data-configuration","page":"Neural Graph Ordinary Differential Equations","title":"Model and data configuration","text":"","category":"section"},{"location":"tutorials/graph_node/","page":"Neural Graph Ordinary Differential Equations","title":"Neural Graph Ordinary Differential Equations","text":"nin = size(X, 1)\nnhidden = 16\nnout = length(classes)\nepochs = 40","category":"page"},{"location":"tutorials/graph_node/#Define-Neural-ODE","page":"Neural Graph Ordinary Differential Equations","title":"Define Neural ODE","text":"","category":"section"},{"location":"tutorials/graph_node/","page":"Neural Graph Ordinary Differential Equations","title":"Neural Graph Ordinary Differential Equations","text":"struct NeuralODE{M <: Lux.AbstractExplicitLayer, So, Se, T, K} <:\n       Lux.AbstractExplicitContainerLayer{(:model,)}\n    model::M\n    solver::So\n    sensealg::Se\n    tspan::T\n    kwargs::K\nend\n\nfunction NeuralODE(model::Lux.AbstractExplicitLayer; solver=Tsit5(),\n                   sensealg=InterpolatingAdjoint(; autojacvec=ZygoteVJP()),\n                   tspan=(0.0f0, 1.0f0), kwargs...)\n    return NeuralODE(model, solver, sensealg, tspan, kwargs)\nend\n\nfunction (n::NeuralODE)(x, ps, st)\n    function dudt(u, p, t)\n        u_, st = n.model(u, p, st)\n        return u_\n    end\n    prob = ODEProblem{false}(ODEFunction{false}(dudt), x, n.tspan, ps)\n    return solve(prob, n.solver; sensealg=n.sensealg, n.kwargs...), st\nend\n\nfunction diffeqsol_to_array(x::ODESolution{T, N, <:AbstractVector{<:CuArray}}) where {T, N}\n    return dropdims(gpu(x); dims=3)\nend\ndiffeqsol_to_array(x::ODESolution) = dropdims(Array(x); dims=3)","category":"page"},{"location":"tutorials/graph_node/#Create-and-initialize-the-Neural-Graph-ODE-layer","page":"Neural Graph Ordinary Differential Equations","title":"Create and initialize the Neural Graph ODE layer","text":"","category":"section"},{"location":"tutorials/graph_node/","page":"Neural Graph Ordinary Differential Equations","title":"Neural Graph Ordinary Differential Equations","text":"function create_model()\n    node_chain = Chain(ExplicitGCNConv(nhidden => nhidden, relu),\n                       ExplicitGCNConv(nhidden => nhidden, relu))\n\n    node = NeuralODE(node_chain; save_everystep=false, reltol=1e-3, abstol=1e-3,\n                     save_start=false)\n\n    model = Chain(ExplicitGCNConv(nin => nhidden, relu), node, diffeqsol_to_array,\n                  Dense(nhidden, nout))\n\n    rng = Random.default_rng()\n    Random.seed!(rng, 0)\n\n    ps, st = Lux.setup(rng, model)\n    ps = ComponentArray(ps) |> device\n    st = updategraph(st, g) |> device\n\n    return model, ps, st\nend","category":"page"},{"location":"tutorials/graph_node/#Define-the-loss-function","page":"Neural Graph Ordinary Differential Equations","title":"Define the loss function","text":"","category":"section"},{"location":"tutorials/graph_node/","page":"Neural Graph Ordinary Differential Equations","title":"Neural Graph Ordinary Differential Equations","text":"logitcrossentropy(ŷ, y) = mean(-sum(y .* logsoftmax(ŷ); dims=1))\n\nfunction loss(x, y, mask, model, ps, st)\n    ŷ, st = model(x, ps, st)\n    return logitcrossentropy(ŷ[:, mask], y), st\nend\n\nfunction eval_loss_accuracy(X, y, mask, model, ps, st)\n    ŷ, _ = model(X, ps, st)\n    l = logitcrossentropy(ŷ[:, mask], y[:, mask])\n    acc = mean(onecold(ŷ[:, mask]) .== onecold(y[:, mask]))\n    return (loss=round(l; digits=4), acc=round(acc * 100; digits=2))\nend","category":"page"},{"location":"tutorials/graph_node/#Train-the-model","page":"Neural Graph Ordinary Differential Equations","title":"Train the model","text":"","category":"section"},{"location":"tutorials/graph_node/","page":"Neural Graph Ordinary Differential Equations","title":"Neural Graph Ordinary Differential Equations","text":"function train()\n    model, ps, st = create_model()\n\n    # Optimizer\n    opt = Optimisers.Adam(0.01f0)\n    st_opt = Optimisers.setup(opt, ps)\n\n    # Training Loop\n    for epoch in 1:epochs\n        (l, st), back = pullback(p -> loss(X, ytrain, train_mask, model, p, st), ps)\n        gs = back((one(l), nothing))[1]\n        st_opt, ps = Optimisers.update(st_opt, ps, gs)\n        @show eval_loss_accuracy(X, y, val_mask, model, ps, st)\n    end\nend\n\ntrain()","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = NeuralGraphPDE","category":"page"},{"location":"#NeuralGraphPDE","page":"Home","title":"NeuralGraphPDE","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for NeuralGraphPDE.","category":"page"},{"location":"#Features","page":"Home","title":"Features","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Layers and graphs are coupled and decoupled at the same time: You can bind a graph to a layer at initialization, but the graph is stored in st, not in the layer. They are decoupled in the sense that you can easily update or change the graph by changing st:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using NeuralGraphPDE, GraphNeuralNetworks, Random, Lux\ng = rand_graph(5, 4; bidirected=false)\nx = randn(3, g.num_nodes)\n\n# create layer\nl = ExplicitGCNConv(3 => 5; initialgraph=g)\n\n# setup layer\nrng = Random.default_rng()\nRandom.seed!(rng, 0)\n\nps, st = Lux.setup(rng, l)\n\n# forward pass\ny, st = l(x, ps, st)    # you don't need to feed a graph explicitly\n\n#change the graph\nnew_g = rand_graph(5, 7; bidirected=false)\nst = updategraph(st, new_g)\n\ny, st = l(x, ps, st)","category":"page"},{"location":"","page":"Home","title":"Home","text":"You can omit the keyword argument initalgraph at initialization, and then call updategraph on st to put the graph in it. All gnn layers can work smoothly with other layers defined by Lux.","category":"page"},{"location":"","page":"Home","title":"Home","text":"g = rand_graph(5, 4; bidirected=false)\nx = randn(3, g.num_nodes)\n\nmodel = Chain(Dense(3 => 5), ExplicitGCNConv(5 => 5), ExplicitGCNConv(5 => 3))  # you don't need to use `g` for initalization\n# setup layer\nrng = Random.default_rng()\nRandom.seed!(rng, 0)\n\nps, st = Lux.setup(rng, model) # the default graph is empty\nst = updategraph(st, g) # put the graph in st\n\n# forward pass\ny, st = model(x, ps, st)","category":"page"},{"location":"","page":"Home","title":"Home","text":"An unified interface for graph level tasks. As pointed out here, GNNs are difficult to work well with other neural networks when the input graph is changing. This will not be an issue here. You have an unified interface y, st = model(x, ps, st). There are several benefits to doing so:\nEach layer can take in different graphs.\nYou can modify the graph inside a layer and return it.\nMultigraphs. A layer can take in any number of graphs in st.\nTrainable node embeddings and nontrainable features are seperately stored in x and st.graph.","category":"page"},{"location":"#Limitations","page":"Home","title":"Limitations","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"We assume all graphs have the same structure.\nThe input must be a matrix or a named tuple of matrices.\n","category":"page"}]
}
