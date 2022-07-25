# NeuralGraphPDE

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MilkshakeForReal.github.io/NeuralGraphPDE.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MilkshakeForReal.github.io/NeuralGraphPDE.jl/dev/)
[![Build Status](https://github.com/MilkshakeForReal/NeuralGraphPDE.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MilkshakeForReal/NeuralGraphPDE.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MilkshakeForReal/NeuralGraphPDE.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MilkshakeForReal/NeuralGraphPDE.jl)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

This package is based on [GraphNeuralNetwork.jl](https://github.com/CarloLucibello/GraphNeuralNetworks.jl) and [Lux.jl](https://github.com/avik-pal/Lux.jl).

The goal is to extend Neural (Graph) ODE to Neural Graph PDE (WIP). Be aware of potential breaking changes.

This library will focus on (only a few equivariant) GNNs related to PDEs. This is not a general GNN package. Although you can write any custom convolutional layers if you want.

## References

 1. Iakovlev V, Heinonen M, Lähdesmäki H. Learning continuous-time PDEs from sparse data with graph neural networks[J]. arXiv preprint arXiv:2006.08956, 2020.
 2. Poli M, Massaroli S, Rabideau C M, et al. Continuous-depth neural models for dynamic graph prediction[J]. arXiv preprint arXiv:2106.11581, 2021.
 3. Chamberlain B, Rowbottom J, Gorinova M I, et al. Grand: Graph neural diffusion[C]. International Conference on Machine Learning. PMLR, 2021: 1407-1418.
 4. Brandstetter J, Worrall D, Welling M. Message passing neural PDE solvers[J]. arXiv preprint arXiv:2202.03376, 2022.
 5. Li Z, Kovachki N, Azizzadenesheli K, et al. Neural operator: Graph kernel network for partial differential equations[J]. arXiv preprint arXiv:2003.03485, 2020.
