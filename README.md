# NeuralGraphPDE

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MilkshakeForReal.github.io/NeuralGraphPDE.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MilkshakeForReal.github.io/NeuralGraphPDE.jl/dev/)
[![Build Status](https://github.com/MilkshakeForReal/NeuralGraphPDE.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MilkshakeForReal/NeuralGraphPDE.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MilkshakeForReal/NeuralGraphPDE.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MilkshakeForReal/NeuralGraphPDE.jl)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

This package is based on [GraphNeuralNetwork.jl](https://github.com/CarloLucibello/GraphNeuralNetworks.jl) and [Lux.jl](https://github.com/avik-pal/Lux.jl) to produce explicit GNN layers.

The goal is to extend Neural (Graph) ODE to Neural Graph PDE (WIP). Be aware of potential breaking changes.

This library will focus on (only a few) **equivariant** GNNs. This is not a general GNN package.

## References

 1. Iakovlev V, Heinonen M, Lähdesmäki H. Learning continuous-time PDEs from sparse data with graph neural networks[J]. arXiv preprint arXiv:2006.08956, 2020.
 2. Poli M, Massaroli S, Park J, et al. Graph neural ordinary differential equations[J]. arXiv preprint arXiv:1911.07532, 2019.
