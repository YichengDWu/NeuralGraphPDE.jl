using NeuralGraphPDE
using GraphNeuralNetworks
using Random
using Lux
using Test

@testset "layers" begin
    @testset "conv" begin
        @testset "gcn" begin
            s = [1,1,2,3]
            t = [2,3,1,1]
            g = GNNGraph(s, t)
            x = randn(3, g.num_nodes)
            l = ExplicitGCNConv(3 => 5) 

            rng = Random.default_rng()
            Random.seed!(rng, 0)

            ps, st = Lux.setup(rng, l)
            @test st == (;)
            y,st = l(g, x, ps, st)   
            @test size(y) == (5, g.num_nodes)
            @test st == (;)
        end
    end
end
