using NeuralGraphPDE
using GraphNeuralNetworks
using Random
using Lux
using Lux: parameterlength
using Test

@testset "layers" begin
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    s = [1, 1, 2, 3]
    t = [2, 3, 1, 1]
    g = GNNGraph(s, t)
    @testset "conv" begin
        @testset "gcn" begin
            x = randn(3, g.num_nodes)
            l = ExplicitGCNConv(3 => 5, initialgraph=g)

            ps, st = Lux.setup(rng, l)
            @test st == (graph=g,)
            y, st = l(x, ps, st)
            @test size(y) == (5, g.num_nodes)
            @test st == (graph=g,)
        end

        @testset "edge" begin
            x = randn(3, g.num_nodes)
            u = rand(4, g.num_nodes)
            nn = Dense(4 + 4 + 3 => 5)
            l = ExplicitEdgeConv(nn, initialgraph=g)

            ps, st = Lux.setup(rng, l)
            @test st == (ϕ=(;), graph=g)
            y, _ = l((u=u, x=x), ps, st)
            @test size(y) == (5, g.num_nodes)

            e = x[:, s] - x[:, t]
            y2, _ = l(u, e, ps, st)
            @test y2 == y
        end
    end
end
