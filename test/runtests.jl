using NeuralGraphPDE
using Test

@testset "layers" begin
    @testset "conv" begin
        @test "gcn" begin
            s = [1,1,2,3]
            t = [2,3,1,1]
            g = GNNGraph(s, t)
            x = randn(3, g.num_nodes)
            l = ExplicitGCNConv(3 => 5) 

            # setup layer
            rng = Random.default_rng()
            Random.seed!(rng, 0)

            ps, st = Lux.setup(rng, l)

            # forward pass
            y = l(g, x, ps, st)   
            @test size(y) == (5, g.num_nodes)
        end
    end
end
