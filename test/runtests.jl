using NeuralGraphPDE
using GraphNeuralNetworks
using Random
using Lux
using Lux: parameterlength
using Test

@testset "layers" begin
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    s = [1,1,2,3]
    t = [2,3,1,1]
    g = GNNGraph(s, t)
    @testset "basic" begin
        @testset "Chain" begin
            x = randn(3, g.num_nodes)

            model = Chain(ExplicitGCNConv(3 => 5),
                          ExplicitGCNConv(5 => 5))

            ps, st = Lux.setup(rng, model)
            y, st = model(g,x,ps,st)
            @test size(y) == (5, g.num_nodes)
            y2, _ = model.layers[1](g, x, ps.layer_1, st)
            y3, _ = model.layers[2](g, y2, ps.layer_2, st)
            @test y == y3
        end
        @testset "WithStaticGraph" begin
            x = randn(3, g.num_nodes)

            model = ExplicitGCNConv(3 => 5) 
            wg = WithStaticGraph(model, g)
            @test parameterlength(wg) == parameterlength(model)

            ps, st = Lux.setup(rng, model)
            @test model(g, x, ps, st) == wg(x, ps, st)

            # With Chain
            model = Chain(ExplicitGCNConv(3 => 5),
                          ExplicitGCNConv(5 => 5))
            wg = WithStaticGraph(model, g)
            @test parameterlength(wg) == parameterlength(model)

            ps, st = Lux.setup(rng, model)
            @test model(g, x, ps, st) == wg(x, ps, st)
        end
    end
    @testset "conv" begin
        @testset "gcn" begin
            x = randn(3, g.num_nodes)
            l = ExplicitGCNConv(3 => 5) 

            ps, st = Lux.setup(rng, l)
            @test st == (;)
            y, st = l(g, x, ps, st)   
            @test size(y) == (5, g.num_nodes)
            @test st == (;)
        end

        @testset "edge" begin
            x = randn(3, g.num_nodes)
            u = rand(4, g.num_nodes)
            l = ExplicitEdgeConv(Dense(4+4+3 => 5)) 

            ps, st = Lux.setup(rng, l)
            y, _ = l(g, (u=u,x=x), ps, st)
            @test size(y) == (5, g.num_nodes)

            e = x[:,s]-x[:,t]
            y2, _ = l(g, u, e, ps, st)
            @test y2 == y
        end
    end
end
