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
            u = randn(4, g.num_nodes)
            g = GNNGraph(g, ndata=(; x=rand(3, g.num_nodes)))
            nn = Dense(4 + 4 + 3 => 5)
            l = ExplicitEdgeConv(nn, initialgraph=g)

            ps, st = Lux.setup(rng, l)
            @test st == (ϕ=NamedTuple(), graph=g)
            y, _ = l(u, ps, st)
            @test size(y) == (5, g.num_nodes)
        end
    end
end

@testset "utilities" begin @testset "updategraph" begin
    g = rand_graph(5, 4, bidirected=false)
    x = randn(3, g.num_nodes)

    l = ExplicitGCNConv(3 => 5, initialgraph=g)

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    ps, st = Lux.setup(rng, l)
    new_g = rand_graph(5, 7, bidirected=false)
    new_st = updategraph(st, new_g)
    @test new_st.graph === new_g

    model = Chain(ExplicitGCNConv(3 => 5, initialgraph=g),
                  ExplicitGCNConv(5 => 5, initialgraph=g))
    ps, st = Lux.setup(rng, model)
    new_st = updategraph(st, new_g)
    @test new_st.layer_1.graph === new_st.layer_2.graph === new_g
end end
