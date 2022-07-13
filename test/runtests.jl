using NeuralGraphPDE
using GraphNeuralNetworks
using Random
using Lux
using Lux: parameterlength
using Test
import Flux: batch, unbatch
using SafeTestsets

@testset "layers" begin
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    s = [1, 1, 2, 3]
    t = [2, 3, 1, 1]
    g = GNNGraph(s, t)
    T = Float32
    @testset "conv" begin
        @testset "gcn" begin
            x = randn(T, 3, g.num_nodes)
            l = ExplicitGCNConv(3 => 5, initialgraph = g)

            ps, st = Lux.setup(rng, l)
            @test st == (graph = g,)
            y, st = l(x, ps, st)
            @test size(y) == (5, g.num_nodes)
            @test st == (graph = g,)
        end

        @testset "edge" begin
            u = randn(T, 4, g.num_nodes)
            gh = GNNGraph(g, ndata = (; x = rand(T, 3, g.num_nodes)))
            nn = Dense(4 + 4 + 3 => 5)
            l = ExplicitEdgeConv(nn, initialgraph = gh)

            ps, st = Lux.setup(rng, l)
            @test st == (ϕ = NamedTuple(), graph = gh)
            y, _ = l(u, ps, st)
            @test size(y) == (5, gh.num_nodes)
        end

        @testset "VMH" begin
            u = randn(T, 4, g.num_nodes)
            gh = GNNGraph(g, ndata = (; x = rand(T, 3, g.num_nodes)))

            ϕ = Dense(4 + 4 + 3 => 5)
            γ = Dense(5 + 4 => 7)
            l = VMHConv(ϕ, γ, initialgraph = gh)

            rng = Random.default_rng()
            ps, st = Lux.setup(rng, l)

            ps, st = Lux.setup(rng, l)
            @test st == (ϕ = NamedTuple(), γ = NamedTuple(), graph = gh)
            y, _ = l(u, ps, st)
            @test size(y) == (7, gh.num_nodes)
        end

        @testset "MPPDE" begin
            @testset "With theta" begin
                gh = GNNGraph(g,
                              ndata = (; u = rand(2, g.num_nodes),
                                       x = rand(3, g.num_nodes)),
                              gdata = (; θ = rand(4)))

                h = randn(T, 5, g.num_nodes)
                ϕ = Dense(5 + 5 + 2 + 3 + 4 => 5)
                ψ = Dense(5 + 5 + 4 => 7)
                l = MPPDEConv(ϕ, ψ, initialgraph = gh)

                ps, st = Lux.setup(rng, l)
                @test st.graph == gh

                y, st = l(h, ps, st)

                @test st.graph == gh
                @test size(y) == (7, g.num_nodes)
            end

            @testset "edge features" begin
                gh = GNNGraph(g,
                              edata = (u = rand(2, g.num_edges),
                                       x = rand(3, g.num_edges)),
                              gdata = (; θ = rand(4)))

                h = randn(T, 5, g.num_nodes)
                ϕ = Dense(5 + 5 + 2 + 3 + 4 => 5)
                ψ = Dense(5 + 5 + 4 => 7)
                l = MPPDEConv(ϕ, ψ, initialgraph = gh)

                ps, st = Lux.setup(rng, l)
                y, st = l(h, ps, st)
                @test size(y) == (7, g.num_nodes)
            end

            @testset "batched graph" begin
                gh = GNNGraph(g,
                              ndata = (u = rand(2, g.num_nodes),
                                       x = rand(3, g.num_nodes)),
                              gdata = (; θ = rand(4)))
                gh = batch([gh, copy(gh)])

                h = randn(T, 5, gh.num_nodes)
                ϕ = Dense(5 + 5 + 2 + 3 + 4 => 5)
                ψ = Dense(5 + 5 + 4 => 7)
                l = MPPDEConv(ϕ, ψ, initialgraph = gh)

                ps, st = Lux.setup(rng, l)
                y, st = l(h, ps, st)
                @test size(y) == (7, gh.num_nodes)

                h = randn(T, 5, g.num_nodes, 2)

                ps, st = Lux.setup(rng, l)
                y, st = l(h, ps, st)
                @test size(y) == (7, gh.num_nodes)
            end

            @testset "Without theta" begin
                gh = GNNGraph(g,
                              ndata = (; u = rand(2, g.num_nodes),
                                       x = rand(3, g.num_nodes)))

                h = randn(T, 5, gh.num_nodes)
                ϕ = Dense(5 + 5 + 2 + 3 => 5)
                ψ = Dense(5 + 5 => 7)
                l = MPPDEConv(ϕ, ψ, initialgraph = gh)

                rng = Random.default_rng()
                ps, st = Lux.setup(rng, l)
                @test st.graph == gh

                y, st = l(h, ps, st)
                @test st.graph == gh

                @test size(y) == (7, gh.num_nodes)
            end
        end

        @testset "GNOConv" begin
            gh = rand_graph(10, 6)

            gh = GNNGraph(gh,
                          ndata = (; a = rand(2, gh.num_nodes), x = rand(3, gh.num_nodes)))
            in_chs, out_chs = 5, 7
            h = randn(in_chs, gh.num_nodes)
            ϕ = Dense(2 + 2 + 3 + 3 => in_chs * out_chs)
            l = GNOConv(in_chs => out_chs, ϕ, initialgraph = gh)

            rng = Random.default_rng()
            ps, st = Lux.setup(rng, l)

            y, st = l(h, ps, st)
            @test size(y) == (out_chs, gh.num_nodes)

            l = GNOConv(in_chs => out_chs, ϕ, initialgraph = gh)
            rng = Random.default_rng()
            ps, st = Lux.setup(rng, l)

            y, st = l(h, ps, st)
            @test size(y) == (out_chs, gh.num_nodes)

            gh = GNNGraph(gh, ndata = NamedTuple(),
                          edata = rand(2 + 2 + 3 + 3, gh.num_edges))

            st = updategraph(st, gh)
            y, st = l(h, ps, st)

            @test size(y) == (out_chs, gh.num_nodes)
        end
    end
end

@testset "utilities" begin @testset "updategraph" begin
    g = rand_graph(5, 4, bidirected = false)
    x = randn(3, g.num_nodes)

    l = ExplicitGCNConv(3 => 5, initialgraph = g)

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    ps, st = Lux.setup(rng, l)
    new_g = rand_graph(5, 7, bidirected = false)
    new_st = updategraph(st, new_g)
    @test new_st.graph === new_g

    model = Chain(ExplicitGCNConv(3 => 5, initialgraph = g),
                  ExplicitGCNConv(5 => 5, initialgraph = g))
    ps, st = Lux.setup(rng, model)
    new_st = updategraph(st, new_g)
    @test new_st.layer_1.graph === new_st.layer_2.graph === new_g
end end
