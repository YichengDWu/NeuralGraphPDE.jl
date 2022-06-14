using NeuralGraphPDE
using Test

@testset "NeuralGraphPDE.jl" begin
    @test foo(0)<1E-4
end
