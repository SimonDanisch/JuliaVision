using Test, HorizonRunner, DNNKernels, JSON3, SHA

@testset "Horizon artifact shard index" begin
    source = DNNKernels.readsafetensors(joinpath(HorizonRunner.tinydir(), "weights.safetensors"))
    digests = Dict(k => bytes2hex(sha256(reinterpret(UInt8, vec(v)))) for (k,v) in source)
    part = (; artifact="horizon32b-tiny", file="weights.safetensors")
    mktempdir() do dir
        function index(; format=1, shards=[part], tensor_sha256=digests)
            open(joinpath(dir, "weights.index.json"), "w") do io
                JSON3.write(io, (; format, shards, tensor_sha256))
            end
        end
        index()
        loaded = HorizonRunner.horizon32bweights(; dir)
        @test keys(loaded) == keys(source)
        @test all(isequal(loaded[k], source[k]) for k in keys(source))
        index(shards=[part, part])
        @test_throws ErrorException HorizonRunner.horizon32bweights(; dir)
        index(tensor_sha256=Dict("missing.weight" => repeat("0", 64)))
        @test_throws ErrorException HorizonRunner.horizon32bweights(; dir)
        index(format=2)
        @test_throws ErrorException HorizonRunner.horizon32bweights(; dir)
    end
end
