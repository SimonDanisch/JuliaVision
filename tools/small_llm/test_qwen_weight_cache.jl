using Test, JSON3
import DNNKernels, Mantle
function test_qwen_weight_cache(device)
    Q=SmallLLMChecks.Qwen35Experiment;C=Q.QwenWeightCache;DK=DNNKernels
    mktempdir() do directory
        checkpoint=joinpath(directory,"checkpoint");mkdir(checkpoint)
        write(joinpath(checkpoint,"config.json"),"{}")
        write(joinpath(checkpoint,"model.safetensors.index.json"),JSON3.write((;weight_map=Dict("weight"=>"weights.bin"))))
        source=joinpath(checkpoint,"weights.bin");write(source,UInt8[1,2,3])
        base=joinpath(directory,"cache");opened=C.open_cache(checkpoint,base,device);cache=opened.cache
        owned=Any[];weight=Mantle.Buffer(device,reshape(Float32[sin(i/9) for i in 1:128*256],128,256))
        quantized=Any[]
        bytes(a)=reinterpret(UInt8,vec(Array(a)))
        try
            @testset "source-fingerprinted existing quantcache adapter" begin
                for (key,quantize,kind) in (("matrix",DK.quantizeq4k,:q4k),("head",DK.quantizeq6k,:q6k))
                    packed=quantize(device,weight);push!(quantized,packed)
                    DK.cachestore!(cache,key,(128,256),packed)
                end
                DK.closecache!(cache)
                reopened=C.open_cache(checkpoint,base,device);cache=reopened.cache
                @test reopened.fingerprint==opened.fingerprint
                for (key,packed,kind) in zip(("matrix","head"),quantized,(:q4k,:q6k))
                    @test DK.iscached(cache,key,(128,256),kind)
                    @test !DK.iscached(cache,key,(129,256),kind)
                    restored=C.load_owned!(device,cache,key,owned)
                    @test size(restored)==size(packed)
                    @test all(bytes(a)==bytes(b) for (a,b) in zip(DK.cachearrays(packed),DK.cachearrays(restored)))
                end
                DK.closecache!(cache)
                write(source,UInt8[1,2,4]) # same shape, size and filename
                changed=C.open_cache(checkpoint,base,device)
                @test changed.fingerprint!=opened.fingerprint
                @test isempty(changed.cache.entries)
                DK.closecache!(changed.cache)
                write(source,UInt8[1,2,3])
                write(joinpath(opened.directory,"source.json"),JSON3.write((;fingerprint="wrong")))
                @test_throws ArgumentError C.open_cache(checkpoint,base,device)
            end
        finally
            DK.closecache!(cache)
            foreach(Mantle.free!,owned)
            foreach(q->foreach(Mantle.free!,DK.cachearrays(q)),quantized)
            Mantle.free!(weight)
        end
    end
    (;passed=true,scope="tiny Q4/Q6 cache byte parity and stale-source rejection; full-model cache not yet qualified")
end
