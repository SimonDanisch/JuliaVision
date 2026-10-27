include("qwen36_native_cache.jl")
using Test,SHA,JSON3

function test_qwen36_native_cache(R,device)
    C=Qwen36NativeCache
    fixture=joinpath(@__DIR__,"qwen36_moe_tiny_reference")
    source=bytes2hex(open(sha256,joinpath(fixture,"model.safetensors")))
    @testset "Native MoE cache policy and BF16 conversion" begin
        @test C.native_name("model.language_model.layers.3.mlp.experts.down_proj")=="layers.3.mlp.experts.down_proj"
        @test C.native_name("model.layers.0.input_layernorm.weight")=="layers.0.input_layernorm.weight"
        @test C.native_name("lm_head.weight")=="lm_head"
        @test C.native_name("model.visual.blocks.0.weight")===nothing
        @test C.policy("layers.0.mlp.gate.weight",(2048,256))=="dense"
        @test C.policy("layers.0.mlp.experts.gate_up_proj",(2048,1024,256))=="experts_q4k"
        @test C.policy("lm_head",(2048,248320))=="q6k"
        @test C.policy("embed_tokens.weight",(2048,248320))=="raw"
        @test_throws DimensionMismatch C.policy("layers.0.mlp.experts.down_proj",(255,512,4))
        @test C.floating(UInt16[0x3f80,0xbf80,0x3f00],"BF16")==Float32[1,-1,0.5]
        @test_throws ArgumentError C.floating(UInt16[1],"U16")
    end
    rows=Any[]
    mktempdir() do temp
        cache=C.open_cache(fixture,joinpath(temp,"owned"),device)
        @testset "Native MoE tensor receipts and reload identity" begin
            @test_throws Exception C.finalize!(cache)
            @test_throws Exception C.pack_shard!(R,cache,fixture,"model.safetensors",device;expected_sha256="wrong")
            count=C.pack_shard!(R,cache,fixture,"model.safetensors",device;expected_sha256=source)
            @test count==length(C.expected_names(cache.config))
            @test C.finalize!(cache)==count
            @test C.pack_shard!(R,cache,fixture,"model.safetensors",device;expected_sha256=source)==count
            @test C.open_cache(fixture,cache.directory,device).fingerprint==cache.fingerprint
            unowned=joinpath(temp,"unowned");mkdir(unowned);write(joinpath(unowned,"keep"),"untouched")
            @test_throws Exception C.open_cache(fixture,unowned,device)
            @test read(joinpath(unowned,"keep"),String)=="untouched"
            model=C.loadmodel(R,cache,device);reloaded=nothing;s=nothing;t=nothing
            try
                reloaded=C.loadmodel(R,cache,device)
                @test model.weights["lm_head"] isa C.DK.Q6KMatrix
                @test model.weights["layers.0.mlp.gate.weight"] isa R.Q.DenseWeight
                @test model.weights["layers.0.mlp.experts.gate_up_proj"] isa R.M.Q4Experts
                @test model.weights["lm_head"]!==model.weights["embed_tokens.weight"]
                s=R.session(model;capacity=32);t=R.session(reloaded;capacity=32)
                reference=JSON3.read(read(joinpath(fixture,"reference.json"),String))
                for row in reference.rows
                    before=Mantle.submissions(device)
                    R.step!(s,Int(row.token));R.step!(t,Int(row.token));Mantle.waitidle(device)
                    @test Mantle.submissions(device)-before==2
                    logits=Array(s.logits);other=Array(t.logits);want=Float32.(row.logits)
                    @test all(isfinite,logits)
                    @test reinterpret(UInt32,logits)==reinterpret(UInt32,other)
                    @test maximum(abs,logits-want)<0.15f0
                    push!(rows,(;token=row.token,max_logit_error=maximum(abs,logits-want),argmax_match=argmax(logits)==argmax(want)))
                end
            finally
                s===nothing||close(s);t===nothing||close(t)
                close(model);reloaded===nothing||close(reloaded)
            end
            name="norm.weight";bin=joinpath(cache.directory,C.stem(name)*".bin")
            open(bin,"r+") do io
                first=read(io,UInt8);seekstart(io);write(io,first⊻0x01)
            end
            @test_throws Exception C.validate_entry(cache,name)
            @test_throws Exception C.loadmodel(R,cache,device)
        end
    end
    rows
end
