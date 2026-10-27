# Qwen3.6 prompt chunks against sequential decode, and the chunk kernels against
# their single-column forms. Every comparison is bitwise: a chunk up to the
# shared emitter's GEMV width runs the decode arithmetic column by column, which
# is what lets the text harness reproduce one-token-at-a-time ingestion.
isdefined(@__MODULE__,:Qwen36MoEExperiment) || include("qwen36_moe_model.jl")
isdefined(@__MODULE__,:Qwen36GGUF) || include("qwen36_gguf.jl")
isdefined(@__MODULE__,:load_qwen36_moe_fixture) || include("qwen36_moe_fixture_loader.jl")
using Test,Random,SHA,JSON3
import Mantle

bitsof(a)=reinterpret(UInt32,vec(Array(a)))

function test_qwen36_prefill(device;directory=joinpath(@__DIR__,"qwen36_moe_tiny_reference"))
    R=Qwen36MoEExperiment
    ids=rand(MersenneTwister(36),0:255,48)
    @testset "Qwen3.6 prompt chunks reproduce sequential decode" begin
        model=load_qwen36_moe_fixture(R,directory,device)
        try
            # A session defaults to its checkpoint's recurrent head order.
            @test R.recurrent_layout(model)===:native
            @test R.recurrent_layout(R.Model(device,model.config,Dict{String,Any}(),Any[],:gguf_q4ks))===:tiled
            for optimizations in ((),R.DEFAULT_OPTIMIZATIONS),attention in (:grouped,:split),
                (count,chunk,prefix) in ((17,8,0),(19,8,3),(35,16,0))
                batched=nothing;reference=nothing
                try
                    batched=R.session(model;capacity=64,debug_layers=true,attention,optimizations)
                    reference=R.session(model;capacity=64,debug_layers=true,attention,optimizations)
                    for t in ids[1:prefix];R.step!(batched,t);R.step!(reference,t);end
                    prompt=ids[prefix+1:prefix+count]
                    R.prefill!(batched,prompt;chunk)
                    foreach(t->R.step!(reference,t),prompt)
                    Mantle.waitidle(device)
                    @test batched.position==reference.position==prefix+count
                    @test bitsof(batched.logits)==bitsof(reference.logits)
                    for il in keys(reference.recurrent),(got,want) in zip(batched.recurrent[il],reference.recurrent[il])
                        @test bitsof(got)==bitsof(want)
                    end
                    for il in keys(reference.kv),(got,want) in zip(batched.kv[il],reference.kv[il])
                        @test bitsof(got)==bitsof(want)
                    end
                    for (got,want) in zip(batched.layer_outputs,reference.layer_outputs)
                        @test bitsof(got)==bitsof(want)
                    end
                    first=batched.position
                    R.decode!(batched,4);R.decode!(reference,4);Mantle.waitidle(device)
                    @test Array(batched.predictions)[first:first+3]==Array(reference.predictions)[first:first+3]
                    @test bitsof(batched.logits)==bitsof(reference.logits)
                finally
                    batched===nothing||close(batched);reference===nothing||close(reference)
                end
            end
            s=R.session(model;capacity=16)
            try
                @test_throws ArgumentError R.prefill!(s,ids[1:4];chunk=0)
                @test_throws ArgumentError R.prefill!(s,[-1,0])
                @test_throws ArgumentError R.prefill!(s,ids[1:17])
            finally
                close(s)
            end
        finally
            close(model)
        end
    end
    nothing
end

function test_qwen36_chunk_kernels(device)
    G=Qwen36GGUF;R=G.R;Q=G.Q;DK=G.DK
    folder=joinpath(@__DIR__,"gguf_q5_reference")
    reference=JSON3.read(read(joinpath(folder,"reference.json"),String))
    @assert bytes2hex(open(sha256,joinpath(folder,"reference.safetensors")))==reference.tensor_sha256
    tensors=DK.readsafetensors(joinpath(folder,"reference.safetensors"))
    rng=MersenneTwister(5);owned=Any[];plan=nothing
    upload(x)=begin b=Mantle.Buffer(device,x);push!(owned,b);b;end
    function run(graph)
        plan=Mantle.record!(Mantle.Plan(graph;budget=Inf));Mantle.run!(plan);Mantle.waitidle(device)
        Mantle.free!(plan);plan=nothing
    end
    try
        @testset "Q5 product over columns is the single-column product" begin
            for k in (256,512,1024),n in (1,5)
                rows=17
                A=G.Q5KMatrix(upload(reinterpret(DK.UVec4,vec(tensors["q5.$k.bytes"]))),rows,k)
                x=randn(rng,Float32,k,n)
                out=upload(fill(NaN32,rows,n));columns=[upload(fill(NaN32,rows)) for _ in 1:n]
                graph=Mantle.Graph(device)
                Q.linear_batch!(graph,out,A,upload(x),n;name="q5_columns")
                for j in 1:n
                    Q.linear!(graph,columns[j],A,upload(x[:,j]);name="q5_column.$j")
                end
                run(graph)
                result=Array(out)
                @test all(isfinite,result)
                for j in 1:n
                    @test reinterpret(UInt32,result[:,j])==bitsof(columns[j])
                end
            end
        end
        @testset "Batched Q4 embedding is the per-token embedding" begin
            vocab=17;width=512
            packed=upload(DK.q4k_quantize(randn(rng,Float32,vocab,width)));W=DK.Q4KMatrix(packed,vocab,width)
            tokens=Int32[0,16,3,3,1];n=length(tokens)
            out=upload(fill(NaN32,width,n));columns=[upload(fill(NaN32,width)) for _ in 1:n]
            graph=Mantle.Graph(device)
            R.declare_embedding_batch!(graph,out,W,upload(tokens),width,n)
            for j in 1:n
                R.declare_embedding!(graph,columns[j],W,upload(tokens[j:j]),width)
            end
            run(graph)
            result=Array(out)
            for j in 1:n
                @test reinterpret(UInt32,result[:,j])==bitsof(columns[j])
            end
        end
    finally
        plan===nothing||Mantle.free!(plan)
        foreach(Mantle.free!,reverse(owned))
    end
    nothing
end
