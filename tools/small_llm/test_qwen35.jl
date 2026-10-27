using Test, Random, LinearAlgebra, Mantle, KernelAbstractions
using .Qwen35Experiment
const QE=Qwen35Experiment

function test_qwen35_kernels(device)
    rng=MersenneTwister(3517)
    owned=Any[]
    upload(x)=begin a=Mantle.Buffer(device,x);push!(owned,a);a;end
    try
        @testset "Qwen native contiguous head repetition" begin
            conv=randn(rng,Float32,8192)
            q=upload(fill(NaN32,4096));k=upload(fill(NaN32,4096));src=upload(conv)
            g=Mantle.Graph(device)
            QE.dispatch(g,QE.repeat_qk!,(q,k,src,Int32(32),Int32(16)),4096)
            Mantle.runonce!(g)
            wantq=repeat(reshape(conv[1:2048],128,16);inner=(1,2))
            wantk=repeat(reshape(conv[2049:4096],128,16);inner=(1,2))
            @test reshape(Array(q),128,32)==wantq
            @test reshape(Array(k),128,32)==wantk
        end
        @testset "Qwen split attention output coverage and stable softmax" begin
            for n in (1,7,64,65,129)
                heads=16;kvheads=4;capacity=192;chunks=cld(capacity,64)
                qh=randn(rng,Float32,256,heads)
                kh=randn(rng,Float32,256,kvheads,capacity)
                vh=randn(rng,Float32,256,kvheads,capacity)
                gateh=rand(rng,Float32,256,heads)
                q,k,v,gate=upload(qh),upload(kh),upload(vh),upload(gateh)
                pm=upload(fill(NaN32,heads*chunks));ps=upload(fill(NaN32,heads*chunks))
                pa=upload(fill(NaN32,256*heads*chunks));out=upload(fill(NaN32,256,heads))
                pos=Mantle.GPURef(device,Int32(n-1));push!(owned,pos)
                g=Mantle.Graph(device)
                QE.dispatch(g,QE.attention_partial!,
                    (pm,ps,pa,q,k,v,pos,Int32(kvheads),Int32(heads÷kvheads),Int32(chunks),
                     Val(Mantle.caps(device).subgroup)),heads*chunks*256)
                QE.dispatch(g,QE.BR.attention_combine_kernel!,
                    (out,pm,ps,pa,gate,pos,Int32(chunks)),heads*256)
                Mantle.runonce!(g)
                want=zeros(Float32,256,heads)
                for h in 1:heads
                    kv=(h-1)÷(heads÷kvheads)+1
                    scores=[dot(Float64.(qh[:,h]),Float64.(kh[:,kv,p]))/16 for p in 1:n]
                    probs=exp.(scores.-maximum(scores));probs./=sum(probs)
                    want[:,h]=Float32.(Float64.(vh[:,kv,1:n])*probs).*gateh[:,h]
                end
                got=Array(out)
                @test all(isfinite,got)
                @test got ≈ want atol=3f-6 rtol=3f-5
            end
        end
    finally
        foreach(Mantle.free!,owned)
    end
    nothing
end
