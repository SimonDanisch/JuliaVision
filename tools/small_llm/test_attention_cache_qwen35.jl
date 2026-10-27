using Test,Random,LinearAlgebra
import Mantle,KernelAbstractions

function test_qwen35_attention_cache(device)
    Q=Qwen35Experiment
    rng=MersenneTwister(3519)
    results=Any[]
    @testset "Shared attention kernels and KV precision" begin
        # R=4 is used by 2B/4B; R=8 by the 35B-A3B MoE.
        cases=[(r,n,scale) for r in (1,2,4,8) for n in (1,64,65,129) for scale in (1f0,)]
        append!(cases,[(r,n,scale) for r in (4,8) for n in (7,63,257) for scale in (1f0,8f0)])
        for (ratio,n,scale) in cases, T in (Float32,Float16)
            owned=Any[]
            upload(x)=begin a=Mantle.Buffer(device,x);push!(owned,a);a;end
            try
                kvheads=2;heads=ratio*kvheads;capacity=320;chunks=cld(capacity,64)
                qh=scale*randn(rng,Float32,256,heads)
                kh=T.(randn(rng,Float32,256,kvheads,capacity))
                vh=T.(randn(rng,Float32,256,kvheads,capacity))
                # Reading any unoccupied row must not contaminate the result.
                kh[:,:,n+1:end].=T(NaN);vh[:,:,n+1:end].=T(NaN)
                gateh=rand(rng,Float32,256,heads)
                q,k,v,gate=upload(qh),upload(kh),upload(vh),upload(gateh)
                pm=upload(fill(NaN32,heads*chunks));ps=upload(fill(NaN32,heads*chunks))
                pa=upload(fill(NaN32,256*heads*chunks))
                out=upload(fill(NaN32,256,heads))
                pos=Mantle.GPURef(device,Int32(n-1));push!(owned,pos)
                want=zeros(Float32,256,heads)
                for h in 1:heads
                    kv=(h-1)÷ratio+1
                    scores=[dot(Float64.(qh[:,h]),Float64.(kh[:,kv,p]))/16 for p in 1:n]
                    probs=exp.(scores.-maximum(scores));probs./=sum(probs)
                    want[:,h]=Float32.(Float64.(vh[:,kv,1:n])*probs).*gateh[:,h]
                end
                for mode in (:split,:grouped)
                    fill!(pm,NaN32);fill!(ps,NaN32);fill!(pa,NaN32);fill!(out,NaN32)
                    g=Mantle.Graph(device)
                    Q.declare_attention!(g,mode,pm,ps,pa,q,k,v,pos,kvheads,heads,chunks;name="test.attention")
                    Q.dispatch(g,Q.BR.attention_combine_kernel!,
                        (out,pm,ps,pa,gate,pos,Int32(chunks)),heads*256;name="test.combine")
                    plan=Mantle.record!(Mantle.Plan(g;budget=Inf))
                    try
                        before=Mantle.submissions(device)
                        Mantle.run!(plan);KernelAbstractions.synchronize(device)
                        @test Mantle.submissions(device)-before==1
                        got=Array(out)
                        @test all(isfinite,got)
                        @test got ≈ want atol=8f-6 rtol=5f-5
                        push!(results,(;ratio,n,scale,cache=string(T),mode=string(mode),
                            maxabs=maximum(abs,got-want),relrms=norm(got-want)/norm(want)))
                    finally
                        Mantle.free!(plan)
                    end
                end
            finally
                foreach(Mantle.free!,owned)
            end
        end
    end
    @testset "KV write conversion after normalization and RoPE" begin
        owned=Any[]
        upload(x)=begin a=Mantle.Buffer(device,x);push!(owned,a);a;end
        try
            nkv=2;capacity=64
            k=upload(randn(rng,Float32,256*nkv));v=upload(randn(rng,Float32,256*nkv))
            norm=upload(ones(Float32,256))
            pos=Mantle.GPURef(device,Int32(63));push!(owned,pos)
            pairs=[(upload(zeros(T,256,nkv,capacity)),upload(zeros(T,256,nkv,capacity))) for T in (Float32,Float16)]
            g=Mantle.Graph(device)
            for (kc,vc) in pairs
                Q.dispatch(g,Q.prepare_kv!,(kc,vc,k,v,norm,pos,Int32(nkv),1f7,1f-6),nkv*256;name="test.write.$(eltype(kc))")
            end
            Mantle.runonce!(g)
            for (full,half) in zip(pairs[1],pairs[2])
                @test Array(half)==Float16.(Array(full))
                @test all(iszero,Array(half)[:,:,1:63])
            end
        finally
            foreach(Mantle.free!,owned)
        end
    end
    results
end
