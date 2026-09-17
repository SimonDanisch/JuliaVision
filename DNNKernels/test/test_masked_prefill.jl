using Test, DNNKernels, Mantle, KernelAbstractions, Random, LinearAlgebra

@testset "staged masked prefill with strided caches" begin
    be=Mantle.LavaBackend(); caps=DNNKernels.caps(be)
    if caps.coopmat && caps.coopmatsubgroup==32 && caps.tile==16
        rng=MersenneTwister(711)
        qh=randn(rng,Float16,128,128,2,1).*Float16(.2)
        kh=randn(rng,Float16,128,512,2,1).*Float16(.2)
        vh=randn(rng,Float16,128,512,2,1)
        q=DNNKernels.toback(be,qh)
        k=view(DNNKernels.toback(be,kh),:,1:128,:,:)
        v=view(DNNKernels.toback(be,vh),:,1:128,:,:)
        out=similar(q)
        g=DNNKernels.Graph("empty",String[],String[],String[],
            Dict{String,DNNKernels.Buffer}(),String[],DNNKernels.Op[])
        ctx=DNNKernels.Ctx(Dict{String,Any}(),g,(;),be;ws=nothing)
        for offset in (0,63)
            mh=Float16[s<=mod(r-1,16)+offset+1 ? 0 : -65504
                       for s in 1:128,r in 1:128,h in 1:1,b in 1:1]
            mask=DNNKernels.toback(be,mh)
            DNNKernels.reset!(ctx.ws)
            DNNKernels.maskedprefill!(ctx,out,q,k,v,mask,.08838835f0)
            got=Array(out); want=similar(qh)
            for h in 1:2
                scores=Float16.(Float32.(kh[:,1:128,h,1])'*Float32.(qh[:,:,h,1]))
                scores=Float32.(Float16.(Float16.(scores.*.08838835f0).+mh[:,:,1,1]))
                p=exp.(scores.-maximum(scores;dims=1))
                p=Float16.(p./sum(p;dims=1))
                want[:,:,h,1]=Float32.(vh[:,1:128,h,1])*Float32.(p)
            end
            @test all(isfinite,got)
            @test maximum(abs,Float32.(got).-Float32.(want))<.005
        end
        @test !DNNKernels.maskedprefill_applicable(ctx,q,k,v,zeros(Float16,128,128,1,1))
    else
        @test_skip false
    end
end

# The query floor is a measured routing decision (see `maskedprefill.jl`), so it
# is worth pinning: a grouped-query 128-token prompt folds to `nq = 1024`, where
# the staged route is no faster and drifts further from the unfused graph.
@testset "staged masked prefill query floor" begin
    be=Mantle.LavaBackend(); caps=DNNKernels.caps(be)
    if caps.coopmat && caps.coopmatsubgroup==32 && caps.tile==16
        nk=512
        k=DNNKernels.toback(be,zeros(Float16,128,nk,8,1))
        v=DNNKernels.toback(be,zeros(Float16,128,nk,8,1))
        g=DNNKernels.Graph("empty",String[],String[],String[],
            Dict{String,DNNKernels.Buffer}(),String[],DNNKernels.Op[])
        ctx=DNNKernels.Ctx(Dict{String,Any}(),g,(;),be;ws=nothing)
        for (nq,want) in ((1024,false),(2048,true),(4096,true))
            q=DNNKernels.toback(be,zeros(Float16,128,nq,8,1))
            mask=DNNKernels.toback(be,zeros(Float16,nk,nq,1,1))
            @test DNNKernels.maskedprefill_applicable(ctx,q,k,v,mask)==want
        end
    else
        @test_skip false
    end
end

@testset "strided SwiGLU retains rounded arithmetic" begin
    be=Mantle.LavaBackend()
    x=DNNKernels.toback(be,randn(MersenneTwister(21),Float16,1024,32,1))
    g=view(x,1:512,:,:); u=view(x,513:1024,:,:)
    out=similar(x,Float16,512,32,1); ref=similar(out)
    gr,ur=DNNKernels.stridedroot(g),DNNKernels.stridedroot(u)
    DNNKernels.swiglu_strided_kernel!(be,256)(out,
        reshape(gr[1],length(gr[1])),reshape(ur[1],length(ur[1])),
        Int32(gr[2]+1),Int32(ur[2]+1),Int32.(strides(g)),Int32.(strides(u));ndrange=size(out))
    DNNKernels.swiglu_kernel!(be,256)(ref,copy(g),copy(u),Int64(length(out));ndrange=length(out))
    @test Array(out)==Array(ref)
end
