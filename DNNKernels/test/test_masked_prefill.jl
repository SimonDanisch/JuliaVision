import KernelInterface
using Test, DNNKernels, Mantle, KernelAbstractions, Random, LinearAlgebra

# The staged masked attention as `fused.maskedattention` declares it: dense
# operands (the declared form takes the cache window as a dense copy).
function staged_attention(dev,q,k,v,mask,scale)
    DK=DNNKernels
    g=Mantle.Graph(dev)
    ctx=DK.EmitCtx(DK.Graph("attn",String[],String[],String[],Dict{String,DK.Buffer}(),String[],DK.Op[]),
        g,dev,(;),Dict{String,Any}(),Set{String}(),Ref("attn"),Any[],Dict{String,Any}())
    out=Mantle.Buffer(dev,Float16,size(q))
    DK.maskedprefill_dispatch!(ctx,DK.Op("attn","fused.maskedattention",String[],"attn",Dict{String,Any}()),
        out,q,k,v,mask,scale)
    plan=Mantle.record!(Mantle.Plan(g))
    try
        Mantle.run!(plan);Mantle.waitidle(dev)
        Array(Mantle.storage(out))
    finally
        Mantle.free!(plan);Mantle.free!(out)
    end
end

@testset "staged masked prefill" begin
    dev=Mantle.Device();caps=Mantle.caps(dev)
    if caps.coopmat && caps.coopmatsubgroup==32 && caps.tile==16
        rng=MersenneTwister(711)
        qh=randn(rng,Float16,128,128,2,1).*Float16(.2)
        kh=randn(rng,Float16,128,512,2,1).*Float16(.2)
        vh=randn(rng,Float16,128,512,2,1)
        q=Mantle.Buffer(dev,qh)
        k=Mantle.Buffer(dev,kh[:,1:128,:,:]);v=Mantle.Buffer(dev,vh[:,1:128,:,:])
        for offset in (0,63)
            mh=Float16[s<=mod(r-1,16)+offset+1 ? 0 : -65504
                       for s in 1:128,r in 1:128,h in 1:1,b in 1:1]
            mask=Mantle.Buffer(dev,mh)
            got=staged_attention(dev,q,k,v,mask,.08838835f0)
            want=similar(qh)
            for h in 1:2
                scores=Float16.(Float32.(kh[:,1:128,h,1])'*Float32.(qh[:,:,h,1]))
                scores=Float32.(Float16.(Float16.(scores.*.08838835f0).+mh[:,:,1,1]))
                p=exp.(scores.-maximum(scores;dims=1))
                p=Float16.(p./sum(p;dims=1))
                want[:,:,h,1]=Float32.(vh[:,1:128,h,1])*Float32.(p)
            end
            @test all(isfinite,got)
            @test maximum(abs,Float32.(got).-Float32.(want))<.005
            Mantle.free!(mask)
        end
        foreach(Mantle.free!,(q,k,v))
    else
        @test_skip false
    end
end

# The query floor is a measured routing decision (see `maskedprefill.jl`), so it
# is worth pinning: a grouped-query 128-token prompt folds to `nq = 1024`, where
# the staged route is no faster and drifts further from the unfused graph.
@testset "staged masked prefill query floor" begin
    dev=Mantle.Device();caps=Mantle.caps(dev)
    if caps.coopmat && caps.coopmatsubgroup==32 && caps.tile==16
        nk=512
        k=Mantle.Buffer(dev,zeros(Float16,128,nk,8,1));v=Mantle.Buffer(dev,zeros(Float16,128,nk,8,1))
        for (nq,want) in ((1024,false),(2048,true),(4096,true))
            q=Mantle.Buffer(dev,zeros(Float16,128,nq,8,1));mask=Mantle.Buffer(dev,zeros(Float16,nk,nq,1,1))
            @test DNNKernels.maskedprefill_applicable(caps,q,k,v,mask)==want
            Mantle.free!(q);Mantle.free!(mask)
        end
        @test !DNNKernels.maskedprefill_applicable(caps,k,k,v,Mantle.Buffer(dev,zeros(Float16,nk,nk,1,1)))
        foreach(Mantle.free!,(k,v))
    else
        @test_skip false
    end
end

@testset "strided SwiGLU retains rounded arithmetic" begin
    be=Mantle.defaultbackend()
    x=DNNKernels.toback(be,randn(MersenneTwister(21),Float16,1024,32,1))
    g=view(x,1:512,:,:); u=view(x,513:1024,:,:)
    out=similar(x,Float16,512,32,1); ref=similar(out)
    gr,ur=DNNKernels.stridedroot(g),DNNKernels.stridedroot(u)
    KernelInterface.Kernel(be, DNNKernels.swiglu_strided_kernel!)(out,
        reshape(gr[1],length(gr[1])),reshape(ur[1],length(ur[1])),
        Int32(gr[2]+1),Int32(ur[2]+1),Int32.(strides(g)),Int32.(strides(u));
        ndrange=length(out), workgroupsize=256)
    KernelInterface.Kernel(be, DNNKernels.swiglu_kernel!)(
        ref,copy(g),copy(u),Int64(length(out)); ndrange=length(out), workgroupsize=256)
    @test Array(out)==Array(ref)
end
