import KernelInterface
using Test, DNNKernels, Mantle, KernelAbstractions, Random, LinearAlgebra

# The staged route as a model reaches it: a declared `fused.maskedattention` op
# with a query long enough for it (`nq >= 2048`, see `maskedprefill.jl`), planned
# and replayed, against the same fp16-rounded reference the eager form had.
@testset "staged masked prefill, declared" begin
    be=Mantle.defaultbackend(); dev=Mantle.Device(be); caps=DNNKernels.caps(be)
    E,nq,nk,H=128,2048,256,2
    if DNNKernels.maskedprefill_applicable(caps,zeros(Float16,E,nq,H,1),zeros(Float16,E,nk,H,1),
                                           zeros(Float16,E,nk,H,1),zeros(Float16,nk,nq,1,1))
        rng=MersenneTwister(711)
        qh=randn(rng,Float16,E,nq,H,1).*Float16(.2)
        kh=randn(rng,Float16,E,nk,H,1).*Float16(.2)
        vh=randn(rng,Float16,E,nk,H,1)
        scale=.08838835f0
        for offset in (0,63)
            mh=Float16[s<=mod(r-1,16)+offset+1 ? 0 : -65504
                       for s in 1:nk,r in 1:nq,h in 1:1,b in 1:1]
            A(p...)=Dict{String,Any}(p...)
            buf(id,kind,shape)=DNNKernels.Buffer(id,kind,Any[shape...],Float16,"",(0,0),"","",A())
            bs=[buf("q",:external,(1,H,nq,E)),buf("k",:external,(1,H,nk,E)),
                buf("v",:external,(1,H,nk,E)),buf("mask",:external,(1,1,nq,nk)),
                buf("o",:transient,(1,H,nq,E))]
            ops=[DNNKernels.Op("o","fused.maskedattention",["q","k","v","mask"],"o",A("scale"=>scale))]
            g=DNNKernels.Graph("attn",String[],["q","k","v","mask"],["o"],
                               Dict(b.id=>b for b in bs),[b.id for b in bs],ops)
            ins=(Mantle.Buffer(dev,qh),Mantle.Buffer(dev,kh),Mantle.Buffer(dev,vh),Mantle.Buffer(dev,mh))
            plan=DNNKernels.planfor(dev,g,Dict{String,Any}(),(;))
            got=Float32.(Array(first(DNNKernels.replay!(plan,"attn",ins))))
            kernels=[d.kernel for pp in plan.plan.passes for d in Mantle.dispatches(pp.pass)]
            Mantle.free!(plan.plan)
            @test DNNKernels.masked_batch_gemm! in kernels
            want=similar(got)
            for h in 1:H
                scores=Float16.(Float32.(kh[:,:,h,1])'*Float32.(qh[:,:,h,1]))
                scores=Float32.(Float16.(Float16.(scores.*scale).+mh[:,:,1,1]))
                p=exp.(scores.-maximum(scores;dims=1))
                p=Float16.(p./sum(p;dims=1))
                want[:,:,h,1]=Float32.(vh[:,:,h,1])*Float32.(p)
            end
            @test all(isfinite,got)
            @test maximum(abs,got.-want)<.005
        end
    else
        @test_skip false
    end
end

# The query floor is a measured routing decision (see `maskedprefill.jl`), so it
# is worth pinning: a grouped-query 128-token prompt folds to `nq = 1024`, where
# the staged route is no faster and drifts further from the unfused graph. The
# decision is the device's and the operands' shapes, so host arrays ask it.
@testset "staged masked prefill query floor" begin
    caps=DNNKernels.caps(Mantle.defaultbackend())
    if caps.coopmat && caps.coopmatsubgroup==32 && caps.tile==16
        nk=512
        k=zeros(Float16,128,nk,8,1); v=zeros(Float16,128,nk,8,1)
        for (nq,want) in ((1024,false),(2048,true),(4096,true))
            q=zeros(Float16,128,nq,8,1); mask=zeros(Float16,nk,nq,1,1)
            @test DNNKernels.maskedprefill_applicable(caps,q,k,v,mask)==want
        end
        # A mask of another shape is not this route's.
        @test !DNNKernels.maskedprefill_applicable(caps,zeros(Float16,128,2048,8,1),k,v,
                                                   zeros(Float16,nk,1024,1,1))
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
