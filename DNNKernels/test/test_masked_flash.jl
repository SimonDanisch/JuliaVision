using Test, DNNKernels, Mantle, KernelAbstractions, Random, LinearAlgebra

@testset "masked attention fusion boundaries" begin
    dk=DNNKernels
    b(id,shape;of="",dtype=Float16)=dk.Buffer(id,isempty(of) ? :transient : :view,
        Any[shape...],dtype,"",(0,0),of,isempty(of) ? "" : "view.default",Dict{String,Any}())
    buffers=Dict(x.id=>x for x in (
        b("q",(1,2,4,16)),b("k",(1,2,8,16)),b("v",(1,2,8,16)),
        b("q3",(2,4,16);of="q"),b("k3",(2,16,8);of="k"),
        b("v3",(2,8,16);of="v"),b("mask",(1,1,4,8)),
        b("scores",(2,4,8)),b("scaled",(1,2,4,8)),b("p",(1,2,4,8)),
        b("p3",(2,4,8);of="p"),b("out",(2,4,16))))
    f=dk.FusedOp((dk.Rounded(Float16,*),dk.Rounded(Float16,+)),
        ((dk.In(1),dk.Konst(0.25)),(dk.Tmp(1),dk.In(2))))
    ops=[dk.Op("scores","bmm.default",["q3","k3"],"scores",Dict{String,Any}()),
         dk.Op("scaled","fused.elementwise",["scores","mask"],"scaled",Dict{String,Any}("fused"=>f)),
         dk.Op("p","_softmax.default",["scaled"],"p",Dict{String,Any}("arg1"=>-1)),
         dk.Op("out","bmm.default",["p3","v3"],"out",Dict{String,Any}())]
    graph(outputs,buffers=buffers)=dk.Graph("masked",String[],["q","k","v","mask"],
        outputs,buffers,collect(keys(buffers)),ops)
    fused,n=dk.fusemaskedattention(graph(["out"]))
    @test n==1
    @test only(fused.ops).ins==["q","k","v","mask"]
    @test only(fused.ops).aten=="fused.maskedattention"
    @test last(dk.fusemaskedattention(graph(["out","scores"])))==0
    widened=copy(buffers); widened["p"]=b("p",(1,2,4,8);dtype=Float32)
    @test last(dk.fusemaskedattention(graph(["out"],widened)))==0
    saved=dk.FUSEATTENTION[]
    try
        dk.FUSEATTENTION[]=false
        @test last(dk.fusemaskedattention(Dict("masked"=>graph(["out"]))))==0
    finally
        dk.FUSEATTENTION[]=saved
    end
end

@testset "masked cooperative attention" begin
    # The cooperative-matrix KERNELS, not `dev.coopmat`: the staged kernels are
    # emitted against a literal `_lava_coopmat_load_f16_16x16_a`, so the tile must
    # be 16 AND the tree defining them must be present. See `DNNKernels.coopmatkernels`.
    if !DNNKernels.coopmatkernels(DNNKernels.Ctx(Mantle.defaultbackend()).dev)
        @test_skip DNNKernels.coopmatkernels(DNNKernels.Ctx(Mantle.defaultbackend()).dev)
    else
    backend = Mantle.defaultbackend()
    caps = DNNKernels.caps(backend)
    if caps.coopmat && caps.coopmatsubgroup == 32
        rng = MersenneTwister(24)
        for (lq,lk) in ((8,128),(256,512))
            qh = randn(rng,Float16,128,lq,2,1) .* Float16(0.2)
            kh = randn(rng,Float16,128,lk,2,1) .* Float16(0.2)
            vh = randn(rng,Float16,128,lk,2,1)
            q,k,v = map(a->DNNKernels.toback(backend,a),(qh,kh,vh))
            out = similar(q)
            g = DNNKernels.Graph("empty",String[],String[],String[],Dict{String,DNNKernels.Buffer}(),String[],DNNKernels.Op[])
            ctx = DNNKernels.Ctx(Dict{String,Any}(),g,(;),backend;ws=DNNKernels.nothing)
            plan = DNNKernels.flashcm_plan(caps,q,k,v,nothing;clamp=true)
            @test plan isa DNNKernels.FlashCMPlan
            for offset in (0, lk÷2)
                maskh = [s <= min(lk,offset+mod(r-1,max(1,lq÷8))+1) ? Float16(0) : -floatmax(Float16)
                         for s in 1:lk, r in 1:lq, h in 1:1, b in 1:1]
                mask = DNNKernels.toback(backend,maskh)
                scale = Float32(1/sqrt(128))
                DNNKernels.reset!(ctx.ws)
                DNNKernels.sdpaflashcm!(ctx,out,plan,q,k,v,scale;mask)
                got = Array(out)
                want = similar(qh)
                for h in 1:2
                    scores = Float16.(Float16.(Float32.(kh[:,:,h,1])'*Float32.(qh[:,:,h,1])) .* scale)
                    scores = Float32.(Float16.(scores .+ maskh[:,:,1,1]))
                    p = exp.(scores .- maximum(scores;dims=1))
                    p = Float16.(p ./ sum(p;dims=1))
                    want[:,:,h,1] = Float32.(vh[:,:,h,1])*Float32.(p)
                end
                @test all(isfinite,got)
                @test maximum(abs,Float32.(got).-Float32.(want)) < 0.005
            end
        end
    else
        @test_skip false
    end
    end
end

# The op as a graph declares it. Its only implementation was the eager arm,
# deleted with the eager op library on 2026-09-23; `fusemaskedattention` kept
# building it, so every graph it fired on was refused with "no emit method",
# Horizon 32B's prefill among them. One case per plan family `emitop!` picks.
@testset "fused.maskedattention, declared" begin
    dk = DNNKernels
    backend = Mantle.LavaBackend()
    dev = Mantle.todevice(backend)
    caps = Mantle.caps(dev)
    if caps.coopmat && caps.coopmatsubgroup == 32
        buf(id, shape, kind = :external) =
            dk.Buffer(id, kind, Any[shape...], Float16, "", (0, 0), "", "", Dict{String,Any}())
        rng = MersenneTwister(7)
        H, E = 2, 128
        # (lq, lk): a short query over few keys (register tiling), over many (keys
        # split eight ways), a long query (flash), a longer one (staged prefill).
        for (lq, lk) in ((8, 128), (8, 512), (256, 512), (2048, 256))
            qh = randn(rng, Float16, E, lq, H, 1) .* Float16(0.2)
            kh = randn(rng, Float16, E, lk, H, 1) .* Float16(0.2)
            vh = randn(rng, Float16, E, lk, H, 1)
            # Causal, with the query block placed at the end of the keys.
            maskh = [s <= lk - lq + r ? Float16(0) : -floatmax(Float16)
                     for s in 1:lk, r in 1:lq, _ in 1:1, _ in 1:1]
            scale = 1 / sqrt(E)
            bufs = Dict("q" => buf("q", (1, H, lq, E)), "k" => buf("k", (1, H, lk, E)),
                        "v" => buf("v", (1, H, lk, E)), "mask" => buf("mask", (1, 1, lq, lk)),
                        "out" => buf("out", (H, lq, E), :transient))
            op = dk.Op("out", "fused.maskedattention", ["q", "k", "v", "mask"], "out",
                       Dict{String,Any}("scale" => scale))
            g = dk.Graph("masked", String[], ["q", "k", "v", "mask"], ["out"], bufs,
                         collect(keys(bufs)), [op], Vector{Vector{String}}())
            args = map(a -> dk.toback(backend, a), (qh, kh, vh, maskh))
            plan = dk.planfor(dev, g, Dict{String,Any}(), NamedTuple())
            got = Float32.(Array(only(dk.replay!(plan, "masked", args))))
            Mantle.free!(plan.plan)
            want = zeros(Float32, E, lq, H)
            for h in 1:H
                scores = Float16.(Float16.(Float32.(kh[:, :, h, 1])' * Float32.(qh[:, :, h, 1])) .* Float16(scale))
                scores = Float32.(Float16.(scores .+ maskh[:, :, 1, 1]))
                p = exp.(scores .- maximum(scores; dims = 1))
                p = Float16.(p ./ sum(p; dims = 1))
                want[:, :, h] = Float32.(vh[:, :, h, 1]) * Float32.(p)
            end
            @test size(got) == (E, lq, H)
            @test all(isfinite, got)
            @test maximum(abs, got .- want) < 0.005
            @test dk.maskedprefill_applicable(caps, qh, kh, vh, maskh) == (lq >= 2048)
        end
    else
        @test_skip false
    end
end
