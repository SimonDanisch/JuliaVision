module ActivationPreparationTests
using Test, Random, Statistics
import DNNKernels as DK
import Mantle as M

bits(x)=collect(reinterpret(UInt32,vec(Array(M.storage(x)))))

function run(dev)
    rng=MersenneTwister(726)
    @testset "shared activation preparation" begin
        for T in (Float16,Float32), k in (256,1024), n in (1,3,8)
            xh=T.(randn(rng,Float32,k,n));xh[1:32].=0
            x=M.Buffer(dev,xh)
            ys=[M.Buffer(dev,T,(k,n)) for _ in 1:2]
            plans=Any[];preps=Any[]
            # No CPU inference: compare the existing materialized GPU producer
            # and quantizer with the fused declaration, on changing inputs.
            f=DK.SwiGLUMap{T}()
            try
                for fused in (false,true)
                    g=M.Graph(dev);c=DK.ActivationPrep(g)
                    y=ys[Int(fused)+1]
                    if fused
                        p=DK.declare_mapped_q8!(c,:v,y,f,(x,x))
                    else
                        M.dispatch!(g,DK.swiglu_kernel!,(y,x,x,Int64(k*n)),k*n)
                        p=DK.declare_q8!(c,:v,y)
                    end
                    @test DK.declare_q8!(c,:v,y)===p
                    @test c.reused==1
                    # A new logical version MUST be quantized again.
                    @test DK.declare_q8!(c,(:v,2),y)!==p
                    DK.invalidate_q8!(c)
                    @test isempty(c.entries)
                    push!(preps,p)
                    push!(plans,M.record!(M.Plan(g;alias=false,budget=Inf)))
                end
                @test length(plans[1].passes)==length(plans[2].passes)+1
                for repeat in 1:2
                    foreach(M.run!,plans);M.waitidle(dev)
                    @test Array(ys[1])==Array(ys[2])
                    @test bits(preps[1].q)==bits(preps[2].q)
                    @test bits(preps[1].scales)==bits(preps[2].scales)
                    xh .*= T(-0.75);copyto!(x,xh)
                end
            finally
                foreach(M.free!,plans);foreach(M.free!,(ys...,x))
            end
        end
        g=M.Graph(dev);c=DK.ActivationPrep(g)
        @test_throws DimensionMismatch DK.declare_q8!(c,:bad,M.Transient.Buffer(g,Float32,(255,1)))
        @test_throws DimensionMismatch DK.declare_q8!(c,:bad,M.Transient.Buffer(g,Float32,(256,)))
        out=M.Transient.Buffer(g,Float32,(256,1))
        @test_throws DimensionMismatch DK.declare_mapped_q8!(c,:bad,out,+,(M.Transient.Buffer(g,Float32,(256,2)),))
        @test_throws ArgumentError DK.declare_mapped_q8!(c,:bad,out,identity,())
        @test_throws ArgumentError DK.declare_mapped_q8!(c,:bad,out,identity,(out,))
        @test_throws ArgumentError DK.declare_mapped_q8!(c,:bad,out,identity,(M.viewof(out,(256,1)),))
        backing=M.Transient.Buffer(g,Float32,(512,1))
        a=M.viewof(backing,(256,1));b=M.viewof(backing,(256,1);offset=256)
        pa=DK.declare_q8!(c,:layout,a)
        @test DK.declare_q8!(c,:layout,M.viewof(backing,(256,1)))===pa
        @test DK.declare_q8!(c,:layout,b)!==pa
        @test DK.activation_report(c).reused==1
        # In-graph writes must create a new representation, not reuse the
        # pre-write one. Check both snapshots after replay, without a CPU solver.
        x=M.Buffer(dev,randn(rng,Float32,256,1));mg=M.Graph(dev)
        mc=DK.ActivationPrep(mg);p0=DK.declare_q8!(mc,:x,x)
        M.dispatch!(mg,DK.denseew!,(x,(x,),Base.Fix2(*,0f0),Int32(256)),256)
        DK.invalidate_q8!(mc)
        p1=DK.declare_q8!(mc,:x,x)
        pl=M.record!(M.Plan(mg;alias=false,budget=Inf))
        try
            @test p0!==p1
            M.run!(pl);M.waitidle(dev)
            @test any(!iszero,Array(M.storage(p0.scales)))
            @test all(iszero,Array(M.storage(p1.scales)))
            @test all(iszero,bits(p1.q))
        finally;M.free!(pl);M.free!(x);end
    end
end

function graph(k,n,T;expression=true,observe=false,rows=40)
    b(id,kind,shape;key="")=DK.Buffer(id,kind,Any[shape...],T,key,(0,0),"","",Dict{String,Any}())
    bs=[b("x",:external,(n,k)),b("t",:transient,(n,k)),
        b("a",:weight,(k,rows);key="a"),b("b",:weight,(k,rows);key="b"),
        b("y",:transient,(n,rows)),b("z",:transient,(n,rows))]
    ops=DK.Op[]
    if expression
        f=DK.FusedOp((DK.Rounded(T,+),),((DK.In(1),DK.Konst(T(0.125))),))
        push!(ops,DK.Op("expression","fused.elementwise",["x"],"t",Dict{String,Any}("fused"=>f)))
    end
    src=expression ? "t" : "x"
    push!(ops,DK.Op("first","mm.default",[src,"a"],"y",Dict{String,Any}()))
    push!(ops,DK.Op("second","mm.default",[src,"b"],"z",Dict{String,Any}()))
    DK.Graph("activation_graph",String[],["x"],observe && expression ? ["y","z","t"] : ["y","z"],
        Dict(x.id=>x for x in bs),[x.id for x in bs],ops)
end

function benchmark(dev;repeats=8,replays=128)
    rng=MersenneTwister(729);be=M.backend(dev)
    weights=Dict{String,Any}();plans=Any[];samples=Any[]
    old=(DK.SHARE_ACTIVATIONS[],DK.FUSE_ACTIVATION_Q8[])
    g=graph(4096,1,Float32;rows=1024)
    x=M.Buffer(dev,randn(rng,Float32,4096,1))
    names=("unfused","reuse","automatic")
    try
        for key in ("a","b")
            w=DK.toback(be,Float16.(0.01f0.*randn(rng,Float32,1024,4096)))
            try;weights[key]=DK.quantizeq4k(be,w)
            finally;DK.releasedevice!(w);end
        end
        reference=nothing
        for (sharing,fusion) in ((false,false),(true,false),(true,true))
            DK.SHARE_ACTIVATIONS[]=sharing;DK.FUSE_ACTIVATION_Q8[]=fusion
            p=DK.planfor(dev,g,weights,(;));push!(plans,p)
            output=Array.(DK.replay!(p,g.name,(x,)))
            reference===nothing ? (reference=output) : (@assert output==reference)
        end
        for trial in 1:repeats
            for i in (isodd(trial) ? (1:3) : (3:-1:1))
                p=plans[i].plan;M.waitidle(dev)
                before=M.submissions(dev)
                elapsed=@elapsed begin
                    for _ in 1:replays;M.run!(p);end
                    M.waitidle(dev)
                end
                push!(samples,(;trial,condition=names[i],seconds=elapsed,replays,
                    microseconds=1e6*elapsed/replays,submissions=M.submissions(dev)-before,
                    passes=length(p.passes)))
            end
        end
        (;samples,median_us=Dict(n=>median(s.microseconds for s in samples if s.condition==n) for n in names),
            outputs_bitidentical=true,scope="Generic expression feeding two Q4_K matmuls, 4096 input/1024 output, one column; not full-model throughput")
    finally
        DK.SHARE_ACTIVATIONS[],DK.FUSE_ACTIVATION_Q8[]=old
        foreach(M.free!,plans);DK.releaseweights!(weights);M.free!(x)
    end
end

function integration(dev)
    rng=MersenneTwister(727);be=M.backend(dev)
    @testset "automatic activation lowering" begin
        old=(DK.SHARE_ACTIVATIONS[],DK.FUSE_ACTIVATION_Q8[])
        try
            for T in (Float16,Float32), n in (1,3), expression in (false,true), observe in (false,true)
                k=256;weights=Dict{String,Any}();plans=Any[]
                try
                    for key in ("a","b")
                        w=DK.toback(be,Float16.(0.01f0.*randn(rng,Float32,40,k)))
                        try;weights[key]=DK.quantizeq4k(be,w)
                        finally;DK.releasedevice!(w);end
                    end
                    g=graph(k,n,T;expression,observe)
                    for (sharing,fusion) in ((false,false),(true,false),(true,true))
                        DK.SHARE_ACTIVATIONS[]=sharing;DK.FUSE_ACTIVATION_Q8[]=fusion
                        push!(plans,DK.planfor(dev,g,weights,(;)))
                    end
                    @test length(plans[1].plan.passes)==length(plans[2].plan.passes)+1
                    mapped=expression && M.caps(dev).subgroup>=64
                    @test length(plans[2].plan.passes)==length(plans[3].plan.passes)+Int(mapped)
                    for _ in 1:2
                        x=M.Buffer(dev,T.(randn(rng,Float32,k,n)))
                        try
                            outputs=[Array.(DK.replay!(p,g.name,(x,))) for p in plans]
                            @test outputs[1]==outputs[2]
                            @test outputs[1]==outputs[3]
                        finally;M.free!(x);end
                    end
                    # Bound results keep the shared lowering: a write to bound
                    # storage drops the representations prepared from it, so an
                    # aliasing caller gets what the unshared lowering computes
                    # (test_semanticstandard.jl pins that write).
                    x=M.Buffer(dev,T.(randn(rng,Float32,k,n)))
                    bound=Dict(id=>M.Buffer(dev,T,Tuple(reverse(g.buffers[id].shape))) for id in g.outputs)
                    try
                        mg=M.Graph(dev)
                        result=DK.emitgraph!(mg,g,weights,(;);inputs=Dict("x"=>x),outputs=bound)
                        p=M.record!(M.Plan(mg;budget=Inf))
                        try
                            names=[String(q.pass.name) for q in p.passes]
                            @test count(s->endswith(s,".q8act"),names)==Int(!mapped)
                            @test any(s->endswith(s,".mapped_q8"),names)==mapped
                            M.run!(p);M.waitidle(dev)
                            expected=Array.(DK.replay!(plans[1],g.name,(x,)))
                            @test Array(result["y"])==expected[1]
                        finally;M.free!(p);end
                    finally;M.free!(x);foreach(M.free!,values(bound));end
                    # Input-only module composition has fresh private outputs
                    # and must receive the SAME optimizations as Model/call.
                    x=M.Buffer(dev,T.(randn(rng,Float32,k,n)))
                    y=M.Buffer(dev,T,(40,n));p=nothing
                    try
                        mg=M.Graph(dev)
                        result=DK.emitgraph!(mg,g,weights,(;);inputs=Dict("x"=>x))
                        M.dispatch!(mg,DK.denseew!,(y,(result["y"],),identity,Int32(40n)),40n)
                        p=M.record!(M.Plan(mg;budget=Inf))
                        names=[String(q.pass.name) for q in p.passes]
                        @test count(s->endswith(s,".q8act"),names)==Int(!mapped)
                        @test any(s->endswith(s,".mapped_q8"),names)==mapped
                        M.run!(p);M.waitidle(dev)
                        expected=Array.(DK.replay!(plans[1],g.name,(x,)))
                        @test Array(y)==expected[1]
                    finally
                        p===nothing||M.free!(p);M.free!(x);M.free!(y)
                    end
                finally
                    foreach(M.free!,plans);DK.releaseweights!(weights)
                end
            end
        finally
            DK.SHARE_ACTIVATIONS[],DK.FUSE_ACTIVATION_Q8[]=old
        end
    end
end

function swiglumm(dev)
    rng=MersenneTwister(728);be=M.backend(dev)
    @testset "SwiGLU product activation lowering" begin
        old=(DK.SHARE_ACTIVATIONS[],DK.FUSE_ACTIVATION_Q8[])
        try
            for T in (Float16,Float32), n in (1,3)
                k=256;g=graph(k,n,T)
                buffers=copy(g.buffers)
                for id in ("b","z","t");delete!(buffers,id);end
                buffers["up"]=DK.Buffer("up",:external,Any[n,k],T,"",(0,0),"","",Dict{String,Any}())
                op=DK.Op("down","fused.swiglumm",["x","up","a"],"y",Dict{String,Any}("dtype"=>T))
                sg=DK.Graph("swiglumm",String[],["x","up"],["y"],buffers,
                    [filter(id->haskey(buffers,id),g.order);"up"],[op])
                w=DK.toback(be,Float16.(0.01f0.*randn(rng,Float32,40,k)))
                weights=Dict{String,Any}();plans=Any[]
                try
                    weights["a"]=DK.quantizeq4k(be,w)
                    for enabled in (false,true)
                        DK.SHARE_ACTIVATIONS[]=enabled;DK.FUSE_ACTIVATION_Q8[]=enabled
                        push!(plans,DK.planfor(dev,sg,weights,(;)))
                    end
                    @test length(plans[1].plan.passes)-length(plans[2].plan.passes)==Int(M.caps(dev).subgroup>=64)
                    for _ in 1:2
                        inputs=Tuple(M.Buffer(dev,T.(randn(rng,Float32,k,n))) for _ in 1:2)
                        try
                            @test Array.(DK.replay!(plans[1],sg.name,inputs))==Array.(DK.replay!(plans[2],sg.name,inputs))
                        finally;foreach(M.free!,inputs);end
                    end
                finally
                    foreach(M.free!,plans);DK.releaseweights!(weights);DK.releasedevice!(w)
                end
            end
        finally
            DK.SHARE_ACTIVATIONS[],DK.FUSE_ACTIVATION_Q8[]=old
        end
    end
end
end
