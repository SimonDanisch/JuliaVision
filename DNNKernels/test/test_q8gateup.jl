using Test, DNNKernels, Mantle, Random

@testset "stacked INT8 gate/up reduction" begin
    dev = Mantle.Device()
    rng = MersenneTwister(89)
    h, m, mp = 37, 74, 76
    for T in (Float16, Float32), O in (Float16, Float32), splits in (1,3), reverse in (false,true)
        ph = randn(rng, Float32, mp, 1, splits)
        sc = rand(rng, Float32, m)
        p, s = Mantle.Buffer(dev, ph), Mantle.Buffer(dev, sc)
        outs = [Mantle.Buffer(dev, O, (h,1)) for _ in 1:2]
        gateoff, upoff = reverse ? (h,0) : (0,h)
        plans = Any[]
        for (i,fused) in enumerate((false,true))
            g = Mantle.Graph(dev)
            if fused
                Mantle.dispatch!(g, DNNKernels.q8gateup_reduce!,
                    (outs[i],p,s,Int32(h),Int32(mp),Int32(splits),Int32(gateoff),Int32(upoff),Val(T)),h;group=256)
            else
                projected = Mantle.Transient.Buffer(g,T,(m,1))
                Mantle.dispatch!(g,DNNKernels.q8reduce_kernel!,
                    (projected,p,s,s,Int32(m),Int32(mp),Int32(splits),Val(false)),m;group=256)
                Mantle.dispatch!(g,DNNKernels.swiglu_strided_kernel!,
                    (outs[i],projected,projected,Int32(gateoff+1),Int32(upoff+1),
                     (Int32(1),Int32(m)),(Int32(1),Int32(m))),h;group=256)
            end
            push!(plans,Mantle.record!(Mantle.Plan(g)))
        end
        try
            for _ in 1:2
                foreach(Mantle.run!,plans)
                @test Array(outs[1]) == Array(outs[2])
                sums = copy(ph[1:m,1,1])
                for j in 2:splits; sums .+= ph[1:m,1,j]; end
                values = Float32.(T.(sums .* sc))
                gv, uv = values[gateoff+1:gateoff+h], values[upoff+1:upoff+h]
                want = O.(Float32.(O.(gv ./ (1f0 .+ exp.(-gv)))) .* uv)
                @test isapprox(vec(Array(outs[2])),want;rtol=O===Float16 ? 0.002 : 2e-6,atol=2e-6)
                ph .*= -0.75f0; copyto!(p,ph)
            end
        finally
            foreach(Mantle.free!,plans)
            foreach(Mantle.free!,(outs...,p,s))
        end
    end
end

function q8gateup_testgraph(h,k,n; down=false, reverse=false, escape=false, reshaped=false)
    dk = DNNKernels
    attrs(p...) = Dict{String,Any}(p...)
    buf(id,shape;kind=:transient,key="",of="",viewop="",a=attrs()) =
        dk.Buffer(id,kind,Any[shape...],Float16,key,(0,0),of,viewop,a)
    go,uo = reverse ? (h,0) : (0,h)
    buffers = Dict(b.id=>b for b in (
        buf("x",(n,k);kind=:external),buf("w",(k,2h);kind=:weight,key="w"),
        buf("stack",(n,2h)),
        buf("gate",(n,h);kind=:view,of="stack",viewop="slice.Tensor",
            a=attrs("arg1"=>1,"arg2"=>go,"arg3"=>go+h)),
        buf("up",(n,h);kind=:view,of="stack",viewop="slice.Tensor",
            a=attrs("arg1"=>1,"arg2"=>uo,"arg3"=>uo+h)),
        buf("y",(n,down ? 32 : h))))
    if down; buffers["down"] = buf("down",(h,32);kind=:weight,key="down"); end
    if reshaped
        for id in ("gate","up")
            buffers[id*"view"] = buf(id*"view",(1,n,h);kind=:view,of=id,
                viewop="view.default",a=attrs("arg1"=>[1,n,h]))
        end
    end
    halves = reshaped ? ["gateview","upview"] : ["gate","up"]
    ops = [dk.Op("stack","mm.default",["x","w"],"stack",attrs()),
        dk.Op("y",down ? "fused.swiglumm" : "fused.swiglu",
            down ? [halves;"down"] : halves,"y",attrs("dtype"=>Float16))]
    dk.Graph("gateup",String[],["x"],escape ? ["y","gate"] : ["y"],
             buffers,collect(keys(buffers)),ops)
end

@testset "INT8 gate/up lowering and refusals" begin
    dev = Mantle.Device()
    rng = MersenneTwister(90)
    h,k = 37,131
    a = DNNKernels.QInt8Matrix(Mantle.Buffer(dev,rand(rng,UInt32,cld(2h,4),k)),
                              Mantle.Buffer(dev,rand(rng,Float32,2h).*0.01f0),2h)
    dw = DNNKernels.QInt8Matrix(Mantle.Buffer(dev,rand(rng,UInt32,8,h)),
                               Mantle.Buffer(dev,fill(0.002f0,32)),32)
    weights = Dict{String,Any}("w"=>a,"down"=>dw)
    oldflag = DNNKernels.Q8GATEUP[]
    try
        for n in (1,2), down in (false,true), reverse in (false,true), escape in (false,true), reshaped in (false,true)
            graph = q8gateup_testgraph(h,k,n;down,reverse,escape,reshaped)
            plans = Any[]
            for enabled in (false,true)
                DNNKernels.Q8GATEUP[] = enabled
                push!(plans,DNNKernels.planfor(dev,graph,weights,(;)))
            end
            try
                fused = n == 1 && !escape
                names = [String(p.pass.name) for p in plans[2].plan.passes]
                @test any(n -> occursin("gateup",n),names) == fused
                @test length(plans[1].plan.passes) - length(plans[2].plan.passes) == Int(fused)
                for _ in 1:2
                    b = Mantle.Buffer(dev,Float16.(0.1f0 .* randn(rng,Float32,k,n)))
                    try
                        old = Array.(DNNKernels.replay!(plans[1],"gateup",(b,)))
                        got = Array.(DNNKernels.replay!(plans[2],"gateup",(b,)))
                        @test got == old
                    finally
                        Mantle.free!(b)
                    end
                end
            finally
                foreach(Mantle.free!,plans)
            end
        end
        g = q8gateup_testgraph(h,k,1)
        mm,sw = g.ops
        match(graph;bound=Dict{String,Any}()) = DNNKernels.q8gateupmatch(graph,mm,sw,weights,(;),Set{String}(),bound)
        @test match(g)
        @test !match(g;bound=Dict("up"=>nothing))
        extra = DNNKernels.Op("other","clone.default",["gate"],"other",Dict{String,Any}())
        g2 = DNNKernels.Graph(g.name,g.symbols,g.inputs,g.outputs,g.buffers,g.order,[g.ops;extra])
        @test !match(g2)
        gd = q8gateup_testgraph(h,k,1;down=true)
        rotated = copy(weights)
        rotated["down"] = DNNKernels.ConvRotQInt8Matrix(dw.q,dw.scale,dw.m,256)
        @test !DNNKernels.q8gateupmatch(gd,gd.ops...,rotated,(;),Set{String}(),Dict{String,Any}())
        for options in ((;keepall=true),(;before=ctx->nothing),(;skip=("y",)))
            _,ctx = DNNKernels.emitgraph(dev,g,weights,(;);options...)
            try
                @test haskey(ctx.res,"stack")
            finally
                DNNKernels.freeowned!(ctx)
            end
        end
        buffers = copy(g.buffers)
        buffers["spare"] = DNNKernels.Buffer("spare",:transient,Any[1,k],Float16,"",(0,0),"","",Dict{String,Any}())
        unrelated = DNNKernels.Op("spare","clone.default",["x"],"spare",Dict{String,Any}())
        separated = DNNKernels.Graph(g.name,g.symbols,g.inputs,[g.outputs;"spare"],
            buffers,[g.order;"spare"],[mm,unrelated,sw])
        _,ctx = DNNKernels.emitgraph(dev,separated,weights,(;))
        try
            @test haskey(ctx.res,"stack")
        finally
            DNNKernels.freeowned!(ctx)
        end
    finally
        DNNKernels.Q8GATEUP[] = oldflag
        foreach(Mantle.free!,(a.q,a.scale,dw.q,dw.scale))
    end
end
