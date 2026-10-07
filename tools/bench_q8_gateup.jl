using DNNKernels, Mantle, KernelAbstractions, Random, Statistics, Test

function bench_q8_gateup(; shapes=((1024,2048),(26624,5120)), repeats=12, inner=20,
                          graph=false)
    dev = Mantle.Device()
    backend = Mantle.backend(dev)
    rng = MersenneTwister(88)
    rows = []
    for (h,k) in shapes
        m, mg = 2h, cld(2h,4)
        q = Mantle.Buffer(dev, rand(rng,UInt32,mg,k))
        scale = Mantle.Buffer(dev, fill(0.001f0,m))
        b = Mantle.Buffer(dev, Float16.(0.1f0 .* randn(rng,Float32,k,1)))
        outs = [Mantle.Buffer(dev,Float16,(h,)) for _ in 1:2]
        plans = Any[]
        owners = Any[]
        splits = DNNKernels.q8split(m,k)
        for (i,fused) in enumerate((false,true))
            if graph
                aten = bench_gateup_graph(h,k)
                oldflag = DNNKernels.Q8GATEUP[]
                rp = try
                    DNNKernels.Q8GATEUP[] = fused
                    DNNKernels.planfor(dev,aten,Dict{String,Any}("w"=>DNNKernels.QInt8Matrix(q,scale,m)),(;))
                finally
                    DNNKernels.Q8GATEUP[] = oldflag
                end
                DNNKernels.replay!(rp,"gateup",(b,))
                push!(owners,rp)
                push!(plans,rp.plan)
                continue
            end
            g = Mantle.Graph(dev)
            parts = Mantle.Transient.Buffer(g,Float32,(4mg,1,splits))
            Mantle.dispatch!(g,DNNKernels.q8gemv_kernel!,
                (parts,q,b,Int32(mg),Int32(4mg),Int32(k),Int32(cld(k,splits)),Int32(mg*splits)),
                mg*splits;group=256)
            if fused
                Mantle.dispatch!(g,DNNKernels.q8gateup_reduce!,
                    (outs[i],parts,scale,Int32(h),Int32(4mg),Int32(splits),Int32(0),Int32(h),Val(Float16)),h;group=256)
            else
                projected = Mantle.Transient.Buffer(g,Float16,(m,))
                Mantle.dispatch!(g,DNNKernels.q8reduce_kernel!,
                    (projected,parts,scale,scale,Int32(m),Int32(4mg),Int32(splits),Val(false)),m;group=256)
                v = DNNKernels.swiglurun((h,),(1,),(1,),(1,))
                if v === nothing
                    Mantle.dispatch!(g,DNNKernels.swiglu_strided_kernel!,
                        (outs[i],projected,projected,Int32(1),Int32(h+1),(Int32(1),),(Int32(1),)),h;group=256)
                else
                    Mantle.dispatch!(g,DNNKernels.swiglu_run_kernel!,
                        (outs[i],projected,projected,Int32(1),Int32(h+1),Int32(1),
                         (Int32(1),),(Int32(1),),(Int32(1),),Val(v),(h÷v,)),h÷v;group=256)
                end
            end
            push!(plans,Mantle.record!(Mantle.Plan(g)))
        end
        try
            for _ in 1:3, p in plans; Mantle.run!(p); end
            KernelAbstractions.synchronize(backend)
            values = graph ? [Array(only(p.outputs)) for p in owners] : Array.(outs)
            @test values[1] == values[2]
            times = (Float64[],Float64[])
            for r in 1:repeats, i in (isodd(r) ? (1,2,2,1) : (2,1,1,2))
                t = @elapsed begin
                    for _ in 1:inner; Mantle.run!(plans[i]); end
                    KernelAbstractions.synchronize(backend)
                end
                push!(times[i],1000t/inner)
            end
            old,new = median.(times)
            row = (;h,k,graph,old_ms=old,new_ms=new,speedup=old/new)
            push!(rows,row); println(row); flush(stdout)
        finally
            foreach(Mantle.free!,graph ? owners : plans)
            foreach(Mantle.free!,(outs...,q,scale,b))
        end
    end
    rows
end

function bench_gateup_graph(h,k)
    dk = DNNKernels
    a(p...) = Dict{String,Any}(p...)
    buf(id,shape;kind=:transient,key="",of="",viewop="",attrs=a()) =
        dk.Buffer(id,kind,Any[shape...],Float16,key,(0,0),of,viewop,attrs)
    buffers = Dict(b.id=>b for b in (
        buf("x",(1,k);kind=:external),buf("w",(k,2h);kind=:weight,key="w"),
        buf("stack",(1,2h)),buf("y",(1,h)),
        buf("gate",(1,h);kind=:view,of="stack",viewop="slice.Tensor",
            attrs=a("arg1"=>1,"arg2"=>0,"arg3"=>h)),
        buf("up",(1,h);kind=:view,of="stack",viewop="slice.Tensor",
            attrs=a("arg1"=>1,"arg2"=>h,"arg3"=>2h))))
    ops = [dk.Op("stack","mm.default",["x","w"],"stack",a()),
           dk.Op("y","fused.swiglu",["gate","up"],"y",a())]
    dk.Graph("gateup",String[],["x"],["y"],buffers,collect(keys(buffers)),ops)
end
