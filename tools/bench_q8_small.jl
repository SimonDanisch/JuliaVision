using DNNKernels, Mantle, KernelAbstractions, Random, Statistics, Test

# Record both algorithms, warm them, then alternate order to limit clock drift.
# `smallbatch=false` keeps the previous GEMM/dequantization path for comparison.
# With widths=(1,), `epi` compares separate and fused decode activations.
function bench_q8_small(; shapes=((53248,5120),(10240,5120),(5120,8192),(5120,26624)),
                         widths=(2,4,8), repeats=6, inner=10, epi=identity)
    dev = Mantle.Device()
    backend = Mantle.backend(dev)
    rng = MersenneTwister(84)
    results = []
    for (m,k) in shapes
        a = DNNKernels.QInt8Matrix(Mantle.Buffer(dev,rand(rng,UInt32,cld(m,4),k)),
                                   Mantle.Buffer(dev,fill(0.001f0,m)),m)
        for n in widths
            b = Mantle.Buffer(dev,Float16.(0.1f0 .* randn(rng,Float32,k,n)))
            outs = [Mantle.Buffer(dev,Float16,(m,n)) for _ in 1:2]
            plans = Any[]
            for (i,small) in enumerate((false,true))
                g = Mantle.Graph(dev)
                aten = DNNKernels.Graph("q8bench",String[],String[],String[],
                    Dict{String,DNNKernels.Buffer}(),String[],DNNKernels.Op[])
                ctx = DNNKernels.EmitCtx(aten,g,dev,(;),Dict{String,Any}(),Set{String}(),
                    Ref("out"),Any[],Dict{String,Any}())
                op = DNNKernels.Op("mm","mm.default",String[],"out",Dict{String,Any}())
                DNNKernels.gemm!(ctx,op,outs[i],a,b;smallbatch=small,
                                 epi=small ? epi : identity)
                if !small && epi !== identity
                    DNNKernels.ewdispatch!(ctx,outs[i],(m,n),(outs[i],),
                        (DNNKernels.bcstrides((m,n),(m,n)),),epi;name="activation")
                end
                push!(plans,Mantle.record!(Mantle.Plan(g)))
            end
            try
                for _ in 1:3, p in plans; Mantle.run!(p); end
                KernelAbstractions.synchronize(backend)
                reference, got = map(x->Float32.(Array(x)),outs)
                @test isapprox(got,reference;rtol=0.002,atol=0.002)
                times = (Float64[],Float64[])
                for repeat in 1:repeats
                    for i in (isodd(repeat) ? (1,2,2,1) : (2,1,1,2))
                        t = @elapsed begin
                            for _ in 1:inner; Mantle.run!(plans[i]); end
                            KernelAbstractions.synchronize(backend)
                        end
                        push!(times[i],1000t/inner)
                    end
                end
                old,new = median.(times)
                row = (;m,k,n,old_ms=old,new_ms=new,speedup=old/new)
                push!(results,row)
                println(row); flush(stdout)
            finally
                foreach(Mantle.free!,plans)
                foreach(Mantle.free!,(outs...,b))
            end
        end
        foreach(Mantle.free!,(a.q,a.scale))
    end
    results
end
