# Additional diagnostic tasks, kept separate from the first frozen pilot.
const JULIA_ADVANCED_TASKS = [
    (id="row_softmax",category="base",prompt="Write Julia function solve(x::Matrix{Float64}) returning a Float64 matrix of row-wise softmax probabilities without modifying x. Avoid overflow for large finite entries. If a row contains +Inf, split its probability equally among the +Inf entries and give all others zero. An all -Inf row is uniform. Throw ArgumentError if ANY entry is NaN, even in a row also containing +Inf. Preserve the shape of empty matrices. No packages; code only.",
     tests="""
     x=[1000. 1001. 1002.; -1000. -1001. -1002.]; saved=copy(x); p=solve(x)
     @test size(p)==size(x); @test x==saved
     @test vec(sum(p,dims=2)) ≈ ones(2)
     @test p[1,:] ≈ exp.([-2.,-1.,0.])./sum(exp.([-2.,-1.,0.]))
     @test solve([Inf 1. Inf; -Inf -Inf -Inf]) ≈ [0.5 0. 0.5; 1/3 1/3 1/3]
     @test solve([3. -Inf -Inf]) == [1. 0. 0.]
     @test_throws ArgumentError solve([Inf NaN 0.])
     @test_throws ArgumentError solve([0. 1.; 2. NaN])
     @test size(solve(zeros(0,3)))==(0,3)
     @test size(solve(zeros(4,0)))==(4,0)
     """, reference="""
     function solve(x::Matrix{Float64})
         any(isnan,x) && throw(ArgumentError("NaN input"))
         p=similar(x); n=size(x,2); n==0 && return p
         for i in axes(x,1)
             row=@view x[i,:]; ninf=count(==(Inf),row)
             if ninf>0
                 p[i,:].=ifelse.(row.==Inf,1/ninf,0.0)
             elseif all(==(-Inf),row)
                 p[i,:].=1/n
             else
                 p[i,:].=exp.(row.-maximum(row)); p[i,:]./=sum(@view p[i,:])
             end
         end
         p
     end
     """),
    (id="ki_launch",category="base",prompt="Using current KernelInterface.jl, write function solve!(kernel, out, x). kernel is an ALREADY COMPILED KernelInterface.Kernel whose backend is stored in kernel.backend; its body doubles x into out. Validate equal lengths or throw DimensionMismatch BEFORE launch. Launch it with workgroup size 64 and ndrange=length(x), using the KernelInterface kernel-call interface (not CUDA, not KernelAbstractions @kernel). KernelInterface rounds ndrange up automatically, and zero ndrange launches nothing. Synchronize kernel.backend after launch; return the exact out object. Code only including imports.",
     setup="""
     import KernelInterface as KI
     mutable struct MockJuliaBackend <: KI.Backend
         calls::Vector{Any}
         waits::Int
     end
     KI.max_work_group_dims(::MockJuliaBackend)=(1024,1024,64)
     KI.max_work_group_size(::KI.Kernel{MockJuliaBackend})=1024
     function KI.launch(k::KI.Kernel{MockJuliaBackend},groups::Dims{3},items::Dims{3},args::Tuple;kwargs...)
         isempty(kwargs) || error("unsupported launch keywords")
         push!(k.backend.calls,(groups,items,length(args)))
         out,x=args; out.=2 .*x; nothing
     end
     KI.synchronize(b::MockJuliaBackend)=(b.waits+=1;nothing)
     """, tests="""
     backend=MockJuliaBackend(Any[],0); k=KI.Kernel(backend,nothing)
     for n in (0,1,63,65,129)
         x=Float32.(1:n); out=similar(x)
         before=length(backend.calls)
         @test solve!(k,out,x) === out; @test out==2 .*x
         @test length(backend.calls)==before+(n>0)
         if n>0; @test last(backend.calls)==((cld(n,64),1,1),(64,1,1),2); end
     end
     @test backend.waits==5
     before=length(backend.calls)
     @test_throws DimensionMismatch solve!(k,zeros(2),zeros(3))
     @test length(backend.calls)==before
     """, reference="""
     import KernelInterface as KI
     function solve!(kernel,out,x)
         length(out)==length(x) || throw(DimensionMismatch("lengths differ"))
         kernel(out,x;workgroupsize=64,ndrange=length(x))
         KI.synchronize(kernel.backend)
         out
     end
     """),
    (id="makie_update",category="makie",prompt="Using current Makie.jl, write function solve!(plot, x, y, colors) for an EXISTING scatter plot originally constructed with separate x and y vectors and numeric colors. Replace BOTH positional input arrays and the color attribute coherently in ONE Makie.update! call, so point count can change without intermediate length mismatches. Use Makie's positional argument input names, not separate Observable assignments. Return the exact same plot. Do not recreate plots, figures or axes; code only including imports.",
     tests="""
     f=Figure(); a=Axis(f[1,1]); p=scatter!(a,[0.,1.],[2.,3.],color=[0.,1.])
     @test solve!(p,[1.,2.,4.],[7.,5.,6.],[0.2,0.4,0.8]) === p
     @test Tuple.(p[1][])==[(1.,7.),(2.,5.),(4.,6.)]
     @test p.color[] ≈ [0.2,0.4,0.8]
     @test length(a.scene.plots)==1
     @test solve!(p,[9.],[8.],[0.5]) === p
     @test Tuple.(p[1][])==[(9.,8.)]; @test p.color[] == [0.5]
     function count_update_calls(e)
         e isa Expr || return 0
         f=e.head==:call && !isempty(e.args) ? first(e.args) : nothing
         named=f===:update! || (f isa Expr && f.head==:. && last(f.args)==QuoteNode(:update!))
         Int(named)+sum(count_update_calls,e.args;init=0)
     end
     @test count_update_calls(Meta.parseall(read("/input/candidate.jl",String)))==1
     """,reference="""
     using Makie
     function solve!(plot,x,y,colors)
         Makie.update!(plot;arg1=x,arg2=y,color=colors)
         plot
     end
     """),
]
