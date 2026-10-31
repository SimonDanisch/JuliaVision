# Prompt, executable contract, and a trusted implementation used to validate the
# evaluator. Never include tests/reference in the model prompt.
const JULIA_TASKS = [
    (id="stable_groups", category="base", prompt="Write Julia function solve(keys, values). Return a Vector of Pair mapping each distinct key to the sum of its values, preserving FIRST occurrence order of keys. Inputs have equal length; keys are strings, values are Int. Empty input must return an empty vector. No packages. Code only.",
     tests="""
     @test solve(["b","a","b","c","a"],[3,1,-2,0,8]) == ["b"=>1,"a"=>9,"c"=>0]
     @test isempty(solve(String[],Int[]))
     @test solve(["z","z"],[-2,2]) == ["z"=>0]
     @test solve(["x","y","x","z"],[typemax(Int)÷4,2,1,3]) == ["x"=>(typemax(Int)÷4+1),"y"=>2,"z"=>3]
     """, reference="""
     function solve(keys, values)
         order=String[]; sums=Dict{String,Int}()
         for (k,v) in zip(keys,values)
             haskey(sums,k) || push!(order,k)
             sums[k]=get(sums,k,0)+v
         end
         [k=>sums[k] for k in order]
     end
     """),
    (id="missing_windows", category="base", prompt="Write Julia function solve(x, width) returning the mean of non-missing values in every COMPLETE sliding window of width elements. x contains Float64 or missing. An all-missing window returns missing. width must be positive (throw ArgumentError otherwise); width greater than length(x) returns an empty vector. Do not modify x. No packages. Code only.",
     tests="""
     @test isequal(solve([1.0,missing,3.0,missing],2),[1.0,3.0,3.0])
     @test isequal(solve([missing,missing,2.0],2),[missing,2.0])
     @test solve([1.0,2.0,6.0],3) ≈ [3.0]
     @test isempty(solve(Float64[],1))
     @test isempty(solve([1.0],2))
     @test_throws ArgumentError solve([1.0],0)
     @test_throws ArgumentError solve([1.0],-1)
     original=[1.0,missing,2.0]; solve(original,2); @test isequal(original,[1.0,missing,2.0])
     """, reference="""
     function solve(x,width)
         width>0 || throw(ArgumentError("positive width required"))
         out=Union{Missing,Float64}[]
         for i in 1:max(0,length(x)-width+1)
             total=0.0; count=0
             for j in i:i+width-1
                 if !ismissing(x[j]); total+=x[j]; count+=1; end
             end
             push!(out,count==0 ? missing : total/count)
         end
         out
     end
     """),
    (id="alias_stencil", category="base", prompt="Write Julia function solve!(out, x) computing a 1D three-point mean: out[i]=(x[i-1]+x[i]+x[i+1])/3 for interior indices, with unchanged endpoint values copied from x. Inputs are equal-length ordinary Float64 vectors. Crucially out may be x itself, or overlapping views into one vector; results must use ORIGINAL x values. Handle lengths 0,1,2. Return out; no packages. Code only.",
     tests="""
     for n in (0,1,2,3,17)
         x=Float64.(1:n).^2; expected=copy(x)
         for i in 2:n-1; expected[i]=(x[i-1]+x[i]+x[i+1])/3; end
         y=similar(x); @test solve!(y,x) === y; @test y ≈ expected
         z=copy(x); solve!(z,z); @test z ≈ expected
     end
     a=Float64.(1:20).^2; original=copy(a[1:17]); expected=copy(original)
     for i in 2:16; expected[i]=sum(original[i-1:i+1])/3; end
     solve!(@view(a[2:18]), @view(a[1:17])); @test a[2:18] ≈ expected
     """, reference="""
     function solve!(out,x)
         original=copy(x); copyto!(out,original)
         for i in 2:length(x)-1
             out[i]=(original[i-1]+original[i]+original[i+1])/3
         end
         out
     end
     """),
    (id="ka_tail", category="kernel", prompt="Using KernelAbstractions, write an @kernel named axpby_kernel!(y,x,a,b) and function solve!(y,x,a,b). Launch it using get_backend(y), workgroup size 64, ndrange=length(y), and synchronize the backend before returning y. Kernel computes y[i]=a*x[i]+b*y[i], has an explicit bounds guard, and must work on CPU arrays with lengths 0,1,63,65,129. Inputs are separate Float32 vectors; a,b Float32. Skip launch for empty vectors. Use the standard KernelAbstractions API; code only, including imports.",
     tests="""
     for n in (0,1,63,65,129)
         x=Float32.(1:n)./7; y=Float32.(n:-1:1)./11; expected=2f0.*x .- 0.5f0.*y
         @test solve!(y,x,2f0,-0.5f0) === y
         @test y ≈ expected
     end
     """, reference="""
     using KernelAbstractions
     @kernel function axpby_kernel!(y,x,a,b)
         i=@index(Global,Linear)
         if i<=length(y); @inbounds y[i]=a*x[i]+b*y[i]; end
     end
     function solve!(y,x,a,b)
         isempty(y) && return y
         backend=get_backend(y)
         axpby_kernel!(backend,64)(y,x,a,b;ndrange=length(y))
         KernelAbstractions.synchronize(backend)
         y
     end
     """),
    (id="makie_link", category="makie", prompt="Using Makie (not GLMakie), write function makeplot(x,y) returning (fig, ax1, ax2). Figure size (720,480), two vertically stacked Axis objects; plot y versus x as lines on ax1 and abs.(y) versus x as scatter on ax2. Link only x axes with linkxaxes!. Upper title: Signal; upper ylabel: Amplitude; lower xlabel: Time (s); lower ylabel: Magnitude. No display, save, or backend activation. Code only, including imports.",
     tests="""
     f,a,b=makeplot(collect(1:5),[-2.,1.,0.,3.,-1.])
     @test f isa Makie.Figure; @test a isa Makie.Axis; @test b isa Makie.Axis
     @test a.title[] == "Signal"; @test a.ylabel[] == "Amplitude"
     @test b.xlabel[] == "Time (s)"; @test b.ylabel[] == "Magnitude"
     @test length(a.scene.plots)==1; @test length(b.scene.plots)==1
     @test a.scene.plots[1] isa Makie.Lines; @test b.scene.plots[1] isa Makie.Scatter
     @test Tuple.(a.scene.plots[1][1][]) == collect(zip(1:5,[-2.,1.,0.,3.,-1.]))
     @test Tuple.(b.scene.plots[1][1][]) == collect(zip(1:5,[2.,1.,0.,3.,1.]))
     @test a in b.xaxislinks; @test b in a.xaxislinks
     @test isempty(a.yaxislinks); @test isempty(b.yaxislinks)
     @test Tuple(Makie.size(f.scene)) == (720,480)
     """, reference="""
     using Makie
     function makeplot(x,y)
         f=Figure(size=(720,480))
         a=Axis(f[1,1],title="Signal",ylabel="Amplitude")
         b=Axis(f[2,1],xlabel="Time (s)",ylabel="Magnitude")
         lines!(a,x,y); scatter!(b,x,abs.(y)); linkxaxes!(a,b)
         (f,a,b)
     end
     """),
    (id="makie_band", category="makie", prompt="Using Makie (not GLMakie), write makeplot(x,mu,sigma) returning (fig,ax). Figure size (640,400), axis xlabel Time, ylabel Value, title Estimate. Plot a semi-transparent blue band between mu .- sigma and mu .+ sigma, then a blue mean line labelled Mean. Add axislegend(ax,position=:lt). Do not display, save, or activate a backend. Code only including imports.",
     tests="""
     f,a=makeplot(collect(1:4),[1.,2.,3.,2.],fill(0.5,4))
     @test f isa Makie.Figure; @test a isa Makie.Axis
     @test a.title[] == "Estimate"; @test a.xlabel[] == "Time"; @test a.ylabel[] == "Value"
     @test length(a.scene.plots)==2
     @test a.scene.plots[1] isa Makie.Band
     @test a.scene.plots[2] isa Makie.Lines
     @test a.scene.plots[2].label[] == "Mean"
     @test Tuple.(a.scene.plots[1][1][]) == collect(zip(1:4,[0.5,1.5,2.5,1.5]))
     @test Tuple.(a.scene.plots[1][2][]) == collect(zip(1:4,[1.5,2.5,3.5,2.5]))
     @test Tuple.(a.scene.plots[2][1][]) == collect(zip(1:4,[1.,2.,3.,2.]))
     c=Makie.to_color(a.scene.plots[1].color[])
     @test (c.r,c.g,c.b)==(0,0,1); @test 0<c.alpha<1
     @test Makie.to_color(a.scene.plots[2].color[]) == Makie.to_color(:blue)
     @test any(x->x isa Makie.Legend,f.content)
     @test Tuple(Makie.size(f.scene)) == (640,400)
     """, reference="""
     using Makie
     function makeplot(x,mu,sigma)
         f=Figure(size=(640,400)); a=Axis(f[1,1],xlabel="Time",ylabel="Value",title="Estimate")
         band!(a,x,mu.-sigma,mu.+sigma,color=(:blue,0.25))
         lines!(a,x,mu,color=:blue,label="Mean"); axislegend(a,position=:lt)
         (f,a)
     end
     """),
]
