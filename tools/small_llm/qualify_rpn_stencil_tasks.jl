include("rpn_stencil_tasks.jl")
isdefined(@__MODULE__,:RPNPointwiseInterface)||include("rpn_pointwise_interface.jl")
isdefined(@__MODULE__,:JuliaTaskEval)||include("julia_eval.jl")
isdefined(@__MODULE__,:VULKAN_KERNEL_SETUP)||include("vulkan_kernel_tasks.jl")
using JSON3,SHA

# PRIVATE references and wrong-arithmetic programs. Do not transfer to inference.
function private_rpn_stencil_fixtures()
    [
        (;program="x_next_periodic\nx_prev_periodic\nsub\nhalf\nmul\nfinish",mutant="x_prev_periodic\nx_next_periodic\nsub\nhalf\nmul\nfinish",expected="0.5f0*(x[mod1(i+1,n)]-x[mod1(i-1,n)])"),
        (;program="x0\nx_prev_periodic\nx_next_periodic\nadd\ntwo\nx0\nmul\nsub\nhalf\nmul\nadd\nfinish",mutant="x0\nx_prev_periodic\nx_next_periodic\nadd\ntwo\nx0\nmul\nadd\nhalf\nmul\nadd\nfinish",expected="x[i]+0.5f0*(x[mod1(i-1,n)]+x[mod1(i+1,n)]-2f0*x[i])"),
        (;program="x_prev_clamped\nx0\nsub\nsquare\nx_next_clamped\nx0\nsub\nsquare\nadd\nfinish",mutant="x_prev_clamped\nx0\nsub\nsquare\nx_next_clamped\nx0\nsub\nsquare\nsub\nfinish",expected="(x[max(1,i-1)]-x[i])^2+(x[min(n,i+1)]-x[i])^2"),
        (;program="x_next_periodic\nx0\nsub\nzero\nmax\nfinish",mutant="x_next_periodic\nx0\nsub\nzero\nmin\nfinish",expected="max(x[mod1(i+1,n)]-x[i],0f0)"),
        (;program="x0\nzero\nmax\none\nmin\nfinish",mutant="x0\nzero\nmax\none\nmax\nfinish",expected="clamp(x[i],0f0,1f0)"),
        (;program="x_prev_clamped\nx0\nmax\nx_next_clamped\nmax\nx_prev_clamped\nx0\nmin\nx_next_clamped\nmin\nsub\nfinish",mutant="x_prev_clamped\nx0\nmax\nx_next_clamped\nmax\nx_prev_clamped\nx0\nmin\nx_next_clamped\nmin\nadd\nfinish",expected="max(x[max(1,i-1)],x[i],x[min(n,i+1)])-min(x[max(1,i-1)],x[i],x[min(n,i+1)])"),
        (;program="x0\nx_prev_periodic\nsub\nx_next_periodic\nx0\nsub\nmin\nzero\nmax\nfinish",mutant="x0\nx_prev_periodic\nsub\nx_next_periodic\nx0\nsub\nmax\nzero\nmax\nfinish",expected="max(min(x[i]-x[mod1(i-1,n)],x[mod1(i+1,n)]-x[i]),0f0)"),
        (;program="x_prev_periodic\nx_next_periodic\nadd\nsquare\nx0\nsquare\nsub\nfinish",mutant="x_prev_periodic\nx_next_periodic\nadd\nsquare\nx0\nsquare\nadd\nfinish",expected="(x[mod1(i-1,n)]+x[mod1(i+1,n)])^2-x[i]^2")
    ]
end

function private_rpn_stencil_task(task,fixture)
    tests="""
    for n in (0,1,2,3,7,31,65), family in 1:3
        x=Float32[family==1 ? sin(Float32(i)*0.71f0)*2.3f0+0.2f0 :
            family==2 ? (isodd(i) ? -1.5f0 : 2.25f0) : Float32(i)*0.17f0-0.8f0 for i in 1:max(n,1)]
        initial=fill(-913f0,n+7);expected=copy(initial)
        for i in 1:n;expected[i]=$(fixture.expected);end
        cpu=copy(initial);kernel!(cpu,x,Int32(n))
        device=__bench_launch__(kernel!,initial,x,Int32(n);threads=n)
        @test cpu ≈ expected atol=2f-5 rtol=2f-5
        @test device ≈ expected atol=2f-5 rtol=2f-5
        @test cpu[n+1:end]==initial[n+1:end]
        @test device[n+1:end]==initial[n+1:end]
    end
    """
    setup=replace(VULKAN_KERNEL_SETUP,
        "plan=Mantle.record!(Mantle.Plan(graph;budget=Inf));Mantle.run!(plan)"=>
        "plan=Mantle.record!(Mantle.Plan(graph;budget=Inf));before=Mantle.submissions(__bench_device);Mantle.run!(plan);@assert Mantle.submissions(__bench_device)-before==1")
    setup!=VULKAN_KERNEL_SETUP||error("one-submission assertion not installed")
    (;id=task.id,category="vulkan_kernel",setup,tests)
end

function qualify_rpn_stencil_tasks()
    destination=joinpath(@__DIR__,"rpn_stencil_reference_validation.json")
    ispath(destination)&&error("preserve prior qualification")
    tasks=rpn_stencil_tasks();fixtures=private_rpn_stencil_fixtures();length(tasks)==length(fixtures)==8||error("cohort length")
    sources=("rpn_stencil_tasks.jl","qualify_rpn_stencil_tasks.jl","rpn_pointwise_interface.jl","rpn_prefix_grammar.jl",
        "typed_pointwise_kernel_lowering.jl","julia_eval.jl","julia_sandbox.sh","vulkan_kernel_tasks.jl")
    hashes=Dict(f=>bytes2hex(open(sha256,joinpath(@__DIR__,f))) for f in sources)
    rows=Any[]
    save(complete)=open(io->JSON3.pretty(io,(;complete,scope="PRIVATE authored reference/mutant CPU + software-Vulkan qualification; NOT model quality",source_sha256=hashes,rows)),destination,"w")
    save(false)
    for (task,fixture) in zip(tasks,fixtures)
        private=private_rpn_stencil_task(task,fixture)
        lower=RPNPointwiseInterface.lower(fixture.program);mutant=RPNPointwiseInterface.lower(fixture.mutant)
        lower.accepted&&mutant.accepted||error("trusted program/mutant syntax rejection")
        correct=JuliaTaskEval.evaluate(private,lower.serial_code*"\n"*lower.code)
        incorrect=JuliaTaskEval.evaluate(private,mutant.serial_code*"\n"*mutant.code)
        push!(rows,(;id=task.id,correct,incorrect));save(false)
        correct.passed||error("reference failed $(task.id): $(correct.diagnostic)")
        !incorrect.passed&&occursin("Test Failed",incorrect.diagnostic)&&!occursin("Error During Test",incorrect.diagnostic)||error("mutant not rejected mathematically: $(task.id)")
        println(task.id," reference PASS + admitted wrong arithmetic rejected");flush(stdout)
    end
    save(true);(;complete=true,fixtures=length(rows))
end
