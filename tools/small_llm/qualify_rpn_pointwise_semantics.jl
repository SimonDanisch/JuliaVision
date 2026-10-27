isdefined(@__MODULE__,:RPNPointwiseInterface)||include("rpn_pointwise_interface.jl")
isdefined(@__MODULE__,:JuliaTaskEval)||include("julia_eval.jl")
isdefined(@__MODULE__,:VULKAN_KERNEL_SETUP)||include("vulkan_kernel_tasks.jl")
using JSON3,SHA

"""PRIVATE trusted programs/mutants ONLY. Never a generation input."""
function qualify_rpn_pointwise_semantics()
    destination=joinpath(@__DIR__,"rpn_pointwise_semantics_validation.json")
    isfile(destination)&&error("preserve semantic qualification")
    fixtures=(
        (;id="ordered_subtraction",program="x0\nx_prev_periodic\nsub\nfinish",mutant="x_prev_periodic\nx0\nsub\nfinish",expected="x[i]-x[mod1(i-1,n)]"),
        (;id="float_constants",program="x0\ntwo\nmul\none\nsub\nfinish",mutant="x0\ntwo\nmul\none\nadd\nfinish",expected="2f0*x[i]-1f0"),
        (;id="clamped_neighbors",program="x_prev_clamped\nx_next_clamped\nmin\nfinish",mutant="x_prev_clamped\nx_next_clamped\nmax\nfinish",expected="min(x[max(1,i-1)],x[min(n,i+1)])"),
        (;id="unary_square",program="x_prev_periodic\nx0\nsub\nsquare\nfinish",mutant="x_prev_periodic\nx0\nsub\nsquare\nneg\nfinish",expected="(x[mod1(i-1,n)]-x[i])^2"),
        (;id="unary_sign_max",program="x0\nneg\none\nmax\nfinish",mutant="x0\nneg\none\nmin\nfinish",expected="max(-x[i],1f0)"))
    rows=Any[]
    sources=("rpn_prefix_grammar.jl","rpn_pointwise_interface.jl","qualify_rpn_pointwise_semantics.jl",
        "typed_pointwise_kernel_lowering.jl","julia_eval.jl","julia_sandbox.sh","vulkan_kernel_tasks.jl")
    hashes=Dict(f=>bytes2hex(open(sha256,joinpath(@__DIR__,f))) for f in sources)
    save(complete)=open(io->JSON3.pretty(io,(;complete,scope="PRIVATE trusted/mutant compiler semantics qualification on CPU + SOFTWARE Vulkan, NOT model quality",rows,source_sha256=hashes)),destination,"w")
    save(false)
    setup=replace(VULKAN_KERNEL_SETUP,
        "plan=Mantle.record!(Mantle.Plan(graph;budget=Inf));Mantle.run!(plan)"=>
        "plan=Mantle.record!(Mantle.Plan(graph;budget=Inf));before=Mantle.submissions(__bench_device);Mantle.run!(plan);@assert Mantle.submissions(__bench_device)-before==1")
    setup!=VULKAN_KERNEL_SETUP||error("submission assertion insertion failed")
    for fixture in fixtures
        tests="""
        for n in (0,1,2,3,31,65)
            x=Float32[sin(Float32(i)*0.71f0)+0.2f0*cos(Float32(i)*0.37f0) for i in 1:max(n,1)]
            initial=fill(-913f0,n+7);expected=copy(initial)
            for i in 1:n;expected[i]=$(fixture.expected);end
            cpu=copy(initial);kernel!(cpu,x,Int32(n))
            device=__bench_launch__(kernel!,initial,x,Int32(n);threads=n)
            @test cpu ≈ expected atol=2f-6 rtol=2f-6
            @test device ≈ expected atol=2f-6 rtol=2f-6
            @test cpu[n+1:end]==initial[n+1:end]
            @test device[n+1:end]==initial[n+1:end]
        end
        """
        task=(;id="rpn_"*fixture.id,category="vulkan_kernel",setup,tests)
        lower=RPNPointwiseInterface.lower(fixture.program)
        lower.accepted||error("trusted program rejected")
        correct=JuliaTaskEval.evaluate(task,lower.serial_code*"\n"*lower.code)
        mutant=RPNPointwiseInterface.lower(fixture.mutant)
        mutant.accepted||error("mutant must pass syntax admission")
        incorrect=JuliaTaskEval.evaluate(task,mutant.serial_code*"\n"*mutant.code)
        push!(rows,(;id=fixture.id,correct,incorrect))
        save(false)
        correct.passed||error("trusted semantic failure: $(fixture.id): $(correct.diagnostic)")
        !incorrect.passed&&occursin("Test Failed",incorrect.diagnostic)&&!occursin("Error During Test",incorrect.diagnostic)||
            error("wrong arithmetic not rejected as a semantic failure")
        println(fixture.id," CPU/SOFTWARE reference PASS + wrong arithmetic rejected");flush(stdout)
    end
    save(true)
    (;complete=true,fixtures=length(rows))
end
