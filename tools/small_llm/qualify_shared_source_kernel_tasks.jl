isdefined(@__MODULE__,:shared_source_kernel_tasks) || include("shared_source_kernel_tasks.jl")
isdefined(@__MODULE__,:CheckedTypedPointwiseLowering) || include("typed_pointwise_kernel_lowering.jl")
isdefined(@__MODULE__,:JuliaTaskEval) || include("julia_eval.jl")
using Test,JSON3,SHA

function qualify_shared_source_kernel_tasks()
    output=joinpath(@__DIR__,"shared_source_kernel_reference_validation.json")
    isfile(output) && error("preserve prior qualification")
    cpu_setup="""
    function __bench_launch__(f,initial,args...;threads)
        out=copy(initial);f(out,args...);out
    end
    """
    rows=Any[]
    checks=@testset "Authored shared-source CPU/software-GPU references" begin
        for task in shared_source_kernel_tasks()
            lowered=CheckedTypedPointwiseLowering.lower(task.reference_cpu)
            @test lowered.accepted
            cpu_task=(;id=task.id*"_cpu",category="base",setup=cpu_setup,tests=task.tests)
            cpu=JuliaTaskEval.evaluate(cpu_task,task.reference_cpu)
            @test cpu.passed
            gpu=JuliaTaskEval.evaluate(task,lowered.code)
            @test gpu.passed
            # The same general compiler must preserve a wrong result, not
            # recognize the task and replace it with its correct formula.
            mutant=replace(task.reference_cpu,r"out\[i\]=(.*)"=>s"out[i]=-(\1)")
            @test mutant!=task.reference_cpu
            wrong=CheckedTypedPointwiseLowering.lower(mutant)
            @test wrong.accepted
            rejected=JuliaTaskEval.evaluate(task,wrong.code)
            @test !rejected.passed
            @test occursin("Test Failed",rejected.diagnostic) && !occursin("Error During Test",rejected.diagnostic)
            push!(rows,(;id=task.id,cpu,gpu,mutant=rejected,certificate=lowered.certificate))
            println(task.id," CPU+software-GPU correct, wrong arithmetic rejected");flush(stdout)
        end
    end
    counts=Test.get_test_counts(checks)
    report=(;complete=true,scope="Four trusted authored reference fixtures, each CPU+software Vulkan+wrong-arithmetic mutant. NO model outputs or quality evidence.",
        passed=counts.passes+counts.cumulative_passes,rows,
        source_sha256=Dict(f=>bytes2hex(open(sha256,joinpath(@__DIR__,f))) for f in
            ("shared_source_kernel_tasks.jl","qualify_shared_source_kernel_tasks.jl","typed_pointwise_kernel_lowering.jl",
                "julia_eval.jl","julia_sandbox.sh")))
    open(io->JSON3.pretty(io,report),output,"w");report
end
