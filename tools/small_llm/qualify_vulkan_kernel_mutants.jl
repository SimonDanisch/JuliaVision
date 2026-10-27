isdefined(@__MODULE__,:JuliaTaskEval) || include("julia_eval.jl")
isdefined(@__MODULE__,:vulkan_kernel_tasks) || include("vulkan_kernel_tasks.jl")
using Test, JSON3, SHA

function qualify_vulkan_kernel_mutants()
    rows=Any[]
    @testset "Compiled wrong kernels must fail numerical contracts" begin
        for task in vulkan_kernel_tasks()
            replacements=Dict("ki_body_affine"=>("muladd(alpha,x[i],beta)"=>"(muladd(alpha,x[i],beta)+1f0)"),
                "ki_body_clamped_stencil"=>("/3f0"=>"/2f0"),
                "ki_body_column_major_scale"=>("x[i]*scales[row]"=>"(x[i]*scales[row]+1f0)"),
                "ki_body_atomic_max"=>("max(out[labels[i]],x[i])"=>"min(out[labels[i]],x[i])"))
            code=replace(task.reference,replacements[task.id])
            code!=task.reference || error("mutant did not change the reference")
            result=JuliaTaskEval.evaluate(task,code)
            compiled=occursin("Test Failed",result.diagnostic)&&!occursin("Error During Test",result.diagnostic)
            push!(rows,(;id=task.id,rejected=!result.passed,compiled,code_sha256=result.code_sha256,diagnostic=result.diagnostic))
            open(io->JSON3.pretty(io,rows),joinpath(@__DIR__,"vulkan_kernel_mutant_progress_v2.json"),"w")
            @test !result.passed
            @test compiled
        end
    end
    report=(;complete=all(row->row.rejected&&row.compiled,rows),rows,
        task_source_sha256=bytes2hex(sha256(read(joinpath(@__DIR__,"vulkan_kernel_tasks.jl")))),
        sandbox_sha256=bytes2hex(sha256(read(joinpath(@__DIR__,"julia_sandbox.sh")))))
    open(io->JSON3.pretty(io,report),joinpath(@__DIR__,"vulkan_kernel_mutant_validation.json"),"w");report
end
