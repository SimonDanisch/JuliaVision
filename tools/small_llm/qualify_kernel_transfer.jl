isdefined(@__MODULE__, :kernel_transfer_tasks) || include("kernel_transfer_tasks.jl")
isdefined(@__MODULE__, :JuliaTaskEval) || include("julia_eval.jl")
using Test, JSON3, SHA

function qualify_kernel_transfer()
    output=joinpath(@__DIR__,"kernel_transfer_validation.json")
    isfile(output) && error("preserve previous qualification")
    mutations=Dict(
        "transfer_clamped_gather"=>("x[clamp(indices[i],Int32(1),m)]"=>"x[clamp(indices[i],Int32(1),m)]+1f0"),
        "transfer_sincos_pair"=>("out_cos[i]=cos(x[i])"=>"out_cos[i]=sin(x[i])"),
        "transfer_flat_transpose"=>("x[c+r*rows+Int32(1)]"=>"x[i]"),
        "transfer_atomic_sum"=>("+= x[i]"=>"+= 2f0*x[i]"))
    rows=Any[]
    save()=open(io->JSON3.pretty(io,(;complete=false,rows)),output,"w")
    @testset "Prospective kernel task harness, software Vulkan only" begin
        for task in kernel_transfer_tasks()
            gold=JuliaTaskEval.evaluate(task,task.reference)
            push!(rows,(;id=task.id,kind="reference",result=gold));save()
            @test gold.passed
            mutant=replace(task.reference,mutations[task.id])
            @test mutant!=task.reference
            bad=JuliaTaskEval.evaluate(task,mutant)
            push!(rows,(;id=task.id,kind="arithmetic_mutant",result=bad));save()
            @test !bad.passed
            @test occursin("Test Failed",bad.diagnostic)
            @test !occursin("Error During Test",bad.diagnostic)
        end
    end
    report=(;complete=true,scope="authored prospective tasks: trusted references and numerical mutants; software Vulkan, no model outputs",rows,
        source_sha256=Dict(file=>bytes2hex(sha256(read(joinpath(@__DIR__,file)))) for file in
            ("kernel_transfer_tasks.jl","qualify_kernel_transfer.jl","vulkan_kernel_tasks.jl","julia_eval.jl","julia_sandbox.sh")))
    open(io->JSON3.pretty(io,report),output,"w");report
end
