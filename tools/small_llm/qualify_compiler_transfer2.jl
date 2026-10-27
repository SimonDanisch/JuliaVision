isdefined(@__MODULE__,:CanonicalScalarInterface) || include("canonical_scalar_interface.jl")
isdefined(@__MODULE__,:compiler_transfer2_tasks) || include("compiler_transfer2_tasks.jl")
isdefined(@__MODULE__,:JuliaTaskEval) || include("julia_eval.jl")
using JSON3,SHA,Test

function verify_transfer2_frozen_compiler()
    path=joinpath(@__DIR__,"compiler_transfer2_frozen_compiler.json")
    p=JSON3.read(read(path,String));p.complete||error("compiler not frozen")
    for (file,hash) in pairs(p.source_sha256)
        bytes2hex(open(sha256,joinpath(@__DIR__,String(file))))==hash||error("compiler modified after freeze: $file")
    end
    path
end

function qualify_compiler_transfer2()
    frozen=verify_transfer2_frozen_compiler();rows=Any[]
    checks=@testset "Fresh transfer2 PRIVATE software-Vulkan references" begin
        for task in compiler_transfer2_tasks()
            r=CanonicalScalarInterface.lower(task.reference_value,task.interface_id)
            @test r.accepted
            correct=JuliaTaskEval.evaluate(task,r.code)
            @test correct.passed
            wrong=CanonicalScalarInterface.lower("-("*task.reference_value*")",task.interface_id)
            @test wrong.accepted
            rejected=JuliaTaskEval.evaluate(task,wrong.code)
            @test !rejected.passed
            @test occursin("Test Failed",rejected.diagnostic)&&!occursin("Error During Test",rejected.diagnostic)
            push!(rows,(;id=task.id,correct,rejected))
            println(task.id," reference PASS, wrong arithmetic rejected");flush(stdout)
        end
    end
    n=Test.get_test_counts(checks)
    packet=(;complete=true,scope="Trusted references/mutants ONLY, before any new task generation; no model quality",passed=n.passes+n.cumulative_passes,rows,
        frozen_compiler_sha256=bytes2hex(open(sha256,frozen)),
        source_sha256=Dict(f=>bytes2hex(open(sha256,joinpath(@__DIR__,f))) for f in ("compiler_transfer2_tasks.jl","qualify_compiler_transfer2.jl")))
    output=joinpath(@__DIR__,"compiler_transfer2_reference_validation.json");isfile(output)&&error("preserve qualification")
    open(io->JSON3.pretty(io,packet),output,"w");packet
end
