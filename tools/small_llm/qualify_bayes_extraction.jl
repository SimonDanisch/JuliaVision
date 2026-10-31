include("bayes_extraction_tasks.jl")
include("bayes_extraction_references.jl")
include("bayes_extraction_admission.jl")
include("portable_evidence.jl")
using Test,SHA
import Mantle

function qualify_bayes_extraction(device;destination=joinpath(@__DIR__,"bayes_extraction_reference_validation.json"))
    occursin("llvmpipe",lowercase(sprint(show,device)))||error("This qualification uses SOFTWARE Vulkan only")
    isfile(destination)&&error("Preserve qualification")
    tasks=bayes_extraction_tasks();references=bayes_extraction_references();rows=Any[]
    tests=@testset "Independent rational answers through shared Bayesian kernels" begin
        @test length(tasks)==length(references)==8
        for (task,reference) in zip(tasks,references)
            answer=JSON3.write(reference.network);admitted=admit_bayes_extraction(answer,"bayes_network")
            @test admitted.accepted
            @test admitted.packed.query==Symbol(task.query)
            @test bayes_extraction_faithful(admitted.network,JSON3.read(answer))
            wrong=JSON3.read(answer,Dict{String,Any});wrong["nodes"][1]["rows"][1]["p"]=reverse(wrong["nodes"][1]["rows"][1]["p"])
            @test !bayes_extraction_faithful(JSON3.read(JSON3.write(wrong)),admitted.network)
            packed=admitted.packed
            execution=PortableEvidence.prepare_bayes(device,packed.factors,packed.query,packed.observed)
            try
                before=Mantle.submissions(device);PortableEvidence.run!(execution);Mantle.waitidle(device)
                @test Mantle.submissions(device)-before==1
                @test only(Array(execution.outputs.status))==0
                probability=Array(execution.outputs.posterior)[only(findall(==(task.target),packed.query_states))]
                @test isapprox(probability,Float64(reference.answer);atol=2e-6,rtol=0)
                push!(rows,(;id=task.id,answer=Float64(reference.answer),shared_kernel_probability=probability,network_sha256=bytes2hex(sha256(answer))))
            finally
                close(execution)
            end
        end
        @test admit_bayes_extraction("{\"probability\":0.25}","direct_probability").probability==0.25
        for invalid in ("{\"probability\":true}","{\"probability\":2}","{\"probability\":0.2,\"explanation\":1}","prefix {\"probability\":0.2}",repeat("[",17)*repeat("]",17))
            @test !admit_bayes_extraction(invalid,"direct_probability").accepted
        end
        for arm in ("direct_probability","bayes_network")
            @test length(bayes_extraction_messages(first(tasks),arm))==2
        end
    end
    counts=Test.get_test_counts(tests)
    result=(;complete=true,checks=counts.passes+counts.cumulative_passes,rows,
        scope="REFERENCE/schema/software-Vulkan arithmetic qualification ONLY; no neural extraction/quality result. Same inference kernels as hardware GPUs, one replay/one submission before explicit grading readback.",
        sources=Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in ("bayes_extraction_tasks.jl","bayes_extraction_references.jl","bayes_extraction_admission.jl","bayes_json.jl","portable_evidence.jl","qualify_bayes_extraction.jl")))
    open(io->JSON3.pretty(io,result),destination,"w");result
end
