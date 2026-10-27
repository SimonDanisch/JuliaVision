# ROOT ONLY: mocked reference answers test orchestration, never model quality.
include("score_julia_stencil_transfer.jl")
using Test
function test_julia_stencil_transfer_end_to_end(device)
    original=joinpath(@__DIR__,"julia_stencil_transfer_results/conditions.json")
    mktempdir() do directory
        cp(original,joinpath(directory,"conditions.json"))
        state=julia_stencil_transfer_state(directory);p=state.packet
        for (job,(expression,_)) in zip(p.jobs,private_julia_stencil_transfer_fixtures())
            attempt=joinpath(directory,String(job.id)*"_attempt.json")
            open(io->JSON3.write(io,(;scope="MOCK REFERENCE ONLY")),attempt,"w")
            code="function kernel!(out::Vector{Float32},x::Vector{Float32},n::Int32)\nfor i=1:n\nout[i]="*expression*"\nend\nend"
            d=p.decoding
            sampling=(;seed=d.seed,topk=d.topk,temperature=d.temperature,top_p=d.top_p,presence=d.presence,draws=65)
            row=(;id=job.id,messages=job.messages,normal_final_waitidle=true,conditions_sha256=julia_transfer_digest(state.path),
                attempt_sha256=julia_transfer_digest(attempt),input_sha256=bytes2hex(sha256(JSON3.write(job.messages))),
                code,truncated=false,budget_tokens=512,decode_steps_executed=64,decode_submissions=64,eos_output_poll_chunk=64,sampling)
            open(io->JSON3.pretty(io,row),joinpath(directory,String(job.id)*".json"),"w")
        end
        @testset "Mocked Julia transfer orchestration, NOT intelligence" begin
            @test_throws ErrorException score_julia_stencil_transfer(device;directory)
            @test select_julia_stencil_transfer(;directory).complete
            result=score_julia_stencil_transfer(device;directory)
            @test result.complete
            @test result.total==result.direct_passed==result.hybrid_passed==8
            @test all(r->r.selected=="neural_baseline",result.rows)
            @test all(isfile(joinpath(directory,String(j.id)*"_selection.json")) for j in p.jobs)
        end
    end
end
