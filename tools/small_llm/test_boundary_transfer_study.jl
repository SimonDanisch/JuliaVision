include("score_boundary_transfer.jl")
using Test
function test_boundary_transfer_study(device)
    @testset "Three-arm transfer policy, public-only decisions" begin
        bank=RPNExampleSemantics.Bank([(;input=Float32[2,-1,0.5,-2],output=Float32[0,3,0,2.5]),
            (;input=Float32[-0.75],output=Float32[0])])
        wrong="zero\nx_prev_periodic\nx0\nsub\nmax\nfinish"
        choice=boundary_transfer_decision(wrong,bank,:clamped)
        @test choice.example.selected=="neural_baseline"
        @test choice.example.program==wrong
        @test choice.boundary.selected=="symbolic_public_boundary"
        @test StencilBoundaryContract.accepts(choice.boundary.program,:clamped)
        @test PublicExampleFallback.matches(choice.boundary.program,bank)
        unknown=boundary_transfer_decision(wrong,bank,:unknown)
        @test unknown.example==unknown.boundary
        failed=boundary_transfer_decision(wrong,bank,:clamped;infrastructure_ok=false)
        @test failed.example.selected==failed.boundary.selected=="none"
        @test failed.plain_proposal.candidates==failed.boundary_proposal.candidates==0
        empty=boundary_transfer_decision("",bank,:clamped)
        @test empty.example.selected=="symbolic_public_examples"
        @test empty.boundary.selected=="symbolic_public_boundary"
        @test_throws ArgumentError boundary_transfer_decision(wrong,bank,:invalid)
        @test_throws ArgumentError boundary_transfer_decision(wrong,bank,:clamped;infrastructure_ok=1)
        @test length(boundary_transfer_tasks())==8
        @test all(t->StencilBoundaryContract.recognize(t.specification)!=:unknown,boundary_transfer_tasks())
    end
    mktempdir() do directory
        p=freeze_boundary_transfer(;directory)
        conditions_sha256=julia_transfer_digest(joinpath(directory,"conditions.json"))
        for (job,(expression,_)) in zip(p.jobs,private_boundary_transfer_fixtures())
            path=joinpath(directory,String(job.id)*"_attempt.json")
            open(io->JSON3.write(io,(;scope="MOCK reference qualification ONLY")),path,"w")
            source="function kernel!(out::Vector{Float32},x::Vector{Float32},n::Int32)\nfor i=1:n\nout[i]="*expression*"\nend\nend"
            d=p.decoding;sampling=(;seed=d.seed,topk=d.topk,temperature=d.temperature,top_p=d.top_p,presence=d.presence,draws=65)
            row=(;id=job.id,messages=job.messages,normal_final_waitidle=true,conditions_sha256,
                attempt_sha256=julia_transfer_digest(path),input_sha256=bytes2hex(sha256(JSON3.write(job.messages))),
                code=source,truncated=false,budget_tokens=512,decode_steps_executed=64,decode_submissions=64,eos_output_poll_chunk=64,sampling)
            open(io->JSON3.pretty(io,row),joinpath(directory,String(job.id)*".json"),"w")
        end
        @testset "Mocked reference pipeline, NOT model intelligence" begin
            @test_throws ErrorException score_boundary_transfer(device;directory)
            @test select_boundary_transfer(;directory).complete
            result=score_boundary_transfer(device;directory)
            @test result.complete&&result.total==8
            @test all(==(8),values(result.passed))
            @test all(r->r.example_selection==r.boundary_selection=="neural_baseline",result.rows)
        end
    end
end
