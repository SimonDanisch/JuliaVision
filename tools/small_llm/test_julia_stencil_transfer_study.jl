include("julia_stencil_transfer_study.jl")
using Test
function test_julia_stencil_transfer_study()
    @testset "Prospective Julia transfer receipt and public-only selector" begin
        tasks=julia_stencil_transfer_tasks()
        @test length(tasks)==length(unique(t.id for t in tasks))==8
        @test all(t->length(t.public_cases)>=2,tasks)
        @test all(t->length(julia_stencil_transfer_messages(t))==2,tasks)
        job=(;id="mock",messages=["user"=>"public"])
        decoding=(;max_tokens=512,seed=20261025,topk=20,temperature=0.6f0,top_p=0.95f0,presence=0f0,output_chunk=64)
        p=(;decoding)
        sampling=(;seed=decoding.seed,topk=decoding.topk,temperature=decoding.temperature,top_p=decoding.top_p,presence=decoding.presence,draws=65)
        m=(;id=job.id,messages=job.messages,normal_final_waitidle=true,conditions_sha256="c",attempt_sha256="a",
            input_sha256=bytes2hex(sha256(JSON3.write(job.messages))),budget_tokens=512,decode_steps_executed=64,
            decode_submissions=64,sampling,eos_output_poll_chunk=64,truncated=false)
        validate(row)=validate_julia_stencil_receipt(row,job,p;conditions_sha256="c",attempt_sha256="a")
        @test validate(m)===nothing
        for patch in ((;id="wrong"),(;messages=["user"=>"changed"]),(;normal_final_waitidle=false),
                (;conditions_sha256="changed"),(;attempt_sha256="changed"),(;input_sha256="changed"),
                (;budget_tokens=1024),(;decode_steps_executed=513),(;decode_submissions=63),
                (;sampling=(;sampling...,draws=64)),(;sampling=(;sampling...,seed=7)),
                (;eos_output_poll_chunk=128),(;truncated=1))
            @test_throws ErrorException validate(merge(m,patch))
        end
        @test validate((;id=job.id,messages=job.messages,normal_final_waitidle=true,conditions_sha256="c",attempt_sha256="a",inference_error="failed"))===nothing
        bank=RPNExampleSemantics.Bank(tasks[1].public_cases)
        source="function kernel!(out::Vector{Float32},x::Vector{Float32},n::Int32)\nfor i=1:n\nout[i]=abs(x[i])\nend\nend"
        neural=JuliaStencilInterface.lower(source)
        @test neural.accepted
        @test PublicExampleFallback.choose(neural.program,bank,(;found=false,program="")).selected=="neural_baseline"
        proposal=PublicExampleSynthesis.solve(bank;max_cost=8)
        @test proposal.found
        @test PublicExampleFallback.choose("",bank,proposal).selected=="symbolic_public_examples"
        @test PublicExampleFallback.choose("",bank,proposal;infrastructure_ok=false).selected=="none"
    end
end
