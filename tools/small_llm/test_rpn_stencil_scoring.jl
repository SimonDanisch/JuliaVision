include("score_rpn_stencil_study.jl")
using Test
function test_rpn_stencil_scoring()
    @testset "RPN paired score receipts fail closed, no generation" begin
        job=(;id="test_disabled",task="test",arm="disabled",enabled=false,messages=["user"=>"public task"])
        p=(;decoding=(;max_tokens=512,seed=1,topk=20,temperature=0.6f0,top_p=0.95f0,presence=0f0),grammar=(;operation_budget=32,max_depth=8))
        candidate=(;id=job.id,task=job.task,arm=job.arm,messages=job.messages,conditions_sha256="conditions",attempt_sha256="attempt",
            normal_final_waitidle=true,budget_tokens=512,decode_submissions=64,decode_steps_executed=64,truncated=false,
            input_sha256=bytes2hex(sha256(JSON3.write(job.messages))),
            sampling=(;p.decoding...,draws=65),grammar=(;p.grammar...,enabled=false,rejected=0,done=0))
        check(c,j=job)=validate_rpn_stencil_receipt(c,j,p;conditions_sha256="conditions",attempt_sha256="attempt")
        @test check(candidate)===nothing
        for mutation in ((;id="wrong"),(;task="wrong"),(;arm="enabled"),(;messages=["user"=>"hidden answer"]),
                (;conditions_sha256="wrong"),(;attempt_sha256="wrong"),(;normal_final_waitidle=false),(;budget_tokens=256),
                (;decode_submissions=63),(;decode_submissions=513,decode_steps_executed=513),
                (;sampling=(;candidate.sampling...,draws=64)),(;sampling=(;candidate.sampling...,seed=2)),
                (;grammar=(;candidate.grammar...,enabled=true)),(;grammar=(;candidate.grammar...,rejected=1)),(;input_sha256="wrong"))
            @test_throws ErrorException check((;candidate...,mutation...))
        end
        enabled=(;job...,id="test_enabled",arm="enabled",enabled=true)
        enabled_candidate=(;candidate...,id=enabled.id,arm=enabled.arm,grammar=(;candidate.grammar...,enabled=true,done=1))
        @test check(enabled_candidate,enabled)===nothing
        @test_throws ErrorException check((;enabled_candidate...,truncated=true),enabled)
        @test_throws ErrorException check((;enabled_candidate...,grammar=(;enabled_candidate.grammar...,done=0)),enabled)
        @test !RPNPointwiseInterface.parse_program("```\nx0\nfinish\n```").accepted
        @test RPNPointwiseInterface.parse_program("x0\nfinish").accepted
        @test check((;candidate...,inference_error="logical failure"))===nothing
    end
end
