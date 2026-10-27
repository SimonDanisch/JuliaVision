include("score_rpn_public_example_study.jl")
using Test

function test_rpn_public_example_study()
    @testset "Public-example receipt controls and dead-end scoring" begin
        job=(;id="fixture_enabled",task="fixture",arm="enabled",enabled=true,
            public_cases=[(;input=Float32[2],output=Float32[0.5])],messages=["user"=>"public fixture"])
        p=(;grammar=(;operation_budget=32,max_depth=8),public_example_tolerance=(;atol=0f0,rtol=0f0),
            decoding=(;max_tokens=512,output_chunk=64,seed=20261024,topk=20,temperature=0.6f0,top_p=0.95f0,presence=0f0))
        bank=public_example_bank(job,p)
        k=(;enabled=true,accepted=true,raw_status=0,sampler_status=0,rejected=0,done=1,
            public_examples_matched=true,operation_budget=32,max_depth=8,examples_sha256=public_example_bank_digest(bank),reason="completed")
        c=(;id=job.id,task=job.task,arm=job.arm,messages=job.messages,normal_final_waitidle=true,
            conditions_sha256="conditions",attempt_sha256="attempt",input_sha256=bytes2hex(sha256(JSON3.write(job.messages))),
            answer="half\nfinish",budget_tokens=512,decode_steps_executed=64,decode_submissions=64,eos_output_poll_chunk=64,
            sampling=(;seed=20261024,topk=20,temperature=0.6f0,top_p=0.95f0,presence=0f0,draws=65),constraint=k,truncated=false)
        validate(candidate=c,j=job)=validate_public_example_receipt(candidate,j,p;conditions_sha256="conditions",attempt_sha256="attempt")
        @test validate()===nothing
        for bad in ((;id="other"),(;conditions_sha256="other"),(;attempt_sha256="other"),(;messages=["user"=>"secret"]),
                (;normal_final_waitidle=false),(;input_sha256="other"),(;budget_tokens=513),(;decode_submissions=63),
                (;eos_output_poll_chunk=32),(;decode_steps_executed=0),(;truncated=true),
                (;constraint=merge(k,(;enabled=false))),(;constraint=merge(k,(;examples_sha256="other"))),
                (;constraint=merge(k,(;sampler_status=1))),(;constraint=merge(k,(;public_examples_matched=false))),
                (;constraint=merge(k,(;done=0))),(;constraint=merge(k,(;accepted=false))),
                (;sampling=merge(c.sampling,(;draws=64))),(;sampling=merge(c.sampling,(;topk=19))))
            @test_throws ErrorException validate(merge(c,bad))
        end
        dead=merge(c,(;answer="x0\n",truncated=false,constraint=merge(k,(;accepted=false,done=0,rejected=1,sampler_status=1,public_examples_matched=false,reason="no_admissible_token_or_sampler_rejection"))))
        @test validate(dead)===nothing # retained failure, not an infrastructure exception
        invalid=merge(c,(;constraint=merge(k,(;accepted=false,raw_status=1,sampler_status=1,reason="invalid_raw_neural_logits"))))
        @test validate(invalid)===nothing
        @test recompute_public_example_match("half\nfinish",bank)
        @test !recompute_public_example_match("one\nfinish",bank)
        @test !recompute_public_example_match("half",bank)
        @test !recompute_public_example_match("half\nadd\nfinish",bank)
        @test_throws ErrorException run_rpn_public_example_study() # ROOT guard before GPU
        for (task,fixture) in zip(rpn_public_example_tasks(),private_public_example_fixtures())
            j=(;public_cases=task.public_cases)
            @test recompute_public_example_match(fixture.program,public_example_bank(j,p))
            @test !recompute_public_example_match("zero\nfinish",public_example_bank(j,p))
        end
    end
end
