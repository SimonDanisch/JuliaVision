include("score_pubmedqa_reading.jl")
using Test
function test_pubmedqa_receipt()
    messages=[Dict("system"=>"research classification"),Dict("user"=>"public question")]
    job=(;id="a",messages)
    decoding=(;max_tokens=128,seed=20261025,topk=20,temperature=0.6f0,top_p=0.95f0,presence=0f0,output_chunk=64)
    p=(;decoding)
    sampling=(;seed=decoding.seed,topk=decoding.topk,temperature=decoding.temperature,top_p=decoding.top_p,presence=decoding.presence,draws=65)
    receipt=(;id="a",messages,normal_final_waitidle=true,conditions_sha256="c",attempt_sha256="a",
        input_sha256=bytes2hex(sha256(JSON3.write(messages))),budget_tokens=128,decode_steps_executed=64,
        decode_submissions=64,sampling,eos_output_poll_chunk=64,truncated=false,answer="yes")
    check(x)=validate_pubmedqa_receipt(x,job,p;condition_hash="c",attempt_hash="a")
    @testset "Research classification receipts bind public prompts and budgets" begin
        @test check(receipt)===nothing
        for change in ((;id="b"),(;messages=reverse(messages)),(;normal_final_waitidle=false),
                (;conditions_sha256="bad"),(;attempt_sha256="bad"),(;input_sha256="bad"),
                (;budget_tokens=129),(;decode_steps_executed=0),(;decode_steps_executed=129),
                (;decode_submissions=63),(;eos_output_poll_chunk=128),(;truncated=1),
                (;sampling=(;sampling...,draws=64)),(;sampling=(;sampling...,seed=1)),
                (;sampling=(;sampling...,temperature=1f0)))
            @test_throws ErrorException check((;receipt...,change...))
        end
        @test check((;id="a",messages,normal_final_waitidle=true,conditions_sha256="c",attempt_sha256="a",inference_error="preserved"))===nothing
    end
end
