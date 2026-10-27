include("audit_identical_prompt_outputs.jl")
using Test
function test_identical_prompt_outputs()
    messages=[Dict("user"=>"public question")]
    row(id)=(;id,messages,normal_final_waitidle=true,input_sha256=bytes2hex(sha256(JSON3.write(messages))),
        sampling=(;seed=1,topk=20,temperature=0.6,top_p=0.95,presence=0),budget_tokens=128,eos_output_poll_chunk=64,
        answer="yes",input_tokens=10,generated_tokens=1,truncated=false)
    @testset "Repeated prompt audit distinguishes seeds from discrepancies" begin
        r=identical_prompt_groups([row("a"),row("b")])
        @test r.all_agree&&r.group_count==r.comparisons==1
        r=identical_prompt_groups([row("a"),(;row("b")...,answer="no")])
        @test !r.all_agree
        r=identical_prompt_groups([row("a"),(;row("b")...,sampling=(;row("b").sampling...,seed=2))])
        @test r.group_count==0&&r.comparisons==0
        @test_throws ArgumentError identical_prompt_groups([(;row("a")...,input_sha256="bad")])
        @test_throws ArgumentError identical_prompt_groups([(;row("a")...,normal_final_waitidle=false)])
    end
end
