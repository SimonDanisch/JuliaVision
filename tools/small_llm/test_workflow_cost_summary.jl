include("workflow_cost_summary.jl")
using Test
function test_workflow_cost_summary()
    jobs=[(;id="a",arm="direct"),(;id="b",arm="public_docs")]
    packet=(;jobs,condition_legend=(;direct="self_review",public_docs="tool_revision"))
    primary=(;complete=true,rows=[(;id="a",status="passed"),(;id="b",status="failed")])
    receipt(id)=(;id,normal_final_waitidle=true,input_tokens=100,generated_tokens=15,decode_steps_executed=64,
        decode_submissions=64,elapsed_seconds=2.5,truncated=false)
    receipts=Dict(id=>receipt(id) for id in ("a","b"))
    @testset "Quality and incremental timed cost remain distinct" begin
        result=workflow_cost_totals(packet,primary,receipts)
        @test result.arms[1].passed==1
        @test result.arms[1].label=="self_review"
        @test result.arms[2].label=="tool_revision"
        @test result.arms[1].timed_seconds_per_pass==2.5
        @test result.arms[2].timed_seconds_per_pass===nothing
        @test result.arms[1].generated_tokens==15
        @test result.arms[1].decode_steps==64
        @test occursin("NOT",result.timer)
        @test occursin("incremental",result.baseline_cost)
        @test_throws ArgumentError workflow_cost_totals(packet,(;primary...,complete=false),receipts)
        @test_throws ArgumentError workflow_cost_totals((;packet...,jobs=vcat(jobs,[jobs[1]])),primary,receipts)
        @test_throws ArgumentError workflow_cost_totals(packet,primary,Dict("a"=>receipt("a")))
        for replacement in ((;normal_final_waitidle=false),(;id="wrong"),(;generated_tokens=65),
                (;decode_submissions=63),(;input_tokens=-1),(;elapsed_seconds=NaN),(;elapsed_seconds=-1))
            invalid=copy(receipts);invalid["b"]=(;receipt("b")...,replacement...)
            @test_throws ArgumentError workflow_cost_totals(packet,primary,invalid)
        end
        errorrows=Dict{String,Any}(receipts)
        errorrows["b"]=(;id="b",normal_final_waitidle=true,inference_error="failure")
        result=workflow_cost_totals(packet,primary,errorrows)
        @test result.arms[2].cost_missing==1
        @test result.rows[2].generation_timer_seconds===nothing
        @test result.arms[2].timed_seconds_per_pass===nothing
    end
end
