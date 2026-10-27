include("bayes_thinking_admission.jl")
using Test

function test_bayes_thinking_admission()
    @testset "One whole final Bayesian artifact after thinking" begin
        good=(;answer="Reasoning may mention {\"probability\":0.9}.\n</think>\n{\"probability\":0.25}",
            truncated=false,finished_thinking=true,forced_closures=0,thinking_budget=1024)
        @test bayes_thinking_final(good).answer=="{\"probability\":0.25}"
        @test admit_bayes_thinking(good,"direct_probability").probability==0.25
        @test admit_bayes_thinking((;good...,forced_closures=1),"direct_probability").accepted
        for change in ((;truncated=true),(;truncated=0),(;finished_thinking=false),
                (;forced_closures=true),(;forced_closures=2),(;thinking_budget=0),
                (;answer="{\"probability\":0.25}"),(;answer="reason</think>"),
                (;answer="</think>{\"probability\":0.25}</think>{\"probability\":0.5}"),
                (;answer="<think>reason</think>{\"probability\":0.25}"),
                (;answer="reason</think>{\"probability\":0.25}<|im_end|>"),
                (;answer=repeat("x",131073)))
            @test !admit_bayes_thinking((;good...,change...),"direct_probability").accepted
        end
        @test !bayes_thinking_final((;answer=good.answer)).accepted
        @test !bayes_thinking_final((;good...,inference_error="synthetic")).accepted
        for suffix in ("prose {\"probability\":0.25}","{\"probability\":0.25} trailing", "{\"probability\":0.25}\n{\"probability\":0.5}")
            @test !admit_bayes_thinking((;good...,answer="reason</think>"*suffix),"direct_probability").accepted
        end
        fenced=(;good...,answer="λ Bayesian reasoning</think>\n```json\n{\"probability\":0.25}\n```")
        @test admit_bayes_thinking(fenced,"direct_probability").probability==0.25
        @test !admit_bayes_thinking((;good...,answer="reason</think>{\"probability\":true}"),"direct_probability").accepted
        @test bayes_whole_json_object("{\"x\":\"}\\\"[\",\"nested\":[{}]}\n")
        for source in ("{}x","{} {}","[]","{]","{\"x\":\"unterminated}","{\"x\":[}")
            @test !bayes_whole_json_object(source)
        end
    end
end
