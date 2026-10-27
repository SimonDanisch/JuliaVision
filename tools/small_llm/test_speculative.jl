# Speculative greedy decoding against plain greedy decoding on the 35B: a prompt
# that asks for its own code back with a rename, so drafts are offered, accepted
# and refused. With decode's attention in the verify chunk the two outputs are
# the same tokens; the replay after a refusal is what keeps the recurrent state
# on the plain run's path.
isdefined(@__MODULE__,:ingest!) || include("text_backbone.jl")
using Test

const SPEC_CODE="""
function accumulate_scores(xs::Vector{Float64}, weights::Vector{Float64})
    total = 0.0
    for i in eachindex(xs, weights)
        x = xs[i]
        w = weights[i]
        total += x * w
    end
    return total / sum(weights)
end
"""

spec_prompt(tokenizer)=SmallLLMChecks.QwenText.encode(tokenizer,
    "<|im_start|>user\n$(SPEC_CODE)\nRewrite this function with the loop variable `x` renamed to `value`. Output only the code.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")

function plain_greedy(model,ids,n;kv_type,capacity)
    s=Q36.session(model;capacity,attention=:flash,kv_type)
    try
        ingest!(s,ids);Mantle.waitidle(model.device)
        out=Int[];next=Int(only(Array(s.inner.best)))
        for _ in 1:n
            push!(out,next);Q36.step!(s,next);next=Int(only(Array(s.inner.best)))
        end
        out
    finally
        close(s)
    end
end

function speculative_greedy(model,ids,n;kv_type,capacity,chunk_attention,kmax=8)
    s=Q36.session(model;capacity,attention=:flash,kv_type);sp=nothing
    try
        ingest!(s,ids);Mantle.waitidle(model.device)
        sp=Q36.Speculator(s;kmax,deep=chunk_attention ? 0 : typemax(Int))
        out=Q36.speculative_decode!(sp,ids,n)
        out,(;steps=sp.steps,drafted=sp.drafted,accepted=sp.accepted)
    finally
        sp===nothing||close(sp);close(s)
    end
end

function test_speculative(model,tokenizer;n=160)
    ids=spec_prompt(tokenizer);capacity=length(ids)+n+32
    @testset "speculative greedy decoding" begin
        for kv_type in (Float32,Int8)
            want=plain_greedy(model,ids,n;kv_type,capacity)
            got,stats=speculative_greedy(model,ids,n;kv_type,capacity,chunk_attention=false)
            @test got==want
            # The prompt is copied back: drafts were offered, taken and refused.
            @test stats.accepted>0
            @test stats.drafted>stats.accepted
            @test length(got)==n
        end
    end
end
