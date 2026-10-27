# Speculative greedy decoding with drafts from Qwen3.6's MTP layer against plain
# greedy decoding on the 35B. With decode's attention in the verify the tokens
# are the same whatever the drafts; that most drafts are accepted is what shows
# the layer is wired as the checkpoint was trained (a wrong norm, state or
# position still decodes correctly, with almost nothing accepted).
isdefined(@__MODULE__,:ingest!) || include("text_backbone.jl")
isdefined(@__MODULE__,:spec_prompt) || include("test_speculative.jl")
using Test

function mtp_greedy(model,ids,n;kv_type,capacity,k=2,chunk=INGEST_CHUNK[],deep=typemax(Int))
    s=Q36.session(model;capacity,attention=:flash,kv_type);sp=nothing
    try
        sp=Q36.MTPSpeculator(s;k,deep)
        Q36.prefill!(sp,ids;chunk)
        out=Q36.speculative_decode!(sp,ids,n)
        out,(;steps=sp.steps,drafted=sp.drafted,accepted=sp.accepted)
    finally
        sp===nothing||close(sp);close(s)
    end
end

# Plain greedy decoding of two turns: `n1` tokens after `ids1`, the last left
# pending; then `ids2` after it, and `n2` tokens.
function plain_turns(model,ids1,n1,ids2,n2;kv_type,capacity)
    s=Q36.session(model;capacity,attention=:flash,kv_type)
    try
        turn(ids,n)=begin
            ingest!(s,ids);Mantle.waitidle(model.device)
            out=[Int(only(Array(s.inner.best)))]
            while length(out)<n
                Q36.step!(s,out[end]);push!(out,Int(only(Array(s.inner.best))))
            end
            out
        end
        first=turn(ids1,n1)
        first,turn([first[end];ids2],n2)
    finally
        close(s)
    end
end

function mtp_turns(model,ids1,n1,ids2,n2;kv_type,capacity,k)
    s=Q36.session(model;capacity,attention=:flash,kv_type);sp=nothing
    try
        sp=Q36.MTPSpeculator(s;k,deep=typemax(Int))
        Q36.prefill!(sp,ids1)
        first=Q36.speculative_decode!(sp,ids1,n1)
        pending=sp.pending
        Q36.prefill!(sp,[first[end];ids2])
        first,Q36.speculative_decode!(sp,[ids1;first;ids2],n2),pending
    finally
        sp===nothing||close(sp);close(s)
    end
end

function test_mtp(model,tokenizer;n=120)
    Q36.hasmtp(model) || Q36.loadmtp!(model,QWEN36_MTP)
    ids=spec_prompt(tokenizer);capacity=length(ids)+2n+64
    @testset "MTP drafts" begin
        @test_throws ArgumentError Q36.loadmtp!(model,QWEN36_MTP)
        @testset "prompt chunks with MTP columns leave the trunk's logits as they were" begin
            for chunk in (16,64)
                s1=Q36.session(model;capacity,attention=:flash,kv_type=Float32)
                s2=Q36.session(model;capacity,attention=:flash,kv_type=Float32);sp=nothing
                try
                    ingest!(s1,ids;chunk);sp=Q36.MTPSpeculator(s2);Q36.prefill!(sp,ids;chunk)
                    @test Array(s1.inner.logits)==Array(s2.inner.logits)
                finally
                    sp===nothing||close(sp);close(s1);close(s2)
                end
            end
        end
        for kv_type in (Float32,Int8)
            want=plain_greedy(model,ids,n;kv_type,capacity)
            for k in 1:3
                got,stats=mtp_greedy(model,ids,n;kv_type,capacity,k)
                @test got==want
                # Most first drafts are taken; every round drafts k.
                @test stats.accepted>=0.6*stats.drafted
                @test stats.steps<n÷(1+0.6k)
            end
        end
        @testset "a turn after generated tokens (the flush of waiting states)" begin
            follow=SmallLLMChecks.QwenText.encode(tokenizer,"<|im_end|>\n<|im_start|>user\nNow rename `w` to `weight` as well.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
            want=plain_turns(model,ids,n,follow,n;kv_type=Float32,capacity=2capacity)
            got1,got2,pending=mtp_turns(model,ids,n,follow,n;kv_type=Float32,capacity=2capacity,k=3)
            @test (got1,got2)==want
            @test pending>=1
        end
        @testset "the session cannot move without its MTP layer" begin
            s=Q36.session(model;capacity,attention=:flash,kv_type=Float32);sp=nothing
            try
                ingest!(s,ids[1:4])
                @test_throws ArgumentError Q36.MTPSpeculator(s)
                close(s);s=Q36.session(model;capacity,attention=:flash,kv_type=Float32)
                sp=Q36.MTPSpeculator(s);Q36.prefill!(sp,ids)
                Q36.step!(s,1)
                @test_throws ArgumentError Q36.speculative_decode!(sp,ids,4)
                @test_throws ArgumentError Q36.prefill!(sp,ids[1:2])
            finally
                sp===nothing||close(sp);close(s)
            end
        end
    end
end
