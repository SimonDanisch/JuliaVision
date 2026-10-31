# The harness verbs over one backbone. Prompt ingestion must reproduce
# sequential decode bit for bit: the frozen study receipts depend on it, and
# 32-column chunks stopped doing so once wider products took the fp16 K-quant
# GEMM (stable_groups answered in 136 tokens instead of the receipt's 150).
isdefined(@__MODULE__,:textsession) || include("text_backbone.jl")
using Test
import Mantle

function test_text_backbone(model,prompt;decode=8)
    family=describe(model).family
    length(prompt)>2INGEST_CHUNK[] || throw(ArgumentError("prompt must span more than two chunks"))
    @testset "ingest! reproduces sequential feed! ($family)" begin
        capacity=nextpow(2,length(prompt)+decode+1)
        chunked=textsession(model;capacity);sequential=nothing
        try
            sequential=textsession(model;capacity)
            ingest!(chunked,prompt);foreach(t->feed!(sequential,t),prompt)
            Mantle.waitidle(model.device)
            @test chunked.position==sequential.position==length(prompt)
            @test reinterpret(UInt32,Array(chunked.logits))==reinterpret(UInt32,Array(sequential.logits))
            generate!(chunked,decode);generate!(sequential,decode);Mantle.waitidle(model.device)
            n=length(prompt)
            @test Array(chunked.predictions)[n:n+decode-1]==Array(sequential.predictions)[n:n+decode-1]
            @test reinterpret(UInt32,Array(chunked.logits))==reinterpret(UInt32,Array(sequential.logits))
        finally
            close(chunked);sequential===nothing||close(sequential)
        end
    end
    nothing
end
