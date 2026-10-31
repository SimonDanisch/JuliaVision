include("projection_memory_generation.jl")
using Test

"""Integration invariant, not a quality result: zero-weight memory is identity."""
function verify_projection_model_hook(model,tokenizer,data)
    messages=["system"=>"You are a Julia programmer. Return only executable Julia code.",
        "user"=>"Write Julia function double(x) returning 2*x. Code only."]
    direct=generate_julia_answer(model,tokenizer,messages;max_tokens=32,output_chunk=128)
    hooked=generate_projection_memory_answer(model,tokenizer,messages,data;weight=0f0,max_tokens=32)
    @test hooked.answer==direct.answer
    @test hooked.code==direct.code
    @test hooked.memory_status==0
    @test hooked.decode_submissions==direct.decode_submissions==32
    (;checks=4,exact_answer=true,status=hooked.memory_status,
        scope="zero-weight hook integration ONLY; not intelligence",runtime_provenance=inference_runtime_provenance())
end
