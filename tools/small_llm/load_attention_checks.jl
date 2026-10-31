# Loading this file creates no device. Use the workspace root Julia environment.
# CPU: SmallLLMChecks.run_checks(Mantle.Device("llvmpipe"); output="cpu.json")
# Radeon: SmallLLMChecks.run_checks(Mantle.Device("radeon"); output="radeon.json")
module SmallLLMChecks
using JSON3,SHA
import Mantle
include("qwen35.jl")
include("test_attention_cache_qwen35.jl")
include("bench_qwen35.jl")
include("verify_attention_model_qwen35.jl")
include("test_prefill_qwen35.jl")
include("test_attention_session_qwen35.jl")
include("moe_qwen35.jl")
include("test_moe_qwen35.jl")
include("test_moe_quant_qwen35.jl")
include("quality_attention_qwen35.jl")
include("test_quality_attention_qwen35.jl")
include("qwen_text.jl")

function source_hashes()
    files=("qwen35.jl","grouped_attention_qwen35.jl","prefill_qwen35.jl",
        "moe_qwen35.jl","moe_quant_qwen35.jl","test_attention_cache_qwen35.jl","test_moe_qwen35.jl",
        "test_moe_quant_qwen35.jl",
        "quality_attention_qwen35.jl","moe_transformers_reference.json")
    Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in files)
end

function run_checks(device;output=nothing)
    attention=test_qwen35_attention_cache(device)
    test_qwen35_dense_projection(device)
    moe=test_qwen35_moe(device;fixture=joinpath(@__DIR__,"moe_transformers_reference.json"))
    quant_experts=test_qwen35_quant_experts(device)
    sessions=test_qwen35_attention_sessions(device)
    test_attention_logit_metrics()
    result=(;scope="shared-kernel arithmetic and synthetic integration, not full MoE checkpoint or intelligence evaluation",
        subgroup=Mantle.caps(device).subgroup,attention_cases=length(attention),attention,moe,quant_experts,sessions,
        source_sha256=source_hashes())
    output===nothing || open(io->JSON3.pretty(io,result),output,"w")
    result
end

function prepare_quality_corpus(tokenizer_dir;source_files,tokens_per_file=257,output=nothing)
    tokens_per_file>=2 || throw(ArgumentError("need at least one target token"))
    tokenizer=QwenText.Tokenizer(tokenizer_dir)
    segments=Any[]
    for path in source_files
        bytes=read(path)
        ids=QwenText.encode(tokenizer,String(copy(bytes)))
        length(ids)>=2 || throw(ArgumentError("corpus segment too short"))
        push!(segments,(;source=abspath(path),source_sha256=bytes2hex(sha256(bytes)),
            input_ids=ids[1:min(tokens_per_file,length(ids))]))
    end
    result=(;scope="frozen local diagnostic corpus, not held-out benchmark or general reasoning evaluation",
        tokenizer_sha256=bytes2hex(sha256(read(joinpath(tokenizer_dir,"tokenizer.json")))),segments)
    output===nothing || open(io->JSON3.pretty(io,result),output,"w")
    result
end
end
