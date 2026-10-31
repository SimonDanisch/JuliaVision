using JSON3, SHA
import BonsaiRunner
function prepare_native_conversations(;checkpoint="gen/qwen35-4b",output=joinpath(@__DIR__,"conversation_validation_inputs.json"))
    tokenizer=SmallLLMChecks.QwenText.Tokenizer(checkpoint)
    cases=Any[]
    function append_case(id,messages;thinking=false,validate_logits=false,source=nothing)
        text=SmallLLMChecks.QwenText.render_chat(messages;thinking)
        ids=SmallLLMChecks.QwenText.chat_ids(tokenizer,messages;thinking)
        push!(cases,(;id,messages=[(;role=first(m),content=last(m)) for m in messages],thinking,validate_logits,
            julia_prompt=text,julia_ids=ids,source))
    end
    # Raw generation prompts only, never scoring contracts/reference code.
    for (id,file) in (
        ("baseline_short","julia_tasks_results/stable_groups_baseline.json"),
        ("missing_repair","julia_jet_v4_results/missing_windows_repair.json"),
        ("kernel_repair","julia_kernel_results/ka_tail_repair.json"))
        path=joinpath(@__DIR__,file);candidate=JSON3.read(read(path,String))
        messages=[String(first(keys(m)))=>String(first(values(m))) for m in candidate.messages]
        append_case(id,messages;validate_logits=true,source=(;file,sha256=bytes2hex(sha256(read(path)))))
    end
    append_case("trim_edges",["system"=>"  Be helpful.\n","user"=>"\n Hello.  "])
    append_case("old_thinking",["user"=>"First question","assistant"=>"<think>\nOld reasoning\n</think>\n\nOld answer\n","user"=>"New question"])
    append_case("post_user_assistant",["user"=>"Question","assistant"=>"Answer"])
    append_case("thinking_generation",["user"=>"Question"];thinking=true)
    result=(;scope="CPU-native template/full-model prefill parity inputs; no model-generated code execution",
        checkpoint=abspath(checkpoint),cases,
        source_sha256=Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in ("qwen_text.jl","prepare_conversation_validation.jl")))
    open(io->JSON3.pretty(io,result),output,"w")
    result
end
