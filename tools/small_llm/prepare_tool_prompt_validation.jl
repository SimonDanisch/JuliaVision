using JSON3, SHA
function prepare_tool_prompt_validation(;checkpoint="gen/qwen35-4b")
    T=SmallLLMChecks.QwenText;tokenizer=T.Tokenizer(checkpoint);cases=Any[]
    function add(id,messages,thinking)
        push!(cases,(;id,messages=[(;role=first(m),content=last(m)) for m in messages],thinking,
            validate_logits=false,julia_prompt=T.render_chat(messages;thinking),julia_ids=T.chat_ids(tokenizer,messages;thinking)))
    end
    api=JSON3.read(read(joinpath(@__DIR__,"api_generalization_results/registration.json"),String))
    for task in api.rows, arm in ("direct","api_docs")
        messages=[String(first(keys(m)))=>String(first(values(m))) for m in task.messages]
        arm=="api_docs" && (messages[end]="user"=>(last(messages[end])*"\n\nInstalled package API documentation (may be incomplete; use only as relevant):\n"*String(task.evidence)))
        add(String(task.id)*"_"*arm,messages,false)
    end
    jet=JSON3.read(read(joinpath(@__DIR__,"thinking_jet_results/registration.json"),String))
    for task in jet.rows, arm in ("self_review","jet_feedback")
        messages=[String(first(keys(m)))=>String(first(values(m))) for m in task.messages]
        request="Review the complete prior solution for compliance with the original request. Fix any problems and return the complete executable Julia solution."
        arm=="jet_feedback" && (request*="\n\nPublic parser/dispatch/JET diagnostics follow. They are advisory, contain no private assertions or expected values, and are not a proof of correctness:\n"*String(task.evidence))
        push!(messages,"user"=>request)
        add(String(task.id)*"_"*arm,messages,true)
    end
    report=(;checkpoint=abspath(checkpoint),scope="actual frozen tool-experiment prompt template qualification only",cases)
    open(io->JSON3.pretty(io,report),joinpath(@__DIR__,"tool_prompt_validation_inputs.json"),"w")
    (;cases=length(cases),input_tokens=[(;id=c.id,tokens=length(c.julia_ids)) for c in cases])
end
