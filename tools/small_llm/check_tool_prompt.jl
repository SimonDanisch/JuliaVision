using JSON3
function check_tool_prompt(id,messages,thinking,tokenizer)
    path=joinpath(@__DIR__,"tool_prompt_native_templates.json")
    reference=JSON3.read(read(path,String))
    row=only(filter(r->r.id==id,reference.rows))
    row.identical || error("native prompt fixture did not qualify")
    ids=SmallLLMChecks.QwenText.chat_ids(tokenizer,messages;thinking)
    ids==Int.(row.native_ids) || error("experiment prompt drifted from native-qualified fixture: $id")
    true
end
