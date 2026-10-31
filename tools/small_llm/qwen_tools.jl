# Qwen3.5/3.6 tool calling: the chat template's tool subset, the XML-style calls
# the model writes, and the text a tool response appends to a running session.
# Rendering is checked against the checkpoint's own Jinja template
# (test_qwen_tools.jl), as QwenText.render_chat is for plain chats.
module QwenTools

"""A tool the model may call: `parameters` are `name => description` pairs,
all strings and all required."""
struct Tool
    name::String
    description::String
    parameters::Vector{Pair{String,String}}
end

"""The tool as the template's `tojson` prints it: Python's `json.dumps` with its
default `", "`/`": "` separators, keys in this order, non-ASCII kept."""
function tooljson(t::Tool)
    props=join(("$(pystr(k)): {\"type\": \"string\", \"description\": $(pystr(d))}" for (k,d) in t.parameters),", ")
    req=join((pystr(k) for (k,_) in t.parameters),", ")
    "{\"type\": \"function\", \"function\": {\"name\": $(pystr(t.name)), \"description\": $(pystr(t.description)), " *
    "\"parameters\": {\"type\": \"object\", \"properties\": {$props}, \"required\": [$req]}}}"
end

function pystr(s::AbstractString)
    io=IOBuffer();print(io,'"')
    for c in s
        if c=='"';print(io,"\\\"")
        elseif c=='\\';print(io,"\\\\")
        elseif c=='\n';print(io,"\\n")
        elseif c=='\r';print(io,"\\r")
        elseif c=='\t';print(io,"\\t")
        elseif c=='\b';print(io,"\\b")
        elseif c=='\f';print(io,"\\f")
        elseif UInt32(c)<0x20;print(io,"\\u",string(UInt32(c);base=16,pad=4))
        else print(io,c)
        end
    end
    print(io,'"');String(take!(io))
end

const INSTRUCTIONS="\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>"

"""A call the model made: the function name and its string arguments in order."""
struct Call
    name::String
    arguments::Vector{Pair{String,String}}
end

"""A chat message: `role` is system, user, assistant or tool; an assistant
message may carry reasoning and calls."""
struct Message
    role::String
    content::String
    reasoning::Union{Nothing,String}
    calls::Vector{Call}
end
Message(role,content)=Message(role,content,nothing,Call[])

istoolresponse(c)=startswith(c,"<tool_response>")&&endswith(c,"</tool_response>")

"""The template's rendering of `messages` with `tools`, ending in the assistant
generation prompt. Plain-text content only, as in QwenText.render_chat."""
function render(messages::AbstractVector{Message};tools::AbstractVector{Tool}=Tool[],thinking::Bool=true)
    isempty(messages) && throw(ArgumentError("no messages"))
    all(m->m.role in ("system","user","assistant","tool"),messages) || throw(ArgumentError("unsupported role"))
    io=IOBuffer()
    sys=first(messages).role=="system" ? strip(first(messages).content) : nothing
    if !isempty(tools)
        print(io,"<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>")
        foreach(t->print(io,"\n",tooljson(t)),tools)
        print(io,"\n</tools>",INSTRUCTIONS)
        sys===nothing || isempty(sys) || print(io,"\n\n",sys)
        print(io,"<|im_end|>\n")
    elseif sys!==nothing
        print(io,"<|im_start|>system\n",sys,"<|im_end|>\n")
    end
    lastquery=findlast(m->m.role=="user"&&!istoolresponse(strip(m.content)),messages)
    lastquery===nothing && throw(ArgumentError("no user query"))
    for (i,m) in enumerate(messages)
        content=String(strip(m.content))
        if m.role=="system"
            i==1 || throw(ArgumentError("system message must be first"))
        elseif m.role=="user"
            print(io,"<|im_start|>user\n",content,"<|im_end|>\n")
        elseif m.role=="assistant"
            reasoning=""
            if m.reasoning!==nothing
                reasoning=m.reasoning
            elseif occursin("</think>",content)
                parts=split(content,"</think>")
                reasoning=String(lstrip(last(split(rstrip(first(parts),'\n'),"<think>")),'\n'))
                content=String(lstrip(last(parts),'\n'))
            end
            reasoning=String(strip(reasoning))
            if i>lastquery
                print(io,"<|im_start|>assistant\n<think>\n",reasoning,"\n</think>\n\n",content)
            else
                print(io,"<|im_start|>assistant\n",content)
            end
            for (j,c) in enumerate(m.calls)
                print(io,j>1 ? "\n<tool_call>\n" : isempty(strip(content)) ? "<tool_call>\n" : "\n\n<tool_call>\n","<function=",c.name,">\n")
                for (k,v) in c.arguments
                    print(io,"<parameter=",k,">\n",v,"\n</parameter>\n")
                end
                print(io,"</function>\n</tool_call>")
            end
            print(io,"<|im_end|>\n")
        else
            i>1 && messages[i-1].role!="tool" && print(io,"<|im_start|>user")
            print(io,"\n<tool_response>\n",content,"\n</tool_response>")
            (i==length(messages) || messages[i+1].role!="tool") && print(io,"<|im_end|>\n")
        end
    end
    print(io,"<|im_start|>assistant\n<think>\n")
    thinking || print(io,"\n</think>\n\n")
    String(take!(io))
end

"""What follows the model's `</tool_call>` once the tool has answered: the end
of its turn, the response turn, and the next generation prompt, as `render`
would write them for the conversation so far."""
responsesuffix(result::AbstractString;thinking::Bool=true)=
    "<|im_end|>\n<|im_start|>user\n<tool_response>\n"*strip(result)*"\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n"*
    (thinking ? "" : "\n</think>\n\n")

"""The call at the end of generated `text`, which stopped at `</tool_call>`, or
an error message saying why it is not a call."""
function parsecall(text::AbstractString)
    start=findlast("<tool_call>",text)
    start===nothing && return "no <tool_call> before </tool_call>"
    body=text[last(start)+1:end]
    body=replace(body,r"</tool_call>\s*$"=>"")
    m=match(r"^\s*<function=([A-Za-z_][A-Za-z0-9_]*)>\n(.*?)</function>\s*$"s,body)
    m===nothing && return "expected <function=NAME> ... </function> inside <tool_call>"
    args=Pair{String,String}[]
    rest=m.captures[2]
    for p in eachmatch(r"<parameter=([A-Za-z_][A-Za-z0-9_]*)>\n?(.*?)\n?</parameter>\n?"s,rest)
        push!(args,String(p.captures[1])=>String(p.captures[2]))
    end
    stripped=replace(rest,r"<parameter=([A-Za-z_][A-Za-z0-9_]*)>\n?(.*?)\n?</parameter>\n?"s=>"")
    isempty(strip(stripped)) || return "text outside <parameter=...> blocks"
    Call(String(m.captures[1]),args)
end
end
