isdefined(@__MODULE__, :admit_bayes_extraction) || include("bayes_extraction_admission.jl")

"""Select one completed thinking response's entire final segment, never a JSON fragment.

The chat template opens thinking before generation. Generated text therefore
must contain exactly one closing marker and no new opening marker. The caller
must supply the raw answer, NOT JuliaTaskEval.extract_code's convenience field.
"""
function bayes_thinking_final(receipt)
    reject(reason) = (;accepted=false, reason, answer="")
    hasproperty(receipt, :inference_error) && return reject("inference_error")
    for field in (:answer, :truncated, :finished_thinking, :forced_closures, :thinking_budget)
        hasproperty(receipt, field) || return reject("missing completion metadata")
    end
    receipt.truncated === false || return reject("truncated or invalid completion flag")
    receipt.finished_thinking === true || return reject("unfinished thinking")
    forced=receipt.forced_closures
    forced isa Integer && !(forced isa Bool) && forced in (0,1) || return reject("invalid closure count")
    budget=receipt.thinking_budget
    budget isa Integer && !(budget isa Bool) && budget>0 || return reject("invalid thinking budget")
    receipt.answer isa AbstractString || return reject("invalid answer type")
    answer=String(receipt.answer)
    ncodeunits(answer)<=131072 || return reject("bounded response")
    occursin("<think>",answer) && return reject("unexpected opening marker")
    segments=split(answer,"</think>";keepempty=true)
    length(segments)==2 || return reject("exactly one thinking closure required")
    final=strip(last(segments))
    isempty(final) && return reject("empty final answer")
    # Generation already cuts at the first EOS. Refuse raw control tokens here
    # rather than selecting a second answer or silently discarding text.
    any(t->occursin(t,final),("<|im_end|>","<|im_start|>","<|endoftext|>")) &&
        return reject("unexpected control token")
    (;accepted=true, reason="entire final segment", answer=String(final))
end

function admit_bayes_thinking(receipt,arm)
    final=bayes_thinking_final(receipt)
    reject(reason)=(;accepted=false,reason,probability=nothing,network=nothing,packed=nothing)
    final.accepted || return reject(final.reason)
    try
        source=bayes_extraction_payload(final.answer)
        bayes_whole_json_object(source) || return reject("one whole JSON object required")
        admit_bayes_extraction(source,arm)
    catch err
        err isa InterruptException && rethrow()
        reject(sprint(showerror,err))
    end
end

"""Check object extent before JSON3, which may otherwise ignore trailing bytes."""
function bayes_whole_json_object(source)
    isempty(source) && return false
    first(source)=='{' || return false
    stack=Char[];quoted=false;escaped=false;finished=false
    for c in source
        if finished
            c in (' ','\t','\n','\r') || return false
        elseif quoted
            if escaped;escaped=false
            elseif c=='\\';escaped=true
            elseif c=='"';quoted=false
            end
        elseif c=='"';quoted=true
        elseif c in ('{','[')
            push!(stack,c)
            length(stack)<=16 || return false
        elseif c in ('}',']')
            isempty(stack) && return false
            pop!(stack)==(c=='}' ? '{' : '[') || return false
            isempty(stack) && (finished=true)
        end
    end
    finished && !quoted && isempty(stack)
end
