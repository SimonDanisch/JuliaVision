# Generation with tools the model calls itself. Decoding stays on the device in
# 16-token chunks; a chunk that runs past `</tool_call>` is rewound to its start
# and the kept tokens are replayed through the prefill, which reproduces
# sequential decode bit for bit at this width. The tool's answer is ingested
# into the SAME session: nothing before it is recomputed.
isdefined(@__MODULE__,:textsession) || include("text_backbone.jl")
isdefined(@__MODULE__,:QwenTools) || include("qwen_tools.jl")
isdefined(@__MODULE__,:GPUSampling) || include("gpu_sampling.jl")
isdefined(@__MODULE__,:JuliaTaskEval) || include("julia_eval.jl")
import Mantle,KernelInterface,Atomix

# ── Tools ────────────────────────────────────────────────────────────────────

"""Tools a kernel answer may call. Neither executes model code."""
module JuliaTools
import ..QwenTools,KernelInterface,Atomix

const MAXCHARS=4000
clip(s)=length(s)<=MAXCHARS ? s : first(s,MAXCHARS)*"\n[truncated]"

"""The first syntax error in `code`, as the sandbox would print it, or "no syntax errors"."""
function check_syntax(code::AbstractString)
    try
        Base.JuliaSyntax.parseall(Base.JuliaSyntax.SyntaxNode,code;filename="candidate.jl")
    catch err
        # A parse error is the answer; anything else is a fault of this tool.
        err isa Base.JuliaSyntax.ParseError || rethrow()
        return clip(sprint(showerror,err))
    end
    "no syntax errors"
end

# Modules `lookup` resolves, by the names code writes for them.
const ROOTS=Dict("KI"=>KernelInterface,"KernelInterface"=>KernelInterface,"Atomix"=>Atomix,"Base"=>Base,"Core"=>Core)

docstring(M::Module,name::Symbol)=string(Base.Docs.doc(Base.Docs.Binding(M,name)))

exported(M::Module)=(e=sort!(string.(filter(n->n!==nameof(M),names(M))));isempty(e) ? "nothing" : join(e,", "))
listing(M::Module)="`$M` exports: "*exported(M)*
    "\nOther names it defines: "*join(sort!([String(n) for n in names(M;all=true) if !(n in names(M)) &&
        occursin(r"^@?[A-Za-z_][A-Za-z0-9_!]*$",String(n)) && !startswith(String(n),"_") && n!==nameof(M)]),", ")

function signatures(f)
    ms=collect(methods(f))
    lines=[replace(sprint(show,m),r" @ .*$"=>"") for m in ms[1:min(end,12)]]
    length(ms)>12 && push!(lines,"... $(length(ms)-12) more")
    join(lines,"\n")
end

"""Documentation of an installed name: a module's docs and the names it defines,
a function's or macro's docstring and methods, or, for a name that does not
exist, that fact and what its module does define."""
function lookup(name::AbstractString)
    parts=split(replace(strip(name,[' ','\n','`']),r"\(\)$"=>""),'.')
    root=get(ROOTS,String(first(parts)),nothing)
    root===nothing && return "`$(first(parts))` is not one of the modules available here: "*join(sort!(collect(keys(ROOTS))),", ")
    M=root;path=String(first(parts))
    for (k,p) in enumerate(parts[2:end])
        sym=Symbol(p)
        isdefined(M,sym) || return clip("`$path.$p` is not defined. "*listing(M))
        value=getglobal(M,sym);path*="."*p
        if value isa Module && k<length(parts)-1
            M=value
        elseif k==length(parts)-1
            doc=docstring(M,sym)
            value isa Module && return clip(doc*"\n\n"*listing(value))
            (value isa Function || startswith(p,"@")) && return clip(doc*"\n\nMethods:\n"*signatures(value))
            return clip(doc*"\n\n`$path` is a $(typeof(value)).")
        else
            return "`$path` is not a module, so `$name` does not resolve"
        end
    end
    clip(string(Base.Docs.doc(M))*"\n\n"*listing(M))
end

"""Tools offered in one answer, how each call is answered, and what to release
when the answer is done."""
struct Toolbox
    tools::Vector{QwenTools.Tool}
    runners::Dict{String,Function}
    release::Function
end
Base.close(b::Toolbox)=(b.release();nothing)

const CHECK_SYNTAX=QwenTools.Tool("check_syntax","Parse Julia code and report its first syntax error. Does not run the code.",
    ["code"=>"Complete Julia source to parse."])
const LOOKUP=QwenTools.Tool("lookup","Documentation of an installed Julia name: a module (`Atomix`, `KernelInterface`), function, macro (`Atomix.@atomic`) or type. For a name that does not exist it says so and lists what the module defines.",
    ["name"=>"The name as written in code, e.g. `KI.get_global_id` or `Base.@atomic`."])

notools()=Toolbox(QwenTools.Tool[],Dict{String,Function}(),()->nothing)
parse_lookup()=Toolbox([CHECK_SYNTAX,LOOKUP],Dict{String,Function}("check_syntax"=>c->check_syntax(c["code"]),"lookup"=>c->lookup(c["name"])),()->nothing)

"""The response to a parsed call, or to the reason a call did not parse."""
function respond(box::Toolbox,call)
    call isa AbstractString && return "error: malformed tool call: "*call
    f=get(box.runners,call.name,nothing)
    f===nothing && return "error: unknown function `$(call.name)`; available: "*(isempty(box.runners) ? "none" : join(sort!(collect(keys(box.runners))),", "))
    args=Dict(call.arguments)
    spec=only(filter(t->t.name==call.name,box.tools))
    missing=[k for (k,_) in spec.parameters if !haskey(args,k)]
    isempty(missing) || return "error: missing parameter(s) "*join(missing,", ")
    f(args)
end
end

isdefined(@__MODULE__,:SandboxWorker) || include("julia_worker.jl")

const RUN_JULIA=QwenTools.Tool("run_julia",
    "Run Julia code in a sandbox with a software Vulkan GPU, to test a kernel. `using Mantle`, `import KernelInterface as KI` and `import Atomix` are already done, and `__bench_launch__(kernel!, out, args...; threads=n)` launches `kernel!` on `n` invocations over device copies of the arrays and returns `out` after the kernel. Returns what the code printed and the value of its last expression, or the error. For testing only: your final answer must not call `__bench_launch__`.",
    ["code"=>"Julia code that defines kernel! and tests it."])

"""Parse, lookup and run: one sandbox worker for the answer, started at its
first run and stopped when the answer is done."""
function parse_lookup_run()
    worker=SandboxWorker()
    box=JuliaTools.parse_lookup()
    JuliaTools.Toolbox([box.tools;RUN_JULIA],
        merge(box.runners,Dict{String,Function}("run_julia"=>c->describe_run(run_code(worker,c["code"])))),()->close(worker))
end

# ── Session rewind ───────────────────────────────────────────────────────────

"""Recorded copies of a session's recurrent state into a second set of buffers
and back. Attention caches need none: positions past the restored one are
written again before they are read."""
struct Rewind
    saved::Vector{Any}
    save::Any
    restore::Any
end

function Rewind(s)
    device=s.model.device;bufs=collect(Iterators.flatten(values(s.recurrent)))
    saved=Any[];plans=Any[]
    try
        for b in bufs;push!(saved,Mantle.Buffer(device,eltype(b),size(b)));end
        function copies(pairs)
            g=Mantle.Graph(device)
            for (k,(dst,src)) in enumerate(pairs)
                Mantle.dispatch!(g,BR_COPY,(dst,src,Int32(length(src))),length(src);group=256,name="rewind.copy_$k")
            end
            p=Mantle.record!(Mantle.Plan(g;budget=Inf));push!(plans,p);p
        end
        Rewind(saved,copies(zip(saved,bufs)),copies(zip(bufs,saved)))
    catch
        foreach(Mantle.free!,plans);foreach(Mantle.free!,saved);rethrow()
    end
end
const BR_COPY=SmallLLMChecks.Qwen35Experiment.BR.copy_kernel!
Base.close(r::Rewind)=(Mantle.free!(r.save);Mantle.free!(r.restore);foreach(Mantle.free!,r.saved);nothing)

# ── The loop ─────────────────────────────────────────────────────────────────

const TOOL_CALL_END=248059   # `</tool_call>`
# "Use the tools ... return only the code" left the model deciding between the
# two inside its reasoning until the budget ran out; this asks for one, then the other.
const TOOL_SYSTEM="You are a Julia GPU programmer. You can call tools to check that names exist and that code parses. When you are confident, give your final answer as the kernel code only."
stops(tokenizer)=(tokenizer.bpe.eos,tokenizer.bpe.bos)

const THINK_END=248069       # `</think>`
# What closes a round's reasoning at its budget: Qwen's documented wording for a
# thinking budget, followed by the closing tag the model would have written.
const THINK_CLOSE="\n\nConsidering the limited time, I have to give the solution based on the thinking directly now.\n</think>\n\n"

"""Generate an answer to `messages`, the model calling the tools in `toolbox`.

`max_tokens` bounds the tokens the model generates over all rounds; tool
responses and forced closes are context, not budget. `think_budget` bounds each
round's reasoning: at that many tokens without `</think>`, `THINK_CLOSE` is
appended exactly there (rewinding the chunk that ran past it). Returns the final
answer, every call with its response, and the token accounting."""
function tool_julia_answer(model,tokenizer,messages;max_tokens,thinking=true,seed,topk=20,temperature=0.6f0,top_p=0.95f0,
                           presence=0f0,chunk=16,maxcalls=6,response_tokens=1200,toolbox=JuliaTools.parse_lookup(),
                           think_budget=nothing)
    T=SmallLLMChecks.QwenText
    ids=T.encode(tokenizer,QwenTools.render(messages;tools=toolbox.tools,thinking))
    close_ids=T.encode(tokenizer,THINK_CLOSE)
    capacity=nextpow(2,length(ids)+max_tokens+maxcalls*response_tokens+4length(close_ids)+64)
    capacity<=maxcapacity(model) || error("prompt exceeds experiment capacity")
    sampler=GPUSampling.State(model.device,Int(model.config.vocab_size);seed=UInt32(seed),topk,
        temperature=Float32(temperature),top_p=Float32(top_p),presence=Float32(presence),fallback=tokenizer.bpe.eos)
    s=nothing;rw=nothing;started=time()
    try
        s=textsession(model;capacity,logit_hook=(g,l)->GPUSampling.declare!(g,l,sampler))
        rw=Rewind(s)
        GPUSampling.activate!(sampler,false);ingest!(s,ids[1:end-1])
        GPUSampling.activate!(sampler,true);feed!(s,last(ids))
        # `round` holds this round's tokens from its first sampled one, which
        # each feed leaves pending at the current position.
        pending()=Int(only(Array(s.predictions)[s.position:s.position]))
        generated=Int[];round=Int[];calls=Any[];rewinds=0;finished=false;context=0;forced=0;thoughts=Int[]
        thinking_open=thinking;thought=0
        # Count a sampled token into the round; whether it ends the scan.
        function accept!(t)
            push!(generated,t);push!(round,t)
            if thinking_open
                thought+=1
                t==THINK_END && (thinking_open=false;push!(thoughts,thought))
            end
            t==TOOL_CALL_END || t in stops(tokenizer) ? :stop :
                thinking_open && think_budget!==nothing && thought>=think_budget ? :budget : :none
        end
        # Append `tokens` after the kept ones: rewind the chunk if it ran on,
        # replay, ingest, and leave the next sampled token pending.
        function continue_with!(P,preds,stop,n,counter,tokens)
            replay=if stop<n
                Mantle.run!(rw.restore);setposition!(s,P);rewinds+=1
                fixdraws!(sampler,counter+UInt32(stop),preds[stop+2:end])
                [preds[1:stop+1];tokens[1:end-1]]
            else
                [preds[end:end];tokens[1:end-1]]
            end
            context+=length(tokens)
            GPUSampling.activate!(sampler,false);ingest!(s,replay)
            GPUSampling.activate!(sampler,true);feed!(s,last(tokens))
        end
        event=accept!(pending())
        P=s.position;preds=[last(round)];stop=0;n=0;counter=only(Array(sampler.counter))
        while !finished
            if event===:none
                length(generated)<max_tokens || break
                P=s.position;n=min(chunk,max_tokens-length(generated))
                Mantle.run!(rw.save);counter=only(Array(sampler.counter))
                generate!(s,n)
                preds=Int.(Array(s.predictions)[P:P+n])     # the pending token, then n new ones
                stop=n
                for j in 1:n
                    event=accept!(preds[j+1])
                    event===:none || (stop=j;break)
                end
                event===:none && continue
            end
            if event===:budget
                forced+=1;push!(thoughts,thought);thinking_open=false
                continue_with!(P,preds,stop,n,counter,close_ids);append!(round,close_ids)
            elseif last(round)==TOOL_CALL_END
                call=QwenTools.parsecall(T.decode(tokenizer,round;skipspecial=false))
                response=length(calls)<maxcalls ? JuliaTools.respond(toolbox,call) :
                    "error: the tool-call limit is reached; give your final answer now"
                push!(calls,(;call=call isa QwenTools.Call ? (;name=call.name,arguments=call.arguments) : nothing,
                    malformed=call isa AbstractString ? call : nothing,response))
                continue_with!(P,preds,stop,n,counter,T.encode(tokenizer,QwenTools.responsesuffix(response;thinking)))
                empty!(round);thinking_open=thinking;thought=0
            else
                finished=true;break
            end
            event=accept!(pending())
            P=s.position;preds=[last(round)];stop=0;n=0;counter=only(Array(sampler.counter))
        end
        Mantle.waitidle(model.device)
        only(Array(sampler.status))==0 || error("sampler rejected nonfinite/all-masked logits")
        answer=T.decode(tokenizer,round;skipspecial=false)
        stop=findfirst(t->t in stops(tokenizer),round)
        (;answer=stop===nothing ? answer : T.decode(tokenizer,round[1:stop-1];skipspecial=false),
         code=JuliaTaskEval.extract_code(answer;thinking),calls,input_tokens=length(ids),generated_tokens=length(generated),
         # The conversation's length; the session's position can run past its
         # end by up to a chunk, since nothing is rewound after the final token.
         budget_tokens=max_tokens,context_tokens=length(ids)+length(generated)+context,truncated=!finished,rewinds,
         forced_closes=forced,thinking_tokens=thoughts,
         finished_thinking=!thinking||occursin("</think>",answer),elapsed_seconds=time()-started)
    finally
        rw===nothing||close(rw);s===nothing||close(s);close(sampler)
    end
end

"""After a rewind: the draw counter as if the discarded tokens were never
sampled, and their presence counts removed."""
function fixdraws!(sampler,counter::UInt32,discarded)
    sampler.counter[:]=UInt32[counter]
    seen=Array(sampler.seen)
    for t in discarded;seen[t+1]-=Int32(1);end
    sampler.seen[:]=seen
    nothing
end
