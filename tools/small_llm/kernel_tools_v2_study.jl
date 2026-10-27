# Second kernel tools registration: the compile/run tool and a per-round thinking
# budget, with the arms that separate their effects. Same tasks, seeds, sampler
# and generation budget as the kernel evidence and first tools studies, so every
# answer pairs with theirs on task and seed. Pre-registered with the tool
# specifications, system prompts, budget and close wording before generation.
#
#   tools_run_budget  check_syntax, lookup, run_julia; thinking budget per round
#   tools_budget      check_syntax, lookup; same budget (the run tool's effect)
#   direct_budget     no tools, the direct prompt; same budget (the budget's effect)
isdefined(@__MODULE__,:freeze_kernel_tools) || include("kernel_tools_study.jl")
using JSON3,SHA

const TOOLS_V2_DIR=joinpath(@__DIR__,"kernel_tools_v2_results")
const TOOLS_V2_THINK_BUDGET=3072
const TOOLS_V2_ARMS=[
    (;name="tools_run_budget",system=TOOL_SYSTEM,toolbox=parse_lookup_run),
    (;name="tools_budget",system=TOOL_SYSTEM,toolbox=JuliaTools.parse_lookup),
    (;name="direct_budget",system=KERNEL_SYSTEM,toolbox=JuliaTools.notools)]
const TOOLS_V2_SOURCES=("kernel_tools_v2_study.jl","julia_worker.jl",TOOLS_SOURCES...)

armtools(arm)=(b=arm.toolbox();try [QwenTools.tooljson(t) for t in b.tools] finally close(b) end)

function freeze_kernel_tools_v2(model,backbone,tokenizer;directory=TOOLS_V2_DIR,evidence=EVIDENCE_DIR,first=TOOLS_DIR)
    path=joinpath(directory,"conditions.json");isfile(path) && error("kernel tools v2 conditions already frozen")
    base=JSON3.read(read(joinpath(evidence,"conditions.json"),String))
    tasks=evidencetasks()
    for (t,r) in zip(tasks,base.tasks)
        (digest(t.prompt),digest(t.tests),digest(t.reference))==(r.prompt_sha256,r.tests_sha256,r.reference_sha256) || error("contract drift: $(t.id)")
    end
    packet=(;scope="the eight authored kernel tasks: compile/run tool and per-round thinking budget, with control arms; pairs with the evidence and first tools studies by task and seed",
        hypotheses=["the thinking budget turns out-of-budget answers into answers (direct_budget vs direct, tools_budget vs tools_parse_lookup)",
            "run_julia removes kernel-contract failures (tools_run_budget vs tools_budget)"],
        backbone=basename(backbone),backbone_sha256=isgguf(backbone) ? Qwen36GGUF.CHECKPOINT_SHA256 : nothing,
        decoding=base.decoding,seeds=base.seeds,maxcalls=TOOLS_MAXCALLS,think_budget=TOOLS_V2_THINK_BUDGET,think_close=THINK_CLOSE,
        budget="max_tokens bounds the tokens generated over all rounds; tool responses and forced closes are context, not budget",
        arms=[(;name=a.name,system=a.system,tools=armtools(a)) for a in TOOLS_V2_ARMS],
        comparison=(;evidence=filehash(joinpath(evidence,"conditions.json")),first_tools=filehash(joinpath(first,"conditions.json"))),
        tasks=base.tasks,source_sha256=Dict(f=>filehash(joinpath(@__DIR__,f)) for f in TOOLS_V2_SOURCES))
    mkpath(directory);open(io->JSON3.pretty(io,packet),path,"w");packet
end

function load_kernel_tools_v2(;directory=TOOLS_V2_DIR)
    path=joinpath(directory,"conditions.json");packet=JSON3.read(read(path,String))
    for (file,hash) in pairs(packet.source_sha256)
        filehash(joinpath(@__DIR__,String(file)))==hash || error("kernel tools v2 source drift: $file")
    end
    tasks=evidencetasks()
    for (t,r) in zip(tasks,packet.tasks)
        (digest(t.prompt),digest(t.tests),digest(t.reference))==(r.prompt_sha256,r.tests_sha256,r.reference_sha256) || error("contract drift: $(t.id)")
    end
    for (a,r) in zip(TOOLS_V2_ARMS,packet.arms)
        (a.name,a.system,armtools(a))==(String(r.name),String(r.system),String.(r.tools)) || error("arm drift: $(a.name)")
    end
    packet.think_close==THINK_CLOSE || error("close wording drift")
    (;directory,path,packet,tasks)
end

function run_kernel_tools_v2(model,tokenizer;directory=TOOLS_V2_DIR)
    (;path,packet,tasks)=load_kernel_tools_v2(;directory);d=packet.decoding
    conditions=filehash(path);runtime=inference_runtime_provenance()
    for (s,seed) in enumerate(packet.seeds),(k,task) in enumerate(tasks),arm in circshift(TOOLS_V2_ARMS,k+s)
        output=outputfile(directory,task,seed,arm.name);isfile(output) && continue
        messages=[QwenTools.Message("system",arm.system),QwenTools.Message("user",task.prompt)]
        box=arm.toolbox()
        a=try
            tool_julia_answer(model,tokenizer,messages;max_tokens=Int(d.max_tokens),thinking=Bool(d.thinking),seed=UInt32(seed),
                topk=Int(d.topk),temperature=Float32(d.temperature),top_p=Float32(d.top_p),presence=Float32(d.presence),
                maxcalls=Int(packet.maxcalls),toolbox=box,think_budget=Int(packet.think_budget))
        finally
            close(box)
        end
        saverow(output,(;task=task.id,seed,arm=arm.name,system=arm.system,prompt=task.prompt,backbone=describe(model),
            runtime_provenance=runtime,conditions_sha256=conditions,a...))
        println(task.id," s",seed," ",arm.name,": ",length(a.calls)," calls, ",a.forced_closes," forced, ",
            a.generated_tokens," tokens, truncated=",a.truncated);flush(stdout)
    end
    nothing
end

function score_kernel_tools_v2(;directory=TOOLS_V2_DIR,evidence=EVIDENCE_DIR,first=TOOLS_DIR)
    (;path,packet,tasks)=load_kernel_tools_v2(;directory);conditions=filehash(path)
    filehash(joinpath(evidence,"conditions.json"))==packet.comparison.evidence &&
        filehash(joinpath(first,"conditions.json"))==packet.comparison.first_tools || error("comparison drift")
    others=Dict{Tuple{String,Int,String},String}()
    for x in JSON3.read(read(joinpath(evidence,"summary.json"),String)).rows;others[(String(x.id),Int(x.seed),String(x.arm))]=String(x.status);end
    for x in JSON3.read(read(joinpath(first,"summary.json"),String)).rows;others[(String(x.id),Int(x.seed),TOOLS_ARM)]=String(x.status);end
    rows=Any[]
    for seed in packet.seeds,task in tasks,arm in TOOLS_V2_ARMS
        output=outputfile(directory,task,seed,arm.name)
        isfile(output) || (push!(rows,(;id=task.id,seed,arm=arm.name,status="pending"));continue)
        c=JSON3.read(read(output,String));c.conditions_sha256==conditions || error("candidate conditions drift: $output")
        s=JuliaTaskEval.evaluate(task,String(c.code);output=replace(output,".json"=>"_score.json"))
        names=[x.call===nothing ? "malformed" : String(x.call.name) for x in c.calls]
        push!(rows,(;id=task.id,seed,arm=arm.name,status=s.passed ? "passed" : "failed",class=failureclass(c,s),
            calls=length(c.calls),runs=count(==("run_julia"),names),lookups=count(==("lookup"),names),
            syntax_checks=count(==("check_syntax"),names),malformed=count(==("malformed"),names),forced=c.forced_closes,
            generated_tokens=c.generated_tokens,context_tokens=c.context_tokens,seconds=c.elapsed_seconds,truncated=c.truncated))
    end
    done=filter(r->r.status!="pending",rows)
    totals=Dict(a.name=>(;passed=count(r->r.arm==a.name&&r.status=="passed",done),evaluated=count(r->r.arm==a.name,done),
        classes=Dict(cl=>count(r->r.arm==a.name&&r.class==cl,done) for cl in unique(r.class for r in done)),
        minutes=sum((r.seconds for r in done if r.arm==a.name);init=0.0)/60,
        calls=sum((r.calls for r in done if r.arm==a.name);init=0),runs=sum((r.runs for r in done if r.arm==a.name);init=0),
        forced=sum((r.forced for r in done if r.arm==a.name);init=0)) for a in TOOLS_V2_ARMS)
    pair(arm,other)=(;wins=count(r->r.arm==arm&&r.status=="passed"&&others[(r.id,r.seed,other)]=="failed",done),
        losses=count(r->r.arm==arm&&r.status=="failed"&&others[(r.id,r.seed,other)]=="passed",done))
    for r in done;others[(r.id,r.seed,r.arm)]=r.status;end
    paired=Dict("direct_budget vs direct"=>pair("direct_budget","direct"),
        "tools_budget vs tools_parse_lookup"=>pair("tools_budget",TOOLS_ARM),
        "tools_run_budget vs tools_budget"=>pair("tools_run_budget","tools_budget"),
        "tools_run_budget vs direct"=>pair("tools_run_budget","direct"),
        "tools_run_budget vs api_pack"=>pair("tools_run_budget","api_pack"),
        "tools_run_budget vs runtime_feedback"=>pair("tools_run_budget","runtime_feedback"))
    pertask=Dict(t.id=>Dict(a.name=>count(r->r.id==t.id&&r.arm==a.name&&r.status=="passed",done) for a in TOOLS_V2_ARMS) for t in tasks)
    report=(;scope=packet.scope,totals,paired,pertask,rows)
    open(io->JSON3.pretty(io,report),joinpath(directory,"summary.json"),"w");report
end
