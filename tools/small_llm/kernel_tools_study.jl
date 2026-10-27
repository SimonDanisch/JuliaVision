# The kernel tasks with tools the model calls itself (check_syntax, lookup), on
# the kernel evidence study's tasks, seeds, sampler and generation budget, so
# each answer pairs with that study's arms on the same task and seed. Those arms
# are deterministic and are read from their registration, not regenerated.
# Pre-registered: the tool specifications, the system prompt and the call limit
# are frozen with the conditions before any generation.
isdefined(@__MODULE__,:freeze_kernel_evidence) || include("kernel_evidence_study.jl")
isdefined(@__MODULE__,:tool_julia_answer) || include("tool_julia.jl")
using JSON3,SHA

const TOOLS_DIR=joinpath(@__DIR__,"kernel_tools_results")
const TOOLS_ARM="tools_parse_lookup"
const TOOLS_MAXCALLS=6
const TOOLS_SOURCES=("kernel_tools_study.jl","tool_julia.jl","qwen_tools.jl","qwen_tools_template_fixture.json",EVIDENCE_SOURCES...)

function freeze_kernel_tools(model,backbone,tokenizer;directory=TOOLS_DIR,evidence=EVIDENCE_DIR)
    path=joinpath(directory,"conditions.json");isfile(path) && error("kernel tools conditions already frozen")
    # The evidence study is complete; its packet is read by hash, not re-checked
    # against today's sources (which moved on after it froze).
    basepath=joinpath(evidence,"conditions.json")
    base=(;path=basepath,packet=JSON3.read(read(basepath,String)))
    tasks=evidencetasks()
    String.(getproperty.(base.packet.tasks,:id))==getproperty.(tasks,:id) || error("task drift against the evidence study")
    for (t,r) in zip(tasks,base.packet.tasks)
        (digest(t.prompt),digest(t.tests),digest(t.reference))==(r.prompt_sha256,r.tests_sha256,r.reference_sha256) || error("contract drift: $(t.id)")
    end
    packet=(;scope="the eight authored kernel tasks with model-called tools (check_syntax, lookup); pairs with the kernel evidence study by task and seed",
        hypothesis="model-called parse and lookup tools raise the pass rate over direct, and solve the nonexistent-API tasks that feedback alone does not",
        backbone=basename(backbone),backbone_sha256=isgguf(backbone) ? Qwen36GGUF.CHECKPOINT_SHA256 : nothing,
        decoding=base.packet.decoding,seeds=base.packet.seeds,arm=TOOLS_ARM,maxcalls=TOOLS_MAXCALLS,
        budget="max_tokens bounds the tokens generated over all rounds; tool responses are context, not budget",
        system=TOOL_SYSTEM,tools=[QwenTools.tooljson(t) for t in JuliaTools.parse_lookup().tools],
        comparison=(;conditions=relpath(base.path,@__DIR__),conditions_sha256=filehash(base.path)),
        tasks=base.packet.tasks,
        source_sha256=Dict(f=>filehash(joinpath(@__DIR__,f)) for f in TOOLS_SOURCES))
    mkpath(directory);open(io->JSON3.pretty(io,packet),path,"w");packet
end

function load_kernel_tools(;directory=TOOLS_DIR)
    path=joinpath(directory,"conditions.json");packet=JSON3.read(read(path,String))
    for (file,hash) in pairs(packet.source_sha256)
        filehash(joinpath(@__DIR__,String(file)))==hash || error("kernel tools source drift: $file")
    end
    tasks=evidencetasks()
    for (t,r) in zip(tasks,packet.tasks)
        (digest(t.prompt),digest(t.tests),digest(t.reference))==(r.prompt_sha256,r.tests_sha256,r.reference_sha256) || error("contract drift: $(t.id)")
    end
    [QwenTools.tooljson(t) for t in JuliaTools.parse_lookup().tools]==String.(packet.tools) || error("tool drift")
    packet.system==TOOL_SYSTEM || error("system prompt drift")
    (;directory,path,packet,tasks)
end

function run_kernel_tools(model,tokenizer;directory=TOOLS_DIR)
    (;path,packet,tasks)=load_kernel_tools(;directory);d=packet.decoding
    conditions=filehash(path);runtime=inference_runtime_provenance()
    for seed in packet.seeds,task in tasks
        output=outputfile(directory,task,seed,TOOLS_ARM);isfile(output) && continue
        messages=[QwenTools.Message("system",TOOL_SYSTEM),QwenTools.Message("user",task.prompt)]
        a=tool_julia_answer(model,tokenizer,messages;max_tokens=Int(d.max_tokens),thinking=Bool(d.thinking),seed=UInt32(seed),
            topk=Int(d.topk),temperature=Float32(d.temperature),top_p=Float32(d.top_p),presence=Float32(d.presence),
            maxcalls=Int(packet.maxcalls))
        saverow(output,(;task=task.id,seed,arm=TOOLS_ARM,system=TOOL_SYSTEM,prompt=task.prompt,backbone=describe(model),
            runtime_provenance=runtime,conditions_sha256=conditions,a...))
        println(task.id," s",seed,": ",length(a.calls)," calls, ",a.generated_tokens," tokens, truncated=",a.truncated);flush(stdout)
    end
    nothing
end

function score_kernel_tools(;directory=TOOLS_DIR,evidence=EVIDENCE_DIR)
    (;path,packet,tasks)=load_kernel_tools(;directory);conditions=filehash(path)
    filehash(joinpath(evidence,"conditions.json"))==packet.comparison.conditions_sha256 || error("comparison drift")
    base=JSON3.read(read(joinpath(evidence,"summary.json"),String))
    rows=Any[]
    for seed in packet.seeds,task in tasks
        output=outputfile(directory,task,seed,TOOLS_ARM)
        isfile(output) || (push!(rows,(;id=task.id,seed,status="pending"));continue)
        c=JSON3.read(read(output,String));c.conditions_sha256==conditions || error("candidate conditions drift: $output")
        s=JuliaTaskEval.evaluate(task,String(c.code);output=replace(output,".json"=>"_score.json"))
        names=[x.call===nothing ? "malformed" : String(x.call.name) for x in c.calls]
        push!(rows,(;id=task.id,seed,status=s.passed ? "passed" : "failed",class=failureclass(c,s),calls=length(c.calls),
            lookups=count(==("lookup"),names),syntax_checks=count(==("check_syntax"),names),malformed=count(==("malformed"),names),
            generated_tokens=c.generated_tokens,context_tokens=c.context_tokens,seconds=c.elapsed_seconds,truncated=c.truncated))
    end
    done=filter(r->r.status!="pending",rows)
    other(arm,r)=only(q.status for q in base.rows if q.arm==arm&&q.id==r.id&&q.seed==r.seed)
    paired=Dict(arm=>(;wins=count(r->r.status=="passed"&&other(arm,r)=="failed",done),
        losses=count(r->r.status=="failed"&&other(arm,r)=="passed",done))
        for arm in ("direct","api_pack","runtime_feedback","feedback_pack"))
    report=(;scope=packet.scope,passed=count(r->r.status=="passed",done),evaluated=length(done),
        classes=Dict(cl=>count(r->r.class==cl,done) for cl in unique(r.class for r in done)),
        calls=(;per_answer=sum(r.calls for r in done;init=0)/max(1,length(done)),answers_with_calls=count(r->r.calls>0,done),
            lookups=sum(r.lookups for r in done;init=0),syntax_checks=sum(r.syntax_checks for r in done;init=0),
            malformed=sum(r.malformed for r in done;init=0)),
        gpu_minutes=sum(r.seconds for r in done;init=0.0)/60,paired,
        pertask=Dict(t.id=>count(r->r.id==t.id&&r.status=="passed",done) for t in tasks),rows)
    open(io->JSON3.pretty(io,report),joinpath(directory,"summary.json"),"w");report
end
