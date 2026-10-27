# Evidence and restriction on the eight authored kernel tasks: which way of
# handing the model API knowledge helps, at one budget, sampler and seed set.
# Pre-registered: tasks, arms, the exact evidence texts, seeds, decoding, the
# public smoke probes and the member-name tables are frozen before any
# generation. Retrieved evidence is stored, not recomputed, so an index change
# cannot move it.
#
# Arms on every task and seed:
#   direct          the prompt alone
#   api_pack        the curated installed API pack (kernel_api_pack.json)
#   retrieved_docs  BM25 over installed docs (Atomix included), the prompt as query
#   member_mask     the prompt alone, sampled through MemberNameConstraint
# Revisions of a direct candidate whose public smoke run raises:
#   runtime_feedback  the exception text
#   feedback_pack     the exception text plus the API pack cards of every module
#                     the exception names; identical to runtime_feedback when it
#                     names none, and then not generated twice
isdefined(@__MODULE__,:FrozenStudy) || include("transfer_studies.jl")
isdefined(@__MODULE__,:kernel_transfer_tasks) || include("kernel_transfer_tasks.jl")
isdefined(@__MODULE__,:MemberNameConstraint) || include("member_name_constraint.jl")
isdefined(@__MODULE__,:JuliaDocsRetrieval) || include("julia_docs_retrieval.jl")
import Atomix,KernelInterface
using JSON3,SHA

declare_sampling!(g,logits,c::MemberNameConstraint.State,sampler)=MemberNameConstraint.declare!(g,logits,c,sampler)

const EVIDENCE_DIR=joinpath(@__DIR__,"kernel_evidence_results")
const EVIDENCE_SEEDS=(20261024,20261025,20261026,20261027,20261028)
const EVIDENCE_DECODING=(;thinking=true,max_tokens=6144,output_chunk=128,temperature=0.6,top_p=0.95,topk=20,presence=0.0)
const EVIDENCE_ARMS=("direct","api_pack","retrieved_docs","member_mask")
const REVISION_ARMS=("runtime_feedback","feedback_pack")
const EVIDENCE_MODULES=["KI"=>KernelInterface,"KernelInterface"=>KernelInterface,"Atomix"=>Atomix]
# Which API pack cards document which module the answer can name.
const PACK_CARD_MODULES=Dict("kernelinterface_global_id"=>"KernelInterface","atomix_readme"=>"Atomix",
    "atomix_atomic_implementation"=>"Atomix")
const MODULE_NAMES=Dict("KI"=>"KernelInterface","KernelInterface"=>"KernelInterface","Atomix"=>"Atomix")
const EVIDENCE_SOURCES=("kernel_evidence_study.jl","member_name_constraint.jl","julia_docs_retrieval.jl",
    "kernel_public_smoke.jl","kernel_transfer_tasks.jl","vulkan_kernel_tasks.jl","kernel_api_pack.json",TRANSFER_SOURCES...)

# Public smoke inputs for the transfer tasks, in the style of KERNEL_PUBLIC_PROBES:
# small inputs the prompt describes, recording values or errors, never compared.
const TRANSFER_PUBLIC_PROBES=Dict(
    "transfer_clamped_gather"=>"__bench_launch__(kernel!,fill(-9f0,5),Float32[2,-3,1,8],Int32[0,2,9],Int32(3),Int32(4);threads=3)",
    "transfer_sincos_pair"=>"__bench_launch__(kernel!,fill(-9f0,4),fill(-9f0,4),Float32[0,1,-1],Int32(3);threads=3)",
    "transfer_flat_transpose"=>"__bench_launch__(kernel!,fill(-9f0,8),Float32[1,2,3,4,5,6,0,0],Int32(2),Int32(3);threads=6)",
    "transfer_atomic_sum"=>"__bench_launch__(kernel!,zeros(Float32,2),Int32[1,2,1],Float32[1,2,3],Int32(3);threads=3)")
evidenceprobe(task)=get(KERNEL_PUBLIC_PROBES,task.id,get(TRANSFER_PUBLIC_PROBES,task.id,nothing))
evidencetasks()=[vulkan_kernel_tasks();kernel_transfer_tasks()]

const REVIEW="Review your prior kernel against the original request. Fix any issues and return the complete executable kernel code."
const CONTEXT_HEADER="\n\nInstalled public API/source context:\n"
const FEEDBACK_HEADER="\n\nPublic software-Vulkan compilation/execution reported this exception (no private tests or oracle answers):\n"
const MODULE_DOCS_HEADER="\n\nInstalled documentation of the modules the exception names:\n"

"""The API pack cards, checked by their text hashes. Their origin files are
provenance only: one was copied from BonsaiRunner/src/kernels.jl, which has
changed since, while the card text is what the model reads."""
function evidencecards()
    packet=JSON3.read(read(joinpath(@__DIR__,"kernel_api_pack.json"),String))
    for c in packet.cards
        digest(String(c.text))==c.text_sha256 || error("API card text drift: $(c.id)")
    end
    [(;id=String(c.id),text=String(c.text)) for c in packet.cards]
end
cardtext(cards)=join(("### "*c.id*"\n"*c.text for c in cards),"\n\n")

"""The installed docs the retrieval arm searches: the default index plus Atomix."""
function evidenceindex()
    atomix=pkgdir(Atomix)
    JuliaDocsRetrieval.build_index(;directories=(JuliaDocsRetrieval.DOC_DIRECTORIES...,joinpath(atomix,"src"),joinpath(atomix,"docs","src")),
        files=[joinpath(atomix,"README.md")])
end

"""Modules an exception names: a listed module in backticks, alone or as `M.name`."""
namedmodules(error)=sort!(unique!([MODULE_NAMES[n] for n in keys(MODULE_NAMES) if occursin(Regex("`"*n*"[.`]"),error)]))

directmessages(task,context="")=["system"=>KERNEL_SYSTEM,"user"=>task.prompt*context]

"""Run the qualification gate, store every evidence text, and freeze."""
function freeze_kernel_evidence(model,backbone,tokenizer;directory=EVIDENCE_DIR)
    path=joinpath(directory,"conditions.json");isfile(path) && error("kernel evidence conditions already frozen")
    tasks=evidencetasks();cards=evidencecards();index=evidenceindex()
    tables=MemberNameConstraint.Tables(tokenizer,EVIDENCE_MODULES,Int(model.config.vocab_size))
    enc(s)=SmallLLMChecks.QwenText.encode(tokenizer,s)
    qualification=map(tasks) do task
        reference=JuliaTaskEval.evaluate(task,task.reference)
        probe=evidenceprobe(task)
        smoke=diagnose_kernel_public_smoke((;id=task.id,prompt=task.prompt),task.reference;probe)
        broken=replace(task.reference,"KI.get_global_id"=>"KI.get_global_idx")
        refused=diagnose_kernel_public_smoke((;id=task.id,prompt=task.prompt),broken;probe)
        mask=MemberNameConstraint.trace(tables,enc("```julia\n"*task.reference*"```"))
        (;id=task.id,reference_passed=reference.passed,
         smoke_reference_completed=smoke.analysis_completed&&smoke.payload.completed,
         smoke_broken_raised=refused.analysis_completed&&!refused.payload.completed,
         mask_accepts_reference=mask.refused===nothing,mask_member_positions=mask.constrained)
    end
    all(q->q.reference_passed&&q.smoke_reference_completed&&q.smoke_broken_raised&&q.mask_accepts_reference,qualification) ||
        error("qualification failed: $qualification")
    retrieved=Dict(task.id=>JuliaDocsRetrieval.format_evidence(JuliaDocsRetrieval.retrieve_api(index,task.prompt)) for task in tasks)
    packet=(;scope="pre-registered evidence/restriction comparison on eight authored kernel tasks; development evidence, not a benchmark",
        hypotheses=["api_pack and retrieved_docs raise the pass rate over direct",
            "member_mask removes nonexistent-name failures without lowering the pass rate elsewhere",
            "feedback_pack beats runtime_feedback where the exception names a module"],
        backbone=basename(backbone),backbone_sha256=isgguf(backbone) ? Qwen36GGUF.CHECKPOINT_SHA256 : nothing,
        decoding=EVIDENCE_DECODING,seeds=collect(EVIDENCE_SEEDS),arms=collect(EVIDENCE_ARMS),revision_arms=collect(REVISION_ARMS),
        revision_trigger="the direct candidate's public smoke run raises (syntax, compiler or runtime exception)",
        retention="a revision arm scores its revision when triggered, the direct candidate otherwise",
        tasks=[(;id=t.id,prompt_sha256=digest(t.prompt),setup_sha256=digest(t.setup),tests_sha256=digest(t.tests),
            reference_sha256=digest(t.reference),probe=evidenceprobe(t)) for t in tasks],
        qualification,api_pack=cardtext(cards),pack_card_modules=PACK_CARD_MODULES,retrieved_docs=retrieved,
        member_modules=[name=>string(M) for (name,M) in EVIDENCE_MODULES],member_names=tables.members,
        source_sha256=Dict(f=>filehash(joinpath(@__DIR__,f)) for f in EVIDENCE_SOURCES))
    mkpath(directory);open(io->JSON3.pretty(io,packet),path,"w");packet
end

function load_kernel_evidence(;directory=EVIDENCE_DIR)
    path=joinpath(directory,"conditions.json");packet=JSON3.read(read(path,String))
    for (file,hash) in pairs(packet.source_sha256)
        filehash(joinpath(@__DIR__,String(file)))==hash || error("kernel evidence source drift: $file")
    end
    tasks=evidencetasks()
    String.(getproperty.(packet.tasks,:id))==getproperty.(tasks,:id) || error("task order drift")
    for (t,r) in zip(tasks,packet.tasks)
        (digest(t.prompt),digest(t.setup),digest(t.tests),digest(t.reference))==
            (r.prompt_sha256,r.setup_sha256,r.tests_sha256,r.reference_sha256) || error("contract drift: $(t.id)")
    end
    (;directory,path,packet,tasks)
end

outputfile(directory,task,seed,arm)=joinpath(directory,"$(task.id)_s$(seed)_$(arm).json")
diagnosticfile(directory,task,seed)=joinpath(directory,"$(task.id)_s$(seed)_diagnostic.json")

function firstmessages(packet,task,arm)
    arm=="api_pack" ? directmessages(task,CONTEXT_HEADER*String(packet.api_pack)) :
    arm=="retrieved_docs" ? directmessages(task,CONTEXT_HEADER*String(packet.retrieved_docs[Symbol(task.id)])) :
    directmessages(task)
end

function revisionmessages(packet,task,code,error,arm)
    request=REVIEW*FEEDBACK_HEADER*error
    if arm=="feedback_pack"
        modules=namedmodules(error)
        cards=[c for c in evidencecards() if get(PACK_CARD_MODULES,c.id,"") in modules]
        isempty(cards) || (request*=MODULE_DOCS_HEADER*cardtext(cards))
    end
    ["system"=>KERNEL_SYSTEM,"user"=>task.prompt,"assistant"=>code,"user"=>request]
end

function evidenceanswer(model,tokenizer,tables,messages,seed,arm,d)
    constraint=arm=="member_mask" ? MemberNameConstraint.State(model.device,tables;thinking=Bool(d.thinking)) : nothing
    try
        answer=sampled_julia_answer(model,tokenizer,messages;max_tokens=Int(d.max_tokens),thinking=Bool(d.thinking),
            output_chunk=Int(d.output_chunk),seed=UInt32(seed),temperature=Float32(d.temperature),
            top_p=Float32(d.top_p),topk=Int(d.topk),presence=Float32(d.presence),constraint)
        (;answer...,constraint=constraint===nothing ? nothing : MemberNameConstraint.report(constraint))
    finally
        constraint===nothing || close(constraint)
    end
end

function saverow(output,row)
    isfile(output) && error("refuse regeneration: $output")
    open(io->JSON3.pretty(io,row),output,"w")
end

"""Generate every pending job with a loaded backbone: the four first-attempt
arms in a rotated order per task and seed, then the smoke runs, then revisions."""
function run_kernel_evidence(model,tokenizer;directory=EVIDENCE_DIR)
    (;path,packet,tasks)=load_kernel_evidence(;directory);d=packet.decoding
    tables=MemberNameConstraint.Tables(tokenizer,EVIDENCE_MODULES,Int(model.config.vocab_size))
    # JSON stores each module=>names pair as a one-entry object.
    [String(only(keys(o)))=>String.(only(values(o))) for o in packet.member_names]==tables.members || error("member tables drift")
    conditions=filehash(path);runtime=inference_runtime_provenance()
    row(task,seed,arm,messages,answer;extra...)=(;task=task.id,seed,arm,messages,backbone=describe(model),
        runtime_provenance=runtime,conditions_sha256=conditions,extra...,answer...)
    for (s,seed) in enumerate(packet.seeds),(k,task) in enumerate(tasks)
        arms=circshift(collect(EVIDENCE_ARMS),k+s)
        for arm in arms
            output=outputfile(directory,task,seed,arm);isfile(output) && continue
            messages=firstmessages(packet,task,arm)
            answer=evidenceanswer(model,tokenizer,tables,messages,seed,arm,d)
            saverow(output,row(task,seed,arm,messages,answer))
            println(task.id," s",seed," ",arm,": ",answer.generated_tokens," tokens, truncated=",answer.truncated);flush(stdout)
        end
    end
    for seed in packet.seeds,task in tasks
        direct=JSON3.read(read(outputfile(directory,task,seed,"direct"),String))
        diagnostic=diagnosticfile(directory,task,seed)
        if !isfile(diagnostic)
            o=diagnose_kernel_public_smoke((;id=task.id,prompt=task.prompt),String(direct.code);probe=evidenceprobe(task))
            open(io->JSON3.pretty(io,o),diagnostic,"w")
        end
        o=JSON3.read(read(diagnostic,String))
        o.code_sha256==digest(String(direct.code)) || error("diagnostic drift: $(task.id) s$seed")
        o.analysis_completed&&!o.payload.completed || continue
        error=String(o.payload.error)
        for arm in REVISION_ARMS
            output=outputfile(directory,task,seed,arm);isfile(output) && continue
            messages=revisionmessages(packet,task,String(direct.code),error,arm)
            provenance=(;direct_sha256=filehash(outputfile(directory,task,seed,"direct")),diagnostic_sha256=filehash(diagnostic),
                named_modules=namedmodules(error))
            if arm=="feedback_pack" && messages==revisionmessages(packet,task,String(direct.code),error,"runtime_feedback")
                # Same messages and seed give the same answer; record it, do not regenerate.
                same=JSON3.read(read(outputfile(directory,task,seed,"runtime_feedback"),String))
                saverow(output,(;same...,arm,same_as="runtime_feedback"))
                continue
            end
            answer=evidenceanswer(model,tokenizer,tables,messages,seed,arm,d)
            saverow(output,row(task,seed,arm,messages,answer;provenance...))
            println(task.id," s",seed," ",arm,": ",answer.generated_tokens," tokens, truncated=",answer.truncated);flush(stdout)
        end
    end
    nothing
end

"""Why a candidate failed, read off its scoring diagnostic."""
function failureclass(candidate,score)
    score.passed && return "passed"
    isempty(String(candidate.code)) && return "no_answer_within_budget"
    d=String(score.diagnostic)
    occursin("ParseError",d) && return "syntax"
    (occursin("does not define",d)||occursin("UndefVarError",d)||occursin("no method matching",d)) && return "unknown_or_misused_api"
    occursin("KernelError",d) && return "kernel_contract"
    occursin("Test Failed",d) && return "wrong_result"
    "other"
end

function score_kernel_evidence(;directory=EVIDENCE_DIR)
    (;path,packet,tasks)=load_kernel_evidence(;directory);conditions=filehash(path);rows=Any[]
    scored=Dict{String,Any}()
    function score(task,output)
        get!(scored,output) do
            c=JSON3.read(read(output,String))
            c.conditions_sha256==conditions || error("candidate conditions drift: $output")
            s=JuliaTaskEval.evaluate(task,String(c.code);output=replace(output,".json"=>"_score.json"))
            (;c,s,class=failureclass(c,s))
        end
    end
    for seed in packet.seeds,task in tasks
        for arm in EVIDENCE_ARMS
            output=outputfile(directory,task,seed,arm)
            isfile(output) || (push!(rows,(;id=task.id,seed,arm,status="pending"));continue)
            (;c,class)=score(task,output)
            push!(rows,(;id=task.id,seed,arm,status=class=="passed" ? "passed" : "failed",class,
                generated_tokens=c.generated_tokens,input_tokens=c.input_tokens,seconds=c.elapsed_seconds,
                constraint=c.constraint))
        end
        diagnostic=diagnosticfile(directory,task,seed)
        for arm in REVISION_ARMS
            isfile(diagnostic) || (push!(rows,(;id=task.id,seed,arm,status="pending"));continue)
            o=JSON3.read(read(diagnostic,String));triggered=o.analysis_completed&&!o.payload.completed
            output=triggered ? outputfile(directory,task,seed,arm) : outputfile(directory,task,seed,"direct")
            isfile(output) || (push!(rows,(;id=task.id,seed,arm,status="pending"));continue)
            (;c,class)=score(task,output)
            push!(rows,(;id=task.id,seed,arm,status=class=="passed" ? "passed" : "failed",class,
                action=triggered ? "revision" : "retain_direct",generated_tokens=c.generated_tokens,
                input_tokens=c.input_tokens,seconds=c.elapsed_seconds,named_modules=get(c,:named_modules,nothing)))
        end
    end
    arms=(EVIDENCE_ARMS...,REVISION_ARMS...)
    totals=Dict(arm=>(;passed=count(r->r.arm==arm&&r.status=="passed",rows),evaluated=count(r->r.arm==arm&&r.status!="pending",rows),
        classes=Dict(cl=>count(r->r.arm==arm&&get(r,:class,"")==cl,rows) for cl in unique(get(r,:class,"pending") for r in rows)),
        seconds=sum((r.seconds for r in rows if r.arm==arm&&r.status!="pending");init=0.0)) for arm in arms)
    # Paired against direct on the same task and seed.
    paired=Dict(arm=>(;wins=count(rows) do r
            r.arm==arm&&r.status=="passed"&&any(q->q.arm=="direct"&&q.id==r.id&&q.seed==r.seed&&q.status=="failed",rows)
        end,losses=count(rows) do r
            r.arm==arm&&r.status=="failed"&&any(q->q.arm=="direct"&&q.id==r.id&&q.seed==r.seed&&q.status=="passed",rows)
        end) for arm in arms if arm!="direct")
    pertask=Dict(t.id=>Dict(arm=>count(r->r.id==t.id&&r.arm==arm&&r.status=="passed",rows) for arm in arms) for t in tasks)
    report=(;scope=packet.scope,backbone=packet.backbone,seeds=packet.seeds,totals,paired,pertask,rows)
    open(io->JSON3.pretty(io,report),joinpath(directory,"summary.json"),"w");report
end
