# Frozen task studies rerun on another backbone, as NEW registrations. The tasks,
# prompts, private contracts and decoding settings are the original packet's and
# are checked by hash; today's sources and the backbone are recorded afresh. The
# original packets, their outputs and their drift checks are never touched.
#
# A study answers `studydir`, `studyoriginal`, `studysources`, `studyjobs`,
# `studydecoding` and `studyevaluate`; transfer_kernel_studies.jl has the kernel
# study and its runtime-feedback extension.
isdefined(@__MODULE__,:textsession) || include("text_backbone.jl")
isdefined(@__MODULE__,:sampled_julia_answer) || include("sampled_julia.jl")
isdefined(@__MODULE__,:inference_runtime_provenance) || include("runtime_provenance.jl")
using JSON3,SHA

abstract type FrozenStudy end

filehash(path)=bytes2hex(sha256(read(path)))
digest(s::AbstractString)=bytes2hex(sha256(s))

const TRANSFER_SOURCES=("transfer_studies.jl","text_backbone.jl","run_julia_tasks.jl","sampled_julia.jl",
    "gpu_sampling.jl","julia_eval.jl","julia_sandbox.sh","qwen35.jl","prefill_qwen35.jl","qwen_text.jl",
    "qwen36_moe_model.jl","qwen36_prefill.jl","qwen36_optim.jl","qwen36_gguf.jl","qwen36_gguf_kernels.jl",
    "moe_qwen35.jl","moe_quant_qwen35.jl")

# A study that revises another registration's outputs checks it runs as they did.
checkregistration(::FrozenStudy,backbone,decoding)=nothing

"""Register a rerun of `study` under `name` on the checkpoint at `backbone`.

`overrides` replaces decoding settings (a larger budget, say) and is recorded
next to them, so the packet states every way it departs from the original.
"""
function freeze_transfer(study::FrozenStudy,name,backbone;overrides::NamedTuple=(;))
    dir=studydir(study,name);path=joinpath(dir,"conditions.json")
    isfile(path) && error("$name conditions already frozen")
    base=JSON3.read(read(studyoriginal(study),String))
    decoding=merge((;(Symbol(k)=>v for (k,v) in pairs(studydecoding(study,base)))...),overrides)
    checkregistration(study,backbone,decoding)
    mkpath(dir);jobs=studyjobs(study,dir,base)
    packet=(;scope=String(base.scope)*"; rerun on $(basename(backbone)) as a new registration",
        original_conditions=relpath(studyoriginal(study),@__DIR__),original_conditions_sha256=filehash(studyoriginal(study)),
        backbone=basename(backbone),backbone_sha256=isgguf(backbone) ? Qwen36GGUF.CHECKPOINT_SHA256 : nothing,
        decoding,decoding_overrides=overrides,
        jobs=[(;id=j.id,arm=j.arm,messages_sha256=digest(JSON3.write(j.messages))) for j in jobs],
        source_sha256=Dict(f=>filehash(joinpath(@__DIR__,f)) for f in (TRANSFER_SOURCES...,studysources(study)...)))
    open(io->JSON3.pretty(io,packet),path,"w");packet
end

function load_transfer(study::FrozenStudy,name)
    dir=studydir(study,name);path=joinpath(dir,"conditions.json")
    packet=JSON3.read(read(path,String))
    for (file,hash) in pairs(packet.source_sha256)
        filehash(joinpath(@__DIR__,String(file)))==hash || error("transfer source drift: $file")
    end
    filehash(studyoriginal(study))==packet.original_conditions_sha256 || error("original conditions drift")
    jobs=studyjobs(study,dir,JSON3.read(read(studyoriginal(study),String)))
    [(;j.id,j.arm) for j in jobs]==[(;id=String(r.id),arm=String(r.arm)) for r in packet.jobs] || error("job drift")
    all(digest(JSON3.write(j.messages))==r.messages_sha256 for (j,r) in zip(jobs,packet.jobs)) || error("message drift")
    (;dir,path,packet,jobs)
end

outputpath(dir,job)=joinpath(dir,job.id*"_"*job.arm*".json")

"""Generate every pending job with an already loaded backbone; one sampled
attempt per job at the registered settings, each saved as it completes."""
function run_transfer(study::FrozenStudy,name,model,tokenizer)
    (;dir,path,packet,jobs)=load_transfer(study,name);d=packet.decoding;done=String[]
    runtime=inference_runtime_provenance()
    for job in jobs
        output=outputpath(dir,job);isfile(output) && continue
        answer=sampled_julia_answer(model,tokenizer,job.messages;max_tokens=Int(d.max_tokens),thinking=Bool(d.thinking),
            output_chunk=Int(d.output_chunk),seed=UInt32(d.seed),temperature=Float32(d.temperature),
            top_p=Float32(d.top_p),topk=Int(d.topk),presence=Float32(d.presence))
        row=(;task=job.id,arm=job.arm,messages=job.messages,backbone=describe(model),runtime_provenance=runtime,
            conditions_sha256=filehash(path),job.provenance...,answer...)
        open(io->JSON3.pretty(io,row),output,"w");push!(done,job.id)
        println(job.id," ",job.arm," saved; ",answer.generated_tokens," tokens, truncated=",answer.truncated);flush(stdout)
    end
    done
end

"""A generated candidate, checked to be this job's under this registration."""
function candidate(path,job,output)
    c=JSON3.read(read(output,String))
    (c.task,c.arm)==(job.id,job.arm) && c.conditions_sha256==filehash(path) || error("candidate identity drift: $(job.id)")
    all(getproperty(c,k)==v for (k,v) in pairs(job.provenance)) || error("candidate provenance drift: $(job.id)")
    c
end

function score_transfer(study::FrozenStudy,name)
    (;dir,path,packet,jobs)=load_transfer(study,name);rows=Any[]
    for job in jobs
        output=outputpath(dir,job)
        isfile(output) || (push!(rows,(;job.id,job.arm,status="pending"));continue)
        c=candidate(path,job,output)
        result=studyevaluate(study,job,String(c.code);output=replace(output,".json"=>"_score.json"))
        push!(rows,(;job.id,job.arm,status=result.passed ? "passed" : "failed",
            truncated=c.truncated,generated_tokens=c.generated_tokens))
    end
    report=(;scope=packet.scope,backbone=packet.backbone,rows,passed=count(r->r.status=="passed",rows),
        evaluated=count(r->r.status!="pending",rows),pending=count(r->r.status=="pending",rows))
    open(io->JSON3.pretty(io,report),joinpath(dir,"summary.json"),"w");report
end

include("transfer_kernel_studies.jl")
