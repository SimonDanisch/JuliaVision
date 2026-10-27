# Included by transfer_studies.jl. The authored kernel study and its
# runtime-feedback extension as transferable studies.
isdefined(@__MODULE__,:load_vulkan_kernel_conditions) || include("vulkan_kernel_benchmark.jl")
isdefined(@__MODULE__,:diagnose_kernel_public_smoke) || include("kernel_public_smoke.jl")

const KERNEL_SYSTEM="You are a Julia GPU programmer. Return only the requested executable kernel code."

"""The four authored KernelInterface kernel bodies, scored on llvmpipe."""
struct VulkanKernelStudy<:FrozenStudy end

studydir(::VulkanKernelStudy,name)=joinpath(@__DIR__,"vulkan_kernel_results_"*name)
studyoriginal(::VulkanKernelStudy)=joinpath(@__DIR__,"vulkan_kernel_results","conditions.json")
studysources(::VulkanKernelStudy)=("transfer_kernel_studies.jl","vulkan_kernel_benchmark.jl","vulkan_kernel_tasks.jl")
studydecoding(::VulkanKernelStudy,packet)=packet.baseline
studyevaluate(::VulkanKernelStudy,job,code;output)=JuliaTaskEval.evaluate(job.task,code;output)

"""The tasks of the original packet, checked against today's definitions."""
function kerneltasks(packet)
    tasks=vulkan_kernel_tasks()
    String.(getproperty.(packet.tasks,:id))==getproperty.(tasks,:id) || error("task order drift")
    for (task,row) in zip(tasks,packet.tasks)
        all(digest(getproperty(task,field))==getproperty(row,Symbol(field,"_sha256"))
            for field in (:prompt,:setup,:tests)) || error("contract drift: $(task.id)")
    end
    tasks
end

studyjobs(::VulkanKernelStudy,dir,packet)=[(;id=task.id,arm="sampled_baseline",task,provenance=(;),
    messages=["system"=>KERNEL_SYSTEM,"user"=>task.prompt]) for task in kerneltasks(packet)]

"""One revision of every kernel whose public smoke run raised: self-review, or
self-review with the exception text, in the original's counterbalanced order.

`baseline` names the kernel-study transfer whose outputs are revised. Its
outputs are complete records, so they are read with identity checks only; the
source checks guard regenerating them, not reading them.
"""
struct KernelFeedbackStudy<:FrozenStudy
    baseline::String
end

studydir(::KernelFeedbackStudy,name)=joinpath(@__DIR__,"kernel_runtime_feedback_results_"*name)
studyoriginal(::KernelFeedbackStudy)=joinpath(@__DIR__,"kernel_runtime_feedback_results","conditions.json")
studysources(::KernelFeedbackStudy)=("transfer_kernel_studies.jl","kernel_runtime_feedback.jl","kernel_public_smoke.jl",
    "vulkan_kernel_benchmark.jl","vulkan_kernel_tasks.jl")
studydecoding(::KernelFeedbackStudy,packet)=packet.decoding
studyevaluate(::KernelFeedbackStudy,job,code;output)=JuliaTaskEval.evaluate(job.task,code;output)

baselinedir(s::KernelFeedbackStudy)=studydir(VulkanKernelStudy(),s.baseline)
baselinepacket(s::KernelFeedbackStudy)=JSON3.read(read(joinpath(baselinedir(s),"conditions.json"),String))

function checkregistration(s::KernelFeedbackStudy,backbone,decoding)
    base=baselinepacket(s)
    basename(backbone)==base.backbone || error("revisions must run on their baselines' backbone, $(base.backbone)")
    want=(;(Symbol(k)=>v for (k,v) in pairs(base.decoding))...)
    decoding==want || error("revisions must decode as their baselines did: $decoding vs $want")
end

"""Every task's baseline candidate, its public diagnostic and whether it is
eligible for revision. Diagnostics are computed once into `dir`."""
function feedbackstate(s::KernelFeedbackStudy,dir)
    conditions=joinpath(baselinedir(s),"conditions.json")
    original=JSON3.read(read(studyoriginal(VulkanKernelStudy()),String))
    map(enumerate(kerneltasks(original))) do (i,task)
        source=joinpath(baselinedir(s),task.id*"_sampled_baseline.json")
        c=JSON3.read(read(source,String))
        c.task==task.id && c.arm=="sampled_baseline" && c.conditions_sha256==filehash(conditions) ||
            error("baseline identity drift: $(task.id)")
        diagnostic=joinpath(dir,task.id*"_diagnostic.json")
        if !isfile(diagnostic)
            observation=diagnose_kernel_public_smoke((;id=task.id,prompt=task.prompt),String(c.code))
            open(io->JSON3.pretty(io,observation),diagnostic,"w")
        end
        observation=JSON3.read(read(diagnostic,String))
        observation.code_sha256==digest(String(c.code)) || error("diagnostic drift: $(task.id)")
        (;i,task,source,diagnostic,baseline=c,observation,
         eligible=observation.analysis_completed&&!observation.payload.completed)
    end
end

function studyjobs(s::KernelFeedbackStudy,dir,packet)
    jobs=Any[]
    for st in feedbackstate(s,dir)
        st.eligible || continue
        for arm in (isodd(st.i) ? ("self_review","runtime_feedback") : ("runtime_feedback","self_review"))
            request="Review your prior kernel against the original request. Fix any issues and return the complete executable kernel code."
            arm=="runtime_feedback" && (request*="\n\nPublic software-Vulkan compilation/execution reported this exception (no private tests or oracle answers):\n"*String(st.observation.payload.error))
            push!(jobs,(;id=st.task.id,arm,task=st.task,
                provenance=(;baseline_sha256=filehash(st.source),diagnostic_sha256=filehash(st.diagnostic)),
                messages=["system"=>KERNEL_SYSTEM,"user"=>st.task.prompt,"assistant"=>String(st.baseline.code),"user"=>request]))
        end
    end
    jobs
end

"""The original's retention rule: an eligible task scores its one revision even
when worse; a task whose smoke run raised nothing keeps its baseline."""
function score_transfer(s::KernelFeedbackStudy,name)
    (;dir,path,packet,jobs)=load_transfer(s,name);rows=Any[]
    arms=String.(JSON3.read(read(studyoriginal(s),String)).arms)
    for st in feedbackstate(s,dir),arm in arms
        output=joinpath(dir,st.task.id*"_"*arm*".json")
        c=if st.eligible
            job=only(filter(j->(j.id,j.arm)==(st.task.id,arm),jobs))
            isfile(output) || (push!(rows,(;id=st.task.id,arm,status="pending"));continue)
            candidate(path,job,output)
        else
            st.baseline
        end
        result=JuliaTaskEval.evaluate(st.task,String(c.code);output=replace(output,".json"=>"_score.json"))
        push!(rows,(;id=st.task.id,arm,status=result.passed ? "passed" : "failed",
            action=st.eligible ? "single_revision" : "retain_baseline",truncated=c.truncated,generated_tokens=c.generated_tokens))
    end
    report=(;scope=packet.scope,backbone=packet.backbone,baseline=s.baseline,rows,
        totals=Dict(arm=>(;passed=count(r->r.arm==arm&&r.status=="passed",rows),
            pending=count(r->r.arm==arm&&r.status=="pending",rows)) for arm in arms))
    open(io->JSON3.pretty(io,report),joinpath(dir,"summary.json"),"w");report
end
