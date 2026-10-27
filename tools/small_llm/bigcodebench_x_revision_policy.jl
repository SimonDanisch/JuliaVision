isdefined(@__MODULE__,:PublicProgramDiagnostics)||include("public_program_diagnostics.jl")
isdefined(@__MODULE__,:QualifiedBindingDiagnostics)||include("qualified_binding_diagnostics.jl")
using JSON3,SHA
bcbx_policy_digest(path)=bytes2hex(open(sha256,path))

function freeze_bcbx_revision_policy(;path=joinpath(@__DIR__,"bigcodebench_x_revision_policy.json"))
    ispath(path)&&error("preserve registered policy")
    parentpath=joinpath(@__DIR__,"bigcodebench_x_results/conditions.json")
    parent=JSON3.read(read(parentpath,String));length(parent.jobs)==8||error("fixed cohort")
    for file in ("public_program_diagnostics_validation.json","qualified_binding_diagnostics_validation.json")
        p=JSON3.read(read(joinpath(@__DIR__,file),String));p.complete||error("analyzer qualification")
        for (source,hash) in pairs(p.source_sha256)
            bcbx_policy_digest(joinpath(@__DIR__,String(source)))==hash||error("analyzer drift")
        end
    end
    order=Any[]
    for (i,job) in enumerate(parent.jobs),mode in (isodd(i) ? ("public_static","self_review") : ("self_review","public_static"))
        push!(order,(;parent_id=job.id,task=job.task,mode,id=replace(String(job.id),"_baseline"=>"_"*mode)))
    end
    packet=(;scope="POSTHOC workflow development: policy constructed after observing first external baseline API failure, not an independent held-out intervention. ALL8 tasks retained, no selection by hidden scores.",
        policy="One revision per task per arm, counterbalanced, same original task/prior CODE and identical fixed decoding budget. Public-static arm gets syntax/dependency-contract/literal installed-namespace diagnostics; self-review arm gets no diagnostics. Neither reads benchmark tests, expected answers or grading traces. No generated code executed by analyzers. No arithmetic/API replacements inferred.",
        parent_conditions_sha256=bcbx_policy_digest(parentpath),order,
        decoding=(;parent.decoding...,max_tokens=4096),thinking_budget=2048,
        completion_gate="Derive final messages ONLY when all8 baseline outputs exist; verify receipts and capacity. Freeze final generation+grader sources before any revision. This policy alone does NOT authorize skipping qualification or creating model-quality claims.",
        source_sha256=Dict(file=>bcbx_policy_digest(joinpath(@__DIR__,file)) for file in
            ("bigcodebench_x_revision_policy.jl","public_program_diagnostics.jl","qualified_binding_diagnostics.jl",
                "julia_namespace_contract.json","public_program_diagnostics_validation.json","qualified_binding_diagnostics_validation.json")))
    open(io->JSON3.pretty(io,packet),path,"w");packet
end

function bcbx_public_revision_messages(original_messages,code,mode,bindings)
    mode in ("public_static","self_review")||error("unknown policy arm")
    messages=[String(first(keys(m)))=>String(first(values(m))) for m in original_messages]
    push!(messages,"assistant"=>String(code))
    instruction="Revise the preceding standalone Julia program against the original specification and the published package contract. Return ONLY the complete corrected program. Independently verify algorithm, stdin/stdout format and API calls. No hidden benchmark tests or expected answers are supplied.\n\n"
    if mode=="public_static"
        instruction*=PublicProgramDiagnostics.render(PublicProgramDiagnostics.analyze(code))*"\n"*
            QualifiedBindingDiagnostics.render(QualifiedBindingDiagnostics.analyze(code,bindings))
    else
        instruction*="Self-review control: no tool diagnostics are supplied."
    end
    push!(messages,"user"=>instruction);messages
end
