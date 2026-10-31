include("bigcodebench_x_study.jl")
include("bigcodebench_x_revision_policy.jl")

function freeze_bcbx_revisions(;directory=joinpath(@__DIR__,"bigcodebench_x_revision_results"),
        baseline_directory=joinpath(@__DIR__,"bigcodebench_x_results"),
        tokenizer_dir=abspath(joinpath(@__DIR__,"../../../../gen/qwen35-4b")))
    ispath(directory)&&!isempty(readdir(directory))&&error("preserve revision registration")
    policy_path=joinpath(@__DIR__,"bigcodebench_x_revision_policy.json")
    policy=JSON3.read(read(policy_path,String))
    for (file,hash) in pairs(policy.source_sha256)
        bcbx_digest(joinpath(@__DIR__,String(file)))==hash||error("registered policy drift")
    end
    baseline=laptop_shared_source_state(baseline_directory)
    bcbx_digest(baseline.path)==policy.parent_conditions_sha256||error("parent conditions drift")
    sources=merge(Dict(String(k)=>String(v) for (k,v) in pairs(baseline.packet.source_sha256)),
        Dict(String(k)=>String(v) for (k,v) in pairs(policy.source_sha256)),
        Dict(f=>bcbx_digest(joinpath(@__DIR__,f)) for f in ("bigcodebench_x_revisions.jl","score_bigcodebench_x_revisions.py",
            "bigcodebench_x_secure_audit.py","bigcodebench_x_secure_audit_validation.json")))
    tokenizer=SmallLLMChecks.QwenText.Tokenizer(tokenizer_dir)
    bindings=JSON3.read(read(joinpath(@__DIR__,"julia_namespace_contract.json"),String)).bindings
    originals=Dict{String,Any}();hashes=Dict{String,String}();prepared=Dict{String,Any}()
    for job in baseline.packet.jobs
        path=joinpath(baseline_directory,String(job.id)*".json");isfile(path)||error("baseline generation not complete: $(job.id)")
        c=JSON3.read(read(path,String));attempt=replace(path,".json"=>"_attempt.json")
        c.attempt_sha256==bcbx_digest(attempt)||error("baseline attempt drift")
        hashes[String(job.id)]=bcbx_digest(path);originals[String(job.id)]=c
        if hasproperty(c,:inference_error)
            prepared[String(job.id)]=(;eligible=false,reason="upstream_inference_error",messages=nothing);continue
        end
        c.conditions_sha256==bcbx_digest(baseline.path)&&c.messages==job.messages||error("baseline prompt drift")
        c.decode_submissions==c.decode_steps_executed&&c.sampling.draws==c.decode_steps_executed+1||error("baseline submission invariant")
        c.budget_tokens==baseline.packet.decoding.max_tokens&&c.thinking_budget==baseline.packet.thinking_budget||error("baseline budget drift")
        for field in (:seed,:topk,:temperature,:top_p,:presence)
            a=getproperty(c.sampling,field);b=getproperty(baseline.packet.decoding,field)
            (field in (:temperature,:top_p,:presence) ? Float32(a)==Float32(b) : a==b)||error("baseline sampling drift")
        end
        messages=Dict(mode=>bcbx_public_revision_messages(job.messages,c.code,mode,bindings) for mode in ("public_static","self_review"))
        fits=all(length(SmallLLMChecks.QwenText.chat_ids(tokenizer,m;thinking=true))+Int(policy.decoding.max_tokens)+32<=8192 for m in values(messages))
        prepared[String(job.id)]=(;eligible=fits,reason=fits ? nothing : "paired_context_capacity_failure",messages)
    end
    jobs=Any[]
    for item in policy.order
        state=prepared[String(item.parent_id)];original=only(filter(j->j.id==item.parent_id,baseline.packet.jobs))
        messages=state.messages===nothing ? [String(first(keys(m)))=>String(first(values(m))) for m in original.messages] : state.messages[String(item.mode)]
        push!(jobs,(;id=item.id,task=item.task,arm=item.mode,parent_id=item.parent_id,
            eligible=state.eligible,skip_reason=state.reason,messages))
    end
    p=baseline.packet
    packet=(;scope=policy.scope,policy=policy.policy,jobs,decoding=policy.decoding,thinking_budget=policy.thinking_budget,
        parent_conditions_sha256=p.parent_conditions_sha256,gate_sha256=p.gate_sha256,portable_manifest_sha256=p.portable_manifest_sha256,
        baseline_conditions_sha256=bcbx_digest(baseline.path),baseline_generation_sha256=hashes,
        policy_sha256=bcbx_digest(policy_path),snapshot_manifest_sha256=p.snapshot_manifest_sha256,
        reference_validation_sha256=p.reference_validation_sha256,adapter_validation_sha256=p.adapter_validation_sha256,
        project_sha256=p.project_sha256,manifest_sha256=p.manifest_sha256,source_sha256=sources,
        grading="Unchanged original external tests, test-blind execution. Both arms kept for ALL8; upstream/capacity skips retained, not cherry-picked. Final scorer qualification required before generation.")
    mkpath(directory);open(io->JSON3.pretty(io,packet),joinpath(directory,"conditions.json"),"w");packet
end

function run_bcbx_revisions(;directory=joinpath(@__DIR__,"bigcodebench_x_revision_results"))
    s=laptop_shared_source_state(directory);p=s.packet
    p.policy_sha256==bcbx_digest(joinpath(@__DIR__,"bigcodebench_x_revision_policy.json"))||error("policy drift")
    pending=filter(j->!isfile(joinpath(directory,String(j.id)*".json")),p.jobs)
    isempty(pending)&&return (;complete=true)
    job=first(pending)
    if !job.eligible
        path=joinpath(directory,String(job.id)*".json")
        open(io->JSON3.pretty(io,(;id=job.id,not_generated=true,reason=job.skip_reason,conditions_sha256=bcbx_digest(s.path))),path,"w")
        return (;not_generated=true,id=job.id)
    end
    run_bigcodebench_x_study(;directory)
end
