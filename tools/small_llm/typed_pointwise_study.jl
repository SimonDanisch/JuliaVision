isdefined(@__MODULE__,:kernel_transfer_state) || include("kernel_transfer_benchmark.jl")
isdefined(@__MODULE__,:CheckedTypedPointwiseLowering) || include("typed_pointwise_kernel_lowering.jl")

function score_typed_pointwise_study()
    qualificationpath=joinpath(@__DIR__,"typed_pointwise_lowering_validation.json")
    qualification=JSON3.read(read(qualificationpath,String))
    qualification.complete || error("software compiler qualification required")
    for (file,hash) in pairs(qualification.source_sha256)
        vulkan_benchmark_hash(joinpath(@__DIR__,String(file)))==hash || error("compiler qualification drift")
    end
    state=kernel_transfer_state();directory=joinpath(@__DIR__,"typed_pointwise_study_results");mkpath(directory)
    conditions=joinpath(directory,"conditions.json")
    datasets=("kernel_transfer_results","safe_reasoning_budget_results","safe_budget_transfer_extension_results")
    if !isfile(conditions)
        packet=(;scope="POST-HOC typed-array compiler expansion motivated by the controller pilot's host-style gather; NOT neural or independent held-out evidence",
            policy="uniform typed pointwise lowering then unchanged void-tail normalization, no best-score selection; all registered cases retained, no generation or retries",
            compiler_qualification_sha256=vulkan_benchmark_hash(qualificationpath),
            source_sha256=Dict(f=>vulkan_benchmark_hash(joinpath(@__DIR__,f)) for f in
                ("typed_pointwise_kernel_lowering.jl","typed_pointwise_study.jl","kernel_void_tail.jl")),
            dataset_conditions_sha256=Dict(d=>vulkan_benchmark_hash(joinpath(@__DIR__,d,"conditions.json")) for d in datasets))
        open(io->JSON3.pretty(io,packet),conditions,"w")
    end
    p=JSON3.read(read(conditions,String))
    p.compiler_qualification_sha256==vulkan_benchmark_hash(qualificationpath) || error("qualification receipt drift")
    for (file,hash) in pairs(p.source_sha256)
        vulkan_benchmark_hash(joinpath(@__DIR__,String(file)))==hash || error("study source drift")
    end
    rows=Any[]
    for dataset in datasets
        protocol=joinpath(@__DIR__,dataset,"conditions.json")
        vulkan_benchmark_hash(protocol)==p.dataset_conditions_sha256[Symbol(dataset)] || error("dataset protocol drift")
        registration=JSON3.read(read(protocol,String))
        task_ids=dataset=="safe_reasoning_budget_results" ? [String(registration.task)] :
            dataset=="safe_budget_transfer_extension_results" ? unique(String(c.task) for c in registration.order) : [t.id for t in state.tasks]
        for taskid in task_ids,arm in ("no_addon","api_pack")
            id=taskid*"_"*arm;source=joinpath(dirname(protocol),id*".json")
            if !isfile(source)
                push!(rows,(;dataset,id,status="pending"));continue
            end
            candidate=JSON3.read(read(source,String))
            candidate.conditions_sha256==vulkan_benchmark_hash(protocol) || error("generation protocol drift")
            if hasproperty(candidate,:inference_error)
                push!(rows,(;dataset,id,status="inference_error"));continue
            end
            code=String(candidate.code);lowered=CheckedTypedPointwiseLowering.lower(code)
            intermediate=lowered.accepted ? lowered.code : code
            tail=KernelVoidTail.normalize(intermediate);selected=tail.accepted ? tail.code : intermediate
            scorepath=joinpath(directory,dataset*"_"*id*"_score.json")
            rawpath=replace(source,".json"=>"_raw_score.json")
            raw=isfile(rawpath) ? JSON3.read(read(rawpath,String)) : nothing
            reused=raw!==nothing && raw.code_sha256==bytes2hex(sha256(selected))
            result=if reused
                raw
            elseif isfile(scorepath)
                prior=JSON3.read(read(scorepath,String))
                prior.code_sha256==bytes2hex(sha256(selected)) || error("compiler score drift")
                prior
            else
                task=only(filter(t->t.id==taskid,state.tasks))
                JuliaTaskEval.evaluate(task,selected;output=scorepath)
            end
            push!(rows,(;dataset,id,status=result.passed ? "passed" : "failed",lowered=lowered.accepted,
                admission=lowered.reason,void_tail=tail.accepted,reused_raw_score=reused,
                generation_sha256=vulkan_benchmark_hash(source),code_sha256=result.code_sha256))
        end
    end
    report=(;scope=p.scope,policy=p.policy,rows,complete=all(r->r.status!="pending",rows),conditions_sha256=vulkan_benchmark_hash(conditions))
    open(io->JSON3.pretty(io,report),joinpath(directory,"summary.json"),"w");report
end
