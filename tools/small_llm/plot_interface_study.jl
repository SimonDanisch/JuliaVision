include("plot_interface_tasks.jl")
using JSON3,SHA
function freeze_plot_interface(;token_count,directory=joinpath(@__DIR__,"plot_interface_results"))
    ispath(directory)&&!isempty(readdir(directory))&&error("Preserve plot registration")
    digest(path)=bytes2hex(sha256(read(path)))
    referencepath=joinpath(@__DIR__,"plot_interface_reference_validation.json")
    reference=JSON3.read(read(referencepath,String))
    reference.complete&&length(reference.rows)==8&&all(r->r.correct_reference&&r.wrong_color_rejected,reference.rows)||error("References unqualified")
    for (file,hash) in pairs(reference.sources)
        digest(joinpath(@__DIR__,String(file)))==hash||error("Reference/frontend drift")
    end
    proofs=("safe_plot_builder_validation.json","safe_plot_julia_validation.json")
    for file in proofs
        proof=JSON3.read(read(joinpath(@__DIR__,file),String));proof.complete||error("Construction/render unqualified")
        for (name,hash) in pairs(proof.sources)
            digest(joinpath(@__DIR__,String(name)))==hash||error("Render/reference source drift")
        end
    end
    tasks=plot_interface_tasks();jobs=Any[]
    for (i,task) in enumerate(tasks),arm in (isodd(i) ? ("julia_source","plot_spec") : ("plot_spec","julia_source"))
        messages=plot_interface_messages(task,arm);tokens=token_count(messages)
        tokens isa Integer&&!(tokens isa Bool)&&tokens>0&&tokens+1024+32<=8192||error("Capacity failure, do not filter tasks")
        id=task.id*"_"*arm;basename(id)==id||error("Safe job IDs required")
        push!(jobs,(;id,task=task.id,arm,messages,input_tokens=tokens))
    end
    parentpath=joinpath(@__DIR__,"pubmedqa_retrieval_results/conditions.json");parent=JSON3.read(read(parentpath,String))
    for (file,hash) in pairs(parent.source_sha256)
        digest(joinpath(@__DIR__,String(file)))==hash||error("Qualified inference runtime drift")
    end
    result=(;ready_for_generation=true,
        scope="NEW8 authored related-family static Makie requests. Restricted Julia constructor program vs declarative JSON, SAME trusted drawing backend. Not unrestricted Julia, public held-out, model-weight/frontier or aesthetic-preference evidence.",
        policy="All16 prompts/one seed/budgets/frontends/references frozen BEFORE ANY plot answers. Same public API/schema contract in BOTH arms, only requested output representation differs. Counterbalanced order, one immutable answer, no retry/hidden feedback. Both use same whole-artifact fence normalization. No generated Julia executes: restricted constructor source compiles to checked drawing IR. Independent private specification matching; ALL16 generation workers retired before grading/rendering. Report admission, specification correctness and actual render/layout separately.",
        jobs,decoding=(;max_tokens=1024,thinking=false,seed=20261025,topk=20,temperature=0.6f0,top_p=0.95f0,presence=0f0,output_chunk=64),
        condition_legend=(;julia_source="restricted_native_Julia_constructor_source",plot_spec="declarative_plot_specification"),
        source_sha256=parent.source_sha256,portable_manifest_sha256=parent.portable_manifest_sha256,
        parent_runtime_conditions_sha256=digest(parentpath),project_sha256=parent.project_sha256,manifest_sha256=parent.manifest_sha256,
        public_task_sha256=digest(joinpath(@__DIR__,"plot_interface_tasks.jl")),registration_source_sha256=digest(@__FILE__),
        private_reference_validation_sha256=digest(referencepath),render_validation_sha256=Dict(file=>digest(joinpath(@__DIR__,file)) for file in proofs),
        private_scorer_sha256=digest(joinpath(@__DIR__,"score_plot_interface.jl")),
        private_sources=Dict(file=>digest(joinpath(@__DIR__,file)) for file in
            ("plot_interface_references.jl","plot_interface_admission.jl","safe_plot_spec.jl","safe_plot_julia_source.jl","score_pubmedqa_reading.jl","pubmedqa_metrics.jl","pubmedqa_reading_protocol.jl")))
    mkpath(directory);open(io->JSON3.pretty(io,result),joinpath(directory,"conditions.json"),"w");result
end
