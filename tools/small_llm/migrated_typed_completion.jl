isdefined(@__MODULE__,:CheckedTypedPointwiseLowering)||include("typed_pointwise_kernel_lowering.jl")
isdefined(@__MODULE__,:KernelVoidTail)||include("kernel_void_tail.jl")
using JSON3,SHA

function migrated_typed_choice(code)
    lowered=CheckedTypedPointwiseLowering.lower(code)
    intermediate=lowered.accepted ? lowered.code : code
    tail=KernelVoidTail.normalize(intermediate)
    (;code=tail.accepted ? tail.code : intermediate,lowered=lowered.accepted,
        admission=lowered.reason,void_tail=tail.accepted)
end

"""Freeze decisions for ALL migrated controller answers+matched controls.
Only public candidate source goes to the compiler, never task IDs or grades.
Historical Bosgame slots/results remain untouched; this is a new posthoc audit.
"""
function freeze_migrated_typed_completion(;directory=joinpath(@__DIR__,"migrated_typed_completion_results"))
    ispath(directory)&&!isempty(readdir(directory))&&error("Preserve decisions")
    digest(path)=bytes2hex(sha256(read(path)))
    proofpath=joinpath(@__DIR__,"typed_pointwise_lowering_validation.json")
    proof=JSON3.read(read(proofpath,String));proof.complete||error("Compiler unqualified")
    for (file,hash) in pairs(proof.source_sha256)
        digest(joinpath(@__DIR__,String(file)))==hash||error("Compiler qualification drift")
    end
    rows=Any[]
    datasets=("laptop_pending_results","laptop_controller_controls_results")
    parents=Dict{String,String}()
    for dataset in datasets
        root=joinpath(@__DIR__,dataset);path=joinpath(root,"conditions.json")
        p=JSON3.read(read(path,String));length(p.jobs)==6||error("Entire six-condition cohort required")
        parents[dataset]=digest(path)
        for job in p.jobs
            source=joinpath(root,String(job.id)*".json");candidate=JSON3.read(read(source,String))
            candidate.conditions_sha256==digest(path)&&candidate.messages==job.messages||error("Original receipt drift")
            hasproperty(candidate,:inference_error)&&error("Retain infrastructure failures explicitly; do not quietly transform them")
            selected=migrated_typed_choice(String(candidate.code))
            push!(rows,(;dataset,id=job.id,task=job.task,arm=job.arm,
                generation_sha256=digest(source),raw_code_sha256=bytes2hex(sha256(String(candidate.code))),
                selected...,selected_code_sha256=bytes2hex(sha256(selected.code))))
        end
    end
    result=(;complete=true,scope="POSTHOC SAME12 migrated controller/control answers, compiler admission expansion ONLY; not new neural attempts or same-engine equivalence with historical Bosgame outputs",
        policy="All12 decisions frozen before grade lookup. Uniform typed pointwise lowering then frozen void-tail normalization, no task-ID rules or arithmetic repair. Reuse a prior grade ONLY for exactly equal source bytes; otherwise ungraded here, never silently call it a failure/pass.",
        jobs=rows,parent_conditions_sha256=parents,qualification_sha256=digest(proofpath),
        source_sha256=Dict(f=>digest(joinpath(@__DIR__,f)) for f in
            ("migrated_typed_completion.jl","typed_pointwise_kernel_lowering.jl","kernel_void_tail.jl")))
    mkpath(directory);open(io->JSON3.pretty(io,result),joinpath(directory,"conditions.json"),"w");result
end

function score_migrated_typed_completion(;directory=joinpath(@__DIR__,"migrated_typed_completion_results"))
    digest(path)=bytes2hex(sha256(read(path)))
    conditionpath=joinpath(directory,"conditions.json");p=JSON3.read(read(conditionpath,String))
    for (file,hash) in pairs(p.source_sha256)
        digest(joinpath(@__DIR__,String(file)))==hash||error("Study source drift")
    end
    rows=map(p.jobs) do job
        root=joinpath(@__DIR__,String(job.dataset));source=joinpath(root,String(job.id)*".json")
        digest(joinpath(root,"conditions.json"))==p.parent_conditions_sha256[Symbol(job.dataset)]&&digest(source)==job.generation_sha256||error("Parent drift")
        scores=Any[]
        for variant in ("raw","compiler")
            path=joinpath(root,String(job.id)*"_"*variant*"_score.json")
            isfile(path)||continue
            score=JSON3.read(read(path,String))
            score.code_sha256==job.selected_code_sha256&&push!(scores,(;score,path))
        end
        if isempty(scores)
            return (;dataset=job.dataset,id=job.id,status="ungraded_new_source",lowered=job.lowered,admission=job.admission)
        end
        length(unique(s.score.passed for s in scores))==1||error("Conflicting equal-source grades")
        rawpath=joinpath(root,String(job.id)*"_raw_score.json");raw=JSON3.read(read(rawpath,String))
        raw.code_sha256==job.raw_code_sha256||error("Raw grade drift")
        (;dataset=job.dataset,id=job.id,status=first(scores).score.passed ? "passed" : "failed",
            raw_passed=raw.passed,lowered=job.lowered,admission=job.admission,
            selected_code_sha256=job.selected_code_sha256,reused_grade_sha256=[digest(s.path) for s in scores])
    end
    result=(;complete=all(r->r.status in ("passed","failed"),rows),scope=p.scope,policy=p.policy,rows,
        conditions_sha256=digest(conditionpath),
        interpretation="Exact existing source-score reuse, not new execution/hardware qualification. Earlier software-jail tests are ordinary nonadversarial functional checks, NOT the later external test-blind audit or independent held-out evidence.")
    open(io->JSON3.pretty(io,result),joinpath(directory,"summary.json"),"w");result
end
