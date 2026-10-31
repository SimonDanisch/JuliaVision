isdefined(@__MODULE__,:kernel_feedback_state) || include("kernel_runtime_feedback.jl")
isdefined(@__MODULE__,:CheckedPointwiseLowering) || include("checked_pointwise_lowering.jl")
isdefined(@__MODULE__,:KernelVoidTail) || include("kernel_void_tail.jl")

function freeze_kernel_compiler_composition()
    state=kernel_feedback_state();directory=joinpath(@__DIR__,"kernel_compiler_composition_results");mkpath(directory)
    path=joinpath(directory,"conditions.json");isfile(path) && error("compiler composition already frozen")
    qualifications=("pointwise_lowering_validation.json","kernel_void_tail_validation.json")
    for file in qualifications
        packet=JSON3.read(read(joinpath(@__DIR__,file),String));packet.complete || error("unqualified compiler pass")
        for (source,hash) in pairs(packet.source_sha256)
            vulkan_benchmark_hash(joinpath(@__DIR__,String(source)))==hash || error("qualification source drift")
        end
    end
    packet=(;scope="post-hoc DEVELOPMENT composition: same frozen public-error/self-review candidates plus deterministic compiler passes; not held-out/model-only intelligence evidence",
        rule="use original public-exception policy to choose each arm's code; uniformly attempt restricted pointwise lowering then append a void tail. No hidden-score selection or extra model generation",
        limitations=["motivated by observed serial-loop and void-return failures",
            "remaining model outputs were still pending at registration; every registered task/arm remains in the report",
            "void-tail pass preserves statements but cannot make arbitrary explicit or macro-generated returns void",
            "software numerical tests are neither Radeon timing nor general bounds/race proofs"],
        feedback_conditions_sha256=vulkan_benchmark_hash(state.path),
        qualifications=Dict(f=>vulkan_benchmark_hash(joinpath(@__DIR__,f)) for f in qualifications),
        source_sha256=Dict(f=>vulkan_benchmark_hash(joinpath(@__DIR__,f)) for f in
            ("kernel_compiler_composition.jl","checked_pointwise_lowering.jl","pointwise_kernel_lowering.jl",
                "kernel_void_tail.jl","test_checked_pointwise_lowering.jl")))
    open(io->JSON3.pretty(io,packet),path,"w");packet
end

function score_kernel_compiler_composition()
    state=kernel_feedback_state();directory=joinpath(@__DIR__,"kernel_compiler_composition_results")
    path=joinpath(directory,"conditions.json");packet=JSON3.read(read(path,String))
    packet.feedback_conditions_sha256==vulkan_benchmark_hash(state.path) || error("public policy drift")
    for (file,hash) in pairs(packet.source_sha256)
        vulkan_benchmark_hash(joinpath(@__DIR__,String(file)))==hash || error("compiler source drift")
    end
    for (file,hash) in pairs(packet.qualifications)
        vulkan_benchmark_hash(joinpath(@__DIR__,String(file)))==hash || error("qualification receipt drift")
    end
    rows=Any[]
    for task in state.baseline.tasks
        originalpath=joinpath(dirname(state.baseline.path),task.id*"_baseline.json")
        original=JSON3.read(read(originalpath,String))
        diagnosticpath=joinpath(state.directory,task.id*"_diagnostic.json")
        observation=JSON3.read(read(diagnosticpath,String))
        observation.code_sha256==bytes2hex(sha256(String(original.code))) || error("public observation drift")
        eligible=observation.analysis_completed && !observation.payload.completed
        for arm in state.packet.arms
            selected=eligible ? joinpath(state.directory,task.id*"_"*String(arm)*".json") : originalpath
            if !isfile(selected)
                push!(rows,(;task=task.id,arm,status="pending"));continue
            end
            candidate=JSON3.read(read(selected,String))
            if eligible
                candidate.conditions_sha256==vulkan_benchmark_hash(state.path) && candidate.task==task.id && candidate.arm==arm &&
                    candidate.baseline_sha256==vulkan_benchmark_hash(originalpath) &&
                    candidate.diagnostic_sha256==vulkan_benchmark_hash(diagnosticpath) || error("revision provenance drift")
            else
                candidate.conditions_sha256==state.packet.baseline_conditions_sha256 && candidate.task==task.id || error("baseline provenance drift")
            end
            code=String(candidate.code);pointwise=CheckedPointwiseLowering.lower(code)
            after_pointwise=pointwise.accepted ? pointwise.code : code
            tail=KernelVoidTail.normalize(after_pointwise);chosen=tail.accepted ? tail.code : after_pointwise
            output=joinpath(directory,task.id*"_"*String(arm)*".json")
            if !isfile(output)
                artifact=(;task=task.id,arm,conditions_sha256=vulkan_benchmark_hash(path),
                    selected_source_sha256=vulkan_benchmark_hash(selected),original_code_sha256=bytes2hex(sha256(code)),
                    pointwise_accepted=pointwise.accepted,pointwise_reason=pointwise.reason,
                    certificate=pointwise.certificate,void_tail_accepted=tail.accepted,void_tail_reason=tail.reason,code=chosen)
                open(io->JSON3.pretty(io,artifact),output,"w")
            end
            saved=JSON3.read(read(output,String))
            saved.task==task.id && saved.arm==arm && saved.conditions_sha256==vulkan_benchmark_hash(path) &&
                saved.selected_source_sha256==vulkan_benchmark_hash(selected) && saved.code==chosen || error("compiler artifact drift")
            result=JuliaTaskEval.evaluate(task,chosen;output=replace(output,".json"=>"_score.json"))
            rawpath=joinpath(state.directory,task.id*"_"*String(arm)*"_score.json")
            raw=JSON3.read(read(rawpath,String));raw.code_sha256==bytes2hex(sha256(code)) || error("original score drift")
            push!(rows,(;task=task.id,arm,status=result.passed ? "passed" : "failed",original_passed=raw.passed,
                pointwise=pointwise.accepted,void_tail=tail.accepted))
        end
    end
    report=(;scope=packet.scope,rule=packet.rule,rows,
        totals=Dict(String(arm)=>(;passed=count(r->r.arm==arm&&r.status=="passed",rows),
            original_passed=count(r->r.arm==arm&&hasproperty(r,:original_passed)&&r.original_passed,rows),
            pending=count(r->r.arm==arm&&r.status=="pending",rows)) for arm in state.packet.arms))
    open(io->JSON3.pretty(io,report),joinpath(directory,"summary.json"),"w");report
end
