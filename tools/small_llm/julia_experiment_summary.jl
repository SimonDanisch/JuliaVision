module JuliaExperimentSummary
using JSON3, SHA
include("julia_tasks.jl")

function summarize(;output_dir=joinpath(@__DIR__,"julia_tasks_results"),
                     validation_file=joinpath(@__DIR__,"julia_harness_validation.json"),
                     tasks=JULIA_TASKS,output=joinpath(output_dir,"summary.json"))
    readjson(path)=JSON3.read(read(path,String))
    validation=readjson(validation_file)
    qualified=[task for task in tasks if any(r->r.id==task.id&&r.passed,validation.references)]
    excluded=[task.id for task in tasks if !(task in qualified)]
    conditions=Any[]
    baseline_passed=0
    for task in qualified
        path=joinpath(output_dir,task.id*"_baseline_score.json")
        isfile(path) && readjson(path).passed && (baseline_passed+=1)
    end
    for arm in ("baseline","review","feedback","feedback_docs")
        passed=0;pending=0;generation_seconds=0.0;public_probe_seconds=0.0
        input_tokens=0;budget_tokens=0;truncated=0;rows=Any[]
        for task in qualified
            basepath=joinpath(output_dir,task.id*"_baseline_score.json")
            baseresult=isfile(basepath) ? readjson(basepath) : nothing
            # Repair applies only to failures. A solved baseline is retained,
            # rather than giving it a new chance to regress or extra attempts.
            retained=arm!="baseline" && baseresult!==nothing && baseresult.passed
            source_arm=retained ? "baseline" : arm
            scorepath=joinpath(output_dir,task.id*"_"*source_arm*"_score.json")
            result=isfile(scorepath) ? readjson(scorepath) : nothing
            result===nothing ? (pending+=1) : (passed+=result.passed)
            genpath=joinpath(output_dir,task.id*"_"*source_arm*".json")
            if isfile(genpath)
                gen=readjson(genpath)
                generation_seconds+=get(gen,:elapsed_seconds,0.0)
                input_tokens+=get(gen,:input_tokens,0)
                budget_tokens+=get(gen,:budget_tokens,0)
                truncated+=get(gen,:truncated,false)
            end
            if arm in ("feedback","feedback_docs") && !retained
                probe=joinpath(output_dir,task.id*"_public_probe.json")
                isfile(probe) && (public_probe_seconds+=get(readjson(probe),:elapsed_seconds,0.0))
            end
            push!(rows,(;task=task.id,status=result===nothing ? "pending" : result.passed ? "passed" : "failed",retained_baseline=retained))
        end
        push!(conditions,(;arm,passed,total=length(qualified),pending,
            complete=pending==0,accuracy=pending==0 ? passed/length(qualified) : nothing,
            net_extra_tasks_solved=pending==0 ? passed-baseline_passed : nothing,
            generation_seconds,public_probe_seconds,input_tokens,budget_tokens,truncated,rows))
    end
    native=Any[]
    for task in qualified
        source=joinpath(output_dir,task.id*"_native_cpu.json")
        score=joinpath(output_dir,task.id*"_native_cpu_score.json")
        isfile(source)&&isfile(score) || continue
        push!(native,(;task=task.id,passed=readjson(score).passed,
            precision=readjson(source).precision,scope="independent native control; different precision and output cap, not a paired quantization ablation"))
    end
    result=(;scope="small authored Julia diagnostic, single greedy non-thinking candidate; not held-out evidence of frontier intelligence",
        qualified_tasks=[t.id for t in qualified],excluded_unqualified_reference=excluded,
        baseline_passed,conditions,native,
        repair_policy="one 512-token revision per arm; self-review vs public-probe feedback vs feedback plus docs; private scoring assertions never enter repair prompts",
        timing_policy="generation times include prefill/fixed-budget decode/readback but exclude model load; public probe cold-process CPU cost recorded separately; not matched total wall time",
        not_evaluated=["rendered Makie aesthetic quality","full Radeon generated-kernel execution","broad held-out Julia ability","medical/legal/tax advice quality"],
        provenance_note="Raw prompts, supplied evidence and generated responses are retained per candidate. The following hashes describe the analysis-time source snapshot, not a claim that every earlier long-lived worker loaded that exact snapshot.",
        analysis_source_sha256=Dict(file=>bytes2hex(sha256(read(joinpath(@__DIR__,file)))) for file in
            ("julia_tasks.jl","julia_eval.jl","julia_sandbox.sh","run_julia_tasks.jl","julia_docs_retrieval.jl","models.toml")))
    open(io->JSON3.pretty(io,result),output,"w")
    result
end
end
