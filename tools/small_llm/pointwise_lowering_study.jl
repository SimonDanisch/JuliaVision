isdefined(@__MODULE__,:PointwiseKernelLowering) || include("pointwise_kernel_lowering.jl")
isdefined(@__MODULE__,:load_vulkan_kernel_conditions) || include("vulkan_kernel_benchmark.jl")

function freeze_pointwise_lowering_study()
    baseline=load_vulkan_kernel_conditions();directory=joinpath(@__DIR__,"pointwise_lowering_results");mkpath(directory)
    path=joinpath(directory,"conditions.json");isfile(path) && error("lowering policy already frozen")
    qualification=joinpath(@__DIR__,"pointwise_lowering_validation.json")
    packet=JSON3.read(read(qualification,String));packet.complete || error("unqualified lowering")
    for (file,hash) in pairs(packet.source_sha256)
        vulkan_benchmark_hash(joinpath(@__DIR__,String(file)))==hash || error("compiler qualification drift")
    end
    sources=Dict(t.id=>vulkan_benchmark_hash(joinpath(dirname(baseline.path),t.id*"_baseline.json")) for t in baseline.tasks)
    report=(;scope="post-hoc DEVELOPMENT diagnostic motivated by observed serial-loop baseline failure; not held-out or model-only intelligence evidence",
        rule="apply the same restricted compiler to ALL four frozen candidates; admitted transformation replaces baseline regardless of numerical score, rejected input retains baseline; no extra model generation",
        preconditions="distinct input/output storage, original serial reads in bounds, admitted Float32/Int subset and one-dimensional launch",
        qualification_sha256=vulkan_benchmark_hash(qualification),baseline_conditions_sha256=vulkan_benchmark_hash(baseline.path),
        baseline_outputs_sha256=sources,
        source_sha256=Dict(file=>vulkan_benchmark_hash(joinpath(@__DIR__,file)) for file in
            ("pointwise_lowering_study.jl","pointwise_kernel_lowering.jl")))
    open(io->JSON3.pretty(io,report),path,"w");report
end

function run_pointwise_lowering_study()
    baseline=load_vulkan_kernel_conditions();directory=joinpath(@__DIR__,"pointwise_lowering_results")
    path=joinpath(directory,"conditions.json");packet=JSON3.read(read(path,String))
    packet.baseline_conditions_sha256==vulkan_benchmark_hash(baseline.path) || error("baseline drift")
    packet.qualification_sha256==vulkan_benchmark_hash(joinpath(@__DIR__,"pointwise_lowering_validation.json")) || error("qualification drift")
    for (file,hash) in pairs(packet.source_sha256)
        vulkan_benchmark_hash(joinpath(@__DIR__,String(file)))==hash || error("lowering study drift")
    end
    rows=Any[]
    for task in baseline.tasks
        source=joinpath(dirname(baseline.path),task.id*"_baseline.json")
        packet.baseline_outputs_sha256[Symbol(task.id)]==vulkan_benchmark_hash(source) || error("candidate drift")
        candidate=JSON3.read(read(source,String));original=String(candidate.code)
        lowered=PointwiseKernelLowering.lower(original)
        output=joinpath(directory,task.id*".json")
        if !isfile(output)
            chosen=lowered.accepted ? lowered.code : original
            artifact=(;task=task.id,conditions_sha256=vulkan_benchmark_hash(path),
                baseline_sha256=vulkan_benchmark_hash(source),lowered...,selected_code=chosen)
            open(io->JSON3.pretty(io,artifact),output,"w")
        end
        saved=JSON3.read(read(output,String))
        saved.task==task.id && saved.conditions_sha256==vulkan_benchmark_hash(path) &&
            saved.baseline_sha256==vulkan_benchmark_hash(source) || error("transformed output drift")
        chosen=lowered.accepted ? lowered.code : original
        saved.selected_code==chosen || error("transformation not deterministic")
        result=JuliaTaskEval.evaluate(task,chosen;output=replace(output,".json"=>"_score.json"))
        original_score=JSON3.read(read(replace(source,".json"=>"_score.json"),String))
        original_score.code_sha256==bytes2hex(sha256(original)) || error("baseline score drift")
        push!(rows,(;task=task.id,status=result.passed ? "passed" : "failed",
            baseline_passed=original_score.passed,action=lowered.accepted ? "restricted_lowering" : "retain_baseline",reason=lowered.reason))
    end
    report=(;scope=packet.scope,rule=packet.rule,rows,passed=count(r->r.status=="passed",rows),
        baseline_passed=count(r->r.baseline_passed,rows),pending=0,
        limitation="functional software-Vulkan tests plus restricted independence argument; NOT measured Radeon performance or correctness outside stated preconditions")
    open(io->JSON3.pretty(io,report),joinpath(directory,"summary.json"),"w");report
end
