include("score_laptop_shared_source.jl")
include("invariant_pointwise_lowering.jl")

"""Uniform posthoc compiler over BOTH arms, preserving frozen interface scores."""
function score_invariant_pointwise_study()
    validation=joinpath(@__DIR__,"invariant_pointwise_validation.json")
    v=JSON3.read(read(validation,String));v.complete||error("compiler unqualified")
    for (file,hash) in pairs(v.source_sha256)
        bytes2hex(open(sha256,joinpath(@__DIR__,String(file))))==hash||error("compiler validation drift")
    end
    directory=joinpath(@__DIR__,"invariant_pointwise_study_results");mkpath(directory)
    source_directory=joinpath(@__DIR__,"laptop_shared_source_results")
    s=laptop_shared_source_state(source_directory);rows=Any[];tasks=shared_source_kernel_tasks()
    cpu_setup="function __bench_launch__(f,initial,args...;threads);out=copy(initial);f(out,args...);out;end"
    for job in s.packet.jobs
        path=joinpath(source_directory,String(job.id)*".json")
        if !isfile(path)
            push!(rows,(;id=job.id,status="pending"));continue
        end
        candidate=JSON3.read(read(path,String))
        candidate.conditions_sha256==bytes2hex(open(sha256,s.path))||error("protocol drift")
        if hasproperty(candidate,:inference_error)
            push!(rows,(;id=job.id,status="inference_error"));continue
        end
        task=only(filter(t->t.id==job.task,tasks));code=String(candidate.code)
        lowered=InvariantPointwiseLowering.lower(code)
        target=lowered.accepted ? lowered.code : code
        tail=KernelVoidTail.normalize(target);selected=tail.accepted ? tail.code : target
        function evaluate_once(task,body,suffix)
            scorepath=joinpath(directory,String(job.id)*"_"*suffix*"_score.json")
            if isfile(scorepath)
                old=JSON3.read(read(scorepath,String));old.code_sha256==bytes2hex(sha256(body))||error("score drift");old
            else
                JuliaTaskEval.evaluate(task,body;output=scorepath)
            end
        end
        cpu=lowered.accepted ? evaluate_once((;id=task.id,category="base",setup=cpu_setup,tests=task.tests),code,"cpu") : nothing
        gpu=evaluate_once(task,selected,"gpu")
        push!(rows,(;id=job.id,arm=job.arm,task=job.task,status=gpu.passed&&(cpu===nothing||cpu.passed) ? "passed" : "failed",
            compiler_applied=lowered.accepted,certificate=lowered.certificate,cpu_passed=cpu===nothing ? nothing : cpu.passed,
            gpu_passed=gpu.passed,generation_sha256=bytes2hex(open(sha256,path))))
    end
    report=(;scope="POSTHOC uniform compiler admission expansion motivated by one observed scalar-alias failure. Same compiler route for BOTH arms; no model regeneration, arithmetic repair or replacement of frozen interface scores. NOT held-out/model-only evidence.",
        rows,complete=all(r->r.status in ("passed","failed","inference_error"),rows),
        conditions_sha256=bytes2hex(open(sha256,s.path)),validation_sha256=bytes2hex(open(sha256,validation)),
        source_sha256=Dict(f=>bytes2hex(open(sha256,joinpath(@__DIR__,f))) for f in
            ("invariant_pointwise_study.jl","invariant_pointwise_lowering.jl","typed_pointwise_kernel_lowering.jl")))
    open(io->JSON3.pretty(io,report),joinpath(directory,"summary.json"),"w");report
end
