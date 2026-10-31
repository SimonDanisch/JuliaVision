include("bayes_extraction_admission.jl")
using JSON3,SHA

function bayes_cpu_diagnostic_row(result,expected)
    failed=hasproperty(result,:inference_error)
    complete=!failed&&result.eos_completed&&!result.truncated_or_time_limited
    admitted=complete ? admit_bayes_extraction(String(result.answer),"direct_probability") : (;accepted=false,reason=failed ? "inference_error" : "time_or_token_limited",probability=nothing)
    error=admitted.accepted ? abs(admitted.probability-expected) : nothing
    (;id=result.id,status=error!==nothing&&error<=1e-5 ? "passed" : "failed",admitted=admitted.accepted,
        admission_reason=admitted.reason,probability=admitted.probability,expected_probability=expected,absolute_error=error,
        eos_completed=complete,inference_error=failed)
end

function score_bayes_cpu_reference(;directory=joinpath(@__DIR__,"bayes_cpu_reference_results"),requests=joinpath(@__DIR__,"bayes_cpu_reference_requests.json"),primary=joinpath(@__DIR__,"bayes_extraction_results"))
    digest(path)=bytes2hex(sha256(read(path)))
    completion=JSON3.read(read(joinpath(directory,"completion.json"),String));completion.complete&&completion.jobs==8||error("Complete ALL8 native diagnostics first")
    packet=JSON3.read(read(requests,String));conditions=JSON3.read(read(joinpath(directory,"conditions.json"),String))
    conditions.requests_sha256==digest(requests)&&conditions.script_sha256==digest(joinpath(@__DIR__,"reference_bayes_greedy_cpu.py"))||error("Native request/script drift")
    conditions.device=="cpu"&&conditions.do_sample===false||error("Native CPU/greedy setting drift")
    JSON3.read(read(joinpath(directory,"tokenizer_validation.json"),String)).all_match||error("Native and Julia tokenizers differ")
    primarypath=joinpath(primary,"summary.json");original=JSON3.read(read(primarypath,String))
    original.complete&&packet.original_primary_summary_sha256==digest(primarypath)||error("Original score drift")
    packet.original_conditions_sha256==digest(joinpath(primary,"conditions.json"))||error("Original conditions drift")
    before=Dict(String(r.id)=>r for r in original.rows if r.arm=="direct_probability")
    length(packet.jobs)==length(before)==8||error("ALL8 required")
    rows=map(packet.jobs) do job
        path=joinpath(directory,job.id*".json");result=JSON3.read(read(path,String))
        result.id==job.id||error("Output identity drift")
        (;bayes_cpu_diagnostic_row(result,before[job.id].expected_probability)...,output_sha256=digest(path),original_sampled_status=before[job.id].status)
    end
    curve=[(;absolute_tolerance=tolerance,correct=count(r->r.absolute_error!==nothing&&r.absolute_error<=tolerance,rows)) for tolerance in (1e-5,1e-4,1e-3,1e-2)]
    result=(;complete=true,scope="POSTHOC ALL8 SAME original prompts. Native FP32/greedy CPU diagnostic, different precision AND decoding, NOT isolated quantization effect, independent transfer or replacement primary score.",
        rows,total=8,passed=count(r->r.status=="passed",rows),curve,
        eos_completed=count(r->r.eos_completed,rows),admitted=count(r->r.admitted,rows),conditions_sha256=digest(joinpath(directory,"conditions.json")),
        original_summary_sha256=digest(primarypath),scorer_sha256=digest(@__FILE__),
        caveat="Even a large difference cannot uniquely identify Q4, sampler or engine error. Existing native-logit parity and a matched greedy quantized control would be needed for stronger diagnosis. Incomplete/time-limited outputs remain failures at this diagnostic budget; no regeneration.")
    open(io->JSON3.pretty(io,result),joinpath(directory,"summary.json"),"w");result
end
