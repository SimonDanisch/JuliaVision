# Physical-device fixture gate ONLY, no model or task-quality measurements.
include("device_finish.jl")
include("test_finite_choice_constraint.jl")
include("test_finite_choice_constraint_gpu.jl")
include("runtime_provenance.jl")
using JSON3,SHA,Test

const FINITE_CHOICE_GATE_SOURCES=("finite_choice_constraint.jl","finite_choice_constraint_gpu.jl",
    "test_finite_choice_constraint.jl","test_finite_choice_constraint_gpu.jl","token_prefix_trie.jl",
    "safe_reasoning_budget.jl","reasoning_budget.jl","gpu_sampling.jl","device_finish.jl",
    "runtime_provenance.jl","run_finite_choice_gate.jl")

function run_finite_choice_gate(expected_sources;output=joinpath(@__DIR__,"finite_choice_native_gate/result.json"))
    normpath(pwd())=="/tmp/small-llm-bosgame-20261025"&&Sys.total_memory()<=40*1024^3||error("Laptop snapshot only")
    any(s->isdefined(Main,s),(:model,:sampled_julia_answer,:run_julia_stencil_transfer))&&error("Fresh no-model worker required")
    Set(String.(keys(expected_sources)))==Set(FINITE_CHOICE_GATE_SOURCES)||error("Exact public source set required")
    for (f,h) in pairs(expected_sources)
        bytes2hex(sha256(read(joinpath(@__DIR__,String(f)))))==h||error("Source drift")
    end
    ispath(output)&&error("Preserve first physical attempt")
    mkpath(dirname(output));islink(dirname(output))&&error("Linked report directory")
    report=Dict{String,Any}("complete"=>false,"pid"=>getpid(),"source_sha256"=>expected_sources,
        "scope"=>"Finite canonical support fixtures and one-submission mask/sample/advance; no full-model or semantic/API selection quality result",
        "runtime_provenance"=>inference_runtime_provenance())
    save()=open(io->JSON3.pretty(io,report),output,"w")
    let
        let
            save();device=Mantle.Device("radeon")
            try
                for (name,tests) in (("host_checks",test_finite_choice_constraint()),("device_checks",test_finite_choice_constraint_gpu(device)))
                    counts=Test.get_test_counts(tests);report[name]=counts.passes+counts.cumulative_passes;save()
                end
            finally finish_device!(device) end
            report["complete"]=true;report["normal_final_waitidle"]=true;save();report
        end
    end
end
