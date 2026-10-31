include("test_rpn_example_semantics.jl")
include("test_rpn_example_semantics_gpu.jl")
include("test_rpn_example_constraint.jl")
include("test_rpn_example_constraint_gpu.jl")
include("device_finish.jl")
include("runtime_provenance.jl")
using Test,JSON3,SHA

const EXAMPLE_GATE_SOURCES=("rpn_example_semantics.jl","rpn_example_semantics_gpu.jl",
    "rpn_example_constraint.jl","rpn_example_constraint_gpu.jl",
    "test_rpn_example_semantics.jl","test_rpn_example_semantics_gpu.jl",
    "test_rpn_example_constraint.jl","test_rpn_example_constraint_gpu.jl",
    "rpn_prefix_grammar.jl","rpn_prefix_grammar_gpu.jl","token_prefix_trie.jl",
    "safe_reasoning_budget.jl","reasoning_budget.jl","gpu_sampling.jl",
    "device_finish.jl","runtime_provenance.jl","laptop_example_constraint_gate.jl")

function freeze_laptop_example_constraint_gate(;directory=joinpath(@__DIR__,"laptop_example_constraint_gate"))
    ispath(directory)&&!isempty(readdir(directory))&&error("Preserve gate")
    proofs=Dict{String,String}()
    for file in ("rpn_example_semantics_software_validation.json","rpn_example_constraint_software_validation.json")
        path=joinpath(@__DIR__,file);p=JSON3.read(read(path,String));p.complete||error("Unqualified software core")
        for (source,hash) in pairs(p.sources)
            basename(String(source))==String(source)&&bytes2hex(sha256(read(joinpath(@__DIR__,String(source)))))==hash||error("Source drift")
        end
        proofs[file]=bytes2hex(sha256(read(path)))
    end
    p=(;scope="Hardware infrastructure ONLY, synthetic public fixtures; no model loaded or task generation. Requires later fresh-worker full-model disabled-hook identity before any model-quality use.",
        expected_checks=1627,software_proofs=proofs,
        sources=Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in EXAMPLE_GATE_SOURCES))
    mkpath(directory);open(io->JSON3.pretty(io,p),joinpath(directory,"conditions.json"),"w");p
end

function run_laptop_example_constraint_gate(;directory=joinpath(@__DIR__,"laptop_example_constraint_gate"))
    normpath(pwd())=="/tmp/small-llm-bosgame-20261025"&&Sys.total_memory()<=40*1024^3||error("Laptop snapshot only, no setup")
    (isdefined(Main,:model)||isdefined(Main,:qwenmodel))&&error("Fresh kernel-only worker required")
    path=joinpath(directory,"conditions.json");p=JSON3.read(read(path,String))
    Set(String.(keys(p.sources)))==Set(EXAMPLE_GATE_SOURCES)||error("Source set")
    for (file,hash) in pairs(p.sources)
        bytes2hex(sha256(read(joinpath(@__DIR__,String(file)))))==hash||error("Source drift")
    end
    output=joinpath(directory,"result.json");ispath(output)&&error("Preserve gate attempt")
    report=Dict{String,Any}("complete"=>false,"pid"=>getpid(),"conditions_sha256"=>bytes2hex(sha256(read(path))),
        "scope"=>p.scope,"runtime_provenance"=>inference_runtime_provenance())
    save()=open(io->JSON3.pretty(io,report),output,"w")
    let
        let
            save();device=Mantle.Device("radeon")
            try
                tests=@testset "Public-example arithmetic and resident constraint hardware" begin
                    test_rpn_example_semantics()
                    test_rpn_example_semantics_gpu(device)
                    test_rpn_example_constraint()
                    test_rpn_example_constraint_gpu(device)
                end
                c=Test.get_test_counts(tests);report["passed"]=c.passes+c.cumulative_passes
                report["passed"]==p.expected_checks||error("Check count drift")
                save()
            finally
                finish_device!(device)
            end
            report["normal_final_waitidle"]=true;report["complete"]=true;save();report
        end
    end
end
