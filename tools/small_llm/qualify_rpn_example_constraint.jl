using Test,JSON3,SHA
include("test_rpn_example_constraint.jl")
include("test_rpn_example_constraint_gpu.jl")

function qualify_rpn_example_constraint(device;destination=joinpath(@__DIR__,"rpn_example_constraint_software_validation.json"))
    isfile(destination)&&error("Preserve qualification")
    occursin("llvmpipe",lowercase(string(device)))||error("software-only gate")
    parent=joinpath(@__DIR__,"rpn_example_semantics_software_validation.json")
    proof=JSON3.read(read(parent,String));proof.complete||error("numeric evaluator unqualified")
    for (file,hash) in pairs(proof.sources)
        basename(String(file))==String(file)&&bytes2hex(sha256(read(joinpath(@__DIR__,String(file)))))==hash||error("numeric evaluator drift")
    end
    cpu=test_rpn_example_constraint();software=test_rpn_example_constraint_gpu(device)
    count(t)=let c=Test.get_test_counts(t);c.passes+c.cumulative_passes end
    files=("rpn_example_constraint.jl","rpn_example_constraint_gpu.jl",
        "test_rpn_example_constraint.jl","test_rpn_example_constraint_gpu.jl","qualify_rpn_example_constraint.jl",
        "rpn_example_semantics.jl","rpn_prefix_grammar.jl","rpn_prefix_grammar_gpu.jl",
        "token_prefix_trie.jl","safe_reasoning_budget.jl","gpu_sampling.jl")
    result=(;complete=true,scope="Infrastructure only, trusted synthetic logits: shared-token prefix safety, resident sampler+numeric example state, invalid raw logits remain errors, forced incorrect finish fails closed. No full-model identity or physical-device qualification, no neural task-quality attempt.",
        cpu_checks=count(cpu),software_checks=count(software),device=string(device),
        parent_numeric_proof_sha256=bytes2hex(sha256(read(parent))),
        sources=Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in files))
    open(destination,"w") do io;JSON3.write(io,result);end
    println(JSON3.write(result));result
end
