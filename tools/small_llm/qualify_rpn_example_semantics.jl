using Test,JSON3,SHA
include("test_rpn_example_semantics.jl")
include("test_rpn_example_semantics_gpu.jl")

function qualify_rpn_example_semantics(device;destination=joinpath(@__DIR__,"rpn_example_semantics_software_validation.json"))
    isfile(destination)&&error("Preserve qualification")
    occursin("llvmpipe",lowercase(string(device)))||error("software-only gate")
    cpu=test_rpn_example_semantics();software=test_rpn_example_semantics_gpu(device)
    count(t)=let c=Test.get_test_counts(t);c.passes+c.cumulative_passes end
    files=("rpn_example_semantics.jl","rpn_example_semantics_gpu.jl",
        "test_rpn_example_semantics.jl","test_rpn_example_semantics_gpu.jl","qualify_rpn_example_semantics.jl")
    result=(;complete=true,scope="Infrastructure only: explicitly public-example numeric state. No model integration, no new model attempts, no correctness beyond finite supplied examples. GPU Float64 support required for overflow-safe tolerance comparison.",
        cpu_checks=count(cpu),software_checks=count(software),device=string(device),
        sources=Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in files))
    open(destination,"w") do io;JSON3.write(io,result);end
    println(JSON3.write(result));result
end
