isdefined(@__MODULE__,:test_knn_memory) || include("test_knn_memory.jl")
using JSON3, SHA
isdefined(@__MODULE__,:inference_runtime_provenance) || include("runtime_provenance.jl")

function validate_knn_memory(device;output)
    isfile(output) && error("validation evidence already exists; preserve previous attempts")
    started=time();counts=Test.get_test_counts(test_knn_memory(device))
    Mantle.waitidle(device)
    report=(;scope="shared representation-keyed memory kernel qualification ONLY; no model integration or intelligence claim",
        device=string(Mantle.devicename(device)),passed=counts.passes+counts.cumulative_passes,
        failed=counts.fails+counts.cumulative_fails,errors=counts.errors+counts.cumulative_errors,
        elapsed_seconds=time()-started,runtime_provenance=inference_runtime_provenance(),
        source_sha256=Dict(file=>bytes2hex(sha256(read(joinpath(@__DIR__,file)))) for file in
            ("knn_memory.jl","test_knn_memory.jl","validate_knn_memory.jl")))
    open(io->JSON3.pretty(io,report),output,"w")
    report
end
