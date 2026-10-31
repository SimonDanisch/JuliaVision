include("run_projection_memory_pilot.jl")

function require_projection_hook_gate()
    path=joinpath(@__DIR__,"projection_model_hook_validation.json")
    isfile(path) || error("full-model identity gate required")
    gate=JSON3.read(read(path,String))
    gate.complete && gate.normal_final_waitidle || error("incomplete identity gate")
    current=inference_runtime_provenance()
    for (package,descriptor) in current
        descriptor.source_sha256==Dict(String(k)=>String(v) for (k,v) in pairs(gate.result.runtime_provenance[Symbol(package)].source_sha256)) || error("engine drift since identity gate")
    end
    file="projection_memory_generation.jl"
    gate.source_sha256[Symbol(file)]==bytes2hex(sha256(read(joinpath(@__DIR__,file)))) || error("hook drift")
    registration=joinpath(@__DIR__,"projection_datastore","registration.json")
    gate.datastore_registration_sha256==bytes2hex(sha256(read(registration))) || error("datastore drift")
    gate
end

function run_qualified_projection_pilot(;seconds=210)
    # Leave extra room for CPU checkpoint hashing before the pilot's GPU work.
    seconds>0 || throw(ArgumentError("time budget"))
    require_projection_hook_gate()
    run_projection_memory_pilot(;seconds)
end
