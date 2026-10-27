include("run_bayes_native_audit.jl")
include("bayes_thinking_admission.jl")

"""ROOT-only parity packet for ALL admitted thinking-generated networks.
Expectations are software execution of each GENERATED network, including wrong
ones; no task-answer/reference lookup occurs here. Transfer RAM only to a fresh
no-model worker after ALL16 generation workers retire.
"""
function prepare_bayes_thinking_native_audit(device;directory=joinpath(@__DIR__,"bayes_thinking_results"),destination=joinpath(@__DIR__,"bayes_thinking_native_audit/private_packet.json"))
    occursin("llvmpipe",lowercase(sprint(show,device)))||error("Software reference device required")
    isfile(destination)&&error("Preserve private parity packet")
    digest(path)=bytes2hex(sha256(read(path)))
    conditions=joinpath(directory,"conditions.json");summarypath=joinpath(directory,"summary.json")
    p=JSON3.read(read(conditions,String));summary=JSON3.read(read(summarypath,String))
    summary.complete&&length(summary.rows)==16&&length(p.jobs)==16||error("Finish ALL16 generation/scoring first")
    summary.conditions_sha256==digest(conditions)||error("Score provenance drift")
    for file in ("portable_evidence.jl","bayes_json.jl","bayes_thinking_admission.jl","bayes_extraction_admission.jl")
        p.private_source_sha256[Symbol(file)]==digest(joinpath(@__DIR__,file))||error("Parser/solver drift")
    end
    generation_pids=Int[]
    for job in p.jobs
        path=joinpath(directory,job.id*".json");receipt=JSON3.read(read(path,String))
        receipt.normal_final_waitidle===true&&receipt.conditions_sha256==digest(conditions)||error("Generation cleanup/provenance")
        attemptpath=joinpath(directory,job.id*"_attempt.json");attempt=JSON3.read(read(attemptpath,String))
        receipt.attempt_sha256==digest(attemptpath)||error("Attempt drift")
        attempt.pid isa Integer&&!(attempt.pid isa Bool)&&attempt.pid>0||error("Generation PID")
        push!(generation_pids,Int(attempt.pid))
    end
    length(unique(generation_pids))==16||error("Distinct generation workers required")
    rows=Any[]
    for row in filter(r->r.arm=="bayes_network"&&r.admitted,summary.rows)
        path=joinpath(directory,row.id*".json");digest(path)==row.generation_sha256||error("Generation drift")
        receipt=JSON3.read(read(path,String));admitted=admit_bayes_thinking(receipt,"bayes_network")
        admitted.accepted||error("Admission drift")
        network=String(bayes_extraction_payload(bayes_thinking_final(receipt).answer));packed=admitted.packed
        e=PortableEvidence.prepare_bayes(device,packed.factors,packed.query,packed.observed)
        expected=try
            PortableEvidence.run!(e);Mantle.waitidle(device)
            encode_bayes_snapshot(bayes_native_snapshot(e))
        finally close(e) end
        push!(rows,(;id=row.id,generation_sha256=row.generation_sha256,network,network_sha256=bytes2hex(sha256(network)),expected))
    end
    isempty(rows)&&error("No admitted networks; no physical parity job justified")
    packet=(;scope="Numerical parity for ALL admitted thinking-generated networks, including incorrect/unfaithful ones; no quality-score changes",
        conditions_sha256=digest(conditions),primary_summary_sha256=digest(summarypath),generation_pids,rows,
        source_sha256=Dict(f=>digest(joinpath(@__DIR__,f)) for f in BAYES_NATIVE_SOURCES),preparation_source_sha256=digest(@__FILE__))
    bytes=Vector{UInt8}(codeunits(JSON3.write(packet)));hash=bytes2hex(sha256(bytes))
    validate_bayes_native_payload(bytes,hash)
    mkpath(dirname(destination));open(io->write(io,bytes),destination,"w")
    (;complete=true,cases=length(rows),input_sha256=hash,destination=abspath(destination),privacy="ROOT-only packet; RAM transfer AFTER generation retirement; never write expected arrays on inference-worker filesystem")
end
