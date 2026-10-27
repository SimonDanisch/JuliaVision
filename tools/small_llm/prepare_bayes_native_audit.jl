include("run_bayes_native_audit.jl")
include("bayes_extraction_admission.jl")

"""ROOT-only post-generation packing. Expectations come from SOFTWARE execution
of the SAME generated networks, NOT from problem answers or reference networks.
Never copy this private packet to the Laptop filesystem; transfer bytes in RAM.
"""
function prepare_bayes_native_audit(device;directory=joinpath(@__DIR__,"bayes_extraction_results"),destination=joinpath(@__DIR__,"bayes_extraction_native_audit/private_packet.json"))
    occursin("llvmpipe",lowercase(sprint(show,device)))||error("SOFTWARE reference device required")
    isfile(destination)&&error("Preserve private native packet")
    digest(path)=bytes2hex(sha256(read(path)))
    conditions=joinpath(directory,"conditions.json");summarypath=joinpath(directory,"summary.json")
    p=JSON3.read(read(conditions,String));summary=JSON3.read(read(summarypath,String))
    summary.complete&&length(summary.rows)==16&&length(p.jobs)==16||error("Finish ALL16 and retire generation workers first")
    summary.conditions_sha256==digest(conditions)||error("Score provenance")
    for file in ("portable_evidence.jl","bayes_json.jl")
        p.private_sources[Symbol(file)]==digest(joinpath(@__DIR__,file))||error("Shared solver/parser changed since primary grading")
    end
    generation_pids=Int[]
    for job in p.jobs
        receipt=JSON3.read(read(joinpath(directory,job.id*".json"),String))
        receipt.normal_final_waitidle&&receipt.conditions_sha256==digest(conditions)||error("Generation cleanup/provenance")
        attempt=JSON3.read(read(joinpath(directory,job.id*"_attempt.json"),String))
        attempt.pid isa Integer&&!(attempt.pid isa Bool)&&attempt.pid>0||error("Generation PID required")
        push!(generation_pids,Int(attempt.pid))
    end
    rows=Any[]
    for row in filter(r->r.arm=="bayes_network"&&r.admitted,summary.rows)
        path=joinpath(directory,row.id*".json");digest(path)==row.generation_sha256||error("Generation drift")
        receipt=JSON3.read(read(path,String));admitted=admit_bayes_extraction(String(receipt.answer),"bayes_network")
        admitted.accepted||error("Admission drift")
        network=bayes_extraction_payload(String(receipt.answer));packed=admitted.packed
        execution=PortableEvidence.prepare_bayes(device,packed.factors,packed.query,packed.observed)
        expected=try
            PortableEvidence.run!(execution);Mantle.waitidle(device)
            encode_bayes_snapshot(bayes_native_snapshot(execution))
        finally
            close(execution)
        end
        push!(rows,(;id=row.id,generation_sha256=row.generation_sha256,network,network_sha256=bytes2hex(sha256(network)),expected))
    end
    isempty(rows)&&error("No admitted networks: no physical parity job is justified")
    packet=(;scope="Parity of ALL admitted SAME model networks, including unfaithful/wrong ones; no model intelligence score changes",
        conditions_sha256=digest(conditions),primary_summary_sha256=digest(summarypath),generation_pids=unique(generation_pids),rows,
        source_sha256=Dict(f=>digest(joinpath(@__DIR__,f)) for f in BAYES_NATIVE_SOURCES),preparation_source_sha256=digest(@__FILE__))
    bytes=Vector{UInt8}(codeunits(JSON3.write(packet)));hash=bytes2hex(sha256(bytes))
    validate_bayes_native_payload(bytes,hash)
    mkpath(dirname(destination));open(io->write(io,bytes),destination,"w")
    (;complete=true,cases=length(rows),input_sha256=hash,destination=abspath(destination),
        privacy="ROOT-only packet. RAM transfer to fresh no-model Laptop worker only AFTER all generation retired; retire audit worker before any further generation.")
end
