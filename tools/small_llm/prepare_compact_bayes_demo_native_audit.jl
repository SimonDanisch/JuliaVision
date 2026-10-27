include("run_compact_bayes_demo_native_audit.jl")
include("compact_bayes_admission.jl")
include("compact_bayes_demo_interruption.jl")

function prepare_compact_bayes_demo_native_audit(device;directory=joinpath(@__DIR__,"compact_bayes_demo_results"),
        destination=joinpath(@__DIR__,"compact_bayes_demo_native_audit/private_packet.json"))
    occursin("llvmpipe",lowercase(sprint(show,device)))||error("Software reference required")
    ispath(destination)&&error("Preserve private packet")
    digest(path)=bytes2hex(sha256(read(path)))
    path=joinpath(directory,"conditions.json");p=JSON3.read(read(path,String));hash=digest(path)
    scorepath=joinpath(directory,"summary.json");s=JSON3.read(read(scorepath,String))
    s.complete&&s.all_slots_terminal&&length(s.rows)==length(p.jobs)==8&&s.conditions_sha256==hash||error("Finish ALL8 slots and score first")
    isempty(compact_bayes_demo_pending(directory,p,hash))||error("Unfinished generation")
    for f in ("compact_bayes_admission.jl","compact_bayes_json.jl","bayes_json.jl","portable_evidence.jl")
        p.private_source_sha256[Symbol(f)]==digest(joinpath(@__DIR__,f))||error("Parser/solver drift")
    end
    pids=Int[]
    for job in p.jobs
        attemptpath=joinpath(directory,job.id*"_attempt.json");a=JSON3.read(read(attemptpath,String))
        a.id==job.id&&a.conditions_sha256==hash&&a.pid isa Integer&&!(a.pid isa Bool)&&a.pid>0||error("Attempt identity/PID")
        push!(pids,Int(a.pid))
        compact_bayes_demo_lost(directory,job,hash)&&continue
        m=JSON3.read(read(joinpath(directory,job.id*".json"),String))
        m.normal_final_waitidle&&m.conditions_sha256==hash&&m.attempt_sha256==digest(attemptpath)||error("Generation cleanup/provenance")
    end
    length(unique(pids))==8||error("Distinct actual generation workers required")
    rows=Any[]
    for row in filter(r->r.admitted,s.rows)
        output=joinpath(directory,row.id*".json");digest(output)==row.generation_sha256||error("Generation drift")
        m=JSON3.read(read(output,String));a=admit_compact_bayes(String(m.answer));a.accepted||error("Admission drift")
        network=JSON3.write(a.network);packed=a.packed
        e=PortableEvidence.prepare_bayes(device,packed.factors,packed.query,packed.observed)
        expected=try
            PortableEvidence.run!(e);Mantle.waitidle(device);encode_bayes_snapshot(bayes_native_snapshot(e))
        finally close(e) end
        push!(rows,(;id=row.id,generation_sha256=row.generation_sha256,network,network_sha256=bytes2hex(sha256(network)),expected))
    end
    isempty(rows)&&error("No admitted networks; no physical job justified")
    packet=(;conditions_sha256=hash,primary_summary_sha256=digest(scorepath),generation_pids=pids,rows,
        scope="Software expectations computed from SAME generated networks, including wrong ones; no hidden-answer correction",
        source_sha256=Dict(f=>digest(joinpath(@__DIR__,f)) for f in COMPACT_DEMO_NATIVE_SOURCES),preparation_source_sha256=digest(@__FILE__))
    bytes=Vector{UInt8}(codeunits(JSON3.write(packet)));hash=bytes2hex(sha256(bytes))
    validate_compact_demo_native_payload(bytes,hash)
    mkpath(dirname(destination));open(io->write(io,bytes),destination,"w")
    (;complete=true,cases=length(rows),input_sha256=hash,destination=abspath(destination),privacy="ROOT-only file; RAM transfer to fresh no-model Laptop worker only")
end
