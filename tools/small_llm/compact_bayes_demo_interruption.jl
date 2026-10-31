using JSON3,SHA

"""Only an explicit process-loss record can close a missing answer. Never retry it."""
function compact_bayes_demo_lost(directory,job,conditions_sha256)
    path=joinpath(directory,String(job.id)*"_lost.json")
    isfile(path)||return false
    isfile(joinpath(directory,String(job.id)*".json"))&&error("Both output and loss record")
    m=JSON3.read(read(path,String));attempt=joinpath(directory,String(job.id)*"_attempt.json")
    isfile(attempt)||error("Missing original attempt")
    a=JSON3.read(read(attempt,String))
    m.id==a.id==job.id&&m.pid==a.pid&&m.pid>0||error("Loss identity mismatch")
    m.conditions_sha256==a.conditions_sha256==conditions_sha256||error("Loss registration mismatch")
    m.attempt_sha256==bytes2hex(sha256(read(attempt)))||error("Lost attempt drift")
    m.terminal_status=="lost_process"&&m.original_process_absent===true&&m.shared_lock_observed_free===true||error("Unverified process loss")
    m.normal_final_waitidle===false&&m.regeneration_permitted===false||error("Must not claim normal completion/retry")
    true
end

function compact_bayes_demo_pending(directory,p,conditions_sha256)
    ids=String[String(j.id) for j in p.jobs]
    length(ids)==8&&length(unique(ids))==8||error("ALL8 unique original slots required")
    pending=filter(p.jobs) do job
        output=joinpath(directory,String(job.id)*".json")
        lost=compact_bayes_demo_lost(directory,job,conditions_sha256)
        !isfile(output)&&!lost
    end
    any(isfile(joinpath(directory,String(j.id)*"_attempt.json")) for j in pending)&&error("Unresolved attempt; no regeneration")
    pending
end
