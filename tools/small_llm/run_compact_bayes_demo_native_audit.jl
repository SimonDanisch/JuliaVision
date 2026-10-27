# Fresh no-model process. Private expectations are transferred in RAM only.
include("run_bayes_native_audit.jl")
const COMPACT_DEMO_NATIVE_SOURCES=(BAYES_NATIVE_SOURCES...,"run_compact_bayes_demo_native_audit.jl")

function validate_compact_demo_native_payload(bytes,expected_sha256)
    length(bytes)<=262144&&bytes2hex(sha256(bytes))==expected_sha256||error("Private packet digest/size")
    p=JSON3.read(copy(bytes))
    1<=length(p.rows)<=8&&length(unique(r.id for r in p.rows))==length(p.rows)||error("Unique bounded cohort")
    length(p.generation_pids)==8&&length(unique(p.generation_pids))==8&&
        all(x->x isa Integer&&!(x isa Bool)&&x>0,p.generation_pids)||error("All8 actual attempt PIDs required")
    Set(String.(keys(p.source_sha256)))==Set(COMPACT_DEMO_NATIVE_SOURCES)||error("Exact public source set")
    for (f,h) in pairs(p.source_sha256)
        basename(String(f))==String(f)&&bytes2hex(sha256(read(joinpath(@__DIR__,String(f)))))==h||error("Source drift")
    end
    for row in p.rows
        bytes2hex(sha256(String(row.network)))==row.network_sha256||error("Generated network drift")
        packed=BayesJSON.compile(String(row.network);max_states=256)
        all(length(getproperty(row.expected,k))==length(packed.query_states) for k in (:prior,:posterior,:log_likelihood,:defined))||error("Snapshot dimensions")
        row.expected.status isa Integer&&!(row.expected.status isa Bool)&&row.expected.status in (0,1,2)||error("Solver status")
        all(x->x isa Integer&&!(x isa Bool)&&x in (0,1),row.expected.defined)||error("Defined flags")
        for key in (:prior,:posterior,:log_likelihood);bayes_parity_float.(getproperty(row.expected,key));end
        bayes_parity_float(row.expected.log_evidence)
    end
    p
end

function run_compact_demo_native_audit(bytes;expected_sha256,output)
    normpath(pwd())=="/tmp/small-llm-bosgame-20261025"&&Sys.total_memory()<=40*1024^3||error("Laptop snapshot only")
    any(s->isdefined(Main,s),(:model,:qwenmodel,:sampled_julia_answer,:run_julia_stencil_transfer))&&error("Fresh NO-MODEL worker required")
    p=validate_compact_demo_native_payload(bytes,expected_sha256)
    any(pid->isdir("/proc/"*string(pid)),p.generation_pids)&&error("Generation process still present or PID reused")
    ispath(output)&&error("Preserve physical first attempt");mkpath(dirname(output))
    report=Dict{String,Any}("complete"=>false,"pid"=>getpid(),"input_sha256"=>expected_sha256,
        "conditions_sha256"=>p.conditions_sha256,"primary_summary_sha256"=>p.primary_summary_sha256,
        "scope"=>"Numerical parity/replay of ALL admitted compact demonstration-generated networks, including incorrect ones. Infrastructure only, no quality-score change.",
        "privacy"=>"No-model worker; expectations only in RAM after all8 attempts retired, including lost process. No private arrays written on inference worker.",
        "runtime_provenance"=>inference_runtime_provenance(),"rows"=>Any[])
    save()=open(io->JSON3.pretty(io,report),output,"w")
    let
        let
            save();device=Mantle.Device("radeon")
            try
                tests=test_bayes_native_parity(device);counts=Test.get_test_counts(tests)
                report["harness_checks"]=counts.passes+counts.cumulative_passes;save()
                for row in p.rows
                    result=audit_bayes_native_case(device,String(row.network),row.expected)
                    push!(report["rows"],(;id=row.id,generation_sha256=row.generation_sha256,network_sha256=row.network_sha256,result));save()
                    result.passed||error("Physical generated-network parity failed")
                end
            finally finish_device!(device) end
            report["complete"]=true;report["normal_final_waitidle"]=true;save();nothing
        end
    end
end
