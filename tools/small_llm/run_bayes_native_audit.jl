# Fresh NO-MODEL grading process; private reference values are MEMORY ONLY.
include("device_finish.jl")
include("test_bayes_native_parity.jl")
include("runtime_provenance.jl")
using JSON3,SHA,Test
const BAYES_NATIVE_SOURCES=("run_bayes_native_audit.jl","bayes_native_parity.jl","test_bayes_native_parity.jl",
    "bayes_json.jl","portable_evidence.jl","device_finish.jl","runtime_provenance.jl")

function validate_bayes_native_payload(bytes,expected_sha256)
    length(bytes)<=262144&&bytes2hex(sha256(bytes))==expected_sha256||error("Bounded private packet digest mismatch")
    p=JSON3.read(bytes)
    1<=length(p.rows)<=8&&length(unique(r.id for r in p.rows))==length(p.rows)||error("Bounded unique cohort required")
    length(p.generation_pids)==16&&length(unique(p.generation_pids))==16&&
        all(x->x isa Integer&&!(x isa Bool)&&x>0,p.generation_pids)||error("All16 distinct generation worker PIDs required")
    Set(String.(keys(p.source_sha256)))==Set(BAYES_NATIVE_SOURCES)||error("Grader source set mismatch")
    for (file,hash) in pairs(p.source_sha256)
        basename(String(file))==String(file)&&bytes2hex(sha256(read(joinpath(@__DIR__,String(file)))))==hash||error("Grader source drift")
    end
    for row in p.rows
        bytes2hex(sha256(String(row.network)))==row.network_sha256||error("Generated network drift")
        packed=BayesJSON.compile(String(row.network);max_states=256)
        all(length(getproperty(row.expected,k))==length(packed.query_states) for k in (:prior,:posterior,:log_likelihood,:defined))||error("Expected output dimensions")
        row.expected.status isa Integer&&!(row.expected.status isa Bool)&&row.expected.status in (0,1,2)||error("Expected solver status")
        all(x->x isa Integer&&!(x isa Bool)&&x in (0,1),row.expected.defined)||error("Expected defined flags")
        for key in (:prior,:posterior,:log_likelihood);bayes_parity_float.(getproperty(row.expected,key));end
        bayes_parity_float(row.expected.log_evidence)
    end
    p
end

function run_bayes_native_audit(bytes;expected_sha256,output)
    normpath(pwd())=="/tmp/small-llm-bosgame-20261025"&&Sys.total_memory()<=40*1024^3||error("Laptop snapshot only")
    any(s->isdefined(Main,s),(:model,:qwenmodel,:sampled_julia_answer,:run_julia_stencil_transfer))&&error("Fresh NO-MODEL worker required")
    p=validate_bayes_native_payload(bytes,expected_sha256)
    getpid() in p.generation_pids&&error("Retire ALL generation workers first")
    ispath(output)&&error("Preserve native first attempt")
    report=Dict{String,Any}("complete"=>false,"pid"=>getpid(),"input_sha256"=>expected_sha256,
        "conditions_sha256"=>p.conditions_sha256,"primary_summary_sha256"=>p.primary_summary_sha256,
        "scope"=>"Numerical parity/replay for SAME model-generated networks on physical Laptop GPU; NOT new neural answers, independent intelligence or quality-score changes",
        "privacy"=>"Reference arrays transferred MEMORY ONLY after all generation retired; never written to Laptop disk. This no-model worker must retire before another generation.",
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
                    result.passed||error("Physical shared-kernel parity failed: $(row.id)")
                end
            finally
                finish_device!(device)
            end
            report["complete"]=true;report["normal_final_waitidle"]=true;save();report
        end
    end
end
