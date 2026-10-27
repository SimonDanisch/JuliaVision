# Fresh no-model grading process only. No private values are written to disk.
include("rpn_kernel_device_audit.jl")
include("test_rpn_kernel_device_audit.jl")
include("device_finish.jl")
include("runtime_provenance.jl")
using JSON3,SHA,Test
const WORKFLOW_NATIVE_SOURCES=("rpn_kernel_device_audit.jl","test_rpn_kernel_device_audit.jl",
    "run_workflow_native_audit.jl","rpn_pointwise_interface.jl","rpn_prefix_grammar.jl",
    "typed_pointwise_kernel_lowering.jl","device_finish.jl","runtime_provenance.jl")

function validate_workflow_native_payload(bytes,expected_sha256)
    bytes2hex(sha256(bytes))==expected_sha256||error("Private packet digest mismatch")
    p=JSON3.read(bytes)
    1<=length(p.rows)<=64&&length(unique(r.id for r in p.rows))==length(p.rows)&&length(p.controls)==4||error("Bounded cohort/control mismatch")
    Set(String.(keys(p.source_sha256)))==Set(WORKFLOW_NATIVE_SOURCES)||error("Grader source set mismatch")
    for (f,h) in pairs(p.source_sha256)
        basename(String(f))==String(f)&&bytes2hex(sha256(read(joinpath(@__DIR__,String(f)))))==h||error("Grader source drift")
    end
    p
end

function run_workflow_native_audit(bytes;expected_sha256,output)
    normpath(pwd())=="/tmp/small-llm-bosgame-20261025"&&Sys.total_memory()<=40*1024^3||error("Laptop snapshot only")
    (isdefined(Main,:model)||isdefined(Main,:qwenmodel))&&error("No model may be loaded")
    p=validate_workflow_native_payload(bytes,expected_sha256)
    getpid() in p.generation_pids&&error("Retire all generation workers first")
    ispath(output)&&error("Preserve native attempt")
    report=Dict{String,Any}("complete"=>false,"input_sha256"=>expected_sha256,"pid"=>getpid(),
        "conditions_sha256"=>p.conditions_sha256,"primary_summary_sha256"=>p.primary_summary_sha256,
        "scope"=>p.scope,"privacy"=>"Ephemeral private probes; fresh no-model grading context", "rows"=>Any[],"controls"=>Any[],
        "runtime_provenance"=>inference_runtime_provenance())
    save()=open(io->JSON3.pretty(io,report),output,"w")
    let
        let
            save();device=Mantle.Device("radeon")
            try
                tests=test_rpn_kernel_device_audit(device);counts=Test.get_test_counts(tests)
                report["grader_checks"]=counts.passes+counts.cumulative_passes;save()
                for row in p.controls
                    result=audit_rpn_kernel_device(device,String(row.program),row.cases)
                    push!(report["controls"],(;id=row.id,expected_pass=row.expected_pass,result));save()
                    result.admitted&&result.passed==row.expected_pass||error("Physical control failed")
                end
                for row in p.rows
                    admitted=RPNPointwiseInterface.parse_program(String(row.program)).accepted
                    admitted==row.admitted||error("Admission drift")
                    result=admitted ? audit_rpn_kernel_device(device,String(row.program),row.cases) : (;admitted=false,passed=false)
                    push!(report["rows"],(;id=row.id,arm=row.arm,generation_sha256=row.generation_sha256,primary_status=row.primary_status,result));save()
                end
            finally
                finish_device!(device)
            end
            report["complete"]=true;report["normal_final_waitidle"]=true;save();report
        end
    end
end
