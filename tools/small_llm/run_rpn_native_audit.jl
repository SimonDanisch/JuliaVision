include("rpn_kernel_device_audit.jl")
include("test_rpn_kernel_device_audit.jl")
include("device_finish.jl")
include("runtime_provenance.jl")
using JSON3,Test

const RPN_NATIVE_AUDIT_SOURCES=("rpn_kernel_device_audit.jl","test_rpn_kernel_device_audit.jl","run_rpn_native_audit.jl",
    "rpn_pointwise_interface.jl","rpn_prefix_grammar.jl","typed_pointwise_kernel_lowering.jl",
    "device_finish.jl","runtime_provenance.jl")

function validate_rpn_native_audit_payload(bytes,expected_sha256)
    bytes2hex(sha256(bytes))==expected_sha256||error("private audit input digest mismatch")
    p=JSON3.read(bytes)
    length(p.rows)==16&&length(unique(r.id for r in p.rows))==16&&length(p.controls)==4||error("audit cohort/control mismatch")
    Set(String.(keys(p.source_sha256)))==Set(RPN_NATIVE_AUDIT_SOURCES)||error("audit source set mismatch")
    for (file,hash) in pairs(p.source_sha256)
        basename(String(file))==String(file)&&bytes2hex(open(sha256,joinpath(@__DIR__,String(file))))==hash||error("audit source drift: $file")
    end
    p
end

"""Fresh GRADING-only Laptop process; private values stay in memory, never a prompt."""
function run_rpn_native_audit(bytes;expected_sha256,
        output=joinpath(@__DIR__,"rpn_native_audit_result.json"))
    normpath(pwd())=="/tmp/small-llm-bosgame-20261025"&&Sys.total_memory()<=40*1024^3||error("Laptop snapshot grading only; no device setup")
    (isdefined(Main,:model)||isdefined(Main,:qwenmodel))&&error("fresh grading-only process required")
    p=validate_rpn_native_audit_payload(bytes,expected_sha256)
    getpid() in p.generation_pids&&error("retire ALL generation workers before private audit")
    ispath(output)&&error("preserve native audit attempt")
    report=Dict{String,Any}("complete"=>false,"input_sha256"=>expected_sha256,"pid"=>getpid(),
        "conditions_sha256"=>p.conditions_sha256,"primary_summary_sha256"=>p.primary_summary_sha256,
        "scope"=>p.scope,"privacy"=>"No model loaded; private inputs/expectations are ephemeral grading arguments only",
        "runtime_provenance"=>inference_runtime_provenance(),"controls"=>Any[],"rows"=>Any[])
    save()=open(io->JSON3.pretty(io,report),output,"w")
    let
        save();device=Mantle.Device("radeon")
        try
            tests=test_rpn_kernel_device_audit(device);c=Test.get_test_counts(tests)
            report["grader_checks"]=c.passes+c.cumulative_passes;save()
            for control in p.controls
                result=audit_rpn_kernel_device(device,String(control.program),control.cases)
                push!(report["controls"],(;id=control.id,expected_pass=control.expected_pass,result));save()
                result.admitted&&result.passed==control.expected_pass||error("native reference/mutant control failed")
            end
            for row in p.rows
                admitted=RPNPointwiseInterface.parse_program(String(row.program)).accepted
                admitted==row.admitted||error("syntax admission drift")
                result=admitted ? audit_rpn_kernel_device(device,String(row.program),row.cases) :
                    (;admitted=false,passed=false,rows=Any[])
                push!(report["rows"],(;id=row.id,arm=row.arm,generation_sha256=row.generation_sha256,
                    primary_status=row.primary_status,result));save()
            end
        finally
            finish_device!(device)
        end
        report["normal_final_waitidle"]=true;report["complete"]=true;save();report
    end
end
