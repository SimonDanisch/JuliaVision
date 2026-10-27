isdefined(@__MODULE__,:CheckedPointwiseLowering) || include("checked_pointwise_lowering.jl")
using JSON3, SHA

function audit_checked_pointwise_study()
    directory=joinpath(@__DIR__,"pointwise_lowering_results")
    conditions=JSON3.read(read(joinpath(directory,"conditions.json"),String))
    qualification_path=joinpath(@__DIR__,"pointwise_lowering_validation.json")
    bytes2hex(sha256(read(qualification_path)))==conditions.qualification_sha256 || error("qualification drift")
    qualification=JSON3.read(read(qualification_path,String))
    for (file,hash) in pairs(qualification.source_sha256)
        bytes2hex(sha256(read(joinpath(@__DIR__,String(file)))))==hash || error("qualified runtime/compiler drift")
    end
    for (file,hash) in pairs(conditions.source_sha256)
        bytes2hex(sha256(read(joinpath(@__DIR__,String(file)))))==hash || error("V1 source drift")
    end
    rows=Any[]
    for (id,hash) in pairs(conditions.baseline_outputs_sha256)
        source=joinpath(@__DIR__,"vulkan_kernel_results",String(id)*"_baseline.json")
        bytes2hex(sha256(read(source)))==hash || error("baseline drift")
        candidate=JSON3.read(read(source,String));checked=CheckedPointwiseLowering.lower(String(candidate.code))
        receiptpath=joinpath(directory,String(id)*"_score.json")
        receipt=JSON3.read(read(receiptpath,String))
        receipt.id==String(id) || error("receipt task identity mismatch")
        chosen=checked.accepted ? checked.code : String(candidate.code)
        receipt.code_sha256==bytes2hex(sha256(chosen)) || error("checked selection differs; needs NEW execution, not receipt reuse")
        push!(rows,(;task=String(id),accepted=checked.accepted,reason=checked.reason,certificate=checked.certificate,
            code_sha256=receipt.code_sha256,receipt_sha256=bytes2hex(sha256(read(receiptpath))),
            status=receipt.passed ? "passed" : "failed"))
    end
    report=(;scope="stricter admission audit of the SAME post-hoc four-candidate compiler experiment, not another independent score",
        rule="reuse an existing software-Vulkan receipt ONLY when the checked selected code is byte-identical",
        correction="V1's generic admission allowed floating loop limits and shadowed names; these are rejected here. Frozen historical artifacts are not rewritten.",
        limitation="restricted transformation with explicit disjointness/in-bounds/ordinary-arithmetic/1D-launch preconditions; not an arbitrary Julia compiler or Radeon performance result",
        rows,passed=count(r->r.status=="passed",rows),
        baseline_passed=JSON3.read(read(joinpath(directory,"summary.json"),String)).baseline_passed,pending=0,
        source_sha256=Dict(file=>bytes2hex(sha256(read(joinpath(@__DIR__,file)))) for file in
            ("checked_pointwise_lowering.jl","pointwise_kernel_lowering.jl","audit_checked_pointwise_study.jl","test_checked_pointwise_lowering.jl")))
    open(io->JSON3.pretty(io,report),joinpath(@__DIR__,"checked_pointwise_study.json"),"w");report
end
