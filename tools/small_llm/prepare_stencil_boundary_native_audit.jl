# ROOT-only physical parity preparation for already-selected posthoc workflow.
include("run_rpn_native_audit.jl")
using SHA,JSON3
function prepare_stencil_boundary_native_audit(;directory=joinpath(@__DIR__,"stencil_boundary_native_audit"))
    ispath(directory)&&!isempty(readdir(directory))&&error("Preserve native audit")
    digest(path)=bytes2hex(sha256(read(path)))
    parentdir=joinpath(@__DIR__,"julia_stencil_native_audit")
    parent=JSON3.read(read(joinpath(parentdir,"input_PRIVATE.json"),String))
    receipt=JSON3.read(read(joinpath(parentdir,"result.json"),String))
    receipt.complete&&receipt.normal_final_waitidle&&receipt.input_sha256==digest(joinpath(parentdir,"input_PRIVATE.json"))||error("Prior physical parity missing")
    all(r->r.result.passed==(r.primary_status=="passed"),receipt.rows)||error("Prior physical disagreement")
    study=joinpath(@__DIR__,"stencil_boundary_development_results")
    summarypath=joinpath(study,"summary.json");summary=JSON3.read(read(summarypath,String))
    summary.complete&&summary.total==8||error("Complete posthoc selection/scoring first")
    summary.conditions_sha256==digest(joinpath(study,"conditions.json"))||error("Posthoc conditions drift")
    rows=Any[]
    for row in parent.rows
        if row.arm=="direct"
            push!(rows,row)
        else
            path=joinpath(study,String(row.task)*".json");s=JSON3.read(read(path,String))
            score=only(filter(r->r.id==row.task,summary.rows))
            score.selection_sha256==digest(path)||error("Boundary selection drift")
            program=String(s.selected.program)
            push!(rows,(;id=String(row.task)*"_boundary",task=row.task,arm="boundary",program,
                admitted=RPNPointwiseInterface.parse_program(program).accepted,cases=row.cases,
                primary_status=score.status,generation_sha256=digest(path)))
        end
    end
    length(rows)==16&&length(unique(r.id for r in rows))==16||error("Paired audit cohort")
    mkpath(directory)
    joint=(;scope="Physical parity of SAME8 baseline/posthoc boundary workflow pairs. NOT independent transfer, new model attempts or a pooled intelligence score",rows=[(;r.id,r.primary_status,r.generation_sha256) for r in rows],
        posthoc_summary_sha256=digest(summarypath),prior_native_result_sha256=digest(joinpath(parentdir,"result.json")))
    jointpath=joinpath(directory,"primary_summary.json");open(io->JSON3.pretty(io,joint),jointpath,"w")
    packet=(;scope=joint.scope,conditions_sha256=digest(joinpath(study,"conditions.json")),primary_summary_sha256=digest(jointpath),
        generation_pids=parent.generation_pids,rows,controls=parent.controls,
        source_sha256=Dict(f=>digest(joinpath(@__DIR__,f)) for f in RPN_NATIVE_AUDIT_SOURCES),
        preparation_source_sha256=digest(@__FILE__))
    bytes=Vector{UInt8}(codeunits(JSON3.write(packet)))
    open(io->write(io,bytes),joinpath(directory,"input_PRIVATE.json"),"w")
    (;sha256=bytes2hex(sha256(bytes)),rows=16,controls=4)
end
