# ROOT ONLY. Private probes are ephemeral grading arguments, never model input.
include("score_julia_stencil_transfer.jl")
include("run_rpn_native_audit.jl")

function prepare_julia_stencil_native_audit(;directory=joinpath(@__DIR__,"julia_stencil_native_audit"))
    ispath(directory)&&!isempty(readdir(directory))&&error("Preserve audit preparation")
    study=joinpath(@__DIR__,"julia_stencil_transfer_results")
    summarypath=joinpath(study,"summary.json");summary=JSON3.read(read(summarypath,String))
    summary.complete&&summary.total==8||error("Complete ALL generation/selection/primary scoring first")
    state=julia_stencil_transfer_state(study);p=state.packet
    summary.conditions_sha256==julia_transfer_digest(state.path)||error("Primary provenance drift")
    rows=Any[];pids=Int[]
    for (job,(_,formula)) in zip(p.jobs,private_julia_stencil_transfer_fixtures())
        selectionpath=joinpath(study,String(job.id)*"_selection.json");s=JSON3.read(read(selectionpath,String))
        primary=only(filter(r->r.id==job.id,summary.rows))
        primary.selection_sha256==julia_transfer_digest(selectionpath)||error("Selection drift")
        attempt=JSON3.read(read(joinpath(study,String(job.id)*"_attempt.json"),String));push!(pids,Int(attempt.pid))
        for (arm,program,source) in (("direct",String(s.direct.program),joinpath(study,String(job.id)*".json")),
                ("hybrid",String(s.selected.program),selectionpath))
            admitted=RPNPointwiseInterface.parse_program(program).accepted
            push!(rows,(;id=String(job.id)*"_"*arm,task=String(job.task),arm,program,admitted,
                cases=private_julia_stencil_transfer_cases(formula),
                primary_status=getproperty(primary.outcomes,Symbol(arm)).passed ? "passed" : "failed",
                generation_sha256=julia_transfer_digest(source)))
        end
    end
    controls=Any[]
    for (id,program,want,values) in (
            ("constant_reference","two\none\nsub\nhalf\nmul\nfinish",true,fill(0.5f0,3)),
            ("constant_mutant","two\none\nadd\nhalf\nmul\nfinish",false,fill(0.5f0,3)),
            ("neg_reference","x0\nneg\nfinish",true,Float32[2,-0.5,-3]),
            ("neg_mutant","x0\nfinish",false,Float32[2,-0.5,-3]))
        initial=fill(-971f0,10);expected=copy(initial);expected[1:3]=values
        push!(controls,(;id,program,expected_pass=want,cases=[(;n=3,x=Float32[-2,0.5,3],initial,expected)]))
    end
    length(rows)==16&&length(unique(r.id for r in rows))==16||error("Joint audit cohort")
    mkpath(directory)
    joint=(;scope="Physical-device parity of SAME8 direct/hybrid workflow pairs; overlapping outputs are not independent intelligence evidence",rows=[(;r.id,r.primary_status,r.generation_sha256) for r in rows],
        primary_summary_sha256=julia_transfer_digest(summarypath))
    jointpath=joinpath(directory,"primary_summary.json");open(io->JSON3.pretty(io,joint),jointpath,"w")
    packet=(;scope=joint.scope,conditions_sha256=julia_transfer_digest(state.path),primary_summary_sha256=julia_transfer_digest(jointpath),
        generation_pids=unique(pids),rows,controls,
        source_sha256=Dict(f=>julia_transfer_digest(joinpath(@__DIR__,f)) for f in RPN_NATIVE_AUDIT_SOURCES))
    bytes=Vector{UInt8}(codeunits(JSON3.write(packet)))
    open(io->write(io,bytes),joinpath(directory,"input_PRIVATE.json"),"w")
    (;sha256=bytes2hex(sha256(bytes)),bytes=length(bytes),rows=length(rows),controls=length(controls))
end
