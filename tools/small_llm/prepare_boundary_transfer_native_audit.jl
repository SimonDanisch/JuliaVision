# ROOT ONLY. Construct private values AFTER all generation and selection.
include("score_boundary_transfer.jl")
include("run_workflow_native_audit.jl")
function prepare_boundary_transfer_native_audit(;directory=joinpath(@__DIR__,"boundary_transfer_native_audit"))
    ispath(directory)&&!isempty(readdir(directory))&&error("Preserve native preparation")
    study=joinpath(@__DIR__,"boundary_transfer_results");state=julia_stencil_transfer_state(study);p=state.packet
    summarypath=joinpath(study,"summary.json");summary=JSON3.read(read(summarypath,String))
    summary.complete&&summary.total==8&&summary.conditions_sha256==julia_transfer_digest(state.path)||error("Primary study incomplete")
    rows=Any[];pids=Int[]
    for (job,(_,formula)) in zip(p.jobs,private_boundary_transfer_fixtures())
        selectionpath=joinpath(study,String(job.id)*"_selection.json");s=JSON3.read(read(selectionpath,String))
        primary=only(filter(r->r.id==job.id,summary.rows))
        primary.selection_sha256==julia_transfer_digest(selectionpath)||error("Selection drift")
        attempt=JSON3.read(read(joinpath(study,String(job.id)*"_attempt.json"),String));push!(pids,Int(attempt.pid))
        for (arm,program) in (("direct",String(s.direct.program)),("example",String(s.choices.example.program)),("boundary",String(s.choices.boundary.program)))
            push!(rows,(;id=String(job.id)*"_"*arm,task=String(job.task),arm,program,
                admitted=RPNPointwiseInterface.parse_program(program).accepted,cases=private_boundary_transfer_cases(formula),
                primary_status=getproperty(primary.outcomes,Symbol(arm)).passed ? "passed" : "failed",generation_sha256=julia_transfer_digest(selectionpath)))
        end
    end
    controls=Any[]
    for (id,program,want,values) in (("constant_reference","half\nfinish",true,fill(0.5f0,3)),
            ("constant_mutant","one\nfinish",false,fill(0.5f0,3)),
            ("neg_reference","x0\nneg\nfinish",true,Float32[2,-0.5,-3]),
            ("neg_mutant","x0\nfinish",false,Float32[2,-0.5,-3]))
        initial=fill(-983f0,10);expected=copy(initial);expected[1:3]=values
        push!(controls,(;id,program,expected_pass=want,cases=[(;n=3,x=Float32[-2,0.5,3],initial,expected)]))
    end
    length(rows)==24||error("Three-arm native cohort")
    mkpath(directory)
    packet=(;scope="Physical-device parity of SAME8 three-arm transfer workflow outcomes, not new neural attempts/independent tasks or a pooled score",rows,controls,generation_pids=unique(pids),
        conditions_sha256=julia_transfer_digest(state.path),primary_summary_sha256=julia_transfer_digest(summarypath),
        source_sha256=Dict(f=>julia_transfer_digest(joinpath(@__DIR__,f)) for f in WORKFLOW_NATIVE_SOURCES))
    bytes=Vector{UInt8}(codeunits(JSON3.write(packet)))
    open(io->write(io,bytes),joinpath(directory,"input_PRIVATE.json"),"w")
    (;sha256=bytes2hex(sha256(bytes)),bytes=length(bytes),rows=24,controls=4)
end
