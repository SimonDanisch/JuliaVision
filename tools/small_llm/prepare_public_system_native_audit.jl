# ROOT ONLY: ephemeral private probes after ALL model generation has finished.
include("score_public_example_hybrid.jl")
include("run_rpn_native_audit.jl")

function prepare_public_system_native_audit(;directory=joinpath(@__DIR__,"public_system_native_audit"))
    ispath(directory)&&!isempty(readdir(directory))&&error("Preserve native audit preparation")
    modeldir=joinpath(@__DIR__,"rpn_public_example_results")
    symbolicdir=joinpath(@__DIR__,"public_example_synthesis_results")
    hybriddir=joinpath(@__DIR__,"public_example_hybrid_results")
    summaries=[JSON3.read(read(joinpath(d,"summary.json"),String)) for d in (modeldir,symbolicdir,hybriddir)]
    all(s->s.complete,summaries)||error("Complete ALL primary outputs/grading first")
    model=JSON3.read(read(joinpath(modeldir,"conditions.json"),String));tasks=rpn_public_example_tasks();fixtures=private_public_example_fixtures()
    rows=Any[];pids=Int[]
    function add(id,task,arm,program,source,status)
        index=only(findall(t->t.id==task,tasks));cases=private_public_example_cases(fixtures[index].formula)
        admitted=RPNPointwiseInterface.parse_program(program).accepted
        push!(rows,(;id,task,arm,program,admitted,cases,primary_status=status,generation_sha256=example_study_digest(source)))
    end
    for job in model.jobs
        source=joinpath(modeldir,String(job.id)*".json");c=JSON3.read(read(source,String))
        hasproperty(c,:inference_error)&&error("Resolve inference errors before native comparison")
        score=only(filter(r->r.id==job.id,summaries[1].rows))
        add("model_"*String(job.id),String(job.task),String(job.arm),String(c.answer),source,score.status)
        attempt=JSON3.read(read(replace(source,".json"=>"_attempt.json"),String));push!(pids,Int(attempt.pid))
    end
    for task in tasks
        source=joinpath(symbolicdir,task.id*".json");c=JSON3.read(read(source,String));c.result.found||error("Audit packet requires explicit symbolic program")
        score=only(filter(r->r.task==task.id,summaries[2].rows))
        add("symbolic_"*task.id,task.id,"symbolic",String(c.result.program),source,score.status)
        source=joinpath(hybriddir,task.id*".json");c=JSON3.read(read(source,String))
        score=only(filter(r->r.task==task.id,summaries[3].rows))
        add("hybrid_"*task.id,task.id,"hybrid",String(c.selected.program),source,score.status)
    end
    length(rows)==16&&length(unique(r.id for r in rows))==16||error("Audit cohort")
    controls=Any[]
    for (id,program,want,expected_values) in (
            ("constant_reference","two\none\nsub\nhalf\nmul\nfinish",true,fill(0.5f0,3)),
            ("constant_mutant","two\none\nadd\nhalf\nmul\nfinish",false,fill(0.5f0,3)),
            ("neg_reference","x0\nneg\nfinish",true,Float32[2,-0.5,-3]),
            ("neg_mutant","x0\nfinish",false,Float32[2,-0.5,-3]))
        initial=fill(-913f0,10);expected=copy(initial);expected[1:3]=expected_values
        push!(controls,(;id,program,expected_pass=want,cases=[(;n=3,x=Float32[-2,0.5,3],initial,expected)]))
    end
    mkpath(directory)
    joint=(;scope="Native parity only for16 already scored neural/symbolic/hybrid outputs; overlapping programs are NOT pooled intelligence evidence",rows=[(;id=r.id,primary_status=r.primary_status,generation_sha256=r.generation_sha256) for r in rows],
        primary_summaries_sha256=[example_study_digest(joinpath(d,"summary.json")) for d in (modeldir,symbolicdir,hybriddir)])
    jointpath=joinpath(directory,"primary_summary.json");open(io->JSON3.pretty(io,joint),jointpath,"w")
    packet=(;scope=joint.scope,conditions_sha256=example_study_digest(joinpath(modeldir,"conditions.json")),
        primary_summary_sha256=example_study_digest(jointpath),generation_pids=unique(pids),rows,controls,
        source_sha256=Dict(f=>example_study_digest(joinpath(@__DIR__,f)) for f in RPN_NATIVE_AUDIT_SOURCES))
    bytes=Vector{UInt8}(codeunits(JSON3.write(packet)))
    output=joinpath(directory,"input_PRIVATE.json");open(io->write(io,bytes),output,"w")
    (;sha256=bytes2hex(sha256(bytes)),bytes=length(bytes),rows=length(rows),controls=length(controls))
end
