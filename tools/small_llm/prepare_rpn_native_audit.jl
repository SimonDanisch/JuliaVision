# ROOT-only. Do not deploy this source or its fixture/formula imports.
include("score_rpn_stencil_study.jl")
isdefined(@__MODULE__,:RPN_NATIVE_AUDIT_SOURCES)||include("run_rpn_native_audit.jl")

function private_rpn_native_cases(fixture)
    reference=Core.eval(Module(gensym(:PrivateRPNReference)),Meta.parse("(x,i,n)->("*fixture.expected*")"))
    cases=Any[]
    for n in (0,1,2,3,7,31,65),family in 1:3
        x=Float32[family==1 ? sin(Float32(i)*0.71f0)*2.3f0+0.2f0 :
            family==2 ? (isodd(i) ? -1.5f0 : 2.25f0) : Float32(i)*0.17f0-0.8f0 for i in 1:max(n,1)]
        initial=fill(-913f0,n+7);expected=copy(initial)
        for i in 1:n;expected[i]=Base.invokelatest(reference,x,i,n);end
        push!(cases,(;n,x,initial,expected))
    end
    cases
end

"""After all generations finish: prepare SAME-input physical-device audit.

No new model attempts or changed primary scores. Private test values may only
go to a fresh GRADING-only process, never a live/future generation process.
"""
function prepare_rpn_native_audit(;directory=joinpath(@__DIR__,"rpn_stencil_results"))
    path=joinpath(directory,"native_audit_input_PRIVATE.json")
    ispath(path)&&error("preserve native audit input")
    state=rpn_stencil_state(directory);p=state.packet
    summarypath=joinpath(directory,"summary.json");summary=JSON3.read(read(summarypath,String))
    summary.complete&&summary.conditions_sha256==laptop_rpn_digest(state.path)||error("finish all frozen generations/scoring first")
    validation=JSON3.read(read(joinpath(@__DIR__,"rpn_native_audit_orchestration_validation.json"),String))
    validation.complete&&validation.boundary_checks==9&&validation.preparation_checks==43||error("native audit orchestration not qualified")
    for (file,hash) in pairs(validation.source_sha256)
        basename(String(file))==String(file)&&laptop_rpn_digest(joinpath(@__DIR__,String(file)))==hash||error("native audit qualification source drift")
    end
    [r.id for r in summary.rows]==[j.id for j in p.jobs]||error("cohort drift")
    tasks=rpn_stencil_tasks();fixtures=private_rpn_stencil_fixtures();rows=Any[];generation_pids=Int[]
    for job in p.jobs
        output=joinpath(directory,String(job.id)*".json");attempt=replace(output,".json"=>"_attempt.json")
        candidate=JSON3.read(read(output,String));receipt=JSON3.read(read(attempt,String))
        validate_rpn_stencil_receipt(candidate,job,p;conditions_sha256=laptop_rpn_digest(state.path),attempt_sha256=laptop_rpn_digest(attempt))
        push!(generation_pids,Int(receipt.pid))
        index=only(findall(t->t.id==job.task,tasks))
        raw=hasproperty(candidate,:answer) ? String(candidate.answer) : ""
        admitted=RPNPointwiseInterface.parse_program(raw).accepted
        push!(rows,(;id=job.id,task=job.task,arm=job.arm,program=raw,
            generation_sha256=laptop_rpn_digest(output),primary_status=only(filter(r->r.id==job.id,summary.rows)).status,
            admitted,cases=admitted ? private_rpn_native_cases(fixtures[index]) : Any[]))
    end
    controls=Any[]
    for index in (1,5)
        fixture=fixtures[index];cases=private_rpn_native_cases(fixture)
        push!(controls,(;id=tasks[index].id*"_trusted",program=fixture.program,expected_pass=true,cases))
        push!(controls,(;id=tasks[index].id*"_wrong",program=fixture.mutant,expected_pass=false,cases))
    end
    packet=(;scope="Posthoc SAME-program physical-device correctness audit, NOT new neural attempts, independent tasks, changed primary grading, or speed evidence",
        privacy="Input/expected arrays and trusted/mutant controls are PRIVATE. Ephemeral fresh grading-only worker after all inference workers retired. Never load a model in that process; retire before any future generation.",
        conditions_sha256=laptop_rpn_digest(state.path),primary_summary_sha256=laptop_rpn_digest(summarypath),
        preparer_sha256=laptop_rpn_digest(@__FILE__),generation_pids=unique(generation_pids),controls,rows,
        source_sha256=Dict(f=>laptop_rpn_digest(joinpath(@__DIR__,f)) for f in RPN_NATIVE_AUDIT_SOURCES))
    open(io->JSON3.pretty(io,packet),path,"w");packet
end
