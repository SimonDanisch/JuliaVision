using JSON3,SHA,Test
isdefined(@__MODULE__,:RPNPointwiseInterface)||include("rpn_pointwise_interface.jl")

function rpn_repetition_run(actions)
    isempty(actions)&&return 0
    best=1;current=1
    for i in 2:length(actions)
        current=actions[i]==actions[i-1] ? current+1 : 1;best=max(best,current)
    end
    best
end

"""Completed-output diagnostics only; no private formulas/probes or new generation."""
function audit_rpn_format_followup(;destination=joinpath(@__DIR__,"rpn_format_followup_audit.json"))
    isfile(destination)&&error("Preserve completed-output audit")
    groups=Any[]
    for folder in ("rpn_stencil_results","rpn_stencil_contract_v2_results")
        directory=joinpath(@__DIR__,folder);conditions=joinpath(directory,"conditions.json")
        summarypath=joinpath(directory,"summary.json");p=JSON3.read(read(conditions,String));s=JSON3.read(read(summarypath,String))
        s.complete&&s.conditions_sha256==bytes2hex(sha256(read(conditions)))||error("Unfinished or drifted cohort")
        [r.id for r in s.rows]==[j.id for j in p.jobs]||error("Score order")
        rows=Any[]
        for (job,score) in zip(p.jobs,s.rows)
            path=joinpath(directory,String(job.id)*".json");output=JSON3.read(read(path,String))
            output.conditions_sha256==s.conditions_sha256||error("Candidate provenance")
            parsed=RPNPointwiseInterface.parse_program(String(output.answer))
            parsed.accepted==score.syntax_accepted||error("Admission disagreement")
            actions=parsed.accepted ? parsed.actions[1:end-1] : Int32[]
            counts=Dict(RPNPrefixGrammar.ACTIONS[a]=>count(==(a),actions) for a in unique(actions))
            push!(rows,(;id=job.id,arm=job.arm,status=score.status,syntax_accepted=parsed.accepted,
                generated_tokens=output.generated_tokens,truncated=output.truncated,
                operations=parsed.accepted ? length(actions) : nothing,
                extra_closure_operations=parsed.accepted ? max(0,length(actions)-32) : nothing,
                longest_identical_action_run=rpn_repetition_run(actions),action_counts=counts,
                output_sha256=bytes2hex(sha256(read(path)))))
        end
        push!(groups,(;folder,conditions_sha256=s.conditions_sha256,summary_sha256=bytes2hex(sha256(read(summarypath))),rows))
    end
    checks=@testset "Completed RPN output audit integrity" begin
        @test length(groups)==2
        @test all(g->length(g.rows)==16,groups)
        @test rpn_repetition_run(Int32[])==0
        @test rpn_repetition_run(Int32[1,1,2,2,2,1])==3
        @test all(r->r.status!="passed"||r.syntax_accepted,Iterators.flatten(g.rows for g in groups))
    end
    c=Test.get_test_counts(checks)
    result=(;complete=true,scope="POSTHOC output-format/repetition diagnostics, not new model attempts. Private expected values, formulas and probe arrays not read. Do not pool these related cohorts into an intelligence score.",
        checks=c.passes+c.cumulative_passes,source_sha256=bytes2hex(sha256(read(@__FILE__))),groups)
    open(destination,"w") do io;JSON3.write(io,result);end
    println("Output-only diagnostic checks=",result.checks);result
end
