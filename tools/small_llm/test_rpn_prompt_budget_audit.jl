include("audit_rpn_prompt_budget.jl")
using Test
function test_rpn_prompt_budget_audit()
    @testset "Literal prompt audit retains primary results and stricter endpoint" begin
        mktempdir() do directory
            jobs=[(;id="short",arm="enabled"),(;id="long",arm="enabled"),(;id="pending",arm="disabled")]
            conditions=joinpath(directory,"conditions.json")
            write(conditions,JSON3.write((;jobs,grammar=(;operation_budget=32,max_depth=8))))
            hash=bytes2hex(sha256(read(conditions)))
            summarypath=joinpath(directory,"summary.json")
            write(summarypath,JSON3.write((;complete=false,conditions_sha256=hash,rows=[(;j...,status=j.id=="pending" ? "pending" : "passed") for j in jobs])))
            #32 normal pushes/unaries followed by one closure: legal in the
            #frozen grammar, not legal under the literal32-op prompt wording.
            long="x0\nx0\n"*repeat("neg\n",30)*"add\nfinish"
            @test RPNPointwiseInterface.parse_program(long).accepted
            for (id,answer) in (("short","x0\nfinish"),("long",long))
                write(joinpath(directory,id*".json"),JSON3.write((;answer,conditions_sha256=hash)))
            end
            before=read(summarypath)
            audit=audit_rpn_prompt_budget(;directory)
            @test !audit.complete
            @test audit.rows[1].strict_success
            @test audit.rows[2].grammar_admitted&&audit.rows[2].operations==33
            @test !audit.rows[2].literal_compliance&&!audit.rows[2].strict_success
            @test audit.rows[2].primary_status=="passed"
            @test audit.rows[3].status=="not_auditable"
            @test read(summarypath)==before
            altered=JSON3.read(String(before),Dict{String,Any})
            altered["rows"][1]["generation_sha256"]="wrong"
            write(summarypath,JSON3.write(altered))
            @test_throws ErrorException audit_rpn_prompt_budget(;directory)
        end
    end
end
