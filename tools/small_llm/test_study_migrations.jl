include("audit_study_migrations.jl")
using Test
function test_study_migrations()
    @testset "Migration closure does not regenerate or overwrite old slots" begin
        mktempdir() do root
            old=mkpath(joinpath(root,"old"));new=mkpath(joinpath(root,"new"))
            save(path,value)=open(io->JSON3.write(io,value),path,"w")
            digest(path)=bytes2hex(sha256(read(path)))
            jobs=[(;id="one",messages=[Dict("user"=>"public")])]
            oldpath=joinpath(old,"conditions.json");save(oldpath,(;jobs))
            newpath=joinpath(new,"conditions.json");save(newpath,(;jobs,parent_conditions_sha256=digest(oldpath)))
            save(joinpath(new,"one_attempt.json"),(;id="one"))
            save(joinpath(new,"one.json"),(;id="one",messages=jobs[1].messages,conditions_sha256=digest(newpath),attempt_sha256=digest(joinpath(new,"one_attempt.json"))))
            summarypath=joinpath(new,"summary.json")
            summary=(;complete=true,conditions_sha256=digest(newpath),rows=[(;id="one",status="failed")])
            save(summarypath,summary)
            result=audit_study_migration(old,new,:parent_conditions_sha256)
            @test result.complete
            @test only(result.rows).old_slot=="unattempted_preserved"
            @test only(result.rows).new_slot=="scored_on_different_qualified_runtime"
            save(summarypath,(;summary...,complete=false))
            @test_throws ErrorException audit_study_migration(old,new,:parent_conditions_sha256)
            save(summarypath,summary)
            write(joinpath(old,"one_attempt.json"),"{}")
            @test_throws ErrorException audit_study_migration(old,new,:parent_conditions_sha256)
        end
    end
end
