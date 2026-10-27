include("compact_bayes_demo_interruption.jl")
using Test
function test_compact_bayes_demo_interruption()
    @testset "Process loss is not normal waitidle or model output" begin
        mktempdir() do directory
            jobs=[(;id="j$i") for i in 1:8];p=(;jobs);hash="frozen"
            save(path,x)=open(io->JSON3.pretty(io,x),path,"w")
            @test length(compact_bayes_demo_pending(directory,p,hash))==8
            attempt=(;id="j2",pid=3756467,conditions_sha256=hash)
            path=joinpath(directory,"j2_attempt.json");save(path,attempt)
            @test_throws ErrorException compact_bayes_demo_pending(directory,p,hash)
            lost=(;attempt...,terminal_status="lost_process",attempt_sha256=bytes2hex(sha256(read(path))),
                original_process_absent=true,shared_lock_observed_free=true,normal_final_waitidle=false,regeneration_permitted=false)
            loss=joinpath(directory,"j2_lost.json");save(loss,lost)
            @test compact_bayes_demo_lost(directory,jobs[2],hash)
            @test [j.id for j in compact_bayes_demo_pending(directory,p,hash)]==["j$i" for i in (1,3,4,5,6,7,8)]
            save(loss,(;lost...,normal_final_waitidle=true))
            @test_throws ErrorException compact_bayes_demo_lost(directory,jobs[2],hash)
            save(loss,(;lost...,regeneration_permitted=true))
            @test_throws ErrorException compact_bayes_demo_lost(directory,jobs[2],hash)
            save(loss,(;lost...,pid=1))
            @test_throws ErrorException compact_bayes_demo_lost(directory,jobs[2],hash)
            save(loss,lost);save(joinpath(directory,"j2.json"),(;not_an_answer=true))
            @test_throws ErrorException compact_bayes_demo_lost(directory,jobs[2],hash)
        end
    end
end
