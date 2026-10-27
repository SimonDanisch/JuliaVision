include("plot_interface_study.jl")
include("score_plot_interface.jl")
using Test
function test_plot_interface_study()
    @testset "Mock plotting study freezes16 and waits before private grading" begin
        mktempdir() do directory
            p=freeze_plot_interface(;token_count=_ -> 1000,directory)
            @test length(p.jobs)==16
            @test length(unique(j.task for j in p.jobs))==8
            @test all(j->j.input_tokens==1000,p.jobs)
            @test all(!haskey(p.source_sha256,file) for file in ("plot_interface_references.jl","safe_plot_julia_source.jl","score_plot_interface.jl"))
            @test_throws ErrorException freeze_plot_interface(;token_count=_ -> 1000,directory)
            @test_throws ErrorException score_plot_interface(;directory)
            digest(path)=bytes2hex(sha256(read(path)))
            save(path,value)=open(io->JSON3.write(io,value),path,"w")
            conditions=joinpath(directory,"conditions.json");tasks=unique(String(j.task) for j in p.jobs)
            references=plot_interface_references();expected=Dict(tasks[i]=>references[i] for i in 1:8)
            for job in p.jobs
                attempt=joinpath(directory,job.id*"_attempt.json");save(attempt,(;id=job.id))
                messages=JSON3.read(JSON3.write(job.messages))
                reference=expected[job.task]
                answer=job.arm=="julia_source" ? plot_interface_native_reference(SafePlotSpec.parse_spec(JSON3.write(reference))) : JSON3.write(reference)
                receipt=(;id=job.id,messages,answer,normal_final_waitidle=true,conditions_sha256=digest(conditions),attempt_sha256=digest(attempt),
                    input_sha256=bytes2hex(sha256(JSON3.write(messages))),budget_tokens=1024,decode_steps_executed=64,
                    decode_submissions=64,eos_output_poll_chunk=64,truncated=false,
                    sampling=(;seed=20261025,topk=20,temperature=0.6f0,top_p=0.95f0,presence=0f0,draws=65))
                save(joinpath(directory,job.id*".json"),receipt)
            end
            result=score_plot_interface(;directory)
            @test result.complete
            @test all(a->a.admitted==a.specification_correct==a.total==8,result.arms)
            @test all(r->r.actual_render=="not_yet_attempted"&&r.aesthetic_score===nothing,result.rows)
            path=joinpath(directory,first(p.jobs).id*".json")
            changed=Dict{String,Any}(String(k)=>v for (k,v) in pairs(JSON3.read(read(path,String))))
            changed["answer"]="run(cmd)";save(path,changed)
            @test score_plot_interface(;directory).rows[1].status=="failed"
        end
    end
end
