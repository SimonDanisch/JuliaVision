include("pubmedqa_retrieval_protocol.jl")
include("score_pubmedqa_retrieval.jl")
using Test
function test_pubmedqa_retrieval_scoring()
    @testset "Four-arm scorer keeps every condition and strict labels" begin
        mktempdir() do root
            directory=mkpath(joinpath(root,"results"));private=mkpath(joinpath(root,"private"))
            save(path,value)=open(io->JSON3.write(io,value),path,"w")
            digest(path)=bytes2hex(sha256(read(path)))
            goldpath=joinpath(private,"gold.json");save(goldpath,Dict("1"=>"yes","2"=>"no"))
            manifestpath=joinpath(root,"manifest.json");save(manifestpath,(;gold_sha256=digest(goldpath)))
            tasks=[(;pmid=string(i),question="topic$(i)",abstract_context=["topic$(i)"]) for i in 1:2]
            corpus=[(;qualified=t.pmid,text=only(t.abstract_context)) for t in tasks]
            jobs=PubMedQARetrievalProtocol.jobs(tasks,corpus)
            decoding=(;max_tokens=128,seed=20261025,topk=20,temperature=0.6f0,top_p=0.95f0,presence=0f0,output_chunk=64)
            packet=(;jobs,decoding,scope="mock",policy="all8",snapshot_manifest_sha256=digest(manifestpath),
                private_scorer_sha256=digest(joinpath(@__DIR__,"score_pubmedqa_retrieval.jl")),
                private_metric_sources=Dict(f=>digest(joinpath(@__DIR__,f)) for f in
                    ("pubmedqa_metrics.jl","pubmedqa_reading_protocol.jl","score_pubmedqa_reading.jl")))
            path=joinpath(directory,"conditions.json");save(path,packet)
            pending=score_pubmedqa_retrieval(;directory,snapshot=root)
            @test !pending.complete&&pending.arms===nothing
            @test length(pending.rows)==8
            for job in jobs
                attempt=joinpath(directory,job.id*"_attempt.json");save(attempt,(;id=job.id))
                messages=JSON3.read(JSON3.write(job.messages))
                receipt=(;id=job.id,messages,normal_final_waitidle=true,conditions_sha256=digest(path),attempt_sha256=digest(attempt),
                    input_sha256=bytes2hex(sha256(JSON3.write(messages))),budget_tokens=128,decode_steps_executed=64,
                    decode_submissions=64,eos_output_poll_chunk=64,truncated=false,answer="yes",
                    sampling=(;seed=decoding.seed,topk=decoding.topk,temperature=decoding.temperature,top_p=decoding.top_p,presence=decoding.presence,draws=65))
                save(joinpath(directory,job.id*".json"),receipt)
            end
            result=score_pubmedqa_retrieval(;directory,snapshot=root)
            @test result.complete&&length(result.arms)==4
            @test all(a->a.correct==1&&a.total==2,result.arms)
            output=joinpath(directory,first(jobs).id*".json")
            changed=Dict{String,Any}(String(k)=>v for (k,v) in pairs(JSON3.read(read(output,String))))
            changed["answer"]="YES.";save(output,changed)
            @test score_pubmedqa_retrieval(;directory,snapshot=root).rows[1].status=="invalid_answer"
            changed["input_sha256"]="different";save(output,changed)
            @test_throws ErrorException score_pubmedqa_retrieval(;directory,snapshot=root)
        end
    end
end
