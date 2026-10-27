using JSON3,SHA,Test

function test_shared_source_kernel_protocol()
    checks=@testset "Shared-source protocol PRIVATE CPU MOCKS" begin
        mktempdir() do root
            put(path,x)=begin mkpath(dirname(path));open(io->JSON3.pretty(io,x),path,"w") end
            hash(path)=bytes2hex(open(sha256,path))
            parentpath=joinpath(root,"parent.json");put(parentpath,(;scope="PRIVATE MOCK"))
            apipath=joinpath(root,"api.json");put(apipath,(;scope="PRIVATE MOCK"))
            put(joinpath(root,"shared_source_kernel_reference_validation.json"),
                (;complete=true,rows=fill(nothing,4),source_sha256=Dict()))
            for file in ("shared_source_kernel_benchmark.jl","shared_source_kernel_tasks.jl","typed_pointwise_kernel_lowering.jl","kernel_void_tail.jl")
                cp(joinpath(@__DIR__,file),joinpath(root,file))
            end
            decoding=(;max_tokens=2048,seed=20261024,topk=20,temperature=0.6,top_p=0.95,presence=0.0,output_chunk=128)
            parent=(;path=parentpath,packet=(;decoding))
            api=(;path=apipath,text="SAME_PUBLIC_API_FIXTURE")
            tasks=[(;id=id,category="vulkan_kernel",setup="PRIVATE MOCK",tests="PRIVATE MOCK",
                cpu_prompt="cpu_"*id,gpu_prompt="gpu_"*id,reference_cpu="REFERENCE_ONLY_DO_NOT_EXPOSE") for id in ("a","b","c","d")]
            fixture=Module(gensym(:SharedSourceProtocolFixture))
            declarations=quote
                using JSON3,SHA
                const TEST_ROOT=$root
                const PARENT=$(QuoteNode(parent))
                const API=$(QuoteNode(api))
                const TASKS=$(QuoteNode(tasks))
                const GENERATIONS=Ref(0)
                const FINISHES=Ref(0)
                safe_reasoning_budget_study_state(;require_hardware=false)=PARENT
                shared_source_kernel_tasks()=TASKS
                load_kernel_api_pack()=API
                vulkan_benchmark_hash(path)=bytes2hex(open(sha256,path))
                qualified_qwen_cache()=:mock_cache
                inference_runtime_provenance()=(;scope="PRIVATE CPU MOCK, NOT MODEL EVIDENCE")
                finish_device!(device)=(FINISHES[]+=1;nothing)
                module Mantle
                    Device(name)=:mock_device
                end
                module SmallLLMChecks
                    module QwenText
                        Tokenizer(path)=:mock_tokenizer
                    end
                    module Qwen35Experiment
                        loadmodel(args...;kwargs...)=IOBuffer()
                    end
                end
                module JuliaTaskEval
                    using JSON3,SHA
                    const CALLS=Ref(0)
                    function evaluate(task,code;output)
                        CALLS[]+=1
                        result=(;passed=!occursin("bad_math",code),code_sha256=bytes2hex(sha256(code)))
                        open(io->JSON3.pretty(io,result),output,"w");result
                    end
                end
                function sampled_safe_budget_julia_answer(model,tokenizer,messages;kwargs...)
                    @assert kwargs[:max_tokens]==2048 && kwargs[:thinking_budget]==1024
                    @assert kwargs[:seed]==UInt32(20261024) && kwargs[:topk]==20
                    @assert kwargs[:temperature]==0.6f0 && kwargs[:top_p]==0.95f0 && kwargs[:presence]==0f0
                    GENERATIONS[]+=1
                    prompt=last(last(messages))
                    # Only one private mock condition fails inference. Later
                    # continuation must retain it, never retry that condition.
                    startswith(prompt,"cpu_c") && error("mock backend failure")
                    code=if startswith(prompt,"cpu_b")
                        "not an admitted function"
                    elseif startswith(prompt,"cpu_d")
                        "function kernel!(out::Vector{Float32},x::Vector{Float32},n::Int32);for i=1:n;bad_math=x[i]*2f0;out[i]=bad_math;end;nothing;end"
                    elseif startswith(prompt,"cpu_")
                        "function kernel!(out::Vector{Float32},x::Vector{Float32},n::Int32);for i=1:n;out[i]=x[i];end;nothing;end"
                    else
                        "function kernel!(out,x,n::Int32);return nothing;end"
                    end
                    (;code,forced_closures=1,truncated=false)
                end
            end
            declarations.head=:toplevel;Core.eval(fixture,declarations)
            Base.include(fixture,joinpath(@__DIR__,"typed_pointwise_kernel_lowering.jl"))
            Base.include(fixture,joinpath(@__DIR__,"kernel_void_tail.jl"))
            source=joinpath(@__DIR__,"shared_source_kernel_benchmark.jl")
            Base.include_string(fixture,replace(read(source,String),"@__DIR__"=>"TEST_ROOT"),source)
            Base.invokelatest() do
                packet=fixture.freeze_shared_source_kernel_benchmark()
                @test length(packet.jobs)==8
                @test [String(j.arm) for j in packet.jobs]==["cpu_source","gpu_source","gpu_source","cpu_source","cpu_source","gpu_source","gpu_source","cpu_source"]
                @test all(occursin(api.text,last(last(job.messages))) for job in packet.jobs)
                @test !occursin("REFERENCE_ONLY_DO_NOT_EXPOSE",JSON3.write(packet))
                initial=fixture.score_shared_source_kernel_benchmark()
                @test !initial.complete && all(row->row.status=="pending",initial.rows)
                err=try fixture.run_shared_source_kernel_benchmark();nothing catch e;e end
                @test err isa ErrorException
                @test fixture.GENERATIONS[]==5 && fixture.FINISHES[]==1
                directory=joinpath(root,"shared_source_kernel_results")
                failedpath=joinpath(directory,"c_cpu_source.json");failurehash=hash(failedpath)
                partial=fixture.score_shared_source_kernel_benchmark()
                @test !partial.complete
                @test count(row->row.status=="inference_error",partial.rows)==1
                @test count(row->row.status=="pending",partial.rows)==3
                @test fixture.run_shared_source_kernel_benchmark().complete
                @test fixture.GENERATIONS[]==8 && fixture.FINISHES[]==2
                @test hash(failedpath)==failurehash
                completed=fixture.score_shared_source_kernel_benchmark()
                @test completed.complete
                @test count(row->row.status=="passed",completed.rows)==5
                @test count(row->row.status=="failed",completed.rows)==2
                @test count(row->row.status=="inference_error",completed.rows)==1
                @test only(filter(row->row.id=="b_cpu_source",completed.rows)).stage=="compiler_admission"
                @test only(filter(row->row.id=="d_cpu_source",completed.rows)).stage=="execution"
                before=fixture.JuliaTaskEval.CALLS[]
                @test fixture.score_shared_source_kernel_benchmark().complete
                @test fixture.JuliaTaskEval.CALLS[]==before
                @test fixture.run_shared_source_kernel_benchmark().complete
                @test fixture.GENERATIONS[]==8 && fixture.FINISHES[]==2
                attempt=joinpath(directory,"a_cpu_source_attempt.json");put(attempt,(;tampered=true))
                err=try fixture.score_shared_source_kernel_benchmark();nothing catch e;e end
                @test err isa ErrorException
                @test occursin("attempt receipt drift",sprint(showerror,err))
            end
        end
    end
    output=joinpath(@__DIR__,"shared_source_kernel_protocol_cpu_validation.json")
    isfile(output) && error("preserve previous qualification")
    counts=Test.get_test_counts(checks)
    report=(;complete=true,passed=counts.passes+counts.cumulative_passes,
        scope="PRIVATE CPU MOCKS ONLY, no model or hardware evidence",
        source_sha256=Dict(f=>bytes2hex(open(sha256,joinpath(@__DIR__,f))) for f in
            ("test_shared_source_kernel_protocol.jl","shared_source_kernel_benchmark.jl")))
    open(io->JSON3.pretty(io,report),output,"w");report
end
