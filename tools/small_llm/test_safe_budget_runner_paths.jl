using JSON3,SHA,Test

"""Exercise the frozen runner with private CPU mocks, never hardware receipts."""
function test_safe_budget_runner_paths()
    tests=@testset "Frozen runner lifecycle, CPU mocks ONLY" begin
        for scenario in (:success,:partial,:generation_error,:load_error,:device_error,:close_error,:finish_error)
            mktempdir() do root
                put(path,x)=begin mkpath(dirname(path));open(io->JSON3.pretty(io,x),path,"w") end
                digest(path)=bytes2hex(open(sha256,path))
                baseline=joinpath(root,"baseline","conditions.json");put(baseline,(;fixture=true))
                pilot=joinpath(root,"pilot","conditions.json");put(pilot,(;fixture=true))
                put(joinpath(dirname(pilot),"summary.json"),(;complete=true,scope="CPU MOCK ONLY"))
                tasks=[(;id=id) for id in ("pilot","a","b","c")]
                decoding=(;max_tokens=2048,seed=20261024,topk=20,temperature=0.6,top_p=0.95,presence=0.0,output_chunk=128)
                parent=(;path=pilot,packet=(;task="pilot",decoding),state=(;path=baseline,tasks))
                for task in tasks[2:end],arm in ("api_pack","no_addon")
                    put(joinpath(dirname(baseline),task.id*"_"*arm*".json"),
                        (;conditions_sha256=digest(baseline),messages=[Dict("user"=>task.id*"_"*arm)]))
                end
                source=joinpath(@__DIR__,"safe_budget_transfer_extension.jl")
                cp(source,joinpath(root,basename(source)))
                fixture=Module(gensym(:RunnerLifecycleFixture))
                declarations=quote
                    using JSON3,SHA
                    const TEST_ROOT=$root
                    const PARENT=$(QuoteNode(parent))
                    const CLOCK=Ref(0.0)
                    const FINISHED=Ref(0)
                    const CLOSE_ERROR=Ref(false)
                    const FINISH_ERROR=Ref(false)
                    const GENERATE=Ref{Any}(nothing)
                    mutable struct MockModel
                        open::Bool
                    end
                    Base.isopen(m::MockModel)=m.open
                    function Base.close(m::MockModel)
                        m.open=false
                        CLOSE_ERROR[] && error("simulated model close failure")
                        nothing
                    end
                    time()=CLOCK[]
                    safe_reasoning_budget_study_state(;require_hardware=false)=PARENT
                    vulkan_benchmark_hash(path)=bytes2hex(open(sha256,path))
                    qualified_qwen_cache()=:mock_cache
                    inference_runtime_provenance()=(;scope="PRIVATE CPU MOCK, NOT INFERENCE")
                    function finish_device!(device)
                        FINISHED[]+=1
                        FINISH_ERROR[] && error("simulated final waitidle failure")
                        nothing
                    end
                    sampled_safe_budget_julia_answer(args...;kwargs...)=GENERATE[](args...;kwargs...)
                    module Mantle
                        const CREATE=Ref{Any}(nothing)
                        Device(name)=CREATE[](name)
                    end
                    module SmallLLMChecks
                        module QwenText
                            Tokenizer(path)=:mock_tokenizer
                        end
                        module Qwen35Experiment
                            const LOAD=Ref{Any}(nothing)
                            loadmodel(args...;kwargs...)=LOAD[](args...;kwargs...)
                        end
                    end
                end
                declarations.head=:toplevel
                Core.eval(fixture,declarations)
                Base.include_string(fixture,replace(read(source,String),"@__DIR__"=>"TEST_ROOT"),source)
                Base.invokelatest() do
                    creates=Ref(0);loads=Ref(0);generations=Ref(0);models=Any[]
                    fixture.CLOSE_ERROR[]=scenario==:close_error
                    fixture.FINISH_ERROR[]=scenario==:finish_error
                    fixture.Mantle.CREATE[]=name->begin
                        @test name=="radeon"
                        creates[]+=1
                        scenario==:device_error && error("simulated device setup failure")
                        :mock_device
                    end
                    fixture.SmallLLMChecks.Qwen35Experiment.LOAD[]=(args...;kwargs...)->begin
                        loads[]+=1
                        scenario==:load_error && error("simulated load failure")
                        model=fixture.MockModel(true);push!(models,model);model
                    end
                    fixture.GENERATE[]=(model,tokenizer,messages;kwargs...)->begin
                        @test isopen(model)
                        @test kwargs[:thinking_budget]==1024 && kwargs[:max_tokens]==2048
                        @test kwargs[:seed]==UInt32(20261024)
                        sample_id=last(only(messages));directory=joinpath(root,"safe_budget_transfer_extension_results")
                        @test isfile(joinpath(directory,sample_id*"_attempt.json"))
                        @test !isfile(joinpath(directory,sample_id*".json"))
                        generations[]+=1
                        scenario==:generation_error && generations[]==1 && error("simulated generation failure")
                        fixture.CLOCK[]+=scenario==:partial ? 100 : 1
                        (;code="mock output, never executed",truncated=false,finished_thinking=true,forced_closures=1)
                    end
                    packet=fixture.freeze_safe_budget_transfer_extension()
                    directory=joinpath(root,"safe_budget_transfer_extension_results")
                    result=try fixture.run_safe_budget_transfer_extension() catch e;e end
                    @test fixture.FINISHED[]==(scenario==:device_error ? 0 : 1)
                    @test creates[]==1
                    @test loads[]==(scenario==:device_error ? 0 : 1)
                    @test all(model->!isopen(model),models)
                    if scenario in (:load_error,:device_error)
                        @test result isa ErrorException
                        @test generations[]==0
                        @test !any(endswith("_attempt.json"),readdir(directory))
                    elseif scenario in (:close_error,:finish_error)
                        @test result isa ErrorException
                        @test generations[]==6
                        @test all(isfile(joinpath(directory,row.id*".json")) for row in packet.order)
                    elseif scenario==:generation_error
                        @test result isa ErrorException
                        @test generations[]==1
                        id=first(packet.order).id
                        row=JSON3.read(read(joinpath(directory,id*".json"),String))
                        @test occursin("simulated generation failure",row.inference_error)
                        @test row.attempt_sha256==digest(joinpath(directory,id*"_attempt.json"))
                        failurehash=digest(joinpath(directory,id*".json"))
                        @test fixture.run_safe_budget_transfer_extension().complete
                        @test generations[]==6
                        @test digest(joinpath(directory,id*".json"))==failurehash
                        @test fixture.FINISHED[]==2
                        @test all(model->!isopen(model),models)
                    else
                        expected=scenario==:partial ? 2 : 6
                        @test result.complete==(scenario==:success)
                        @test generations[]==expected
                        @test length(result.completed)==expected
                        for row in packet.order[1:expected]
                            output=joinpath(directory,row.id*".json")
                            saved=JSON3.read(read(output,String))
                            @test saved.attempt_sha256==digest(joinpath(directory,row.id*"_attempt.json"))
                            @test saved.conditions_sha256==digest(joinpath(directory,"conditions.json"))
                        end
                        if scenario==:success
                            before=readdir(directory)
                            @test fixture.run_safe_budget_transfer_extension().complete
                            @test creates[]==loads[]==1
                            @test readdir(directory)==before
                        else
                            saved_hashes=Dict(row.id=>digest(joinpath(directory,row.id*".json")) for row in packet.order[1:2])
                            @test !fixture.run_safe_budget_transfer_extension().complete
                            @test fixture.run_safe_budget_transfer_extension().complete
                            @test generations[]==6
                            @test fixture.FINISHED[]==3 && creates[]==loads[]==3
                            @test all(model->!isopen(model),models)
                            @test all(digest(joinpath(directory,id*".json"))==hash for (id,hash) in saved_hashes)
                        end
                    end
                end
            end
        end
    end
    output=joinpath(@__DIR__,"safe_budget_runner_paths_cpu_validation.json")
    isfile(output) && error("preserve previous qualification")
    counts=Test.get_test_counts(tests)
    report=(;complete=true,passed=counts.passes+counts.cumulative_passes,
        scope="PRIVATE CPU MOCKS ONLY: orchestration/receipt/cleanup logic, NOT hardware or intelligence evidence",
        source_sha256=Dict(f=>bytes2hex(open(sha256,joinpath(@__DIR__,f))) for f in
            ("test_safe_budget_runner_paths.jl","safe_budget_transfer_extension.jl")))
    open(io->JSON3.pretty(io,report),output,"w");report
end
