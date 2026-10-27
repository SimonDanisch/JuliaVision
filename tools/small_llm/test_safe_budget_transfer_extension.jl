using JSON3,SHA,Test

function test_safe_budget_transfer_extension()
    @testset "Extension registration and immutable receipts, CPU mocks only" begin
        mktempdir() do root
            put(path,x)=begin mkpath(dirname(path));open(io->JSON3.pretty(io,x),path,"w") end
            hash(path)=bytes2hex(open(sha256,path))
            baseline=joinpath(root,"baseline","conditions.json");put(baseline,(;fixture=true))
            parentpath=joinpath(root,"pilot","conditions.json");put(parentpath,(;fixture=true))
            put(joinpath(dirname(parentpath),"summary.json"),(;complete=true,scope="PRIVATE CPU MOCK"))
            tasks=[(;id=id) for id in ("pilot","remaining_a","remaining_b","remaining_c")]
            parent=(;path=parentpath,packet=(;task="pilot",decoding=(;max_tokens=2048)),state=(;path=baseline,tasks))
            for task in tasks[2:end],arm in ("api_pack","no_addon")
                put(joinpath(dirname(baseline),task.id*"_"*arm*".json"),(;conditions_sha256=hash(baseline)))
            end
            path=joinpath(@__DIR__,"safe_budget_transfer_extension.jl")
            cp(path,joinpath(root,basename(path)))
            fixture=Module(gensym(:TransferExtensionFixture))
            Core.eval(fixture,:(using JSON3,SHA))
            Core.eval(fixture,:(const TEST_ROOT=$root))
            Core.eval(fixture,:(const parent_fixture=$(QuoteNode(parent))))
            Core.eval(fixture,:(safe_reasoning_budget_study_state(;require_hardware=false)=parent_fixture))
            Core.eval(fixture,:(vulkan_benchmark_hash(p)=bytes2hex(open(sha256,p))))
            Core.eval(fixture,:(qualified_qwen_cache()=error("unexpected GPU preparation")))
            Base.include_string(fixture,replace(read(path,String),"@__DIR__"=>"TEST_ROOT"),path)
            call(name)=Base.invokelatest(()->getfield(fixture,name)())
            packet=call(:freeze_safe_budget_transfer_extension)
            @test length(packet.order)==6
            @test [row.task for row in packet.order]==repeat(["remaining_a","remaining_b","remaining_c"];inner=2)
            @test [row.arm for row in packet.order]==["api_pack","no_addon","no_addon","api_pack","api_pack","no_addon"]
            initial=call(:score_safe_budget_transfer_extension)
            @test !initial.complete
            @test all(row->row.status=="pending",initial.rows)
            directory=joinpath(root,"safe_budget_transfer_extension_results")
            firstid=first(packet.order).id
            put(joinpath(directory,firstid*"_attempt.json"),(;fixture=true))
            err=try call(:run_safe_budget_transfer_extension);nothing catch e;e end
            @test err isa ErrorException
            @test occursin("unfinished prior attempt",sprint(showerror,err))
            interrupted=call(:score_safe_budget_transfer_extension)
            @test first(interrupted.rows).status=="interrupted_attempt"
            for row in packet.order
                attempt=joinpath(directory,row.id*"_attempt.json")
                isfile(attempt)||put(attempt,(;fixture=true))
                put(joinpath(directory,row.id*".json"),(;conditions_sha256=hash(joinpath(directory,"conditions.json")),
                    original_sha256=packet.original_sha256[row.id],attempt_sha256=hash(attempt),inference_error="CPU mock"))
            end
            completed=call(:score_safe_budget_transfer_extension)
            @test completed.complete
            @test length(completed.rows)==6
            @test all(row->row.status=="inference_error",completed.rows)
            put(joinpath(directory,firstid*"_attempt.json"),(;tampered=true))
            err=try call(:score_safe_budget_transfer_extension);nothing catch e;e end
            @test err isa ErrorException
            @test occursin("attempt receipt drift",sprint(showerror,err))
            original=joinpath(dirname(baseline),firstid*".json")
            put(original,(;tampered=true))
            err=try call(:safe_budget_transfer_extension_state);nothing catch e;e end
            @test err isa ErrorException
            @test occursin("original output drift",sprint(showerror,err))
        end
    end
end
