using JSON3, SHA, Test

function test_safe_reasoning_receipts()
    tests=@testset "Interrupted-attempt receipts (CPU mocks, no device)" begin
        mktempdir() do root
            writejson(path,value)=begin
                mkpath(dirname(path));open(io->JSON3.pretty(io,value),path,"w")
            end
            digest(path)=bytes2hex(open(sha256,path))
            baseline=joinpath(root,"baseline");directory=joinpath(root,"safe_reasoning_budget_results")
            transfer=joinpath(baseline,"conditions.json");writejson(transfer,(;fixture=true))
            qualification=joinpath(root,"safe_reasoning_budget_software_validation.json")
            writejson(qualification,(;fixture=true))
            originals=Dict{String,String}()
            for arm in ("no_addon","api_pack")
                path=joinpath(baseline,"fixture_task_"*arm*".json");writejson(path,(;fixture=true,arm))
                originals[arm]=digest(path)
            end
            gate="safe_reasoning_budget_full_model_validation.json"
            # These are mock receipts in a PRIVATE temporary directory, never
            # evidence accepted by the real experiment or a fake hardware pass.
            writejson(joinpath(root,gate),(;complete=true,normal_final_waitidle=true,
                hardware_fixture_passed=156,full_model_passed=9,source_sha256=Dict()))
            packet=(;scope="CPU MOCK ONLY",policy="mock",task="fixture_task",arms=["no_addon","api_pack"],
                transfer_conditions_sha256=digest(transfer),source_sha256=Dict(),
                software_qualification_sha256=digest(qualification),original_sha256=originals,required_hardware_gate=gate)
            conditions=joinpath(directory,"conditions.json");writejson(conditions,packet)
            fixture=(;path=transfer,tasks=[(;id="fixture_task")])
            module_fixture=Module(gensym(:SafeReceiptFixture))
            Core.eval(module_fixture,:(using JSON3,SHA))
            Core.eval(module_fixture,:(const TEST_ROOT=$root))
            Core.eval(module_fixture,:(const fixture_state=$(QuoteNode(fixture))))
            Core.eval(module_fixture,:(kernel_transfer_state()=fixture_state))
            Core.eval(module_fixture,:(vulkan_benchmark_hash(path)=bytes2hex(open(sha256,path))))
            Core.eval(module_fixture,:(sampled_safe_budget_julia_answer(args...;kwargs...)=error("unexpected generation")))
            Core.eval(module_fixture,:(qualified_qwen_cache()=error("unexpected GPU preparation")))
            path=joinpath(@__DIR__,"safe_reasoning_budget_study.jl")
            # Exercise the actual frozen implementation, redirecting ONLY its
            # artifact root. Existing imports are satisfied by CPU-only mocks.
            source=replace(read(path,String),"@__DIR__"=>"TEST_ROOT")
            Base.include_string(module_fixture,source,path)
            score=()->Base.invokelatest(getfield(module_fixture,:score_safe_reasoning_budget_study))
            run=()->Base.invokelatest(getfield(module_fixture,:run_safe_reasoning_budget_study))
            initial=score();@test !initial.complete
            @test all(r->r.status=="pending",initial.rows)
            attempt=joinpath(directory,"fixture_task_no_addon_attempt.json")
            writejson(attempt,(;fixture=true))
            interrupted=score();@test !interrupted.complete
            @test first(interrupted.rows).status=="interrupted_attempt"
            err=try run();nothing catch error;error end
            @test err isa ErrorException
            @test occursin("unfinished prior attempt",sprint(showerror,err))
            for arm in ("no_addon","api_pack")
                receipt=joinpath(directory,"fixture_task_"*arm*"_attempt.json")
                isfile(receipt) || writejson(receipt,(;fixture=true))
                writejson(joinpath(directory,"fixture_task_"*arm*".json"),
                    (;conditions_sha256=digest(conditions),original_sha256=originals[arm],
                        attempt_sha256=digest(receipt),inference_error="simulated backend failure"))
            end
            failures=score();@test failures.complete
            @test all(r->r.status=="inference_error",failures.rows)
            @test all(r->r.status!="failed",failures.rows)
            writejson(attempt,(;fixture="tampered"))
            err=try score();nothing catch error;error end
            @test err isa ErrorException
            @test occursin("attempt receipt drift",sprint(showerror,err))
        end
    end
    counts=Test.get_test_counts(tests)
    output=joinpath(@__DIR__,"safe_reasoning_receipt_validation.json")
    isfile(output) && error("preserve previous qualification")
    report=(;complete=true,passed=counts.passes+counts.cumulative_passes,
        scope="private CPU mock artifact tests ONLY, no device/hardware/model evidence",
        source_sha256=Dict(f=>bytes2hex(open(sha256,joinpath(@__DIR__,f))) for f in
            ("test_safe_reasoning_receipts.jl","safe_reasoning_budget_study.jl")))
    open(io->JSON3.pretty(io,report),output,"w");report
end
