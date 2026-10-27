using JSON3,SHA,Test

function compare_safe_budget_pilot()
    root=@__DIR__;rows=Any[]
    hash(path)=bytes2hex(open(sha256,path))
    @testset "Frozen paired budget comparison integrity" begin
        for arm in ("no_addon","api_pack")
            name="transfer_clamped_gather_"*arm*".json"
            original=joinpath(root,"kernel_transfer_results",name)
            controlled=joinpath(root,"safe_reasoning_budget_results",name)
            before=JSON3.read(read(original,String));after=JSON3.read(read(controlled,String))
            @test after.original_sha256==hash(original)
            @test before.input_sha256==after.input_sha256
            @test before.messages==after.messages
            @test before.budget_tokens==after.budget_tokens==2048
            for field in (:seed,:topk,:temperature,:top_p,:presence)
                @test getproperty(before.sampling,field)==getproperty(after.sampling,field)
            end
            @test before.decode_steps_executed==before.decode_submissions
            @test after.decode_steps_executed==after.decode_submissions
            # Compare without copying private reasoning text into this audit.
            # Decoded-prefix equality is diagnostic, not full token-ID parity.
            prefix=first(split(String(after.answer),"</think>";limit=2))
            prefix_check=(;matches_original=startswith(String(before.answer),prefix),
                utf8_bytes=ncodeunits(prefix),sha256=bytes2hex(sha256(prefix)))
            scores=Dict{String,Any}()
            for variant in ("raw","compiler")
                bp=replace(original,".json"=>"_"*variant*"_score.json")
                ap=replace(controlled,".json"=>"_"*variant*"_score.json")
                b=JSON3.read(read(bp,String));a=JSON3.read(read(ap,String))
                scores[variant]=(;before_passed=b.passed,after_passed=a.passed,
                    original_score_sha256=hash(bp),controlled_score_sha256=hash(ap))
            end
            push!(rows,(;arm,scores,pre_closure_decoded_prefix=prefix_check,
                before=(;generated=before.generated_tokens,executed_steps=before.decode_steps_executed,
                    truncated=before.truncated,finished_thinking=before.finished_thinking,elapsed_seconds=before.elapsed_seconds),
                after=(;generated=after.generated_tokens,executed_steps=after.decode_steps_executed,
                    truncated=after.truncated,finished_thinking=after.finished_thinking,
                    forced_closures=after.forced_closures,elapsed_seconds=after.elapsed_seconds),
                original_sha256=hash(original),controlled_sha256=hash(controlled)))
        end
    end
    report=(;scope="Single authored gather task, post-hoc budget intervention; two context arms, NOT two independent tasks",
        result="API-pack raw kernel changes fail->pass; no-addon remains fail; no separate compiler gain in this pilot",
        limits=["One fixed seed, one task; not frontier intelligence or independent held-out evidence",
            "Reported elapsed excludes cold model load/session setup; EOS polling executes extra tokens beyond first EOS",
            "API-context benefit with the controller is one paired observation, not evidence that arbitrary knowledge is now integrated"],
        rows,source_sha256=hash(@__FILE__))
    output=joinpath(root,"safe_budget_pilot_comparison.json")
    open(io->JSON3.pretty(io,report),output,"w");report
end
