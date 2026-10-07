using DNNKernels, HorizonRunner, BonsaiRunner, Mantle, KernelAbstractions, Test, Statistics

function check_bonsai_e2e(model; chunks=(0,1,2,3,8,16), newtokens=8)
    results = []
    @testset "full Bonsai decode and chunked prefill" begin
        for prompt in ("Compute 2 + 2.", "Explain why the sky is blue.")
            ids = BonsaiRunner.encode(model.tokenizer,
                BonsaiRunner.chatprompt(["user"=>prompt]))
            reference = nothing
            for chunk in chunks
                s = BonsaiRunner.session(model)
                try
                    logits = if chunk == 0
                        last([Float32.(Array(BonsaiRunner.step!(s,token))) for token in ids])
                    else
                        Float32.(Array(BonsaiRunner.prefill!(s,ids;chunk)))
                    end
                    tokens = Int[]
                    current = logits
                    for i in 1:newtokens
                        @test all(isfinite,current)
                        token = argmax(current)-1
                        push!(tokens,token)
                        i == newtokens || (current = Float32.(Array(BonsaiRunner.step!(s,token))))
                    end
                    @test all(isfinite,logits)
                    if reference === nothing
                        reference = (;logits,tokens)
                    end
                    rel = maximum(abs,logits-reference.logits)/max(maximum(abs,reference.logits),eps(Float32))
                    @test rel < 0.01
                    @test tokens == reference.tokens
                    row = (;prompt,chunk,prompt_tokens=length(ids),relative_max=rel,tokens)
                    push!(results,row); println(row); flush(stdout)
                finally
                    BonsaiRunner.release!(s)
                end
            end
        end
    end
    results
end

function q8mode!(enabled)
    DNNKernels.Q8SMALL[] = enabled
    DNNKernels.Q8EPILOGUE[] = enabled
    DNNKernels.Q8GATEUP[] = enabled
end

function e2eplans(model)
    plans = DNNKernels.RecordedPlan[]
    for v in values(model.scratch)
        v isa DNNKernels.RecordedPlan && push!(plans,v)
        v isa DNNKernels.Model && append!(plans,e2eplans(v))
    end
    plans
end

function release_e2e!(model)
    cache = get(model.scratch,(:horizon_generation_cache,),nothing)
    for v in values(model.scratch)
        v isa DNNKernels.Model && release_e2e!(v)
    end
    DNNKernels.releaseplans!(model)
    cache === nothing || DNNKernels.releasedevice!(cache)
end

function check_horizon_q8_e2e(h; repeats=3, newtokens=12)
    original = (DNNKernels.Q8SMALL[],DNNKernels.Q8EPILOGUE[],DNNKernels.Q8GATEUP[])
    base = h.model
    models = [HorizonRunner.Horizon32B(h.backend,
        DNNKernels.Model(base.graphs,base.weights,base.device,base.memevery,base.memframes,base.topk),
        h.maxlen,h.prefill,h.vocab) for _ in 1:2]
    results = []
    try
        @testset "full Horizon INT8 optimization A/B" begin
            for prompt in (Int64[0,250018],Int64[0,250018,2672],
                           Int64[0,250018,2672,200,46348,803,2853,18147])
                output = []
                for i in 1:2
                    q8mode!(i==2)
                    tokens = HorizonRunner.generate(models[i],prompt;maxnew=newtokens)
                    @test all(p -> all(isfinite,Array(first(p.outputs))),e2eplans(models[i].model))
                    push!(output,tokens)
                end
                @test output[1] == output[2]
                samples = (Float64[],Float64[])
                for r in 1:repeats, i in (isodd(r) ? (1,2) : (2,1))
                    q8mode!(i==2)
                    t = @elapsed actual = HorizonRunner.generate(models[i],prompt;maxnew=newtokens)
                    @test all(p -> all(isfinite,Array(first(p.outputs))),e2eplans(models[i].model))
                    @test actual == output[1]
                    push!(samples[i],t)
                end
                old,new = median.(samples)
                row = (;prompt_tokens=length(prompt),tokens=output[1],old_seconds=old,
                       new_seconds=new,speedup=old/new,old_samples=samples[1],new_samples=samples[2])
                push!(results,row); println(row); flush(stdout)
            end
            passes = [sum(length(v.plan.passes) for v in e2eplans(m.model)) for m in models]
            @test passes[2] < passes[1]
            println((recorded_passes=passes,))
        end
    finally
        DNNKernels.Q8SMALL[],DNNKernels.Q8EPILOGUE[],DNNKernels.Q8GATEUP[] = original
        foreach(m->release_e2e!(m.model),models)
    end
    results
end

function check_horizon_cold_plans(h, reference; rounds=8)
    base = h.model
    original = (DNNKernels.Q8SMALL[],DNNKernels.Q8EPILOGUE[],DNNKernels.Q8GATEUP[])
    try
        @testset "Horizon fresh-plan stress" begin
            for r in 1:rounds
                models = [HorizonRunner.Horizon32B(h.backend,
                    DNNKernels.Model(base.graphs,base.weights,base.device,base.memevery,base.memframes,base.topk),
                    h.maxlen,h.prefill,h.vocab) for _ in 1:2]
                try
                    for (j,prompt) in enumerate((Int64[0,250018],Int64[0,250018,2672],
                            Int64[0,250018,2672,200,46348,803,2853,18147])), i in 1:2
                        q8mode!(i==2)
                        tokens = HorizonRunner.generate(models[i],prompt;maxnew=12)
                        @test tokens == reference[j].tokens
                        @test all(p -> all(isfinite,Array(first(p.outputs))),e2eplans(models[i].model))
                    end
                finally
                    foreach(m -> release_e2e!(m.model),models)
                end
                println("fresh-plan round=",r," generations=",6r); flush(stdout)
            end
        end
    finally
        DNNKernels.Q8SMALL[],DNNKernels.Q8EPILOGUE[],DNNKernels.Q8GATEUP[] = original
    end
    nothing
end
