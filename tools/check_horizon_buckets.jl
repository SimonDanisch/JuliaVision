# Load `bucket_horizon = horizon32b(dir=...)` first; reuses its device weights.
using HorizonRunner, DNNKernels, Mantle, Test, Statistics

function check_horizon_buckets(recorded)
    base = recorded.model
    fixedgraphs = Dict(n => base.graphs[n] for n in HorizonRunner.GRAPHS)
    immediate = DNNKernels.Model(fixedgraphs, base.weights, base.device,
        base.memevery, base.memframes, base.topk; record=false)
    reference = Horizon32B(recorded.backend, immediate, recorded.maxlen,
                          recorded.prefill, recorded.vocab)
    prompts = [Int64[0,250018,2672,200,46348,803,2853,18147,1094,293,
                      21409,31479,15,250019,250018,142036,200,250029,200],
               Int64[0,250018,2672,200,13940,265,48700,2605,485,8836,293,
                      329,13313,93167,2003,13,330,15598,501,317,1027,20413,
                      15,250019,250018,142036,200,250029,200]]
    @testset "full Horizon prompt/attention buckets" begin
        for (i, prompt) in enumerate(prompts)
            # Cross both 32- and 64-slot boundaries during ordinary generation.
            expected = generate(reference, prompt; maxnew=52, bucketed=false, verbose=true)
            println("BUCKET_REFERENCE prompt=$i tokens=$expected")
            flush(stdout)
            for iteration in 1:2
                elapsed = @elapsed actual = generate(recorded, prompt; maxnew=52, verbose=true)
                println("BUCKET_GENERATION prompt=$i iteration=$iteration exact=$(actual == expected) total_s=$elapsed")
                flush(stdout)
                @test actual == expected
                println("BUCKET_MEMORY pool_gib=",
                    Mantle.reserved(Mantle.pool(recorded.model.device)) / 2.0^30)
                flush(stdout)
            end
        end
    end
    nothing
end

check_horizon_buckets(bucket_horizon)
