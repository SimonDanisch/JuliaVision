# Run after loading `prefill_model` with record=false (see the prefill check).
using HorizonRunner, DNNKernels, Test

function check_generation_replay(reference)
    m = reference.model
    recordedmodel = DNNKernels.Model(m.graphs, m.weights, m.backend,
        m.memevery, m.memframes, m.topk; record=true,
        record_maxpasses=m.record_maxpasses)
    recorded = Horizon32B(reference.backend, recordedmodel, reference.maxlen,
                         reference.prefill, reference.vocab)
    prompts = [Int64[0,250018,2672,200,46348,803,2853,18147,1094,293,
                       21409,31479,15,250019,250018,142036,200,250029,200],
               Int64[0,250018,2672,200,13940,265,48700,2605,485,8836,293,
                       329,13313,93167,2003,13,330,15598,501,317,1027,20413,
                       15,250019,250018,142036,200,250029,200]]
    @testset "full Horizon repeated generation" begin
        for (i, prompt) in enumerate(prompts)
            expected = generate(reference, prompt; maxnew=12, verbose=true)
            for iteration in 1:2
                actual = generate(recorded, prompt; maxnew=12, verbose=true)
                println("GENERATION prompt=", i, " iteration=", iteration,
                        " exact=", actual==expected, " tokens=", actual)
                flush(stdout)
                @test actual == expected
            end
        end
        plans = [v for (k,v) in recordedmodel.scratch if k isa Tuple &&
                 !isempty(k) && first(k) === :mantleplan]
        @test length(plans) == 2
        @test count(p -> p.plan.recording isa Mantle.RecordingSequence, plans) == 1
    end
    recorded
end

using Mantle
recorded_horizon = check_generation_replay(prefill_model)
nothing
