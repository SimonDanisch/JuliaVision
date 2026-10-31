using Test, LinearAlgebra
import Mantle, KernelAbstractions

function test_qwen35_prefill(model,metadata;kv_type=Float32,attention=:split)
    ids=repeat(Int.(metadata.input_ids),6)
    @testset "Qwen batched prefill preserves decode state" begin
        for (count,chunk,prefix) in ((17,8,0),(19,8,3),(35,16,0))
            batched=Qwen35Experiment.session(model;capacity=128,debug_layers=true,kv_type,attention)
            reference=nothing
            try
                reference=Qwen35Experiment.session(model;capacity=128,debug_layers=true,kv_type,attention)
                for token in ids[1:prefix]
                    Qwen35Experiment.step!(batched,token)
                    Qwen35Experiment.step!(reference,token)
                end
                prompt=ids[prefix+1:prefix+count]
                Qwen35Experiment.prefill!(batched,prompt;chunk)
                for token in prompt
                    Qwen35Experiment.step!(reference,token)
                end
                @test batched.position==reference.position==prefix+count
                @test Array(batched.logits) ≈ Array(reference.logits) atol=2f-4 rtol=1f-4
                @test Qwen35Experiment.greedy(batched)==Qwen35Experiment.greedy(reference)
                for il in keys(reference.recurrent)
                    for (got,want) in zip(batched.recurrent[il],reference.recurrent[il])
                        @test Array(got) ≈ Array(want) atol=2f-4 rtol=1f-4
                    end
                end
                for il in keys(reference.kv)
                    for (got,want) in zip(batched.kv[il],reference.kv[il])
                        @test Array(got) ≈ Array(want) atol=2f-4 rtol=1f-4
                    end
                end
                for (got,want) in zip(batched.layer_outputs,reference.layer_outputs)
                    @test Array(got) ≈ Array(want) atol=2f-4 rtol=1f-4
                end
                first=batched.position
                Qwen35Experiment.decode!(batched,4)
                Qwen35Experiment.decode!(reference,4)
                @test Array(batched.predictions)[first:first+3]==Array(reference.predictions)[first:first+3]
                @test Array(batched.logits) ≈ Array(reference.logits) atol=2f-4 rtol=1f-4
                @test_throws ArgumentError Qwen35Experiment.prefill!(batched,[0];chunk=0)
                @test_throws ArgumentError Qwen35Experiment.prefill!(batched,[-1])
            finally
                reference===nothing || close(reference)
                close(batched)
            end
        end
    end
    nothing
end
