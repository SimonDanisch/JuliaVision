using Test, JSON3, LinearAlgebra
import DNNKernels
using .Qwen35Experiment

function verify_qwen35(model,dir;debug_layers=false)
    metadata=JSON3.read(read(joinpath(dir,"reference.json"),String))
    reference=DNNKernels.readsafetensors(joinpath(dir,"reference.safetensors"))
    s=Qwen35Experiment.session(model;capacity=256,debug_layers)
    rows=Any[]
    try
        @testset "Qwen native checkpoint reference" begin
            for (i,token) in enumerate(metadata.input_ids)
                got=Array(Qwen35Experiment.step!(s,token));want=reference["logits"][:,i]
                @test all(isfinite,got)
                row=(;step=i,maxabs=maximum(abs,got-want),relrms=norm(got-want)/norm(want),
                       greedy=argmax(got)-1,reference=argmax(want)-1)
                push!(rows,row)
                if model.quantize===:none
                    @test row.maxabs<5f-5
                    @test row.relrms<5f-6
                    @test row.greedy==row.reference
                end
            end
            if debug_layers && model.quantize===:none
                for (i,buffer) in enumerate(s.layer_outputs)
                    got=Array(buffer);want=reference["layer.$(i-1)"]
                    @test norm(got-want)/norm(want)<5f-6
                end
            end
        end
    finally
        close(s)
    end
    rows
end

# Deliberate readback in the reference path only, to validate device feedback
# against explicit token feeding before using the feedback path for timing.
function verify_qwen35_feedback(model,metadata;n=8)
    resident=Qwen35Experiment.session(model;capacity=256)
    reference=nothing
    try
        reference=Qwen35Experiment.session(model;capacity=256)
        @testset "Qwen device-resident greedy feedback" begin
            for token in metadata.input_ids
                Qwen35Experiment.step!(resident,token)
                Qwen35Experiment.step!(reference,token)
            end
            first_prediction=resident.position
            Qwen35Experiment.decode!(resident,n)
            expected=Int[]
            for _ in 1:n
                token=Qwen35Experiment.greedy(reference)
                push!(expected,token)
                Qwen35Experiment.step!(reference,token)
            end
            got=Int.(Array(resident.predictions)[first_prediction:first_prediction+n-1])
            @test got==expected
            @test resident.position==reference.position
            @test Array(resident.logits) ≈ Array(reference.logits) atol=1f-6 rtol=1f-6
            @test_throws ArgumentError Qwen35Experiment.decode!(resident,-1)
            @test_throws ArgumentError Qwen35Experiment.decode!(resident,resident.capacity)
        end
    finally
        reference===nothing || close(reference)
        close(resident)
    end
    nothing
end
