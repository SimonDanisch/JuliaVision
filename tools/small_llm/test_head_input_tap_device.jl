include("head_input_tap.jl")
using Test

function test_head_input_tap_device(device)
    Q=SmallLLMChecks.Qwen35Experiment
    @testset "Read-only native head-input capture" begin
        width=512;outputs=7;owned=Any[];plans=Any[]
        upload(x)=begin buffer=Mantle.Buffer(device,x);push!(owned,buffer);buffer;end
        weights=Q.DenseWeight(upload(reshape(Float32[cos(i*.07) for i in 1:width*outputs],width,outputs)),outputs,width)
        input=upload(Float32[sin(i*.03) for i in 1:width])
        query=upload(zeros(Float32,width));direct=upload(zeros(Float32,outputs));tapped=upload(zeros(Float32,outputs))
        model=(;config=(;hidden_size=width),weights=Dict{String,Any}("lm_head"=>weights))
        try
            graph=Mantle.Graph(device);Q.linear!(graph,direct,weights,input;name="direct_head")
            plain=Mantle.record!(Mantle.Plan(graph;budget=Inf));push!(plans,plain)
            with_head_input_tap(model,query) do
                graph=Mantle.Graph(device);Q.linear!(graph,tapped,model.weights["lm_head"],input;name="tapped_head")
                hooked=Mantle.record!(Mantle.Plan(graph;budget=Inf));push!(plans,hooked)
                for scale in (1f0,-2f0)
                    values=scale.*Float32[sin(i*.03) for i in 1:width];input[:]=values
                    Mantle.run!(plain)
                    before=Mantle.submissions(device);Mantle.run!(hooked)
                    @test Mantle.submissions(device)-before==1
                    @test Array(query)==values
                    @test Array(tapped)==Array(direct)
                    @test all(isfinite,Array(tapped))
                end
            end
            @test model.weights["lm_head"]===weights
        finally
            foreach(Mantle.free!,reverse(plans));foreach(Mantle.free!,reverse(owned))
        end
    end
end
