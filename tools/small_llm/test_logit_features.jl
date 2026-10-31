using Test
import Mantle
isdefined(@__MODULE__,:LogitFeatures) || include("logit_features.jl")

function test_logit_features(device)
    @testset "Shared output-projection features and bounded capture" begin
        for width in (3,257,512,513)
            vocabulary=2width+1;ids=Int32.(0:2:2width-2)
            original=Float32[sin(i*0.17)+0.001i for i in 1:vocabulary]
            feature=LogitFeatures.Features(device,vocabulary,ids)
            capture=LogitFeatures.Capture(device,width,2)
            logits=Mantle.Buffer(device,original);query=Mantle.Buffer(device,zeros(Float32,width));plan=nothing
            selected=Float64.(original[ids.+1]);centered=selected.-sum(selected)/width
            expected=centered./sqrt(sum(abs2,centered)/width)
            try
                graph=Mantle.Graph(device);LogitFeatures.declare!(graph,query,logits,feature)
                LogitFeatures.declare_capture!(graph,query,capture)
                plan=Mantle.record!(Mantle.Plan(graph;budget=Inf))
                Mantle.run!(plan)
                @test only(Array(capture.counter))==0
                @test all(iszero,Array(capture.history))
                @test Array(query) ≈ expected atol=2e-5 rtol=2e-4
                @test Array(logits)==original
                LogitFeatures.activate!(capture)
                before=Mantle.submissions(device);Mantle.run!(plan)
                @test Mantle.submissions(device)-before==1
                @test only(Array(capture.counter))==1
                @test Array(capture.history)[:,1] ≈ expected atol=2e-5 rtol=2e-4
                logits[:]=2f0.*original.+3f0;Mantle.run!(plan)
                @test only(Array(capture.counter))==2
                @test only(Array(capture.status))==0
                @test Array(capture.history)[:,2] ≈ expected atol=2e-5 rtol=2e-4
                saved=Array(capture.history);Mantle.run!(plan)
                @test only(Array(capture.counter))==2
                @test only(Array(capture.status))==1
                @test Array(capture.history)==saved
                LogitFeatures.reset!(capture);LogitFeatures.activate!(capture,false)
                Mantle.run!(plan)
                @test only(Array(capture.counter))==0
                @test only(Array(capture.status))==0
                for invalid in (fill(0.1f0,vocabulary),fill(NaN32,vocabulary),fill(Inf32,vocabulary))
                    logits[:]=invalid;Mantle.run!(plan)
                    @test all(isnan,Array(query))
                end
            finally
                plan===nothing || Mantle.free!(plan)
                close(feature);close(capture);Mantle.free!(logits);Mantle.free!(query)
            end
        end
        @test_throws ArgumentError LogitFeatures.Features(device,3,Int32[0,0])
        @test_throws ArgumentError LogitFeatures.Features(device,3,Int32[-1,2])
        @test_throws ArgumentError LogitFeatures.Features(device,3,Int32[0,3])
        @test_throws ArgumentError LogitFeatures.Features(device,3,Bool[false,true])
        @test_throws ArgumentError LogitFeatures.Capture(device,0,2)
        @test_throws ArgumentError LogitFeatures.Capture(device,2,0)
    end
end
