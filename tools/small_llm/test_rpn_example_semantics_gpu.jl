isdefined(@__MODULE__,:RPNExampleSemanticsGPU)||include("rpn_example_semantics_gpu.jl")
using Test
import Mantle

function test_rpn_example_semantics_gpu(device)
    R=RPNExampleSemantics;G=RPNExampleSemanticsGPU
    # Public generic opcode fixtures, independent of all stencil task formulas.
    programs=[Int32[a,17] for a in 1:9]
    append!(programs,[Int32[1,10,17],Int32[1,11,17]])
    append!(programs,[Int32[1,8,a,17] for a in 12:16])
    @testset "Public-example shared CPU/device numeric state" begin
        for n in (1,3,65),actions in programs
            x=Float32[Float32(i%7-3)*0.25f0 for i in 1:n];features=R.features(x)
            expected=if length(actions)==2
                action=first(actions)
                action<=5 ? vec(features[action,:]) : fill(action==6 ? 0f0 : action==7 ? 1f0 : action==8 ? 2f0 : 0.5f0,n)
            else
                action=actions[end-1]
                action==10 ? -x : action==11 ? x.*x : action==12 ? x.+2f0 :
                    action==13 ? x.-2f0 : action==14 ? x.*2f0 : action==15 ? min.(x,2f0) : max.(x,2f0)
            end
            buffers=Any[];upload(x)=(b=Mantle.Buffer(device,x);push!(buffers,b);b)
            plan=nothing
            try
                stack=upload(fill(-999f0,8n+7));f=upload(vec(features));status=upload(zeros(Int32,n+7))
                target=upload(expected);out=upload(fill(Int32(-9),n+7))
                graph=Mantle.Graph(device);depth=Int32(0)
                for action in actions
                    Mantle.dispatch!(graph,G.apply_action!,
                        (stack,f,status,action,depth,Int32(8),Int32(n)),n;group=64)
                    depth+=action<=9 ? Int32(1) : action<=11 || action==17 ? Int32(0) : Int32(-1)
                end
                Mantle.dispatch!(graph,G.check_examples!,
                    (out,stack,target,status,depth,Int32(8),0f0,0f0,Int32(n)),n;group=64)
                plan=Mantle.record!(Mantle.Plan(graph;budget=Inf))
                for replay in 1:2
                    stack[:]=fill(-999f0,8n+7);status[:]=zeros(Int32,n+7);out[:]=fill(Int32(-9),n+7)
                    before=Mantle.submissions(device);Mantle.run!(plan)
                    @test Mantle.submissions(device)-before==1
                    actual=Array(stack);flags=Array(out);statuses=Array(status)
                    @test actual[1:8:8n]==expected
                    @test actual[8n+1:end]==fill(-999f0,7)
                    @test flags[1:n]==ones(Int32,n)
                    @test flags[n+1:end]==fill(Int32(-9),7)
                    @test all(iszero,statuses)
                end
            finally
                plan===nothing||Mantle.free!(plan)
                foreach(Mantle.free!,reverse(buffers))
            end
        end
        # No successful finish on wrong examples, nonfinite features, overflow
        # or underflow. A failed earlier action stays failed after later pushes.
        for (input,expected,actions) in (
                (Float32[1],Float32[2],Int32[1,17]),
                (Float32[Inf],Float32[1],Int32[1,7,17]),
                (Float32[floatmax(Float32)],Float32[1],Int32[1,11,17]),
                (Float32[1],Float32[1],Int32[12,7,17]))
            buffers=Any[];upload(x)=(b=Mantle.Buffer(device,x);push!(buffers,b);b)
            plan=nothing
            try
                stack=upload(fill(-999f0,15));f=upload(fill(only(input),5));status=upload(Int32[0])
                target=upload(expected);out=upload(Int32[-9,-9])
                graph=Mantle.Graph(device);depth=Int32(0)
                for action in actions
                    Mantle.dispatch!(graph,G.apply_action!,
                        (stack,f,status,action,depth,Int32(8),Int32(1)),1;group=64)
                    depth=action<=9 ? depth+Int32(1) : action<=11 || action==17 ? depth : max(Int32(0),depth-Int32(1))
                end
                Mantle.dispatch!(graph,G.check_examples!,
                    (out,stack,target,status,Int32(1),Int32(8),0f0,0f0,Int32(1)),1;group=64)
                plan=Mantle.record!(Mantle.Plan(graph;budget=Inf));before=Mantle.submissions(device)
                Mantle.run!(plan)
                @test Mantle.submissions(device)-before==1
                @test Array(out)==Int32[0,-9]
                @test Array(stack)[9:end]==fill(-999f0,7)
            finally
                plan===nothing||Mantle.free!(plan)
                foreach(Mantle.free!,reverse(buffers))
            end
        end
    end
end
