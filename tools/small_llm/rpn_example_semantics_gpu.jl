isdefined(@__MODULE__,:RPNExampleSemantics)||include("rpn_example_semantics.jl")
module RPNExampleSemanticsGPU
import ..RPNExampleSemantics
import KernelInterface as KI

function apply_action!(stack,features,status,action::Int32,depth::Int32,max_depth::Int32,n::Int32)
    example=Int32(KI.get_global_id().x)
    if example<=n
        @inbounds if status[example]==0
            status[example]=RPNExampleSemantics.step!(stack,features,example,action,depth,max_depth)
        end
    end
    nothing
end

function check_examples!(out,stack,expected,status,depth::Int32,max_depth::Int32,atol::Float32,rtol::Float32,n::Int32)
    example=Int32(KI.get_global_id().x)
    if example<=n
        @inbounds out[example]=status[example]==0 &&
            RPNExampleSemantics.matches(stack,expected,example,depth,max_depth,atol,rtol) ? Int32(1) : Int32(0)
    end
    nothing
end
end
