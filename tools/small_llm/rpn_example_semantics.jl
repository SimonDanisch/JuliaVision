"""Shared numeric prefix evaluation for explicitly supplied PUBLIC examples.

Not a natural-language parser, completeness oracle or proof of correctness.
An example match must never replace independent hidden functional tests.
This module does not read task IDs, benchmark formulas or reference programs.
"""
module RPNExampleSemantics

"""Apply one admitted action for one example, with flattened column-major data.

features holds five rows/example: current, periodic previous/next, clamped
previous/next. stack holds max_depth rows/example. depth is BEFORE the action.
Returns zero on success, one on invalid input or nonfinite result. A rejected
operation leaves stack untouched; other examples are independent writers.
"""
@inline function step!(stack,features,example::Int32,action::Int32,depth::Int32,max_depth::Int32)
    example>0&&max_depth>=2&&0<=depth<=max_depth&&1<=action<=17||return Int32(1)
    example<=length(stack)÷max_depth&&example<=length(features)÷5||return Int32(1)
    example<=typemax(Int32)÷max_depth&&example<=typemax(Int32)÷5||return Int32(1)
    base=(example-Int32(1))*max_depth
    feature_base=(example-Int32(1))*Int32(5)
    # Check sizes before @inbounds; no overflow permitted by host admission.
    base+max_depth<=length(stack)&&feature_base+Int32(5)<=length(features)||return Int32(1)
    if action==17
        depth==1||return Int32(1)
        @inbounds return Int32(!isfinite(stack[base+Int32(1)]))
    elseif action<=9
        depth<max_depth||return Int32(1)
        value=if action<=5
            @inbounds features[feature_base+action]
        elseif action==6;0f0
        elseif action==7;1f0
        elseif action==8;2f0
        else;0.5f0
        end
        isfinite(value)||return Int32(1)
        @inbounds stack[base+depth+Int32(1)]=value
    elseif action<=11
        depth>=1||return Int32(1)
        @inbounds value=stack[base+depth]
        result=action==10 ? -value : value*value
        isfinite(result)||return Int32(1)
        @inbounds stack[base+depth]=result
    else
        depth>=2||return Int32(1)
        @inbounds left=stack[base+depth-Int32(1)]
        @inbounds right=stack[base+depth]
        isfinite(left)&&isfinite(right)||return Int32(1)
        result=if action==12;left+right
        elseif action==13;left-right
        elseif action==14;left*right
        elseif action==15;min(left,right)
        else;max(left,right)
        end
        isfinite(result)||return Int32(1)
        @inbounds stack[base+depth-Int32(1)]=result
    end
    Int32(0)
end

"""Finish admission from finite supplied examples; no vacuous empty-example pass."""
@inline function matches(stack,expected,example::Int32,depth::Int32,max_depth::Int32,atol::Float32,rtol::Float32)
    example>0&&depth==1&&max_depth>=2&&isfinite(atol)&&atol>=0&&isfinite(rtol)&&rtol>=0||return false
    example<=length(stack)÷max_depth&&example<=typemax(Int32)÷max_depth||return false
    slot=(example-Int32(1))*max_depth+Int32(1)
    slot<=length(stack)&&example<=length(expected)||return false
    @inbounds actual=stack[slot]
    @inbounds target=expected[example]
    isfinite(actual)&&isfinite(target)||return false
    # Float64 comparison prevents overflow of the error/tolerance arithmetic.
    abs(Float64(actual)-Float64(target))<=Float64(atol)+Float64(rtol)*abs(Float64(target))
end

function features(x::AbstractVector{Float32})
    isempty(x)&&throw(ArgumentError("examples must contain at least one input point"))
    length(x)<=typemax(Int32)÷64||throw(ArgumentError("example cohort too large"))
    all(isfinite,x)||throw(ArgumentError("nonfinite public example"))
    n=length(x);out=Matrix{Float32}(undef,5,n)
    for i in 1:n
        out[1,i]=x[i];out[2,i]=x[mod1(i-1,n)];out[3,i]=x[mod1(i+1,n)]
        out[4,i]=x[max(1,i-1)];out[5,i]=x[min(n,i+1)]
    end
    out
end

struct Bank
    features::Vector{Float32}
    expected::Vector{Float32}
    case_offsets::Vector{Int32}
    max_depth::Int32
    atol::Float32
    rtol::Float32
end

"""Validate/flatten caller-supplied example pairs, retaining per-case boundaries.

Periodic/clamped neighbors are computed separately within each input, never
across adjacent examples. The caller must freeze and expose these examples to
BOTH model arms, and must not construct them from private grading probes.
"""
function Bank(cases;max_depth=8,atol=0f0,rtol=0f0,max_points=256)
    max_depth isa Integer&&!(max_depth isa Bool)&&2<=max_depth<=64||throw(ArgumentError("invalid stack bound"))
    max_points isa Integer&&!(max_points isa Bool)&&1<=max_points<=65536||throw(ArgumentError("invalid example cap"))
    atol=Float32(atol);rtol=Float32(rtol)
    isfinite(atol)&&atol>=0&&isfinite(rtol)&&rtol>=0||throw(ArgumentError("invalid tolerances"))
    isempty(cases)&&throw(ArgumentError("public examples cannot be empty"))
    packed=Float32[];expected=Float32[];offsets=Int32[1]
    for case in cases
        x=case.input;y=case.output
        x isa AbstractVector{Float32}&&y isa AbstractVector{Float32}||throw(ArgumentError("Float32 example vectors required"))
        !isempty(x)&&length(x)==length(y)||throw(ArgumentError("invalid example shape"))
        length(x)<=max_points-length(expected)||throw(ArgumentError("public example cap exceeded"))
        all(isfinite,y)||throw(ArgumentError("nonfinite example target"))
        append!(packed,vec(features(x)));append!(expected,y);push!(offsets,Int32(length(expected)+1))
    end
    Bank(packed,expected,offsets,Int32(max_depth),atol,rtol)
end
end
