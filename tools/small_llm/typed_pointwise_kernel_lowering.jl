# Separate post-hoc compiler version; never substitute into frozen study scores.
module TypedPointwiseKernelLowering

"""Conservative serial-pointwise-loop lowering, not an arbitrary Julia compiler.

Admits one `for i=1:N` loop, Float32 output and Float32/Int32 vector arguments, Int/Int32 scalar
arguments, pure scalar expressions and exactly one `out[i]` store. No output
reads, loop-carried state, side effects, macros, nested control flow or existing
device intrinsics are admitted. Requires distinct input/output arrays and an
already in-bounds serial loop. These are preconditions, NOT inferred proofs.
Only ASTs are inspected here; generated code is never executed in this process.
"""
function lower(code::AbstractString)
    reject(reason)=(;accepted=false,reason,code="",certificate=nothing)
    ncodeunits(code)<=65536 || return reject("source too large")
    tree=Meta.parseall(code)
    statements(x)=x isa Expr && x.head in (:toplevel,:block) ? filter(y->!(y isa LineNumberNode),x.args) : Any[x]
    top=Any[]
    for statement in statements(tree)
        if statement isa Expr && statement.head in (:using,:import)
            sprint(Base.show_unquoted,statement) in ("using KernelInterface","import KernelInterface","import KernelInterface as KI") ||
                return reject("unknown import binding")
        else
            push!(top,statement)
        end
    end
    length(top)==1 || return reject("one function definition required")
    definition=only(top)
    definition isa Expr && definition.head in (:function,:(=)) || return reject("function syntax required")
    signature,body=definition.args
    signature isa Expr && signature.head==:call && first(signature.args)==:kernel! || return reject("kernel! only")
    arrays=Set{Symbol}();scalars=Set{Symbol}();arguments=Any[];elements=Dict{Symbol,Symbol}()
    for argument in signature.args[2:end]
        argument isa Expr && argument.head==:(::) && length(argument.args)==2 && first(argument.args) isa Symbol ||
            return reject("explicit simple argument types required")
        name,type=argument.args
        name in arrays || name in scalars ? (return reject("duplicate argument")) : nothing
        if type isa Expr && type.head==:curly && type.args[1] in (:Vector,:AbstractVector) && length(type.args)==2 && type.args[2] in (:Float32,:Int32)
            push!(arrays,name);elements[name]=type.args[2];push!(arguments,Expr(:(::),name,Expr(:curly,:AbstractVector,type.args[2])))
        elseif type in (:Int,:Int32)
            push!(scalars,name);push!(arguments,argument)
        else
            return reject("only Float32/Int32 vectors and signed Int/Int32 parameters supported")
        end
    end
    isempty(arguments) && return reject("missing output")
    output=first(first(arguments).args);output in arrays && elements[output]==:Float32 || return reject("first argument must be Float32 output vector")
    parent=filter(x->x!==:nothing && !(x isa Expr && x.head==:return && x.args==Any[:nothing]),statements(body))
    length(parent)==1 || return reject("one loop and optional return nothing required")
    loop=only(parent);loop isa Expr && loop.head==:for || return reject("one serial loop required")
    binding,loopbody=loop.args
    binding isa Expr && binding.head==:(=) && first(binding.args) isa Symbol || return reject("simple induction variable required")
    index,range=binding.args
    index in union(arrays,scalars) && return reject("induction variable shadows argument")
    range isa Expr && range.head==:call && range.args[1]==:(:) && length(range.args)==3 && range.args[2]===1 ||
        return reject("only range 1:N supported")
    limit=range.args[3]
    purecalls=Set((:+,:-,:*,:/,:div,:mod,:mod1,:min,:max,:clamp,:muladd,:Float32,:Int32,:Int))
    function pure(x,known;allow_arrays=true)
        x isa Symbol && return x in known
        x isa Number && return isfinite(x)
        x isa Expr || return false
        if x.head==:ref && length(x.args)==2
            return allow_arrays && first(x.args) in setdiff(arrays,Set([output])) && pure(last(x.args),known;allow_arrays)
        end
        x.head==:call && first(x.args) in purecalls && all(y->pure(y,known;allow_arrays),x.args[2:end])
    end
    pure(limit,scalars;allow_arrays=false) || return reject("loop limit must be pure scalar expression")
    known=union(scalars,Set([index]));stores=0
    for statement in statements(loopbody)
        statement===:nothing && continue
        statement isa Expr && statement.head==:(=) && length(statement.args)==2 || return reject("only local assignments and output store admitted")
        lhs,rhs=statement.args
        pure(rhs,known) || return reject("unknown/impure expression or loop-carried dependency")
        if lhs isa Symbol
            lhs in union(arrays,scalars,Set([index])) && return reject("cannot mutate arguments or induction variable")
            push!(known,lhs)
        elseif lhs isa Expr && lhs.head==:ref && lhs.args==Any[output,index]
            stores+=1
        else
            return reject("only direct output[induction] stores admitted")
        end
    end
    stores==1 || return reject("exactly one per-iteration output store required")
    lowered=Expr(:function,Expr(:call,:kernel!,arguments...),quote
        $index=KI.get_global_id().x
        if $index<=$limit
            $loopbody
        end
        nothing
    end)
    result="import KernelInterface as KI\n"*sprint(Base.show_unquoted,lowered)*"\n"
    (;accepted=true,reason="admitted disjoint pointwise subset",code=result,
        certificate=(;output=String(output),induction=String(index),stores_per_invocation=1,
            limit=sprint(Base.show_unquoted,limit),
            preconditions=["distinct input/output storage","original serial accesses are in bounds",
                "Float32 output and Float32/Int32 input vectors, Int/Int32 scalars, ordinary Base arithmetic bindings",
                "one-dimensional launch with unique one-based global x IDs, covering 1:N"],
            guarantee="under these preconditions, each valid invocation executes exactly one original iteration and writes only its unique output index; no claim about whether the original arithmetic solves the task"))
end
end

module CheckedTypedPointwiseLowering
import ..TypedPointwiseKernelLowering

"""Narrower admission guard for the separate typed-array lowering.

The expression whitelist also admits floating loop limits. Such a range can
have a different induction type, so a native-Int global index is not a general
semantics-preserving replacement. Reject those limits and shadowed API/math
bindings; the frozen original remains unchanged.
"""
function lower(code::AbstractString)
    result=TypedPointwiseKernelLowering.lower(code)
    result.accepted || return result
    reject(reason)=(;accepted=false,reason,code="",certificate=nothing)
    nodes=filter(x->!(x isa LineNumberNode) && !(x isa Expr && x.head in (:using,:import)),Meta.parseall(code).args)
    signature,body=only(nodes).args
    scalar_names=Set{Symbol}();bindings=Set{Symbol}()
    for arg in signature.args[2:end]
        name,type=arg.args;push!(bindings,name)
        type in (:Int,:Int32) && push!(scalar_names,name)
    end
    function collect_bindings(x)
        x isa Expr || return
        if x.head==:(=) && first(x.args) isa Symbol
            push!(bindings,first(x.args))
        end
        foreach(collect_bindings,x.args)
    end
    collect_bindings(body)
    reserved=Set((:KI,:Base,:nothing,:+,:-,:*,:/,:div,:mod,:mod1,:min,:max,:clamp,:muladd,:Float32,:Int32,:Int))
    isempty(intersect(bindings,reserved)) || return reject("shadowed runtime or arithmetic binding")
    integer_ops=Set((:+,:-,:*,:div,:mod,:mod1,:min,:max,:clamp,:Int32,:Int))
    function integer_limit(x)
        x isa Symbol && return x in scalar_names
        x isa Int && return true
        x isa Int32 && return true
        x isa Expr && x.head==:call && first(x.args) in integer_ops && all(integer_limit,x.args[2:end])
    end
    integer_limit(Meta.parse(result.certificate.limit)) || return reject("loop bound is not established Int/Int32 arithmetic")
    (;result...,certificate=(;result.certificate...,
        additional_admission="integer-only loop bound, native-Int induction for 1:N, no shadowed runtime/math bindings"))
end
end

