"""A declarative frontend for handwritten neural operators.

`standardop!` records operations of the shared IR (the kinds imported graphs
use), `semanticop!`/`semantickernel!` record opaque operations with explicit
tensor operands, including in-place/multi-output state changes. `emitsemantic!`
runs the shared pattern and fusion passes before lowering into the underlying
Mantle graph. Operation lowerings are ordinary `emitop!` methods; arbitrary
callbacks are not graph operations.
"""
mutable struct SemanticGraph
    dev::M.Device
    mantle::M.Graph
    name::String
    buffers::Dict{String,Buffer}
    order::Vector{String}
    ops::Vector{Op}
    bindings::Dict{String,Any}
    current::IdDict{Any,String}
    weights::IdDict{Any,String}
    # Packed/quantized weights by key, lowered through the emitter's weight table.
    objects::Dict{String,Any}
    # The caller tensor root behind every bound version, for the hazard check.
    roots::Dict{String,Any}
    emitted::Bool
end
SemanticGraph(dev::M.Device;name="handwritten_neural") = SemanticGraph(dev,M.Graph(dev),name,
    Dict{String,Buffer}(),String[],Op[],Dict{String,Any}(),IdDict{Any,String}(),
    IdDict{Any,String}(),Dict{String,Any}(),Dict{String,Any}(),false)

M.Transient.Buffer(g::SemanticGraph,T::Type,dims)=M.Transient.Buffer(g.mantle,T,dims)

const SemanticTensor = Union{M.Buffer,M.TransientBuffer,M.ResourceView,M.GPURef}
semanticroot(x)=x
semanticroot(x::M.ResourceView)=semanticroot(x.parent)
semanticbinding(x)=x
semanticbinding(x::M.GPURef)=M.viewof(x,(1,))
semanticdims(x)=size(x)
semanticdims(x::M.GPURef)=(1,)
# The logical element type a weight object stands for in the IR.
semanticdtype(x::SemanticTensor)=eltype(x)
semanticdtype(::KQuantMatrix)=Float32

"""A tensor owned by the semantic graph. Only standard operations produce one;
fusion may remove it, otherwise the emitter gives it transient storage."""
struct SemanticValue
    id::String
    dims::Dims
    dtype::DataType
end
Base.size(v::SemanticValue)=v.dims
Base.eltype(v::SemanticValue)=v.dtype
Base.length(v::SemanticValue)=prod(v.dims)
Base.ndims(v::SemanticValue)=length(v.dims)

semanticconstant(x)=isbits(x) || x===nothing || x isa AbstractString
semanticconstant(x::Tuple)=all(semanticconstant,x)
semanticconstant(x::Vector)=all(semanticconstant,x)
semanticconstant(x::AbstractDict)=all(semanticconstant(k) && semanticconstant(v) for (k,v) in x)

"""Register weights by key before their first use. Device tensors are bound
directly; other weight objects (e.g. `KQuantMatrix`) reach the emitter's weight
table, so standard products select their packed lowerings."""
function semanticweights!(g::SemanticGraph,weights::AbstractDict)
    g.emitted && error("Cannot add weights to an emitted semantic graph")
    for (name,x) in weights
        root=x isa SemanticTensor ? semanticroot(x) : x
        haskey(g.current,root) && throw(ArgumentError("Declare weights before their first operation"))
        g.weights[root]=String(name)
        x isa SemanticTensor || (g.objects[String(name)]=x)
    end
    g
end

function semanticvalue!(g::SemanticGraph,root;write=false,binding=root)
    old=get(g.current,root,nothing)
    !write && old!==nothing && return old
    id="v$(length(g.order)+1)";key=get(g.weights,root,"")
    kind=write ? :transient : isempty(key) ? :external : :weight
    g.buffers[id]=Buffer(id,kind,Any[reverse(semanticdims(binding))...],eltype(root),key,
        (0,0),"","",Dict{String,Any}())
    push!(g.order,id);g.bindings[id]=semanticbinding(binding);g.current[root]=id
    kind===:weight || (g.roots[id]=root)
    id
end

# A weight object has no binding: its key names it in the emitter's weight table.
function semanticweight!(g::SemanticGraph,x)
    old=get(g.current,x,nothing)
    old===nothing || return old
    id="v$(length(g.order)+1)"
    g.buffers[id]=Buffer(id,:weight,Any[reverse(size(x))...],semanticdtype(x),g.weights[x],
        (0,0),"","",Dict{String,Any}())
    push!(g.order,id);g.current[x]=id
    id
end

struct SemanticTensorArg
    input::Int
    output::Int # -2: input, -1: single output, >=0: tuple output index
    dims::Tuple
    offset::Int
    view::Bool
end
function semanticarg!(g::SemanticGraph,x::SemanticTensor,ins;output=-2)
    root=semanticroot(x)
    i=if output>=-1
        0
    else
        id=semanticvalue!(g,root);j=findfirst(==(id),ins)
        j===nothing ? (push!(ins,id);length(ins)) : j
    end
    SemanticTensorArg(i,output,semanticdims(x),x isa M.ResourceView ? x.offset : 0,x isa M.ResourceView)
end
# Opaque operations may read graph-owned values, never write them.
function semanticarg!(g::SemanticGraph,x::SemanticValue,ins;output=-2)
    output==-2 || throw(ArgumentError("Opaque operations write caller tensors, not graph values"))
    j=findfirst(==(semanticinput!(g,x)),ins)
    j===nothing && (push!(ins,x.id);j=length(ins))
    SemanticTensorArg(j,-2,x.dims,0,false)
end
semanticarg!(g,x::Tuple,ins;kw...)=map(a->semanticarg!(g,a,ins;kw...),x)
function semanticarg!(g,x,ins;kw...)
    isbits(x) || x===nothing || throw(ArgumentError(
        "Semantic argument $(typeof(x)) is not a declared tensor or value constant"))
    x
end

"""Declare one operation. `writes` names its written tensor argument positions.

Writes advance allocation-root versions, so different chunk views and residual
aliases cannot hide a dependency. Multiple written roots form a standard tuple
result with getitem views. Attributes contain constants/descriptors, never
untracked device arrays. Custom numerical semantics belong in `emitop!`.
"""
function semanticop!(g::SemanticGraph,kind::AbstractString,args::Tuple;
                     writes=(1,),attrs=Dict{String,Any}())
    g.emitted && error("Cannot append to an emitted semantic graph")
    semanticconstant(attrs) || throw(ArgumentError("Semantic attributes must not hide tensor resources"))
    isempty(writes) && throw(ArgumentError("A semantic operation must declare its results"))
    all(i->i isa Integer && 1<=i<=length(args),writes) || throw(ArgumentError("Invalid semantic output position"))
    roots=Any[]
    for i in writes
        args[i] isa SemanticTensor || throw(ArgumentError("Semantic output must be a declared tensor"))
        root=semanticroot(args[i])
        haskey(g.weights,root) && throw(ArgumentError("Semantic operation cannot overwrite a weight"))
        any(x->x===root,roots) || push!(roots,root)
    end
    ins=String[semanticvalue!(g,root) for root in roots]
    encoded=map(Tuple(enumerate(args))) do (i,x)
        output=i in writes ? (length(roots)==1 ? -1 : findfirst(r->r===semanticroot(x),roots)-1) : -2
        semanticarg!(g,x,ins;output)
    end
    out=if length(roots)==1
        semanticvalue!(g,only(roots);write=true)
    else
        id="v$(length(g.order)+1)"
        shapes=[Any[reverse(semanticdims(r))...] for r in roots]
        g.buffers[id]=Buffer(id,:transient,Any[],eltype(first(roots)),"",(0,0),"","",
            Dict{String,Any}("shapes"=>shapes,"dtypes"=>[eltype(r) for r in roots]))
        push!(g.order,id);g.bindings[id]=Tuple(semanticbinding(r) for r in roots)
        for (i,root) in enumerate(roots)
            child="$(id)_$(i-1)"
            g.buffers[child]=Buffer(child,:view,shapes[i],eltype(root),"",(0,0),id,"getitem",
                Dict{String,Any}("arg1"=>i-1))
            push!(g.order,child);g.bindings[child]=semanticbinding(root);g.current[root]=child
            g.roots[child]=root
        end
        id
    end
    a=Dict{String,Any}(attrs);a["arguments"]=encoded
    opid="op$(length(g.ops)+1)"
    any(op->op.id==opid,g.ops) && throw(ArgumentError("Duplicate semantic operation name $opid"))
    push!(g.ops,Op(opid,String(kind),ins,out,a))
    out
end

"""
    kernelresults(kernel, args) -> positions

The argument positions a device kernel writes. Each kernel declares its own
method, so `semantickernel!` refuses a kernel whose side effects are unknown.
"""
kernelresults(kernel,args::Tuple)=kernelresults(kernel)

"""Declare a device kernel as one opaque operation of the semantic IR. `kind`
names it in reports; its writes come from `kernelresults`."""
function semantickernel!(g::SemanticGraph,kernel,args::Tuple,ndrange;
                         kind::AbstractString=string(nameof(kernel)),group::Integer=256,
                         name::AbstractString=string(nameof(kernel)))
    semanticop!(g,"dnn.kernel",args;writes=kernelresults(kernel,args),
        attrs=Dict{String,Any}("kernel"=>kernel,"kind"=>String(kind),"extent"=>ndrange,
            "group"=>Int(group),"label"=>String(name)))
    nothing
end
# This package's own device kernels as opaque semantic operations.
for (kernel,writes) in ((q8act_kernel!,(1,2)),(mapped_q8_kernel!,(1,2,3)),(mapped_q8_parallel!,(1,2,3)),
        (swiglu_kernel!,(1,)),(depthwise_conv4_kernel!,(1,2)),(gated_delta_net_kernel!,(1,2)),(gated_delta_net_chunk_kernel!,(1,2)),
        (hadamard1024_kernel!,(1,)),(ptq1_coopmat_kernel!,(1,)),(ptq1_mul_mm_kernel!,(1,)),
        (ptq1_getrows_kernel!,(1,)))
    @eval kernelresults(::typeof($kernel))=$writes
end
for kernel in (values(Q4K_GEMV_KERNELS)...,values(Q6K_GEMV_KERNELS)...)
    @eval kernelresults(::typeof($kernel))=(1,)
end
function emitop!(ctx::EmitCtx,op::Op,::Val{Symbol("dnn.kernel")})
    a=op.attrs
    M.dispatch!(ctx.g,a["kernel"],semanticargument(ctx,op,a["arguments"]),a["extent"];
        group=a["group"],name=a["label"])
    nothing
end

# A whole tensor, possibly reshaped: standard operands carry no window offsets.
wholetensor(x)=true
wholetensor(x::M.ResourceView)=x.offset==0 && length(x)==length(semanticroot(x))

function semanticinput!(g::SemanticGraph,x::SemanticValue)
    b=get(g.buffers,x.id,nothing)
    b!==nothing && b.kind===:transient && !haskey(g.bindings,x.id) &&
        b.shape==Any[reverse(x.dims)...] && b.dtype===x.dtype ||
        throw(ArgumentError("Semantic value $(x.id) does not belong to this graph"))
    x.id
end
function semanticinput!(g::SemanticGraph,x::SemanticTensor)
    wholetensor(x) || throw(ArgumentError(
        "Standard operands are whole tensors; declare a window with an opaque operation"))
    id=semanticvalue!(g,semanticroot(x))
    want=Any[reverse(semanticdims(x))...]
    g.buffers[id].shape==want && return id
    vid="v$(length(g.order)+1)"
    g.buffers[vid]=Buffer(vid,:view,want,eltype(x),"",(0,0),id,"view.default",Dict{String,Any}())
    push!(g.order,vid)
    vid
end
semanticinput!(g::SemanticGraph,x)=haskey(g.weights,x) ? semanticweight!(g,x) :
    throw(ArgumentError("Standard operand $(typeof(x)) is not a tensor, graph value or registered weight"))

# Element-local kinds: each output element depends only on the same element of
# each input, so the result may overwrite an input of identical extent.
const SEMANTIC_INPLACE = Set(["add.Tensor","sub.Tensor","mul.Tensor","div.Tensor",
    "fused.swiglu",keys(UNARY_FUSED)...])

"""
    standardop!(g, aten, inputs; out, attrs) -> SemanticValue | nothing

Declare an operation of the shared DNNKernels IR: the kinds, operand order and
attributes an imported graph uses, so pattern and fusion passes can rewrite it.
Operands are whole tensors (or whole-tensor reshapes), values this graph
produced, or weights registered with `semanticweights!`.

`out=(T, dims)` makes a graph-owned value. A caller tensor as `out` is
overwritten whole and advances its version; it may also be an input only for
element-local kinds. `name` is the op id, and so the name of its passes.
"""
function standardop!(g::SemanticGraph,aten::AbstractString,inputs::Tuple;out,
                     attrs=Dict{String,Any}(),name::Union{Nothing,AbstractString}=nothing)
    g.emitted && error("Cannot append to an emitted semantic graph")
    semanticconstant(attrs) || throw(ArgumentError("Semantic attributes must not hide tensor resources"))
    opid=name===nothing ? "op$(length(g.ops)+1)" : String(name)
    any(op->op.id==opid,g.ops) && throw(ArgumentError("Duplicate semantic operation name $opid"))
    ins=String[semanticinput!(g,x) for x in inputs]
    value=nothing
    id=if out isa Tuple
        T,dims=out
        T isa DataType && dims isa Dims || throw(ArgumentError("Graph value requires (T, dims)"))
        vid="v$(length(g.order)+1)"
        g.buffers[vid]=Buffer(vid,:transient,Any[reverse(dims)...],T,"",(0,0),"","",Dict{String,Any}())
        push!(g.order,vid);value=SemanticValue(vid,dims,T)
        vid
    else
        out isa SemanticTensor && wholetensor(out) || throw(ArgumentError(
            "Standard result must be a whole caller tensor or (T, dims)"))
        root=semanticroot(out)
        haskey(g.weights,root) && throw(ArgumentError("Standard operation cannot overwrite a weight"))
        for x in inputs
            x isa SemanticTensor && semanticroot(x)===root || continue
            aten in SEMANTIC_INPLACE && semanticdims(x)==semanticdims(out) || throw(ArgumentError(
                "$aten cannot overwrite its own operand"))
        end
        semanticvalue!(g,root;write=true,binding=out)
    end
    push!(g.ops,Op(opid,String(aten),ins,id,Dict{String,Any}(attrs)))
    value
end

function semanticargument(ctx,op,x::SemanticTensorArg)
    r=x.output==-2 ? operand(ctx,op.ins[x.input]) :
      x.output==-1 ? dest(ctx) : dest(ctx,x.output)
    # A version a standard op wrote through a whole-tensor reshape is bound to
    # that reshape; an opaque reader of the whole tensor sees its own shape.
    x.view ? M.viewof(r,x.dims;offset=x.offset) : size(r)==x.dims ? r : M.viewof(r,x.dims)
end
semanticargument(ctx,op,x::Tuple)=map(a->semanticargument(ctx,op,a),x)
semanticargument(ctx,op,x)=x

function semanticgraph(g::SemanticGraph)
    produced=Set(op.out for op in g.ops)
    inputs=[id for id in g.order if !(id in produced) && haskey(g.bindings,id) &&
            g.buffers[id].kind!==:view]
    outputs=[id for (root,id) in g.current if root isa SemanticTensor && !(root isa M.TransientBuffer) &&
             (id in produced || g.buffers[id].kind===:view)]
    Graph(g.name,String[],inputs,sort!(outputs),g.buffers,g.order,g.ops)
end

# The version of a caller tensor that `id` reads, through reshapes and aliases.
function semanticversion(g::SemanticGraph,graph::Graph,id::AbstractString)
    while !haskey(g.roots,id)
        b=get(graph.buffers,id,nothing)
        (b===nothing || b.kind!==:view || isempty(b.of)) && return nothing
        id=b.of
    end
    id
end
# Bound tuple elements of each multi-output result.
function semanticchildren(g::SemanticGraph,graph::Graph)
    out=Dict{String,Vector{String}}()
    for (id,b) in graph.buffers
        b.kind===:view && b.viewop=="getitem" && haskey(g.roots,id) && push!(get!(out,b.of,String[]),id)
    end
    out
end
semanticwrites(g::SemanticGraph,children,op::Op)=
    haskey(g.roots,op.out) ? String[op.out] : get(children,op.out,String[])

"""
    semantichazards(g, ir, fused) -> op ids

Caller tensors bind every version of a root to the same storage. The shared
passes assume SSA storage, so they may move a read of one version past a later
write of its root (a producer fused into a consumer runs at the consumer), or
stop writing a bound version (a consumer folded into its producer). Returns the
original ops whose rewriting caused either.
"""
function semantichazards(g::SemanticGraph,ir::Graph,fused::Graph)
    kept=Set(op.id for op in fused.ops)
    absorbed=[op for op in ir.ops if !(op.id in kept)]
    irchildren=semanticchildren(g,ir);children=semanticchildren(g,fused)
    writers=Dict{String,String}()
    for op in ir.ops, w in semanticwrites(g,irchildren,op)
        writers[w]=op.id
    end
    current=Dict{UInt,String}()
    for (id,root) in g.roots
        haskey(writers,id) || (current[objectid(root)]=id)
    end
    written=Set{String}();culprits=Set{String}()
    for op in fused.ops
        for i in op.ins
            v=semanticversion(g,fused,i)
            v===nothing && continue
            get(current,objectid(g.roots[v]),nothing)==v && continue
            readers=[o.id for o in absorbed if any(j->semanticversion(g,ir,j)==v,o.ins)]
            union!(culprits,isempty(readers) ? [get(writers,v,op.id)] : readers)
        end
        for w in semanticwrites(g,children,op)
            current[objectid(g.roots[w])]=w;push!(written,w)
        end
    end
    for (v,id) in writers
        v in written || push!(culprits,id)
    end
    culprits
end

# Hidden behind a kind no pass matches, so no pass rewrites them.
pinops(ir::Graph,pinned)=Graph(ir.name,ir.symbols,ir.inputs,ir.outputs,ir.buffers,ir.order,
    [op.id in pinned ? Op(op.id,"semantic.pinned",op.ins,op.out,
        merge(op.attrs,Dict{String,Any}("semantic.kind"=>op.aten))) : op for op in ir.ops],ir.fusion)
function unpinops(g::Graph)
    ops=map(g.ops) do op
        op.aten=="semantic.pinned" || return op
        a=Dict{String,Any}(op.attrs);kind=pop!(a,"semantic.kind")
        Op(op.id,kind,op.ins,op.out,a)
    end
    Graph(g.name,g.symbols,g.inputs,g.outputs,g.buffers,g.order,ops,g.fusion)
end

"""
    semanticfusion(g) -> (graph, report)

The shared pattern and fusion passes over a semantic graph. Rewrites that would
reorder a read across a write of the same caller tensor are refused by pinning
the ops involved and running the passes again.
"""
function semanticfusion(g::SemanticGraph)
    ir=semanticgraph(g);pinned=Set{String}()
    while true
        fused,patterns=patternfusion(pinops(ir,pinned))
        fused,generic=genericfusion(fused)
        fused=unpinops(fused)
        culprits=semantichazards(g,ir,fused)
        isempty(culprits) && return fused,(;patterns...,generic...,pinned=length(pinned))
        issubset(culprits,pinned) && error(
            "Semantic fusion hazard on pinned operations $(sort!(collect(culprits)))")
        union!(pinned,culprits)
    end
end

function emitsemantic!(g::SemanticGraph)
    g.emitted && error("Semantic graph was already emitted")
    graph,report=semanticfusion(g)
    emitgraph!(g.mantle,graph,g.objects,(;);inputs=g.bindings)
    g.emitted=true
    (;graph,report,mantle=g.mantle)
end

# Activation preparation participates in the IR in declaration order. Emitting
# it eagerly into `g.mantle` would put quantization before its semantic producer.
activationbuffer(g::SemanticGraph,T,dims)=M.Transient.Buffer(g.mantle,T,dims)
const SEMANTIC_ACTIVATION_KERNELS=(("q8",q8act_kernel!,(1,2)),
    ("mapped_q8",mapped_q8_kernel!,(1,2,3)),
    ("parallel_mapped_q8",mapped_q8_parallel!,(1,2,3)))
function activationdispatch!(g::SemanticGraph,k,args,n;group,name)
    i=findfirst(t->t[2]===k,SEMANTIC_ACTIVATION_KERNELS)
    i===nothing && error("Unregistered activation preparation kernel")
    kind,_,writes=SEMANTIC_ACTIVATION_KERNELS[i]
    semanticop!(g,"dnn.$kind",args;writes,
        attrs=Dict{String,Any}("extent"=>n,"group"=>group,"label"=>String(name)))
    nothing
end
for (kind,kernel,_) in SEMANTIC_ACTIVATION_KERNELS
    tag=Symbol("dnn.$kind")
    @eval function emitop!(ctx::EmitCtx,op::Op,::Val{$(QuoteNode(tag))})
        a=op.attrs;args=semanticargument(ctx,op,a["arguments"])
        M.dispatch!(ctx.g,$kernel,args,a["extent"];group=a["group"],name=a["label"])
        nothing
    end
end
