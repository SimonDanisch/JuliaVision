"""
Graph loading and shape resolution.

Shapes arrive symbolic in `h` and `w` (the stride-16 plane). Nothing here is a
type parameter: resolution is resolved by `evalshape` into plain integers at
instantiate time, so changing it never invokes the Julia compiler.
"""

struct Buffer
    id::String
    kind::Symbol            # :weight :external :transient :view :host
    shape::Vector{Any}      # ints and symbolic strings, in torch order
    dtype::DataType
    key::String             # weight only: name in weights.safetensors
    live::Tuple{Int,Int}    # transient only: op indices, inclusive
    of::String              # view only: parent buffer id
    viewop::String          # view only: which aten op produced it
    attrs::Dict{String,Any}
end

struct Op
    id::String
    aten::String
    ins::Vector{String}
    out::String
    attrs::Dict{String,Any}
    # `runop!` dispatches on `Val(Symbol(aten))`. Building that per call meant
    # interning a Symbol — a hash and a global-table lookup — for every op on
    # every step, which showed up in the profile as `_Symbol` + `hash`. The name
    # is fixed when the graph is loaded, so the tag is too. Dispatch on it is
    # still dynamic, exactly as before; only the interning is gone.
    tag::Val
    Op(id, aten, ins, out, attrs) = new(id, aten, ins, out, attrs, Val(Symbol(aten)))
end

struct Graph
    name::String
    symbols::Vector{String}
    inputs::Vector{String}
    outputs::Vector{String}
    buffers::Dict{String,Buffer}
    order::Vector{String}   # buffer ids in declaration order
    ops::Vector{Op}
    # Inductor's fusion decisions for this graph, as lists of op ids that its
    # codegen put in one Triton kernel (tools/dump_plan.py). Advisory: dropping
    # it changes speed, never results. Empty when no plan has been merged.
    fusion::Vector{Vector{String}}
end

# Every rewrite pass (`foldbatchnorm`, `foldrelu`, `hoistcasts`, `dropdead`)
# rebuilds a Graph; this keeps them from having to thread the plan through.
Graph(name, symbols, inputs, outputs, buffers, order, ops) =
    Graph(name, symbols, inputs, outputs, buffers, order, ops, Vector{String}[])

jget(o, k, default) = haskey(o, k) ? o[k] : default

"""
    plainattr(v)

Convert a JSON3 value to a plain Julia one, once, at load time.

Attributes are constant for the life of the graph, but `ints()` used to walk the
lazy `JSON3.Array` and allocate a fresh `Vector` *on every execution of every
op*. That measured 154 of ~430 profiler samples inside `execute!` — 36% of the
whole host-side dispatch cost — for re-deriving values that never change.
"""
plainattr(v::JSON3.Array) = [plainattr(x) for x in v]
plainattr(v::JSON3.Object) = Dict{String,Any}(String(k) => plainattr(x) for (k, x) in pairs(v))
plainattr(v) = v

attrdict(o) = haskey(o, :attrs) ?
    Dict{String,Any}(String(k) => plainattr(v) for (k, v) in pairs(o.attrs)) :
    Dict{String,Any}()

function Buffer(o)
    kind = Symbol(o.kind)
    shape = haskey(o, :shape) ? Any[x isa Integer ? Int(x) : String(x) for x in o.shape] : Any[]
    dtype = haskey(o, :dtype) ? DTYPE_NAMES[String(o.dtype)] : Float32
    live = haskey(o, :live) ? (Int(o.live[1]), Int(o.live[2])) : (0, 0)
    attrs = attrdict(o)
    # host scalars carry their symbolic size expression at the top level
    haskey(o, :expr) && (attrs["expr"] = o.expr)
    # A multi-output op (`native_layer_norm`, `_scaled_dot_product_*`,
    # `max_pool2d_with_indices`) declares `shapes`/`dtypes` instead of
    # `shape`/`dtype`. Dropping them, as this did, left `planslab` unable to size
    # those results, so every one of them allocated from the pool on every call —
    # a handful of buffers in MatAnyone, but 151 of them in SAM 2's encoder,
    # which is its whole transformer.
    if haskey(o, :shapes)
        attrs["shapes"] = Any[s === nothing ? nothing :
                              Any[x isa Integer ? Int(x) : String(x) for x in s] for s in o.shapes]
    end
    if haskey(o, :dtypes)
        # `get`, not indexing: a tuple element the runtime never reads can carry a
        # dtype nothing else in the graph uses — sdpa's philox seed and offset are
        # `uint64` — and an unrecognised one should mean "do not plan this
        # element", not "refuse to load the graph".
        attrs["dtypes"] = Any[d === nothing ? nothing :
                              get(DTYPE_NAMES, String(d), nothing) for d in o.dtypes]
    end
    Buffer(String(o.id), kind, shape, dtype,
           haskey(o, :key) ? String(o.key) : "",
           live,
           haskey(o, :of) && o.of !== nothing ? String(o.of) : "",
           haskey(o, :op) ? String(o.op) : "",
           attrs)
end

# `complex64`/`complex128` are torch's names for a *pair* of floats — the rotary
# embedding packs its sin/cos that way (`view_as_complex`), so a graph carrying
# RoPE fails to load without them.
const DTYPE_NAMES = Dict("float32" => Float32, "float64" => Float64, "float16" => Float16,
                         "int64" => Int64, "int32" => Int32, "bool" => Bool, "uint8" => UInt8,
                         "complex64" => ComplexF32, "complex128" => ComplexF64)

function Op(o)
    Op(String(o.id), String(o.aten), String[String(x) for x in o.ins_],
       String(o.out),
       attrdict(o))
end

"""
    loadgraph(path) -> Graph
"""
function loadgraph(path::AbstractString)
    j = JSON3.read(read(path, String))
    buffers = Dict{String,Buffer}()
    order = String[]
    for o in j.buffers
        b = Buffer(o)
        buffers[b.id] = b
        push!(order, b.id)
    end
    ops = Op[]
    for o in j.ops
        push!(ops, Op(String(o.id), String(o.aten),
                      String[String(x) for x in o["in"]], String(o.out),
                      attrdict(o)))
    end
    fusion = Vector{String}[]
    if haskey(j, :fusion_groups)
        for grp in j.fusion_groups
            push!(fusion, String[String(x) for x in grp.ops])
        end
    end
    Graph(String(j.name), String[String(s) for s in j.symbols],
          String[String(s) for s in j.inputs], String[String(s) for s in j.outputs],
          buffers, order, ops, fusion)
end

"""
    evalshape(shape, dims) -> NTuple{N,Int}

Resolve a symbolic torch-order shape against `dims = (h=..., w=...)` and return
it **reversed**, i.e. in the `(x, y, c, n)` order the arrays actually use.

The expressions torch.export emits are products, quotients and sums of the base
symbols; they are evaluated here rather than at kernel launch so no kernel ever
divides by a runtime value.
"""
function evalshape(shape, dims)
    out = Int[]
    for s in shape
        push!(out, s isa Integer ? Int(s) : evalexpr(String(s), dims))
    end
    Tuple(reverse(out))
end

function evalexpr(e::AbstractString, dims)
    ex = Meta.parse(e)
    evalexpr(ex, dims)
end

function evalexpr(ex, dims)
    ex isa Integer && return Int(ex)
    ex isa Symbol && return Int(getproperty(dims, ex))
    if ex isa Expr && ex.head === :call
        f = ex.args[1]
        as = [evalexpr(a, dims) for a in ex.args[2:end]]
        f === :+ && return sum(as)
        f === :* && return prod(as)
        f === :- && return length(as) == 1 ? -as[1] : as[1] - sum(as[2:end])
        # torch emits floor division for the downsampling paths
        (f === :/ || f === Symbol("//")) && return as[1] ÷ as[2]
        f === :% && return mod(as[1], as[2])
        # sympy prints these in function form
        f === :Mod && return mod(as[1], as[2])
        f === :Max && return maximum(as)
        f === :Min && return minimum(as)
        (f === :floor || f === :floordiv) && return length(as) == 1 ? as[1] : as[1] ÷ as[2]
        f === :ceiling && return length(as) == 1 ? as[1] : cld(as[1], as[2])
        f === :Pow && return as[1]^as[2]
        error("unsupported shape expression: $ex")
    end
    error("unsupported shape expression: $ex")
end
