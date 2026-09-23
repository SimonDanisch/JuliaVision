"""
The three ATen ops Hunyuan3D-2.1's shape denoiser needs and nothing else here
did: `_fused_rms_norm`, `topk` and `scatter.src`.

That is the whole new-op surface of a 3.05B-parameter diffusion transformer —
931 ops in the exported graph, 928 of which the runtime already had. Each of the
three is asserted against its definition written out, not against a second
implementation, because the failure mode for all three is an argument in the
wrong position or a dim mapped the wrong way round, and a second implementation
written from the same misreading agrees with the first.

The dim mapping is the thing to be suspicious of. `readsafetensors` reverses
extents, so a torch `(T, E)` tensor is a Julia `(E, T)` array and torch's `dim=1`
is Julia's `1` — via `jdim`, which is `n - d`, not `d + 1`. Both `topk` and
`scatter` take a `dim`, and getting it wrong on a square test tensor is
invisible. Nothing here is square.

## The properties each case exists for

**topk** — ties. `topk_body` selects by the key `(value descending, index
ascending)`, so equal values resolve to the lower index, which is what torch
returns. A row of all-equal values makes every implementation return *some*
answer; only the right one returns `0, 1`.

**scatter** — that it is out of place, and that it reads its index operand
through `Int(...)`. `topk` above hands back indices in the *value* dtype rather
than as `int64` (see its docstring), so the index arriving at a scatter is
routinely a float array, and a method signature that demanded integers would
work in a unit test and fail on the graph that motivated it.

**_fused_rms_norm** — the accumulator. Squares of Float16 values above 256
exceed Float16's 65504, so a mean-square accumulated at the operand's own
precision saturates to `Inf` and the whole normalisation group comes back as
zeros — silently, with no NaN to trip over. Hunyuan3D normalises 128-element
attention heads in fp16 and its activations do reach that range. The last case
here is a direct probe of it: values at 300, whose squares are 90000.
"""

using Test, DNNKernels, Mantle
const DK = DNNKernels

# `shape` is in TORCH order, as `loadgraph` produces it.
tbuf(id, kind, shape, dtype; key = "", of = "", viewop = "", attrs = Dict{String,Any}()) =
    DK.Buffer(id, kind, Any[shape...], dtype, key, (0, 0), of, viewop, attrs)

"""A `getitem` view onto element `i` (0-based) of a tuple-valued op."""
titem(id, parent, i, shape, dtype) =
    tbuf(id, :view, shape, dtype; of = parent, viewop = "getitem.default",
         attrs = Dict{String,Any}("arg1" => i))

"""
The output buffer of a TUPLE-valued op, carrying the per-element `shapes` and
`dtypes` an export files on one.

The `getitem` views name which elements are read; this is what says how many
there are. An element nothing reads is given `nothing` for its dtype and gets no
storage at all, which is how sdpa's philox seed and offset are declared.

A declaration needs this and an interpreted run did not: `execute!` put the
whole tuple in a table under the op id, so a parent with no element metadata
worked there and has no destination here.
"""
ttuple(id, kind, shapes, dtypes) =
    DK.Buffer(id, kind, Any[], Float32, "", (0, 0), "", "",
              Dict{String,Any}("shapes" => Any[Any[s...] for s in shapes],
                               "dtypes" => Any[dtypes...]))

"""
Declare a one-op graph, run it, and return its outputs as host arrays.

`planfor` resolves each output itself, which matters here because both ops with
a tuple result are reached through a `getitem` view, and a view is not a buffer
with storage of its own — the emit is what knows which parent it reads.

The reference every case below compares against is the op's DEFINITION written
out, so this runs the implementation under test exactly once. That is the point
of the file: a second implementation written from the same misreading of an
argument position agrees with the first.
"""
function run1(bufs, ops, inputs, outputs, weights = Dict{String,Any}())
    ins = Dict{String,Any}(inputs)
    inids = collect(keys(ins))
    g = DK.Graph("t", String[], inids, outputs,
                 bufs, collect(keys(bufs)), ops, Vector{Vector{String}}())
    backend = Mantle.LavaBackend()
    args = Tuple(DK.toback(backend, ins[id]) for id in inids)
    plan = DK.planfor(Mantle.todevice(backend), g, weights, NamedTuple())
    res = Dict{String,Any}(o => Array(v)
                           for (o, v) in zip(outputs, DK.replay!(plan, "t", args)))
    Mantle.free!(plan.plan)
    res
end

# ─────────────────────────────────────────────────────────────────── topk

"""
`aten::topk(self, k, dim=-1, largest=true)` written out, on one row.

Sorts by `(value descending, index ascending)` and takes the first `k`. This is
the definition the kernel implements, spelled as a sort so that it shares no
structure with the bounded scan `topk_body` performs.
"""
function topk_ref(row::AbstractVector, k::Int)
    p = sortperm(eachindex(row); by = i -> (-row[i], i))[1:k]
    (row[p], p .- 1)          # torch indices are 0-based
end

@testset "topk.default" begin
    T, E, k = 5, 8, 2         # nothing square: a wrong dim would reshape or throw
    torch = Float32[(i * 13 + j * 7) % 11 for i in 1:T, j in 1:E]   # (T, E) torch
    # ... plus one row of exact ties, which is the only case that pins the
    # index-ordering half of the contract.
    torch[3, :] .= 4f0
    x = permutedims(torch)                                          # (E, T) Julia

    bufs = Dict{String,DK.Buffer}(
        "x"  => tbuf("x", :external, (T, E), Float32),
        "tk" => ttuple("tk", :transient, ((T, k), (T, k)), (Float32, Float32)),
        "v"  => titem("v", "tk", 0, (T, k), Float32),
        "i"  => titem("i", "tk", 1, (T, k), Float32))
    ops = [DK.Op("tk", "topk.default", ["x"], "tk",
                 Dict{String,Any}("arg1" => k, "arg2" => -1,
                                  "arg3" => true, "arg4" => false))]
    out = run1(bufs, ops, ("x" => x,), ["v", "i"])
    vals, inds = Array(out["v"]), Array(out["i"])

    @test size(vals) == (k, T)          # Julia-side: k fastest, as torch (T, k) reverses
    @test size(inds) == (k, T)
    for t in 1:T
        wv, wi = topk_ref(view(torch, t, :), k)
        @test vals[:, t] == wv
        @test inds[:, t] == Float32.(wi)
    end
    # Stated on its own, because a scan that happens to visit indices in order
    # passes the loop above by luck on every non-tied row.
    @test inds[:, 3] == Float32[0, 1]   # all-tied row -> the two lowest indices

    @testset "largest=false is refused, not silently answered" begin
        ops2 = [DK.Op("tk", "topk.default", ["x"], "tk",
                      Dict{String,Any}("arg1" => k, "arg2" => -1,
                                       "arg3" => false, "arg4" => false))]
        @test_throws Exception run1(bufs, ops2, ("x" => x,), ["v", "i"])
    end
end

# ────────────────────────────────────────────────────────────────── scatter

@testset "scatter.src" begin
    T, E, k = 5, 8, 2
    self_t = Float32[100i + j for i in 1:T, j in 1:E]     # (T, E) torch
    idx_t = Float32[(i + 2j) % E for i in 1:T, j in 1:k]  # (T, k), 0-based
    src_t = Float32[-(i * 10 + j) for i in 1:T, j in 1:k]
    # Distinct per row: `scatter` without a `reduce` is undefined when two
    # entries collide, so a test that let them would be asserting on luck.
    for i in 1:T
        idx_t[i, 2] = (idx_t[i, 1] + 1) % E
    end

    bufs = Dict{String,DK.Buffer}(
        "s"   => tbuf("s", :external, (T, E), Float32),
        "i"   => tbuf("i", :external, (T, k), Float32),
        "v"   => tbuf("v", :external, (T, k), Float32),
        "out" => tbuf("out", :transient, (T, E), Float32))
    ops = [DK.Op("out", "scatter.src", ["s", "i", "v"], "out",
                 Dict{String,Any}("arg1" => 1))]
    inputs = ("s" => permutedims(self_t), "i" => permutedims(idx_t),
              "v" => permutedims(src_t))
    got = permutedims(Array(run1(bufs, ops, inputs, ["out"])["out"]))   # back to torch

    want = copy(self_t)
    for i in 1:T, j in 1:k
        want[i, Int(idx_t[i, j]) + 1] = src_t[i, j]
    end
    @test got == want

    # Out of place: `self` is untouched, so anything else reading it still can.
    @test permutedims(Array(inputs[1][2])) == self_t

    @testset "an integer index array works too" begin
        bufs2 = copy(bufs)
        bufs2["i"] = tbuf("i", :external, (T, k), Int32)
        inputs2 = ("s" => permutedims(self_t), "i" => Int32.(permutedims(idx_t)),
                   "v" => permutedims(src_t))
        @test permutedims(Array(run1(bufs2, ops, inputs2, ["out"])["out"])) == want
    end
end

# ───────────────────────────────────────────────────────────── _fused_rms_norm

"""`x * rsqrt(mean(x^2) + eps) * γ` over the last torch axis, in Float64."""
function rmsnorm_ref(x_t::AbstractMatrix, γ::AbstractVector, eps)
    out = similar(x_t, Float64)
    for i in axes(x_t, 1)
        row = Float64.(view(x_t, i, :))
        r = 1 / sqrt(sum(abs2, row) / length(row) + eps)
        out[i, :] .= row .* r .* Float64.(γ)
    end
    out
end

@testset "_fused_rms_norm.default" begin
    N, C, ε = 6, 16, 1f-6
    γ = Float32[1 + 0.01f0 * c for c in 1:C]

    @testset "against the definition" begin
        x_t = Float32[sin(i * 0.7 + c * 0.3) * (1 + 0.1c) for i in 1:N, c in 1:C]
        bufs = Dict{String,DK.Buffer}(
            "x"  => tbuf("x", :external, (N, C), Float32),
            "g"  => tbuf("g", :weight, (C,), Float32; key = "g"),
            "rn" => ttuple("rn", :transient, ((N, C), (N, 1)), (Float32, nothing)),
            "o"  => titem("o", "rn", 0, (N, C), Float32))
        ops = [DK.Op("rn", "_fused_rms_norm.default", ["x", "g"], "rn",
                     Dict{String,Any}("arg1" => Any[C], "arg3" => ε))]
        got = permutedims(Array(run1(bufs, ops, ("x" => permutedims(x_t),), ["o"],
                                     Dict{String,Any}("g" => γ))["o"]))
        @test maximum(abs, got .- rmsnorm_ref(x_t, γ, ε)) < 1e-5
    end

    @testset "no γ" begin
        x_t = Float32[cos(i + c) for i in 1:N, c in 1:C]
        bufs = Dict{String,DK.Buffer}(
            "x"  => tbuf("x", :external, (N, C), Float32),
            "rn" => ttuple("rn", :transient, ((N, C), (N, 1)), (Float32, nothing)),
            "o"  => titem("o", "rn", 0, (N, C), Float32))
        ops = [DK.Op("rn", "_fused_rms_norm.default", ["x"], "rn",
                     Dict{String,Any}("arg1" => Any[C], "arg3" => ε))]
        got = permutedims(Array(run1(bufs, ops, ("x" => permutedims(x_t),), ["o"])["o"]))
        @test maximum(abs, got .- rmsnorm_ref(x_t, ones(Float32, C), ε)) < 1e-5
    end

    # The reason the accumulator is `Float32` and not `eltype(a)`. At 300 the
    # squares are 90000 and Float16 tops out at 65504, so an fp16 accumulation
    # gives `Inf`, `rsqrt(Inf)` is 0, and the group comes back all zeros with
    # nothing raised. That is not hypothetical: this is the dtype and roughly
    # the range Hunyuan3D's q/k norms see.
    @testset "Float16 whose squares overflow Float16" begin
        x_t = Float16[Float16(300) * Float16(1 + 0.01 * ((i + c) % 5)) for i in 1:N, c in 1:C]
        @test !isfinite(sum(abs2, Float16.(x_t[1, :])))     # the trap, confirmed live
        bufs = Dict{String,DK.Buffer}(
            "x"  => tbuf("x", :external, (N, C), Float16),
            "rn" => ttuple("rn", :transient, ((N, C), (N, 1)), (Float16, nothing)),
            "o"  => titem("o", "rn", 0, (N, C), Float16))
        ops = [DK.Op("rn", "_fused_rms_norm.default", ["x"], "rn",
                     Dict{String,Any}("arg1" => Any[C], "arg3" => ε))]
        got = permutedims(Array(run1(bufs, ops, ("x" => permutedims(x_t),), ["o"])["o"]))
        @test all(isfinite, got)
        @test !all(iszero, got)
        # Float16 output, so the tolerance is the dtype's, not the method's.
        @test maximum(abs, Float64.(got) .- rmsnorm_ref(x_t, ones(Float16, C), ε)) < 2e-3
    end
end
