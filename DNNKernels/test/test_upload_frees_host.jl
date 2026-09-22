"""
A checkpoint is resident once, not twice.

`Model`'s upload loop drops its host copy of each tensor as that tensor lands on
the device, and the comment above it says what that is for: a model whose
weights do not fit twice. It was not working. The loop deletes from `host`, and
`host` was never the last reference — the dictionary each host pass was HANDED
stays in `Model`'s frame until `Model` returns, and it names every tensor in the
checkpoint. So did the caller's own dictionary.

Measured on Qwen-Image 2.1's denoiser, 6.9 GB of host weights against 8.3 GB
uploaded: `gc_live_bytes` came out of the upload loop at 7.11 GB with `host`
emptied and a full `GC.gc(true)` just run, and one more `GC.gc(true)` a frame
later — with `Model` gone — reported 176 MB. On an APU, where the host copy and
the device copy come out of the same memory, the staged pipeline peaked at
18.4 GB of anonymous memory plus GTT for a 6.9 GB model, and 9.2 GB once this
was fixed.

The assertion is reachability and not a byte count, because a byte count on a
toy model measures the runtime rather than the bug: a `WeakRef` to a host weight
must be dead once `Model` has returned and the collector has run. That is true
exactly when nothing in `Model`'s frame, and nothing in the caller's dictionary,
still names it.

Device-free and backend-agnostic: `Mantle.eachbackend()` because `LavaBackend`
exists only where Lava does, and what this pins is portable.
"""

using Test, DNNKernels, Mantle

const DKU = DNNKernels

ubuf(id, kind, shape, T, key = "") =
    DKU.Buffer(id, kind, Any[shape...], T, key, (0, 3), "", "", Dict{String,Any}())

"""`y = x .+ w` as a graph, with `n` weights so the upload loop has a list."""
function addgraph(n::Int, len::Int; T = Float32)
    keys_ = ["w$j" for j in 1:n]
    bufs = Dict{String,DKU.Buffer}("x" => ubuf("x", :external, (len,), T))
    ops = DKU.Op[]
    prev = "x"
    for (j, k) in enumerate(keys_)
        bufs[k] = ubuf(k, :weight, (len,), T, k)
        out = j == n ? "y" : "t$j"
        bufs[out] = ubuf(out, j == n ? :transient : :transient, (len,), T)
        push!(ops, DKU.Op("add$j", "add.Tensor", [prev, k], out, Dict{String,Any}()))
        prev = out
    end
    DKU.Graph("add", String[], ["x"], ["y"], bufs, [["x"]; keys_; [o.out for o in ops]], ops)
end

@testset "the upload drops its host copy" begin
    backend = first(Mantle.eachbackend())
    n, len = 8, 4096
    g = addgraph(n, len)
    weights = Dict{String,Any}("w$j" => fill(Float32(j), len) for j in 1:n)
    # Weak, so holding this cannot be what keeps them alive.
    refs = [WeakRef(weights["w$j"]) for j in 1:n]

    model = DKU.Model(Dict("add" => g), weights; backend)

    # The caller's dictionary was taken over, which is half of what makes the
    # loop's `delete!` the last reference. `consume = false` below is the
    # documented way out.
    @test isempty(weights)

    # And nothing in `Model`'s frame kept the other half.
    GC.gc(true); GC.gc(true)
    @test all(r -> r.value === nothing, refs)

    # The weights did reach the device, which is the other thing that could make
    # the assertion above true.
    @test length(model.weights) == n
    @test !any(v -> v isa Array, values(model.weights))
end

@testset "consume = false leaves the caller's dictionary alone" begin
    backend = first(Mantle.eachbackend())
    n, len = 4, 1024
    g = addgraph(n, len)
    weights = Dict{String,Any}("w$j" => fill(Float32(j), len) for j in 1:n)
    model = DKU.Model(Dict("add" => g), weights; backend, consume = false)
    @test length(weights) == n
    @test all(j -> weights["w$j"] == fill(Float32(j), len), 1:n)
    @test length(model.weights) == n
end

@testset "handoff! empties the source, and never the answer" begin
    a = Dict{String,Any}("k" => [1.0])
    b = Dict{String,Any}("k" => a["k"])
    @test DKU.handoff!(a, b) === b
    @test isempty(a)
    # The value survives: emptying a dictionary drops references, not arrays.
    @test b["k"] == [1.0]
    # A pass that had nothing to do returns its argument, and that must not be
    # emptied — this is the whole reason for the `===` guard.
    c = Dict{String,Any}("k" => [2.0])
    @test DKU.handoff!(c, c) === c
    @test c["k"] == [2.0]
end
