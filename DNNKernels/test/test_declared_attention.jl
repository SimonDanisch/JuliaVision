# The declared three-pass attention, and which passes it declares.
#
# `threepass!` asks Mantle for a recordable batched GEMM for BOTH of attention's
# products and falls back to the portable blocked kernels per product. That is one
# decision made in backend-independent code from `Mantle.native_batched_gemm_dispatch!`'s
# answer, so what has to be pinned is that BOTH routes compute the same attention and
# that the route is visible in the plan:
#
#   native    scores (q read transposed in the matrix descriptor) | softmax | apply
#   portable  toLE | scores | softmax | apply
#
# The `toLE` copy is the tell: the native score product transposes `q` in its descriptor,
# so a graph with three passes is the native route and one with four is not. Both are
# exercised on the same device by giving the second shape a sequence length the tiling
# cannot cover, which is how a backend declines.
#
# Over every backend rather than `LavaBackend()`, which most of this suite still
# hardcodes: the routing is Mantle's interface, not one backend's, and the apply
# product's `coldiv` epilogue — softmax leaves its row sums unnormalized and the apply
# divides by them — is the part a backend can get subtly wrong while every shape still
# runs.

using Test
import DNNKernels
import Mantle
import KernelAbstractions

const DK = DNNKernels
const MM = Mantle
const KAb = KernelAbstractions

"""Attention on the host, fp32 over the fp16 operands, normalized at the end."""
function attn_host(qh, kh, vh, scale)
    E, Lq, H, B = size(qh); Lk = size(kh, 2)
    out = zeros(Float32, E, Lq, H, B)
    for b in 1:B, h in 1:H, lq in 1:Lq
        s = [sum(Float32(qh[e, lq, h, b]) * Float32(kh[e, lk, h, b]) for e in 1:E) * scale
             for lk in 1:Lk]
        s .= exp.(s .- maximum(s)); s ./= sum(s)
        for e in 1:E
            out[e, lq, h, b] = sum(s[lk] * Float32(vh[e, lk, h, b]) for lk in 1:Lk)
        end
    end
    out
end

"""One declared `threepass!`, recorded, replayed, and read back."""
function declared_threepass(dev, E, L, H, B)
    qh = Float16.(reshape(0.4 .* sin.(range(0, 9, E*L*H*B)), E, L, H, B))
    kh = Float16.(reshape(0.4 .* cos.(range(0, 7, E*L*H*B)), E, L, H, B))
    vh = Float16.(reshape(0.4 .* sin.(range(0, 5, E*L*H*B)), E, L, H, B))
    q = MM.Buffer(dev, qh); k = MM.Buffer(dev, kh); v = MM.Buffer(dev, vh)
    out = MM.Buffer(dev, Float16, (E, L, H, B))
    graph = MM.Graph(dev)
    op = DK.Op("attn", "fused.sdpa", String[], "attn", Dict{String,Any}())
    emitctx = DK.EmitCtx(
        DK.Graph("attn", String[], String[], String[],
                 Dict{String,DK.Buffer}(), String[], DK.Op[], Vector{Vector{String}}()),
        graph, dev, NamedTuple(), Dict{String,Any}("attn" => out),
        Set{String}(), Ref("attn"), Any[])
    scale = Float32(inv(sqrt(E)))
    DK.threepass!(emitctx, op, out, q, k, v, nothing, scale)
    npasses = length(graph.passes)
    plan = MM.Plan(graph)
    MM.record!(plan); MM.run!(plan); MM.waitidle(dev)
    got = Float32.(Array(MM.storage(out)))
    MM.free!(plan)
    ref = attn_host(qh, kh, vh, scale)
    err = maximum(abs, got .- ref) / max(maximum(abs, ref), eps(Float32))
    # A pass that writes nothing also "matches" a zero reference.
    (; err, npasses, wrote = maximum(abs, got) > 1e-4)
end

"""One declared `native_attention_dispatch!`, recorded, replayed, and read back."""
function declared_fused(dev, E, L, H, B)
    qh = Float16.(reshape(0.4 .* sin.(range(0, 9, E*L*H*B)), E, L, H, B))
    kh = Float16.(reshape(0.4 .* cos.(range(0, 7, E*L*H*B)), E, L, H, B))
    vh = Float16.(reshape(0.4 .* sin.(range(0, 5, E*L*H*B)), E, L, H, B))
    q = MM.Buffer(dev, qh); k = MM.Buffer(dev, kh); v = MM.Buffer(dev, vh)
    out = MM.Buffer(dev, Float16, (E, L, H, B))
    graph = MM.Graph(dev)
    scale = Float32(inv(sqrt(E)))
    MM.native_attention_dispatch!(dev, graph, out, q, k, v; scale, name = "flash") ||
        return nothing
    npasses = length(graph.passes)
    plan = MM.Plan(graph)
    MM.record!(plan); MM.run!(plan); MM.waitidle(dev)
    got = Float32.(Array(MM.storage(out)))
    MM.free!(plan)
    ref = attn_host(qh, kh, vh, scale)
    (; err = maximum(abs, got .- ref) / max(maximum(abs, ref), eps(Float32)),
       npasses, wrote = maximum(abs, got) > 1e-4)
end

function declaredattention(dev, label)
    @testset "declared three-pass attention — $label" begin
        E, H, B = 72, 2, 1
        # Ask the device the same question `threepass!` asks, so the pass count below is
        # an assertion about the ROUTE and not a record of whatever happened.
        offers = let g = MM.Graph(dev)
            sc = MM.Transient.Buffer(g, Float16, (128, 128, H, B))
            qq = MM.Buffer(dev, Float16, (E, 128, H, B))
            kk = MM.Buffer(dev, Float16, (E, 128, H, B))
            MM.native_batched_gemm_dispatch!(dev, g, sc, qq, kk; transpose_a = true,
                                             alpha = 1.0f0, name = "probe")
        end

        r = declared_threepass(dev, E, 128, H, B)
        @test r.wrote
        @test r.err < 3e-3
        # Three passes means `toLE` is gone AND apply carried the row-sum divide itself,
        # because softmax never normalizes — so a wrong `coldiv` shows up as `err`, not
        # as a missing pass.
        @test r.npasses == (offers ? 3 : 4)
        offers || @info "no recordable batched GEMM for these operands on $label; " *
                        "portable route only"

        # A length no tile covers: 12 is not a multiple of eight, so a backend whose
        # native GEMM tiles in eights declines and the portable kernels run. Both routes
        # are then checked against the same host reference on the same device.
        r2 = declared_threepass(dev, E, 12, H, B)
        @test r2.wrote
        @test r2.err < 3e-3
        @test r2.npasses == 4

        # And the FUSED route, which `emitsdpa!` prefers over all three passes where the
        # backend has one. Declared through the same interface into a recorded plan, so
        # this covers the access walk and the replay, not just the kernel's arithmetic.
        rf = declared_fused(dev, E, 128, H, B)
        if rf === nothing
            @info "no recordable fused attention on $label"
        else
            @test rf.npasses == 1
            @test rf.wrote
            @test rf.err < 3e-3
        end
    end
end

for back in MM.eachbackend()
    declaredattention(MM.todevice(back), string(nameof(typeof(back))))
end
