"""
`attn_flash_rows!`: subgroups that own their query rows. Exact where it claims
to be, at the extents that exercise its two slides (a query tile moved back onto
the tensor, a key block re-reading the last key), and taken only where the plan
says so.
"""

using Test, DNNKernels, Lava, KernelAbstractions
import Mantle
using Mantle: LavaBackend

"""fp32 attention of fp16 operands, per head, as `(E, Lq, H, B)`."""
function rowsref(qh, kh, vh, scale)
    E, Lq, H, B = size(qh)
    out = zeros(Float32, E, Lq, H, B)
    for b in 1:B, h in 1:H
        S = (Float32.(kh[:, :, h, b])' * Float32.(qh[:, :, h, b])) .* scale
        S .= exp.(S .- maximum(S; dims = 1))
        S ./= sum(S; dims = 1)
        out[:, :, h, b] = Float32.(vh[:, :, h, b]) * S
    end
    out
end

@testset "flash attention with subgroup-owned rows" begin
    back = LavaBackend()
    ctx = DNNKernels.Ctx(back)
    dev = ctx.dev

    if !(DNNKernels.coopmatkernels(dev) && dev.tile == 16 && dev.coopmatsubgroup == 32)
        @info "attn_flash_rows! needs 16x16 cooperative matrices on 32-lane subgroups; skipping"
    else
        @testset "exact at E=$E, Lq=$Lq, Lk=$Lk, H=$H, B=$B, out $T" for
                (E, Lq, Lk, H, B, T) in (
                    (128, 256, 4118, 2, 1, Float32),   # Qwen-Image 2.1's ragged key axis
                    (128, 64, 32, 1, 1, Float32),      # one whole key block
                    (128, 100, 17, 1, 2, Float32),     # ragged both ways, one short block
                    (128, 40, 1, 1, 1, Float32),       # a single key; queries slide
                    (64, 256, 1000, 2, 1, Float32),
                    (64, 16, 64, 3, 1, Float32),       # fewer queries than a workgroup has rows
                    (128, 128, 500, 1, 1, Float16))
            qh = Float16.(randn(Float32, E, Lq, H, B))
            kh = Float16.(randn(Float32, E, Lk, H, B))
            vh = Float16.(randn(Float32, E, Lk, H, B))
            scale = Float32(1 / sqrt(E))
            out = LavaArray(zeros(T, E, Lq, H, B))
            DNNKernels.sdpaflashrows!(ctx, out, DNNKernels.FlashRowsPlan(E),
                                      LavaArray(qh), LavaArray(kh), LavaArray(vh), scale)
            ref = rowsref(qh, kh, vh, scale)
            # P is fp16, so the error is P's rounding carried through V: it
            # grows as fewer keys share the weight. An fp16 `out` adds its own.
            @test maximum(abs.(Float32.(Array(out)) .- ref)) < (T === Float16 ? 1e-3 : 5e-4)
        end

        @testset "scores far from zero" begin
            # Four times the unit scale puts row maxima near 45 in log2 units, so
            # the running maximum and its rescale are exercised, not just exp(0).
            E, Lq, Lk = 128, 128, 700
            qh = Float16.(4f0 .* randn(Float32, E, Lq, 1, 1))
            kh = Float16.(4f0 .* randn(Float32, E, Lk, 1, 1))
            vh = Float16.(randn(Float32, E, Lk, 1, 1))
            scale = Float32(1 / sqrt(E))
            out = LavaArray(zeros(Float32, E, Lq, 1, 1))
            DNNKernels.sdpaflashrows!(ctx, out, DNNKernels.FlashRowsPlan(E),
                                      LavaArray(qh), LavaArray(kh), LavaArray(vh), scale)
            @test maximum(abs.(Array(out) .- rowsref(qh, kh, vh, scale))) < 5e-4
        end

        @testset "operands read in place from a packed tensor" begin
            # q, k and v as views into one (E, L, 3H, B) projection, the way a
            # fused QKV leaves them: strides of the parent, offsets into it.
            E, L, H, B = 128, 192, 2, 2
            qkvh = Float16.(randn(Float32, E, L, 3H, B))
            qkv = LavaArray(qkvh)
            q = view(qkv, :, :, 1:H, :)
            k = view(qkv, :, :, (H + 1):2H, :)
            v = view(qkv, :, :, (2H + 1):3H, :)
            plan = DNNKernels.flashrows_plan(Mantle.DeviceCaps(dev; cores = 1), q, k, v, nothing)
            @test plan == DNNKernels.FlashRowsPlan(E)
            scale = Float32(1 / sqrt(E))
            out = LavaArray(zeros(Float32, E, L, H, B))
            DNNKernels.sdpaflashrows!(ctx, out, plan, q, k, v, scale)
            ref = rowsref(qkvh[:, :, 1:H, :], qkvh[:, :, (H + 1):2H, :],
                          qkvh[:, :, (2H + 1):3H, :], scale)
            @test maximum(abs.(Array(out) .- ref)) < 5e-4
        end

        @testset "the plan takes what it is written for, and only that" begin
            plan(q, k, v; bias = nothing, caps = dev) = DNNKernels.flashrows_plan(caps, q, k, v, bias)
            gdev = Mantle.Device(back)
            buf(dims...) = Mantle.Buffer(gdev, Float16, dims)
            # Qwen-Image 2.1's attention: 4096 queries, 4118 keys, 32 heads.
            q, k = buf(128, 4096, 32, 1), buf(128, 4118, 32, 1)
            @test plan(q, k, k) == DNNKernels.FlashRowsPlan(128)
            @test plan(q, k, k; bias = k) == DNNKernels.Decline(:bias)
            @test plan(buf(96, 4096, 32, 1), buf(96, 4118, 32, 1), buf(96, 4118, 32, 1)) ==
                  DNNKernels.Decline(:headdim)
            @test plan(buf(128, 8, 32, 1), k, k) == DNNKernels.Decline(:extent)
            # Too few workgroups to fill the device is the key-split kernel's shape.
            @test plan(buf(128, 64, 1, 1), buf(128, 4096, 1, 1), buf(128, 4096, 1, 1)) ==
                  DNNKernels.Decline(:grid)
            # A device the indexing is not written for.
            @test plan(q, k, k; caps = Mantle.DeviceCaps(dev; coopmatsubgroup = 64)) ==
                  DNNKernels.Decline(:subgroup)
            @test plan(q, k, k; caps = Mantle.DeviceCaps(dev; sharedbudget = 16384)) ==
                  DNNKernels.Decline(:shared)
            @test DNNKernels.flashrows_shared(128) == 21504
            # A key row that does not start on a 16-byte boundary cannot be
            # staged in 16-byte chunks.
            # A contiguous view of a device array is itself a device array with
            # the offset inside it, so `stridedroot` reports offset 0 for it: the
            # base's own alignment is what catches it. Found writing this test.
            n = 128 * 4118 * 32
            ka = LavaArray(zeros(Float16, n + 8))
            @test plan(q, reshape(view(ka, 5:(n + 4)), 128, 4118, 32, 1), k) == DNNKernels.Decline(:align)
            @test plan(q, reshape(view(ka, 9:(n + 8)), 128, 4118, 32, 1), k) == DNNKernels.FlashRowsPlan(128)
            # A key stride that is not a multiple of eight elements.
            kb = LavaArray(zeros(Float16, 132, 4118, 32, 1))
            @test plan(q, view(kb, 1:128, :, :, :), k) == DNNKernels.Decline(:align)
        end

        @testset "declared into a graph and replayed" begin
            gdev = Mantle.Device(back)
            E, Lq, Lk, H, B = 128, 192, 300, 2, 1
            qh = Float16.(randn(Float32, E, Lq, H, B))
            kh = Float16.(randn(Float32, E, Lk, H, B))
            vh = Float16.(randn(Float32, E, Lk, H, B))
            q, k, v = Mantle.Buffer(gdev, qh), Mantle.Buffer(gdev, kh), Mantle.Buffer(gdev, vh)
            out = Mantle.Buffer(gdev, Float32, (E, Lq, H, B))
            graph = Mantle.Graph(gdev)
            scale = Float32(1 / sqrt(E))
            DNNKernels.flashrows_dispatch!(graph, out, DNNKernels.FlashRowsPlan(E), q, k, v, scale)
            @test length(graph.passes) == 1
            p = Mantle.Plan(graph)
            Mantle.record!(p)
            Mantle.run!(p); Mantle.waitidle(gdev)
            @test maximum(abs.(Array(Mantle.storage(out)) .- rowsref(qh, kh, vh, scale))) < 5e-4
            # A second replay of the same recording computes the same thing.
            Mantle.run!(p); Mantle.waitidle(gdev)
            @test maximum(abs.(Array(Mantle.storage(out)) .- rowsref(qh, kh, vh, scale))) < 5e-4
            Mantle.free!(p)
        end
    end
end
