# The one-kernel norm path, and the predicate that unlocks it.
#
# `native_layer_norm` and `_fused_rms_norm` each have two forms: one kernel that
# reads its tensor once, and a fallback written as an expression that is six
# passes over it. The gate between them is a LAYOUT question — the kernel indexes
# linearly over a dense buffer, so a lazy `Broadcasted`, a `SubArray` or a
# permuted view has to go the other way, and so does a host `Array`, because the
# kernel is `cpu=false`.
#
# NOT `a isa Mantle.LavaArray`, which is that predicate with one backend's
# concrete type standing in for it. The cost is
# invisible on the backend it named and large everywhere else: on AMDGPU a
# `ROCArray` failed the test, so SAM 2.1's encoder replayed its 96 layer norms as
# 854 graph passes instead of 96 — 8.9 launches apiece for a form whose whole
# point is to be one — and the encoder ran 459 ms against 335 once the gate
# admitted them. It also moved the encoder CLOSER to PyTorch, `add_129` from
# 4.244 to 1.398, because the fallback's accumulation is not the kernel's.
#
# So what is pinned here is the predicate itself, against whatever array the
# backend under test hands out, plus the kernel's answer on that backend.

import KernelInterface as KI
using Test, DNNKernels, Mantle, KernelAbstractions
import GPUArrays

@testset "layer norm workgroup follows the reduction width" begin
    DK = DNNKernels
    @test DK.layernormwg(144) == 32
    @test DK.layernormwg(288) == 32
    @test DK.layernormwg(576) == 32
    @test DK.layernormwg(1152) == 64
    @test DK.layernormwg(4096) == DK.LN_WG
end


@testset "tiled transpose kernels preserve layout and pooling" begin
    DK = DNNKernels
    backend = Mantle.LavaBackend()

    # Float32 channel-major -> Float16 spatial-major transpose/cast.
    N, M = 7, 35
    ah = reshape(Float32.(1:N*M) ./ 13, N, M)
    a = DK.toback(backend, ah)
    d = DK.toback(backend, zeros(Float16, M, N))
    KI.Kernel(backend, DK.transposecast_f32_f16!)(
        d, a, Int32(M), Int32(N); ndrange = (32, 8, 1), workgroupsize = (32, 4, 1))
    KernelAbstractions.synchronize(backend)
    @test Array(d) == Float16.(permutedims(ah, (2, 1)))

    # The positional-embedding route performs the same transpose while adding
    # one fp16 and one fp32 source, with the sum retained in fp32.
    bh = reshape(Float16.(1:N*M) ./ 17, N, M)
    b = DK.toback(backend, bh)
    s = DK.toback(backend, zeros(Float32, M, N))
    KI.Kernel(backend, DK.transposeadd_f16_f32_f32!)(
        s, b, a, Int32(M), Int32(N); ndrange = (32, 8, 1), workgroupsize = (32, 4, 1))
    KernelAbstractions.synchronize(backend)
    @test Array(s) == permutedims(Float32.(bh) .+ ah, (2, 1))

    # The residual-stage variant transposes only its first fp16 source; the
    # second is already in destination order.
    dh = reshape(Float16.(1:N*M) ./ 23, M, N)
    d = DK.toback(backend, dh)
    t = DK.toback(backend, zeros(Float16, M, N))
    KI.Kernel(backend, DK.transposeadd_f16_dense_f16!)(
        t, b, d, Int32(M), Int32(N); ndrange = (32, 8, 1), workgroupsize = (32, 4, 1))
    KernelAbstractions.synchronize(backend)
    @test Array(t) == permutedims(bh, (2, 1)) .+ dh

    # Q occupies the first C channels of a 3C-wide interleaved QKV plane. The
    # pitched pool kernel must skip K/V between pixels and batches.
    C, IW, IH, B = 7, 8, 6, 2
    OW, OH, pitch = IW ÷ 2, IH ÷ 2, 3C
    sh = reshape(Float16.(1:pitch*IW*IH*B), pitch, IW, IH, B)
    src = DK.toback(backend, sh)
    out = DK.toback(backend, zeros(Float16, OW, OH, C, B))
    KI.Kernel(backend, DK.maxpool2transposekernel(Float16))(
        out, src, Int32(0), Int32(pitch * IW * IH), Int32(pitch),
        Int32(pitch * IW), Int32(OW), Int32(OH), Int32(C);
        ndrange = (32, 4, B), workgroupsize = (32, 4, 1))
    KernelAbstractions.synchronize(backend)
    want = [maximum(sh[c, 2x-1:2x, 2y-1:2y, b])
            for x in 1:OW, y in 1:OH, c in 1:C, b in 1:B]
    @test Array(out) == want
end

@testset "the norm gate admits this backend's arrays and no wrapper" begin
    DK = DNNKernels
    backend = Mantle.LavaBackend()
    # `Mantle.storage`: `toback` hands back the pool REGION that owns the
    # weight, and this testset is about the array over it — every line below
    # takes a view of it, permutes it, or broadcasts over it.
    x = Mantle.storage(DK.toback(backend, reshape(collect(Float32, 1:32), 8, 4)))
    # What the gate must accept: this backend's dense array.
    @test x isa GPUArrays.AbstractGPUArray
    # …and what it must still refuse, because their linear order is not the
    # buffer's. These are the reason the fallback exists.
    @test !(view(x, 1:4, :) isa GPUArrays.AbstractGPUArray)
    @test !(PermutedDimsArray(x, (2, 1)) isa GPUArrays.AbstractGPUArray)
    @test !(Broadcast.broadcasted(+, x, 1f0) isa GPUArrays.AbstractGPUArray)
    # A host array is refused for a different reason: `layernorm_kernel!` is
    # `cpu=false`, so there is no method to launch.
    @test !(Array(x) isa GPUArrays.AbstractGPUArray)
end

@testset "the one-kernel layer norm computes what the expression does" begin
    DK = DNNKernels
    backend = Mantle.LavaBackend()
    ctx = (; backend)                      # `layernorm!` asks for nothing else
    C, groups = 12, 5
    h = reshape(Float32.(1:(C * groups)) .* 0.25f0 .- 3.0f0, C, groups)
    γh = Float32.(1:C) ./ C
    βh = Float32.(C:-1:1) ./ 2C
    for (γ, β) in ((nothing, nothing), (γh, nothing), (γh, βh))
        a = DK.toback(backend, h)
        out = similar(a)
        μ = DK.toback(backend, zeros(Float32, groups))
        r = DK.toback(backend, zeros(Float32, groups))
        DK.layernorm!(ctx, out, μ, r, a,
                      γ === nothing ? nothing : DK.toback(backend, γ),
                      β === nothing ? nothing : DK.toback(backend, β),
                      C, 1.0f-5)
        # The reference is the fallback's arithmetic, in the same accumulator.
        m = vec(sum(h; dims = 1) ./ C)
        v = vec(sum(abs2, h .- reshape(m, 1, groups); dims = 1) ./ C)
        want = (h .- reshape(m, 1, groups)) ./ reshape(sqrt.(v .+ 1.0f-5), 1, groups)
        γ === nothing || (want = want .* γ)
        β === nothing || (want = want .+ β)
        @test Array(out) ≈ want rtol = 1.0f-5
        @test Array(μ) ≈ m rtol = 1.0f-5
        @test Array(r) ≈ 1.0f0 ./ sqrt.(v .+ 1.0f-5) rtol = 1.0f-5
    end
end

@testset "subgroup layer norm replaces the shared reduction tree" begin
    DK = DNNKernels
    backend = Mantle.LavaBackend()
    sg = Mantle.caps(Mantle.vk_context(backend)).subgroup
    C, groups = 144, 17
    h = reshape(Float32.(1:(C * groups)) .* 0.001f0 .- 1.0f0, C, groups)
    γh = Float32.(1:C) ./ C
    βh = Float32.(C:-1:1) ./ 2C
    a, γ, β = map(x -> DK.toback(backend, x), (h, γh, βh))
    out = similar(a)
    μ = DK.toback(backend, zeros(Float32, groups))
    r = similar(μ)
    rowsg = DK.layernormwg(C)
    KI.Kernel(backend, DK.layernorm_shfl_kernel!)(
        out, μ, r, a, γ, β, Int32(C), Int32(groups), 1.0f-5,
        Val(sg), Val(rowsg), Val(true), Val(true);
        ndrange = cld(groups, sg ÷ rowsg) * sg, workgroupsize = sg)
    KernelAbstractions.synchronize(backend)

    m = vec(sum(h; dims = 1) ./ C)
    v = vec(sum(abs2, h .- reshape(m, 1, groups); dims = 1) ./ C)
    want = (h .- reshape(m, 1, groups)) ./ reshape(sqrt.(v .+ 1.0f-5), 1, groups)
    want = want .* γh .+ βh
    @test Array(out) ≈ want rtol = 1.0f-5
    @test Array(μ) ≈ m rtol = 1.0f-5
    @test Array(r) ≈ 1.0f0 ./ sqrt.(v .+ 1.0f-5) rtol = 1.0f-5

    # The residual-add form writes the carried residual and normalises that
    # rounded value in the same dispatch.
    ah1 = h .* 0.75f0
    ah2 = h .* 0.25f0
    a1, a2 = map(x -> DK.toback(backend, x), (ah1, ah2))
    sumout = similar(a1); fusedout = similar(a1)
    fusedμ = similar(μ); fusedr = similar(r)
    KI.Kernel(backend, DK.add_layernorm_shfl_kernel!)(
        fusedout, sumout, fusedμ, fusedr, a1, a2, γ, β,
        Int32(C), Int32(groups), 1.0f-5, Val(sg), Val(rowsg), Val(true), Val(true);
        ndrange = cld(groups, sg ÷ rowsg) * sg, workgroupsize = sg)
    KernelAbstractions.synchronize(backend)
    @test Array(sumout) == ah1 .+ ah2
    @test Array(fusedout) == Array(out)
    @test Array(fusedμ) == Array(μ)
    @test Array(fusedr) == Array(r)
end

@testset "subgroup layer norm writes window order directly" begin
    DK = DNNKernels
    backend = Mantle.LavaBackend()
    sg = Mantle.caps(Mantle.vk_context(backend)).subgroup
    C, IW, IH, NX, NY, B = 144, 4, 4, 2, 2, 1
    W, H, groups = IW * NX, IH * NY, IW * NX * IH * NY * B
    h = reshape(Float32.(1:(C * groups)) .* 0.0001f0 .- 2.0f0, C, W, H, B)
    γh, βh = Float32.(1:C) ./ C, Float32.(C:-1:1) ./ 3C
    a, γ, β = map(x -> DK.toback(backend, x), (h, γh, βh))
    out = DK.toback(backend, zeros(Float16, C, IW, IH, NX, NY, B))
    wide = similar(a)
    μ = DK.toback(backend, zeros(Float32, groups)); r = similar(μ)
    refwide = similar(a); refμ = similar(μ); refr = similar(r)
    rowsg = DK.layernormwg(C)
    KI.Kernel(backend, DK.layernorm_shfl_kernel!)(
        refwide, refμ, refr, a, γ, β, Int32(C), Int32(groups), 1.0f-5,
        Val(sg), Val(rowsg), Val(true), Val(true);
        ndrange = cld(groups, sg ÷ rowsg) * sg, workgroupsize = sg)
    KI.Kernel(backend, DK.layernorm_shfl_window4_add_kernel!)(
        out, wide, wide, μ, r, a, a, γ, β, Int32(C), Int32(groups), 1.0f-5,
        Val(sg), Val(rowsg), Val(IW), Val(IH), Val(NX), Val(NY),
        Val(false), Val(true), Val(true), Val(true);
        ndrange = cld(groups, sg ÷ rowsg) * sg, workgroupsize = sg)
    KernelAbstractions.synchronize(backend)

    want = Array(refwide)
    windowed = Array{Float16}(undef, C, IW, IH, NX, NY, B)
    for b in 1:B, wy in 1:NY, wx in 1:NX, iy in 1:IH, ix in 1:IW
        windowed[:, ix, iy, wx, wy, b] =
            Float16.(want[:, ix + IW * (wx - 1), iy + IH * (wy - 1), b])
    end
    @test Array(wide) == want
    @test Array(out) == windowed
end
