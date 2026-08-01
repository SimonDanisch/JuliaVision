"""
The fused attention kernel: correct where it claims to be, and refusing the
shapes where it is not.

It is not on the `sdpa` path — it measured slower (see the file's docstring) —
so this exists to keep it honest for whoever picks the optimisation back up. A
kernel that is wrong and unused is worse than no kernel: it looks available.
"""

using Test, DNNKernels, Lava, KernelAbstractions
const KA = KernelAbstractions

function attnref(qh, kh, vh, scale)
    E, Lq, H, B = size(qh); Lk = size(kh, 2)
    out = zeros(Float32, E, Lq, H, B)
    for b in 1:B, h in 1:H, lq in 1:Lq
        s = [sum(qh[e,lq,h,b] * kh[e,lk,h,b] for e in 1:E) * scale for lk in 1:Lk]
        s .= exp.(s .- maximum(s)); s ./= sum(s)
        for e in 1:E
            out[e,lq,h,b] = sum(s[lk] * vh[e,lk,h,b] for lk in 1:Lk)
        end
    end
    out
end

@testset "fused attention" begin
    back = LavaBackend()

    @testset "exact on the validated tiling" begin
        for (E, L, H, B) in ((72, 64, 1, 1), (72, 128, 2, 1), (72, 256, 2, 1))
            qh = randn(Float32,E,L,H,B) .* 0.2f0
            kh = randn(Float32,E,L,H,B) .* 0.2f0
            vh = randn(Float32,E,L,H,B) .* 0.2f0
            q = DNNKernels.toback(back,qh); k = DNNKernels.toback(back,kh); v = DNNKernels.toback(back,vh)
            o = KA.allocate(back, Float32, E,L,H,B); fill!(o, 0f0)
            scale = Float32(1/sqrt(E))
            @test DNNKernels.sdpaflash!(o, q, k, v, scale; backend = back)
            KA.synchronize(back)
            got = Array(o)
            # A kernel that writes nothing also "matches" a zero reference, so
            # check it produced something before checking what.
            @test maximum(abs, got) > 1e-3
            @test maximum(abs, got .- attnref(qh, kh, vh, scale)) < 1e-5
            q = k = v = o = nothing; GC.gc()
        end
    end

    @testset "exact at every tiling, including the odd slot count" begin
        # `BQ = 32, NT = 256` gives 9 accumulator slots per thread and was wrong
        # from query row 4 on (7.1e-02) until the staging index stopped going
        # through `OpUDiv` — `E = 72` is not a power of two. Varied inputs
        # matter: the broken version was **exact for constant inputs**, which is
        # what disguised it as an accumulator problem for so long.
        E, L, H, B = 72, 128, 2, 2
        qh = randn(Float32,E,L,H,B) .* 0.2f0
        kh = randn(Float32,E,L,H,B) .* 0.2f0
        vh = randn(Float32,E,L,H,B) .* 0.2f0
        q = DNNKernels.toback(back,qh); k = DNNKernels.toback(back,kh); v = DNNKernels.toback(back,vh)
        scale = Float32(1/sqrt(E))
        ref = attnref(qh, kh, vh, scale)
        for (BQ, NT) in ((64, 256), (32, 256), (32, 128), (64, 128))
            @test DNNKernels.flashfits(E, BQ, 32, NT)
            o = KA.allocate(back, Float32, E,L,H,B); fill!(o, 0f0)
            DNNKernels.attn_flash!(back, NT)(o, q, k, v, scale, Val(BQ), Val(32),
                                             Val(E), Val(NT), Int32(L);
                                             ndrange = (NT * div(L, BQ), H, B))
            KA.synchronize(back)
            got = Array(o)
            @test maximum(abs, got) > 1e-3
            @test maximum(abs, got .- ref) < 1e-4
            o = nothing
        end
        q = k = v = nothing; GC.gc()
    end

    @testset "refuses what it cannot compute" begin
        # Shared memory over budget: this tiling launches and writes zeros.
        @test !DNNKernels.flashfits(72, 64, 64, 256)
        @test DNNKernels.flashshared(72, 64, 64) > DNNKernels.FLASH_SHARED_BUDGET[]
        # The validated one fits, in both senses.
        @test DNNKernels.flashfits(72, 64, 32, 256)
        @test DNNKernels.flashshared(72, 64, 32) <= DNNKernels.FLASH_SHARED_BUDGET[]
        # An odd accumulator-slot count used to be refused, because
        # `BQ = 32, NT = 256` computed wrong results. That was `OpUDiv` in the
        # staging index, not the slot count — see `flashfits` — and `splitidx`
        # fixed it, so both are accepted now and both must be exact.
        @test DNNKernels.flashfits(72, 32, 32, 256)       # BQ*E/NT = 9, odd
        @test DNNKernels.flashfits(72, 32, 32, 128)       # BQ*E/NT = 18
        # …and a sequence the tiling does not divide falls back.
        E, L, H, B = 72, 100, 1, 1              # 100 % 64 != 0
        q = KA.allocate(back, Float32, E,L,H,B); fill!(q, 0.01f0)
        k = KA.allocate(back, Float32, E,L,H,B); fill!(k, 0.01f0)
        v = KA.allocate(back, Float32, E,L,H,B); fill!(v, 0.01f0)
        o = KA.allocate(back, Float32, E,L,H,B)
        @test !DNNKernels.sdpaflash!(o, q, k, v, 0.1f0; backend = back)
        q = k = v = o = nothing; GC.gc()
    end
end
