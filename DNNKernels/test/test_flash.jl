"""
The fused attention kernel: correct where it claims to be, refusing the shapes
where it is not, and recordable through the declared `sdpa` path.
"""

using Test, DNNKernels, Lava, KernelAbstractions, Random
import Mantle
using Mantle: LavaBackend
const KA = KernelAbstractions

@testset "subgroup flash uses Mantle's portable cooperative-matrix surface" begin
    src = read(joinpath(pkgdir(DNNKernels), "src", "kernels", "extern", "flash.jl"),
               String)
    portable_lava_name = r"Lava\.(AcceleratedMatrix|Accumulator|MatrixA|MatrixB|coopmat_(load|store|muladd|mul|add|zero|undef|convert|length|getcomp|setcomp))"
    @test match(portable_lava_name, src) === nothing
    # Per-element callbacks are not part of the portable eleven-operation floor:
    # this one remains the explicitly gated VK_NV_cooperative_matrix2 route.
    @test occursin("Lava.coopmat_perelement", src)
end

@testset "flash output can land directly in spatial window order" begin
    E, H, IW, IH, NX, NY, B = 3, 2, 4, 2, 3, 2, 2
    L = IW * IH
    seen = Int[]
    for b in 1:(NX * NY * B), h in 1:H, l in 0:(L - 1), e in 0:(E - 1)
        got = DNNKernels.flashoutindex(e, h, l, b, Val(E), H, L,
                                        Val(IW), Val(IH), Val(NX), Val(NY))
        ix, iy = l % IW, l ÷ IW
        wx, tail = (b - 1) % NX, (b - 1) ÷ NX
        wy, batch = tail % NY, tail ÷ NY
        col = ix + IW * wx + (IW * NX) * (iy + IH * wy + IH * NY * batch)
        @test got == e + E * ((h - 1) + H * col)
        push!(seen, got)
    end
    @test sort(seen) == collect(0:(E * H * IW * NX * IH * NY * B - 1))
end

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
    # The kernel entry points take a context. `Ctx(backend)` builds one with no
    # graph behind it, which is exactly the direct-call case this file is.
    ctx = DNNKernels.Ctx(back)
    # The tiling chooser and the fit predicates take a `DeviceCaps` rather than
    # reading one, so they can be asked about hardware this machine is not.
    dev = ctx.dev

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
        # `BQ = 32, NT = 256` gives 9 accumulator slots per thread and is wrong
        # from query row 4 on (7.1e-02) if the staging index goes
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
        @test DNNKernels.flashshared(72, 64, 64) > DNNKernels.PORTABLE_SHARED_FLOOR
        # The validated one fits, in both senses.
        @test DNNKernels.flashfits(72, 64, 32, 256)
        @test DNNKernels.flashshared(72, 64, 32) <= DNNKernels.PORTABLE_SHARED_FLOOR
        # An odd accumulator-slot count is admitted: `BQ = 32, NT = 256`
        # computing wrong results is `OpUDiv` in the
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

    @testset "scalar flash is a recordable declared pass" begin
        # `gdev`, NOT `dev`. Allocating a buffer needs the concrete device while
        # every predicate in this file takes the `DeviceCaps` bound above, and
        # `@testset` does not open a scope that an assignment stays inside: a
        # `dev = Mantle.Device(back)` here REBOUND the outer `dev` for every
        # testset that follows, so the tiling chooser and the plan builders were
        # handed a `LavaDevice` and threw `MethodError` from there to the end of
        # the file. Only the testsets after this one failed, which is what an
        # order-dependent rebinding looks like.
        gdev = Mantle.Device(back)
        E, L, H, B = 72, 128, 2, 1
        qh = Float16.(randn(Float32, E,L,H,B) .* 0.2f0)
        kh = Float16.(randn(Float32, E,L,H,B) .* 0.2f0)
        vh = Float16.(randn(Float32, E,L,H,B) .* 0.2f0)
        q = Mantle.Buffer(gdev, qh)
        k = Mantle.Buffer(gdev, kh)
        v = Mantle.Buffer(gdev, vh)
        out = Mantle.Buffer(gdev, Float16, (E,L,H,B))
        graph = Mantle.Graph(gdev)
        op = DNNKernels.Op("flash", "fused.sdpa", String[], "flash",
                           Dict{String,Any}())
        emitctx = DNNKernels.EmitCtx(
            DNNKernels.Graph("flash", String[], String[], String[],
                             Dict{String,DNNKernels.Buffer}(), String[],
                             DNNKernels.Op[], Vector{Vector{String}}()),
            graph, gdev, NamedTuple(), Dict{String,Any}("flash" => out),
            Set{String}(), Ref("flash"), Any[], Dict{String,Any}())
        scale = Float32(inv(sqrt(E)))
        @test DNNKernels.scalarflash_dispatch!(emitctx, op, out, q, k, v,
                                                nothing, scale)
        @test length(graph.passes) == 1
        plan = Mantle.Plan(graph)
        Mantle.record!(plan)
        Mantle.run!(plan)
        Mantle.waitidle(gdev)
        got = Array(Mantle.storage(out))
        ref = attnref(Float32.(qh), Float32.(kh), Float32.(vh), scale)
        @test maximum(abs, Float32.(got) .- ref) < 2e-3
        Mantle.free!(plan)
        q = k = v = out = nothing; GC.gc()
    end

    # ── the cooperative-matrix form, which IS on the `sdpa` path ─────────────
    # The context, not a global: `coopmat_gemm_available` asks a DEVICE, and
    # there is no no-argument form reaching for the default one.
    if !Mantle.coopmat_gemm_available(Mantle.vk_context())
        @info "no cooperative-matrix support on this device; skipping the fused path"
    else
        @testset "cooperative-matrix flash: exact at every shipped tiling" begin
            # fp16 operands, because that is what the path requires and what the
            # encoder runs — the tolerance below is the format's, not the
            # algorithm's. Varied inputs for the reason recorded above: the
            # `OpUDiv` failure this kernel's staging indices would otherwise hit
            # is EXACT for constant inputs, so constants prove nothing.
            E, L, H, B = 72, 128, 2, 2
            qh = randn(Float32,E,L,H,B) .* 0.2f0
            kh = randn(Float32,E,L,H,B) .* 0.2f0
            vh = randn(Float32,E,L,H,B) .* 0.2f0
            f16(x) = DNNKernels.toback(back, Float16.(x))
            q, k, v = f16(qh), f16(kh), f16(vh)
            scale = Float32(1/sqrt(E))
            ref = attnref(Float32.(Float16.(qh)), Float32.(Float16.(kh)),
                          Float32.(Float16.(vh)), scale)
            # Both places `O` can live, because `FLASHCM_REGO` is a measured
            # choice and the losing branch still has to be right — a switch whose
            # other side is broken is not a switch.
            for (BR, BC, NW) in DNNKernels.FLASHCM_TILINGS, rego in (false, true),
                lazyrescale in (false, true), held in (false, true)
                # `dev.coopmatsubgroup`, not 32 and not `dev.subgroup`: Lava pins
                # cooperative-matrix modules to COOPMAT_SUBGROUP, which is where
                # this thread count comes from, while RDNA's device default is 64.
                # The literal happened to be right; it stops being a coincidence
                # only once it names which of the two widths it means.
                NW * dev.coopmatsubgroup <= dev.workgrouplimit || continue
                L % BR == 0 && L % BC == 0 || continue
                # The table also contains high-residency candidates whose
                # reduced footprint exists only for an automatically selected,
                # unsplit held-output plan. This exhaustive switch test calls
                # the explicit conservative planner, so those are not admissible
                # on devices whose full `pvs` footprint exceeds the budget.
                EP = cld(E, dev.tile) * dev.tile
                NT = NW * dev.coopmatsubgroup
                DNNKernels.flashcmfits(dev, EP, BR, BC, NT) || continue
                o = KA.allocate(back, Float32, E,L,H,B); fill!(o, 0f0)
                @test DNNKernels.sdpaflashcm!(ctx, o, q, k, v, scale;
                                              BR, BC, NW, rego, lazyrescale, held)
                KA.synchronize(back)
                got = Array(o)
                @test maximum(abs, got) > 1e-3          # it wrote something…
                @test maximum(abs, got .- ref) / maximum(abs, ref) < 5e-3
                o = nothing
            end
            # The graph emitter can remove the sole `(E,L,H,B) -> (E,H,L,B)`
            # materialisation after attention by asking flash to write that
            # physical order itself. The array keeps the logical input shape at
            # the kernel boundary, so reinterpret its bytes with the consumer's
            # shape before comparing.
            plan = DNNKernels.flashcm_plan(dev, q, k, v, nothing; split = false)
            @test plan isa DNNKernels.FlashCMPlan
            o = KA.allocate(back, Float32, E,L,H,B); fill!(o, 0f0)
            DNNKernels.sdpaflashcm!(ctx, o, plan, q, k, v, scale; outperm = true)
            KA.synchronize(back)
            gotperm = reshape(Array(o), E,H,L,B)
            @test maximum(abs, gotperm .- permutedims(ref, (1,3,2,4))) /
                  maximum(abs, ref) < 5e-3
            q = k = v = nothing; GC.gc()
        end

        @testset "cooperative-matrix flash: the tiling chooser" begin
            # 64 x 64 wants 66 KB and must never be offered.
            @test !DNNKernels.flashcmfits(dev, 80, 64, 64, 256)
            @test DNNKernels.flashcmfits(dev, 80, 64, 32, 256)
            # The shipped default, on the encoder's two dominant shapes. `NW` is
            # the table's own scaled to this device's wave: a cooperative-matrix
            # module that runs narrower than the wave (32 against 64 on RDNA 3.5)
            # needs twice the subgroups for the same waves in flight, and
            # `flashcm_tiling` says so with the measurement. `widen` is 1 where
            # the two agree, which is where the table was measured.
            widen = max(max(1, dev.subgroup ÷ dev.coopmatsubgroup),
                        dev.warps >= 64 ? 2 : 1)
            @test DNNKernels.flashcm_tiling(dev, 72, 4096, 4096) == (64, 32, 8 * widen)
            @test DNNKernels.flashcm_tiling(dev, 72, 256, 256) == (64, 32, 8 * widen)
            # A query count no tiling divides is taken clamped only when the
            # padding earns its place. Clamping remains opt-in here; the planner
            # enables it for the exact one-key-tile case below.
            @test DNNKernels.flashcm_tiling(dev, 72, 4, 16) === nothing
            # Clamping is off by default, so a non-dividing extent is refused
            # until the caller asks for it — which only the decoder does.
            @test DNNKernels.flashcm_tiling(dev, 16, 23, 4096) === nothing
            @test DNNKernels.flashcm_tiling(dev, 16, 23, 4096; clamp=true)[1] == 32
            @test DNNKernels.flashcm_tiling(dev, 16, 23, 23; clamp=true) == (32, 32, 8 * widen)
            # Four queries against exactly one 16-key tile are the measured
            # exception: a quarter-full fused tile is still cheaper than two
            # padded GEMMs plus score, softmax and apply passes.
            @test DNNKernels.flashcm_tiling(dev, 72, 4, 16; clamp=true) == (16, 16, 4)
            # A 128-wide head: the 32-key block wants 71816 bytes of shared
            # against 65536 **when `O` is accounted in shared memory**, which is
            # where the narrow-key entry came from. Held in cooperative-matrix
            # fragments there is no `pvs` and the wider block fits, which is
            # worth 13.4% — 42.69 ms against 49.27 at `Lq = Lk = 4096`, and
            # 58.47 to 46.44 on Qwen-Image 2.1's own 4118-key shape.
            @test !DNNKernels.flashcmfits(dev, 128, 64, 32, 16 * dev.coopmatsubgroup)
            @test DNNKernels.flashcmfits(dev, 128, 64, 32, 16 * dev.coopmatsubgroup, true)
            @test DNNKernels.flashcmfits(dev, 128, 64, 16, 16 * dev.coopmatsubgroup)
            @test DNNKernels.flashcm_tiling(dev, 128, 4096, 4118, 32; clamp=true) ==
                  (64, 32, 8 * widen)
            # The 128-row tile is NOT granted the same exception. It is first in
            # the table, and at `E = 72` it displaces `(64, 32)` and runs 70%
            # slower: 12.29 ms against 7.24 on SAM 2's global attention. Fitting
            # is not the same as being worth it.
            @test DNNKernels.flashcm_tiling(dev, 72, 4096, 4096) == (64, 32, 8 * widen)
            # And only there: at `E = 72` the wider block fits and is faster
            # (7.67 ms against 7.89), so that shape keeps it.
            @test DNNKernels.flashcm_tiling(dev, 72, 4096, 4096, 8) == (64, 32, 8 * widen)
            # The 64-token key axis the narrow entry was added for is unmoved.
            @test DNNKernels.flashcm_tiling(dev, 72, 4096, 64, 8) == (64, 16, 8 * widen)
            # Every shipped tiling must satisfy the write-out loop's own
            # divisibility, which `flashcmfits` cannot see (it takes the padded
            # head dimension, and the write-out uses the real one).
            # `dev.coopmatsubgroup`, NOT `dev.subgroup`. They are the same 32 on
            # wave32 hardware, which is why this passed there; on RDNA 3.5 the
            # device default is 64 while Lava pins a cooperative-matrix module to
            # 32, so `dev.subgroup` overstates the workgroup by 2x and every
            # shipped tiling "failed" divisibility. `flashcm_tiling` itself uses
            # `NT = NW * dev.coopmatsubgroup` (flash.jl:1116) and its docstring
            # spells out which of the two the tiling needs.
            for (BR, BC, NW) in DNNKernels.FLASHCM_TILINGS
                @test (BR * 72) % (NW * dev.coopmatsubgroup) == 0
            end
            # A scaled `NW` is only taken where it is admissible: `32x32` at
            # `E = 72` fails `(BR * E) % NT == 0` at twice the subgroups, so that
            # entry keeps the table's own width even on a widening device.
            if widen > 1
                @test (32 * 72) % (8 * widen * dev.coopmatsubgroup) != 0
                @test DNNKernels.flashcm_tiling(dev, 72, 32, 32) == (32, 32, 8)
            end
            # Where `O` lives follows the tiling and not the device: `rego` is on
            # exactly where it costs at most 10 floats a thread, which is the
            # crossing both cards' sweeps found.
            qkv = ntuple(_ -> fill!(KA.allocate(back, Float16, 72, 64, 1, 1),
                                     Float16(0)), 3)
            for (BR, BC, NW) in ((64, 32, 8), (64, 32, 16), (32, 32, 8))
                pl = DNNKernels.flashcm_plan(dev, qkv..., nothing; BR, BC, NW)
                pl isa DNNKernels.Decline && continue
                @test pl.rego == ((BR * pl.EP) ÷ pl.NT <= 10)
            end
        end

        @testset "a key length no tiling divides still takes flash" begin
            # Qwen-Image 2.1 at 1024x1024 over a 22-token prompt: 4096 image
            # queries against 4096 + 22 keys. 4118 is 2 x 29 x 71, so no tile
            # divides it, the strict plan declines, and the declared path fell
            # through to writing a 4096 x 4118 score matrix per layer — 40.2 s
            # per denoising step against 9.6 s at a key length that divides.
            f16(E, L, H, B) = DNNKernels.toback(back, zeros(Float16, E, L, H, B))
            q, k = f16(128, 4096, 32, 1), f16(128, 4118, 32, 1)
            @test DNNKernels.flashcm_plan(dev, q, k, k, nothing) isa DNNKernels.Decline
            padded = DNNKernels.flashcm_padded_plan(dev, q, k, k, nothing)
            @test padded isa DNNKernels.FlashCMPlan
            @test padded.clamp
            # The padding is what it costs: one tile of 4118 keys, 0.3%.
            @test cld(4118, padded.BC) * padded.BC <= 1.01 * 4118
            q = k = nothing; GC.gc()

            # And it stays refused where padding is not cheap. Seventeen queries
            # take a 32-row tile and throw away 88% of it — the kind of waste
            # that measured +2.12 ms of SAM 2 encode for nothing.
            q, k = f16(128, 17, 32, 1), f16(128, 4096, 32, 1)
            @test DNNKernels.flashcm_plan(dev, q, k, k, nothing) isa DNNKernels.Decline
            @test DNNKernels.flashcm_padded_plan(dev, q, k, k, nothing) isa DNNKernels.Decline
            q = k = nothing; GC.gc()

            # The one shape padded at any occupancy: a query shorter than a tile
            # against exactly one key tile.
            q, k = f16(72, 4, 1, 1), f16(72, 16, 1, 1)
            @test DNNKernels.flashcm_padded_plan(dev, q, k, k, nothing) isa DNNKernels.FlashCMPlan
            q = k = nothing; GC.gc()
        end

        @testset "clamped: extents that do not divide the tile" begin
            # The decoder's shapes, which are why CLAMP exists: every one has a
            # 23 in it. Includes Lk = 23 < BC, where the key-block count has to
            # be `cld` — `div` gives ZERO blocks and the loop never runs, which
            # is a silently wrong answer rather than a crash.
            let
                for (E, Lq, Lk, H, B) in ((16, 23, 4096, 8, 1), (32, 23, 23, 8, 1),
                                          (16, 4096, 23, 8, 1), (16, 23, 17, 4, 1))
                    f16r(s, L) = DNNKernels.toback(back,
                                    Float16.(randn(Float32, E, L, H, B) .* 0.2f0))
                    q, k, v = f16r(1, Lq), f16r(2, Lk), f16r(3, Lk)
                    scale = Float32(1/sqrt(E))
                    t = DNNKernels.flashcm_tiling(dev, E, Lq, Lk; clamp=true)
                    @test t !== nothing
                    o = KA.allocate(back, Float32, E, Lq, H, B); fill!(o, 0f0)
                    @test DNNKernels.sdpaflashcm!(ctx, o, q, k, v, scale; clamp = true,
                                                  BR = t[1], BC = t[2], NW = t[3])
                    KA.synchronize(back)
                    got = Array(o)
                    ref = attnref(Float32.(Array(q)), Float32.(Array(k)), Float32.(Array(v)), scale)
                    @test maximum(abs, got) > 1e-3
                    @test maximum(abs, got .- ref) / maximum(abs, ref) < 5e-3
                    q = k = v = o = nothing; GC.gc()
                end
            end
        end

        @testset "the one-pass softmax agrees, including past its headroom" begin
            # `onepass` exponentiates against the PREVIOUS block's
            # maximum and defers the correction, which is exact in exact
            # arithmetic but not bit-identical: `ps` is fp16 and now rounds at a
            # different scale, so this is a tolerance and not `==`.
            #
            # The `4096` case is the one that matters. It is 128 key blocks, so
            # the deferred correction is applied 128 times, and `Lk` that long is
            # also where a row is most likely to meet a block hotter than
            # `FLASH_EXP_HEADROOM` and take the two-pass fallback — the branch
            # that would otherwise never run in a test.
            for (E, Lq, Lk, H, B) in ((72, 128, 256, 2, 2), (72, 64, 4096, 2, 1))
                f16r(s, L) = DNNKernels.toback(back,
                                Float16.(randn(Float32, E, L, H, B) .* 0.2f0))
                q, k, v = f16r(1, Lq), f16r(2, Lk), f16r(3, Lk)
                scale = Float32(1/sqrt(E))
                outs = map((false, true)) do op
                    o = KA.allocate(back, Float32, E, Lq, H, B); fill!(o, 0f0)
                    @test DNNKernels.sdpaflashcm!(ctx, o, q, k, v, scale; onepass = op)
                    KA.synchronize(back)
                    Array(o)
                end
                @test maximum(abs, outs[1]) > 1e-3
                @test maximum(abs, outs[1] .- outs[2]) / maximum(abs, outs[1]) < 1e-3
                q = k = v = nothing; GC.gc()
            end
        end

        @testset "skipping the rescale is exact, not approximate" begin
            # `lazyrescale` skips `O *= exp(m_old - m_new)` on blocks
            # where no row's max moved, i.e. where the factor is exactly one. If
            # that reasoning is ever wrong the two answers differ, so they are
            # compared to each other rather than to a tolerance.
            E, L, H, B = 72, 256, 2, 2
            f16r(s) = DNNKernels.toback(back, Float16.(randn(Float32,E,L,H,B) .* 0.2f0))
            q, k, v = f16r(1), f16r(2), f16r(3)
            scale = Float32(1/sqrt(E))
            outs = map((false, true)) do lazy
                o = KA.allocate(back, Float32, E,L,H,B); fill!(o, 0f0)
                DNNKernels.sdpaflashcm!(ctx, o, q, k, v, scale;
                                        lazyrescale = lazy, onepass = false)
                KA.synchronize(back)
                Array(o)
            end
            @test outs[1] == outs[2]
            q = k = v = nothing; GC.gc()
        end

        @testset "cooperative-matrix flash agrees with the path it replaces" begin
            # This guards the ROUTING, not the kernel: `sdpa` must recognise the
            # shape and fuse it, and the fused answer must match the two-GEMM
            # path it displaced. Flipping a `FLASHCM` switch is not how the
            # second answer is obtained, since there is none, so the alternative
            # path is now called by name — which also makes the comparison
            # independent of what `sdpa` would fall through to next.
            E, L, H, B = 72, 256, 2, 2
            f16r(s) = DNNKernels.toback(back, Float16.(randn(Float32,E,L,H,B) .* 0.2f0))
            q, k, v = f16r(1), f16r(2), f16r(3)
            wctx = DNNKernels.Ctx(back; ws = DNNKernels.nothing)
            scale = Float32(1/sqrt(E))
            # A plan, not a `Bool`: the same call that decides also carries the
            # tiling, so there is nothing left for a second predicate to disagree
            # with.
            @test DNNKernels.flashcm_plan(dev, q, k, v, nothing) isa DNNKernels.FlashCMPlan
            DNNKernels.reset!(wctx.ws)
            fused = Array(DNNKernels.sdpa(wctx, q, k, v, nothing, scale))
            KA.synchronize(back)
            cmplan = DNNKernels.coopmat_sdpa_plan(dev, q, k, v, nothing)
            @test cmplan isa DNNKernels.CoopMatSDPAPlan
            DNNKernels.reset!(wctx.ws)
            o = KA.allocate(back, Float32, E, L, H, B); fill!(o, 0f0)
            DNNKernels.sdpa_coopmat!(wctx, o, cmplan, q, k, v, scale)
            KA.synchronize(back)
            twogemm = Array(o)
            @test maximum(abs, twogemm) > 1e-3
            @test maximum(abs, fused .- twogemm) / maximum(abs, twogemm) < 5e-3
            q = k = v = o = nothing; GC.gc()
        end
    end
end

@testset "the tiling chooser prefers occupancy when the grid cannot fill the device" begin
    # `FLASHCM_TILINGS` is ordered fastest-first *for a launch that fills the
    # card*. SAM 2's decoder is where none does: `Lq = 23` (the mask prompt's
    # token count) against `Lk = 4096` is one query block, so the grid is
    # `1 * H*B` = 8 workgroups on 48 SMs and the kernel measures 0.10 TFLOP/s.
    # Below one workgroup per shader core the chooser therefore picks for grid
    # size instead.
    # Capabilities belong to the concrete execution device.  A KA backend is
    # only a launch descriptor and deliberately carries no device identity.
    dev = DNNKernels.Ctx(LavaBackend()).dev
    tiling(args...) = DNNKernels.flashcm_tiling(dev, args...; clamp = true)

    # Without a batch count it must behave exactly as it always did.
    @test tiling(16, 23, 4096) == tiling(16, 23, 4096, 0)

    with = tiling(16, 23, 4096, 8)
    @test with !== nothing
    @test cld(23, with[1]) * 8 > cld(23, tiling(16, 23, 4096)[1]) * 8
    # ...and the choice is still a legal tiling for the shape.
    @test 2 * 23 >= with[1]

    # The other two decoder shapes: one is already wide, one is not.
    @test tiling(16, 4096, 23, 8) == tiling(16, 4096, 23)
    @test tiling(32, 23, 23, 8) !== nothing

    @testset "inert on every encoder shape" begin
        # These launch 512 workgroups and must keep the table's own order, or the
        # rule has quietly re-tuned the encoder against a decoder measurement.
        for (E, Lq, Lk, nb) in [(72, 4096, 4096, 8), (72, 256, 256, 128),
                                (72, 512, 512, 32), (72, 1024, 1024, 16)]
            @test tiling(E, Lq, Lk, nb) == tiling(E, Lq, Lk)
        end
    end

    @testset "a grid at the threshold is not re-chosen" begin
        # The core count is a floor, not a target: at or above it the table's
        # order stands. Asked by describing a one-core device rather than by
        # writing to a global and restoring it — which a failing `@test` inside
        # the `try` would have skipped, leaving it flipped for everything after.
        one = DNNKernels.M.DeviceCaps(dev; cores = 1)
        onetiling(args...) = DNNKernels.flashcm_tiling(one, args...; clamp = true)
        @test onetiling(16, 23, 4096, 8) == onetiling(16, 23, 4096)
    end
end

@testset "plans: one decision, and a refusal that says why" begin
    back = LavaBackend()
    ctx = DNNKernels.Ctx(back)
    dev = ctx.dev
    E, L, H, B = 72, 256, 2, 1
    f16(s) = DNNKernels.toback(back, Float16.(randn(Float32, E, L, H, B) .* 0.2f0))
    q, k, v = f16(1), f16(2), f16(3)

    # ── What this replaced could drift; this cannot.
    #
    # `flashcm_applicable` ran `flashcm_tiling` and threw the answer away, `sdpa`
    # ran it again, and `sdpaflashcm!` then re-checked six more conditions and
    # could still decline — after `out` was allocated. The plan is the single
    # answer, and it carries the tiling that decision was made with.
    p = DNNKernels.flashcm_plan(dev, q, k, v, nothing)
    @test p isa DNNKernels.FlashCMPlan
    @test (p.BR, p.BC, p.NW) == DNNKernels.flashcm_tiling(dev, E, L, L, H * B)
    # coopmatsubgroup, not subgroup — see the comment at :166. The plan builds
    # NT as NW * dev.coopmatsubgroup (flash.jl:1250).
    @test p.NT == p.NW * dev.coopmatsubgroup
    @test p.EP == cld(E, dev.tile) * dev.tile

    # ── Every refusal is named, so a caller can react and a test can assert
    # which rule fired. `nothing` could do neither.
    @test DNNKernels.flashcm_plan(dev, q, k, v, k).reason === :bias
    @test DNNKernels.flashcm_plan(dev, q, k, v, nothing; BR = 64, BC = 64,
                                  NW = 8).reason === :tiling
    f32 = DNNKernels.toback(back, randn(Float32, E, L, H, B))
    @test DNNKernels.flashcm_plan(dev, f32, k, v, nothing).reason === :eltype
    nocm = DNNKernels.M.DeviceCaps(dev; coopmat = false)
    @test DNNKernels.flashcm_plan(nocm, q, k, v, nothing).reason === :nocoopmat
    @test DNNKernels.flashcm_plan(dev, q, k, v, nothing; BR = 23).reason === :extent

    # ── A plan is an answer about a device, so it can be asked about one this
    # machine does not have. `NW * 32` was a literal in three places, and on a
    # wave64 part it names half the workgroup the tiling assumes.
    # An RDNA3-shaped device: default width 64, but Lava PINS coopmat modules to
    # 32, so the workgroup must still be sized in 32s. Sizing it in 64s is the
    # bug this field exists to prevent, and it is invisible on this card.
    w64 = DNNKernels.M.DeviceCaps(true, 16, 64, 32, 65536, 1024, 40, 0)
    p64 = DNNKernels.flashcm_plan(w64, q, k, v, nothing)
    @test p64 isa DNNKernels.FlashCMPlan
    @test p64.NT == p64.NW * 32              # …not * 64
    @test w64.subgroup == 64                 # the device default is still 64

    # Native wave32 does not mean a narrow workgroup is the occupancy winner.
    # A backend that reports 64 resident subgroups takes the measured 16-wave
    # form without pretending its subgroup width is 64.
    # One synthetic core keeps this test about the workgroup-width choice; a
    # many-core device with this deliberately tiny `qkv` would instead exercise
    # the decoder rule above and choose a smaller BR to create more workgroups.
    resident64 = DNNKernels.M.DeviceCaps(true, 16, 32, 32, 65536, 1024, 1, 64)
    @test DNNKernels.flashcm_tiling(resident64, 72, 4096, 4096) == (128, 16, 16)
    @test DNNKernels.flashcm_tiling(resident64, 72, 256, 256) == (128, 32, 16)
    @test DNNKernels.flashcm_tiling(resident64, 72, 64, 64) == (64, 16, 16)
    pr64 = DNNKernels.flashcm_plan(resident64, q, k, v, nothing)
    @test pr64 isa DNNKernels.FlashCMPlan
    @test pr64.NW == 16
    @test pr64.NT == 512
    @test !pr64.rego
    @test pr64.held
    # The selected 128-row tile fits only because the held direct store removes
    # `pvs`; explicitly disabling that algorithm makes this candidate decline
    # instead of launching beyond its shared-memory budget.
    @test DNNKernels.flashcm_plan(resident64, q, k, v, nothing;
                                  held = false).reason === :tiling
    # Explicit tuning still opts into the algorithm associated with that width;
    # only the automatic occupancy widening preserves the table's arithmetic.
    @test DNNKernels.flashcm_plan(resident64, q, k, v, nothing;
                                  BR = 64, BC = 32, NW = 16).rego

    # A long global key loop keeps the same arithmetic too: fragment holding,
    # not the scalar register accumulator, is the measured winner there.
    qlong = KA.allocate(back, Float16, E, 4096, H, B)
    plong = DNNKernels.flashcm_plan(resident64, qlong, qlong, qlong, nothing)
    @test plong isa DNNKernels.FlashCMPlan
    @test !plong.rego
    @test plong.held

    # ── The clamp is per run, not per process. Two contexts, opposite policies,
    # both alive at once — which the `Ref` it replaced could not express, and
    # which is why SAM 2 had to set it and restore it in a `finally`.
    dctx = DNNKernels.Ctx(back; clampattn = true)
    @test !ctx.clampattn && dctx.clampattn
    E2, Lq2 = 16, 23
    g16(s, L) = DNNKernels.toback(back, Float16.(randn(Float32, E2, L, 8, 1) .* 0.2f0))
    q2, k2, v2 = g16(1, Lq2), g16(2, 4096), g16(3, 4096)
    @test DNNKernels.flashcm_plan(ctx.dev, q2, k2, v2, nothing;
                                  clamp = ctx.clampattn) isa DNNKernels.Decline
    @test DNNKernels.flashcm_plan(dctx.dev, q2, k2, v2, nothing;
                                  clamp = dctx.clampattn) isa DNNKernels.FlashCMPlan
end

# The merge is the whole cost of a wide split, and how it is written decides
# whether the split runs at all.
#
# `splitcount` aims at two workgroups per shader core, so a one-query-block shape
# on a 96-core part asks for 64 splits. With the split count a `Val` and `E`
# looped inside each thread that was a 4608-iteration triple loop with
# compile-time bounds: the driver unrolled it whole, and the 19274-instruction
# shader it produced — 58 spilled VGPRs, 456 spilled SGPRs — never returned.
# `Lk = 4096` took the GPU down rather than giving a wrong answer, so no
# tolerance would have caught it, and 32 splits still fit, which is why every
# shorter key axis passed. (First seen on RADV; the shader is the same one
# everywhere.)
#
# Against the closed form rather than against the fused kernel, so it says what
# the merge owes regardless of what wrote the partials.
@testset "the split merge survives the counts the chooser asks for" begin
    back = LavaBackend()
    dev = DNNKernels.Ctx(back).dev

    # The count that hung is one the chooser asks for, not one a test invented.
    # Asked of a described device, so the number is the rule's and not this
    # machine's.
    caps = DNNKernels.M.DeviceCaps(dev; cores = 96)
    @test DNNKernels.splitcount(caps, 64, 4096, 64, 32, 2) == 64

    function mergeref(p, m)
        E, Lq, H, B, ns = size(p)
        o = zeros(Float32, E, Lq, H, B)
        for b in 1:B, h in 1:H, lq in 1:Lq
            mx = maximum(m[lq, h, b, sp, 1] for sp in 1:ns)
            w = [exp(m[lq, h, b, sp, 1] - mx) for sp in 1:ns]
            L = sum(w[sp] * m[lq, h, b, sp, 2] for sp in 1:ns)
            for e in 1:E
                o[e, lq, h, b] = sum(w[sp] * p[e, lq, h, b, sp] for sp in 1:ns) /
                                 (L == 0 ? 1 : L)
            end
        end
        o
    end

    for (E, Lq, H, B, ns) in ((72, 64, 2, 1, 64), (72, 23, 3, 2, 96))
        ph = randn(Float32, E, Lq, H, B, ns) .* 0.1f0
        mh = cat(randn(Float32, Lq, H, B, ns) .* 0.5f0,
                 rand(Float32, Lq, H, B, ns) .+ 1f0; dims = 5)
        out = KA.allocate(back, Float32, E, Lq, H, B); fill!(out, 0f0)
        # Flat, one element per thread: `n` over `(e, lq, h, b)` and `nrow`
        # over the `(lq, h, b)` the split bookkeeping is indexed by.
        g = DNNKernels.FLASH_MERGE_GROUP
        DNNKernels.attn_flash_cm_merge!(back, g)(out, DNNKernels.toback(back, ph),
                                                 DNNKernels.toback(back, mh),
                                                 Val(ns), Val(E),
                                                 Int32(E * Lq * H * B), Int32(Lq * H * B);
                                                 ndrange = cld(E * Lq * H * B, g) * g)
        KA.synchronize(back)
        @test maximum(abs, Array(out) .- mergeref(ph, mh)) /
              maximum(abs, mergeref(ph, mh)) < 1f-5
    end
end

# A key axis that does not divide the tile, in ONE launch.
#
# The clamped path costs 46% of this kernel and not because of what the mask
# does: a launch with `KCLAMP` set throughout measured 49.57 ms against 37.69
# at Qwen-Image 2.1's shape, for 22 ragged keys of 4118. So the clamp is
# confined to the blocks that are actually short — `ragged` in the kernel — and
# the padded score columns are masked once rather than tested in every softmax
# loop.
#
# This replaced a LAUNCH split: the bulk in one dispatch, the remainder in a
# second, and `attn_flash_cm_merge!` combining their partial softmaxes in a
# third. That form still exists behind `loopsplit = false`, which is the A side
# of the measurement above, and the two must agree numerically.
@testset "a ragged key axis is clamped per block, in one launch" begin
    back = LavaBackend()
    ctx = DNNKernels.Ctx(back)
    dev = ctx.dev
    if dev.coopmat && dev.coopmatsubgroup == 32
        E, Lq, Lk, H = 128, 512, 530, 8
        rng = MersenneTwister(5)
        mk(L) = DNNKernels.toback(back, Float16.(randn(rng, Float32, E, L, H, 1) .* 0.3f0))
        q, k, v = mk(Lq), mk(Lk), mk(Lk)
        scale = Float32(1 / sqrt(E))
        plan = DNNKernels.flashcm_plan(dev, q, k, v, nothing; clamp = true)
        @test plan isa DNNKernels.FlashCMPlan
        @test Lk % plan.BC != 0
        # The ragged axis no longer buys a split, so no `partial` and no merge.
        @test !plan.tailsplit
        @test plan.nsplit == 1

        partial = KA.allocate(back, Float32, E, Lq, H, 1, 2)
        ml = KA.allocate(back, Float32, Lq, H, 1, 2, 2)
        ls = DNNKernels.flash_launches(dev, q, plan, q, k, v, scale, partial, ml)
        @test length(ls) == 1
        @test ls[1].kern === DNNKernels.attn_flash_cm_spatial4!
        # `..., Lq, keys, lazyrescale, partial, ml`: the one launch is told the
        # whole key axis, ragged end included.
        keysof(l) = Int(l.args[end - 3])
        @test keysof(ls[1]) == Lk

        out = KA.allocate(back, Float32, E, Lq, H, 1); fill!(out, 0f0)
        DNNKernels.sdpaflashcm!(ctx, out, plan, q, k, v, scale)
        KA.synchronize(back)
        got = Array(out)

        ref = zeros(Float32, E, Lq, H, 1)
        for h in 1:H
            qh = Float32.(Array(q)[:, :, h, 1])
            kh = Float32.(Array(k)[:, :, h, 1])
            vh = Float32.(Array(v)[:, :, h, 1])
            s = (qh' * kh) .* scale
            for i in 1:Lq
                p = exp.(s[i, :] .- maximum(s[i, :]))
                ref[:, i, h, 1] = vh * (p ./ sum(p))
            end
        end
        @test maximum(abs, got .- ref) / maximum(abs, ref) < 5e-3

        # The keys past the extent must be EXCLUDED, not merely zeroed.
        #
        # Staging writes zeros for them, which gives each a SCORE of zero — and
        # a weight of `exp(0 - m)`, which is not zero. It lands in `l` and
        # divides the whole row. The `-Inf` mask on the score tile is what
        # removes it, and this is the shape that says so: every real score is
        # the same negative number, so `m` is negative, so a padded key would
        # outweigh a real one.
        #
        # `q = -c`, `k = c` constant gives every real score
        # `-|c|^2 E / sqrt(E) = -2.83`, against `0` for a pad — so the PADS set
        # the row maximum and each weighs `1` against a real key's
        # `exp(-2.83) = 0.059`. Fourteen of them against 530 real keys would
        # scale the row by `530*0.059 / (530*0.059 + 14) = 0.69`: a 31% error,
        # not a rounding one.
        nfull = div(Lk, plan.BC) * plan.BC
        npad = cld(Lk, plan.BC) * plan.BC - Lk
        @test 0 < npad < plan.BC
        c = Float16(0.5)
        qc = DNNKernels.toback(back, fill(-c, E, Lq, H, 1))
        kc = DNNKernels.toback(back, fill(c, E, Lk, H, 1))
        vc = DNNKernels.toback(back, Float16.(randn(rng, Float32, E, Lk, H, 1) .* 0.3f0))
        outc = KA.allocate(back, Float32, E, Lq, H, 1); fill!(outc, 0f0)
        DNNKernels.sdpaflashcm!(ctx, outc, plan, qc, kc, vc, scale)
        KA.synchronize(back)
        # Equal scores means a plain mean over the REAL keys, per head.
        meanv = reshape(sum(Float32.(Array(vc)); dims = 2) ./ Lk, E, 1, H, 1)
        gotc = Array(outc)
        @test maximum(abs, gotc .- meanv) / maximum(abs, meanv) < 5e-3
        # And the failure it is guarding against would look like this, so the
        # test cannot pass by both being wrong.
        realw = exp(-Float64(E) * Float64(c)^2 * scale)
        bad = Lk * realw / (Lk * realw + npad)
        @test bad < 0.75
        @test maximum(abs, gotc .- meanv .* bad) / maximum(abs, meanv) > 0.1

        # And the same numbers as the launch split it replaced. Not bit-identical
        # and cannot be: the merge sums the two slices' contributions under a
        # common maximum, so the fp32 accumulation is reassociated. Two orders of
        # magnitude inside the fp16 output's own resolution is the claim.
        split = DNNKernels.flashcm_plan(dev, q, k, v, nothing; clamp = true,
                                        loopsplit = false)
        @test split.tailsplit
        @test split.nsplit == 2
        sls = DNNKernels.flash_launches(dev, q, split, q, k, v, scale, partial, ml)
        @test length(sls) == 3
        @test sls[3].kern === DNNKernels.attn_flash_cm_merge!
        @test keysof(sls[1]) + keysof(sls[2]) == Lk
        @test keysof(sls[1]) == nfull
        @test 0 < keysof(sls[2]) < plan.BC
        fill!(out, 0f0)
        DNNKernels.sdpaflashcm!(ctx, out, split, q, k, v, scale)
        KA.synchronize(back)
        @test maximum(abs, got .- Array(out)) / maximum(abs, ref) < 1e-4

        # Writing the spatial order directly is the one thing a tail split
        # cannot do: its merge writes `out`, so the permutation would simply
        # not happen and nothing would say so. One launch has no such problem.
        @test_throws ArgumentError DNNKernels.flash_launches(
            dev, q, split, q, k, v, scale, partial, ml; outperm = true)

        # A key axis the tile divides takes the same single launch, with every
        # `ragged` branch folded away.
        k2, v2 = mk(512), mk(512)
        plain = DNNKernels.flashcm_plan(dev, q, k2, v2, nothing; clamp = true)
        @test !plain.tailsplit
        @test length(DNNKernels.flash_launches(dev, q, plain, q, k2, v2, scale,
                                               partial, ml)) == 1
    else
        @test_skip false
    end
end

# K and V read as cooperative matrices, rather than staged through shared.
#
# `OpCooperativeMatrixLoadKHR` takes a pointer in any storage class, so the tile
# the tensor core wants can come from the tensor. What that removes is a store
# to shared, a barrier and a load, for both operands, on every key block:
# **37.36 ms against 25.56** at Qwen-Image 2.1's attention, with the same
# numbers out. It also changes which TILING is fastest, which is why the plan
# carries the decision rather than the launcher re-deriving it.
@testset "K and V as tiles: same numbers, and the window that keeps them in bounds" begin
    back = LavaBackend()
    ctx = DNNKernels.Ctx(back)
    dev = ctx.dev

    # The table choice needs no device, so it is asked first and unconditionally.
    @testset "the tiling follows the operand source" begin
        for (E, Lq, Lk, H) in ((128, 4096, 4118, 32), (64, 4096, 4096, 8))
            tiled = DNNKernels.flashcm_tiling(dev, E, Lq, Lk, H; clamp = true,
                                              globalkv = true)
            staged = DNNKernels.flashcm_tiling(dev, E, Lq, Lk, H; clamp = true,
                                               globalkv = false)
            @test tiled !== nothing && staged !== nothing
            @test tiled != staged
            # A 16-row block and the SMALLEST workgroup that can write it out,
            # which is where the measured optimum sits for both head widths.
            # `flashcmfits` already owns that rule — the write-out is
            # `@nexprs 3`, so a launch needs `cld(RT * ET, NW) <= 3` subgroups —
            # and the table's job is only to offer the narrow entries first.
            @test tiled[1] == 16
            EP = cld(E, dev.tile) * dev.tile
            tiles = div(tiled[1], dev.tile) * div(EP, dev.tile)
            @test cld(tiles, tiled[3]) <= 3           # this one can write out
            @test cld(tiles, tiled[3] ÷ 2) > 3        # and nothing narrower can
            # Never wider than the 128 threads AOTriton tunes to here.
            @test tiled[3] * dev.coopmatsubgroup <= 128
            # The staged table goes the other way: it widens for the staging.
            @test staged[3] > tiled[3]
        end
    end

    if dev.coopmat && dev.coopmatsubgroup == 32
        E, Lq, H = 128, 512, 8
        rng = MersenneTwister(9)
        mk(L, x = 0.3f0) = DNNKernels.toback(back, Float16.(randn(rng, Float32, E, L, H, 1) .* x))
        scale = Float32(1 / sqrt(E))

        @testset "the four conditions" begin
            q, k, v = mk(Lq), mk(530), mk(530)
            p = DNNKernels.flashcm_plan(dev, q, k, v, nothing; clamp = true)
            @test p.globalkv
            st = DNNKernels.flashstrides(k)
            # E contiguous, no head padding, one whole tile of keys. Each is a
            # thing the staging did that a tile load cannot.
            @test DNNKernels.flashglobalkv(p.E, p.EP, st, st, 530, p.BC)
            @test !DNNKernels.flashglobalkv(p.E, p.EP, (2, 1, 1, 1), st, 530, p.BC)
            @test !DNNKernels.flashglobalkv(p.E, p.EP, st, (2, 1, 1, 1), 530, p.BC)
            @test !DNNKernels.flashglobalkv(72, 80, st, st, 530, p.BC)
            @test !DNNKernels.flashglobalkv(p.E, p.EP, st, st, p.BC - 1, p.BC)
        end

        @testset "staged and tiled agree, ragged or not" begin
            for Lk in (512, 530)
                q, k, v = mk(Lq), mk(Lk), mk(Lk)
                p = DNNKernels.flashcm_plan(dev, q, k, v, nothing; clamp = true)
                got = map((false, true)) do g
                    o = KA.allocate(back, Float32, E, Lq, H, 1); fill!(o, 0f0)
                    DNNKernels.sdpaflashcm!(ctx, o, p, q, k, v, scale; globalkv = g)
                    KA.synchronize(back)
                    Array(o)
                end
                ref = zeros(Float32, E, Lq, H, 1)
                for h in 1:H
                    qh = Float32.(Array(q)[:, :, h, 1])
                    kh = Float32.(Array(k)[:, :, h, 1])
                    vh = Float32.(Array(v)[:, :, h, 1])
                    sc = (qh' * kh) .* scale
                    for i in 1:Lq
                        pr = exp.(sc[i, :] .- maximum(sc[i, :]))
                        ref[:, i, h, 1] = vh * (pr ./ sum(pr))
                    end
                end
                for o in got
                    @test maximum(abs, o) > 1e-3
                    @test maximum(abs, o .- ref) / maximum(abs, ref) < 5e-3
                end
                # The two paths differ only in where the operand bytes came
                # from, so they must agree far more tightly than either agrees
                # with the reference.
                @test maximum(abs, got[1] .- got[2]) / maximum(abs, ref) < 1e-5
            end
        end

        @testset "a slid window counts the keys it slides over once" begin
            # A ragged block's tile starts at `Lk - BC` so the load stays inside
            # the tensor, which brings keys the PREVIOUS block already counted
            # into view. They are masked to `-Inf`; if they were not, they would
            # be counted twice.
            #
            # Constant `q = -c`, `k = c` makes every real score equal, so the
            # answer is a plain mean over the real keys and a double-counted key
            # moves it by its own share.
            Lk = 530
            c = Float16(0.5)
            q = DNNKernels.toback(back, fill(-c, E, Lq, H, 1))
            k = DNNKernels.toback(back, fill(c, E, Lk, H, 1))
            v = mk(Lk)
            p = DNNKernels.flashcm_plan(dev, q, k, v, nothing; clamp = true)
            @test p.globalkv && Lk % p.BC != 0
            o = KA.allocate(back, Float32, E, Lq, H, 1); fill!(o, 0f0)
            DNNKernels.sdpaflashcm!(ctx, o, p, q, k, v, scale)
            KA.synchronize(back)
            got = Array(o)

            vh = Float32.(Array(v))
            meanv = reshape(sum(vh; dims = 2) ./ Lk, E, 1, H, 1)
            @test maximum(abs, got .- meanv) / maximum(abs, meanv) < 5e-3

            # What double-counting would have produced: the slid-over keys are
            # the ones between the last whole tile and where the block began.
            nfull = div(Lk, p.BC) * p.BC
            dup = (Lk - p.BC + 1):nfull
            @test length(dup) == p.BC - (Lk - nfull)
            bad = reshape((sum(vh; dims = 2) .+ sum(vh[:, dup, :, :]; dims = 2)) ./
                          (Lk + length(dup)), E, 1, H, 1)
            @test maximum(abs, bad .- meanv) / maximum(abs, meanv) > 0.02
            @test maximum(abs, got .- bad) / maximum(abs, meanv) > 0.02
        end

        @testset "asking for it where it does not hold is an error, not a bad read" begin
            q, k, v = mk(Lq), mk(16), mk(16)
            p = DNNKernels.flashcm_plan(dev, q, k, v, nothing; clamp = true)
            @test p isa DNNKernels.FlashCMPlan
            @test !p.globalkv          # 16 keys is under one 32-wide block
            part = KA.allocate(back, Float32, E, Lq, H, 1, 2)
            ml = KA.allocate(back, Float32, Lq, H, 1, 2, 2)
            @test_throws ArgumentError DNNKernels.flash_launches(
                dev, q, p, q, k, v, scale, part, ml; globalkv = true)
        end
    else
        @test_skip false
    end
end

# Which softmax a plan takes reaches the kernel as a `Val`, not as a uniform
# argument.
#
# That is what lets a two-pass plan lose the one-pass loop AND the deferred
# rescale that only the one-pass form reaches: `onep` is statically false, so
# `pre` is a compile-time `true`. As an `Int32` neither folded, because the
# dead loop's stores go to shared memory. Measured on an 8060S at Qwen-Image
# 2.1's attention (`Lq = 4096`, `Lk = 4118` clamped, `E = 128`, 32 heads):
# 36.6 ms against 38.6 interleaved, and the whole 20B denoising step 5.85 s
# against 6.02, with bit-identical output.
@testset "the softmax form is compiled in, not passed in" begin
    back = LavaBackend()
    ctx = DNNKernels.Ctx(back)
    dev = ctx.dev
    if dev.coopmat
        E, Lq, H = 128, 1024, 8
        rng = MersenneTwister(7)
        mk(E) = DNNKernels.toback(back, Float16.(randn(rng, Float32, E, Lq, H, 1) .* 0.3f0))
        scale = Float32(1 / sqrt(E))
        part = KA.allocate(back, Float32, E, Lq, H, 1, 2)
        ml = KA.allocate(back, Float32, Lq, H, 1, 2, 2)
        # The merge launch carries none of these arguments, so ask the spatial
        # ones. This grid fills the device, so there is no merge to filter out
        # at all, which is what the first assertion checks.
        launches(p, q, k, v; kw...) =
            filter(l -> l.kern === DNNKernels.attn_flash_cm_spatial4!,
                   DNNKernels.flash_launches(dev, q, p, q, k, v, scale, part, ml; kw...))

        # Which argument is which, FOUND rather than counted.
        #
        # These used to be offsets from the end of the tuple, and that broke
        # twice as `Val`s were added ahead of them — silently, because a wrong
        # offset still lands on some other `Val{false}`. So flip the setting and
        # take the position that moved: if it is not exactly one position, the
        # argument list changed in a way this test must be told about.
        #
        # `Val` positions only. The tensor arguments are rebuilt per call —
        # `flashflat(stridedroot(q))` is a fresh wrapper each time — so they
        # differ under `!==` on every comparison and say nothing.
        function movedval(a, b, what)
            length(a) == length(b) || error("$what changed the argument COUNT")
            d = findall(i -> a[i] isa Val && a[i] !== b[i], eachindex(a))
            length(d) == 1 || error("$what moved $(length(d)) `Val`s, not 1")
            only(d)
        end
        argat(p, q, k, v; kw...) =
            movedval(launches(p, q, k, v)[1].args,
                     launches(p, q, k, v; kw...)[1].args, keys(kw))

        q, k, v = mk(E), mk(E), mk(E)
        two = DNNKernels.flashcm_plan(dev, q, k, v, nothing)
        @test two isa DNNKernels.FlashCMPlan && !two.onepass
        @test length(launches(two, q, k, v)) == 1

        one = DNNKernels.flashcm_plan(dev, q, k, v, nothing; onepass = true)
        @test one.onepass
        # The softmax form is a plan field rather than a keyword, so its
        # position comes from the two plans differing in exactly that `Val`.
        onepassat = movedval(launches(two, q, k, v)[1].args,
                             launches(one, q, k, v)[1].args, :onepass)
        @test all(l -> l.args[onepassat] === Val(false), launches(two, q, k, v))
        @test all(l -> l.args[onepassat] === Val(true), launches(one, q, k, v))

        # No runtime flag is left to disagree with the `Val`: the trailing
        # arguments are `Lq`, `keys`, `lazyrescale` and the two split buffers.
        @test all(a -> !(a isa Int32), last(launches(two, q, k, v)).args[end-1:end])
        @test last(launches(two, q, k, v)).args[end - 2] isa Int32

        # The two "what is this kernel waiting on" diagnostics are wrong when
        # on, so nothing in the library may turn them on.
        @test all(l -> l.args[argat(two, q, k, v; smoff = true)] === Val(false),
                  launches(two, q, k, v))
        @test all(l -> l.args[argat(two, q, k, v; ldoff = 1)] === Val(0),
                  launches(two, q, k, v))
        # Same for the fp16 score accumulator, which changes numerics, and for
        # the barrier ablation, which races.
        @test all(l -> l.args[argat(two, q, k, v; s16 = true)] === Val(false),
                  launches(two, q, k, v))
        @test all(l -> l.args[argat(two, q, k, v; baroff = true)] === Val(false),
                  launches(two, q, k, v))
        # And the ragged-block clamp IS on: it is what makes a key axis the
        # tile does not divide one launch instead of three.
        @test all(l -> l.args[argat(two, q, k, v; loopsplit = false)] === Val(true),
                  launches(two, q, k, v))

        # That the two forms agree numerically is its own testset above; this
        # one is about which of them the kernel is compiled for.
    else
        @test_skip false
    end
end

# One pass over the scores or two: head width AND rows per warp.
#
# One pass reads each score once and redoes the block when a row's maximum grew
# past the fp16 headroom; two passes read every score twice and never redo. The
# width half is the older measurement: at `Lq = Lk = 4096`, two passes win by
# 7.6% at `E = 128` and 3.1% at 96, one pass by 2.8% at 80 and 3.8% at 64.
#
# Rows per warp is the second half and it dominates, because the more rows a
# warp owns the likelier one of them forces the redo. Measured at `L = 4096`,
# two passes winning:
#
#     BR/NW   E=64 H=8   E=64 H=32   E=80 H=8
#       8      -19.4%      -50.4%     -20.2%
#       4       +0.3%       +0.8%      -2.6%
#       2       +6.5%
#
# The width rule alone was calibrated when the chooser picked large tiles; it
# now picks `BR = 16, NW = 2` for several shapes, where one pass costs 19% to
# 50%.
@testset "the score pass count follows width and rows per warp" begin
    back = LavaBackend()
    dev = DNNKernels.Ctx(back).dev
    if dev.coopmat
        mk(E) = DNNKernels.toback(back, zeros(Float16, E, 256, 4, 1))
        plan(E; kw...) = DNNKernels.flashcm_plan(dev, mk(E), mk(E), mk(E), nothing; kw...)
        wide = plan(128)
        narrow = plan(64)
        @test wide isa DNNKernels.FlashCMPlan && narrow isa DNNKernels.FlashCMPlan
        # A wide head is two passes whatever the tile.
        @test !wide.onepass
        @test !plan(96).onepass
        @test !plan(128; BC = 32, BR = 16, NW = 4).onepass
        # A narrow head follows the tile: 4 rows a warp or fewer keeps one pass,
        # 8 does not.
        @test plan(64; BC = 32, BR = 16, NW = 4).onepass
        @test plan(80; BC = 32, BR = 16, NW = 4).onepass
        @test !plan(64; BC = 32, BR = 16, NW = 2).onepass
        @test !plan(64; BC = 32, BR = 64, NW = 8).onepass
        # Whatever the chooser picked, the rule is the one above.
        @test narrow.onepass == (narrow.BR ÷ narrow.NW < 8)
        # And a caller who asks gets what it asked for, both ways.
        @test plan(128; onepass = true).onepass
        @test !plan(64; BC = 32, BR = 16, NW = 4, onepass = false).onepass
    else
        @test_skip false
    end
end
