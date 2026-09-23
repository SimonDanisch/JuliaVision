"""
ConvRot — Comfy's activation rotation — and the packed INT8 product that reads
its output. Both are what a compact Qwen-Image 2.1 checkpoint runs on every
matmul, and both were rewritten for speed after the model first ran:

  * the transform went from a pass per stage (eight traversals of the tensor) to
    two passes of sixteen elements in registers (four),
  * the declared GEMM stopped dequantising 100 MB of weights per product and
    reads the packed int8 directly, padding its column count to a tile.

What has to hold across both: the arithmetic the checkpoint was validated
against, at a sequence length no tile divides.
"""

import KernelInterface as KI
using Test, DNNKernels, Mantle, KernelAbstractions, Random

const DKA = DNNKernels

@testset "ConvRot and the packed product it feeds" begin
    # Whatever backend is loaded, not a named one: `Mantle.LavaBackend` exists only
    # where Lava does, so naming it made this file ERROR on a machine with a
    # different GPU rather than take the capability skip below that was written
    # for exactly this case.
    backend = first(Mantle.eachbackend())
    caps = DNNKernels.caps(backend)
    # `coopmatkernels`, not `caps.coopmat` alone: Metal reports cooperative
    # matrices (simdgroup matrices are real) while the staged GEMM these tests
    # exercise -- `GEMM_TILINGS`, `q8gemm_tiling`, `conv_coopmat_plan` -- is
    # compiled in only where Lava is. Asking the narrower question is what makes
    # the skip below fire instead of a `Decline(:host)` failing an assertion.
    if DNNKernels.coopmatkernels(caps) && caps.coopmatsubgroup == 32
        rng = MersenneTwister(11)
        # A pass per stage, on the host, rounding to fp16 in between — the form
        # the kernel replaced.
        function staged(x::Matrix{Float16}, G::Int)
            K = size(x, 1); src = copy(x); dst = similar(src)
            for stage in 0:(round(Int, log(4, G)) - 1)
                s = 4^stage; period = 4s
                for col in axes(src, 2), k in 0:K-1
                    base = (k ÷ period) * period + (k % s) + 1
                    a, b, c, d = Float32.((src[base, col], src[base + s, col],
                                           src[base + 2s, col], src[base + 3s, col]))
                    row = (k ÷ s) % 4
                    v = row == 0 ? a+b+c-d : row == 1 ? a+b-c+d :
                        row == 2 ? a-b+c+d : -a+b+c+d
                    dst[k + 1, col] = Float16(v * 0.5f0)
                end
                src, dst = dst, src
            end
            src
        end

        xh = Float16.(randn(rng, Float32, 512, 6) .* 0.3f0)
        x = DNNKernels.toback(backend, xh)
        o = KernelAbstractions.allocate(backend, Float16, 512, 6)
        n = Int64(length(xh))
        for (S, _) in ((1, nothing), (16, nothing))
            KI.Kernel(backend, DNNKernels.convrot_pass_kernel!)(
                o, S == 1 ? x : o, Val(S), Val(16), Val(256), n; ndrange = length(xh) ÷ 16, workgroupsize = 256)
        end
        KernelAbstractions.synchronize(backend)
        want = staged(xh, 256)
        # One fp16 ulp at this magnitude: the compiler reassociates the sum.
        @test maximum(abs, Float32.(Array(o)) .- Float32.(want)) <= 0.001
        # Orthogonal, and its own inverse — what makes an embedding row
        # recoverable by applying it twice.
        KI.Kernel(backend, DNNKernels.convrot_pass_kernel!)(o, o, Val(1), Val(16), Val(256), n;
                                                      ndrange = length(xh) ÷ 16, workgroupsize = 256)
        KI.Kernel(backend, DNNKernels.convrot_pass_kernel!)(o, o, Val(16), Val(16), Val(256), n;
                                                      ndrange = length(xh) ÷ 16, workgroupsize = 256)
        KernelAbstractions.synchronize(backend)
        @test maximum(abs, Float32.(Array(o)) .- Float32.(xh)) <= 0.01

        # The declared packed GEMM against the dequantise-and-reuse path it
        # replaced, at Qwen-Image's own column count: 4118 = 2 x 29 x 71, which
        # no tile divides, so this is the padding path.
        M, K, N = 256, 512, 4118
        A = DNNKernels.quantizeint8(backend,
            DNNKernels.toback(backend, Float16.(randn(rng, Float32, M, K) .* 0.2f0)))
        B = DNNKernels.toback(backend, Float16.(randn(rng, Float32, K, N) .* 0.2f0))
        @test DNNKernels.q8gemm_columns(N) == 4224
        @test DNNKernels.q8gemm_columns(64) == 64
        @test DNNKernels.q8gemm_tiling(caps, Float16, M, K, DNNKernels.q8gemm_columns(N)) !== nothing

        buf(id, shape; kind = :transient, dtype = Float16) =
            DKA.Buffer(id, kind, Any[shape...], dtype, id == "w" ? "w" : "", (0, 0), "", "",
                       Dict{String,Any}())
        buffers = Dict(b.id => b for b in (buf("x", (N, K); kind = :external),
                                           buf("w", (M, K); kind = :weight),
                                           buf("y", (N, M))))
        # `mm(x, w)` is `w * x` in the reversed layout — the weight is the
        # matrix operand, which is what `mmplan` sends to the int8 path.
        ops = [DKA.Op("y", "mm.default", ["x", "w"], "y", Dict{String,Any}())]
        g = DKA.Graph("q8", String[], ["x"], ["y"], buffers, collect(keys(buffers)), ops)
        plan = DNNKernels.planfor(Mantle.todevice(backend), g, Dict{String,Any}("w" => A), (;))
        got = Float32.(Array(first(DNNKernels.replay!(plan, "q8", (B,)))))
        Mantle.free!(plan.plan)

        ctx = DNNKernels.Ctx(Dict{String,Any}(), g, (;), backend)
        out = KernelAbstractions.allocate(backend, Float16, M, N)
        DNNKernels.matmul!(ctx, DNNKernels.MMInt8Plan(), out, A, B, nothing, identity)
        KernelAbstractions.synchronize(backend)
        want2 = Float32.(Array(out))
        @test size(got) == (M, N)
        @test maximum(abs, got .- want2) <= 0.02maximum(abs, want2)

        # The same product feeding something else, so its destination is
        # INTERNAL. `declare!` then builds that destination at the padded width
        # and the op's result is a view of it — no pass discards the padding,
        # which at Qwen-Image's widest product is 202 MB read and written.
        buffers2 = Dict(b.id => b for b in (buf("x", (N, K); kind = :external),
                                            buf("w", (M, K); kind = :weight),
                                            buf("y", (N, M)),
                                            buf("z", (N, M))))
        ops2 = [DKA.Op("y", "mm.default", ["x", "w"], "y", Dict{String,Any}()),
                DKA.Op("z", "clamp.default", ["y"], "z",
                       Dict{String,Any}("arg1" => -1000, "arg2" => 1000))]
        g2 = DKA.Graph("q8chain", String[], ["x"], ["z"], buffers2,
                       collect(keys(buffers2)), ops2)
        plan2 = DNNKernels.planfor(Mantle.todevice(backend), g2,
                                   Dict{String,Any}("w" => A), (;))
        chained = Float32.(Array(first(DNNKernels.replay!(plan2, "q8chain", (B,)))))
        @test count(p -> occursin("unpad", String(p.pass.name)), plan2.plan.passes) == 0
        @test maximum(abs, chained .- clamp.(want2, -1000, 1000)) <= 0.02maximum(abs, want2)
        Mantle.free!(plan2.plan)
    else
        @test_skip false
    end
end

# The checkpoint's `(K, M)` int8 becomes the GEMM's `(M/4, K)` packed words, and
# the two disagree on which axis is contiguous.
#
# A thread per output word with the word index fast reads four bytes `K` apart
# and, across a wave, sixty-four groups of those `4K` apart: one byte of every
# line fetched. Qwen-Image 2.1's `12288 x 4096` measured 189.9 ms, 505 MB/s, and
# 224 such weights are what made loading the 20B denoiser 39 s. Each thread now
# takes `Q8PACK_WORDS` CONSECUTIVE words with `k` fast, which makes the write a
# full line and every read coalesced: 3.62 ms, 26.5 GB/s, and the whole upload
# 39.2 s to 10.3.
#
# Pinned here on shapes that exercise both tail guards, because the packing is
# what every weight in a compact checkpoint goes through and a wrong byte is
# not something a later test would localise.
# The conditioner's W4A8 pack has the same shape of problem and takes the same
# fix. At Qwen3-VL-8B's `4096 x 12288` it measured **121.59 ms against 4.43**,
# bit for bit the same words, and that decode runs once per weight over a 6.9
# GB checkpoint.
@testset "the W4A8 checkpoint decodes into the same words, stacked or not" begin
    # Whatever backend is loaded, not a named one: `Mantle.LavaBackend` exists only
    # where Lava does, so naming it made this file ERROR on a machine with a
    # different GPU rather than take the capability skip below that was written
    # for exactly this case.
    backend = first(Mantle.eachbackend())
    caps = DNNKernels.caps(backend)
    # `coopmatkernels`, not `caps.coopmat` alone: Metal reports cooperative
    # matrices (simdgroup matrices are real) while the staged GEMM these tests
    # exercise -- `GEMM_TILINGS`, `q8gemm_tiling`, `conv_coopmat_plan` -- is
    # compiled in only where Lava is. Asking the narrower question is what makes
    # the skip below fire instead of a `Decline(:host)` failing an assertion.
    if DNNKernels.coopmatkernels(caps)
        rng = MersenneTwister(13)
        # `M` a multiple of four and not; a stacked part at a non-zero group
        # offset; a group size that is not the k-block.
        for (K, M, GS, goff, mgd) in ((512, 1024, 128, 0, 256),
                                      (512, 1020, 128, 0, 255),
                                      (256, 260, 64, 7, 72))
            q  = rand(rng, UInt8, K ÷ 2, M)
            # `f8e4m3fn`'s NaN encoding is `x & 0x7f == 0x7f`, and a NaN scale
            # reaches `unsafe_trunc(Int8, NaN)`, which is undefined and need not
            # agree between the host and the device. A checkpoint has none.
            sr = rand(rng, UInt8, K ÷ GS, M)
            sr[(sr .& 0x7f) .== 0x7f] .= 0x00
            cb = Float32.(randn(rng, 16))
            packed = KernelAbstractions.allocate(backend, UInt32, mgd, K)
            fill!(packed, UInt32(0))
            DKA.w4a8pack!(backend, packed, DKA.toback(backend, reinterpret(Int8, q)),
                          DKA.toback(backend, sr), DKA.toback(backend, cb),
                          K, M, GS, mgd, goff)
            KernelAbstractions.synchronize(backend)
            got = Array(packed)

            mg = cld(M, 4)
            ref = zeros(UInt32, mgd, K)
            for k in 0:(K - 1), rg in 0:(mg - 1)
                w = UInt32(0)
                for r in 0:3
                    m = 4rg + r
                    m < M || continue
                    byte = q[(k ÷ 2) + 1, m + 1]
                    code = iseven(k) ? (byte & 0x0f) : (byte >> 4)
                    s = DKA.f8e4m3fn(sr[(k ÷ GS) + 1, m + 1])
                    v = round(clamp(cb[Int(code) + 1] * s, -127f0, 127f0))
                    w |= UInt32(reinterpret(UInt8, unsafe_trunc(Int8, v))) << (8r)
                end
                ref[rg + goff + 1, k + 1] = w
            end
            @test got == ref
        end
    else
        @test_skip false
    end
end

@testset "the int8 checkpoint packs four rows to a word, whatever the tail" begin
    # Whatever backend is loaded, not a named one: `Mantle.LavaBackend` exists only
    # where Lava does, so naming it made this file ERROR on a machine with a
    # different GPU rather than take the capability skip below that was written
    # for exactly this case.
    backend = first(Mantle.eachbackend())
    caps = DNNKernels.caps(backend)
    # `coopmatkernels`, not `caps.coopmat` alone: Metal reports cooperative
    # matrices (simdgroup matrices are real) while the staged GEMM these tests
    # exercise -- `GEMM_TILINGS`, `q8gemm_tiling`, `conv_coopmat_plan` -- is
    # compiled in only where Lava is. Asking the narrower question is what makes
    # the skip below fire instead of a `Decline(:host)` failing an assertion.
    if DNNKernels.coopmatkernels(caps)
        rng = MersenneTwister(11)
        # `M` a multiple of four and not; `M/4` a multiple of the words a thread
        # takes and not; `K` beyond one workgroup.
        for (K, M) in ((512, 1024), (512, 1022), (256, 132), (768, 4 * 32 * 3))
            q = rand(rng, Int8, K, M)
            scale = rand(rng, Float32, M) .+ 0.5f0
            A = DKA.convrotqint8(backend, q, scale; group_size = 256)
            KernelAbstractions.synchronize(backend)
            got = Array(A.q)
            MG = cld(M, 4)
            @test size(got) == (MG, K)
            ref = zeros(UInt32, MG, K)
            for k in 1:K, g in 0:(MG - 1)
                w = UInt32(0)
                for r in 0:3
                    m = 4g + r
                    m < M || continue
                    w |= UInt32(reinterpret(UInt8, q[k, m + 1])) << (8r)
                end
                ref[g + 1, k] = w
            end
            @test got == ref
            @test size(A) == (M, K)
            @test Array(A.scale) ≈ scale
        end
    else
        @test_skip false
    end
end
