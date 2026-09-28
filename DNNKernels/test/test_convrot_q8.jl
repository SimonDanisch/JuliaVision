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

# The declared ConvRot product rotates its activations straight into the
# `K x NP` buffer the packed GEMM reads. It used to rotate into a `K x N` one
# and copy that into the padded buffer with `padcols_kernel!`: 85 ms of a
# Qwen-Image 2.1 step, all of it bytes moved for nothing. The pad columns are
# left unwritten, since output column `j` reads only column `j` of `B` and the
# output's pad is discarded.
#
# Both packed kernels behind it: `N = 4118` pads to 4224 and takes the pipelined
# kernel, `N = 300` pads to 384 and takes the staged tiling.
@testset "ConvRot writes the padded GEMM input directly" begin
    backend = first(Mantle.eachbackend())
    caps = DNNKernels.caps(backend)
    if DNNKernels.coopmatkernels(caps) && caps.coopmatsubgroup == 32
        rng = MersenneTwister(12)
        Mm, K = 256, 512
        A = DKA.convrotqint8(backend, rand(rng, Int8.(-40:40), K, Mm),
                             rand(rng, Float32, Mm) .* 0.01f0 .+ 0.002f0; group_size = 256)
        buf(id, shape; kind = :transient, dtype = Float16) =
            DKA.Buffer(id, kind, Any[shape...], dtype, id == "w" ? "w" : "", (0, 0), "", "",
                       Dict{String,Any}())
        for (N, pipelined) in ((4118, true), (300, false))
            NP = DKA.q8gemm_columns(N)
            @test (DKA.q8gemm_pipelined_tile(caps, Float16, Mm, K, NP) !== nothing) == pipelined
            @test DKA.q8gemm_tiling(caps, Float16, Mm, K, NP) !== nothing
            B = DKA.toback(backend, Float16.(randn(rng, Float32, K, N) .* 0.5f0))
            buffers = Dict(b.id => b for b in (buf("x", (N, K); kind = :external),
                                               buf("w", (Mm, K); kind = :weight),
                                               buf("y", (N, Mm)),
                                               buf("z", (N, Mm))))
            ops = [DKA.Op("y", "mm.default", ["x", "w"], "y", Dict{String,Any}()),
                   DKA.Op("z", "clamp.default", ["y"], "z",
                          Dict{String,Any}("arg1" => -1000, "arg2" => 1000))]
            g = DKA.Graph("q8rot", String[], ["x"], ["z"], buffers, collect(keys(buffers)), ops)
            plan = DKA.planfor(Mantle.todevice(backend), g, Dict{String,Any}("w" => A), (;))
            got = Float32.(Array(first(DKA.replay!(plan, "q8rot", (B,)))))
            passnames = [String(p.pass.name) for p in plan.plan.passes]
            @test count(n -> occursin("padB", n), passnames) == 0
            @test count(n -> occursin("convrot", n), passnames) == 2
            Mantle.free!(plan.plan)

            ctx = DKA.Ctx(Dict{String,Any}(), g, (;), backend)
            out = KernelAbstractions.allocate(backend, Float16, Mm, N)
            DKA.matmul!(ctx, DKA.MMConvRotInt8Plan(), out, A, B, nothing, identity)
            KernelAbstractions.synchronize(backend)
            want = clamp.(Float32.(Array(out)), -1000, 1000)
            @test size(got) == (Mm, N)
            @test maximum(abs, got .- want) <= 0.01maximum(abs, want)
        end
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
            # `w4a8pack!` declares into a graph now: the stacked form decodes
            # several parts into one pack, and those are passes of one graph.
            dev = Mantle.todevice(backend)
            g = Mantle.Graph(dev)
            packed = Mantle.Buffer(dev, zeros(UInt32, mgd, K))
            DKA.w4a8pack!(g, packed, Mantle.Buffer(dev, reinterpret(Int8, q)),
                          Mantle.Buffer(dev, sr), Mantle.Buffer(dev, cb),
                          K, M, GS, mgd, goff)
            DKA.runonce!(g)
            KernelAbstractions.synchronize(backend)
            got = Array(Mantle.storage(packed))

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

# A SwiGLU whose only reader is a product with a ConvRot weight is folded into
# it (`fuseswiglumm`), and the rotation's first pass computes the SwiGLU as it
# reads: the product, 101 MB a layer at Qwen-Image 2.1's `12288 x 4118`, is
# never written. Same arithmetic in the same order, so the result is the
# unfused graph's to the bit, and a weight without a rotation gets the two
# unfused ops back.
#
# `N = 4118` and `N = 300` pad (to the pipelined and the staged kernel), and
# `N = 256` does not, which rotates into a plain `K x N` buffer instead.
@testset "a SwiGLU is computed inside the ConvRot that reads it" begin
    backend = first(Mantle.eachbackend())
    caps = DNNKernels.caps(backend)
    if DNNKernels.coopmatkernels(caps) && caps.coopmatsubgroup == 32
        rng = MersenneTwister(14)
        Mm, K = 256, 512
        A(p...) = Dict{String,Any}(p...)
        buf(id, kind, shape; of = "", viewop = "", attrs = A()) =
            DKA.Buffer(id, kind, Any[shape...], Float16, id == "w" ? "w" : "", (0, 0),
                       of, viewop, attrs)
        # `x` is the stacked gate/up product, `(N, 2K)` in torch's order; the
        # SwiGLU's result reaches the product through a reshape, as it does in
        # the export.
        function mlpgraph(N; readers = 1, output = false)
            bs = [buf("x", :external, (N, 2K)),
                  buf("gate", :view, (N, K); of = "x", viewop = "slice.Tensor",
                      attrs = A("arg1" => 1, "arg2" => 0, "arg3" => K)),
                  buf("up", :view, (N, K); of = "x", viewop = "slice.Tensor",
                      attrs = A("arg1" => 1, "arg2" => K, "arg3" => 2K)),
                  buf("h", :transient, (1, N, K)),
                  buf("hv", :view, (N, K); of = "h", viewop = "view.default",
                      attrs = A("arg1" => Any[N, K])),
                  buf("w", :weight, (Mm, K)),
                  buf("y", :transient, (N, Mm)),
                  buf("z", :transient, (N, Mm)),
                  buf("h2", :transient, (1, N, K))]
            ops = [DKA.Op("h", "fused.swiglu", ["gate", "up"], "h", A()),
                   DKA.Op("y", "mm.default", ["hv", "w"], "y", A()),
                   DKA.Op("z", "clamp.default", ["y"], "z", A("arg1" => -1000, "arg2" => 1000))]
            readers == 2 && push!(ops, DKA.Op("h2", "clamp.default", ["h"], "h2",
                                              A("arg1" => -1000, "arg2" => 1000)))
            outs = output ? ["z", "h"] : readers == 2 ? ["z", "h2"] : ["z"]
            DKA.Graph("mlp", String[], ["x"], outs, Dict(b.id => b for b in bs),
                      [b.id for b in bs], ops)
        end
        function run(g, w, x)
            plan = DKA.planfor(Mantle.todevice(backend), g, Dict{String,Any}("w" => w), (;))
            got = Array(first(DKA.replay!(plan, "mlp", (x,))))
            names = [String(p.pass.name) for p in plan.plan.passes]
            Mantle.free!(plan.plan)
            got, names
        end

        # The rewrite, and the two SwiGLUs it has to leave alone: one with a
        # second reader, and one the graph returns.
        g, n = DKA.fuseswiglumm(mlpgraph(300))
        @test n == 1
        op = only(o for o in g.ops if o.aten == "fused.swiglumm")
        @test op.ins == ["gate", "up", "w"] && op.out == "y" && op.attrs["dtype"] === Float16
        @test !any(o -> o.aten == "fused.swiglu", g.ops)
        @test DKA.fuseswiglumm(mlpgraph(300; readers = 2))[2] == 0
        @test DKA.fuseswiglumm(mlpgraph(300; output = true))[2] == 0

        wrot = DKA.convrotqint8(backend, rand(rng, Int8.(-40:40), K, Mm),
                                rand(rng, Float32, Mm) .* 0.01f0 .+ 0.002f0; group_size = 256)
        wq8 = DKA.quantizeint8(backend,
            DKA.toback(backend, Float16.(randn(rng, Float32, Mm, K) .* 0.2f0)))
        for N in (4118, 300, 256)
            x = DKA.toback(backend, Float16.(randn(rng, Float32, 2K, N) .* 2f0))
            want, _ = run(mlpgraph(N), wrot, x)
            got, names = run(first(DKA.fuseswiglumm(mlpgraph(N))), wrot, x)
            @test got == want
            @test "y.convrot.1" in names && "y.convrot.2" in names
            # No SwiGLU pass, no copy into a padded input, and no copy out of a
            # padded output: `paddedcolumns` has to know the fused op too.
            @test !any(n -> n == "h" || occursin("swiglu", n) || occursin("padB", n) ||
                            occursin("unpad", n), names)

            want, _ = run(mlpgraph(N), wq8, x)
            got, names = run(first(DKA.fuseswiglumm(mlpgraph(N))), wq8, x)
            @test got == want
            @test "y.swiglu" in names
        end
    else
        @test_skip false
    end
end
