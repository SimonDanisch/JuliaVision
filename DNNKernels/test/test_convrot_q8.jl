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

using Test, DNNKernels, Mantle, KernelAbstractions, Random

const DKA = DNNKernels

@testset "ConvRot and the packed product it feeds" begin
    backend = Mantle.LavaBackend()
    caps = DNNKernels.caps(backend)
    if caps.coopmat && caps.coopmatsubgroup == 32
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
            DNNKernels.convrot_pass_kernel!(backend, 256)(
                o, S == 1 ? x : o, Val(S), Val(16), Val(256), n; ndrange = length(xh) ÷ 16)
        end
        KernelAbstractions.synchronize(backend)
        want = staged(xh, 256)
        # One fp16 ulp at this magnitude: the compiler reassociates the sum.
        @test maximum(abs, Float32.(Array(o)) .- Float32.(want)) <= 0.001
        # Orthogonal, and its own inverse — what makes an embedding row
        # recoverable by applying it twice.
        DNNKernels.convrot_pass_kernel!(backend, 256)(o, o, Val(1), Val(16), Val(256), n;
                                                      ndrange = length(xh) ÷ 16)
        DNNKernels.convrot_pass_kernel!(backend, 256)(o, o, Val(16), Val(16), Val(256), n;
                                                      ndrange = length(xh) ÷ 16)
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
