"""
The declared dense convolution against its definition, and the index type of the
implicit GEMM behind it.

`conv2d_igemm_ki!` computes its indices in the integer type of its extents, and
the emitter passes `Int32` whenever every operand fits (`convindex`). In `Int64`
every `÷` by a runtime extent is a 64-bit division, which RADV expands in
software: the Qwen-Image 2.1 VAE's 1024x1024 `convolution_37` compiled to 125,360
instructions at 4 waves per SIMD and took 1585 ms, against 4,474 instructions,
8 waves and 335 ms in `Int32`. The same rewrite moved the write-back's index
decomposition out of its row loop.

Nothing checked this kernel's output directly before; the parity gates reach it
only through whole models. The cases cover what the rewrite touched: more than
one output-channel tile, a batch, stride, padding and dilation, no bias, and a
shape that splits its reduction (`splitk > 1`), with and without a fused relu.
"""

using Test, Random
import DNNKernels, Mantle
const DK = DNNKernels
const MC = Mantle

"`(W, H, Cin, N)` convolved with `(KW, KH, Cin, Cout)`, in Float64, on the host."
function convref(x, w, b, stride, pad, dil, act)
    W, H, Cin, N = size(x)
    KW, KH, _, Cout = size(w)
    OW = (W + 2pad[1] - dil[1] * (KW - 1) - 1) ÷ stride[1] + 1
    OH = (H + 2pad[2] - dil[2] * (KH - 1) - 1) ÷ stride[2] + 1
    y = zeros(Float64, OW, OH, Cout, N)
    for n in 1:N, k in 1:Cout, oh in 1:OH, ow in 1:OW
        s = b === nothing ? 0.0 : Float64(b[k])
        for cin in 1:Cin, kh in 1:KH, kw in 1:KW
            ix = (ow - 1) * stride[1] - pad[1] + (kw - 1) * dil[1] + 1
            iy = (oh - 1) * stride[2] - pad[2] + (kh - 1) * dil[2] + 1
            (1 <= ix <= W && 1 <= iy <= H) || continue
            s += Float64(w[kw, kh, cin, k]) * Float64(x[ix, iy, cin, n])
        end
        y[ow, oh, k, n] = act === :relu ? max(s, 0.0) : s
    end
    return y
end

"Emit one `convolution.default` into a plan on `dev`, run it, and return the result,
the implicit-GEMM dispatches it declared and the library calls."
function convplan(dev, x, w, b, stride, pad, dil, act)
    y = convref(x, w, b, stride, pad, dil, :none)          # for the output shape only
    g = MC.Graph(dev)
    res = Dict{String,Any}("x" => MC.Buffer(dev, x), "w" => MC.Buffer(dev, w),
                           "y" => MC.Buffer(dev, Float32, size(y)))
    ins = ["x", "w"]
    if b !== nothing
        res["b"] = MC.Buffer(dev, b)
        push!(ins, "b")
    end
    # The attributes are torch's, height before width; `stride`, `pad` and `dil`
    # here are in the reversed layout's axis order, width first.
    attrs = Dict{String,Any}("arg3" => collect(reverse(stride)), "arg4" => collect(reverse(pad)),
                             "arg5" => collect(reverse(dil)), "arg6" => false,
                             "arg7" => [0, 0], "arg8" => 1)
    act === :none || (attrs["act"] = string(act))
    op = DK.Op("y", "convolution.default", ins, "y", attrs)
    emitctx = DK.EmitCtx(DK.Graph("y", String[], String[], String[],
                                  Dict{String,DK.Buffer}(), String[], DK.Op[],
                                  Vector{Vector{String}}()),
                         g, dev, NamedTuple(), res, Set{String}(), Ref("y"), Any[],
                         Dict{String,Any}())
    DK.emitop!(emitctx, op, Val(Symbol("convolution.default")))
    plan = MC.Plan(g)
    MC.record!(plan)
    igemm = [d for pp in plan.passes for d in pp.pass.dispatches
             if d.kernel === DK.conv2d_igemm_ki!]
    calls = [d for pp in plan.passes for d in pp.pass.dispatches if !(d.kernel isa Function)]
    MC.run!(plan)
    MC.waitidle(dev)
    got = Array(MC.storage(res["y"]))
    MC.free!(plan)
    foreach(MC.free!, values(res))
    return got, igemm, calls
end

@testset "convindex" begin
    @test DK.convindex(zeros(Float32, 4), zeros(Float32, 4), zeros(Float32, 4)) === Int32
    # Past `typemax(Int32)` elements an offset no longer fits. Ranges stand in for
    # operands that size, which nothing here could allocate.
    @test DK.convindex(1:2^31, 1:2, 1:2) === Int64
    @test DK.convindex(1:2, 1:2^31, 1:2) === Int64
    @test DK.convindex(1:2, 1:2, 1:typemax(Int32)) === Int32
end

function declaredconv(be)
    dev = MC.todevice(be)
    Random.seed!(3)
    #   (W,  H,  Cin, Cout, N, K, stride, pad, dil, bias, act)
    cases = [
        (17, 13, 20, 150, 2, 3, (1, 1), (1, 1), (1, 1), true,  :none),  # two channel tiles, a batch
        (19, 16, 12, 40,  1, 3, (2, 2), (2, 2), (2, 2), true,  :none),  # stride, padding, dilation
        (9,  9,  16, 24,  1, 5, (1, 2), (2, 0), (1, 1), false, :none),  # no bias, uneven strides
        (8,  8,  256, 256, 1, 3, (1, 1), (1, 1), (1, 1), true, :none),  # splits its reduction
        (8,  8,  256, 256, 1, 3, (1, 1), (1, 1), (1, 1), true, :relu),  # ... through the fp32 scratch
    ]
    @testset "a declared convolution matches the definition on $(nameof(typeof(dev)))" begin
        for (W, H, Cin, Cout, N, K, stride, pad, dil, hasbias, act) in cases
            x = 0.2f0 .* randn(Float32, W, H, Cin, N)
            w = 0.2f0 .* randn(Float32, K, K, Cin, Cout)
            b = hasbias ? 0.2f0 .* randn(Float32, Cout) : nothing
            got, igemm, calls = convplan(dev, x, w, b, stride, pad, dil, act)
            want = convref(x, w, b, stride, pad, dil, act)
            @test size(got) == size(want)
            @test maximum(abs.(Float64.(got) .- want)) <= 1e-5 * maximum(abs, want) * sqrt(Cin * K^2)
            # The convolution went to the implicit GEMM or to a library call, and
            # where it is the implicit GEMM, its extents are `Int32`.
            @test !isempty(igemm) || !isempty(calls)
            for d in igemm
                @test all(a -> a isa Int32, d.args[end-8:end])
            end
        end
    end
end

let bes = MC.eachbackend()
    isempty(bes) && @info "no usable backend; skipping the declared convolution"
    foreach(declaredconv, bes)
end
