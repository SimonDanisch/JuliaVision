"""
A 1-D convolution is offered to the backend's library, forward and transposed.

`conv1d` and the 2-D gather are the right PORTABLE kernels for these — `conv1d`
exists so the inner loop does not carry a trip count of 1 — but both lowerings
used to return before the backend was ever asked. The 2-D branch has asked a
library first since RIFE's transposed decoder cost 438 ms of a 581 ms frame on
the gather; the 1-D branch had the same problem and no such ask.

It is Kokoro that pays: every convolution in it is 1-D, and on an M5 they were
1.586 s of a 1.790 s utterance — 89% of the model in 93 ops. The two transposed
ones alone were 803 ms. Each op measured against MPSGraph over the lifted shapes:
394.5 -> 1.23 ms and 408.7 -> 0.57 ms transposed, 11.0 -> 0.08 ms and 26.0 ->
0.12 ms forward.

Two things are checked, and the second is the one that regresses silently.
VALUES, against a Float64 host reference, because a lift that gets an axis wrong
still produces numbers. And the ROUTE: that the op became a library call exactly
when this backend's `native_conv2d_dispatch!` says it takes these operands. That
is a capability question asked of Mantle, so it reads the same on a backend with
no library convolution (where both sides are `false`) as on one with.
"""

using Test, DNNKernels, KernelAbstractions
import Mantle
const DKC = DNNKernels
const MCC = Mantle

convbuf(id, kind, shape, T) =
    DKC.Buffer(id, kind, Any[shape...], T, "", (0, 3), "", "", Dict{String,Any}())

"""
`conv1d`/`conv_transpose1d` as a one-op graph, in torch shapes.

`x` is `(N, Cin, L)`, `w` is `(Cout, Cin, K)` forward and `(Cin, Cout, K)`
transposed, which is the layout an export hands over.
"""
function conv1dgraph(xs, ws, os, stride, pad; transposed, T = Float32, bias = true)
    ins = bias ? ["x", "w", "b"] : ["x", "w"]
    bufs = Dict{String,DKC.Buffer}(
        "x" => convbuf("x", :external, xs, T),
        "w" => convbuf("w", :external, ws, T),
        "y" => convbuf("y", :transient, os, T))
    bias && (bufs["b"] = convbuf("b", :external, (os[2],), T))
    attrs = Dict{String,Any}("arg3" => [stride], "arg4" => [pad], "arg5" => [1],
                             "arg6" => transposed, "arg7" => [0], "arg8" => 1)
    op = DKC.Op("c", "convolution.default", ins, "y", attrs)
    DKC.Graph("c", String[], ins, ["y"], bufs, [ins; "y"], [op])
end

"""The same convolution in Float64 on the host, from the torch-shaped arrays."""
function refconv1d(xh, wh, bh, stride, pad, L_out; transposed)
    N, Cin, L = size(xh)
    y = zeros(Float64, N, transposed ? size(wh, 2) : size(wh, 1), L_out)
    K = size(wh, 3)
    if !transposed
        Cout = size(wh, 1)
        for n in 1:N, co in 1:Cout, o in 1:L_out
            acc = bh === nothing ? 0.0 : Float64(bh[co])
            for ci in 1:Cin, k in 1:K
                i = (o - 1) * stride + (k - 1) - pad + 1
                1 <= i <= L && (acc += Float64(xh[n, ci, i]) * Float64(wh[co, ci, k]))
            end
            y[n, co, o] = acc
        end
    else
        Cout = size(wh, 2)
        for n in 1:N, co in 1:Cout
            bh === nothing || (y[n, co, :] .= Float64(bh[co]))
        end
        for n in 1:N, ci in 1:Cin, co in 1:Cout, i in 1:L, k in 1:K
            o = (i - 1) * stride + (k - 1) - pad + 1
            1 <= o <= L_out &&
                (y[n, co, o] += Float64(xh[n, ci, i]) * Float64(wh[ci, co, k]))
        end
    end
    y
end

"""Whether this backend's library takes the lifted operands, asked without emitting."""
function librarytakes(dev, os, xs, ws, T, stride, pad, transposed, bias)
    lift(d) = (d[3], 1, d[2], d[1])            # torch (N, C, L) -> lifted (L, 1, C, N)
    g = MCC.Graph(dev)
    o = MCC.Buffer(dev, T, lift(os))
    x = MCC.Buffer(dev, T, lift(xs))
    # The weight reverses to `(K, Cout, Cin)` forward and `(K, Cin, Cout)` in the
    # transposed layout; the lift puts the singleton after K in both.
    w = MCC.Buffer(dev, T, (ws[3], 1, ws[2], ws[1]))
    b = bias ? MCC.Buffer(dev, T, (os[2],)) : nothing
    ans = MCC.native_conv2d_dispatch!(dev, g, o, x, w; bias = b,
                                      stride = (stride, 1), pad = (pad, 0),
                                      dilation = (1, 1), groups = 1,
                                      epilogue = identity, transposed, name = "probe")
    return ans !== nothing
end

"""
Did the emitted plan hand the convolution to a library, rather than a kernel?

`pp.dispatches` and not `pp.pass.dispatches`: the pass holds the DECLARATION, and
a declared library call is a `Dispatch` there like any other. What a plan resolved
it to is on the plan's own pass, and `Mantle.Call` is what a library member is.
"""
iscall(plan) = any(plan.passes) do pp
    String(pp.pass.name) == "c" && any(d -> d isa MCC.Call, pp.dispatches)
end

@testset "a 1-D convolution reaches the library — $(nameof(typeof(backend)))" for
        backend in MCC.eachbackend()
    dev = MCC.todevice(backend)
    # Kokoro's two shapes, shrunk: a plain kernel-7 pad-3, and the stride-10
    # transposed upsampler its iSTFTNet head is built from.
    cases = ((:plain, 1, 6, 8, 7, 1, 3, 12),      # N, Cin, Cout, K, stride, pad, L
             (:plain, 1, 4, 4, 3, 2, 1, 10),
             (:transposed, 1, 8, 6, 20, 10, 5, 4),
             (:transposed, 1, 4, 5, 4, 2, 1, 6))
    for (kind, N, Cin, Cout, K, stride, pad, L) in cases, bias in (true, false)
        transposed = kind === :transposed
        L_out = transposed ? (L - 1) * stride - 2pad + K :
                             (L + 2pad - K) ÷ stride + 1
        xs = (N, Cin, L)
        ws = transposed ? (Cin, Cout, K) : (Cout, Cin, K)
        os = (N, Cout, L_out)
        @testset "$kind K=$K s=$stride bias=$bias" begin
            g = conv1dgraph(xs, ws, os, stride, pad; transposed, bias)
            model = DKC.Model(Dict("c" => g), Dict{String,Any}(), backend, 1, 1, 1)
            # Values that make an axis mix-up visible: a constant input cannot
            # tell a transposed weight from its reverse.
            xh = reshape(Float32[sin(i / 3) for i in 1:prod(xs)], xs)
            wh = reshape(Float32[cos(i / 5) / K for i in 1:prod(ws)], ws)
            bh = bias ? Float32[i / 10 for i in 1:Cout] : nothing
            args = Any[DKC.toback(backend, permutedims(xh, (3, 2, 1))),
                       DKC.toback(backend, permutedims(wh, (3, 2, 1)))]
            bias && push!(args, DKC.toback(backend, bh))
            got = Array(only(DKC.call(model, "c", args...; dims = (;))))
            want = refconv1d(xh, wh, bh, stride, pad, L_out; transposed)
            @test size(got) == (L_out, Cout, N)
            @test maximum(abs, Float64.(permutedims(got, (3, 2, 1))) .- want) < 2e-5

            # …and that it went where this backend said it could.
            plan = DKC.planfor(model.device, model.graphs["c"], model.weights, (;))
            @test iscall(plan.plan) ==
                  librarytakes(dev, os, xs, ws, Float32, stride, pad, transposed, bias)
        end
    end
end
