"""
Convolution as an explicit im2col followed by a tensor-core GEMM.

`conv_implicit.jl` never materialises the im2col matrix, which is the right call
for a scalar kernel: it trades memory traffic for index arithmetic that the ALUs
would otherwise be idle for. It cannot use cooperative matrices, though, because
Lava lowers a `OpCooperativeMatrixLoadKHR` from a `PhysicalStorageBuffer`
address — there is no workgroup-storage load — so the B operand has to exist in
global memory before the multiply starts.

Materialising it is cheap here. The reduction extent `CRS = Cin*KH*KW` is large
and the pixel count `NPQ = N*OH*OW` small (the dominant layer is 15x8), so the
im2col matrix for that layer is 553 KB; writing and re-reading it costs a few
microseconds against a multiply that drops from 0.93 ms to 0.027 ms once it runs
on tensor cores.

The orientation matters and it is the *transposed* one:

    out[NPQ, Cout] = col[NPQ, CRS] * w[CRS, Cout]

Column-major, `out`'s first axis is the pixel index — which is exactly the
memory order of `(OW, OH, Cout, N)` — and `w`'s first axis is `CRS`, which is
exactly the memory order of `(KW, KH, Cin, Cout)`. So the weights are already
the B operand with no transpose, and the result lands in the destination's own
layout. Writing it the other way round (`Cout x NPQ`) would need a permute on
both ends.
"""

"""Round up to the cooperative-matrix tile."""
padtile(n::Int) = cld(n, Lava.GEMM_TILE) * Lava.GEMM_TILE

"""
    conv_coopmat_applicable(out, x, w) -> Bool

Whether the tensor-core path can take this convolution.

`NPQ` is padded internally (the im2col kernel simply writes zeros past the last
pixel), but `CRS` and `Cout` are the weight's own extents and padding those would
mean rewriting the weight, so a convolution whose channel counts do not land on
the tile falls back to the implicit-GEMM kernel. In this model that is the
stem (`7x7x3`), the 1x1 layers with a concatenated scalar channel (`Cin` 17,
257, 769) and the single-channel alpha heads — all of them small.
"""
function conv_coopmat_applicable(out, x, w)
    # The operands have to be *on the Lava device*, not merely of a type
    # cooperative matrices could hold: `coopmat_gemm_available` asks the Vulkan
    # context, which answers yes whenever Lava is loaded, so without this the CPU
    # verification run took this path and handed host `Array`s to the SPIR-V
    # compiler.
    x isa Lava.LavaArray && w isa Lava.LavaArray || return false
    eltype(x) === Float16 && eltype(w) === Float16 || return false
    KW, KH, Cin, Cout = size(w)
    CRS = Cin * KH * KW
    CRS % Lava.GEMM_TILE == 0 || return false
    Cout % Lava.GEMM_TILE == 0 || return false

    # Materialising im2col is only worth it when the GEMM reuses each element
    # enough to pay for writing and re-reading it. Every column is read once per
    # output channel, so `Cout` *is* the reuse factor: at `Cout = 16` the
    # 240x128 layer writes a 35 MB matrix to do 0.57 GFLOP, and the traffic
    # (~0.2 ms) costs more than the arithmetic it enables (~0.035 ms) — the
    # implicit-GEMM kernel, which never materialises it, wins outright.
    #
    # The size cap is the same judgement from the other side: that 35 MB is also
    # what makes the workspace OOM when anything else is on the card.
    # Both bounds used to exclude the full-resolution layers, on the estimate that
    # the implicit-GEMM fallback was cheaper for them than im2col's 35 MB of
    # traffic. Measured in situ (`OPDOUBLEFILTER` + capture/replay) that estimate
    # was inverted: `3x3 64->16 @240x128` alone cost **1.411 ms at 0.40 TFLOP/s**,
    # 21% of the step's entire convolution budget, while coopmat reaches 5.1-5.6
    # TFLOP/s on `@120x64` shapes that have 4x LESS tile parallelism.
    #
    # Admitting them: that convolution drops to **0.442 ms** (3.2x) and the step
    # goes **11.08 -> 9.78 ms, 90.2 -> 102.3 steps/s**. Cout=16 is a legal single
    # N-tile; the cap only needed to clear the 35 MB those layers ask for.
    #
    # The lesson worth keeping is that the old bounds were never wrong in
    # reasoning, only in their input — nobody had measured the fallback.
    Cout >= Lava.GEMM_TILE || return false
    padtile(size(out, 4) * size(out, 2) * size(out, 1)) * CRS * sizeof(Float16) <= 48 << 20 ||
        return false
    Lava.coopmat_gemm_available()
end

"""
The im2col matrix, `MP x CRS` column-major, zero past `NPQ` and outside the
input. `MP` is static so the destination stride folds into the address.

The index arithmetic is deliberately `Int32`. Julia hands out `Int64` indices
and Lava emits them as-is, but NVIDIA has no native 64-bit integer unit — adds
and compares are emulated and *division* especially so. This kernel does four
divisions and two remainders per element, which made it the worst case in the
model: 0.038 ms to write 590 KB. Measured on a tight loop, narrowing the
counter alone was worth 1.56x (2100 -> 3282 GFLOP/s). Every extent here is far
inside `Int32`; `MP * CRS` for the largest convolution we take is 17.7M.
"""
@kernel function im2col_kernel!(col, @Const(x), ::Val{MP},
                                ::Val{KW}, ::Val{KH}, ::Val{SX}, ::Val{SY},
                                ::Val{PX}, ::Val{PY}, ::Val{DX}, ::Val{DY},
                                Wid, Hei, OW, OH, NPQ, ntot) where {MP,KW,KH,SX,SY,PX,PY,DX,DY}
    # Flat launch, like `conv_epilogue_kernel!`: a 2-D `ndrange` is partitioned
    # into 2-D workgroups, so a warp spans only a handful of consecutive `m` and
    # the writes to `col` (which is `m`-major) are fragmented.
    lin = @index(Global, Linear)
    # `return` is not permitted in a KernelAbstractions kernel; guard instead.
    if lin <= ntot
        @inbounds begin
            T = eltype(col)
            q = Int32(lin) - Int32(1)
            m = q % Int32(MP) + Int32(1)
            c = q ÷ Int32(MP) + Int32(1)
            v = zero(T)
            if m <= Int32(NPQ)
                npq = m - Int32(1)
                n = npq ÷ Int32(OH * OW)
                r = npq - n * Int32(OH * OW)
                oh = r ÷ Int32(OW)
                ow = r - oh * Int32(OW)
                crs = c - Int32(1)
                kw = crs % Int32(KW)
                t = crs ÷ Int32(KW)
                kh = t % Int32(KH)
                cin = t ÷ Int32(KH)
                ix = ow * Int32(SX) - Int32(PX) + kw * Int32(DX)
                iy = oh * Int32(SY) - Int32(PY) + kh * Int32(DY)
                if Int32(0) <= ix < Int32(Wid) && Int32(0) <= iy < Int32(Hei)
                    v = T(x[ix + Int32(1), iy + Int32(1), cin + Int32(1), n + Int32(1)])
                end
            end
            col[m + Int32(MP) * (c - Int32(1))] = v
        end
    end
end

"""
Scatter the `MP x Cout` GEMM result back into `(OW, OH, Cout, N)` and add the
bias. The GEMM's leading dimension is the padded pixel count, so this is not a
reshape — the rows past `NPQ` are dropped here.
"""
# One lane per output element, in the destination's own linear order.
#
# Launched flat rather than over `(OW, OH, Cout, N)`. A 4-D `ndrange` makes
# KernelAbstractions partition the index space into 4-D workgroups, and
# consecutive lanes then no longer write consecutive memory — the same defect
# that cost Lava's broadcast a factor of six. Measured by running the phase twice
# and differencing: this kernel cost 2.9 ms a step, more than the tensor-core
# GEMM it follows (1.9 ms).
#
# Recovering `(pixel, channel, image)` from the linear index costs two integer
# divisions, done in `Int32` because NVIDIA emulates 64-bit integer division.
@kernel function conv_epilogue_kernel!(out, @Const(C), @Const(bias), ::Val{MP},
                                       ::Val{ACT}, ::Val{SPLITK},
                                       P, Cout, n, plane) where {MP,ACT,SPLITK}
    lin = @index(Global, Linear)
    if lin <= n
        @inbounds begin
            r = Int32(lin) - Int32(1)
            pix = r % Int32(P)              # (ow-1) + OW*(oh-1)
            t = r ÷ Int32(P)
            kc = t % Int32(Cout)            # channel, 0-based
            img = t ÷ Int32(Cout)           # image in the batch, 0-based
            npq = pix + Int32(P) * img
            i = Int32(1) + npq + Int32(MP) * kc
            # The split-K planes are summed here rather than by a separate
            # reduction pass: this kernel already touches every output element.
            v = C[i]
            for s in Int32(1):Int32(SPLITK - 1)
                v += C[i + s * Int32(plane)]
            end
            bias === nothing || (v += Float32(bias[kc + Int32(1)]))
            # Safe here in a way it is not in `conv2d_igemm!`: the splits have
            # been summed by the time the activation is applied.
            ACT === :relu && (v = max(v, zero(v)))
            out[lin] = eltype(out)(v)
        end
    end
end

"""
    convolution_coopmat!(out, x, w, bias, stride, padding, dilation) -> out

Tensor-core path. `conv_coopmat_applicable` decides whether it can run.
"""
function convolution_coopmat!(out, x, w, bias, stride, padding, dilation;
                              ws=nothing, act::Symbol=:none)
    KW, KH, Cin, Cout = size(w)
    Wid, Hei = size(x, 1), size(x, 2)
    OW, OH, _, N = size(out)
    CRS = Cin * KH * KW
    NPQ = N * OH * OW
    MP = padtile(NPQ)
    backend = KernelAbstractions.get_backend(out)

    col = scratch!(ws, backend, Float16, MP, CRS)
    im2col_kernel!(backend)(col, x, Val(MP),
                            Val(KW), Val(KH), Val(stride[1]), Val(stride[2]),
                            Val(padding[1]), Val(padding[2]),
                            Val(dilation[1]), Val(dilation[2]),
                            Wid, Hei, OW, OH, NPQ, MP * CRS; ndrange = MP * CRS)

    _, splitk = Lava.coopmat_gemm_shape(MP, Cout, CRS)
    # With a split there is no separate destination: the epilogue reads the
    # partial planes directly and sums them.
    C = scratch!(ws, backend, Float32, MP, Cout, max(splitk, 1))
    Lava.coopmat_gemm!(C, col, w, MP, Cout, CRS; partials = C, reduce = false)

    conv_epilogue_kernel!(backend)(out, C, bias, Val(MP), Val(act), Val(splitk),
                                   OW * OH, Cout, length(out), MP * Cout;
                                   ndrange = length(out))
    out
end
