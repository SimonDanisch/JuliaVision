"""
Convolution: hand-written, type-parameterized, shared by every model.

ATen `convolution.default` in the reversed `(x, y, c, n)` layout. The weight is
a reversed `(C_out, C_in÷groups, KH, KW)`, i.e. indexed `(kx, ky, ci, co)`.

Kernel extent, stride, padding, dilation and groups are `Val` parameters so the
loop bounds and the group arithmetic fold at compile time. Only the plane
extents come from the launch geometry - resolution is never a type parameter.

Capability dispatch (scalar here, cooperative-matrix later) is selected at
instantiate from a device query, not here.
"""

"""
    accum(T)

Accumulator type for a reduction over `T`. Half precision accumulates in
Float32 because that is what the reference does - tensor cores take fp16
operands and accumulate fp32, and PyTorch's autocast relies on it. Summing a
few thousand products in fp16 instead drifts by percent, not ulps.
"""
@inline accum(::Type{Float16}) = Float32
@inline accum(::Type{T}) where {T} = T

@inline function conv2d(I, x, w, bias,
                        ::Val{SX}, ::Val{SY}, ::Val{PX}, ::Val{PY},
                        ::Val{DX}, ::Val{DY}, ::Val{GROUPS}) where {SX,SY,PX,PY,DX,DY,GROUPS}
    ox, oy, co, n = I
    @inbounds begin
        KX, KY, CIN = size(w, 1), size(w, 2), size(w, 3)
        A = accum(eltype(x))
        acc = bias === nothing ? zero(A) : A(bias[co])
        g = GROUPS == 1 ? 0 : (co - 1) ÷ (size(w, 4) ÷ GROUPS)
        bx = (ox - 1) * SX - PX
        by = (oy - 1) * SY - PY
        for ci in 1:CIN
            xc = g * CIN + ci
            for ky in 1:KY
                iy = by + (ky - 1) * DY + 1
                (iy < 1 || iy > size(x, 2)) && continue
                for kx in 1:KX
                    ix = bx + (kx - 1) * DX + 1
                    (ix < 1 || ix > size(x, 1)) && continue
                    acc = muladd(A(x[ix, iy, xc, n]), A(w[kx, ky, ci, co]), acc)
                end
            end
        end
        acc
    end
end

"""
    onebyone(w, stride, padding, dilation, groups) -> Bool

Whether this convolution is the 1x1 case that is a plain GEMM: unit kernel, unit
stride, no padding, no dilation, one group. `convolution!` routes those straight
to `matmul!`, with no `im2col` and no scatter.

A question about the *problem*, and nothing else. It used to begin with
`CONV_1X1_GEMM[] &&`, so with the switch off a 1x1 convolution reported that it
was not one (`kernel-library-review.md` finding 7); the switch was settled and is
gone (finding 3, tier two).

A 1x1 kernel at stride 1 with no padding is a matrix multiply and nothing else.
`im2col` exists to gather each output pixel's receptive field into a row; when
the field is one pixel the "column matrix" **is** the input, reshaped. And the
product already lands in the right place: `C` is `(NPQ, Cout)` with `NPQ`
contiguous, and the reversed output `(W, H, Cout, N)` is the same bytes in the
same order, so the scatter is the identity too.

Measured on SAM 2's largest, `convolution_4` — 144 -> 256 channels over 256x256,
4.83 GFLOP: **0.860 ms**, of which the GEMM is 0.411. The other 52% is `im2col`
writing an 18.9 MB copy of the input and the epilogue reading `C` back to move it
somewhere it already was. Five of the encoder's seven convolutions are 1x1.

**Reach, audited across every exported graph.** MatAnyone has 55 of these — 28 in
`encode_image` alone — and they are covered: `runtests.jl` verifies its graphs
node by node against PyTorch at both precisions. SAM 2's encoder has 6 of 7.
BasicVSR++ has 510 convolutions and not one 1x1-stride-1-unpadded. Wan's VAE has
6, which cannot take this route because its graphs are fp32 and
`conv_coopmat_plan` refuses them.

The bias is the one thing that does not come free. `coopmat_gemm!`'s bias is
per-*row* of `C` — per pixel here — and a convolution's is per output channel,
which is per column. So it stays a separate broadcast; that is one pass over the
output against the two this removes.
"""
@inline onebyone(w, stride, padding, dilation, groups) =
    groups == 1 && length(stride) == 2 &&
    size(w, 1) == 1 && size(w, 2) == 1 &&
    all(==(1), stride) && all(==(0), padding) && all(==(1), dilation)

"""
    convolution!(ctx, out, x, w, bias, stride, padding, dilation, groups)

`stride`/`padding`/`dilation` are `(x, y)` in the reversed layout, i.e. the
reverse of the `(h, w)` pairs ATen carries.
"""
function convolution!(ctx, out, x, w, bias, stride, padding, dilation, groups;
                      act::Symbol=:none)
    # Dense convolutions go through the implicit-GEMM kernel (conv_implicit.jl);
    # it is the only one with any data reuse. Grouped ones stay here — this model
    # has none, so that path is untuned.
    #
    # `convolution_direct!` (below) is the same kernel on every backend, CPU
    # included, and that CPU run is what validates the GPU run against the
    # reference activations, so it has to be the same source or the verification
    # means nothing.
    #
    # This entry point is NOT that kernel. The branches below reach
    # `conv_coopmat_plan` and from there `Lava.coopmat_gemm!`, which exists
    # only on Lava. Only the fallthrough is backend-independent.
    s = (stride[1], stride[2])
    p = (padding[1], padding[2])
    d = (dilation[1], dilation[2])
    if groups == 1
        # A 1x1 convolution is a GEMM on the input as it already lies — see
        # `onebyone`. Checked before the coopmat path because it is the same
        # product with two passes removed.
        cmplan = conv_coopmat_plan(ctx.dev, out, x, w)
        if ctx.ws !== nothing && onebyone(w, stride, padding, dilation, groups) &&
           cmplan isa ConvCoopMatPlan
            Wi, Hi, Cin, Nb = size(x)
            Cout = size(w, 4)
            matmul!(ctx, reshape(out, Wi * Hi * Nb, Cout), reshape(x, Wi * Hi * Nb, Cin),
                    reshape(w, Cin, Cout), nothing)
            bias === nothing || (out .= out .+ reshape(bias, 1, 1, Cout, 1))
            act === :relu && (out .= max.(out, zero(eltype(out))))
            return out
        end
        # Tensor cores when the extents land on the cooperative-matrix tile and
        # the operands are fp16 (i.e. the autocast export); the scalar
        # implicit-GEMM otherwise. Same arithmetic, ~30x apart on the layers
        # that dominate this model.
        cmplan isa ConvCoopMatPlan &&
            return convolution_coopmat!(ctx, out, cmplan, x, w, bias, s, p, d; act)
        return convolution_igemm!(ctx, out, x, w, bias, s, p, d; act)
    end
    convolution_direct!(ctx, out, x, w, bias, stride, padding, dilation, groups)
    # The grouped path has no epilogue to fold into; this model has no grouped
    # convolutions, so it is left as a separate pass rather than duplicated.
    act === :relu && (out .= max.(out, zero(eltype(out))))
    out
end

@inline function convtranspose2d(I, x, w, bias,
                                 ::Val{SX}, ::Val{SY}, ::Val{PX}, ::Val{PY},
                                 ::Val{DX}, ::Val{DY}, ::Val{G}) where {SX,SY,PX,PY,DX,DY,G}
    ox, oy, co, n = I
    @inbounds begin
        KX, KY = size(w, 1), size(w, 2)
        A = accum(eltype(x))
        acc = bias === nothing ? zero(A) : A(bias[co])
        # Grouped: output channel `co` is fed only by its own group's inputs, and
        # indexes the weight by its position WITHIN the group. The transposed
        # weight is `(kx, ky, C_out/G, C_in)`, so its third axis is already the
        # within-group one — which is exactly the arithmetic the ungrouped path
        # gets right by accident at G = 1 and would get wrong at G > 1.
        cog = G == 1 ? co : (co - 1) % (size(w, 3)) + 1
        g = G == 1 ? 0 : (co - 1) ÷ (size(w, 3))
        cilo = G == 1 ? 1 : g * (size(x, 3) ÷ G) + 1
        cihi = G == 1 ? size(x, 3) : (g + 1) * (size(x, 3) ÷ G)
        # Gather, not scatter: every output element is computed by ONE thread
        # from the inputs that reach it. The scatter reading of a transposed
        # convolution ("each input paints a kernel-sized patch") needs atomics
        # and gives a nondeterministic sum; the condition that decides which
        # inputs reach an output is just the stride divisibility below.
        bx = (ox - 1) + PX
        by = (oy - 1) + PY
        for ci in cilo:cihi
            for ky in 1:KY
                ty = by - (ky - 1) * DY
                (ty < 0 || ty % SY != 0) && continue
                iy = ty ÷ SY + 1
                (iy < 1 || iy > size(x, 2)) && continue
                for kx in 1:KX
                    tx = bx - (kx - 1) * DX
                    (tx < 0 || tx % SX != 0) && continue
                    ix = tx ÷ SX + 1
                    (ix < 1 || ix > size(x, 1)) && continue
                    # torch stores a transposed weight as (C_in, C_out, kH, kW),
                    # so reversed it is (kx, ky, co, ci) — the two channel axes
                    # are the other way round from the ordinary convolution.
                    acc = muladd(A(x[ix, iy, ci, n]), A(w[kx, ky, cog, ci]), acc)
                end
            end
        end
        acc
    end
end

"""
    convtransposesize(in, k, stride, pad, dilation, outpad) -> Int

Output extent of a transposed convolution: it undoes what the forward
convolution's `convsize` did to that axis.
"""
convtransposesize(n, k, s, p, d, op) = (n - 1) * s - 2p + d * (k - 1) + op + 1

"""
    shufflecase(w, stride, padding, dilation, outpad, groups) -> Bool

Whether a transposed convolution is the **non-overlapping** one, where the
receptive fields of neighbouring output pixels do not intersect: stride equal to
the kernel, no padding, no dilation, one group.

That case is not really a convolution. Take the gather in `convtranspose2d`: with
`P = 0`, `D = 1` and `S = K`, the stride-divisibility test `tx % SX == 0` is
satisfied by exactly one `kx` for each output column, namely `kx = dx + 1` where
`dx = (ox - 1) % S`. One input pixel, one weight slice, per output pixel:

    out[co, S*i+dx, S*j+dy] = Σ_ci  x[ci, i, j] * W[ci, co, dx, dy]

which is a GEMM over `(H*W) x C_in x (C_out*S*S)` followed by a depth-to-space
interleave — the pixel-shuffle identity. SAM 2's mask decoder upsamples with two
of these, and they were **3.73 ms of an 8.44 ms decode**, 44%, because the
gather kernel below computes each output element from scratch with no reuse.

Like [`onebyone`](@ref) this asks about the problem only. It used to begin with
`CONVT_GEMM[] &&` — and, because the `const` sat between this docstring and the
function, the prose was attached to the *switch*, so `?shufflecase` answered
nothing. Deleting the settled switch (review finding 3) put the docstring back on
the function it describes.
"""
@inline shufflecase(w, stride, padding, dilation, outpad, groups) =
    groups == 1 && length(stride) == 2 &&
    all(==(1), dilation) && all(==(0), padding) && all(==(0), outpad) &&
    size(w, 1) == stride[1] && size(w, 2) == stride[2]

"""
    phasecase(w, stride, padding, dilation, outpad, groups) -> Bool

Whether a **1-D** transposed convolution can be done as one ordinary convolution
plus an interleave — the OVERLAPPING generalisation of [`shufflecase`](@ref).

`shufflecase` needs `K == S`, so each output position comes from exactly one
input. Every upsampler in the HiFi-GAN family — Kokoro's iSTFTNet included —
uses `K == 2S` with `P == S/2` instead, so neighbouring receptive fields overlap
by half and `shufflecase` refuses. Those calls fell to `convtranspose2d`, one
thread per output element with no reuse, and on Kokoro that is **137 ms of a
217 ms utterance**: 0.019 and 0.032 TF/s where an ordinary convolution of the
same arithmetic runs at 8-20.

The decomposition holds for any `K`, `S`, `P`. Writing `t = o + P`, `d = t % S`,
`q = t ÷ S` and `j = q - i`, the defining sum

    out[o, co] = Σ_i Σ_k x[i, ci] W[k, co, ci]        over  i*S - P + k == o

becomes, for each phase `d` independently,

    out[q*S + d - P, co] = Σ_j Σ_ci x[q - j, ci] W[d + j*S, co, ci]

which is an ordinary correlation of `x` with a phase-sliced kernel of length
`J = cld(K - d, S)`. Stack the `S` phases along the output-channel axis and it is
ONE convolution producing `S*C_out` channels, then an interleave — the same
pixel-shuffle identity `shufflecase` uses, one step more general.

2-D is excluded deliberately: SAM 2's decoder is the only 2-D user and it already
satisfies `shufflecase`, so the extra weight-permute machinery would be untested
code on a path that never runs.
"""
@inline phasecase(w, stride, padding, dilation, outpad, groups) =
    groups == 1 && length(stride) == 2 &&
    size(w, 2) == 1 && stride[2] == 1 && padding[2] == 0 &&      # genuinely 1-D
    all(==(1), dilation) && all(==(0), outpad) &&
    stride[1] > 1 && size(w, 1) >= stride[1]

"""
Interleave the `S` phases of [`phasecase`](@ref)'s convolution into the output.

`y` is `(Q, 1, S*C_out, 1)`; element `(q, d, co)` belongs at output position
`q*S + d - P`. Written from the output's point of view so consecutive threads
write consecutive addresses, and positions whose `q` fell outside the convolution
are zero — they are the ones the padding removed.
"""
@inline function phaseout(I, Y, bias, ::Val{S}, ::Val{P}, Q::Int32) where {S,P}
    ox, oy, co, n = I
    @inbounds begin
        t = ox - 1 + P
        d, q = Lava.splitidx(t, Val(S))
        v = (q >= 0 && q < Int(Q)) ? Y[1 + q, 1, 1 + d + S * (co - 1), 1] : zero(eltype(Y))
        bias === nothing ? v : v + bias[co]
    end
end

"""
Scatter the GEMM's `(H*W, S*S*C_out)` result into `(S*H, S*W, C_out)`.

Reads are the transposed convolution's own output order, so consecutive threads
write consecutive addresses; the gather side is strided but it is a read.
"""
@inline function shuffleout(I, P, bias, ::Val{SX}, ::Val{SY}, W::Int32) where {SX,SY}
    ox, oy, co, n = I
    @inbounds begin
        # `splitidx` returns (remainder, quotient) in that order — the sub-pixel
        # offset first, the input coordinate second.
        dx, i = Lava.splitidx(ox - 1, Val(SX))
        dy, j = Lava.splitidx(oy - 1, Val(SY))
        v = P[1 + i + j * Int(W), 1 + dx + SX * (dy + SY * (co - 1))]
        # `launch!` is a map: return the element, it does the store.
        bias === nothing ? v : v + bias[co]
    end
end

"""
    convolutiontranspose!(ctx, out, x, w, bias, stride, padding, dilation, outpad, groups)

`aten::convolution` with `transposed = true` — SAM 2's mask decoder upsamples
its 64x64 embedding to 256x256 with two of these.

Grouped is supported by the gather kernel — Kokoro's iSTFTNet upsampler uses
depthwise (`groups == C_in`) `ConvTranspose1d`. The GEMM+shuffle fast path stays
ungrouped: it flattens all input channels into one matmul, which is precisely
what groups forbid, so grouped calls take the gather.
"""
function convolutiontranspose!(ctx, out, x, w, bias, stride, padding, dilation, outpad,
                               groups)
    # `output_padding` needs no code. It only chooses the output SIZE — torch
    # uses it to disambiguate which of the several inputs a given output length
    # could have come from — and the gather below already computes each output
    # position from whichever inputs reach it, of which the padded positions
    # have none or few. The extent is `convtransposesize`'s job, which already
    # takes it. Only the GEMM+shuffle path cannot: it assumes the receptive
    # fields tile the output exactly, and `shufflecase` already tests for that.
    #
    # This used to refuse outright. Kokoro's iSTFTNet upsampler uses
    # `output_padding = 1`, and the refusal was the last thing between the port
    # and audio.
    # The non-overlapping case is a GEMM; everything else is the gather below.
    # `size(x, 4) == 1` because the flatten fuses `W` and `H`, which are only
    # adjacent in memory within one batch element.
    # The overlapping 1-D case: one ordinary convolution over `S` stacked phases,
    # then an interleave. Checked before `shufflecase` only in the sense that it
    # is the more general test; `shufflecase`'s `K == S` shapes are left to it
    # because its GEMM needs no phase weight and no padding.
    if ctx.ws !== nothing && size(x, 4) == 1 &&
       !shufflecase(w, stride, padding, dilation, outpad, groups) &&
       phasecase(w, stride, padding, dilation, outpad, groups)
        return convolutiontranspose_phase!(ctx, out, x, w, bias, stride, padding)
    end
    if groups == 1 && ctx.ws !== nothing && size(x, 4) == 1 &&
       shufflecase(w, stride, padding, dilation, outpad, groups)
        Wi, Hi, Ci, _ = size(x)
        KX, KY, Co, _ = size(w)
        xm = reshape(x, Wi * Hi, Ci)
        # `(ci, dx, dy, co)` flattens column-major to the column index the
        # shuffle above reads. The permute is of a *weight*, so it belongs at
        # load time — `hoistpermutes` territory — and is done per call for now.
        wm = reshape(permutedims(w, (4, 1, 2, 3)), Ci, KX * KY * Co)
        P = scratch!(ctx, eltype(out), Wi * Hi, KX * KY * Co)
        matmul!(ctx, P, xm, wm, nothing)
        launch!(ctx, shuffleout, out, P, bias, Val(stride[1]), Val(stride[2]), Int32(Wi))
        return out
    end
    launch!(ctx, convtranspose2d, out, x, w, bias,
            Val(stride[1]), Val(stride[2]), Val(padding[1]), Val(padding[2]),
            Val(dilation[1]), Val(dilation[2]), Val(groups))
end

"""
    convolutiontranspose_phase!(ctx, out, x, w, bias, stride, padding)

[`phasecase`](@ref)'s decomposition: build the phase-sliced weight, run ONE
ordinary convolution producing `S*C_out` channels, interleave.

`w` is `(K, 1, C_out, C_in)` — torch stores a transposed weight `C_in`-major and
the reversed layout puts `C_in` last. The phase weight is
`Wp[J-j, 1, d + S*(co-1), ci] = w[d + j*S + 1, 1, co, ci]`, zero where
`d + j*S >= K`. The `J-j` rather than `j+1` is the flip that turns the
correlation `convolution!` computes into the `x[q - j]` this needs; getting it
backwards produces audio that is still speech-shaped, which is why the test
compares against the gather kernel elementwise rather than by ear.

The weight is rebuilt per call. It is a graph constant and belongs at load time
(`hoistpermutes` territory), exactly as the note on `shufflecase`'s `wm` says;
`S*J` slice copies of `(C_out, C_in)` is 5.2 MB on Kokoro's largest, against the
72 ms the old path spent.
"""
function convolutiontranspose_phase!(ctx, out, x, w, bias, stride, padding)
    S, P = stride[1], padding[1]
    K, Cout, Cin = size(w, 1), size(w, 3), size(w, 4)
    M = size(x, 1)
    J = cld(K, S)
    T = eltype(out)

    # `(KW, KH, C_in, C_out)` is what `convolution!` wants, and a TRANSPOSED
    # weight arrives `(KW, KH, C_out, C_in)` — torch stores it `C_in`-major. The
    # permute is once for the whole weight rather than once per phase slice;
    # building `Wp` the other way round silently computes a different product and
    # still returns speech-shaped audio, which is why the test below diffs
    # against the gather kernel elementwise.
    wp = permutedims(w, (1, 2, 4, 3))                 # (K, 1, C_in, C_out)
    Wp = scratch!(ctx, T, J, 1, Cin, S * Cout)
    fill!(Wp, zero(T))
    for j in 0:(J - 1), d in 0:(S - 1)
        kk = d + j * S
        kk < K || continue
        # `J - j` is the flip; `d + S*(co-1)` is the channel-major stacking the
        # interleave above reads back.
        copyto!(view(Wp, J - j, 1, :, (d + 1):S:(d + 1 + S * (Cout - 1))),
                view(wp, kk + 1, 1, :, :))
    end

    Q = M + J - 1                       # `padding = J-1` on both sides, stride 1
    Y = scratch!(ctx, T, Q, 1, S * Cout, 1)
    convolution!(ctx, Y, x, Wp, nothing, [1, 1], [J - 1, 0], [1, 1], 1)
    launch!(ctx, phaseout, out, Y, bias, Val(S), Val(P), Int32(Q))
    out
end

"""One thread per output element, no reuse. Kept for grouped convolutions and as
the reference the implicit-GEMM kernel is checked against."""
function convolution_direct!(ctx, out, x, w, bias, stride, padding, dilation, groups)
    launch!(ctx, conv2d, out, x, w, bias,
            Val(stride[1]), Val(stride[2]), Val(padding[1]), Val(padding[2]),
            Val(dilation[1]), Val(dilation[2]), Val(groups))
end

"""
    convsize(inplane, k, stride, padding, dilation) -> Int

ATen's output extent, evaluated host-side so no kernel recomputes it.
"""
convsize(n, k, s, p, d) = (n + 2p - d * (k - 1) - 1) ÷ s + 1

"""
1-D convolution. `aten::convolution` covers 1/2/3-D under one schema and the
channel-attention blocks (channel_attn.py) use the 1-D form, so it is a
separate body rather than a 2-D one with a degenerate axis - the inner loop
should not carry a trip count of 1.

Layout `(x, c, n)`; weight reversed `(C_out, C_in÷groups, K)` = `(kx, ci, co)`.
"""
@inline function conv1d(I, x, w, bias, ::Val{SX}, ::Val{PX}, ::Val{DX},
                        ::Val{GROUPS}) where {SX,PX,DX,GROUPS}
    ox, co, n = I
    @inbounds begin
        KX, CIN = size(w, 1), size(w, 2)
        A = accum(eltype(x))
        acc = bias === nothing ? zero(A) : A(bias[co])
        g = GROUPS == 1 ? 0 : (co - 1) ÷ (size(w, 3) ÷ GROUPS)
        bx = (ox - 1) * SX - PX
        for ci in 1:CIN
            xc = g * CIN + ci
            for kx in 1:KX
                ix = bx + (kx - 1) * DX + 1
                (ix < 1 || ix > size(x, 1)) && continue
                acc = muladd(A(x[ix, xc, n]), A(w[kx, ci, co]), acc)
            end
        end
        acc
    end
end

convolution1d!(ctx, out, x, w, bias, stride, padding, dilation, groups) =
    launch!(ctx, conv1d, out, x, w, bias, Val(stride[1]), Val(padding[1]),
            Val(dilation[1]), Val(groups))
