"""
Layer norm in one kernel.

Written as an expression — mean, centre, variance, rsqrt, scale-and-shift — it
is six passes over the tensor:

    sum(a)                     read a
    t .= a .- μ                read a, write t
    sum(abs2, t)               read t
    out .= (a .- μ) .* r ...   read a, write out

Measured on SAM 2's dominant layer-norm shape, `(576, 64, 64, 1)` fp16, 4.7 MB:
**0.341 ms**, against 0.062 for a plain copy of the same tensor. PyTorch runs one
fused `vectorized_layer_norm_kernel` and does all 96 of the encoder's layer
norms in 5.64 ms against our ~37.6.

This does two: one read of `a` from global, one write of `out`. The two
reduction passes and the final pass re-read `a`, but a normalisation group is
`C` elements — 144 to 1152 here, i.e. 288 to 2304 bytes — so after the first
pass it is in L1 and the re-reads are nearly free.

Two reduction passes rather than the one that `E[x²] - μ²` allows, deliberately:
that form cancels catastrophically when the mean dominates the variance, and
mask parity with PyTorch to five decimals is not worth trading for a pass that
comes out of cache anyway.
"""

"""Fallback threads per normalisation group for reductions wider than 2048."""
const LN_WG = 128

"""Workgroup width for layer norm, selected from the reduction width.

Small normalisation groups need useful lanes more than they need parallelism:
on SAM 2's 144/288/576-wide groups, 32 lanes are 2.5--2.9x faster than the old
fixed 128.  At 1152, 64 lanes win.  This depends only on the tensor shape, not
on a backend identity; larger reductions retain the conservative 128-wide
route.
"""
layernormwg(C::Integer) = C <= 576 ? 32 : C <= 2048 ? 64 : LN_WG

"""
    layernorm_kernel!(out, mean, rstd, a, γ, β, C, eps)

One workgroup per normalisation group. `γ`/`β` are indexed along the normalised
axis and may be the same array for every group; `HASG`/`HASB` fold them out
entirely when the graph did not supply them, so the no-affine case pays nothing.

`mean` and `rstd` are written because `native_layer_norm` returns them as tuple
elements and something downstream may read them; they are one scalar per group,
so the cost is noise.
"""
function layernorm_kernel!(out, mean, rstd, a, γ,
                                             β, C::Int32, eps::Float32,
                                             ::Val{WG}, ::Val{HASG}, ::Val{HASB}) where
                                            {WG,HASG,HASB}
    red = KI.localmemory(Float32, Val((WG,)), Val(1))
    g = KI.get_group_id().x - 1
    t = KI.get_local_id().x - 1
    base = g * Int(C)
    n = Float32(C)

    # ── pass 1: the mean
    s = 0.0f0
    i = t
    @inbounds while i < C
        s += Float32(a[base + i + 1])
        i += WG
    end
    @inbounds red[t + 1] = s
    KI.barrier()
    # Tree reduction. The barrier is outside the `if`, so every lane reaches it
    # — a barrier in divergent control flow is undefined, and on this compiler
    # that is not a theoretical concern.
    stride = WG ÷ 2
    while stride > 0
        @inbounds if t < stride
            red[t + 1] += red[t + 1 + stride]
        end
        KI.barrier()
        stride ÷= 2
    end
    @inbounds μ = red[1] / n
    KI.barrier()                      # nothing may overwrite `red` until all have read it

    # ── pass 2: the variance, about that mean
    s2 = 0.0f0
    i = t
    @inbounds while i < C
        d = Float32(a[base + i + 1]) - μ
        s2 += d * d
        i += WG
    end
    @inbounds red[t + 1] = s2
    KI.barrier()
    stride = WG ÷ 2
    while stride > 0
        @inbounds if t < stride
            red[t + 1] += red[t + 1 + stride]
        end
        KI.barrier()
        stride ÷= 2
    end
    @inbounds r = 1.0f0 / sqrt(red[1] / n + eps)
    KI.barrier()

    # ── pass 3: normalise, scale, shift
    i = t
    @inbounds while i < C
        x = (Float32(a[base + i + 1]) - μ) * r
        if HASG
            x *= Float32(γ[i + 1])
        end
        if HASB
            x += Float32(β[i + 1])
        end
        out[base + i + 1] = x
        i += WG
    end
    @inbounds if t == 0
        mean[g + 1] = μ
        rstd[g + 1] = r
    end
    return nothing
end

"""
Layer norm with exactly one hardware subgroup per normalisation group.

The ordinary kernel's shared-memory tree is the portable fallback. Backends
which report a subgroup width can reduce each partial sum with the portable
KernelInterface intrinsic instead: no shared scratch and no reduction barriers.
"""
function layernorm_shfl_kernel!(out, mean, rstd,
                                                       a, γ,
                                                       β, C::Int32,
                                                       NG::Int32, eps::Float32,
                                                       ::Val{SG}, ::Val{ROW},
                                                      ::Val{HASG}, ::Val{HASB}) where
                                                      {SG,ROW,HASG,HASB}
    block = KI.get_group_id().x - 1
    t = KI.get_local_id().x - 1
    lane = t % ROW
    chunk = t ÷ ROW
    g = block * div(SG, ROW) + chunk
    valid = g < NG
    base = g * Int(C)
    n = Float32(C)

    s = 0.0f0
    i = lane
    @inbounds while valid && i < C
        s += Float32(a[base + i + 1])
        i += ROW
    end
    if ROW == 64
        v = KI.shfl_down(s, 32); lane < 32 && (s += v)
    end
    v = KI.shfl_down(s, 16); lane < 16 && (s += v)
    v = KI.shfl_down(s, 8);  lane < 8  && (s += v)
    v = KI.shfl_down(s, 4);  lane < 4  && (s += v)
    v = KI.shfl_down(s, 2);  lane < 2  && (s += v)
    v = KI.shfl_down(s, 1);  lane < 1  && (s += v)
    μ = KI.shfl(s, chunk * ROW) / n

    s2 = 0.0f0
    i = lane
    @inbounds while valid && i < C
        d = Float32(a[base + i + 1]) - μ
        s2 += d * d
        i += ROW
    end
    if ROW == 64
        v = KI.shfl_down(s2, 32); lane < 32 && (s2 += v)
    end
    v = KI.shfl_down(s2, 16); lane < 16 && (s2 += v)
    v = KI.shfl_down(s2, 8);  lane < 8  && (s2 += v)
    v = KI.shfl_down(s2, 4);  lane < 4  && (s2 += v)
    v = KI.shfl_down(s2, 2);  lane < 2  && (s2 += v)
    v = KI.shfl_down(s2, 1);  lane < 1  && (s2 += v)
    r = 1.0f0 / sqrt(KI.shfl(s2, chunk * ROW) / n + eps)

    i = lane
    @inbounds while valid && i < C
        x = (Float32(a[base + i + 1]) - μ) * r
        HASG && (x *= Float32(γ[i + 1]))
        HASB && (x += Float32(β[i + 1]))
        out[base + i + 1] = x
        i += ROW
    end
    @inbounds if valid && lane == 0
        mean[g + 1] = μ
        rstd[g + 1] = r
    end
    return nothing
end

"""Subgroup layer norm fused with the residual add that produces its input."""
function add_layernorm_shfl_kernel!(out, sumout, mean, rstd,
        a, b, γ, β, C::Int32, NG::Int32,
        eps::Float32, ::Val{SG}, ::Val{ROW}, ::Val{HASG}, ::Val{HASB}) where
        {SG,ROW,HASG,HASB}
    block = KI.get_group_id().x - 1
    t = KI.get_local_id().x - 1
    lane = t % ROW
    chunk = t ÷ ROW
    g = block * div(SG, ROW) + chunk
    valid = g < NG
    base = g * Int(C)
    n = Float32(C)

    s = 0.0f0
    i = lane
    @inbounds while valid && i < C
        # Convert through the residual's declared element type. In particular,
        # f16 + f16 -> f16 must round before the norm observes it.
        z = convert(eltype(sumout), a[base + i + 1] + b[base + i + 1])
        sumout[base + i + 1] = z
        s += Float32(z)
        i += ROW
    end
    if ROW == 64
        v = KI.shfl_down(s, 32); lane < 32 && (s += v)
    end
    v = KI.shfl_down(s, 16); lane < 16 && (s += v)
    v = KI.shfl_down(s, 8);  lane < 8  && (s += v)
    v = KI.shfl_down(s, 4);  lane < 4  && (s += v)
    v = KI.shfl_down(s, 2);  lane < 2  && (s += v)
    v = KI.shfl_down(s, 1);  lane < 1  && (s += v)
    μ = KI.shfl(s, chunk * ROW) / n

    s2 = 0.0f0
    i = lane
    @inbounds while valid && i < C
        d = Float32(sumout[base + i + 1]) - μ
        s2 += d * d
        i += ROW
    end
    if ROW == 64
        v = KI.shfl_down(s2, 32); lane < 32 && (s2 += v)
    end
    v = KI.shfl_down(s2, 16); lane < 16 && (s2 += v)
    v = KI.shfl_down(s2, 8);  lane < 8  && (s2 += v)
    v = KI.shfl_down(s2, 4);  lane < 4  && (s2 += v)
    v = KI.shfl_down(s2, 2);  lane < 2  && (s2 += v)
    v = KI.shfl_down(s2, 1);  lane < 1  && (s2 += v)
    r = 1.0f0 / sqrt(KI.shfl(s2, chunk * ROW) / n + eps)

    i = lane
    @inbounds while valid && i < C
        x = (Float32(sumout[base + i + 1]) - μ) * r
        HASG && (x *= Float32(γ[i + 1]))
        HASB && (x += Float32(β[i + 1]))
        out[base + i + 1] = x
        i += ROW
    end
    @inbounds if valid && lane == 0
        mean[g + 1] = μ
        rstd[g + 1] = r
    end
    return nothing
end


"""
The subgroup layer norm with its result written directly in SAM-style window
order.  `g` remains the dense spatial row used to read `a`; only the output row
is permuted, so the reduction and fp16 store rounding are unchanged.
"""
function layernorm_shfl_window4_add_kernel!(out, wideout, sumout, mean, rstd,
        a, b, γ, β, C::Int32, NG::Int32, eps::Float32,
        ::Val{SG}, ::Val{ROW}, ::Val{IW}, ::Val{IH}, ::Val{NX}, ::Val{NY},
        ::Val{ADD}, ::Val{FULLWIDE}, ::Val{HASG}, ::Val{HASB}) where
        {SG,ROW,IW,IH,NX,NY,ADD,FULLWIDE,HASG,HASB}
    block = KI.get_group_id().x - 1
    t = KI.get_local_id().x - 1
    lane = t % ROW
    chunk = t ÷ ROW
    g = block * div(SG, ROW) + chunk
    valid = g < NG
    base = g * Int(C)
    n = Float32(C)

    s = 0.0f0
    i = lane
    @inbounds while valid && i < C
        z = ADD ? convert(eltype(sumout), a[base + i + 1] + b[base + i + 1]) :
                  a[base + i + 1]
        ADD && (sumout[base + i + 1] = z)
        s += Float32(z)
        i += ROW
    end
    if ROW == 64
        v = KI.shfl_down(s, 32); lane < 32 && (s += v)
    end
    v = KI.shfl_down(s, 16); lane < 16 && (s += v)
    v = KI.shfl_down(s, 8);  lane < 8  && (s += v)
    v = KI.shfl_down(s, 4);  lane < 4  && (s += v)
    v = KI.shfl_down(s, 2);  lane < 2  && (s += v)
    v = KI.shfl_down(s, 1);  lane < 1  && (s += v)
    μ = KI.shfl(s, chunk * ROW) / n

    s2 = 0.0f0
    i = lane
    @inbounds while valid && i < C
        z = ADD ? sumout[base + i + 1] : a[base + i + 1]
        d = Float32(z) - μ
        s2 += d * d
        i += ROW
    end
    if ROW == 64
        v = KI.shfl_down(s2, 32); lane < 32 && (s2 += v)
    end
    v = KI.shfl_down(s2, 16); lane < 16 && (s2 += v)
    v = KI.shfl_down(s2, 8);  lane < 8  && (s2 += v)
    v = KI.shfl_down(s2, 4);  lane < 4  && (s2 += v)
    v = KI.shfl_down(s2, 2);  lane < 2  && (s2 += v)
    v = KI.shfl_down(s2, 1);  lane < 1  && (s2 += v)
    r = 1.0f0 / sqrt(KI.shfl(s2, chunk * ROW) / n + eps)

    W = IW * NX
    H = IH * NY
    x = g % W
    tail = g ÷ W
    y = tail % H
    b = tail ÷ H
    ix, wx = x % IW, x ÷ IW
    iy, wy = y % IH, y ÷ IH
    outrow = ix + IW * (iy + IH * (wx + NX * (wy + NY * b)))
    outbase = outrow * Int(C)
    i = lane
    @inbounds while valid && i < C
        src = ADD ? sumout[base + i + 1] : a[base + i + 1]
        z = (Float32(src) - μ) * r
        HASG && (z *= Float32(γ[i + 1]))
        HASB && (z += Float32(β[i + 1]))
        # The ordinary result is provably dead when this route is selected; one
        # element per row keeps its declared transient live for Mantle without
        # writing the full fp32 tensor that no pass can read.
        (FULLWIDE || i == 0) && (wideout[base + i + 1] = z)
        out[outbase + i + 1] = z
        i += ROW
    end
    @inbounds if valid && lane == 0
        mean[g + 1] = μ
        rstd[g + 1] = r
    end
    return nothing
end

"""
    layernorm!(ctx, out, mean, rstd, a, γ, β, C, eps) -> out

Launch [`layernorm_kernel!`](@ref) over `length(a) ÷ C` groups.

`a` must be contiguous with the normalised axis fastest, which is what the
reversed layout gives for `native_layer_norm`: torch normalises over trailing
dims, and those are the leading Julia ones.
"""
function layernorm!(ctx, out, mean, rstd, a, γ, β, C::Integer, eps::Real)
    backend = ctx.backend
    groups = length(a) ÷ C
    rowsg = layernormwg(C)
    sg = hasproperty(ctx, :dev) ? ctx.dev.subgroup : 0
    dummy = γ === nothing ? (β === nothing ? a : β) : γ
    if sg > 0 && rowsg <= sg && sg % rowsg == 0
        harnesslaunch!(backend, layernorm_shfl_kernel!,
            out, mean, rstd, a,
            γ === nothing ? dummy : γ, β === nothing ? dummy : β,
            Int32(C), Int32(groups), Float32(eps), Val(sg), Val(rowsg),
            Val(γ !== nothing), Val(β !== nothing);
            ndrange = cld(groups, sg ÷ rowsg) * sg, workgroupsize = sg)
        return out
    end
    wg = rowsg
    harnesslaunch!(backend, layernorm_kernel!,
        out, mean, rstd, a,
        γ === nothing ? dummy : γ, β === nothing ? dummy : β,
        Int32(C), Float32(eps), Val(wg), Val(γ !== nothing), Val(β !== nothing);
        ndrange = groups * wg, workgroupsize = wg)
    out
end

"""
    rmsnorm_kernel!(out, rstd, a, γ, C, eps)

Root-mean-square norm — `x * rsqrt(mean(x^2) + eps) * γ` — one workgroup per
group, same layout contract as [`layernorm_kernel!`](@ref).

Not layer norm with the mean forced to zero, and the difference is a whole pass:
without a mean to centre about there is one reduction rather than two, so the
catastrophic-cancellation argument that keeps layer norm on two passes does not
apply here. `sum(x^2)` is the quantity, directly.

The accumulator is `Float32` whatever `a` is. Hunyuan3D normalises 128-element
attention heads in fp16, and squares of fp16 values above 256 already overflow
`Float16`'s 65504 — the sum would saturate to `Inf` and the whole head would come
back as zeros. That is the same failure the instance-norm path hit; here it is
closed by construction rather than narrowed.
"""
function rmsnorm_kernel!(out, rstd, a, γ,
                                           C::Int32, eps::Float32,
                                           ::Val{HASG}) where {HASG}
    red = KI.localmemory(Float32, Val((LN_WG,)), Val(1))
    g = KI.get_group_id().x - 1
    t = KI.get_local_id().x - 1
    base = g * Int(C)
    n = Float32(C)

    # ── pass 1: the mean square
    s = 0.0f0
    i = t
    @inbounds while i < C
        x = Float32(a[base + i + 1])
        s += x * x
        i += LN_WG
    end
    @inbounds red[t + 1] = s
    KI.barrier()
    # The barrier sits outside the `if`, so every lane reaches it — a barrier in
    # divergent control flow is undefined, and on this compiler that is not a
    # theoretical concern.
    stride = LN_WG ÷ 2
    while stride > 0
        @inbounds if t < stride
            red[t + 1] += red[t + 1 + stride]
        end
        KI.barrier()
        stride ÷= 2
    end
    @inbounds r = 1.0f0 / sqrt(red[1] / n + eps)
    KI.barrier()                      # nothing may overwrite `red` until all have read it

    # ── pass 2: scale
    i = t
    @inbounds while i < C
        x = Float32(a[base + i + 1]) * r
        if HASG
            x *= Float32(γ[i + 1])
        end
        out[base + i + 1] = x
        i += LN_WG
    end
    @inbounds if t == 0
        rstd[g + 1] = r
    end
    return nothing
end

# GROUPED RMS norm, as one op.
#
# `layernorm_num_groups = 4` means the mean square is taken over a QUARTER of the
# hidden dim at a time while the gain `γ` still spans the whole of it. `nn.RMSNorm`
# does not survive `run_decompositions()` in that form, so the export spells it
# out: cast, `pow`, `mean`, `add`, `rsqrt`, `mul`, `mul`, cast — eight ops, twice
# a layer, 129 times in a K2 Horizon 32B decode step. Each one touches 20 KB and
# costs a dispatch, and a dispatch on this device is ~13 us whatever it does, so
# the norms alone were ~10 ms of a 182 ms token.
#
# Identical to `rmsnorm_kernel!` except that `γ` is indexed by position within the
# ROW rather than within the group, which is the only thing "grouped" changes.
"""
    rmsgroup(C) -> Int

How wide a workgroup [`groupedrms_kernel!`](@ref) should reduce a row of `C`
with.

`LN_WG` for every row was the wrong answer where `C` is small: 128 lanes each
loading ONE value, then seven barriers to combine them. Qwen-Image 2.1 norms
per head, so `C` is 128 and that is the whole shape of the op -- 131776
workgroups of almost no work.

Measured on this M5 at fixed total elements, in GB/s, with the answers checked
against a host reference at every point:

    C      wg=32   wg=64   wg=128  wg=256
    64      25.3    16.7    11.7     5.8
    128     34.9    37.9    21.4    11.2
    256     64.1    45.7    37.3    20.5
    512     76.7    91.2    62.1    36.8
    4096    75.9    92.3    86.0    93.6

so the group wants to GROW with `C` rather than be a constant. This picks the
measured best or within 8% of it at every column above. It is deliberately not
"one subgroup always": that reading came from an earlier sweep where the kernel
still sized its shared memory from `LN_WG`, so every small-group number in it
was a NaN produced quickly.
"""
rmsgroup(C::Integer) = clamp(nextpow(2, max(1, cld(Int(C), 8))), 32, 256)

function groupedrms_kernel!(out, a, γ, C::Int32,
                                              NG::Int32, eps::Float32,
                                              ::Val{MIDROUND} = Val(false),
                                              ::Val{WG} = Val(LN_WG)) where {MIDROUND,WG}
    # `WG` and not the `LN_WG` constant: the shared array and the tree's first
    # stride have to be the size of the group that was actually LAUNCHED. They
    # were the constant, so launching this at any smaller group read `red` past
    # what the live lanes wrote -- NaN out, and fast, which is how a group-size
    # sweep can look like a 4.9x that is not there.
    red = KI.localmemory(Float32, Val((WG,)), Val(1))
    g = KI.get_group_id().x - 1
    t = KI.get_local_id().x - 1
    base = g * Int(C)
    gof = (g % Int(NG)) * Int(C)      # where this group starts inside the row
    n = Float32(C)

    s = 0.0f0
    i = t
    @inbounds while i < C
        x = Float32(a[base + i + 1])
        s += x * x
        i += WG
    end
    @inbounds red[t + 1] = s
    KI.barrier()
    # The barrier sits outside the `if`, as in `rmsnorm_kernel!` above.
    stride = WG ÷ 2
    while stride > 0
        @inbounds if t < stride
            red[t + 1] += red[t + 1 + stride]
        end
        KI.barrier()
        stride ÷= 2
    end
    @inbounds r = 1.0f0 / sqrt(red[1] / n + eps)
    KI.barrier()

    i = t
    @inbounds while i < C
        y = Float32(a[base + i + 1]) * r
        # `MIDROUND` is where the export put its cast. Qwen-Image 2.1 normalises
        # in fp32, narrows to fp16, and multiplies the fp16 gain into the fp16
        # value; K2 Horizon multiplies first and narrows once. The two differ by
        # a rounding, and a fusion that is not told which one it is replacing
        # silently changes the model it was meant to speed up.
        if MIDROUND
            y = Float32(eltype(out)(y))
        end
        out[base + i + 1] = eltype(out)(y * Float32(γ[gof + i + 1]))
        i += WG
    end
    return nothing
end

"""
    rmsnorm!(ctx, out, rstd, a, γ, C, eps) -> out

Launch [`rmsnorm_kernel!`](@ref) over `length(a) ÷ C` groups. Same contract as
[`layernorm!`](@ref): `a` contiguous with the normalised axis fastest.
"""
function rmsnorm!(ctx, out, rstd, a, γ, C::Integer, eps::Real)
    groups = length(a) ÷ C
    harnesslaunch!(ctx.backend, rmsnorm_kernel!,
        out, rstd, a, γ === nothing ? a : γ,
        Int32(C), Float32(eps), Val(γ !== nothing);
        ndrange = groups * LN_WG, workgroupsize = LN_WG)
    out
end
