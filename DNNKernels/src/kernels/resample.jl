"""
Resampling bodies: adaptive average pooling, bilinear upsampling, max pooling.

Plain functions of `(I, args...)`, launched through `launch!`.
"""

"""
`_adaptive_avg_pool2d`. Output extents come in as `Val` so the window
arithmetic folds where the ratio is exact (it always is on this model's paths).
"""
@inline function adaptive_avg_pool(I, x, ::Val{OX}, ::Val{OY}) where {OX,OY}
    ox, oy, c, n = I
    @inbounds begin
        IX, IY = size(x, 1), size(x, 2)
        x0 = ((ox - 1) * IX) ÷ OX
        x1 = (ox * IX + OX - 1) ÷ OX
        y0 = ((oy - 1) * IY) ÷ OY
        y1 = (oy * IY + OY - 1) ÷ OY
        acc = zero(eltype(x))
        for iy in (y0 + 1):y1, ix in (x0 + 1):x1
            acc += x[ix, iy, c, n]
        end
        acc / ((x1 - x0) * (y1 - y0))
    end
end

adaptive_avg_pool2d!(ctx, out, x) = launch!(ctx, adaptive_avg_pool, out, x,
                                           Val(size(out, 1)), Val(size(out, 2)))

"""
`upsample_bilinear2d` with `align_corners=false`: source coordinate
`(dst + 0.5) / scale - 0.5`, clamped, matching ATen.
"""
# `ALIGN` picks the coordinate convention. torch exposes both and the exported
# graphs use both: BasicVSR++'s SPyNet upsamples flow with `align_corners=true`
# and its pyramid with `false`. Implementing only one silently rescales by
# roughly (n-1)/n — invisible on a big tensor, and enough to move an optical-flow
# field by a pixel, which then shows up as a warp error several ops later.
@inline function upsample_bilinear(I, x, sx::Float32, sy::Float32, ::Val{ALIGN}) where {ALIGN}
    ox, oy, c, n = I
    @inbounds begin
        T = eltype(x)
        IX, IY = size(x, 1), size(x, 2)
        # `sx`/`sy` carry the convention: out/in for align_corners=false,
        # (out-1)/(in-1) for true, so the kernel needs no output extents.
        fx = ALIGN ? Float32(ox - 1) / sx : max((ox - 0.5f0) / sx - 0.5f0, 0.0f0)
        fy = ALIGN ? Float32(oy - 1) / sy : max((oy - 0.5f0) / sy - 0.5f0, 0.0f0)
        # `fastfloor` for the same reason as `upsample_nearest` below: `fx` is
        # bounded below by the `max` above and above by this `min`, so the
        # `InexactError` branch is unreachable and its allocation is not wanted
        # in a kernel.
        #
        # The three floors further down are NOT this case — they floor a
        # coordinate that comes from user data (a sampling grid, a deformable
        # offset), where out of range is genuinely reachable — so they are
        # `safetrunc(Int32, floor(…))` rather than either of the two: `fastfloor`
        # would be undefined there and `floor(Int32, …)` throws, which in a
        # kernel Lava lowers to a `gpu_gc_pool_alloc` whose undef return then
        # feeds the index anyway. `safetrunc` saturates, and every one of those
        # three sites already guards its index against the extents.
        x0 = min(fastfloor(fx), IX - 1)
        y0 = min(fastfloor(fy), IY - 1)
        tx = T(fx - x0)
        ty = T(fy - y0)
        x1 = min(x0 + 1, IX - 1)
        y1 = min(y0 + 1, IY - 1)
        v00 = x[x0 + 1, y0 + 1, c, n]; v10 = x[x1 + 1, y0 + 1, c, n]
        v01 = x[x0 + 1, y1 + 1, c, n]; v11 = x[x1 + 1, y1 + 1, c, n]
        top = muladd(v10 - v00, tx, v00)
        bot = muladd(v11 - v01, tx, v01)
        muladd(bot - top, ty, top)
    end
end

function upsample_bilinear2d!(ctx, out, x; align_corners::Bool = false)
    sx = align_corners ? (size(out, 1) > 1 ? Float32((size(out, 1) - 1) / max(size(x, 1) - 1, 1)) : 1.0f0) :
                         Float32(size(out, 1) / size(x, 1))
    sy = align_corners ? (size(out, 2) > 1 ? Float32((size(out, 2) - 1) / max(size(x, 2) - 1, 1)) : 1.0f0) :
                         Float32(size(out, 2) / size(x, 2))
    launch!(ctx, upsample_bilinear, out, x, sx, sy, Val(align_corners))
end

"""
`upsample_nearest2d`: source index `min(floor(dst * in/out), in - 1)`, which is
what ATen's `nearest_neighbor_compute_source_index` computes.

Note this is `nearest`, not `nearest-exact`: it floors `dst * scale` rather than
`(dst + 0.5) * scale`, so on an even upscale it repeats the *top-left* sample
instead of centring the window. The two differ by half an output pixel and
torch exports them as separate ATen ops, so there is no ambiguity to resolve
here — but they look identical on smooth data and only separate at edges, which
is where the FPN's upsampled features carry their signal.
"""
@inline function upsample_nearest(I, x, sx::Float32, sy::Float32)
    ox, oy, c, n = I
    @inbounds begin
        # `fastfloor`, not `floor(Int, …)`: the `min` below is the bound, so the
        # `InexactError` branch `floor(Int, …)` carries is unreachable — and a
        # throw in a kernel allocates its exception. See `fastfloor`.
        ix = min(fastfloor((ox - 1) * sx), size(x, 1) - 1)
        iy = min(fastfloor((oy - 1) * sy), size(x, 2) - 1)
        x[ix + 1, iy + 1, c, n]
    end
end

# The scale is derived from the extents rather than read from the op's
# `scale_factors`, and the two agree by construction: ATen uses `1/scale_factor`
# when the factors are given and `in/out` when they are not, and the output
# extent it records was computed from those same factors.
upsample_nearest2d!(ctx, out, x) = launch!(ctx, upsample_nearest, out, x,
                                           Float32(size(x, 1) / size(out, 1)),
                                           Float32(size(x, 2) / size(out, 2)))

"""
`cumsum` along one axis, one thread per output element.

Each thread walks the axis from its start, so the work is quadratic in the
scanned extent rather than the linear cost of a proper parallel scan. That is
deliberate: the only `cumsum` in any graph here builds SAM 2's dense positional
encoding from a constant 64x64 tensor, where the whole op is 262k adds and runs
once per decoder call. A work-efficient scan belongs here the moment something
scans a long axis; until then it would be untested code on a path nothing
exercises.
"""
@inline function cumsum_body(I, a, ::Val{D}) where {D}
    @inbounds begin
        acc = zero(eltype(a))
        for k in 1:I[D]
            acc += a[ntuple(j -> j == D ? k : I[j], Val(length(I)))...]
        end
        acc
    end
end

cumsum_dim!(ctx, out, a, d::Integer) = launch!(ctx, cumsum_body, out, a, Val(Int(d)))

@inline function maxpool(I, x, ::Val{KX}, ::Val{KY}, ::Val{SX}, ::Val{SY},
                         ::Val{PX}, ::Val{PY}) where {KX,KY,SX,SY,PX,PY}
    ox, oy, c, n = I
    @inbounds begin
        acc = typemin(eltype(x))
        bx = (ox - 1) * SX - PX
        by = (oy - 1) * SY - PY
        for ky in 1:KY
            iy = by + ky
            (iy < 1 || iy > size(x, 2)) && continue
            for kx in 1:KX
                ix = bx + kx
                (ix < 1 || ix > size(x, 1)) && continue
                acc = max(acc, x[ix, iy, c, n])
            end
        end
        acc
    end
end

"""
`maxpool` over a view described by its root, element offset and strides.

Pooling already visits every input coordinate in its window.  Addressing that
coordinate in the producer's layout avoids first writing a complete permuted
copy merely to read it here.  `XD`, `ST` and `OFF` are declaration-time view
metadata, so they live in the kernel type just like the window parameters do.
"""
@inline function maxpool_strided(I, x, ::Val{XD}, ::Val{ST}, ::Val{OFF},
                                 ::Val{KX}, ::Val{KY}, ::Val{SX}, ::Val{SY},
                                 ::Val{PX}, ::Val{PY}) where
                                {XD,ST,OFF,KX,KY,SX,SY,PX,PY}
    ox, oy, c, n = I
    @inbounds begin
        acc = typemin(eltype(x))
        bx = (ox - 1) * SX - PX
        by = (oy - 1) * SY - PY
        cn = OFF + (c - 1) * ST[3] + (n - 1) * ST[4]
        for ky in 1:KY
            iy = by + ky
            (iy < 1 || iy > XD[2]) && continue
            for kx in 1:KX
                ix = bx + kx
                (ix < 1 || ix > XD[1]) && continue
                acc = max(acc, x[cn + (ix - 1) * ST[1] +
                                     (iy - 1) * ST[2] + 1])
            end
        end
        acc
    end
end

# Pool channel-major `(C, 2W, 2H, B)` storage into dense `(W, H, C, B)` while
# transposing through a padded tile. Literal local-memory types avoid the GPU
# compiler bug described beside the attention transpose kernels.
for T in (Float16, Float32)
    @eval function $(Symbol("maxpool2transpose_pitched_", nameof(T), "!"))(
            out, src, off::Int32, batchstride::Int32,
            xstride::Int32, ystride::Int32, OW::Int32, OH::Int32, C::Int32)
        tile = KI.localmemory($(nameof(T)), Val((33, 32)), Val(1))
        tx, ty = Tuple(KI.get_local_id())
        gx, gy, gz = Tuple(KI.get_group_id())
        c0 = Int32(gx - 1) * Int32(32)
        m0 = Int32(gy - 1) * Int32(32)
        M = OW * OH
        ibase = off + Int32(gz - 1) * batchstride
        obase = Int32(gz - 1) * C * M
        @inbounds begin
            for j in Int32(0):Int32(7)
                c = c0 + Int32(tx)
                m = m0 + Int32(ty) + Int32(4) * j
                if c <= C && m <= M
                    ox = (m - Int32(1)) % OW
                    oy = (m - Int32(1)) ÷ OW
                    p = ibase + c + Int32(2) * ox * xstride +
                        Int32(2) * oy * ystride
                    tile[tx, ty + 4j] = max(max(src[p], src[p + xstride]),
                                            max(src[p + ystride], src[p + ystride + xstride]))
                else
                    tile[tx, ty + 4j] = typemin($(nameof(T)))
                end
            end
            KI.barrier()
            for j in Int32(0):Int32(7)
                m = m0 + Int32(tx)
                c = c0 + Int32(ty) + Int32(4) * j
                m <= M && c <= C &&
                    (out[obase + (c - Int32(1)) * M + m] = tile[ty + 4j, tx])
            end
        end
    end
end

maxpool2transposekernel(::Type{Float16}) = maxpool2transpose_pitched_Float16!
maxpool2transposekernel(::Type{Float32}) = maxpool2transpose_pitched_Float32!

maxpool2d!(ctx, out, x, k, s, p) = launch!(ctx, maxpool, out, x, Val(k[1]), Val(k[2]),
                                           Val(s[1]), Val(s[2]), Val(p[1]), Val(p[2]))

# ----------------------------------------------------------- BasicVSR++ kernels

"""
    avg_pool2d!(out, x, kw, kh, sx, sy, px, py)

Non-adaptive average pooling: fixed window and stride, unlike
`adaptive_avg_pool2d!` which derives the window from the output size.

`count_include_pad` is not modelled — every call in the graphs seen so far pools
with no padding, where the two agree. A padded call would need the divisor to
switch between the window area and the in-bounds count.
"""
function avg_pool2d_kernel!(out, x, kw::Int32, kh::Int32,
                                    sx::Int32, sy::Int32, px::Int32, py::Int32,
                                    W4::Int32, H4::Int32, C4::Int32)
    # 1-D ndrange + `coords4` — see `coords4`; the tuple index costs 3.5x at rank 4.
    # The bounds check `@kernel` used to insert. `launchgroup` fills a workgroup
    # up to 256 and need not divide the ndrange, so the last workgroup runs past
    # the end — `flat4` gives a flat count, not a multiple of the group.
    KI.get_global_id().x <= length(out) || return nothing
    i, j, c, n = coords4(Int32(KI.get_global_id().x) - Int32(1), W4, H4, C4)
    @inbounds begin
        W, H = size(x, 1), size(x, 2)
        acc = zero(Float32)
        x0 = (Int32(i) - Int32(1)) * sx - px
        y0 = (Int32(j) - Int32(1)) * sy - py
        for dy in Int32(0):(kh - Int32(1)), dx in Int32(0):(kw - Int32(1))
            xi = x0 + dx + Int32(1); yi = y0 + dy + Int32(1)
            if 1 <= xi <= W && 1 <= yi <= H
                acc += Float32(x[xi, yi, c, n])
            end
        end
        out[i, j, c, n] = eltype(out)(acc / Float32(kw * kh))
    end
    return nothing
end

function avg_pool2d!(out, x, kw::Integer, kh::Integer, sx::Integer, sy::Integer,
                     px::Integer = 0, py::Integer = 0)
    backend = KernelAbstractions.get_backend(out)
    nd, W4, H4, C4 = flat4(out)
    harnesslaunch!(backend, avg_pool2d_kernel!, out, x, Int32(kw), Int32(kh), Int32(sx), Int32(sy),
                                Int32(px), Int32(py), W4, H4, C4;
                                ndrange = nd, workgroupsize = launchgroup(nd))
    return out
end

"""
    grid_sample2d!(out, x, grid)

Bilinear `grid_sampler_2d` with `align_corners = true` and zero padding — the
combination `flow_warp` uses, and the only one the exported graphs ask for.

`grid` is **`(2, W, H, N)`** in Julia order: torch's grid is `(N, H, W, 2)` and
reversing puts the coordinate axis FIRST, not last. `grid[1, i, j, n]` is the x
to sample, `grid[2, ...]` the y. Getting this backwards costs a `W`-sized output
instead of the real width and shows up much later as a `cat` mismatch.

Out-of-range samples contribute zero rather than clamping, which is what makes a
warp reveal black at the frame edge instead of smearing the border pixel.
"""
function grid_sample2d_kernel!(out, x, grid, ::Val{ALIGN},
                                       ::Val{PAD}, W4::Int32, H4::Int32,
                                       C4::Int32) where {ALIGN,PAD}
    # 1-D ndrange + `coords4`, not `Tuple(KI.get_global_id())`: the tuple index costs
    # 3.5x at rank 4 here, and a bare copy over this ndrange was 88% of this
    # kernel's runtime. See `coords4`.
    #
    # The bounds check `@kernel` used to insert: `launchgroup` fills a workgroup
    # up to 256 and need not divide a flat count.
    KI.get_global_id().x <= length(out) || return nothing
    i, j, c, n = coords4(Int32(KI.get_global_id().x) - Int32(1), W4, H4, C4)
    @inbounds begin
        W, H = size(x, 1), size(x, 2)
        gx = Float32(grid[1, i, j, n]); gy = Float32(grid[2, i, j, n])
        # normalized [-1,1] -> pixel index (1-based)
        px = ALIGN ? (gx + 1.0f0) * 0.5f0 * (Float32(W) - 1.0f0) + 1.0f0 :
                     ((gx + 1.0f0) * Float32(W) - 1.0f0) * 0.5f0 + 1.0f0
        py = ALIGN ? (gy + 1.0f0) * 0.5f0 * (Float32(H) - 1.0f0) + 1.0f0 :
                     ((gy + 1.0f0) * Float32(H) - 1.0f0) * 0.5f0 + 1.0f0
        x0 = safetrunc(Int32, floor(px)); y0 = safetrunc(Int32, floor(py))
        fx = px - Float32(x0); fy = py - Float32(y0)
        acc = zero(Float32)
        for dy in Int32(0):Int32(1), dx in Int32(0):Int32(1)
            xi = x0 + dx; yi = y0 + dy
            wgt = (dx == 0 ? 1.0f0 - fx : fx) * (dy == 0 ? 1.0f0 - fy : fy)
            if PAD === :border
                # clamp instead of dropping: the edge pixel is repeated outward,
                # which is what `grid_sampler_2d`'s padding_mode=1 means
                xc = min(max(xi, Int32(1)), Int32(W)); yc = min(max(yi, Int32(1)), Int32(H))
                acc += wgt * Float32(x[xc, yc, c, n])
            elseif 1 <= xi <= W && 1 <= yi <= H
                acc += wgt * Float32(x[xi, yi, c, n])
            end
        end
        out[i, j, c, n] = eltype(out)(acc)
    end
    return nothing
end

"""
    grid_sample2d!(out, x, grid; align_corners, padding) -> out

**Asynchronous, like every other op launcher here.** This file's five launchers
and `embedding`'s each ended with `KernelAbstractions.synchronize(backend)` — a
full pipeline drain per op, inside `runop!`, inside `execute!`. RIFE runs
**eighteen** `grid_sampler_2d` per interpolation, so that was eighteen drains a
frame; a graph is a queue of dependent kernels and the executor owns the
synchronisation, not the individual op.

Nothing needs the host to wait here: the device orders the writes against any
later kernel that reads `out` through the batch's own buffer dependencies. A
caller that genuinely wants the result on the host syncs there, which is the one
place that can know it.

**What it cost, and why**, sampling `utilization.gpu` from outside the process
during a sustained RIFE loop with this one sync toggled:

                  wall     util    GPU busy   GPU IDLE
    with sync    98.7 ms   68.7%    67.8 ms    30.9 ms
    without      91.9      71.8%    66.0       25.9

GPU busy time is unchanged — the sync does no work — and the whole 6.1 ms is
idle: host recording that was hiding under execution, exposed. An MWE of the same
18 launches reproduces only 0.87 ms of that, because its filler ops were trivial
broadcasts where these are convolutions with plan lookups and scratch allocation.

The row worth staring at is the last one: **RIFE idles ~26 ms even without the
sync, 28% of the model.** 350 ops is more than this host can record while the
card stays fed.
"""
function grid_sample2d!(out, x, grid; align_corners::Bool = true, padding::Symbol = :zeros)
    backend = KernelAbstractions.get_backend(out)
    nd, W4, H4, C4 = flat4(out)
    harnesslaunch!(backend, grid_sample2d_kernel!, out, x, grid, Val(align_corners), Val(padding),
                                   W4, H4, C4;
                                   ndrange = nd, workgroupsize = launchgroup(nd))
    return out
end

"""
    deform_conv2d!(out, x, offset, mask, w, bias, sx, sy, px, py, dx, dy)

Modulated deformable convolution v2 — torchvision's `deform_conv2d`, which is
what BasicVSR++'s `SecondOrderDeformableAlignment` runs 16 times a clip.

Each output position samples its `KW x KH` window at *learned* offsets rather
than on the grid, bilinearly, and scales each tap by a learned mask. So it is a
convolution whose gather pattern is data-dependent: the im2col + GEMM route that
carries the ordinary convolutions here does not apply, because there is no fixed
matrix to build. This is a direct kernel, one thread per output element.

Layouts follow the exported graph (Julia order, torch reversed):
`x` `(W,H,Cin,N)`, `offset` `(W,H,2*dg*KH*KW,N)`, `mask` `(W,H,dg*KH*KW,N)`,
`w` `(KW,KH,Cin/groups,Cout)`. Within the offset channel block torch orders the
taps `(kh, kw)` with y before x, which is the one detail that silently produces a
plausible-but-wrong warp if mirrored.
"""
function deform_conv2d_kernel!(out, x, offset, mask,
                                       w, bias,
                                       sx::Int32, sy::Int32, px::Int32, py::Int32,
                                       dlx::Int32, dly::Int32, dg::Int32, groups::Int32,
                                       ::Val{HASMASK}, W4::Int32, H4::Int32,
                                       C4::Int32) where {HASMASK}
    # The bounds check `@kernel` used to insert. `launchgroup` fills a workgroup
    # up to 256 and need not divide the ndrange, so the last workgroup runs past
    # the end — `flat4` gives a flat count, not a multiple of the group.
    KI.get_global_id().x <= length(out) || return nothing
    i, j, co, n = coords4(Int32(KI.get_global_id().x) - Int32(1), W4, H4, C4)
    @inbounds begin
        W, H, Cin = size(x, 1), size(x, 2), size(x, 3)
        KW, KH, Cpg = size(w, 1), size(w, 2), size(w, 3)
        Cout = size(w, 4)
        g = Int32((co - 1) ÷ (Cout ÷ groups))          # which weight group
        cbase = g * Cpg
        acc = bias === nothing ? zero(Float32) : Float32(bias[co])
        x0 = (Int32(i) - Int32(1)) * sx - px
        y0 = (Int32(j) - Int32(1)) * sy - py
        for kh in Int32(1):Int32(KH), kw in Int32(1):Int32(KW)
            tap = (kh - Int32(1)) * Int32(KW) + (kw - Int32(1))   # torch: y outer
            # The deform group is a partition of the INPUT channels — each group
            # carries its own offset field — so it depends on `ci`, not on the
            # output channel. Deriving it from `co` gives a result that is the
            # right shape and plausible everywhere, and wrong by ~9 in this test.
            for ci in Int32(1):Int32(Cpg)
                cin = cbase + ci
                dgi = Int32((cin - Int32(1)) * dg ÷ Int32(Cin))
                och = dgi * Int32(KH) * Int32(KW) * Int32(2) + tap * Int32(2)
                offy = Float32(offset[i, j, och + Int32(1), n])
                offx = Float32(offset[i, j, och + Int32(2), n])
                m = HASMASK ?
                    Float32(mask[i, j, dgi * Int32(KH) * Int32(KW) + tap + Int32(1), n]) : 1.0f0
                sxp = Float32(x0 + (kw - Int32(1)) * dlx) + offx + 1.0f0
                syp = Float32(y0 + (kh - Int32(1)) * dly) + offy + 1.0f0
                xi0 = safetrunc(Int32, floor(sxp)); yi0 = safetrunc(Int32, floor(syp))
                fx = sxp - Float32(xi0); fy = syp - Float32(yi0)
                wv = Float32(w[kw, kh, ci, co])
                sacc = zero(Float32)
                for ddy in Int32(0):Int32(1), ddx in Int32(0):Int32(1)
                    xi = xi0 + ddx; yi = yi0 + ddy
                    if 1 <= xi <= W && 1 <= yi <= H
                        bw = (ddx == 0 ? 1.0f0 - fx : fx) * (ddy == 0 ? 1.0f0 - fy : fy)
                        sacc += bw * Float32(x[xi, yi, cin, n])
                    end
                end
                acc += wv * m * sacc
            end
        end
        out[i, j, co, n] = eltype(out)(acc)
    end
    return nothing
end

function deform_conv2d!(out, x, offset, mask, w, bias;
                        stride = (1, 1), padding = (0, 0), dilation = (1, 1),
                        groups::Integer = 1, deform_groups::Integer = 1)
    backend = KernelAbstractions.get_backend(out)
    nd, W4, H4, C4 = flat4(out)
    harnesslaunch!(backend, deform_conv2d_kernel!, out, x, offset,
                                   mask === nothing ? offset : mask, w, bias,
                                   Int32(stride[1]), Int32(stride[2]),
                                   Int32(padding[1]), Int32(padding[2]),
                                   Int32(dilation[1]), Int32(dilation[2]),
                                   Int32(deform_groups), Int32(groups),
                                   Val(mask !== nothing), W4, H4, C4;
                                   ndrange = nd, workgroupsize = launchgroup(nd))
    return out
end

"""
    deform_im2col_kernel!(col, x, offset, mask, …)

The gathered window of a modulated deformable convolution, as a matrix, once.

The direct kernel below re-gathers every tap for every OUTPUT channel, and the
gather is the whole cost: at `Cout = 64` that is sixty-four reads of the same
bilinear sample. Nothing in the gather depends on `co` -- the offsets and the mask
are indexed by pixel, deform group and tap, and the sample by the INPUT channel --
so the window can be built once and multiplied by the weights as a GEMM. That is
what torchvision does and what the note above this file used to deny: the pattern
is data-dependent, but the MATRIX is still there to build.

Measured on BasicVSR++, whose `SecondOrderDeformableAlignment` runs sixteen of
these a clip: 52.5 ms each on the direct kernel, 0.0115 TFLOP/s, 71% of the whole
frame.

`col` is `(W*H*N, KW*KH*Cin)`, whose column index is `kw + KW*(kh-1) +
KW*KH*(cin-1)` -- the order `w`'s own `(KW, KH, Cin, Cout)` memory already has, so
the weight reshapes into the GEMM's right operand for free and the result lands
in `out`'s `(W, H, Cout, N)` as it stands.
"""
function deform_im2col_kernel!(col, x, offset, mask,
                                       sx::Int32, sy::Int32, px::Int32, py::Int32,
                                       dlx::Int32, dly::Int32, dg::Int32,
                                       ::Val{HASMASK}, KW::Int32, KH::Int32,
                                       W4::Int32, H4::Int32, NPIX::Int32) where {HASMASK}
    # The bounds check `@kernel` used to insert: the ndrange is `length(col)` and
    # `launchgroup` need not divide it.
    KI.get_global_id().x <= length(col) || return nothing
    t = Int32(KI.get_global_id().x) - Int32(1)
    @inbounds begin
        pix = t % NPIX                       # (i, j, n) flattened, `col`'s row
        rest = t ÷ NPIX
        kw = rest % KW
        rest = rest ÷ KW
        kh = rest % KH
        cin = rest ÷ KH                      # 0-based input channel
        W, H, Cin = size(x, 1), size(x, 2), size(x, 3)
        i = pix % W4
        j = (pix ÷ W4) % H4
        n = pix ÷ (W4 * H4)
        tap = kh * KW + kw                   # torch orders the taps y-outer
        dgi = Int32(cin * dg ÷ Cin)
        och = dgi * KH * KW * Int32(2) + tap * Int32(2)
        offy = Float32(offset[i + Int32(1), j + Int32(1), och + Int32(1), n + Int32(1)])
        offx = Float32(offset[i + Int32(1), j + Int32(1), och + Int32(2), n + Int32(1)])
        m = HASMASK ? Float32(mask[i + Int32(1), j + Int32(1),
                                   dgi * KH * KW + tap + Int32(1), n + Int32(1)]) : 1.0f0
        sxp = Float32(i * sx - px + kw * dlx) + offx + 1.0f0
        syp = Float32(j * sy - py + kh * dly) + offy + 1.0f0
        xi0 = safetrunc(Int32, floor(sxp)); yi0 = safetrunc(Int32, floor(syp))
        fx = sxp - Float32(xi0); fy = syp - Float32(yi0)
        sacc = zero(Float32)
        for ddy in Int32(0):Int32(1), ddx in Int32(0):Int32(1)
            xi = xi0 + ddx; yi = yi0 + ddy
            if 1 <= xi <= W && 1 <= yi <= H
                bw = (ddx == 0 ? 1.0f0 - fx : fx) * (ddy == 0 ? 1.0f0 - fy : fy)
                sacc += bw * Float32(x[xi, yi, cin + Int32(1), n + Int32(1)])
            end
        end
        col[t + Int32(1)] = eltype(col)(m * sacc)
    end
    return nothing
end

"""
    flip!(out, x, dims)

Reverse `x` along `dims` on whatever backend it lives on.

`Base.reverse` allocates and then *scalar-indexes* to fill, which a device array
refuses ("scalar iteration is disallowed"), so the host fallback is not an
option here. The reversed dimensions are a `Val` so the index arithmetic
specialises and the kernel stays branch-free.
"""
function flip_kernel!(out, x, ::Val{DIMS}, W4::Int32, H4::Int32, C4::Int32) where {DIMS}
    # The bounds check `@kernel` used to insert. `launchgroup` fills a workgroup
    # up to 256 and need not divide the ndrange, so the last workgroup runs past
    # the end — `flat4` gives a flat count, not a multiple of the group.
    KI.get_global_id().x <= length(out) || return nothing
    I = coords4(Int32(KI.get_global_id().x) - Int32(1), W4, H4, C4)
    @inbounds begin
        J = ntuple(k -> (k in DIMS) ? size(x, k) - I[k] + 1 : I[k], length(I))
        out[I...] = x[J...]
    end
    return nothing
end

function flip!(out, x, dims::Tuple)
    backend = KernelAbstractions.get_backend(out)
    nd, W4, H4, C4 = flat4(x)
    harnesslaunch!(backend, flip_kernel!, out, x, Val(dims), W4, H4, C4;
                          ndrange = nd, workgroupsize = launchgroup(nd))
    return out
end

# ------------------------------------------------------------- Wan VAE kernels

"""
    convolution3d!(out, x, w, bias, stride, pad, dil, groups)

Direct 3-D convolution. `aten::convolution` covers 1-D, 2-D and 3-D and the rank
is only visible in the length of its stride/padding tuples, so a 3-D call that
falls into the 2-D branch is silently wrong rather than an error — which is why
`runop!` dispatches on that length.

Layouts are the reversed (Julia) order the graphs use: `x` `(W,H,D,Cin,N)`,
`w` `(KW,KH,KD,Cin/groups,Cout)`, `out` `(OW,OH,OD,Cout,N)`.

No im2col here. The Wan VAE's 3-D convolutions are 116 of its 1825 nodes and
mostly small in the depth axis, and an im2col matrix gains a whole extra
dimension — this is the correctness-first path, to be revisited when the VAE is
actually on a hot loop.
"""
function conv3d_kernel!(out, x, w, bias,
                                sx::Int32, sy::Int32, sz::Int32,
                                px::Int32, py::Int32, pz::Int32,
                                dx::Int32, dy::Int32, dz::Int32, groups::Int32,
                                ::Val{ACT}) where {ACT}
    # Three global axes, not the output's five: `@index(Global, NTuple)` carried
    # KernelAbstractions' flattening of an N-dimensional ndrange and
    # `KernelInterface` has none. The launch folds `(W, H, D, Cout, N)` to
    # `(W, H, D * Cout * N)` and the depth, channel and batch axes are recovered
    # here from the output's own extents.
    i, j, dcn = Tuple(KI.get_global_id())
    # The bounds check `@kernel` used to insert; `launchgroup` need not divide.
    (i <= size(out, 1) && j <= size(out, 2) &&
     dcn <= size(out, 3) * size(out, 4) * size(out, 5)) || return nothing
    D_, Cout_ = size(out, 3), size(out, 4)
    k  = (dcn - 1) % D_ + 1
    co = ((dcn - 1) ÷ D_) % Cout_ + 1
    n  = (dcn - 1) ÷ (D_ * Cout_) + 1
    @inbounds begin
        W, H, D, Cin = size(x, 1), size(x, 2), size(x, 3), size(x, 4)
        KW, KH, KD, Cpg = size(w, 1), size(w, 2), size(w, 3), size(w, 4)
        Cout = size(w, 5)
        g = Int32((co - 1) ÷ (Cout ÷ groups))
        cbase = g * Int32(Cpg)
        acc = bias === nothing ? zero(Float32) : Float32(bias[co])
        x0 = (Int32(i) - Int32(1)) * sx - px
        y0 = (Int32(j) - Int32(1)) * sy - py
        z0 = (Int32(k) - Int32(1)) * sz - pz
        for kd in Int32(1):Int32(KD)
            zi = z0 + (kd - Int32(1)) * dz + Int32(1)
            (zi < 1 || zi > D) && continue
            for kh in Int32(1):Int32(KH)
                yi = y0 + (kh - Int32(1)) * dy + Int32(1)
                (yi < 1 || yi > H) && continue
                for kw in Int32(1):Int32(KW)
                    xi = x0 + (kw - Int32(1)) * dx + Int32(1)
                    (xi < 1 || xi > W) && continue
                    for ci in Int32(1):Int32(Cpg)
                        acc += Float32(w[kw, kh, kd, ci, co]) *
                               Float32(x[xi, yi, zi, cbase + ci, n])
                    end
                end
            end
        end
        ACT === :relu && (acc = max(acc, 0.0f0))
        out[i, j, k, co, n] = eltype(out)(acc)
    end
    return nothing
end

function convolution3d!(out, x, w, bias, stride, pad, dil, groups::Integer;
                        act::Symbol = :none)
    backend = KernelAbstractions.get_backend(out)
    # Folded to three axes; `conv3d_kernel!` unfolds the trailing three.
    nd = (size(out, 1), size(out, 2), size(out, 3) * size(out, 4) * size(out, 5))
    harnesslaunch!(backend, conv3d_kernel!, out, x, w, bias === nothing ? w : bias,
                            Int32(stride[1]), Int32(stride[2]), Int32(stride[3]),
                            Int32(pad[1]), Int32(pad[2]), Int32(pad[3]),
                            Int32(dil[1]), Int32(dil[2]), Int32(dil[3]),
                            Int32(groups), Val(act); ndrange = nd,
                            workgroupsize = launchgroup(nd))
    return out
end
