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

adaptive_avg_pool2d!(out, x) = launch!(adaptive_avg_pool, out, x,
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
        x0 = min(floor(Int, fx), IX - 1)
        y0 = min(floor(Int, fy), IY - 1)
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

function upsample_bilinear2d!(out, x; align_corners::Bool = false)
    sx = align_corners ? (size(out, 1) > 1 ? Float32((size(out, 1) - 1) / max(size(x, 1) - 1, 1)) : 1.0f0) :
                         Float32(size(out, 1) / size(x, 1))
    sy = align_corners ? (size(out, 2) > 1 ? Float32((size(out, 2) - 1) / max(size(x, 2) - 1, 1)) : 1.0f0) :
                         Float32(size(out, 2) / size(x, 2))
    launch!(upsample_bilinear, out, x, sx, sy, Val(align_corners))
end

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

maxpool2d!(out, x, k, s, p) = launch!(maxpool, out, x, Val(k[1]), Val(k[2]),
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
@kernel function avg_pool2d_kernel!(out, @Const(x), kw::Int32, kh::Int32,
                                    sx::Int32, sy::Int32, px::Int32, py::Int32)
    i, j, c, n = @index(Global, NTuple)
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
end

function avg_pool2d!(out, x, kw::Integer, kh::Integer, sx::Integer, sy::Integer,
                     px::Integer = 0, py::Integer = 0)
    backend = KernelAbstractions.get_backend(out)
    avg_pool2d_kernel!(backend)(out, x, Int32(kw), Int32(kh), Int32(sx), Int32(sy),
                                Int32(px), Int32(py); ndrange = size(out))
    KernelAbstractions.synchronize(backend)
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
@kernel function grid_sample2d_kernel!(out, @Const(x), @Const(grid), ::Val{ALIGN},
                                       ::Val{PAD}) where {ALIGN,PAD}
    i, j, c, n = @index(Global, NTuple)
    @inbounds begin
        W, H = size(x, 1), size(x, 2)
        gx = Float32(grid[1, i, j, n]); gy = Float32(grid[2, i, j, n])
        # normalized [-1,1] -> pixel index (1-based)
        px = ALIGN ? (gx + 1.0f0) * 0.5f0 * (Float32(W) - 1.0f0) + 1.0f0 :
                     ((gx + 1.0f0) * Float32(W) - 1.0f0) * 0.5f0 + 1.0f0
        py = ALIGN ? (gy + 1.0f0) * 0.5f0 * (Float32(H) - 1.0f0) + 1.0f0 :
                     ((gy + 1.0f0) * Float32(H) - 1.0f0) * 0.5f0 + 1.0f0
        x0 = floor(Int32, px); y0 = floor(Int32, py)
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
end

function grid_sample2d!(out, x, grid; align_corners::Bool = true, padding::Symbol = :zeros)
    backend = KernelAbstractions.get_backend(out)
    grid_sample2d_kernel!(backend)(out, x, grid, Val(align_corners), Val(padding);
                                   ndrange = size(out))
    KernelAbstractions.synchronize(backend)
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
@kernel function deform_conv2d_kernel!(out, @Const(x), @Const(offset), @Const(mask),
                                       @Const(w), @Const(bias),
                                       sx::Int32, sy::Int32, px::Int32, py::Int32,
                                       dlx::Int32, dly::Int32, dg::Int32, groups::Int32,
                                       ::Val{HASMASK}) where {HASMASK}
    i, j, co, n = @index(Global, NTuple)
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
                xi0 = floor(Int32, sxp); yi0 = floor(Int32, syp)
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
end

function deform_conv2d!(out, x, offset, mask, w, bias;
                        stride = (1, 1), padding = (0, 0), dilation = (1, 1),
                        groups::Integer = 1, deform_groups::Integer = 1)
    backend = KernelAbstractions.get_backend(out)
    deform_conv2d_kernel!(backend)(out, x, offset,
                                   mask === nothing ? offset : mask, w, bias,
                                   Int32(stride[1]), Int32(stride[2]),
                                   Int32(padding[1]), Int32(padding[2]),
                                   Int32(dilation[1]), Int32(dilation[2]),
                                   Int32(deform_groups), Int32(groups),
                                   Val(mask !== nothing); ndrange = size(out))
    KernelAbstractions.synchronize(backend)
    return out
end
