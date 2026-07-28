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
@inline function upsample_bilinear(I, x, sx::Float32, sy::Float32)
    ox, oy, c, n = I
    @inbounds begin
        T = eltype(x)
        IX, IY = size(x, 1), size(x, 2)
        fx = max((ox - 0.5f0) / sx - 0.5f0, 0.0f0)
        fy = max((oy - 0.5f0) / sy - 0.5f0, 0.0f0)
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

function upsample_bilinear2d!(out, x)
    launch!(upsample_bilinear, out, x,
            Float32(size(out, 1) / size(x, 1)), Float32(size(out, 2) / size(x, 2)))
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
