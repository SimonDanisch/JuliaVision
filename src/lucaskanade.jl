# Sparse Lucas-Kanade point tracker on the GPU: one thread per feature solves the 2×2
# structure-tensor system iteratively to sub-pixel-track it from `I0` to `I1`. Being
# gradient-based it has none of the NCC peak-locking bias on repetitive texture that
# defeats template matching; the caller pairs it with forward-backward validation and a
# RANSAC global-motion fit. This is the tracking half a virtual-tripod stabiliser needs.

"Bilinear sample of `img` at floating `(x, y)` (device-inlinable; clamps to the border)."
@inline function bilinsample(img, x::Float32, y::Float32)
    W = size(img, 1); H = size(img, 2)
    x = clamp(x, 1.0f0, Float32(W) - 0.001f0)
    y = clamp(y, 1.0f0, Float32(H) - 0.001f0)
    x0 = unsafe_trunc(Int32, x); y0 = unsafe_trunc(Int32, y)
    fx = x - x0; fy = y - y0
    o = one(Int32)
    @inbounds (1 - fx) * (1 - fy) * img[x0, y0] + fx * (1 - fy) * img[x0 + o, y0] +
              (1 - fx) * fy * img[x0, y0 + o] + fx * fy * img[x0 + o, y0 + o]
end

# One thread per feature. `Ix, Iy` are the gradients of `I0`. `win` = half window,
# `iters` = max Gauss-Newton steps. Writes the tracked position and a 0/1 validity flag
# (0 if the point left the frame, the window is textureless, or the solve diverged).
@kernel function lucaskanade_kernel!(qx, qy, valid, @Const(I0), @Const(I1),
                                     @Const(Ix), @Const(Iy), @Const(px), @Const(py),
                                     win::Int32, iters::Int32)
    k = @index(Global)
    W = Int32(size(I0, 1)); H = Int32(size(I0, 2))
    fpx = px[k]; fpy = py[k]
    icx = unsafe_trunc(Int32, fpx + 0.5f0)
    icy = unsafe_trunc(Int32, fpy + 0.5f0)
    v = Int32(0); rx = fpx; ry = fpy
    o = one(Int32)
    if icx > win + Int32(2) && icx < W - win - Int32(2) &&
       icy > win + Int32(2) && icy < H - win - Int32(2)
        a = 0.0f0; b = 0.0f0; c = 0.0f0
        j = -win
        while j <= win
            i = -win
            while i <= win
                @inbounds gx = Ix[icx + i, icy + j]
                @inbounds gy = Iy[icx + i, icy + j]
                a += gx * gx; b += gx * gy; c += gy * gy
                i += o
            end
            j += o
        end
        det = a * c - b * b
        if det > 1.0f-4 && a > 1.0f-4 && c > 1.0f-4
            dx = 0.0f0; dy = 0.0f0; ok = true
            it = o
            while it <= iters
                bx = 0.0f0; by = 0.0f0
                j = -win
                while j <= win
                    i = -win
                    while i <= win
                        s = bilinsample(I1, Float32(icx + i) + dx, Float32(icy + j) + dy)
                        @inbounds tt = s - I0[icx + i, icy + j]
                        @inbounds bx += Ix[icx + i, icy + j] * tt
                        @inbounds by += Iy[icx + i, icy + j] * tt
                        i += o
                    end
                    j += o
                end
                ddx = (c * bx - b * by) / det
                ddy = (-b * bx + a * by) / det
                dx -= ddx; dy -= ddy
                (ddx * ddx + ddy * ddy < 1.0f-4) && break
                if abs(dx) > Float32(win) || abs(dy) > Float32(win)
                    ok = false; break
                end
                it += o
            end
            if ok
                v = o; rx = fpx + dx; ry = fpy + dy
            end
        end
    end
    qx[k] = rx; qy[k] = ry; valid[k] = v
end

"""
    lucaskanade!(qx, qy, valid, I0, I1, Ix, Iy, px, py; win=11, iters=15)

Track the `px, py` feature points from `I0` to `I1` by sparse Lucas-Kanade on the
backend of the arrays (`Ix, Iy` = gradients of `I0`). Fills `qx, qy` with the tracked
positions and `valid` with 1/0 per point. All arrays live on the same KA backend; run
[`gradients!`](@ref) to get `Ix, Iy`. Pair with a reverse call for forward-backward
error rejection.
"""
function lucaskanade!(qx, qy, valid, I0, I1, Ix, Iy, px, py; win::Integer = 11,
                      iters::Integer = 15)
    backend = KA.get_backend(I0)
    lucaskanade_kernel!(backend)(qx, qy, valid, I0, I1, Ix, Iy, px, py,
                                 Int32(win), Int32(iters); ndrange = length(px))
    KA.synchronize(backend)
    return qx, qy, valid
end
