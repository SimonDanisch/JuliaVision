# Trilinear apply of a 3D colour LUT.
#
# This is the second half of the Image-Adaptive 3D LUT port: `NeuralLUTRunner`
# predicts the table and this applies it. It lives here rather than in the graph
# because it is a per-pixel image kernel with no learned parameters — the same
# reason `coloradjust!` is here — and because a LUT is a grading object the user
# can also author or import, so the apply has to work without a network at all.
#
# ## Layout
#
# `lut` is `(D, D, D, 3)` indexed `lut[r, g, b, c]`, which is what
# `readsafetensors` produces from the exporter's torch `(3, D, D, D)` — the raw
# buffer reshaped with reversed dims is the Julia column-major view of the same
# bytes, so no permute happens anywhere. Upstream's CUDA kernel computes
# `r_id + g_id*dim + b_id*dim^2 + c*dim^3` into the row-major tensor, which is
# exactly this array with red fastest-varying.
#
# ## Where the coordinates come from
#
# Upstream uses `binsize = 1.000001 / (D - 1)`, and the fudge factor is load
# bearing rather than sloppy: with a plain `1/(D-1)`, an input of exactly 1.0
# gives `floor(1.0/binsize) == D-1`, whose upper neighbour is one past the end
# of the table. Upstream reads it anyway and multiplies by a zero weight, which
# is an out-of-bounds read that happens to be harmless on CUDA. The 1.000001
# pulls the index back to `D-2` with a fractional part of ~1, so the read stays
# in bounds.
#
# The fudge is kept, so the arithmetic matches PyTorch bit for bit in the
# interior, **and** the upper index is clamped, so an input above 1.0 (which the
# editor can produce — a previous filter in the chain need not have clamped)
# cannot walk off the table. Upstream would read whatever followed it in memory.

"""
    lut3d!(out, img, lut; binsize = nothing)
    lut3d!(img, lut)

Apply a 3D colour LUT by trilinear interpolation.

`lut` is a `(D, D, D, 3)` array of `Float32` indexed `[r, g, b, channel]`, on the
same backend as the image. `img` is any `AbstractMatrix{<:AbstractRGB}`; the
two-argument form works in place.

`binsize` defaults to upstream's `1.000001 / (D - 1)` — see the comment above for
why that constant is not `1 / (D - 1)`. It is exposed because a LUT authored
elsewhere (a `.cube` file, say) is defined on a plain `1/(D-1)` grid, and
applying it on the fudged one is a fraction-of-a-bin shift across the whole
image.

Asynchronous, like every kernel here: synchronize before reading on the host.
"""
function lut3d! end

# `inp` is deliberately not `@Const`: the two-argument form passes the same array
# as both, and `@Const` promises the compiler that binding does not alias a
# written one. The read is pointwise so aliasing is harmless, but the promise
# would be false. `lut` is never written by anyone, so it keeps the annotation.
@kernel function lut3d_kernel!(out, inp, @Const(lut), binsize::Float32,
                               dim::Int32)
    I = @index(Global, Cartesian)
    c = tofloat(inp[I])

    # 0-based bin index and the fraction within the bin, as upstream computes
    # them. `fmod(v, binsize) / binsize` for v >= 0 is `v/binsize - floor(v/binsize)`,
    # so it is written that way to avoid a second division.
    fr = c.r / binsize
    fg = c.g / binsize
    fb = c.b / binsize
    ir = floor(fr)
    ig = floor(fg)
    ib = floor(fb)
    dr = fr - ir
    dg = fg - ig
    db = fb - ib

    # 0-based -> 1-based, and clamped at both ends. The low clamp catches a
    # negative input (an unclamped upstream filter), the high clamp catches the
    # upper neighbour at the top of the table.
    r0 = clamp(unsafe_trunc(Int32, ir) + Int32(1), Int32(1), dim)
    g0 = clamp(unsafe_trunc(Int32, ig) + Int32(1), Int32(1), dim)
    b0 = clamp(unsafe_trunc(Int32, ib) + Int32(1), Int32(1), dim)
    r1 = min(r0 + Int32(1), dim)
    g1 = min(g0 + Int32(1), dim)
    b1 = min(b0 + Int32(1), dim)

    # A negative or >1 input has already been clamped into the end bin above; the
    # fraction has to follow it or the interpolation reintroduces the excursion.
    dr = clamp(dr, 0.0f0, 1.0f0)
    dg = clamp(dg, 0.0f0, 1.0f0)
    db = clamp(db, 0.0f0, 1.0f0)

    w000 = (1.0f0 - dr) * (1.0f0 - dg) * (1.0f0 - db)
    w100 = dr * (1.0f0 - dg) * (1.0f0 - db)
    w010 = (1.0f0 - dr) * dg * (1.0f0 - db)
    w110 = dr * dg * (1.0f0 - db)
    w001 = (1.0f0 - dr) * (1.0f0 - dg) * db
    w101 = dr * (1.0f0 - dg) * db
    w011 = (1.0f0 - dr) * dg * db
    w111 = dr * dg * db

    @inline fetch(ch) =
        w000 * lut[r0, g0, b0, ch] + w100 * lut[r1, g0, b0, ch] +
        w010 * lut[r0, g1, b0, ch] + w110 * lut[r1, g1, b0, ch] +
        w001 * lut[r0, g0, b1, ch] + w101 * lut[r1, g0, b1, ch] +
        w011 * lut[r0, g1, b1, ch] + w111 * lut[r1, g1, b1, ch]

    out[I] = topixel(eltype(out), fetch(Int32(1)), fetch(Int32(2)), fetch(Int32(3)))
end

function lut3d!(out::AbstractMatrix{<:AbstractRGB}, img::AbstractMatrix{<:AbstractRGB},
                lut::AbstractArray{Float32,4}; binsize::Union{Nothing,Real} = nothing)
    size(out) == size(img) ||
        throw(DimensionMismatch("out $(size(out)) does not match img $(size(img))"))
    d = size(lut, 1)
    size(lut) == (d, d, d, 3) ||
        throw(ArgumentError("lut must be (D, D, D, 3), got $(size(lut))"))
    d > 1 || throw(ArgumentError("lut dimension must be > 1, got $d"))

    bs = Float32(binsize === nothing ? 1.000001 / (d - 1) : binsize)
    backend = KA.get_backend(img)
    lut3d_kernel!(backend)(out, img, lut, bs, Int32(d); ndrange = size(img))
    return out
end

lut3d!(img::AbstractMatrix{<:AbstractRGB}, lut::AbstractArray{Float32,4}; kwargs...) =
    lut3d!(img, img, lut; kwargs...)
