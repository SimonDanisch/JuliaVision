"""
    occupiedcoords(logits, resolution) -> Matrix{Int32}

The tail of `sample_sparse_structure`: the decoder's `(64, 64, 64, 1, 1)` logits
thresholded at 0, max-pooled down to `resolution` if that is coarser, then
`argwhere`. Returns `(4, N)`, torch's `(N, 4)` rows `(batch, x, y, z)` as
columns, in `argwhere`'s order.

The order is free: torch's `(b, c, x, y, z)` is Julia's `(z, y, x, c, b)`, so
row-major `argwhere` and column-major `findall` both run `z` fastest.
"""
function occupiedcoords(logits::AbstractArray{<:Real,5}, resolution::Integer)
    occ = view(logits, :, :, :, 1, 1) .> 0
    r = size(occ, 1)
    ratio = r ÷ resolution
    ratio * resolution == r || throw(ArgumentError(
        "a $(r)^3 occupancy does not pool to $(resolution)^3"))
    if ratio > 1
        # `max_pool3d(occ.float(), ratio, ratio, 0) > 0.5`: a cell is occupied
        # when any of its ratio^3 children is.
        blk(i) = (i - 1) * ratio + 1:i * ratio
        occ = [any(view(occ, blk(i), blk(j), blk(k)))
               for i in 1:resolution, j in 1:resolution, k in 1:resolution]
    end
    idx = findall(occ)
    coords = Matrix{Int32}(undef, 4, length(idx))
    for (n, I) in enumerate(idx)
        coords[:, n] .= (0, I[3] - 1, I[2] - 1, I[1] - 1)
    end
    return coords
end

"""
    ropetable(coords; headdim = 128, base = 10000f0) -> Array{Float32,3}

The SLat flow graphs' `rope` input for voxels `coords` (`(4, N)`, as
[`occupiedcoords`](@ref) returns): `(64, N, 2)`, the cos and sin of torch's
`(2, N, 64)` table (`tools/export_trellis2.py`, `real_rotary`).

`SparseRotaryPositionEmbedder`: 21 frequencies `1 / base^(i / 21)` per spatial
axis, the phase `coord * freq` laid out x, y, z, then one zero phase to pad 63 to
64. Float32 throughout, as torch computes it.
"""
function ropetable(coords::AbstractMatrix{<:Integer}; headdim::Integer = 128,
                   base::Float32 = 10000f0)
    P = headdim ÷ 2
    naxes = size(coords, 1) - 1
    fdim = P ÷ naxes
    freqs = [1f0 / base^(Float32(i) / Float32(fdim)) for i in 0:(fdim - 1)]
    n = size(coords, 2)
    table = zeros(Float32, P, n, 2)
    table[(naxes * fdim + 1):P, :, 1] .= 1
    for k in 1:n, a in 1:naxes
        c = Float32(coords[a + 1, k])
        for j in 1:fdim
            s, co = sincos(c * freqs[j])
            table[(a - 1) * fdim + j, k, 1] = co
            table[(a - 1) * fdim + j, k, 2] = s
        end
    end
    return table
end
