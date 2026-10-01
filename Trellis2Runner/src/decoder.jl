"""
    SparseDecoder(dev, weights; out, subdiv)

One of TRELLIS.2's two sparse structured-latent decoders, `SparseUnetVaeDecoder`
with `SparseConvNeXtBlock3d` levels and `SparseResBlockC2S3d` upsamplers: the
shape decoder (`out = 7`, predicts where to subdivide) or the texture decoder
(`out = 6`, `subdiv = false`, guided by the shape decoder's subdivisions).

`weights` is the checkpoint's tensors as `readsafetensors` gives them, or the
path of its `.safetensors`. Every tensor goes to the device once,
in the layout the kernels read: a linear weight as `(out, in)`, a sparse
convolution's `(co, kx, ky, kz, ci)` as the `(co, 27 ci)` matrix the implicit
im2col multiplies, its reduction index `c + ci (kz + 3 ky + 9 kx)`.

Features are `(C, N)` throughout, torch's `(N, C)` rows as columns, in fp16 in
the torso (`use_fp16`) and fp32 at the two ends, as upstream.
"""
struct SparseDecoder
    dev::Mantle.Device
    w::Dict{String,Mantle.Buffer}
    channels::Vector{Int}
    nblocks::Vector{Int}
    out::Int
    subdiv::Bool
end

SparseDecoder(dev::Mantle.Device, path::AbstractString; kw...) = SparseDecoder(dev, readsafetensors(path); kw...)

function SparseDecoder(dev::Mantle.Device, raw::AbstractDict; out::Int, subdiv::Bool,
                       channels = [1024, 512, 256, 128, 64], nblocks = [4, 16, 8, 4, 0])
    w = Dict{String,Mantle.Buffer}()
    for (k, a) in raw
        w[k] = Mantle.Buffer(dev, devicelayout(a))
    end
    # A GEMM operand's rows come in multiples of 16: the subdivision head's eight
    # outputs are padded with zero rows, which the epilogue never writes out.
    subdiv && for i in 0:(length(channels) - 2)
        key = "blocks.$i.$(nblocks[i + 1]).to_subdiv.weight"
        wt = Array(Mantle.storage(w[key]))
        padded = zeros(Float16, 16, size(wt, 2))
        padded[1:size(wt, 1), :] .= wt
        w[key] = Mantle.Buffer(dev, padded)
    end
    return SparseDecoder(dev, w, collect(channels), collect(nblocks), out, subdiv)
end

"""The device layout of one checkpoint tensor (see [`SparseDecoder`](@ref))."""
devicelayout(a::AbstractArray{T,5}) where {T} =
    permutedims(reshape(a, :, size(a, 5)))        # (ci, kz, ky, kx, co) -> (co, 27 ci)
devicelayout(a::AbstractMatrix) = permutedims(a)  # torch (out, in) read as (in, out)
devicelayout(a::AbstractVector) = collect(a)

"""
    Level(coords, res)

The voxels of one decoder level: `coords` `(3, N)` zero-based `(x, y, z)` in a
`res`^3 grid, in the order upstream's `SparseChannel2Spatial` creates them.
"""
struct Level
    coords::Matrix{Int32}
    res::Int
end

Base.length(l::Level) = size(l.coords, 2)

"""
    children(level, mask) -> (Level, parent, subidx)

`SparseChannel2Spatial`: for every voxel in order, the children whose `mask` bit
is set (`(8, N)`), in child order, child `s` at `2 x + (s & 1, s >> 1 & 1,
s >> 2 & 1)`. `parent` and `subidx` (zero-based) say where each child's features
come from.
"""
function children(l::Level, mask::AbstractMatrix{Bool})
    n = count(mask)
    coords = Matrix{Int32}(undef, 3, n)
    parent = Vector{Int32}(undef, n)
    subidx = Vector{Int32}(undef, n)
    k = 0
    for j in 1:length(l), s in 0:7
        mask[s + 1, j] || continue
        k += 1
        coords[1, k] = 2 * l.coords[1, j] + (s & 1)
        coords[2, k] = 2 * l.coords[2, j] + (s >> 1 & 1)
        coords[3, k] = 2 * l.coords[3, j] + (s >> 2 & 1)
        parent[k] = j
        subidx[k] = s
    end
    return Level(coords, 2 * l.res), parent, subidx
end

"""Columns a per-voxel buffer is allocated with: a GEMM extent is a multiple of 64."""
padded(n::Int) = cld(n, 64) * 64

"""
    DecodeResult

What [`decode`](@ref) produces: the head's output `(out, N)` in Float32 on the last
level, every level's voxels, and each upsampler's subdivision mask (what the
texture decoder is guided by).
"""
struct DecodeResult
    out::Matrix{Float32}
    levels::Vector{Level}
    masks::Vector{Matrix{Bool}}
end

"""
    decode(dec, slat, coords; res = 32, guide = nothing, probe = nothing) -> DecodeResult

Run the decoder on a structured latent `slat` `(32, N)` Float32 at voxels
`coords` `(3, N)` of the `res`^3 grid: 32 for the 512 pipeline, 64 for the
1024 cascade (upstream's `set_resolution`, the output grid being 16 times the
input's). `guide` is the shape decoder's result, whose masks a decoder without
its own subdivision head follows.

One graph per level: what crosses a level is decided on the device (which voxels
subdivide) and sizes the next one, so the host reads the mask between them and
builds the next level's voxels. `probe(name, array)` sees each level's input,
the upsampler's input and its logits, named as the dump does.
"""
function decode(dec::SparseDecoder, slat::AbstractMatrix{Float32}, coords::AbstractMatrix{<:Integer};
                res::Int = 32, guide::Union{Nothing,DecodeResult} = nothing, probe = nothing)
    levels, masks, out = decodelevels(dec, slat, coords, length(dec.channels); res, guide, probe)
    return DecodeResult(out, levels, masks)
end

"""
    upsample(dec, slat, coords; res = 32, times = 4) -> Level

The shape decoder's `upsample`: its first `times` levels, subdivisions
included, and the voxels the last of them creates, without the rest of the
decoder. What the 1024 cascade places its high-resolution latent on.
"""
upsample(dec::SparseDecoder, slat::AbstractMatrix{Float32}, coords::AbstractMatrix{<:Integer};
         res::Int = 32, times::Int = 4) = decodelevels(dec, slat, coords, times; res)[1][end]

"""
    cascadecoords(level; resolution = 1024) -> Matrix{Int32}

`sample_shape_slat_cascade`'s tokens: the voxels [`upsample`](@ref) creates
(`level`, on the low-resolution output grid) moved onto the `resolution ÷ 16`
grid the high-resolution latent lives on, `int((c + 0.5) / lr * (resolution ÷
16))` in Float32 as torch computes it, each cell once. `(4, N)` with batch 0,
in `unique(dim = 0)`'s lexicographic order.
"""
function cascadecoords(level::Level; resolution::Int = 1024)
    S = resolution ÷ 16
    q(c) = unsafe_trunc(Int64, (Float32(c) + 0.5f0) / Float32(level.res) * Float32(S))
    keys = unique!(sort!([(q(level.coords[1, j]) * S + q(level.coords[2, j])) * S + q(level.coords[3, j])
                          for j in 1:length(level)]))
    coords = Matrix{Int32}(undef, 4, length(keys))
    for (n, k) in enumerate(keys)
        coords[:, n] .= (0, k ÷ (S * S), k ÷ S % S, k % S)
    end
    return coords
end

"""
The decoder's first `nlevels` levels: every level's voxels, every subdivision
mask, and the head's output when the last level ran (`nothing` otherwise).

A level's graph opens with the upsampler that creates it, both halves (see
[`upsample!`](@ref)), so what crosses from one graph to the next is only the
parent level's voxels and features and where each child comes from.
"""
function decodelevels(dec::SparseDecoder, slat::AbstractMatrix{Float32}, coords::AbstractMatrix{<:Integer},
                      stop::Int; res::Int, guide::Union{Nothing,DecodeResult} = nothing, probe = nothing)
    dec.subdiv || guide !== nothing || throw(ArgumentError(
        "decode: this decoder predicts no subdivision; pass the shape decoder's result as `guide`"))
    dev = dec.dev
    level = Level(Int32.(coords), res)
    levels = [level]
    masks = Matrix{Bool}[]
    nlevels = length(dec.channels)
    up = nothing
    out = nothing
    for l in 0:(stop - 1)
        g = Mantle.Graph(dev)
        n = length(level)
        C = dec.channels[l + 1]
        pos = Mantle.Buffer(dev, level.coords)
        grid = VoxelGrid(g, dev, level, pos)
        x = fill!(Mantle.Buffer(dev, Float16, (C, padded(n))), 0)
        lin = nothing
        if l == 0
            lin = Mantle.Buffer(dev, Float32, (C, n))
            Mantle.dispatch!(g, linear32!, (lin, Mantle.Buffer(dev, Matrix{Float32}(slat)),
                                            dec.w["from_latent.weight"], dec.w["from_latent.bias"],
                                            C, size(slat, 1), n), C * n; name = "from_latent")
            Mantle.dispatch!(g, tofp16!, (x, lin, C * n), C * n; name = "from_latent/fp16")
        else
            upsample!(g, dec, "blocks.$(l - 1).$(dec.nblocks[l])", x, up, pos, grid, level.res, n)
        end
        for b in 0:(dec.nblocks[l + 1] - 1)
            convnext!(g, dec, "blocks.$l.$b", x, pos, grid, level.res, n)
        end
        if l < nlevels - 1
            sub = dec.subdiv ? Mantle.Buffer(dev, Float16, (8, n)) : nothing
            sub === nothing || subdivision!(g, dec, "blocks.$l.$(dec.nblocks[l + 1])", sub, x, n)
            runonce!(g)
            if probe !== nothing
                # `L0.in` is from_latent's Float32 output, before the torso's fp16.
                l == 0 && probe("L0.in", Array(Mantle.storage(lin)))
                probe("L$l.pre", Array(Mantle.storage(x))[:, 1:n])
                sub === nothing || probe("L$l.sub", Array(Mantle.storage(sub)))
            end
            mask = dec.subdiv ? Array(Mantle.storage(sub)) .> 0 : guide.masks[l + 1]
            push!(masks, mask)
            child, parent, subidx = children(level, mask)
            up = Upsampling(level, pos, x, Mantle.Buffer(dev, parent), Mantle.Buffer(dev, subidx),
                            [1; 1 .+ cumsum(vec(count(mask; dims = 1)))])
            level = child
            push!(levels, level)
        else
            head = Mantle.Buffer(dev, Float32, (dec.out, n))
            head!(g, dec, head, x, n)
            runonce!(g)
            probe === nothing || probe("L$l.in", Array(Mantle.storage(x))[:, 1:n])
            out = Array(Mantle.storage(head))
        end
    end
    return levels, masks, out
end

# ── the blocks ────────────────────────────────────────────────────────────────

"""
`SparseConvNeXtBlock3d`: `h = conv(x)`, `LayerNorm32` (affine), `Linear`, SiLU,
`Linear`, `h + x`, in place on `x`.

The norm and the MLP run in column chunks: the hidden layer is four times the
channels, over every voxel 1.7 GB at the 1024 cascade's third level, and each
voxel's column needs only its own. A chunk's hidden layer is bounded like a GEMM
chunk, and so is its normalised input.
"""
function convnext!(g, dec::SparseDecoder, p, x, pos, grid, res, n)
    C, np = size(x)
    h = Mantle.Transient.Buffer(g, Float16, C, np)
    sparseconv!(g, h, x, dec.w["$p.conv.weight"], dec.w["$p.conv.bias"], pos, grid, res, n;
                name = "$p.conv")
    cols = chunkcols(C, 4C, n)
    hn = Mantle.Transient.Buffer(g, Float16, C, cols)
    m = Mantle.Transient.Buffer(g, Float16, 4C, cols)
    for c0 in 0:cols:(n - 1)
        w = min(cols, n - c0)
        wp = padded(w)
        hv = Mantle.viewof(h, (C, wp); offset = C * c0)
        hnv = Mantle.viewof(hn, (C, wp))
        xv = Mantle.viewof(x, (C, wp); offset = C * c0)
        mv = Mantle.viewof(m, (4C, wp))
        Mantle.dispatch!(g, layernorm!, (hnv, hv, dec.w["$p.norm.weight"], dec.w["$p.norm.bias"],
                                         Val(C), w, 1f-6, identity), w; name = "$p.norm")
        linear!(g, mv, dec.w["$p.mlp.0.weight"], hnv, dec.w["$p.mlp.0.bias"], w, silu, nothing;
                name = "$p.mlp.0")
        linear!(g, xv, dec.w["$p.mlp.2.weight"], mv, dec.w["$p.mlp.2.bias"], w, identity, xv;
                name = "$p.mlp.2")
    end
    return x
end

"""
The subdivision logits of `SparseResBlockC2S3d` `p`, on the parent level: the
first eight rows of `to_subdiv(x)`, which the host reads to build the children.
"""
function subdivision!(g, dec::SparseDecoder, p, sub, x, n)
    logits = Mantle.Transient.Buffer(g, Float16, 16, size(x, 2))
    linear!(g, logits, dec.w["$p.to_subdiv.weight"], x, dec.w["$p.to_subdiv.bias"], n,
            identity, nothing; name = "$p.to_subdiv", rows = 8)
    Mantle.dispatch!(g, croprows!, (sub, logits, 8, 16, n), 8n; name = "$p.to_subdiv/crop")
    return sub
end

"""
    Upsampling(level, pos, x, parent, subidx, firstchild)

What a level hands the upsampler that creates the next: its voxels (`pos` on the
device), its features `x`, for each child the parent column and the zero-based
child index ([`children`](@ref)), and on the host where each parent's children
start (`firstchild[p]:firstchild[p + 1] - 1`, children being in parent order).
"""
struct Upsampling
    level::Level
    pos::Mantle.Buffer
    x::Mantle.Buffer
    parent::Mantle.Buffer
    subidx::Mantle.Buffer
    firstchild::Vector{Int}
end

"""
`SparseResBlockC2S3d` `p` into the child level's `x`: `conv1(silu(norm1(xp)))`
on the parent's voxels, whose `8 co` channels become the children's features,
`silu(norm2(.))` of those after channel-to-space, `conv2`, and the skip (the
parent's channels, each repeated) added.

Both halves in the child level's graph, and neither channel-to-space is ever
materialised: `conv1`'s output (eight times the parent's voxels, 1.8 GB going
into the 1024 cascade's last level) is consumed chunk by chunk as the GEMM
produces it, each chunk's children normalised from it before the next chunk
overwrites it, and `conv2`'s epilogue reads the skip from the parent's features.
"""
function upsample!(g, dec::SparseDecoder, p, x, up::Upsampling, pos, grid, res, n)
    xp = up.x
    Cp, npp = size(xp)
    co = size(x, 1)
    m = length(up.level)
    hn = Mantle.Transient.Buffer(g, Float16, Cp, npp)
    Mantle.dispatch!(g, layernorm!, (hn, xp, dec.w["$p.norm1.weight"], dec.w["$p.norm1.bias"],
                                     Val(Cp), m, 1f-6, silu), m; name = "$p.norm1")
    hn2 = Mantle.Transient.Buffer(g, Float16, size(x)...)
    fc = up.firstchild
    norm2 = Chunked() do hc, p0, valid
        # The children of parents `p0 .+ (1:valid)`, whose features are `hc`'s
        # first `valid` columns.
        j0 = fc[p0 + 1]
        k = fc[p0 + valid + 1] - j0
        k > 0 && Mantle.dispatch!(g, spatialnorm!, (hn2, hc, up.parent, up.subidx, Val(co), j0 - 1, k, p0,
                                                    1f-6, silu), k; name = "$p.c2s/norm2")
    end
    sparseconv!(g, norm2, hn, dec.w["$p.conv1.weight"], dec.w["$p.conv1.bias"], up.pos,
                VoxelGrid(g, dec.dev, up.level, up.pos), up.level.res, m; name = "$p.conv1")
    sparseconv!(g, x, hn2, dec.w["$p.conv2.weight"], dec.w["$p.conv2.bias"], pos, grid, res, n;
                name = "$p.conv2", residual = (xp, up.parent, up.subidx, Val(co), Val(Cp)))
    return x
end

"""
The head: back to Float32, `layer_norm` without affine (eps 1e-5, torch's default
for `F.layer_norm`), `output_layer`, in one kernel ([`normlinear!`](@ref)).
"""
function head!(g, dec::SparseDecoder, out, x, n)
    Mantle.dispatch!(g, normlinear!, (out, x, dec.w["output_layer.weight"], dec.w["output_layer.bias"],
                                      Val(size(x, 1)), dec.out, n, 1f-5), n; name = "head")
    return out
end

# ── sparse convolution and linear layers as GEMMs ─────────────────────────────

"""
Bytes one GEMM chunk's operands may take: the im2col matrix or the output
planes. Columns are processed in chunks of at most this, through the same two
transients.
"""
chunkcols(K::Int, M::Int, n::Int; budget = 256 << 20) =
    min(padded(n), max(64, (budget ÷ (4 * max(K, M))) ÷ 64 * 64))

"""
    VoxelGrid(g, dev, level, pos)

The index of a level's voxels (`pos`, its coords on the device) by position,
what the im2col reads a neighbour through and the bake samples the attribute
volume through ([`voxelat`](@ref)): the `res`^3 grid in bricks of 8^3 cells,
`bricks` the dense table of which brick holds voxels (one-based, zero for
none), `cells` 512 entries per such brick, each the voxel's one-based column or
zero. A dense index over the grid is four bytes a cell, 4 GiB at 1024^3, which
no buffer takes; the voxels lie on a surface, so few bricks hold any.

The brick table is built on the host, which has the level's voxels anyway.
"""
struct VoxelGrid{B,C}
    bricks::B
    cells::C
    res::Int
end

function VoxelGrid(g::Mantle.Graph, dev::Mantle.Device, level::Level, pos)
    B = level.res ÷ 8
    table = zeros(Int32, B^3)
    nb = Int32(0)
    for j in 1:length(level)
        k = (level.coords[1, j] >> 3) + B * ((level.coords[2, j] >> 3) + B * (level.coords[3, j] >> 3)) + 1
        table[k] == 0 && (table[k] = nb += Int32(1))
    end
    bricks = Mantle.Buffer(dev, table)
    ncells = 512 * max(Int(nb), 1)
    cells = Mantle.Transient.Buffer(g, Int32, ncells)
    Mantle.dispatch!(g, zerofill!, (cells, ncells), ncells; name = "grid/clear")
    n = length(level)
    Mantle.dispatch!(g, scattergrid!, (bricks, cells, pos, level.res, n), n; name = "grid/scatter")
    return VoxelGrid(bricks, cells, level.res)
end

"""The one-based column of the voxel at `(x, y, z)` (in the grid), zero for none."""
@inline function voxelat(bricks, cells, res, x, y, z)
    B = res >> 3
    @inbounds b = bricks[(x >> 3) + B * ((y >> 3) + B * (z >> 3)) + 1]
    b == 0 && return Int32(0)
    @inbounds return cells[(b - 1) * 512 + (x & 7) + 8 * ((y & 7) + 8 * (z & 7)) + 1]
end

"""
`out = conv(x) + bias` over a submanifold 3^3 neighbourhood (flex_gemm's
`sparse_submanifold_conv3d`), as chunks of: gather the 27 neighbours' features
into an im2col matrix, multiply by the `(co, 27 ci)` weight, add the bias (and
`residual`) and round to fp16.
"""
function sparseconv!(g, out, x, w, b, pos, grid, res, n; name, residual = nothing)
    ci = size(x, 1)
    co = size(w, 1)
    K = 27 * ci
    cols = chunkcols(K, co, n)
    col = Mantle.Transient.Buffer(g, Float16, K, cols)
    gemmchunks!(g, out, w, K, co, n, cols, b, identity, residual; name) do n0, w_
        Mantle.dispatch!(g, im2col3!, (col, x, pos, grid.bricks, grid.cells, res, Val(ci), n0, w_), K * w_;
                         name = "$name/im2col")
        col
    end
    return out
end

"""`out = f(W x + bias) (+ residual)` over the first `n` columns of `x`, in fp16."""
function linear!(g, out, w, x, b, n, f, residual; name, rows = size(out, 1))
    K = size(x, 1)
    M = size(w, 1)
    cols = chunkcols(K, M, n)
    gemmchunks!(g, out, w, K, M, n, cols, b, f, residual; name, rows) do n0, w_
        Mantle.viewof(x, (K, w_); offset = K * n0)
    end
    return out
end

"""
The chunked GEMM both of the above are: `operand(n0, width)` declares the `(K,
width)` right-hand side for columns `n0 .+ (1:width)`, and the epilogue writes
`rows` of the `M` result rows into `out`, or into a [`Chunked`](@ref) output's
chunk.
"""
function gemmchunks!(operand, g, out, w, K, M, n, cols, b, f, residual; name, rows = M)
    # Two chunk widths at most, the full one and the last; the planes are sized
    # for whichever of the two splits the reduction further.
    last = padded(n - (cld(n, cols) - 1) * cols)
    splitk = maximum(wd -> max(Mantle.coopmat_gemm_shape(M, wd, K)[2], 1), (min(cols, padded(n)), last))
    C = Mantle.Transient.Buffer(g, Float32, M, cols, splitk)
    dst = destination(g, out, rows, cols)
    for n0 in 0:cols:(n - 1)
        w_ = min(cols, padded(n - n0))
        B = operand(n0, w_)
        Cw = Mantle.viewof(C, (M, w_, splitk))
        bs = Mantle.coopmat_gemm_shape(M, w_, K)
        Mantle.coopmat_gemm_dispatch!(g, Cw, w, B, M, w_, K; blk_split = bs, partials = Cw,
                                      reduce = false, name = "$name/gemm")
        valid = min(w_, n - n0)
        Mantle.dispatch!(g, epilogue!, (dst, Cw, b, residual, f, Val(max(bs[2], 1)), rows, M,
                                        w_, n0, column(out, n0), valid), rows * valid;
                         name = "$name/epilogue")
        produced!(out, dst, n0, valid)
    end
    return out
end

"""
    Chunked(consume)

A GEMM output that never exists whole. Each column chunk's result goes to one
chunk-sized fp16 transient, and `consume(chunk, n0, valid)` declares what reads
its first `valid` columns (the result's columns `n0 .+ (1:valid)`) before the
next chunk overwrites them.
"""
struct Chunked{F}
    consume::F
end

"""What `gemmchunks!`'s epilogue writes into: the output, or a chunked output's
one `rows × cols` chunk."""
destination(g, out, rows, cols) = out
destination(g, ::Chunked, rows, cols) = Mantle.Transient.Buffer(g, Float16, rows, cols)

"""The destination column a chunk starting at result column `n0` is written from
(zero-based): `n0` itself, or the chunk's own first column."""
column(out, n0) = n0
column(::Chunked, n0) = 0

"""After a chunk's epilogue: nothing, or a chunked output's `consume`."""
produced!(out, dst, n0, valid) = nothing
produced!(out::Chunked, dst, n0, valid) = (out.consume(dst, n0, valid); nothing)

# ── kernels ──────────────────────────────────────────────────────────────────

silu(x) = x / (1f0 + exp(-x))

function zerofill!(a, n)
    i = KI.get_global_id().x
    i <= n && (@inbounds a[i] = zero(eltype(a)))
    return nothing
end

function scattergrid!(bricks, cells, pos, res, n)
    j = KI.get_global_id().x
    j <= n || return nothing
    @inbounds x, y, z = pos[1, j], pos[2, j], pos[3, j]
    B = res >> 3
    @inbounds b = bricks[(x >> 3) + B * ((y >> 3) + B * (z >> 3)) + 1]
    @inbounds cells[(b - 1) * 512 + (x & 7) + 8 * ((y & 7) + 8 * (z & 7)) + 1] = Int32(j)
    return nothing
end

"""
One element of the im2col matrix: row `c + ci k` of column `j` is channel `c` of
the neighbour at offset `(kx, ky, kz) - 1`, `k = kz + 3 ky + 9 kx`, or zero.
"""
function im2col3!(col, x, pos, bricks, cells, res, ::Val{CI}, n0, width) where {CI}
    t = KI.get_global_id().x
    K = 27 * CI
    t <= K * width || return nothing
    r = (t - 1) % K
    j = (t - 1) ÷ K + 1
    c = r % CI + 1
    k = r ÷ CI
    kz, ky, kx = k % 3, k ÷ 3 % 3, k ÷ 9
    v = zero(eltype(col))
    src = n0 + j
    if src <= size(pos, 2)
        @inbounds xx = pos[1, src] + kx - 1
        @inbounds yy = pos[2, src] + ky - 1
        @inbounds zz = pos[3, src] + kz - 1
        if 0 <= xx < res && 0 <= yy < res && 0 <= zz < res
            nb = voxelat(bricks, cells, res, xx, yy, zz)
            nb > 0 && (@inbounds v = x[c, nb])
        end
    end
    @inbounds col[t] = v
    return nothing
end

"""
`out[m, o + j] = f(Σ_s C[m, j, s] + b[m])` for result column `n0 + j`, rounded
to `out`'s type as torch rounds each op, then `+ residual` (see
[`residualat`](@ref)) and rounded again.
"""
function epilogue!(out, C, b, residual, f::F, ::Val{S}, rows, M, width, n0, o, valid) where {F,S}
    t = KI.get_global_id().x
    t <= rows * valid || return nothing
    m = (t - 1) % rows + 1
    j = (t - 1) ÷ rows + 1
    acc = 0f0
    for s in 1:S
        @inbounds acc += C[m + M * ((j - 1) + width * (s - 1))]
    end
    @inbounds acc += Float32(b[m])
    T = eltype(out)
    v = T(f(Float32(T(acc))))
    if residual !== nothing
        v = T(Float32(v) + Float32(residualat(residual, m, n0 + j)))
    end
    @inbounds out[m, o + j] = v
    return nothing
end

"""
Row `m` of a residual's column `j`: a matrix's element, or, for the skip of
`SparseChannel2Spatial` given as `(xprev, parent, subidx, Val(co), Val(cp))`,
the parent's channel `s cp/8 + (m - 1) ÷ r`, `r = co ÷ (cp/8)`
(`repeat_interleave`), of child `j`'s parent at sub-index `s`.
"""
@inline residualat(r, m, j) = @inbounds r[m, j]
@inline function residualat(r::Tuple{Any,Any,Any,Val{CO},Val{CP}}, m, j) where {CO,CP}
    xprev, parent, subidx = r
    @inbounds p = parent[j]
    @inbounds s = subidx[j]
    q = CP ÷ 8
    return @inbounds xprev[s * q + (m - 1) ÷ (CO ÷ q) + 1, p]
end

"""
The mean of a column's `C` channels and the reciprocal of its standard deviation,
in Float32, `at(c)` reading channel `c`.
"""
@inline function moments(at::A, ::Val{C}, eps::Float32) where {A,C}
    s = 0f0
    for c in 1:C
        s += Float32(at(c))
    end
    mu = s / C
    v = 0f0
    for c in 1:C
        d = Float32(at(c)) - mu
        v += d * d
    end
    return mu, 1f0 / sqrt(v / C + eps)
end

"""Column `j` of `y`: channel `c` read by `at(c)`, normalised, the affine applied
when there is one, rounded to `y`'s type, then `f`, rounded again."""
@inline function normcolumn!(y, j, at::A, w, b, ::Val{C}, eps::Float32, f::F) where {A,C,F}
    mu, r = moments(at, Val(C), eps)
    T = eltype(y)
    for c in 1:C
        t = (Float32(at(c)) - mu) * r
        if w !== nothing
            @inbounds t = t * Float32(w[c]) + Float32(b[c])
        end
        @inbounds y[c, j] = T(f(Float32(T(t))))
    end
    return nothing
end

"""
`LayerNorm` over each voxel's channels in Float32 (`LayerNorm32`), with or
without an affine, rounded to `y`'s type, then `f` (SiLU) rounded again.
"""
function layernorm!(y, x, w, b, ::Val{C}, n, eps::Float32, f::F) where {C,F}
    j = KI.get_global_id().x
    j <= n || return nothing
    normcolumn!(y, j, c -> @inbounds(x[c, j]), w, b, Val(C), eps, f)
    return nothing
end

"""
`SparseChannel2Spatial` of `conv1`'s output into `norm2` (no affine), then `f`,
for the `k` children after child `j0`, whose parents are in the chunk `hc` of
`conv1`'s output that starts at parent column `p0 + 1`: child `j` of parent `p`
at sub-index `s` normalises channels `s co .+ (1:co)` of `hc[:, p - p0]`.
"""
function spatialnorm!(y, hc, parent, subidx, ::Val{CO}, j0, k, p0, eps::Float32, f::F) where {CO,F}
    t = KI.get_global_id().x
    t <= k || return nothing
    j = j0 + t
    @inbounds p = parent[j] - p0
    @inbounds s = subidx[j]
    normcolumn!(y, j, c -> @inbounds(hc[s * CO + c, p]), nothing, nothing, Val(CO), eps, f)
    return nothing
end

"""
`layer_norm` without affine then a Float32 linear layer, one voxel per item:
`out[m, j] = Σ_k w[m, k] t_k + b[m]` over the normalised column `t`, which is
recomputed per output rather than stored. Stored it is `C × N` Float32, 1.8 GB
at the cascade's last level, for six outputs.
"""
function normlinear!(out, x, w, b, ::Val{C}, M, n, eps::Float32) where {C}
    j = KI.get_global_id().x
    j <= n || return nothing
    mu, r = moments(c -> @inbounds(x[c, j]), Val(C), eps)
    for m in 1:M
        acc = 0f0
        for k in 1:C
            @inbounds acc += Float32(w[m, k]) * ((Float32(x[k, j]) - mu) * r)
        end
        @inbounds out[m, j] = acc + Float32(b[m])
    end
    return nothing
end

"""A Float32 linear layer, one output element per item: the two small ends."""
function linear32!(out, x, w, b, M, K, n)
    t = KI.get_global_id().x
    t <= M * n || return nothing
    m = (t - 1) % M + 1
    j = (t - 1) ÷ M + 1
    acc = 0f0
    for k in 1:K
        @inbounds acc += Float32(w[m, k]) * Float32(x[k, j])
    end
    @inbounds out[m, j] = acc + Float32(b[m])
    return nothing
end

function tofp16!(y, x, n)
    i = KI.get_global_id().x
    i <= n || return nothing
    @inbounds y[i] = Float16(x[i])
    return nothing
end

"""The first `rows` of every column of `src` into `dst`."""
function croprows!(dst, src, rows, srcrows, n)
    t = KI.get_global_id().x
    t <= rows * n || return nothing
    m = (t - 1) % rows + 1
    j = (t - 1) ÷ rows + 1
    @inbounds dst[m, j] = src[m, j]
    return nothing
end
