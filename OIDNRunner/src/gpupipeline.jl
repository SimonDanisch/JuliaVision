# Upstream's device-side pipeline (core/unet_filter.cpp): radiance and guides
# are uploaded once, the auxiliary networks write their filtered guides on the
# device, the HDR network reads them there, and only the result comes back.

"""Full-resolution images on the device, Float32, three channels each."""
struct DeviceImages
    w::Int
    h::Int
    color::Any
    albedo::Any
    normal::Any
    galbedo::Any     # prefiltered guides, planar (w, h, 3)
    gnormal::Any
    output::Any      # result, planar (w, h, 3)
end
function DeviceImages(backend, w, h)
    a() = KA.allocate(backend, Float32, 3 * w * h)
    return DeviceImages(w, h, a(), a(), a(), a(), a(), a())
end
free!(x::DeviceImages) = foreach(KA.unsafe_free!, (x.color, x.albedo, x.normal, x.galbedo, x.gnormal, x.output))
function free!(t::TileBuffers)
    foreach(KA.unsafe_free!, (t.input, t.output))
    foreach(KA.unsafe_free!, t.a); foreach(KA.unsafe_free!, t.b)
    foreach(KA.unsafe_free!, t.pool[2:end])
    return nothing
end
free!(n::UNet) = foreach(L -> (KA.unsafe_free!(L.Wt); KA.unsafe_free!(L.bias)), values(n.layers))
free!(::Array) = nothing

function deviceimages!(d, w, h)
    img = get(d.workspace, :gpuimages, nothing)
    img isa DeviceImages && img.w == w && img.h == h && return img
    img === nothing || free!(img)
    return d.workspace[:gpuimages] = DeviceImages(d.backend, w, h)
end
function tilebuffers!(d, tw, th)
    t = get(d.workspace, :gputiles, nothing)
    t isa TileBuffers && t.tw == tw && t.th == th && return t
    t === nothing || free!(t)
    return d.workspace[:gputiles] = TileBuffers(d.backend, tw, th)
end
unet!(d, kind) = get!(d.unets, kind) do
    weights = DNNKernels.readsafetensors(joinpath(d.directory, MODEL_NAMES[kind] * ".safetensors"))
    UNet(weights, kind === :hdr ? 9 : 3, d.backend)
end

# Upload without a host copy: RGB/Vec3 matrices are interleaved Float32 already,
# planar arrays are their own layout.
function upload!(dst, a::Matrix{T}) where {T}
    isbitstype(T) && sizeof(T) == 12 || throw(ArgumentError("expected 3 x Float32 pixels, got $T"))
    GC.@preserve a copyto!(dst, unsafe_wrap(Array, Ptr{Float32}(pointer(a)), 3 * length(a)))
    return interleaved(size(a, 1))
end
upload!(dst, a::AbstractMatrix) = upload!(dst, collect(a))
function upload!(dst, a::Array{Float32, 3})
    copyto!(dst, vec(a))
    return planarstrides(size(a, 1), size(a, 2))
end
upload!(dst, a::AbstractArray{<:Real, 3}) = upload!(dst, Array{Float32}(a))

const TILE_OVERLAP = 112   # the large network's receptive field is 202
struct Tile
    x0::Int32; y0::Int32; rw::Int32; rh::Int32; ax::Int32; ay::Int32
    left::Int32; top::Int32; nx::Int32; ny::Int32
end
function tileplan(d, w, h)
    tw = min(d.tilesize, 16 * cld(w, 16)); th = min(d.tilesize, 16 * cld(h, 16))
    stridew = tw - 2 * TILE_OVERLAP; strideh = th - 2 * TILE_OVERLAP
    nx = tw >= w ? 1 : cld(w - 2 * TILE_OVERLAP, stridew)
    ny = th >= h ? 1 : cld(h - 2 * TILE_OVERLAP, strideh)
    tiles = Tile[]
    for iy in 0:(ny - 1), ix in 0:(nx - 1)
        x0 = ix * stridew; y0 = iy * strideh
        rw = min(w - x0, tw); rh = min(h - y0, th)
        left = ix > 0 ? TILE_OVERLAP : 0; right = ix < nx - 1 ? TILE_OVERLAP : 0
        top = iy > 0 ? TILE_OVERLAP : 0; bottom = iy < ny - 1 ? TILE_OVERLAP : 0
        push!(tiles, Tile(x0, y0, rw, rh, tw - 16 * cld(rw, 16), th - 16 * cld(rh, 16),
                          left, top, rw - left - right, rh - top - bottom))
    end
    return tw, th, tiles
end

function runtiles!(d, kind, net, t, tiles, src, ss, g1, s1, g2, s2, scale, dst, sd, invscale)
    be = d.backend
    for tile in tiles
        Mantle.kilaunch!(be, input_kernel!, t.input, src, g1, g2, Val(kind), Val(net.CI), Val(t.tw), Val(t.th),
            tile.ax, tile.ay, tile.x0, tile.y0, tile.rw, tile.rh, ss, s1, s2, scale;
            ndrange = cld(t.tw * t.th, 256) * 256, workgroupsize = 256)
        forward!(be, net, t)
        Mantle.kilaunch!(be, output_kernel!, dst, t.output, Val(kind), Val(net.layers[:dec_conv1c].CO), Val(t.tw),
            tile.ax, tile.ay, tile.x0, tile.y0, tile.left, tile.top, tile.nx, tile.ny, sd, invscale;
            ndrange = cld(Int(tile.nx) * Int(tile.ny), 256) * 256, workgroupsize = 256)
    end
    return dst
end

function auxguides!(d, img, sa, sn, t, tiles)
    pl = planarstrides(img.w, img.h)
    runtiles!(d, :albedo, unet!(d, :albedo), t, tiles, img.albedo, sa, img.albedo, sa, img.albedo, sa,
              1.0f0, img.galbedo, pl, 1.0f0)
    runtiles!(d, :normal, unet!(d, :normal), t, tiles, img.normal, sn, img.normal, sn, img.normal, sn,
              1.0f0, img.gnormal, pl, 1.0f0)
    return pl
end
function download(a, w, h)
    out = Array{Float32}(undef, w, h, 3)
    copyto!(vec(out), a)
    return out
end

function gpudenoise(d, color, albedo, normal, clean_aux, s)
    w, h = size(color, 1), size(color, 2)
    img = deviceimages!(d, w, h)
    sc = upload!(img.color, color); sa = upload!(img.albedo, albedo); sn = upload!(img.normal, normal)
    tw, th, tiles = tileplan(d, w, h)
    t = tilebuffers!(d, tw, th)
    if clean_aux
        g1, s1, g2, s2 = img.albedo, sa, img.normal, sn
    else
        pl = auxguides!(d, img, sa, sn, t, tiles)
        g1, s1, g2, s2 = img.galbedo, pl, img.gnormal, pl
    end
    runtiles!(d, :hdr, unet!(d, :hdr), t, tiles, img.color, sc, g1, s1, g2, s2, s,
              img.output, planarstrides(w, h), 1.0f0 / s)
    return download(img.output, w, h)
end

function gpuprefilter(d, albedo, normal)
    size(albedo, 1) == size(normal, 1) && size(albedo, 2) == size(normal, 2) || error("guide dimensions must match")
    w, h = size(albedo, 1), size(albedo, 2)
    img = deviceimages!(d, w, h)
    sa = upload!(img.albedo, albedo); sn = upload!(img.normal, normal)
    tw, th, tiles = tileplan(d, w, h)
    auxguides!(d, img, sa, sn, tilebuffers!(d, tw, th), tiles)
    return (albedo = download(img.galbedo, w, h), normal = download(img.gnormal, w, h))
end
