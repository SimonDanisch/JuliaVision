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
    buffers::Vector{Mantle.Buffer}
end
function DeviceImages(backend, w, h)
    buffers = Mantle.Buffer[]
    a() = devicearray!(buffers, backend, Float32, (3 * w * h,))
    return DeviceImages(w, h, a(), a(), a(), a(), a(), a(), buffers)
end
free!(x::DeviceImages) = (foreach(Mantle.free!, x.buffers); empty!(x.buffers); nothing)
function free!(t::TileBuffers)
    foreach(Mantle.free!,values(t.plans));empty!(t.plans)
    foreach(Mantle.free!, t.buffers); empty!(t.buffers)
    return nothing
end
free!(n::UNet) = (foreach(Mantle.free!, n.buffers); empty!(n.buffers); nothing)
free!(::Array) = nothing
free!(b::Mantle.Buffer) = Mantle.free!(b)   # a device buffer the workspace holds

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

auxguides!(d, img, sa, sn, t, tiles) = auxguides!(d, img, img.albedo, sa, img.normal, sn, t, tiles)
"""The prefiltered guides of `albedo` and `normal` (strides `sa`, `sn`) into `img.galbedo`, `img.gnormal`."""
function auxguides!(d, img, albedo, sa, normal, sn, t, tiles)
    pl = planarstrides(img.w, img.h)
    runtiles!(d, :albedo, unet!(d, :albedo), t, tiles, albedo, sa, albedo, sa, albedo, sa,
              1.0f0, img.galbedo, pl, 1.0f0)
    runtiles!(d, :normal, unet!(d, :normal), t, tiles, normal, sn, normal, sn, normal, sn,
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

"""
    denoise!(d, out, color, albedo, normal; clean_aux = false, scale = nothing) -> out

`color` denoised with its guides into `out`: w×h device matrices of three
Float32 channels (`RGB{Float32}`, `Vec3f`) on `d`'s device, read and written
where they are. Nothing crosses to the host but the input scale, which is
computed on the device too (`deviceinputscale`).
"""
function denoise!(d::OIDNDenoiser, out, color, albedo, normal; clean_aux::Bool = false, scale = nothing)
    d.precision === Float16 || error("the device pipeline runs the Float16 networks; this denoiser is $(d.precision)")
    KA.get_backend(color) == d.backend ||
        error("the images are on $(KA.get_backend(color)), the denoiser on $(d.backend)")
    return lock(d.lock) do
        w, h = size(color)
        size(albedo) == size(normal) == size(out) == (w, h) || error("radiance, guides and output must match")
        s = scale === nothing ? deviceinputscale(d, color) : Float32(scale)
        isfinite(s) && s > 0 || error("input scale must be positive and finite")
        flat(a) = reinterpret(Float32, vec(a))
        il = interleaved(w)
        img = deviceimages!(d, w, h)
        tw, th, tiles = tileplan(d, w, h)
        t = tilebuffers!(d, tw, th)
        if clean_aux
            g1, s1, g2, s2 = flat(albedo), il, flat(normal), il
        else
            pl = auxguides!(d, img, flat(albedo), il, flat(normal), il, t, tiles)
            g1, s1, g2, s2 = img.galbedo, pl, img.gnormal, pl
        end
        runtiles!(d, :hdr, unet!(d, :hdr), t, tiles, flat(color), il, g1, s1, g2, s2, s,
                  flat(out), il, 1.0f0 / s)
        return out
    end
end

"""
    deviceinputscale(d, color) -> Float32

OIDN's autoexposure (`inputscale`) of a device image: the mean log luminance
over a grid of about 16×16-pixel bins, each bin averaged by one thread.
"""
function deviceinputscale(d, color)
    w, h = size(color)
    nw, nh = cld(w, 16), cld(h, 16)
    dev = Mantle.todevice(d.backend)
    b = get!(() -> Mantle.Buffer(dev, Float32, (nw * nh,)), d.workspace, :binluminance)
    if length(b) != nw * nh
        Mantle.free!(b)
        b = d.workspace[:binluminance] = Mantle.Buffer(dev, Float32, (nw * nh,))
    end
    lum = Mantle.storage(b)
    Mantle.kilaunch!(d.backend, binluminance_kernel!, lum, color, Int32(nw), Int32(nh);
                     ndrange = cld(nw * nh, 256) * 256, workgroupsize = 256)
    total = mapreduce(l -> l > 1.0f-8 ? log2(l) : 0.0f0, +, lum)
    count = mapreduce(l -> l > 1.0f-8 ? 1.0f0 : 0.0f0, +, lum)
    return count > 0 ? 0.18f0 / exp2(total / count) : 1.0f0
end

function binluminance_kernel!(lum, color, nw::Int32, nh::Int32)
    i = Int32(KI.get_global_id().x)
    i <= nw * nh || return nothing
    w, h = Int32(size(color, 1)), Int32(size(color, 2))
    bx, by = (i - Int32(1)) % nw, (i - Int32(1)) ÷ nw
    x0, x1 = bx * w ÷ nw + Int32(1), (bx + Int32(1)) * w ÷ nw
    y0, y1 = by * h ÷ nh + Int32(1), (by + Int32(1)) * h ÷ nh
    acc = 0.0f0
    for y in y0:y1, x in x0:x1
        @inbounds c = color[x, y]
        acc += 0.212671f0 * ksanitize(c.r, 0.0f0, floatmax(Float32)) +
               0.71516f0 * ksanitize(c.g, 0.0f0, floatmax(Float32)) +
               0.072169f0 * ksanitize(c.b, 0.0f0, floatmax(Float32))
    end
    @inbounds lum[i] = acc / Float32((x1 - x0 + Int32(1)) * (y1 - y0 + Int32(1)))
    return nothing
end
