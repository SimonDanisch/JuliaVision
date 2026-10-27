module OIDNRunner

using DNNKernels
using LazyArtifacts
import Mantle
using ColorTypes: RGB
const KI = Mantle.KI
const KA = Mantle.KA
export OIDNDenoiser, denoise, prefilter, release!, install!
install!(d) = error("load VideoEditor to activate OIDNRunner's editor extension")

# Port of OIDN v2.5.1 core/color.{h,cpp} and GPU input/output processing.
# Copyright 2018-2022 Intel Corporation; SPDX-License-Identifier: Apache-2.0.
# Modified: Julia array operations and DNNKernels/Mantle inference.
srgb_forward(y) = y <= 0.0031308f0 ? 12.92f0 * y : 1.055f0 * y^(1.0f0 / 2.4f0) - 0.055f0
srgb_inverse(x) = x <= 0.04045f0 ? x / 12.92f0 : ((x + 0.055f0) / 1.055f0)^2.4f0
pu_forward(y) = y <= 1.5794576f-6 ? 1.41283765f3 * y :
    y <= 3.22087631f-2 ? 1.64593172f0 * y^4.31384981f-1 - 2.94139609f-3 :
    1.92653254f-1 * log(y + 6.26026094f-3) + 9.98620152f-1
pu_inverse(x) = x <= 2.23151711f-3 ? x / 1.41283765f3 :
    x <= 3.70974749f-1 ? ((x + 2.94139609f-3) / 1.64593172f0)^(1.0f0 / 4.31384981f-1) :
    exp((x - 9.98620152f-1) / 1.92653254f-1) - 6.26026094f-3
const PU_MAX = pu_forward(65504.0f0)
sanitize(v, lo, hi) = clamp(isnan(v) ? 0.0f0 : Float32(v), lo, hi)
const MODEL_NAMES = Dict(:albedo => "rt_alb_large", :normal => "rt_nrm_large", :hdr => "rt_hdr_calb_cnrm_large")
modeldir() = @artifact_str("oidn")

mutable struct OIDNDenoiser
    directory::String
    backend::Any
    models::Dict{Symbol, Any}
    tilesize::Int
    precision::DataType
    fuseconv::Bool
    workspace::Dict{Symbol, Any}
    unets::Dict{Symbol, Any}
    lock::ReentrantLock
end
# The ROCm backend when AMDGPU is loaded: the same kernels run 0.146 s per 1080p
# frame there against 0.178 s through Vulkan on gfx1151 (README). Otherwise the
# platform default; nothing here loads AMDGPU.
function preferredbackend()
    for (name, backend) in zip(Mantle.availablebackends(), Mantle.eachbackend())
        name === :rocm && return backend
    end
    return Mantle.defaultbackend()
end
function OIDNDenoiser(directory::AbstractString = modeldir(); backend = preferredbackend(), tilesize = 2048, precision = :auto, fuseconv = true)
    tilesize >= 256 && tilesize % 16 == 0 || throw(ArgumentError("tile size must be a multiple of 16, at least 256"))
    for name in values(MODEL_NAMES), ext in (".json", ".safetensors")
        isfile(joinpath(directory, name * ext)) || error("missing OIDN export: $(name * ext)")
    end
    T = precision === :auto ? (DNNKernels.caps(backend).coopmat ? Float16 : Float32) : precision
    T in (Float16, Float32) || throw(ArgumentError("precision must be :auto, Float16 or Float32"))
    return OIDNDenoiser(abspath(directory), backend, Dict{Symbol, Any}(), tilesize, T, fuseconv, Dict{Symbol, Any}(), Dict{Symbol, Any}(), ReentrantLock())
end
if isdefined(Mantle, :accstore!)
    include("fusedconv.jl")
else
    # The Vulkan matrix-store hook does not exist on a Metal-only installation.
    fusedgraph(g) = g
end
include("unet.jl")
include("unetgraph.jl")
include("gpupipeline.jl")
function workbuffer(d, key, dims, T = Float32)
    a = get(d.workspace, key, nothing)
    if a === nothing || size(a) != dims || eltype(a) !== T
        a = Array{T}(undef, dims)
        d.workspace[key] = a
    end
    return a
end
function inferencegraph(g, T)
    T === Float32 && return g
    # As upstream HIP: fp16 tensors/weights, fp32 matrix accumulation. HDR
    # range is retained by the host PU transform, outside the neural network.
    buffers = Dict(
        k => DNNKernels.Buffer(
            b.id, b.kind, b.shape, b.dtype === Float32 ? T : b.dtype,
            b.key, b.live, b.of, b.viewop, b.attrs
        ) for (k, b) in g.buffers
    )
    return DNNKernels.Graph(g.name, g.symbols, g.inputs, g.outputs, buffers, g.order, g.ops, g.fusion)
end
function model!(d, kind)
    return get!(d.models, kind) do
        name = MODEL_NAMES[kind]
        graph = inferencegraph(DNNKernels.loadgraph(joinpath(d.directory, name * ".json")), d.precision)
        weights = DNNKernels.readsafetensors(joinpath(d.directory, name * ".safetensors"))
        if d.precision === Float16
            weights = Dict(k => Float16.(v) for (k, v) in weights)
        end
        model = DNNKernels.Model(
            Dict(name => graph), weights;
            backend = d.backend
        )
        if d.precision === Float16 && d.fuseconv
            model.graphs[name] = fusedgraph(model.graphs[name])
        end
        return model
    end
end
function release!(d::OIDNDenoiser)
    lock(d.lock) do
        Mantle.waitidle(Mantle.todevice(d.backend))
        for m in values(d.models)
            DNNKernels.releaseplans!(m)
            DNNKernels.releaseweights!(m.weights)
        end
        empty!(d.models)
        foreach(free!, values(d.workspace))
        empty!(d.workspace)
        foreach(free!, values(d.unets))
        empty!(d.unets)
    end
    return nothing
end

# Preserve upstream's 16-pixel bin partition and geometric mean of bin luminance.
@inline channel(img::AbstractArray{<:Real, 3}, x, y, k) = img[x, y, k]
@inline channel(img::AbstractMatrix{<:RGB}, x, y, k) = (v = img[x, y]; k == 1 ? v.r : k == 2 ? v.g : v.b)
@inline channel(img::AbstractMatrix, x, y, k) = img[x, y][k]
function inputscale(color)
    w, h = size(color, 1), size(color, 2); nw = cld(w, 16); nh = cld(h, 16)
    total = 0.0f0;count = 0
    for by in 0:(nh - 1), bx in 0:(nw - 1)
        x0 = bx * w ÷ nw + 1;x1 = (bx + 1) * w ÷ nw
        y0 = by * h ÷ nh + 1;y1 = (by + 1) * h ÷ nh
        lum = 0.0f0
        for y in y0:y1, x in x0:x1
            lum += 0.212671f0 * sanitize(channel(color, x, y, 1), 0.0f0, floatmax(Float32)) +
                0.71516f0 * sanitize(channel(color, x, y, 2), 0.0f0, floatmax(Float32)) +
                0.072169f0 * sanitize(channel(color, x, y, 3), 0.0f0, floatmax(Float32))
        end
        lum /= Float32((x1 - x0 + 1) * (y1 - y0 + 1))
        if lum > 1.0f-8
            total += log2(lum);count += 1
        end
    end
    return count > 0 ? 0.18f0 / exp2(total / Float32(count)) : 1.0f0
end
function planar(image::AbstractMatrix, kind)
    w, h = size(image);out = Array{Float32}(undef, w, h, 3)
    return planar!(out, image, kind)
end
function planar!(out, image::AbstractMatrix, kind)
    w, h = size(image)
    for k in 1:3, y in 1:h
        @inbounds @simd for x in 1:w
            v = image[x, y]
            out[x, y, k] = kind === :normal ? v[k] : (k == 1 ? v.r : k == 2 ? v.g : v.b)
        end
    end
    return out
end
planar!(out, image::AbstractArray{<:Real, 3}, kind) = (out .= image)
planar(image::AbstractArray{<:Real, 3}, kind) = Float32.(image)
planar(image::Array{Float32, 3}, kind) = image
function network(d, kind, input; temporary = false)
    w, h, c = size(input)
    # OIDN's large network receptive field is 202: 112 pixels of aligned overlap.
    overlap = 112
    tw = min(d.tilesize, 16 * cld(w, 16));th = min(d.tilesize, 16 * cld(h, 16))
    stridew = tw - 2 * overlap;strideh = th - 2 * overlap
    nx = tw >= w ? 1 : cld(w - 2 * overlap, stridew)
    ny = th >= h ? 1 : cld(h - 2 * overlap, strideh)
    out = temporary ? workbuffer(d, Symbol(kind, :_output), (w, h, 3)) : Array{Float32}(undef, w, h, 3)
    m = model!(d, kind)
    tile = workbuffer(d, Symbol(kind, :_tile), (tw, th, c, 1), d.precision)
    for iy in 0:(ny - 1), ix in 0:(nx - 1)
        x0 = ix * stridew;y0 = iy * strideh
        rw = min(w - x0, tw);rh = min(h - y0, th)
        ax = tw - 16 * cld(rw, 16);ay = th - 16 * cld(rh, 16)
        fill!(tile, zero(d.precision))
        @views tile[(ax + 1):(ax + rw), (ay + 1):(ay + rh), :, 1] .= input[(x0 + 1):(x0 + rw), (y0 + 1):(y0 + rh), :]
        # Replay copies directly into its persistent input: no temporary GPU
        # upload buffer, second copy or dtype-conversion dispatch per tile.
        result, = DNNKernels.call(m, MODEL_NAMES[kind], tile; dims = (h = th ÷ 16, w = tw ÷ 16))
        pixels = Array(result)
        left = ix > 0 ? overlap : 0;right = ix < nx - 1 ? overlap : 0
        top = iy > 0 ? overlap : 0;bottom = iy < ny - 1 ? overlap : 0
        @views out[(x0 + left + 1):(x0 + rw - right), (y0 + top + 1):(y0 + rh - bottom), :] .=
            pixels[(ax + left + 1):(ax + rw - right), (ay + top + 1):(ay + rh - bottom), :, 1]
    end
    return out
end
function prefilter_unlocked(d, albedo, normal; temporary = false)
    a = planar(albedo, :albedo);n = planar(normal, :normal)
    size(a) == size(n) && size(a, 3) == 3 || error("guide dimensions must match")
    ai = workbuffer(d, :albedo_input, size(a));ni = workbuffer(d, :normal_input, size(n))
    ai .= srgb_forward.(sanitize.(a, 0.0f0, 1.0f0))
    ni .= sanitize.(n, -1.0f0, 1.0f0) .* 0.5f0 .+ 0.5f0
    af = network(d, :albedo, ai; temporary);nf = network(d, :normal, ni; temporary)
    af .= clamp.(srgb_inverse.(max.(af, 0.0f0)), 0.0f0, 1.0f0)
    nf .= clamp.(max.(nf, 0.0f0) .* 2.0f0 .- 1.0f0, -1.0f0, 1.0f0)
    return (albedo = af, normal = nf)
end
prefilter(d::OIDNDenoiser, albedo, normal) = lock(d.lock) do
    d.precision === Float16 && return gpuprefilter(d, albedo, normal)
    prefilter_unlocked(d, albedo, normal)
end
function denoise(d::OIDNDenoiser, color; albedo, normal, clean_aux = false, scale = nothing)
    return lock(d.lock) do
        w, h = size(color, 1), size(color, 2)
        dims = (w, h, 3)
        for image in (color, albedo, normal)
            size(image, 1) == w && size(image, 2) == h &&
                (ndims(image) == 2 || (ndims(image) == 3 && size(image, 3) == 3)) ||
                error("radiance and guides must match and have three channels")
        end
        s = scale === nothing ? inputscale(color) : Float32(scale)
        isfinite(s) && s > 0 || error("input scale must be positive and finite")
        d.precision === Float16 && return gpudenoise(d, color, albedo, normal, clean_aux, s)
        c = planar!(workbuffer(d, :color_plane, dims), color, :color)
        a = planar!(workbuffer(d, :albedo_plane, dims), albedo, :albedo)
        n = planar!(workbuffer(d, :normal_plane, dims), normal, :normal)
        size(c) == size(a) == size(n) || error("radiance and guides must match")
        if !clean_aux
            guides = prefilter_unlocked(d, a, n; temporary = true);a = guides.albedo;n = guides.normal
        end
        input = workbuffer(d, :hdr_input, (w, h, 9))
        @views input[:, :, 1:3] .= pu_forward.(sanitize.(c .* s, 0.0f0, floatmax(Float32))) ./ PU_MAX
        @views input[:, :, 4:6] .= sanitize.(a, 0.0f0, 1.0f0)
        @views input[:, :, 7:9] .= sanitize.(n, -1.0f0, 1.0f0) .* 0.5f0 .+ 0.5f0
        pred = network(d, :hdr, input; temporary = true)
        output = pu_inverse.(max.(pred, 0.0f0) .* PU_MAX) ./ s
        return output
    end
end
end
