"""
UVDoc document unwarping on the GPU through DNNKernels/Mantle.

The 7.9M-parameter PaddlePaddle/UVDoc_safetensors network predicts a backward
grid from an RGB page resized to 488×712. The grid is sampled at the original
resolution, preserving the source dimensions. All images use `img[x,y]`.
Export: `tools/export_uvdoc.py`; bind: `tools/make_artifacts.jl uvdoc`.
Upstream code and weights: Apache-2.0.
"""
module UVDocRunner

using Artifacts, LazyArtifacts, DNNKernels, ColorTypes
import KernelAbstractions as KA
import Mantle

export UVDoc, uvdoc, unwarp, deformation_grid

modeldir() = artifact"uvdoc"
function ready()
    toml = joinpath(@__DIR__, "..", "Artifacts.toml")
    isfile(toml) || return false
    hash = artifact_hash("uvdoc", toml)
    return hash !== nothing && artifact_exists(hash)
end

struct UVDoc{M,A}
    model::M
    input::A
    requests::Channel{Any}
    worker::Task
end

Base.show(io::IO, ::UVDoc) = print(io, "UVDoc(488×712 → document grid)")

"""Load UVDoc and plan its fixed-size GPU network. Reuse it across pages."""
function uvdoc(; backend = Mantle.defaultbackend())
    dir = modeldir()
    model = DNNKernels.Model(Dict("uvdoc" => DNNKernels.loadgraph(joinpath(dir, "uvdoc.json"))),
        DNNKernels.readsafetensors(joinpath(dir, "weights.safetensors")); backend)
    DNNKernels.planahead!(model, "uvdoc")
    input = KA.allocate(model.backend, Float32, 488, 712, 3, 1)
    requests = Channel{Any}(Inf)
    # Mantle submission belongs to the constructing thread. A sticky worker
    # both serializes shared buffers and serves pages from Threads.@spawn.
    worker = @async try
        for (host, reply) in requests
            put!(reply, try
                copyto!(input, host)
                out = only(DNNKernels.call(model, "uvdoc", input; dims = (;)))
                (dropdims(Array(out); dims = 4), nothing)
            catch err
                (nothing, CapturedException(err, catch_backtrace()))
            end)
        end
    finally
        # On the device's own thread, after the last request.
        KA.unsafe_free!(input)
        close(model)
    end
    errormonitor(worker)
    return UVDoc(model, input, requests, worker)
end

"""
    close(m::UVDoc)

Stop serving pages, then free the network and its input on the device.
"""
function Base.close(m::UVDoc)
    close(m.requests)
    wait(m.worker)
    return nothing
end

@inline channels(c) = (Float32(red(c)), Float32(green(c)), Float32(blue(c)))
@inline mix(a, b, f) = a .* (1f0 - f) .+ b .* f

# Border clamping is for resizing only. Image warping below uses zero padding,
# exactly as torch.grid_sample(..., align_corners=True).
@inline function sample(img, x, y)
    w, h = size(img)
    x, y = clamp(x, 1f0, Float32(w)), clamp(y, 1f0, Float32(h))
    a, b = floor(Int, x), floor(Int, y)
    c, d = min(a + 1, w), min(b + 1, h)
    return mix(mix(channels(img[a,b]), channels(img[c,b]), x-a),
               mix(channels(img[a,d]), channels(img[c,d]), x-a), y-b)
end

"""Resize RGB [0,1] with align_corners=True, matching UVDocImageProcessor."""
function prepare(img::AbstractMatrix{<:AbstractRGB})
    isempty(img) && throw(ArgumentError("UVDoc needs a nonempty image"))
    w, h = size(img)
    out = Array{Float32}(undef, 488, 712, 3, 1)
    for y in 1:712, x in 1:488
        px = sample(img, 1f0 + (x-1) * Float32((w-1)/487), 1f0 + (y-1) * Float32((h-1)/711))
        for c in 1:3
            out[x,y,c,1] = px[c]
        end
    end
    return out
end

"""Predict normalized source coordinates, `(grid_width, grid_height, 2)`. Thread safe."""
function deformation_grid(m::UVDoc, img::AbstractMatrix{<:AbstractRGB})
    host = prepare(img)
    reply = Channel{Any}(1)
    put!(m.requests, (host, reply))
    grid, err = take!(reply)
    err === nothing || throw(err)
    return grid
end

@inline function pixel_or_zero(img, x, y)
    return 1 <= x <= size(img,1) && 1 <= y <= size(img,2) ? channels(@inbounds img[x,y]) : (0f0, 0f0, 0f0)
end

"""
    unwarp(img, grid; output_size=size(img))

Apply a normalized backward grid. Fuses bilinear grid upsampling and image
sampling with align_corners=True and zero padding, without a full-size grid
allocation. Returns RGB Float32; values remain in the input's [0,1] range.
"""
function unwarp(img::AbstractMatrix{<:AbstractRGB}, grid::AbstractArray{<:Real,3}; output_size = size(img))
    isempty(img) && throw(ArgumentError("UVDoc needs a nonempty image"))
    gw, gh, nc = size(grid)
    gw > 0 && gh > 0 && nc == 2 || throw(ArgumentError("grid must have shape (width,height,2)"))
    all(isfinite, grid) || throw(ArgumentError("grid contains nonfinite coordinates"))
    W, H = output_size
    W > 0 && H > 0 || throw(ArgumentError("output dimensions must be positive"))
    w, h = size(img)
    out = Matrix{RGB{Float32}}(undef, W, H)
    sx, sy = Float32((gw-1)/max(W-1,1)), Float32((gh-1)/max(H-1,1))
    Threads.@threads for y in 1:H
        for x in 1:W
            gx, gy = min(1f0+(x-1)*sx, Float32(gw)), min(1f0+(y-1)*sy, Float32(gh))
            a, b = floor(Int, gx), floor(Int, gy)
            c, d = min(a+1,gw), min(b+1,gh)
            uv = ntuple(2) do k
                mix(mix(Float32(grid[a,b,k]), Float32(grid[c,b,k]), gx-a),
                    mix(Float32(grid[a,d,k]), Float32(grid[c,d,k]), gx-a), gy-b)
            end
            px, py = (uv[1]+1f0)*0.5f0*(w-1)+1f0, (uv[2]+1f0)*0.5f0*(h-1)+1f0
            if px <= 0 || px >= w+1 || py <= 0 || py >= h+1
                out[x,y] = RGB{Float32}(0,0,0)
                continue
            end
            i, j = floor(Int, px), floor(Int, py)
            rgb = mix(mix(pixel_or_zero(img,i,j), pixel_or_zero(img,i+1,j), px-i),
                      mix(pixel_or_zero(img,i,j+1), pixel_or_zero(img,i+1,j+1), px-i), py-j)
            out[x,y] = RGB{Float32}(rgb...)
        end
    end
    return out
end

"""Flatten a curved page with UVDoc, keeping its resolution by default."""
unwarp(m::UVDoc, img::AbstractMatrix{<:AbstractRGB}; kw...) = unwarp(img, deformation_grid(m, img); kw...)

# Keep first-page compilation out of an interactive scan. Never fetch weights
# during precompilation, and allow environments without a usable GPU to load.
Mantle.@setup_workload begin
    if ready()
        try
            Mantle.@compile_workload begin
                m = uvdoc()
                try
                    page = fill(RGB{Float32}(0.95f0, 0.94f0, 0.93f0), 293, 417)
                    unwarp(m, page)
                finally
                    close(m)
                end
            end
        catch err
            @warn "UVDocRunner: workload skipped; first use will compile" exception = err
        end
    end
end

end
