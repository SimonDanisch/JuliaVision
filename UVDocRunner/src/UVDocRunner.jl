"""
UVDoc document unwarping on the GPU through DNNKernels/Mantle.

The 7.9M-parameter PaddlePaddle/UVDoc_safetensors network predicts a backward
grid from an RGB page resized to 488×712. The grid is sampled at the original
resolution, preserving the source dimensions. All images use `img[x,y]`, and
everything runs on the device: the resize into the network, the network, and
the warp along its grid, the image operations GPUFiltering's. Export: `tools/export_uvdoc.py`; bind:
`tools/make_artifacts.jl uvdoc`. Upstream code and weights: Apache-2.0.

    using UVDocRunner
    m = uvdoc()
    flat = unwarp(m, img)          # img[x, y], any colour; RGB{Float32} back

A caller with the page on the device already writes it into
`UVDocRunner.pageinput!(m, size).pixels`, calls `UVDocRunner.unwarp!(m, page)`
and reads `page.output`. Use a model on the thread that made its device:
Mantle submits from there.
"""
module UVDocRunner

using Artifacts, LazyArtifacts, DNNKernels, ColorTypes
using DNNKernels: recordedplan
import Mantle, GPUFiltering
using GPUFiltering: PlanarMap, Bilinear
using GeometryBasics: Vec, Vec2f

export UVDoc, uvdoc, unwarp
public pageinput!, upload!, unwarp!

modeldir() = artifact"uvdoc"
function ready()
    toml = joinpath(@__DIR__, "..", "Artifacts.toml")
    isfile(toml) || return false
    hash = artifact_hash("uvdoc", toml)
    return hash !== nothing && artifact_exists(hash)
end

# The network's input size.
const WIDTH = 488
const HEIGHT = 712

"""
    PageInput

A page on the model's device: `pixels` to unwarp and `output` for the result,
both `RGB{Float32}` in `[x, y]` and the same size, and the passes between them.
[`pageinput!`](@ref) makes one per page size, and the model frees them all on
`close`.
"""
struct PageInput{P}
    pixels::P
    output::P
    plans::Dict{Symbol, Any}
    staging::Dict{DataType, Any}
end

Base.size(p::PageInput) = size(Mantle.storage(p.pixels))

"""
    UVDoc

The network, its recorded plan, and one [`PageInput`](@ref) per page size seen.
"""
struct UVDoc{M, D}
    model::M
    device::D
    pages::Dict{NTuple{2, Int}, PageInput}
end

Base.show(io::IO, ::UVDoc) = print(io, "UVDoc(488×712 → document grid)")

plan(m::UVDoc) = recordedplan(m.model, "uvdoc")

"""
    record(declare, dev) -> Mantle.Plan

The graph `declare(graph)` builds on `dev`, recorded.
"""
function record(declare, dev)
    g = Mantle.Graph(dev)
    declare(g)
    plan = Mantle.Plan(g)
    Mantle.record!(plan)
    return plan
end

"""
    uvdoc(; backend = Mantle.defaultbackend()) -> UVDoc

Load UVDoc and plan its fixed-size network. Reuse it across pages.
"""
function uvdoc(; backend = Mantle.defaultbackend())
    dir = modeldir()
    model = DNNKernels.Model(Dict("uvdoc" => DNNKernels.loadgraph(joinpath(dir, "uvdoc.json"))),
        DNNKernels.readsafetensors(joinpath(dir, "weights.safetensors")); backend)
    recordedplan(model, "uvdoc")
    return UVDoc(model, model.device, Dict{NTuple{2, Int}, PageInput}())
end

"""
    close(m::UVDoc)

Free the network, its plan and every page input on the device. Call it on the
thread that loaded the model; it cannot be used afterwards.
"""
function Base.close(m::UVDoc)
    Mantle.waitidle(m.device)
    for p in values(m.pages)
        foreach(Mantle.free!, values(p.plans))
        for (buffer, plan) in values(p.staging)
            Mantle.free!(plan); Mantle.free!(buffer)
        end
        Mantle.free!(p.output); Mantle.free!(p.pixels)
    end
    empty!(m.pages)
    close(m.model)
    return nothing
end

"""
    pageinput!(m, (w, h)) -> PageInput

The model's page input of this size, made on first use and kept until the
model is closed.
"""
function pageinput!(m::UVDoc, dims::NTuple{2, Int})
    min(dims...) > 0 || throw(ArgumentError("UVDoc needs a nonempty image, got size $dims"))
    get!(m.pages, dims) do
        pixels = Mantle.Buffer(m.device, RGB{Float32}, dims)
        output = Mantle.Buffer(m.device, RGB{Float32}, dims)
        mp = plan(m)
        w, h = dims
        # RGB in [0, 1] resized to the network's input with align_corners=True,
        # as UVDocImageProcessor does: the corner pixels' centres meet.
        s = Vec2f((w - 1) / (WIDTH - 1), (h - 1) / (HEIGHT - 1))
        map = PlanarMap(s, 0.5f0 .- 0.5f0 .* s, Vec{2, Int32}(WIDTH, HEIGHT))
        plans = Dict{Symbol, Any}(
            :input => record(g -> GPUFiltering.sampleplanar!(g, only(mp.inputs), Mantle.storage(pixels), map, Bilinear(),
                                                             (0.0f0, 0.0f0, 0.0f0), (1.0f0, 1.0f0, 1.0f0), 0.0f0), m.device),
            # The backward grid applied, as torch's grid_sample with
            # align_corners=True and zero padding.
            :warp => record(g -> GPUFiltering.gridsample!(g, Mantle.storage(output), Mantle.storage(pixels), only(mp.outputs)), m.device))
        PageInput(pixels, output, plans, Dict{DataType, Any}())
    end
end

"""
    upload!(m, img) -> PageInput

`img[x, y]`, any colour, copied to the device as the page input of its size.
"""
function upload!(m::UVDoc, img::AbstractMatrix{<:Colorant})
    page = pageinput!(m, size(img))
    buffer, toplan = get!(page.staging, eltype(img)) do
        staging = Mantle.Buffer(m.device, eltype(img), size(img))
        staging, record(g -> GPUFiltering.convertpixels!(g, Mantle.storage(page.pixels), Mantle.storage(staging)), m.device)
    end
    copyto!(buffer, img)
    Mantle.run!(toplan)
    return page
end

"""
    unwarp!(m, page) -> page

Predict the page's deformation grid and write the flattened page into
`page.output`, at the page's resolution.
"""
function unwarp!(m::UVDoc, page::PageInput)
    Mantle.run!(page.plans[:input])
    mp = plan(m)
    DNNKernels.call(m.model, "uvdoc", only(mp.inputs); dims = (;))
    Mantle.run!(page.plans[:warp])
    return page
end

"""
    unwarp(m, img) -> Matrix{RGB{Float32}}

Flatten a curved page with UVDoc, keeping its resolution.
"""
function unwarp(m::UVDoc, img::AbstractMatrix{<:Colorant})
    page = unwarp!(m, upload!(m, img))
    out = Matrix{RGB{Float32}}(undef, size(img))
    copyto!(out, page.output)
    return out
end

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
