"""
PP-DocLayoutV3 on the GPU: a page's regions, what each is, and the order to
read them in.

The 33M-parameter detector (an RT-DETR with a reading-order head, from
PaddleOCR) sees the page resized to 800x800 and proposes 300 regions of 25
kinds: text, paragraph and document titles, tables, images, charts, formulas,
headers, footers, footnotes, seals, page numbers and more. A region is kept
when its kind scores at least `threshold`; the order head scores every pair of
regions for which comes first, and the regions come back in reading order.

    using PPDocLayoutRunner
    m = doclayout()
    for r in regions(m, img)        # img[x, y], any colour
        println(r.kind, " ", r.box)
    end

The resize into the network runs on the device (`GPUFiltering`, bicubic as the
model was trained with); 300 scores, boxes and the order matrix come back, and
the selection and ordering of a few hundred values stay on the host. A caller
with its page on the device already writes it into `pageinput!(m, size).pixels`.
Use a model on the thread that made its device: Mantle submits from there.

Upstream code and weights: Apache-2.0, `PaddlePaddle/PP-DocLayoutV3_safetensors`
through Hugging Face Transformers. Export: `tools/export_ppdoclayout.py`; bind:
`tools/make_artifacts.jl ppdoclayoutv3`.
"""
module PPDocLayoutRunner

using Artifacts, LazyArtifacts, DNNKernels, ColorTypes
using DNNKernels: recordedplan
import Mantle, GPUFiltering, JSON3
using GPUFiltering: PlanarMap, Bicubic
using GeometryBasics: Vec, Vec2f

export DocLayout, doclayout, Region, regions
public pageinput!, upload!

modeldir() = artifact"ppdoclayoutv3"
function ready()
    toml = joinpath(@__DIR__, "..", "Artifacts.toml")
    isfile(toml) || return false
    hash = artifact_hash("ppdoclayoutv3", toml)
    return hash !== nothing && artifact_exists(hash)
end

# The network's input size.
const SIZE = 800

"""
    Region

A region of a page: its `kind` (`:text`, `:paragraph_title`, `:doc_title`,
`:table`, `:image`, `:chart`, `:formula`, `:header`, `:footer`, `:seal`, …),
how sure the detector is of it, and its box `(x0, y0, x1, y1)` in page pixels,
pixel edges as the OCR's boxes.
"""
struct Region
    kind::Symbol
    score::Float32
    box::NTuple{4, Float32}
end

Base.show(io::IO, r::Region) = print(io, "Region(:", r.kind, ", ", round.(Int, r.box), ")")

"""
    PageInput

A page on the model's device: `pixels`, `RGB{Float32}` in `[x, y]`, and the
recorded passes that resize it into the network and convert uploads.
[`pageinput!`](@ref) makes one per page size; the model frees them on `close`.
"""
struct PageInput{P}
    pixels::P
    plans::Dict{Symbol, Any}
    staging::Dict{DataType, Any}
end

Base.size(p::PageInput) = size(Mantle.storage(p.pixels))

"""
    DocLayout

The network and its recorded plan, the 25 region kinds in the order of its
class logits, the host arrays its outputs come back in, and one
[`PageInput`](@ref) per page size seen.
"""
struct DocLayout{M, D}
    model::M
    device::D
    kinds::Vector{Symbol}
    logits::Matrix{Float32}
    boxes::Matrix{Float32}
    order::Matrix{Float32}
    pages::Dict{NTuple{2, Int}, PageInput}
end

Base.show(io::IO, m::DocLayout) = print(io, "DocLayout(", length(m.kinds), " kinds)")

plan(m::DocLayout) = recordedplan(m.model, "ppdoclayoutv3")

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
    doclayout(; backend = Mantle.defaultbackend()) -> DocLayout

Load PP-DocLayoutV3 and plan it. Reuse it across pages.
"""
function doclayout(; backend = Mantle.defaultbackend())
    ready() || throw(ArgumentError(
        "PP-DocLayoutV3 export not found. Generate it with `uv run tools/export_ppdoclayout.py` and bind it " *
        "with `tools/make_artifacts.jl ppdoclayoutv3`."))
    dir = modeldir()
    model = DNNKernels.Model(Dict("ppdoclayoutv3" => DNNKernels.loadgraph(joinpath(dir, "ppdoclayoutv3.json"))),
        DNNKernels.readsafetensors(joinpath(dir, "weights.safetensors")); backend)
    logits, boxes, order = recordedplan(model, "ppdoclayoutv3").outputs
    kinds = Symbol.(JSON3.read(read(joinpath(dir, "labels.json"), String), Vector{String}))
    length(kinds) == size(logits, 1) || throw(ArgumentError("$(length(kinds)) labels for $(size(logits, 1)) classes"))
    return DocLayout(model, model.device, kinds, Matrix{Float32}(undef, size(logits)[1:2]),
                     Matrix{Float32}(undef, size(boxes)[1:2]), Matrix{Float32}(undef, size(order)[1:2]),
                     Dict{NTuple{2, Int}, PageInput}())
end

"""
    close(m::DocLayout)

Free the network, its plan and every page input on the device. Call it on the
thread that loaded the model; it cannot be used afterwards.
"""
function Base.close(m::DocLayout)
    Mantle.waitidle(m.device)
    for p in values(m.pages)
        foreach(Mantle.free!, values(p.plans))
        for (buffer, plan) in values(p.staging)
            Mantle.free!(plan); Mantle.free!(buffer)
        end
        Mantle.free!(p.pixels)
    end
    empty!(m.pages)
    close(m.model)
    return nothing
end

"""
    pageinput!(m, (w, h)) -> PageInput

The model's page input of this size, made on first use and kept until the
model is closed: the page resized into the network's 800x800 input by bicubic
convolution without antialiasing, as the processor it was trained with
(`cv2.resize`), RGB in [0, 1].
"""
function pageinput!(m::DocLayout, dims::NTuple{2, Int})
    min(dims...) > 0 || throw(ArgumentError("a page needs pixels, got size $dims"))
    get!(m.pages, dims) do
        pixels = Mantle.Buffer(m.device, RGB{Float32}, dims)
        w, h = dims
        map = PlanarMap(Vec2f(w / SIZE, h / SIZE), Vec2f(0, 0), Vec{2, Int32}(SIZE, SIZE))
        input = only(plan(m).inputs)
        resize = record(g -> GPUFiltering.sampleplanar!(g, input, Mantle.storage(pixels), map, Bicubic(),
                                                       (0.0f0, 0.0f0, 0.0f0), (1.0f0, 1.0f0, 1.0f0), 0.0f0), m.device)
        PageInput(pixels, Dict{Symbol, Any}(:input => resize), Dict{DataType, Any}())
    end
end

"""
    upload!(m, img) -> PageInput

`img[x, y]`, any colour, copied to the device as the page input of its size.
"""
function upload!(m::DocLayout, img::AbstractMatrix{<:Colorant})
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
    regions(m, page; threshold = 0.5) -> Vector{Region}

The page's regions in reading order. As upstream's post-processing: every
(query, kind) pair whose sigmoid score reaches `threshold` is a region, boxed by
its query, and the regions are ordered by their queries' votes from the order
head, each query counting how many come before it. `page` is a
[`PageInput`](@ref) or an image on the host.
"""
function regions(m::DocLayout, page::PageInput; threshold::Real = 0.5)
    Mantle.run!(page.plans[:input])
    mp = plan(m)
    logits, boxes, order = DNNKernels.call(m.model, "ppdoclayoutv3", only(mp.inputs); dims = (;))
    copyto!(vec(m.logits), logits); copyto!(vec(m.boxes), boxes); copyto!(vec(m.order), order)
    w, h = size(page)
    ranks = orderranks(m.order)
    found = Tuple{Int, Float32, Region}[]
    for q in axes(m.logits, 2), k in axes(m.logits, 1)
        score = sigmoid(m.logits[k, q])
        score >= threshold || continue
        cx, cy, bw, bh = view(m.boxes, :, q)
        box = (clamp((cx - bw / 2) * w, 0, w), clamp((cy - bh / 2) * h, 0, h),
               clamp((cx + bw / 2) * w, 0, w), clamp((cy + bh / 2) * h, 0, h))
        push!(found, (ranks[q], score, Region(m.kinds[k], score, Float32.(box))))
    end
    # By reading order, and the more certain kind first where one query holds two.
    sort!(found; by = f -> (f[1], -f[2]))
    return [last(f) for f in found]
end

regions(m::DocLayout, img::AbstractMatrix{<:Colorant}; kw...) = regions(m, upload!(m, img); kw...)

sigmoid(x) = 1 / (1 + exp(-x))

"""
    orderranks(order) -> Vector{Int}

Each query's place in reading order from the order head's pairwise logits.
`order[j, i]` (torch's `[i, j]`) is how sure the head is that query `i` comes
before query `j`; a query's votes are the chances that each query with a lower
index comes before it, plus the chances that it does not come before each with
a higher one, and the queries are ranked by their votes.
"""
function orderranks(order::AbstractMatrix{Float32})
    n = size(order, 1)
    votes = map(1:n) do j
        before = sum(i -> sigmoid(order[j, i]), 1:(j - 1); init = 0.0f0)
        after = sum(i -> 1 - sigmoid(order[i, j]), (j + 1):n; init = 0.0f0)
        before + after
    end
    return invperm(sortperm(votes))
end

# Keep first-page compilation out of an interactive scan. Never fetch weights
# during precompilation, and allow environments without a usable GPU to load.
Mantle.@setup_workload begin
    if ready()
        try
            Mantle.@compile_workload begin
                m = doclayout()
                try
                    page = fill(RGB{Float32}(0.97f0, 0.97f0, 0.96f0), 620, 877)
                    for y in 200:212, x in 100:500
                        page[x, y] = RGB{Float32}(0.1f0, 0.1f0, 0.1f0)
                    end
                    regions(m, page)
                finally
                    close(m)
                end
            end
        catch err
            @warn "PPDocLayoutRunner: workload skipped; first use will compile" exception = err
        end
    end
end

end
