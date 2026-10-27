"""
PP-OCRv6 medium — text detection and line recognition.

A page goes in, lines of text with their boxes come out, and each line's words
with where they start and end. Two networks: a detector that marks every pixel
inside a line of text (PP-LCNetV4 backbone, a feature-pyramid neck and DB's
differentiable-binarisation head, 22M parameters), and a recogniser that reads
one line at a time (PP-LCNetV4, two SVTR attention blocks and a CTC head over
18710 characters, 19M parameters). Latin script including ä ö ü ß €, and
Chinese, Japanese, Korean and more. A third, PP-LCNet's page classifier, says
which way up a page is.

Upstream: https://github.com/PaddlePaddle/PaddleOCR, through the PyTorch port in
Hugging Face Transformers (`PaddlePaddle/PP-OCRv6_medium_{det,rec}_safetensors`).
License: **Apache-2.0**, code and weights.

Exported by `tools/export_ppocrv6.py`; bound with
`julia --project=<env> tools/make_artifacts.jl ppocrv6_det ppocrv6_rec ppocr_docori`.

Everything between the networks runs on the device too: scaling the page into
the detector's and the classifier's inputs, turning the detector's map into
boxes (connected components and DB's "unclip"), and cutting the lines out at 48
pixels high, a batch of them at a time. The image operations are GPUFiltering's. The page is uploaded once; only the boxes, the recogniser's best
class per time step and four orientation logits come back. CTC decoding and
reading order are a few dozen strings and boxes, and stay on the host.

    using PPOCRRunner
    ocr = ppocr()
    for line in read_lines(ocr, img)       # img[x, y], any colour
        println(line.text)
    end

A caller that has its page on the device already writes it into
`pageinput!(ocr, size).pixels` and passes the [`PageInput`](@ref) instead.
Use a model on the thread that made its device: Mantle submits from there.
"""
module PPOCRRunner

using DNNKernels, KernelAbstractions
import Mantle, GPUFiltering
import KernelInterface as KI
using GPUFiltering: PlanarMap, Area, Bilinear
using GeometryBasics: Vec, Vec2f
using Mantle: @setup_workload, @compile_workload
using LazyArtifacts
using DNNKernels: loadgraph, readsafetensors, Model, recordedplan
using ColorTypes: Colorant, RGB
import JSON3

export PPOCR, ppocr, PageInput, pageinput!, upload!, TextLine, detect, recognize, read_lines, orientation

const KA = KernelAbstractions

"""
    detdir(), recdir(), oridir() -> String

Where the three exports live: their artifacts, cached across every environment
on this machine. Changing them means re-exporting and re-binding, see the
module docstring.
"""
detdir() = @artifact_str("ppocrv6_det")
recdir() = @artifact_str("ppocrv6_rec")
oridir() = @artifact_str("ppocr_docori")

"""
    ready() -> Bool

Whether the exports are installed. The workload branches on this, because it
must not fail on a machine that has not run the exporter.
"""
ready() = isfile(joinpath(detdir(), "ppocrv6_det.json")) && isfile(joinpath(recdir(), "ppocrv6_rec.json")) &&
          isfile(joinpath(oridir(), "ppocr_docori.json"))

# The normalisation the exporter left outside the graphs. Getting these wrong
# gives plausible garbage, not an error, so they are stated once, here.
const DET_MEAN = (0.485f0, 0.456f0, 0.406f0)
const DET_STD = (0.229f0, 0.224f0, 0.225f0)
const REC_MEAN = 0.5f0
const REC_STD = 0.5f0

"""
    REC_HEIGHT, REC_WIDTHS

A line is scaled to `REC_HEIGHT` pixels high keeping its aspect, at most
`last(REC_WIDTHS)` wide, and padded on the right with zeros to the next of
`REC_WIDTHS`, as upstream pads the lines of a batch.

The recogniser is symbolic in its width, and every width is a plan of its own
whose convolutions pick their tiles from it, so a new width is new kernels to
compile. Rounding to a multiple of 64 made every page bring widths of its own:
a fresh session spent a minute compiling on its first page. Eight fixed widths
are planned at load and compiled by the workload, and padding a line by up to
half again costs a few milliseconds of a recogniser that takes twenty.
"""
const REC_HEIGHT = 48
const REC_WIDTHS = (320, 480, 640, 960, 1280, 1920, 2560, 3200)

"""
    recbatches(W, columns = 7680) -> batch sizes

How many lines of input width `W` the recogniser reads in one call, largest
first: 8, 4, 2 or 1, as long as a batch holds at most `columns` input columns.
The lines of one width are read in batches of these sizes, the largest that
still fills, so no batch carries empty slots.

One line underuses the device: short lines are mostly launch and latency, and
eight lines of 320 columns took a third of eight single calls; at 1920 and 3200
columns the gain falls to about a quarter (the Ada, with other work running).
The plans of one model share a memory arena sized by the largest of them. With
`columns` all of them take about 450 MiB on the device, against 140 MiB for
one line per call; 3200 columns at a batch of 8 alone would add another 520.
"""
recbatches(W::Integer, columns::Integer = 7680) = filter(b -> b == 1 || W * b <= columns, (8, 4, 2, 1))

"""
    ORI_SHORT, ORI_SIZE

The orientation classifier sees the page with its short side scaled to
`ORI_SHORT` and the centre `ORI_SIZE` square cut out, as upstream's
`ResizeImage(resize_short=256)` and `CropImage(size=224)`.
"""
const ORI_SHORT = 256
const ORI_SIZE = 224

"""
    BOX_CAPACITY

How many text regions one page may have. The device compacts the boxes into a
buffer this long; a page with more fails loudly instead of losing lines.
"""
const BOX_CAPACITY = 4096

# --------------------------------------------------------------------- kernels

# Connected components of the detector's map above `THRESHOLD`
# (`GPUFiltering.components!`), then per-region statistics and compacted boxes.
# The map never leaves the device.
const THRESHOLD = 0.3f0
const BOX_THRESHOLD = 0.6f0
const UNCLIP = 1.5f0
const MIN_SIZE = 3

# Plain KernelInterface functions: one work item per pixel or region, guarded
# against the whole workgroups a launch rounds up to.
function components_clear!(stats, sums, counter)
    k = Int(KI.get_global_id().x)
    k <= size(stats, 2) || return nothing
    @inbounds begin
        stats[1, k] = typemax(Int32); stats[2, k] = typemax(Int32)
        stats[3, k] = 0; stats[4, k] = 0; stats[5, k] = 0; sums[k] = 0f0
        k == 1 && (counter[1] = 0)
    end
    return nothing
end

# Each region's bounding box, size and probability sum, kept at its label.
function components_stats!(stats, sums, labels, prob, W::Int)
    k = Int(KI.get_global_id().x)
    k <= length(labels) || return nothing
    @inbounds r = labels[k]
    if r != 0
        x = Int32(mod(k - 1, W) + 1); y = Int32((k - 1) ÷ W + 1)
        KA.@atomic :monotonic min(stats[1, r], x)
        KA.@atomic :monotonic min(stats[2, r], y)
        KA.@atomic :monotonic max(stats[3, r], x)
        KA.@atomic :monotonic max(stats[4, r], y)
        KA.@atomic :monotonic stats[5, r] += Int32(1)
        KA.@atomic :monotonic sums[r] += prob[k]
    end
    return nothing
end

# As upstream's DB post-processing: a region whose mean probability is below
# `BOX_THRESHOLD` is dropped, and each box is grown by `area * UNCLIP /
# perimeter`, since the network marks a shrunk core of each line. Boxes are
# axis-aligned, in page pixels: pages are upright and flat before they are read.
function components_boxes!(boxes, counter, stats, sums, shape, W::Int, H::Int)
    k = Int(KI.get_global_id().x)
    k <= size(stats, 2) || return nothing
    @inbounds begin
        count = stats[5, k]; x0 = stats[1, k]; y0 = stats[2, k]; x1 = stats[3, k]; y1 = stats[4, k]
        bw = x1 - x0 + 1; bh = y1 - y0 + 1
        if count > 0 && min(bw, bh) >= MIN_SIZE && sums[k] / count >= BOX_THRESHOLD
            d = Float32(bw) * Float32(bh) * UNCLIP / (2f0 * (bw + bh))
            if min(bw, bh) + 2d >= MIN_SIZE + 2
                slot = KA.@atomic :monotonic counter[1] += Int32(1)
                if slot <= BOX_CAPACITY
                    w = shape[1]; h = shape[2]; scale = min(W / Float32(w), H / Float32(h))
                    boxes[1, slot] = max(0f0, (x0 - 1 - d) / scale); boxes[2, slot] = max(0f0, (y0 - 1 - d) / scale)
                    boxes[3, slot] = min(Float32(w), (x1 + d) / scale); boxes[4, slot] = min(Float32(h), (y1 + d) / scale)
                end
            end
        end
    end
    return nothing
end

# ----------------------------------------------------------------- the model

"""
    PageInput

A page on the model's device: `pixels`, `RGB{Float32}` in `[x, y]`, the maps
that cut the current batch of lines out of it, and the passes that scale it
into the networks. Fill `pixels` on the device, or from
the host with [`upload!`](@ref); [`read_lines`](@ref), [`detect`](@ref),
[`recognize`](@ref) and [`orientation`](@ref) read it. [`pageinput!`](@ref)
makes one per page size, and the model frees them all on `close`.
"""
struct PageInput{P, L}
    pixels::P
    lines::L
    plans::Dict{Any, Any}
    staging::Dict{DataType, Any}
end

Base.size(p::PageInput) = size(Mantle.storage(p.pixels))

"""
    BoxSearch

The device side of turning the detector's map into boxes: component labels,
per-region statistics, the compacted boxes and their count, the page size they
are scaled to, and the plan recorded against the detector's output.
"""
struct BoxSearch{P}
    labels::Any
    stats::Any
    sums::Any
    count::Any
    boxes::Any
    shape::Any
    plan::P
end

function Base.close(b::BoxSearch)
    Mantle.free!(b.plan)
    foreach(Mantle.free!, (b.labels, b.stats, b.sums, b.count, b.boxes, b.shape))
    return nothing
end

"""
    Readback

Host arrays the device's small results are copied into, made once: the box
count and boxes, the page size, the orientation logits, the maps of a batch of
lines, and the recogniser's best class and its probability per time step for
each width and batch size.
"""
struct Readback{L, B, I}
    count::Vector{Int32}
    shape::Vector{Int32}
    boxes::Matrix{Float32}
    logits::L
    lines::Vector{PlanarMap}
    best::Dict{NTuple{2, Int}, B}
    index::Dict{NTuple{2, Int}, I}
end

"""
    PPOCR

The three loaded networks (detector, recogniser, page orientation) with the
plans they were recorded into, the recogniser's character set (index 1 is the
CTC blank), the box search, the readback arrays, and one [`PageInput`](@ref)
per page size seen.
"""
struct PPOCR{B, D, M, S, R}
    backend::B
    device::D
    det::M
    rec::M
    ori::M
    page::NTuple{2, Int}
    characters::Vector{String}
    boxes::S
    host::R
    pages::Dict{NTuple{2, Int}, PageInput}
end

Base.show(io::IO, m::PPOCR) = print(io, "PPOCR(detector ", m.page[1], "x", m.page[2], ", ", length(m.characters), " characters)")

detplan(m::PPOCR) = recordedplan(m.det, "ppocrv6_det")
oriplan(m::PPOCR) = recordedplan(m.ori, "ppocr_docori")
recplan(rec::Model, W::Int, B::Int) = recordedplan(rec, "ppocrv6_rec"; dims = (w = W ÷ 16, b = B))
recplan(m::PPOCR, W::Int, B::Int) = recplan(m.rec, W, B)
maxbatch() = maximum(W -> first(recbatches(W)), REC_WIDTHS)

"""
    record(declare, dev) -> Mantle.Plan

The graph `declare(graph)` builds on `dev`, recorded. A graph orders its passes
by the array objects they name, so each array is one object throughout.
"""
function record(declare, dev)
    g = Mantle.Graph(dev)
    declare(g)
    plan = Mantle.Plan(g)
    Mantle.record!(plan)
    return plan
end

"""
    ppocr(; backend = Mantle.defaultbackend()) -> PPOCR

Load the three networks, plan the detector, the classifier and every
recogniser width, and record the box search against the detector's output.
"""
function ppocr(; backend = Mantle.defaultbackend())
    ready() || throw(ArgumentError(
        "PP-OCRv6 exports not found. Generate them with `uv run tools/export_ppocrv6.py` and bind them " *
        "with `tools/make_artifacts.jl ppocrv6_det ppocrv6_rec ppocr_docori`."))
    det = Model(Dict("ppocrv6_det" => loadgraph(joinpath(detdir(), "ppocrv6_det.json"))),
                readsafetensors(joinpath(detdir(), "weights.safetensors")); backend)
    rec = Model(Dict("ppocrv6_rec" => loadgraph(joinpath(recdir(), "ppocrv6_rec.json"))),
                readsafetensors(joinpath(recdir(), "weights.safetensors")); backend)
    ori = Model(Dict("ppocr_docori" => loadgraph(joinpath(oridir(), "ppocr_docori.json"))),
                readsafetensors(joinpath(oridir(), "weights.safetensors")); backend)
    characters = String.(JSON3.read(read(joinpath(recdir(), "characters.json"), String)))
    dev = det.device
    prob = first(recordedplan(det, "ppocrv6_det").outputs)
    W, H = size(prob, 1), size(prob, 2)
    logits = first(recordedplan(ori, "ppocr_docori").outputs)
    outputs = Dict((w, b) => recplan(rec, w, b).outputs for w in REC_WIDTHS for b in recbatches(w))
    n = W * H
    labels, stats = Mantle.Buffer(dev, Int32, (n,)), Mantle.Buffer(dev, Int32, (5, n))
    sums, count = Mantle.Buffer(dev, Float32, (n,)), Mantle.Buffer(dev, Int32, (1,))
    boxes, shape = Mantle.Buffer(dev, Float32, (4, BOX_CAPACITY)), Mantle.Buffer(dev, Int32, (2,))
    plan = record(dev) do g
        l, map = Mantle.storage(labels), reshape(prob, W, H)
        Mantle.dispatch!(g, components_clear!, (stats, sums, count), n; group = 256, name = "clear components")
        GPUFiltering.components!(g, l, map, >(THRESHOLD))
        Mantle.dispatch!(g, components_stats!, (stats, sums, l, map, W), n; group = 256, name = "component statistics")
        Mantle.dispatch!(g, components_boxes!, (boxes, count, stats, sums, shape, W, H), n; group = 256, name = "compact boxes")
    end
    host = Readback(zeros(Int32, 1), zeros(Int32, 2), zeros(Float32, 4, BOX_CAPACITY),
                    Array{eltype(logits)}(undef, size(logits)), Vector{PlanarMap}(undef, maxbatch()),
                    Dict(w => Array{eltype(o[1])}(undef, size(o[1])) for (w, o) in outputs),
                    Dict(w => Array{eltype(o[2])}(undef, size(o[2])) for (w, o) in outputs))
    return PPOCR(backend, dev, det, rec, ori, (W, H), characters,
                 BoxSearch(labels, stats, sums, count, boxes, shape, plan), host,
                 Dict{NTuple{2, Int}, PageInput}())
end

"""
    close(m::PPOCR)

Free the three networks, their plans, the box search and every page input on
the device. Call it on the thread that loaded the model; it cannot be used
afterwards.
"""
function Base.close(m::PPOCR)
    Mantle.waitidle(m.device)
    for p in values(m.pages)
        foreach(Mantle.free!, values(p.plans))
        for (buffer, plan) in values(p.staging)
            Mantle.free!(plan); Mantle.free!(buffer)
        end
        Mantle.free!(p.lines); Mantle.free!(p.pixels)
    end
    empty!(m.pages)
    close(m.boxes)
    foreach(close, (m.det, m.rec, m.ori))
    return nothing
end

"""
    pageinput!(m, (w, h)) -> PageInput

The model's page input of this size, made on first use and kept until the
model is closed. Its `pixels` are the page the next call reads.
"""
function pageinput!(m::PPOCR, dims::NTuple{2, Int})
    min(dims...) > 0 || throw(ArgumentError("a page needs pixels, got size $dims"))
    get!(m.pages, dims) do
        PageInput(Mantle.Buffer(m.device, RGB{Float32}, dims), Mantle.Buffer(m.device, PlanarMap, (maxbatch(),)),
                  Dict{Any, Any}(), Dict{DataType, Any}())
    end
end

"""
    upload!(m, img) -> PageInput

`img[x, y]`, any colour, copied to the device as the page input of its size.
"""
function upload!(m::PPOCR, img::AbstractMatrix{<:Colorant})
    page = pageinput!(m, size(img))
    T = eltype(img)
    buffer, plan = get!(page.staging, T) do
        staging = Mantle.Buffer(m.device, T, size(img))
        staging, record(g -> GPUFiltering.convertpixels!(g, Mantle.storage(page.pixels), Mantle.storage(staging)), m.device)
    end
    copyto!(buffer, img)
    Mantle.run!(plan)
    return page
end

"""
    pass!(declare, m, page, key)

The graph `declare(graph)` builds for `page`, recorded the first time under
`key`, run.
"""
function pass!(declare, m::PPOCR, page::PageInput, key)
    plan = get!(() -> record(declare, m.device), page.plans, key)
    Mantle.run!(plan)
    return nothing
end

"""
    scale!(graph, input, page, map, fill)

The page through `map` into a network's `(W, H, 3, 1)` input, averaging each
input pixel's cover and normalised for the detector and the classifier.
"""
scale!(g, input, page::PageInput, map::PlanarMap, fill) =
    GPUFiltering.sampleplanar!(g, input, Mantle.storage(page.pixels), map, Area(), DET_MEAN, DET_STD, fill)

# ------------------------------------------------------------------- detection

"""
    detect(m, page) -> Vector{NTuple{4, Float32}}

The lines of text on the page, as boxes `(x0, y0, x1, y1)` in its pixels, in
reading order. `page` is a [`PageInput`](@ref) or an image on the host.
"""
function detect(m::PPOCR, page::PageInput)
    W, H = m.page
    w, h = size(page)
    mp = detplan(m)
    input = only(mp.inputs)
    # Letterboxed: scaled to fit, in the top-left corner, the rest white paper.
    s = min(W / w, H / h)
    map = PlanarMap(Vec2f(1 / s, 1 / s), Vec2f(0, 0), Vec{2, Int32}(round(Int, w * s), round(Int, h * s)))
    pass!(g -> scale!(g, input, page, map, 1.0f0), m, page, :detector)
    DNNKernels.call(m.det, "ppocrv6_det", input; dims = (;))
    host, search = m.host, m.boxes
    host.shape .= (w, h)
    copyto!(search.shape, host.shape)
    Mantle.run!(search.plan)
    copyto!(host.count, search.count)
    count = Int(host.count[1])
    count <= BOX_CAPACITY || throw(ArgumentError(
        "the detector found $count text regions, more than the $BOX_CAPACITY a page may have"))
    boxes = host.boxes
    copyto!(boxes, 1, Mantle.storage(search.boxes), 1, 4count)
    return reading_order([ntuple(j -> boxes[j, k], 4) for k in 1:count])
end

"""
    reading_order(boxes)

Top to bottom, and left to right within a line: boxes whose vertical centres
are closer than half the smaller height count as one line.
"""
function reading_order(boxes::Vector{NTuple{4, Float32}})
    sorted = sort(boxes; by = b -> (b[2] + b[4]) / 2)
    rows = Vector{Vector{NTuple{4, Float32}}}()
    for b in sorted
        cy, bh = (b[2] + b[4]) / 2, b[4] - b[2]
        if !isempty(rows)
            last = rows[end][end]
            if abs(cy - (last[2] + last[4]) / 2) < min(bh, last[4] - last[2]) / 2
                push!(rows[end], b)
                continue
            end
        end
        push!(rows, [b])
    end
    return reduce(vcat, (sort(r; by = first) for r in rows); init = NTuple{4, Float32}[])
end

# ----------------------------------------------------------------- orientation

"""
    orientation(m, page) -> degrees

How far, clockwise, the page must be turned for its text to stand upright:
0, 90, 180 or 270. PP-LCNet's page classifier looks at the centre of the page
with its short side at 256 pixels, so it needs some text there and none of it
has to be recognisable; it reports how far the content is turned clockwise,
and the correction is the rest of the circle.
"""
function orientation(m::PPOCR, page::PageInput)
    w, h = size(page)
    mp = oriplan(m)
    input = only(mp.inputs)
    # The short side to ORI_SHORT and the centre ORI_SIZE square cut out.
    s = ORI_SHORT / min(w, h)
    offset = Vec2f((round(Int, w * s) - ORI_SIZE) ÷ 2, (round(Int, h * s) - ORI_SIZE) ÷ 2) / s
    map = PlanarMap(Vec2f(1 / s, 1 / s), offset, Vec{2, Int32}(ORI_SIZE, ORI_SIZE))
    pass!(g -> scale!(g, input, page, map, 1.0f0), m, page, :orientation)
    logits = m.host.logits
    copyto!(logits, first(DNNKernels.call(m.ori, "ppocr_docori", input; dims = (;))))
    turned = 90 * (argmax(vec(logits)) - 1)
    return mod(360 - turned, 360)
end

# ----------------------------------------------------------------- recognition

"""
    TextLine

One line read off a page: its text, its box `(x0, y0, x1, y1)` in page pixels,
the mean probability of its characters, and its words with where each starts
and ends along the line.
"""
struct TextLine
    text::String
    box::NTuple{4, Float32}
    score::Float32
    words::Vector{Tuple{String, Float32, Float32}}
end

Base.show(io::IO, l::TextLine) = print(io, "TextLine(", repr(l.text), ", ", round.(Int, l.box), ")")

"""
    linewidth(box) -> (W, rw)

The input width `W` of the recogniser that reads `box`, one of `REC_WIDTHS`,
and how much of it, `rw`, the line fills at `REC_HEIGHT` pixels high.
"""
function linewidth((bx0, by0, bx1, by1)::NTuple{4, Float32})
    bw, bh = max(bx1 - bx0, 1.0f0), max(by1 - by0, 1.0f0)
    rw = clamp(ceil(Int, REC_HEIGHT * bw / bh), 1, last(REC_WIDTHS))
    return recwidth(rw), rw
end

"The recogniser width a line `rw` input pixels wide is padded to."
recwidth(rw::Integer) = REC_WIDTHS[findfirst(>=(rw), REC_WIDTHS)]

"""
    recognize(m, page, boxes; retry = true) -> Vector{TextLine}
    recognize(m, page, box; retry = true) -> TextLine

Read the lines inside `boxes` on the page, the lines of one recogniser width in
batches ([`recbatches`](@ref)). CTC decoding: the most likely class per time
step, repeats merged, blanks dropped. Each kept character remembers its time
step, and the steps are spread evenly over the input, which is what gives the
words their positions.

A curved line needs a taller box, and scaling that box to 48 pixels also makes
its letters narrower, which can lose most of the text. With `retry`, the lines
read with less than 0.8 confidence are read once more 1.5 times as wide, in
batches of their own, and the more confident reading of each wins.
"""
function recognize(m::PPOCR, page::PageInput, boxes::AbstractVector{NTuple{4, Float32}}; retry::Bool = true)
    rws = [last(linewidth(box)) for box in boxes]
    lines = recognize(m, page, boxes, rws)
    retry || return lines
    again = [k for k in eachindex(lines) if lines[k].score < 0.8f0 && rws[k] < last(REC_WIDTHS)]
    wider = recognize(m, page, boxes[again], [min(cld(3rws[k], 2), last(REC_WIDTHS)) for k in again])
    for (k, line) in zip(again, wider)
        line.score > lines[k].score && (lines[k] = line)
    end
    return lines
end

recognize(m::PPOCR, page::PageInput, box::NTuple{4, Float32}; retry::Bool = true) =
    only(recognize(m, page, [box]; retry))

# The lines at input widths `rws`: grouped by recogniser width, each group read
# in the largest batches that still fill.
function recognize(m::PPOCR, page::PageInput, boxes::AbstractVector{NTuple{4, Float32}}, rws::AbstractVector{Int})
    lines = Vector{TextLine}(undef, length(boxes))
    widths = recwidth.(rws)
    for W in unique(sort(widths))
        group = findall(==(W), widths)
        while !isempty(group)
            B = first(b for b in recbatches(W) if b <= length(group))
            # Other tasks on the device's thread get a turn between batches.
            yield()
            readbatch!(lines, m, page, boxes, rws, splice!(group, 1:B), W)
        end
    end
    return lines
end

# One batch: the lines cut out of the page into the recogniser's input, read,
# and decoded into `lines` at their indices.
function readbatch!(lines, m::PPOCR, page::PageInput, boxes, rws, batch, W::Int)
    B = length(batch)
    for (slot, k) in enumerate(batch)
        # The box `rw` input pixels wide and REC_HEIGHT high, bilinear without
        # antialiasing as upstream's `cv2.resize`, padded with zeros to `W`.
        bx0, by0, bx1, by1 = boxes[k]
        scale = Vec2f(max(bx1 - bx0, 1.0f0) / rws[k], max(by1 - by0, 1.0f0) / REC_HEIGHT)
        m.host.lines[slot] = PlanarMap(scale, Vec2f(bx0, by0), Vec{2, Int32}(rws[k], REC_HEIGHT))
    end
    copyto!(page.lines, m.host.lines)
    input = only(recplan(m, W, B).inputs)
    pass!(m, page, (:lines, W, B)) do g
        GPUFiltering.sampleplanar!(g, input, Mantle.storage(page.pixels), Mantle.storage(page.lines), Bilinear(),
                                   ntuple(_ -> REC_MEAN, 3), ntuple(_ -> REC_STD, 3), REC_MEAN)
    end
    best, index = DNNKernels.call(m.rec, "ppocrv6_rec", input; dims = (w = W ÷ 16, b = B))
    hb, hi = m.host.best[(W, B)], m.host.index[(W, B)]
    copyto!(hb, best); copyto!(hi, index)
    for (slot, k) in enumerate(batch)
        lines[k] = decode(m, view(hb, :, slot), view(hi, :, slot), W, rws[k], boxes[k])
    end
    return lines
end

function decode(m::PPOCR, best, index, W, rw, box)
    T = length(index)
    chars, steps, probs = String[], Int[], Float32[]
    previous = 0
    for t in 1:T
        k = Int(index[t])
        if k != 0 && k != previous
            push!(chars, m.characters[k + 1])
            push!(steps, t)
            push!(probs, best[t])
        end
        previous = k
    end
    # Input pixels per step, and page pixels per input pixel.
    step = W / T
    f = (box[3] - box[1]) / rw
    xpage(t) = box[1] + Float32((t - 0.5f0) * step * f)
    words = Tuple{String, Float32, Float32}[]
    i = 1
    while i <= length(chars)
        if isempty(strip(chars[i]))
            i += 1
            continue
        end
        j = i
        while j < length(chars) && !isempty(strip(chars[j + 1]))
            j += 1
        end
        half = Float32(step * f / 2)
        push!(words, (join(chars[i:j]), max(box[1], xpage(steps[i]) - half), min(box[3], xpage(steps[j]) + half)))
        i = j + 1
    end
    score = isempty(probs) ? 0.0f0 : sum(probs) / length(probs)
    return TextLine(strip(join(chars)), box, score, words)
end

"""
    read_lines(m, page) -> Vector{TextLine}

Every line of text on the page that reads as something, in reading order.
"""
read_lines(m::PPOCR, page::PageInput) = filter(line -> !isempty(line.text), recognize(m, page, detect(m, page)))

# The same three on an image on the host, uploaded first.
detect(m::PPOCR, img::AbstractMatrix{<:Colorant}) = detect(m, upload!(m, img))
orientation(m::PPOCR, img::AbstractMatrix{<:Colorant}) = orientation(m, upload!(m, img))
read_lines(m::PPOCR, img::AbstractMatrix{<:Colorant}) = read_lines(m, upload!(m, img))

# ---------------------------------------------------------------- the workload
#
# Guarded on the assets and on a working device, so precompiling never fails;
# it drives `read_lines`, the call a scanner makes, on a small synthetic page.
@setup_workload begin
    if ready()
        try
            backend = Mantle.defaultbackend()
            @compile_workload begin
                m = ppocr(; backend)
                try
                    page = fill(RGB{Float32}(0.95f0, 0.95f0, 0.95f0), 620, 877)
                    for y in 200:212, x in 100:500
                        page[x, y] = RGB{Float32}(0.1f0, 0.1f0, 0.1f0)
                    end
                    input = upload!(m, page)
                    read_lines(m, input)
                    orientation(m, input)
                    for W in REC_WIDTHS, B in recbatches(W)
                        box = (100.0f0, 196.0f0, Float32(100 + W * 16 / REC_HEIGHT - 2), 212.0f0)
                        recognize(m, input, fill(box, B); retry = false)
                    end
                finally
                    close(m)
                end
            end
        catch err
            @warn "PPOCRRunner: workload skipped; first use will compile" exception = err
        end
    else
        @info "PPOCRRunner: no export installed — nothing precompiled"
    end
end

end # module
