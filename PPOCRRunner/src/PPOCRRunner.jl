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
boxes (connected components and DB's "unclip"), and cutting each line out at 48
pixels high. The page is uploaded once; only the boxes, the recogniser's best
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
import Mantle
using Mantle: @setup_workload, @compile_workload
using LazyArtifacts
using DNNKernels: loadgraph, readsafetensors, Model, recordedplan
using ColorTypes: Colorant, AbstractRGB, RGB, red, green, blue
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

channels(c::AbstractRGB) = (Float32(red(c)), Float32(green(c)), Float32(blue(c)))

"""
    area(img, x0, x1, y0, y1) -> (r, g, b)

The mean colour of `img[x, y]` over the pixels a footprint from `x0` to `x1`
and `y0` to `y1` (in pixel edges, 0-based) covers. Downscaling by averaging each
output pixel's footprint keeps thin strokes from aliasing away.
"""
@inline function area(img::AbstractMatrix, x0, x1, y0, y1)
    w, h = size(img)
    acc, n = (0.0f0, 0.0f0, 0.0f0), 0
    for y in clamp(floor(Int, y0) + 1, 1, h):clamp(ceil(Int, y1), 1, h),
            x in clamp(floor(Int, x0) + 1, 1, w):clamp(ceil(Int, x1), 1, w)
        acc = acc .+ channels(@inbounds img[x, y])
        n += 1
    end
    return acc ./ max(n, 1)
end

@inline function bilinear(img::AbstractMatrix, x::Float32, y::Float32)
    w, h = size(img)
    x, y = clamp(x, 1.0f0, Float32(w)), clamp(y, 1.0f0, Float32(h))
    xa, ya = min(floor(Int, x), w - 1), min(floor(Int, y), h - 1)
    xa, ya = max(xa, 1), max(ya, 1)
    fx, fy = x - xa, y - ya
    a, b = channels(img[xa, ya]), channels(img[min(xa + 1, w), ya])
    c, d = channels(img[xa, min(ya + 1, h)]), channels(img[min(xa + 1, w), min(ya + 1, h)])
    return (a .* (1 - fx) .+ b .* fx) .* (1 - fy) .+ (c .* (1 - fx) .+ d .* fx) .* fy
end

@kernel function topixels!(out, @Const(src))
    i = @index(Global)
    @inbounds out[i] = convert(RGB{Float32}, src[i])
end

# The page scaled into a network's `(W, H, 3, 1)` input. For the detector it is
# letterboxed: scaled by `s` to fit, in the top-left corner, the rest white
# paper. For the classifier its short side goes to `ORI_SHORT` and the centre
# `W × H` is cut out.
@kernel function scaled!(out, @Const(img), w::Int, h::Int, W::Int, H::Int, orientation::Bool)
    i = @index(Global)
    u = mod(i - 1, W) + 1; v = mod((i - 1) ÷ W, H) + 1; c = (i - 1) ÷ (W * H) + 1
    s = orientation ? ORI_SHORT / min(w, h) : min(W / w, H / h)
    cx = orientation ? (round(Int, w * s) - W) ÷ 2 : 0
    cy = orientation ? (round(Int, h * s) - H) ÷ 2 : 0
    px = 1f0
    if orientation || (u <= round(Int, w * s) && v <= round(Int, h * s))
        f = 1 / s
        rgb = area(img, (cx + u - 1) * f, (cx + u) * f, (cy + v - 1) * f, (cy + v) * f)
        px = rgb[c]
    end
    @inbounds out[i] = (px - DET_MEAN[c]) / DET_STD[c]
end

# The line inside `box[1:4]`, `box[5]` input pixels wide and `REC_HEIGHT` high
# (bilinear without antialiasing, as upstream's `cv2.resize`), normalised and
# padded with zeros to the input's width `W`.
@kernel function line!(out, @Const(img), @Const(box), W::Int)
    i = @index(Global)
    u = mod(i - 1, W) + 1; v = mod((i - 1) ÷ W, REC_HEIGHT) + 1; c = (i - 1) ÷ (W * REC_HEIGHT) + 1
    @inbounds begin
        bx0 = box[1]; by0 = box[2]; bw = max(box[3] - bx0, 1f0); bh = max(box[4] - by0, 1f0); rw = Int(box[5])
    end
    val = 0f0
    if u <= rw
        px = bilinear(img, bx0 + (u - 0.5f0) * (bw / rw) + 0.5f0, by0 + (v - 0.5f0) * (bh / REC_HEIGHT) + 0.5f0)
        val = (px[c] - REC_MEAN) / REC_STD
    end
    @inbounds out[i] = val
end

# Connected components of the detector's map above `threshold`: a monotone
# union-find (a larger root hooks to the smaller), then per-root statistics and
# compacted boxes. The map never leaves the device.
const THRESHOLD = 0.3f0
const BOX_THRESHOLD = 0.6f0
const UNCLIP = 1.5f0
const MIN_SIZE = 3

@kernel function components_init!(parents, stats, sums, counter, @Const(prob))
    k = @index(Global)
    @inbounds begin
        parents[k] = prob[k] > THRESHOLD ? Int32(k) : Int32(0)
        stats[1, k] = typemax(Int32); stats[2, k] = typemax(Int32)
        stats[3, k] = 0; stats[4, k] = 0; stats[5, k] = 0; sums[k] = 0f0
        k == 1 && (counter[1] = 0)
    end
end

@inline function root(parents, k::Int32)
    p = KA.@atomic :monotonic parents[k]
    while p != k
        k = p; p = KA.@atomic :monotonic parents[k]
    end
    k
end

@inline function join!(parents, a::Int32, b::Int32)
    while true
        a = root(parents, a); b = root(parents, b)
        a == b && break
        hi = max(a, b); lo = min(a, b)
        result = KA.@atomicreplace :monotonic parents[hi] hi => lo
        result.success && break
    end
    nothing
end

@kernel function components_merge!(parents, @Const(prob), W::Int)
    k = @index(Global); x = mod(k - 1, W) + 1; y = (k - 1) ÷ W + 1
    if @inbounds prob[k] > THRESHOLD
        if x > 1 && (@inbounds prob[k - 1] > THRESHOLD)
            join!(parents, Int32(k), Int32(k - 1))
        end
        if y > 1
            for dx in -1:1
                j = k - W + dx
                if 1 <= x + dx <= W && (@inbounds prob[j] > THRESHOLD)
                    join!(parents, Int32(k), Int32(j))
                end
            end
        end
    end
end

@kernel function components_stats!(stats, sums, @Const(parents), @Const(prob), W::Int)
    k = @index(Global)
    if @inbounds parents[k] != 0
        r = Int32(k)
        while (@inbounds parents[r]) != r
            @inbounds r = parents[r]
        end
        x = Int32(mod(k - 1, W) + 1); y = Int32((k - 1) ÷ W + 1)
        KA.@atomic min(stats[1, r], x)
        KA.@atomic min(stats[2, r], y)
        KA.@atomic max(stats[3, r], x)
        KA.@atomic max(stats[4, r], y)
        KA.@atomic stats[5, r] += Int32(1)
        KA.@atomic sums[r] += prob[k]
    end
end

# As upstream's DB post-processing: a region whose mean probability is below
# `BOX_THRESHOLD` is dropped, and each box is grown by `area * UNCLIP /
# perimeter`, since the network marks a shrunk core of each line. Boxes are
# axis-aligned, in page pixels: pages are upright and flat before they are read.
@kernel function components_boxes!(boxes, counter, @Const(stats), @Const(sums), @Const(shape), W::Int, H::Int)
    k = @index(Global)
    @inbounds begin
        count = stats[5, k]; x0 = stats[1, k]; y0 = stats[2, k]; x1 = stats[3, k]; y1 = stats[4, k]
        bw = x1 - x0 + 1; bh = y1 - y0 + 1
        if count > 0 && min(bw, bh) >= MIN_SIZE && sums[k] / count >= BOX_THRESHOLD
            d = Float32(bw) * Float32(bh) * UNCLIP / (2f0 * (bw + bh))
            if min(bw, bh) + 2d >= MIN_SIZE + 2
                slot = KA.@atomic counter[1] += Int32(1)
                if slot <= BOX_CAPACITY
                    w = shape[1]; h = shape[2]; scale = min(W / Float32(w), H / Float32(h))
                    boxes[1, slot] = max(0f0, (x0 - 1 - d) / scale); boxes[2, slot] = max(0f0, (y0 - 1 - d) / scale)
                    boxes[3, slot] = min(Float32(w), (x1 + d) / scale); boxes[4, slot] = min(Float32(h), (y1 + d) / scale)
                end
            end
        end
    end
end

# ----------------------------------------------------------------- the model

"""
    PageInput

A page on the model's device: `pixels`, `RGB{Float32}` in `[x, y]`, and the
passes that scale it into the networks. Fill `pixels` on the device, or from
the host with [`upload!`](@ref); [`read_lines`](@ref), [`detect`](@ref),
[`recognize`](@ref) and [`orientation`](@ref) read it. [`pageinput!`](@ref)
makes one per page size, and the model frees them all on `close`.
"""
struct PageInput{P, L}
    pixels::P
    line::L
    plans::Dict{Any, Any}
    staging::Dict{DataType, Any}
end

Base.size(p::PageInput) = size(Mantle.storage(p.pixels))

"""
    BoxSearch

The device side of turning the detector's map into boxes: union-find parents,
per-region statistics, the compacted boxes and their count, the page size they
are scaled to, and the plan recorded against the detector's output.
"""
struct BoxSearch{P}
    parents::Any
    stats::Any
    sums::Any
    count::Any
    boxes::Any
    shape::Any
    plan::P
end

function Base.close(b::BoxSearch)
    Mantle.free!(b.plan)
    foreach(Mantle.free!, (b.parents, b.stats, b.sums, b.count, b.boxes, b.shape))
    return nothing
end

"""
    Readback

Host arrays the device's small results are copied into, made once: the box
count and boxes, the page size, the orientation logits, a line's box, and the
recogniser's best class and its probability per time step for each width.
"""
struct Readback{L, B, I}
    count::Vector{Int32}
    shape::Vector{Int32}
    boxes::Matrix{Float32}
    logits::L
    line::Vector{Float32}
    best::Dict{Int, B}
    index::Dict{Int, I}
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
recplan(rec::Model, W::Int) = recordedplan(rec, "ppocrv6_rec"; dims = (w = W ÷ 16,))
recplan(m::PPOCR, W::Int) = recplan(m.rec, W)

function record(dev, passes...)
    g = Mantle.Graph(dev)
    for (kernel, args, n, name) in passes
        Mantle.dispatch!(g, kernel, args, n; name)
    end
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
    outputs = Dict(w => recplan(rec, w).outputs for w in REC_WIDTHS)
    n = W * H
    parents, stats = Mantle.Buffer(dev, Int32, (n,)), Mantle.Buffer(dev, Int32, (5, n))
    sums, count = Mantle.Buffer(dev, Float32, (n,)), Mantle.Buffer(dev, Int32, (1,))
    boxes, shape = Mantle.Buffer(dev, Float32, (4, BOX_CAPACITY)), Mantle.Buffer(dev, Int32, (2,))
    plan = record(dev,
        (components_init!, (parents, stats, sums, count, prob), n, "component initialization"),
        (components_merge!, (parents, prob, W), n, "component union"),
        (components_stats!, (stats, sums, parents, prob, W), n, "component statistics"),
        (components_boxes!, (boxes, count, stats, sums, shape, W, H), n, "compact boxes"))
    host = Readback(zeros(Int32, 1), zeros(Int32, 2), zeros(Float32, 4, BOX_CAPACITY),
                    Array{eltype(logits)}(undef, size(logits)), zeros(Float32, 5),
                    Dict(w => Array{eltype(o[1])}(undef, size(o[1])) for (w, o) in outputs),
                    Dict(w => Array{eltype(o[2])}(undef, size(o[2])) for (w, o) in outputs))
    return PPOCR(backend, dev, det, rec, ori, (W, H), characters,
                 BoxSearch(parents, stats, sums, count, boxes, shape, plan), host,
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
        Mantle.free!(p.line); Mantle.free!(p.pixels)
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
        PageInput(Mantle.Buffer(m.device, RGB{Float32}, dims), Mantle.Buffer(m.device, Float32, (5,)),
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
        staging, record(m.device, (topixels!, (page.pixels, staging), length(img), "page to Float32"))
    end
    copyto!(buffer, img)
    Mantle.run!(plan)
    return page
end

function pass!(m::PPOCR, page::PageInput, key, kernel, args, n)
    plan = get!(() -> record(m.device, (kernel, args, n, string(key))), page.plans, key)
    Mantle.run!(plan)
    return nothing
end

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
    pass!(m, page, :detector, scaled!, (input, page.pixels, w, h, W, H, false), W * H * 3)
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
    pass!(m, page, :orientation, scaled!, (input, page.pixels, w, h, ORI_SIZE, ORI_SIZE, true), ORI_SIZE * ORI_SIZE * 3)
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
    return REC_WIDTHS[findfirst(>=(rw), REC_WIDTHS)], rw
end

"""
    recognize(m, page, box; retry = true) -> TextLine

Read the line inside `box` on the page. CTC decoding: the most likely class per
time step, repeats merged, blanks dropped. Each kept character remembers its
time step, and the steps are spread evenly over the input, which is what gives
the words their positions.

A curved line needs a taller box, and scaling that box to 48 pixels also makes
its letters narrower, which can lose most of the text. With `retry`, a line
read with less than 0.8 confidence is read once more 1.5 times as wide, and the
more confident reading wins.
"""
function recognize(m::PPOCR, page::PageInput, box::NTuple{4, Float32}; retry::Bool = true)
    _, rw = linewidth(box)
    line = recognize(m, page, box, rw)
    if retry && line.score < 0.8f0 && rw < last(REC_WIDTHS)
        yield()
        wider = recognize(m, page, box, min(cld(3rw, 2), last(REC_WIDTHS)))
        wider.score > line.score && return wider
    end
    return line
end

function recognize(m::PPOCR, page::PageInput, box::NTuple{4, Float32}, rw::Int)
    W = REC_WIDTHS[findfirst(>=(rw), REC_WIDTHS)]
    params = m.host.line
    for k in 1:4
        params[k] = box[k]
    end
    params[5] = rw
    copyto!(page.line, params)
    mp = recplan(m, W)
    input = only(mp.inputs)
    pass!(m, page, (:line, W), line!, (input, page.pixels, page.line, W), W * REC_HEIGHT * 3)
    best, index = DNNKernels.call(m.rec, "ppocrv6_rec", input; dims = (w = W ÷ 16,))
    hb, hi = m.host.best[W], m.host.index[W]
    copyto!(hb, best); copyto!(hi, index)
    return decode(m, vec(hb), vec(hi), W, rw, box)
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
function read_lines(m::PPOCR, page::PageInput)
    lines = TextLine[]
    for box in detect(m, page)
        # Other tasks on the device's thread get a turn between lines.
        yield()
        line = recognize(m, page, box)
        isempty(line.text) || push!(lines, line)
    end
    return lines
end

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
                    for W in REC_WIDTHS
                        recognize(m, input, (100.0f0, 196.0f0, Float32(100 + W * 16 / REC_HEIGHT - 2), 212.0f0))
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
