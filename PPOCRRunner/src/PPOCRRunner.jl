"""
PP-OCRv6 medium — text detection and line recognition.

A page goes in, lines of text with their boxes come out, and each line's words
with where they start and end. Two networks: a detector that marks every pixel
inside a line of text (PP-LCNetV4 backbone, a feature-pyramid neck and DB's
differentiable-binarisation head, 22M parameters), and a recogniser that reads
one line at a time (PP-LCNetV4, two SVTR attention blocks and a CTC head over
18710 characters, 19M parameters). Latin script including ä ö ü ß €, and
Chinese, Japanese, Korean and more.

Upstream: https://github.com/PaddlePaddle/PaddleOCR, through the PyTorch port in
Hugging Face Transformers (`PaddlePaddle/PP-OCRv6_medium_{det,rec}_safetensors`).
License: **Apache-2.0**, code and weights.

Exported by `tools/export_ppocrv6.py`; bound with
`julia --project=<env> tools/make_artifacts.jl ppocrv6_det ppocrv6_rec`.

Ops `DNNKernels` did not have: none.

What is not in the graphs, and is here instead: scaling a page into the
detector's fixed input, turning its probability map into boxes (connected
components and DB's "unclip" expansion), cutting each line out and scaling it
to 48 pixels high, and CTC decoding. All of it on the host: a page has a few
dozen lines and each of those steps is cheap next to the networks.

    using PPOCRRunner
    ocr = ppocr()
    for line in read_lines(ocr, img)       # img[x, y], any AbstractRGB
        println(line.text)
    end
"""
module PPOCRRunner

using Lava, DNNKernels, KernelAbstractions
import Mantle
using Mantle: @setup_workload, @compile_workload
using LazyArtifacts
using DNNKernels: loadgraph, readsafetensors, Model
using ColorTypes: AbstractRGB, RGB, red, green, blue
import JSON3

export PPOCR, ppocr, TextLine, detect, recognize, read_lines, orientation

const KA = KernelAbstractions

"""
    detdir(), recdir() -> String

Where the two exports live: their artifacts, cached across every environment on
this machine. Changing them means re-exporting and re-binding, see the module
docstring.
"""
detdir() = @artifact_str("ppocrv6_det")
recdir() = @artifact_str("ppocrv6_rec")
oridir() = @artifact_str("ppocr_docori")

"""
    ready() -> Bool

Whether both exports are installed. The workload branches on this, because it
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
    PPOCR

The three loaded networks (detector, recogniser, page orientation), their input
buffers (the recogniser's one per width), and the recogniser's character set
(index 1 is the CTC blank).
"""
struct PPOCR{B, M}
    backend::B
    det::M
    rec::M
    ori::M
    page::NTuple{2, Int}
    detinput::Any
    recinputs::Dict{Int, Any}
    oriinput::Any
    characters::Vector{String}
end

Base.show(io::IO, m::PPOCR) = print(io, "PPOCR(detector ", m.page[1], "x", m.page[2], ", ", length(m.characters), " characters)")

"""
    close(m::PPOCR)

Free the three networks and their input buffers on the device. Call it on the
thread that loaded the model; it cannot be used afterwards.
"""
function Base.close(m::PPOCR)
    Mantle.waitidle(m.det.device)
    KA.unsafe_free!(m.detinput)
    KA.unsafe_free!(m.oriinput)
    foreach(KA.unsafe_free!, values(m.recinputs))
    empty!(m.recinputs)
    foreach(close, (m.det, m.rec, m.ori))
    return nothing
end

"""
    ppocr(; backend = Mantle.defaultbackend()) -> PPOCR

Load both networks and plan the detector, whose input size is fixed.
"""
function ppocr(; backend = Mantle.defaultbackend())
    ready() || throw(ArgumentError(
        "PP-OCRv6 exports not found. Generate them with `uv run tools/export_ppocrv6.py` and bind them " *
        "with `tools/make_artifacts.jl ppocrv6_det ppocrv6_rec`."))
    det = Model(Dict("ppocrv6_det" => loadgraph(joinpath(detdir(), "ppocrv6_det.json"))),
                readsafetensors(joinpath(detdir(), "weights.safetensors")); backend)
    rec = Model(Dict("ppocrv6_rec" => loadgraph(joinpath(recdir(), "ppocrv6_rec.json"))),
                readsafetensors(joinpath(recdir(), "weights.safetensors")); backend)
    ori = Model(Dict("ppocr_docori" => loadgraph(joinpath(oridir(), "ppocr_docori.json"))),
                readsafetensors(joinpath(oridir(), "weights.safetensors")); backend)
    DNNKernels.planahead!(det, "ppocrv6_det")
    DNNKernels.planahead!(ori, "ppocr_docori")
    for W in REC_WIDTHS
        DNNKernels.planahead!(rec, "ppocrv6_rec"; dims = (w = W ÷ 16,))
    end
    oriinput = KA.allocate(ori.backend, Float32, ORI_SIZE, ORI_SIZE, 3, 1)
    g = det.graphs["ppocrv6_det"]
    shape = DNNKernels.evalshape(g.buffers[only(g.inputs)].shape, (;))
    detinput = KA.allocate(det.backend, Float32, shape...)
    characters = String.(JSON3.read(read(joinpath(recdir(), "characters.json"), String)))
    return PPOCR(det.backend, det, rec, ori, (shape[1], shape[2]), detinput, Dict{Int, Any}(), oriinput, characters)
end

# ------------------------------------------------------------------- detection

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

"""
    letterbox(img, (W, H), s) -> Array{Float32, 4}

`img[x, y]` scaled by `s` (at most 1 on either side of the box it must fit),
normalised for the detector, in the top-left corner of a `(W, H, 3, 1)` input
whose rest is white paper.
"""
function letterbox(img::AbstractMatrix, (W, H)::NTuple{2, Int}, s::Real)
    w, h = size(img)
    out = Array{Float32}(undef, W, H, 3, 1)
    white = ntuple(c -> (1.0f0 - DET_MEAN[c]) / DET_STD[c], 3)
    for c in 1:3
        out[:, :, c, 1] .= white[c]
    end
    sw, sh = min(W, round(Int, w * s)), min(H, round(Int, h * s))
    f = 1 / s
    Threads.@threads for v in 1:sh
        for u in 1:sw
            px = area(img, (u - 1) * f, u * f, (v - 1) * f, v * f)
            for c in 1:3
                @inbounds out[u, v, c, 1] = (px[c] - DET_MEAN[c]) / DET_STD[c]
            end
        end
    end
    return out
end

"""
    probability(m, img) -> (map, s)

The detector's map for `img[x, y]`: for each pixel of its input, the
probability it lies inside a line of text, and the scale `s` from page pixels
to map pixels.
"""
function probability(m::PPOCR, img::AbstractMatrix)
    W, H = m.page
    s = min(W / size(img, 1), H / size(img, 2))
    copyto!(m.detinput, letterbox(img, m.page, s))
    out = first(DNNKernels.call(m.det, "ppocrv6_det", m.detinput; dims = (;)))
    return Array(out)[:, :, 1, 1], s
end

"""
    components(mask) -> (labels, n)

8-connected components of `mask`, labelled 1 to `n`.
"""
function components(mask::AbstractMatrix{Bool})
    w, h = size(mask)
    labels = zeros(Int32, w, h)
    n = 0
    stack = Tuple{Int, Int}[]
    for y in 1:h, x in 1:w
        (mask[x, y] && labels[x, y] == 0) || continue
        n += 1
        labels[x, y] = n
        push!(stack, (x, y))
        while !isempty(stack)
            cx, cy = pop!(stack)
            for dy in -1:1, dx in -1:1
                nx, ny = cx + dx, cy + dy
                (1 <= nx <= w && 1 <= ny <= h) || continue
                if mask[nx, ny] && labels[nx, ny] == 0
                    labels[nx, ny] = n
                    push!(stack, (nx, ny))
                end
            end
        end
    end
    return labels, n
end

"""
    detect(m, img; threshold = 0.3, box_threshold = 0.6, unclip = 1.5, min_size = 3)
        -> Vector{NTuple{4, Float32}}

The lines of text on `img[x, y]`, as boxes `(x0, y0, x1, y1)` in its pixels, in
reading order. As upstream's DB post-processing: pixels above `threshold` form
regions, a region whose mean probability is below `box_threshold` is dropped,
and each box is grown by `area * unclip / perimeter`, since the network marks a
shrunk core of each line. Boxes are axis-aligned: pages here have been turned
upright and flattened before they are read.
"""
function detect(m::PPOCR, img::AbstractMatrix; threshold = 0.3f0, box_threshold = 0.6f0, unclip = 1.5f0, min_size = 3)
    prob, s = probability(m, img)
    labels, n = components(prob .> threshold)
    w, h = size(prob)
    x0, y0 = fill(typemax(Int32), n), fill(typemax(Int32), n)
    x1, y1 = fill(Int32(0), n), fill(Int32(0), n)
    total, count = zeros(Float32, n), zeros(Int, n)
    for y in 1:h, x in 1:w
        k = labels[x, y]
        k == 0 && continue
        x0[k], x1[k] = min(x0[k], x), max(x1[k], x)
        y0[k], y1[k] = min(y0[k], y), max(y1[k], y)
        total[k] += prob[x, y]
        count[k] += 1
    end
    boxes = NTuple{4, Float32}[]
    for k in 1:n
        bw, bh = x1[k] - x0[k] + 1, y1[k] - y0[k] + 1
        min(bw, bh) < min_size && continue
        total[k] / count[k] < box_threshold && continue
        d = bw * bh * unclip / (2 * (bw + bh))
        min(bw, bh) + 2d < min_size + 2 && continue
        # Map pixel edges back to the page.
        push!(boxes, ((x0[k] - 1 - d) / s, (y0[k] - 1 - d) / s, (x1[k] + d) / s, (y1[k] + d) / s))
    end
    W, H = size(img)
    boxes = [(max(a, 0.0f0), max(b, 0.0f0), min(c, Float32(W)), min(d, Float32(H))) for (a, b, c, d) in boxes]
    return reading_order(boxes)
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
    orientation(m, img) -> degrees

How far, clockwise, `img[x, y]` must be turned for its text to stand upright:
0, 90, 180 or 270. PP-LCNet's page classifier looks at the centre of the page
with its short side at 256 pixels, so it needs some text there and none of it
has to be recognisable; it reports how far the content is turned clockwise,
and the correction is the rest of the circle.
"""
function orientation(m::PPOCR, img::AbstractMatrix)
    w, h = size(img)
    s = ORI_SHORT / min(w, h)
    # The centre crop, in the scaled page's pixels.
    cx, cy = (round(Int, w * s) - ORI_SIZE) ÷ 2, (round(Int, h * s) - ORI_SIZE) ÷ 2
    host = Array{Float32}(undef, ORI_SIZE, ORI_SIZE, 3, 1)
    f = 1 / s
    for v in 1:ORI_SIZE, u in 1:ORI_SIZE
        px = area(img, (cx + u - 1) * f, (cx + u) * f, (cy + v - 1) * f, (cy + v) * f)
        for c in 1:3
            host[u, v, c, 1] = (px[c] - DET_MEAN[c]) / DET_STD[c]
        end
    end
    copyto!(m.oriinput, host)
    logits = vec(Array(first(DNNKernels.call(m.ori, "ppocr_docori", m.oriinput; dims = (;)))))
    turned = 90 * (argmax(logits) - 1)
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

"""
    line_input(img, box) -> (input, width)

The line inside `box`, scaled to `REC_HEIGHT` pixels high (bilinear, without
antialiasing, as upstream's `cv2.resize`), normalised, and padded on the right
with zeros to the next of `REC_WIDTHS`. `width` is how much of the input is
the line.
"""
function line_input(img::AbstractMatrix, (bx0, by0, bx1, by1)::NTuple{4, Float32})
    bw, bh = max(bx1 - bx0, 1.0f0), max(by1 - by0, 1.0f0)
    rw = clamp(ceil(Int, REC_HEIGHT * bw / bh), 1, last(REC_WIDTHS))
    W = REC_WIDTHS[findfirst(>=(rw), REC_WIDTHS)]
    out = zeros(Float32, W, REC_HEIGHT, 3, 1)
    sx, sy = bw / rw, bh / REC_HEIGHT
    for v in 1:REC_HEIGHT, u in 1:rw
        # Pixel centres: output pixel u covers the box's (u-1)sx … u sx.
        px = bilinear(img, bx0 + (u - 0.5f0) * sx + 0.5f0, by0 + (v - 0.5f0) * sy + 0.5f0)
        for c in 1:3
            out[u, v, c, 1] = (px[c] - REC_MEAN) / REC_STD
        end
    end
    return out, rw
end

"""
    recognize(m, img, box) -> TextLine

Read the line inside `box` on `img[x, y]`. CTC decoding: the most likely class
per time step, repeats merged, blanks dropped. Each kept character remembers its
time step, and the steps are spread evenly over the input, which is what gives
the words their positions.
"""
function recognize(m::PPOCR, img::AbstractMatrix, box::NTuple{4, Float32})
    host, rw = line_input(img, box)
    W = size(host, 1)
    input = get!(() -> KA.allocate(m.backend, Float32, W, REC_HEIGHT, 3, 1), m.recinputs, W)
    copyto!(input, host)
    best, index = DNNKernels.call(m.rec, "ppocrv6_rec", input; dims = (w = W ÷ 16,))
    best, index = vec(Array(best)), vec(Array(index))
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
    read_lines(m, img; kw...) -> Vector{TextLine}

Every line of text on `img[x, y]` that reads as something, in reading order.
`kw` go to [`detect`](@ref).
"""
function read_lines(m::PPOCR, img::AbstractMatrix; kw...)
    boxes=detect(m,img;kw...)
    lines=TextLine[]
    # Keep the device-owning Julia thread available for preview events between
    # network calls. The caller still serializes access to the model buffers.
    yield()
    for box in boxes
        line=recognize(m,img,box)
        isempty(line.text) || push!(lines,line)
        yield()
    end
    return lines
end

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
                page = fill(RGB{Float32}(0.95f0, 0.95f0, 0.95f0), 620, 877)
                for y in 200:212, x in 100:500
                    page[x, y] = RGB{Float32}(0.1f0, 0.1f0, 0.1f0)
                end
                read_lines(m, page)
                orientation(m, page)
                for W in REC_WIDTHS
                    recognize(m, page, (100.0f0, 196.0f0, Float32(100 + W * 16 / REC_HEIGHT - 2), 212.0f0))
                end
                KA.synchronize(backend)
            end
        catch err
            @warn "PPOCRRunner: workload skipped; first use will compile" exception = err
        end
    else
        @info "PPOCRRunner: no export installed — nothing precompiled"
    end
end

end # module
