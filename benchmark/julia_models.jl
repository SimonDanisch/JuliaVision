# The Julia half of the suite: one forward pass per model, built from
# `models.toml` and asserted against it.
#
# Every `build` returns `(call, meta)`. `call` takes no arguments and runs one
# forward; everything it needs is constructed here, OUTSIDE the timed region —
# model loading, weight upload and the first compile are not what this measures.
#
# **Each setup asserts the parameters rather than deriving them.** `framesize`
# and a model's own image size are readable here, so it is tempting to let the
# model decide the shape and report whatever it chose. That is how the two sides
# drift: PyTorch would decide separately and the ratio between them would be
# between two different programs. The spec is authoritative and a model that
# disagrees with it is an error, not a row.

using Printf
import TOML
using Colors                    # re-exports N0f8; FixedPointNumbers is not a direct dep
using KernelAbstractions
import Mantle
using Mantle: LavaBackend
using DNNKernels: toback

# All at top level: `using` inside a function body is a syntax error, so the
# per-model builders below cannot load their own package.
using SAM2Runner, WhisperRunner, KokoroRunner, MatAnyoneRunner
using DepthAnythingRunner, NeuralLUTRunner, RIFERunner

const KA = KernelAbstractions
const SPEC = TOML.parsefile(joinpath(@__DIR__, "models.toml"))

"""Deterministic host RGB frame.

Synthetic and not the recorded reference inputs, because the PyTorch side has no
way to read a `safetensors` reference for models whose export does not ship one,
and a benchmark where the two sides feed different data is a benchmark of the
data. None of these models branch on pixel values, so content changes nothing
that is being measured; `sin`/`cos` rather than `rand` so a rerun is the same
frame.
"""
cl(x) = clamp(x, 0.0, 1.0)
rgbframe(w, h) = [RGB{Colors.N0f8}(cl(0.3 + 0.4sin(i / 13)), 0.5, cl(0.4 + 0.3cos(j / 17)))
                  for i in 1:w, j in 1:h]

params(name) = SPEC["models"][name]["params"]

function meta(name)
    m = SPEC["models"][name]
    (; dtype = m["dtype"], shape = m["shape"], note = m["note"], runner = m["runner"])
end

# ── the models ───────────────────────────────────────────────────────────────

function build_sam2(backend)
    p = params("sam2-encode")
    sam = SAM2Runner.defaultmodel(; backend)
    res = p["resolution"]
    img = toback(backend, zeros(Float32, res, res, 3, 1))
    (() -> SAM2Runner.encode(sam, img)), meta("sam2-encode")
end

function build_whisper(backend)
    p = params("whisper-encode")
    m = WhisperRunner.whispermodel(; backend)
    # (frames, bins, batch) — a LavaArray's dimensions are the reverse of torch's.
    mel = toback(backend, zeros(Float32, p["mel_frames"], p["mel_bins"], 1))
    (() -> WhisperRunner.encode(m, mel)), meta("whisper-encode")
end

function build_kokoro(backend)
    p = params("kokoro-speak")
    k = KokoroRunner.Kokoro(; backend)
    ph = p["phonemes"]
    # The token count is the number that silently differed between the two sides
    # before this file existed, so it is checked and not reported.
    n = length(KokoroRunner.tokenize(k, ph))
    n == p["tokens"] || error("kokoro-speak: spec says $(p["tokens"]) tokens, \
                               tokenising the spec's phonemes gives $n")
    # `phonemes` and not `text`: our own G2P is a different program from misaki's
    # and timing it against PyTorch's would compare the two phonemisers.
    # `noise = false` so a rerun is the same waveform.
    (() -> KokoroRunner.speak(k; phonemes = ph, voice = p["voice"], noise = false)),
    meta("kokoro-speak")
end

function build_matanyone(backend)
    p = params("matanyone-step")
    m = MatAnyoneRunner.matanyonemodel(; backend)
    W, H = p["width"], p["height"]
    img = toback(backend, fill(0.5f0, W, H, 3, 1))
    host = zeros(Float32, W, H)
    host[(W ÷ 4):(3W ÷ 4), (H ÷ 4):(3H ÷ 4)] .= 255.0f0
    msk = toback(backend, host)
    # `warmup = 0`: the timed region is one step, and the runner's own warm-up
    # would put several inside it.
    (() -> MatAnyoneRunner.runmatanyone(m, img, msk; warmup = 0)), meta("matanyone-step")
end

function build_depthanything(backend)
    p = params("depthanything")
    m = DepthAnythingRunner.depthanything(; backend)
    img = rgbframe(p["width"], p["height"])
    (() -> DepthAnythingRunner.depthmap!(m, img)), meta("depthanything")
end

function build_neurallut(backend)
    p = params("neurallut")
    m = NeuralLUTRunner.neurallut(; backend)
    img = rgbframe(p["width"], p["height"])   # host RGB{N0f8}, what predictlut declares
    (() -> NeuralLUTRunner.predictlut(m, img)), meta("neurallut")
end

function build_rife(backend)
    p = params("rife")
    m = RIFERunner.rife(; backend)
    w, h = RIFERunner.framesize(m)            # the PADDED size the export was built for
    w == p["width"] && h == p["height"] ||
        error("rife: spec says $(p["width"])x$(p["height"]), the installed export \
               is padded to $(w)x$(h)")
    a, b = rgbframe(w, h), rgbframe(w, h)
    out = similar(a)
    (() -> RIFERunner.interpolate!(out, m, a, b; t = p["timestep"])), meta("rife")
end

const BUILDERS = Dict{String,Function}(
    "sam2-encode"     => build_sam2,
    "whisper-encode"  => build_whisper,
    "kokoro-speak"    => build_kokoro,
    "matanyone-step"  => build_matanyone,
    "depthanything"   => build_depthanything,
    "neurallut"       => build_neurallut,
    "rife"            => build_rife,
)

# Spec order, so the table reads the same way every run and a model added to the
# spec cannot be silently absent from the Julia side.
const ORDER = collect(keys(SPEC["models"]))
sort!(ORDER, by = n -> something(findfirst(==(n),
        ["sam2-encode", "whisper-encode", "kokoro-speak", "matanyone-step",
         "depthanything", "neurallut", "rife"]), typemax(Int)))

for n in ORDER
    haskey(BUILDERS, n) ||
        error("models.toml declares `$n` and julia_models.jl has no builder for it")
end
