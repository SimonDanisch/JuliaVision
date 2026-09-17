"""
Every ported model's forward pass, on this card, through one harness.

    julia --project=. tools/bench_all.jl            # all of them
    julia --project=. tools/bench_all.jl sam2 rife  # a subset

The other half is `tools/baseline_all.py`, which times the same models in
PyTorch. Neither is meaningful without the other: "≥ PyTorch" is a target, and a
target needs a denominator measured the same way on the same card (GUARDRAILS §5).

**Why one file instead of the per-model `bench_*.jl` tools.** Those measure
different things in different ways — some `@elapsed` a cold call, some take a
median, none warm the clock. This card idles at **210 MHz of 3105** and ramps
only under sustained load, so a cold measurement reads several times slow and
looks exactly like a regression. `tools/measure.jl` exists for that: it warms to
a measured plateau, brackets every sample with a clock reading and discards any
sample that dipped. Everything here goes through it, so the rows are comparable
to each other as well as to PyTorch.

Each entry is `name => (setup, call)`. `setup` returns whatever `call` needs and
runs once, outside the timed region — model loading, weight upload and the first
compile are not what this measures.

## Measured 2026-09-17, `Radeon 8060S (RADV STRIX_HALO)`

The first rows this file has ever produced on this machine. It could not load,
for two reasons: a bare `LavaBackend()` has been an `UndefVarError` since the
name moved to Mantle, and `measure.jl` underneath it read the clock through
`nvidia-smi` on an AMD card, so `report()` threw ENOENT before the first
sample. Both are fixed;
see `measure.jl`'s telemetry section for the second.

    sam2-encode      1035.26 ms  ± 2.6%   clock 99%   11/11   1024x1024
    whisper-encode   1261.48 ms  ± 4.6%   clock 98%   11/11   30 s window
    matanyone-step    273.80 ms  ± 4.6%   clock 93%   11/11   512x288
    depthanything     179.44 ms  ± 6.8%   clock 96%   11/11   518x518
    neurallut           6.21 ms  ±25.9%   clock 77%   11/11   256x256
    rife              838.35 ms  ± 2.4%   clock 97%   11/11   1920x1152

Every row gated: 11 of 11 samples kept, at 93-99% of the plateau. `neurallut` is
the exception its own comment predicts: 6 ms of mostly host work, so the card
idles inside the sample, the clock holds at 77% and the spread is 26%. Its number
is worth less than the others and it is not a broken row.

These are a BASELINE and not a comparison: there was nothing to compare against,
because this file has not run here before. They were taken after `folddims!`
gained a `post` map and `_to_copy` gained `castfn`, both of which every model
reads, so a future run has a number to be measured against.
"""

using Printf, Statistics
include(joinpath(@__DIR__, "measure.jl"))

using Lava, DNNKernels, KernelAbstractions
using DNNKernels: readsafetensors, toback
# `LavaBackend` from MANTLE, which owns it. It used to be Lava's and Lava does
# not re-export it, so a bare `LavaBackend()` here has been an `UndefVarError`
# since the move, and this file could not load at all. RayDemo had the same
# break and 35 of the one-off tools beside this one still do.
import Mantle
using Mantle: LavaBackend
# All at top level: `using` inside a function body is a syntax error, so the
# per-model closures below cannot load their own package. Every runner is
# precompiled, so this costs startup time and nothing else.
using SAM2Runner, WhisperRunner, KokoroRunner, MatAnyoneRunner
using DepthAnythingRunner, NeuralLUTRunner, RIFERunner
const KA = KernelAbstractions
const BACKEND = LavaBackend()
sync() = KA.synchronize(BACKEND)

# `N0f8` via Colors, which re-exports it — FixedPointNumbers is not a direct
# dependency of this project and adding one for a synthetic test frame is not
# worth it.
using Colors
cl(x) = clamp(x, 0.0, 1.0)          # N0f8 is [0,1]; an unclamped sine leaves it
rgbframe(w, h) = [RGB{Colors.N0f8}(cl(0.3 + 0.4sin(i / 13)), 0.5, cl(0.4 + 0.3cos(j / 17)))
                  for i in 1:w, j in 1:h]

# References live in `gen/graphs/`, NOT in the artifacts — the tarballs ship what
# a caller needs to RUN a model and leave developer material out. Benchmarking
# wants the real input where it exists, so look there and fall back to synthetic.
const GEN = joinpath(@__DIR__, "..", "gen", "graphs")
function refinput(dir, key, fallback)
    p = joinpath(GEN, dir, "refs.safetensors")
    isfile(p) || return fallback()
    r = readsafetensors(p)
    haskey(r, key) ? r[key] : fallback()
end

# ── the models ───────────────────────────────────────────────────────────────
#
# `nothing` for a model whose package is present but whose port is not: an entry
# that cannot run must say so rather than be quietly absent, or the table reads
# as "everything we have" when it is "everything that happened to load".

const MODELS = Dict{String,Any}(
    "sam2-encode" => () -> begin
        d = SAM2Runner.assetdir()
        sam = SAM2Runner.defaultmodel(; backend = BACKEND)
        img = toback(BACKEND, refinput("sam2-large", "sam2_encoder/in0",
                                       () -> zeros(Float32, 1024, 1024, 3, 1)))
        (() -> SAM2Runner.encode(sam, img), "1024x1024")
    end,

    "whisper-encode" => () -> begin
        m = WhisperRunner.whispermodel(; backend = BACKEND)
        mel = toback(BACKEND, refinput("whisper-fp16", "whisper/in0",
                                       () -> zeros(Float32, 3000, 128, 1)))
        (() -> WhisperRunner.encode(m, mel), "30 s window")
    end,

    "kokoro-speak" => () -> begin
        k = KokoroRunner.Kokoro(; backend = BACKEND)
        txt = "The quick brown fox jumps over the lazy dog."
        (() -> KokoroRunner.speak(k, txt; noise = false), "one sentence")
    end,

    "matanyone-step" => () -> begin
        m = MatAnyoneRunner.matanyonemodel(; backend = BACKEND)
        W, H = 512, 288
        img = toback(BACKEND, fill(0.5f0, W, H, 3, 1))
        host = zeros(Float32, W, H); host[(W÷4):(3W÷4), (H÷4):(3H÷4)] .= 255.0f0
        msk = toback(BACKEND, host)
        (() -> MatAnyoneRunner.runmatanyone(m, img, msk; warmup = 0), "512x288")
    end,

    "depthanything" => () -> begin
        m = DepthAnythingRunner.depthanything(; backend = BACKEND)
        img = rgbframe(518, 518)
        (() -> DepthAnythingRunner.depthmap!(m, img), "518x518")
    end,

    "neurallut" => () -> begin
        m = NeuralLUTRunner.neurallut(; backend = BACKEND)
        img = rgbframe(256, 256)   # host RGB{N0f8}, what predictlut declares
        (() -> NeuralLUTRunner.predictlut(m, img), "256x256 classifier")
    end,

    "rife" => () -> begin
        m = RIFERunner.rife(; backend = BACKEND)
        w, h = RIFERunner.framesize(m)
        a, b = rgbframe(w, h), rgbframe(w, h)
        out = similar(a)
        (() -> RIFERunner.interpolate!(out, m, a, b; t = 0.5), "$(w)x$(h)")
    end,
)

const ORDER = ["sam2-encode", "whisper-encode", "kokoro-speak", "matanyone-step",
               "depthanything", "neurallut", "rife"]

want = isempty(ARGS) ? ORDER : ARGS
report()
results = Dict{String,Any}()
for name in want
    haskey(MODELS, name) || (@warn "no such model" name; continue)
    print(rpad(name, 18)); flush(stdout)
    try
        call, shape = Base.invokelatest(MODELS[name])
        f = () -> (call(); sync())
        plat, _ = plateau(f; seconds = 4)              # warm, and report what it reached
        r = bench(f; sync = nothing, plat, samples = 11, label = name)
        # A row where the clock gate rejected everything used to print `NaN`,
        # which reads as a broken model. It is not: `neurallut` is host-bound, so
        # the GPU idles inside its own sample and never holds the clock however
        # long the sample is made. Print the UNGATED median with a `!` — the
        # number is worth less than a gated one and is worth more than nothing,
        # and the marker is what says which it is.
        ms = r.kept > 0 ? r.median * 1000 :
             median([s.seconds for s in r.samples]) * 1000
        results[name] = (; ms, spread = r.spread, shape, kept = r.kept, clock = r.clock)
        @printf("%9.2f ms%s ±%4.1f%%  clock %3.0f%%  %d/%d  (%s)\n",
                ms, r.kept > 0 ? " " : "!", 100r.spread, 100r.clock,
                r.kept, r.kept + r.rejected, shape)
    catch e
        println("FAILED: ", first(split(sprint(showerror, e), '\n'))[1:min(end, 90)])
    end
    GC.gc(true); Mantle.trim_gpu_pool!()
end

println("\n── summary (median, warm) ──")
for n in want
    haskey(results, n) || continue
    r = results[n]
    # `!` carries through to the summary: a number the clock gate rejected is
    # still a number, and hiding which rows those are is how a summary lies.
    @printf("%-18s %9.2f ms%s  %s\n", n, r.ms, r.kept > 0 ? " " : "!", r.shape)
end
