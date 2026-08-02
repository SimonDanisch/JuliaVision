"""
Image-Adaptive 3D LUT: does the apply match upstream, and does it fit the budget?

    julia --project=. tools/bench_neurallut.jl

Two questions, in the order they have to be answered:

  * **parity of the apply** — `GPUFiltering.lut3d!` against
    `reference_apply.safetensors`, which `tools/ref_neurallut_apply.py` produces
    by building and running *upstream's own* `trilinear_kernel.cu`. Not a
    PyTorch reimplementation: a reimplementation would only show that two
    readings of the same file agree. `verify_neurallut.jl` covers the graph.
  * **the editor budget** — the whole per-frame cost at 4K, which is what
    `models-to-port.md` sets at **< 2 ms**. Reported as three parts, because
    they are paid at different rates: the apply is per frame, the classifier is
    per frame only if the look is being re-predicted, and the 4K -> 256 reduction
    the classifier needs is shared with the thumbnail the editor already draws.

**No engine comparison here** — `GUARDRAILS.md` §6: `perf-plan.md`'s numbers are
all desktop measurements and cross-machine numbers do not compare. This prints
this machine's wall clock and says so.

**Clamping is a deliberate difference from upstream.** `topixel` clamps to
[0, 1] for every kernel in `GPUFiltering`, and the predicted LUT overshoots at
both ends (the reference lands at -0.07 .. 1.03). The parity check therefore
compares against a clamped reference, and the raw overshoot is printed so the
size of what is being clamped stays visible rather than assumed small.
"""

using GPUFiltering, KernelAbstractions, Lava, Printf, Statistics, ColorTypes
using DNNKernels: loadgraph, execute!, readsafetensors, toback,
                  Model, planslab, fusableset, Workspace
const KA = KernelAbstractions

const DIR = normpath(joinpath(@__DIR__, "..", "gen", "graphs", "neurallut"))
const RES = 256           # the classifier's fixed input
const UHD = (3840, 2160)  # "4K" as the editor means it

smclock() = try
    parse(Int, first(split(read(`nvidia-smi --query-gpu=clocks.sm --format=csv,noheader`,
                                String))))
catch
    9999
end

"""
Median of `n` samples, each `reps` launches with one sync. Same shape as
`gemm_bench.jl`'s `timedall`: per-launch syncs measure the sync, not the kernel.
"""
function timed(f, backend; n = 15, reps = 20)
    f(); KA.synchronize(backend)
    for _ in 1:3; f(); end; KA.synchronize(backend)
    ts = Float64[]
    for _ in 1:n
        KA.synchronize(backend)
        t0 = time_ns()
        for _ in 1:reps; f(); end
        KA.synchronize(backend)
        push!(ts, (time_ns() - t0) / 1e6 / reps)   # ms per launch
    end
    median(ts)
end

backend = LavaBackend()
isdir(DIR) || error("no export at $DIR — run `uv run tools/export_neurallut.py`")

# ---------------------------------------------------------------- parity
refp = joinpath(DIR, "reference_apply.safetensors")
isfile(refp) || error("no apply reference at $refp — run `uv run tools/ref_neurallut_apply.py`")
ra = readsafetensors(refp)

# torch (1, 3, H, W) -> Julia (W, H, 3, 1); the image is (W, H) of RGB.
himg = ra["img"]
hgraded = ra["graded"]
hlut = ra["lut"]                     # Julia (D, D, D, 3)
W, H = size(himg, 1), size(himg, 2)

src = [RGB{Float32}(himg[x, y, 1, 1], himg[x, y, 2, 1], himg[x, y, 3, 1])
       for x in 1:W, y in 1:H]
dimg = toback(backend, src)
dout = similar(dimg)
dlut = toback(backend, hlut)

# GUARDRAILS §3: prove every element was written, not just that the written ones
# are right. A kernel that covers three quarters of its destination looks like a
# speed-up and passes a value check on the part it did write. The sentinel is a
# value the kernel cannot legitimately produce — `topixel` clamps to [0, 1], so
# a NaN survives only where nothing was stored.
fill!(dout, RGB{Float32}(NaN32, NaN32, NaN32))
KA.synchronize(backend)

lut3d!(dout, dimg, dlut)
KA.synchronize(backend)
got = Array(dout)

unwritten = count(p -> isnan(red(p)) || isnan(green(p)) || isnan(blue(p)), got)
@printf("coverage %d of %d pixels written%s\n", length(got) - unwritten, length(got),
        unwritten == 0 ? "" : "   ** $unwritten UNWRITTEN **")

# The reference is unclamped; `topixel` clamps. Compare against the clamped
# reference and report the overshoot separately.
over_lo = minimum(hgraded)
over_hi = maximum(hgraded)
function maxdiff(got, hgraded, W, H)
    d = 0.0f0
    for y in 1:H, x in 1:W
        p = got[x, y]
        for (c, v) in enumerate((red(p), green(p), blue(p)))
            want = clamp(hgraded[x, y, c, 1], 0.0f0, 1.0f0)
            d = max(d, abs(Float32(v) - want))
        end
    end
    return d
end
Δ = maxdiff(got, hgraded, W, H)
@printf("apply    max|Δ| = %.3e  vs upstream's trilinear_kernel.cu (clamped)\n", Δ)
@printf("         upstream unclamped range %.4f .. %.4f\n", over_lo, over_hi)

ok = Δ < 1e-5 && unwritten == 0
println(ok ? "APPLY PARITY OK" : "APPLY PARITY FAILED")

# ---------------------------------------------------------------- editor budget
# Through `Model`, not `loadgraph`: it runs the host-side preparation passes
# (fold, hoist, dropdead) that the editor's own path gets, and a benchmark that
# skips them measures a graph nothing ships. Measured on this model:
# `loadgraph` + a slab-less `execute!` reported 14.08 ms for this classifier;
# the same graph through `Model` with a planned slab is 1.73 ms. The slab is
# most of that — without one, every intermediate is a fresh allocation.
model = Model(DIR, joinpath(DIR, "weights.safetensors");
              names = ["neurallut"], backend)
graph = model.graphs["neurallut"]
weights = model.weights
gplan = planslab(graph, (;))
gslab = KA.allocate(backend, UInt8, max(gplan.bytes, 1))
gws = Workspace(backend)
glazy = fusableset(graph)

uhd = KA.allocate(backend, RGB{Float32}, UHD...)
uhd .= RGB{Float32}(0.3f0, 0.5f0, 0.7f0)
uhd_out = similar(uhd)
cls_in = KA.allocate(backend, Float32, RES, RES, 3, 1)

# Heat the card: this one idles at 210 MHz of 2265 and a cold clock is worth
# several x. GUARDRAILS §6.
heat = KA.allocate(backend, Float32, 1 << 22)
heat2 = KA.allocate(backend, Float32, 1 << 22)
for _ in 1:25
    for _ in 1:200; heat .= heat2 .* 1.0001f0 .+ 0.5f0; end
    KA.synchronize(backend)
    smclock() >= 1800 && break
end
clk = smclock()

t_apply = timed(() -> lut3d!(uhd_out, uhd, dlut), backend)
# The classifier's input in one step: bilinear down to 256 and de-interleave to
# the planar (256, 256, 3, 1) the graph takes. `bilinearresize!` is Float32-only
# and would need the frame split into channels first, which is the copy this
# avoids.
t_resize = timed(() -> resizeplanar!(cls_in, uhd), backend)
t_graph = timed(backend; n = 7, reps = 3) do
    execute!(graph, Dict{String,Any}("img" => cls_in), weights; dims = (;), backend,
             slab = gslab, plan = gplan, ws = gws, lazy = glazy)
end

println()
println("editor budget at $(UHD[1])x$(UHD[2]) — THIS MACHINE (RTX 3070 laptop, 8 GB), ",
        "SM clock $clk MHz")
println("cross-machine numbers do not compare; perf-plan.md keeps desktop numbers only")
@printf("  LUT apply (per frame)          %8.3f ms   [target < 2.000]\n", t_apply)
@printf("  4K -> 256 reduce (per predict) %8.3f ms\n", t_resize)
@printf("  classifier graph (per predict) %8.3f ms\n", t_graph)
@printf("  ------------------------------------------\n")
@printf("  grade an already-predicted look %7.3f ms\n", t_apply)
@printf("  re-predict and grade            %7.3f ms\n", t_apply + t_resize + t_graph)
println(t_apply < 2.0 ? "APPLY WITHIN BUDGET" : "APPLY OVER BUDGET")

exit(ok ? 0 : 1)
