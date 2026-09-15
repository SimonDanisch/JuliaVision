# What does attention actually cost inside SAM 2's encode, and does the coopmat2
# kernel change it?
#
# This exists because a microbenchmark and a model disagreed. `attn_flash_cm2!`
# is 27-45% faster than `attn_flash_cm!` on the encoder's own attention shapes,
# measured interleaved and checked against a third implementation — and switching
# the whole model between them moved the encode by -0.1%. One of those two
# numbers is not measuring what it claims to, and per-op device time says which.
#
# **`call`, not `encode`.** `encode` runs a baked Mantle plan: the first call
# records command buffers and the rest replay them, so per-op instrumentation
# never fires and a device-caps flip between two `encode`s changes nothing at
# all. `call` runs the graph op by op, which is the only path `optimes` can see.
# It is slower than the plan and that is fine — this measures the SHARE, and both
# columns pay the same overhead.
#
# `optimes` synchronises around every op, so the total exceeds the step's wall
# time. Read the attention row against the other rows of the same column, and the
# columns against each other; never a total against the encode.
ENV["DISPLAY"] = get(ENV, "DISPLAY", ":99")
using DNNKernels, KernelAbstractions, Printf, Lava
using SAM2Runner: SAM2
using DNNKernels: readsafetensors, toback
const KA = KernelAbstractions
const DK = DNNKernels
const DIR = normpath(joinpath(@__DIR__, "..", "gen", "graphs", "sam2-large"))

"`{aten => (calls, ms)}` for one interpreted encode, device time, serialised."
function ourtimes(sam, img; iters = 3)
    m = sam.model
    DK.call(m, "sam2_encoder", img; dims = sam.dims)     # warm
    KA.synchronize(m.backend)
    t = Dict{String,Tuple{Int,Float64}}()
    m.diag.optimes = t
    for _ in 1:iters
        DK.call(m, "sam2_encoder", img; dims = sam.dims)
    end
    KA.synchronize(m.backend)
    m.diag.optimes = nothing
    Dict(k => (v[1] ÷ iters, v[2] / iters) for (k, v) in t)
end

sam = SAM2(Dict(n => DNNKernels.loadgraph(joinpath(DIR, "$n.json"))
                for n in ("sam2_encoder", "sam2_decoder")),
           readsafetensors(joinpath(DIR, "weights.safetensors"));
           backend = LavaBackend(), res = 1024)
img = toback(sam.model.backend,
             readsafetensors(joinpath(DIR, "refs.safetensors"))["sam2_encoder/in0"])

# Same switch the bench uses: empty `wggran` and the planner decides as if this
# card had no workgroup-scope matrices.
full = Lava.caps()
setcm2(on) = (Mantle.vk_context().caches.caps =
                  on ? full : Lava.DeviceCaps(full; wggran = NTuple{4,Int}[]))

rows = Dict{String,Vector{Tuple{Int,Float64}}}()
for on in (false, true)
    setcm2(on)
    for (k, v) in ourtimes(sam, img)
        push!(get!(rows, k, Tuple{Int,Float64}[]), v)
    end
end
setcm2(true)

println("\nper-op device time, one interpreted encode (cm1 = no workgroup scope):")
@printf("%-44s %6s %10s %10s %8s\n", "aten", "calls", "cm1 ms", "cm2 ms", "delta")
tot = [0.0, 0.0]
for (k, v) in sort(collect(rows); by = x -> -(length(x[2]) == 2 ? x[2][1][2] : 0.0))
    length(v) == 2 || continue
    tot .+= [v[1][2], v[2][2]]
    v[1][2] < 0.3 && continue
    @printf("%-44s %6d %10.3f %10.3f %+7.1f%%\n", k, v[1][1], v[1][2], v[2][2],
            100 * (v[2][2] - v[1][2]) / v[1][2])
end
@printf("%-44s %6s %10.3f %10.3f %+7.1f%%\n", "TOTAL (serialised, not the encode)", "",
        tot[1], tot[2], 100 * (tot[2] - tot[1]) / tot[1])
