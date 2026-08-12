# WHY is the matrix-operand fusion SLOWER? Registers, from the driver.
#
# `flash_cm2_variants.jl` measured the fused softmax +30.8% and +30.9% on two
# shapes a factor of ten apart, interleaved, at identical accuracy. Fewer passes
# over the same data is not supposed to cost more, so the cost is not in the
# passes — the candidate is the register file, which is already at the driver's
# ceiling (255) for this kernel unfused.
#
# (The ablation has since given the other half of the answer: the two ROW
# REDUCTIONS are 38.1% of the kernel against the six per-element passes' 13.5%.
# Fusion cut the cheap half and left the expensive one alone.)
#
# `VK_KHR_pipeline_executable_properties` is the only way to see that without
# Nsight, and it must be enabled BEFORE the device exists.
ENV["DISPLAY"] = get(ENV, "DISPLAY", ":99")
using Lava
Lava.enable_pipeline_executable_properties!()
include(joinpath(@__DIR__, "attn_lab.jl"))
using Printf

const DEV = DK.caps(BACKEND)
const CTX = DK.Ctx(BACKEND; ws = WS)

"Driver statistics for whichever pipelines `setup` newly compiles."
function newpipestats(setup)
    pipes() = Lava.vk_context().caches.pipelines
    before = Set(keys(pipes()))
    DK.reset!(WS); setup(); KA.synchronize(BACKEND)
    fresh = [k for k in keys(pipes()) if !(k in before)]
    [Lava.pipeline_exec_stats(pipes()[k]) for k in fresh]
end

function statline(name, setup)
    for s in newpipestats(setup)
        s === nothing && continue
        vals = join([string(r.name, "=", r.value) for r in s.raw_stats
                     if !(r.value isa Bool)], "  ")
        println(rpad(name, 24), vals)
    end
end

const E, Lq, Lk, H, B = 72, 4096, 4096, 8, 1
q, k, v, out = operands(E, Lq, Lk, H, B)
scale = Float32(1 / sqrt(E))

for (BR, BC, NT) in [(64, 32, 128), (64, 64, 128)]
    plan = DK.flashcm2_plan(DEV, q, k, v, nothing; BR, BC, NT)
    plan isa DK.FlashCM2Plan || continue
    # The same `VARIANTS` the A/B times, so a row here and a row there are the
    # same kernel. `osum` deletes the `Br x Bc` running total as well as a
    # reduction, so its register line is part of the claim.
    for (nm, kw) in VARIANTS
        statline(@sprintf("cm2 %dx%d/%d %s", BR, BC, NT, nm),
                 () -> DK.sdpaflashcm2!(CTX, out, plan, q, k, v, scale; kw...))
    end
end
println("DONE")
