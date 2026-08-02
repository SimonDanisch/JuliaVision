"""
Which structurization pass turns `cis[I % Int32]` into the wrong answer.

    julia --project=. tools/bisect_narrow_index.jl [PassName]

`test_int32_cartesian_miscompile.jl` has eliminated everything reachable by
reading the emitted SPIR-V: the width, the division, the decomposition
arithmetic and the consumption are all correct or identical between the failing
and the working variant. And it cannot be probed at runtime — storing the index
components is exactly the second consumer that makes the fault vanish.

That leaves bisecting the pipeline. `LAVA_SKIP_PASSES` drops one pass; this
script runs the failing kernel once per pass and reports which of three things
happened:

  * **exact** — dropping that pass fixes it, so that pass introduces the fault;
  * **still wrong** — that pass is not implicated;
  * **broke** — the module no longer compiles or validates without it, i.e. the
    pass is load-bearing and this technique cannot rule it in or out.

One pass per process, because `LAVA_SKIP_PASSES` is read at compile time and the
kernel cache would otherwise serve a module built under a different setting.
"""

using Lava, KernelAbstractions
using Lava: cart32
const KA = KernelAbstractions

const PASSES = ["SimplifyCFG", "fixup_structured_cfg", "LowerSwitch",
                "UnifyFunctionExitNodes", "FixIrreducible", "LoopSimplify",
                "StructurizeCFG", "InstCombine", "merge_equivalent_loop_phis",
                "replace_undef_phi_operands", "fixup_post_structurize"]

@kernel function narrow_bcast!(dest, bc, cis, n)
    I = @index(Global, Linear)
    if I <= n
        J = @inbounds cis[I % Int32]
        @inbounds dest[I] = bc[J]
    end
end

"Run the failing kernel; `true` if it now computes the reference."
function exact()
    be = LavaBackend()
    host = reshape(collect(1f0:105f0), 7, 5, 3)
    A = KA.allocate(be, Float32, 7, 5, 3); copyto!(A, host)
    sz = (5, 5, 3); n = prod(sz)
    want = host[2:6, :, :] .* 3
    dummy = KA.allocate(be, Float32, sz...)
    bc = Broadcast.preprocess(dummy, Broadcast.instantiate(
             Broadcast.broadcasted(*, view(A, 2:6, :, :), 3f0)))
    out = KA.allocate(be, Float32, n); fill!(out, 0f0)
    narrow_bcast!(be, 64)(out, bc, CartesianIndices(sz), n; ndrange = n)
    KA.synchronize(be)
    reshape(Array(out), sz) ≈ want
end

if abspath(PROGRAM_FILE) == @__FILE__
    if isempty(ARGS)
        # Driver: one subprocess per pass, so each gets a clean kernel cache.
        println("dropping each pass in turn; baseline (nothing dropped) must be wrong\n")
        for p in vcat(["<none>"], PASSES)
            cmd = `$(Base.julia_cmd()) --project=$(dirname(@__DIR__)) $(@__FILE__) $p`
            env = p == "<none>" ? ENV : merge(ENV, Dict("LAVA_SKIP_PASSES" => p))
            out = try
                read(pipeline(setenv(cmd, env); stderr = devnull), String)
            catch
                "broke\n"
            end
            println("  ", rpad(p, 28), strip(last(split(strip(out), '\n'))))
        end
    else
        println(exact() ? "exact  <-- this pass introduces the fault" : "still wrong")
    end
end
