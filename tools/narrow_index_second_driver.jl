"""
The narrow-index miscompile, on whichever Vulkan driver is selected.

    julia --project=. tools/narrow_index_second_driver.jl                    # NVIDIA
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \\
        julia --project=. tools/narrow_index_second_driver.jl                # lavapipe

This is the one question left about `test_int32_cartesian_miscompile.jl` that
reading the disassembly cannot answer. Every value in the failing module has been
checked and is correct; the address computation is instruction-for-instruction
identical to the working one and reads the same struct offsets. So either the
module is subtly wrong in a way inspection keeps missing, or it is right and the
driver compiles it wrongly — and a second, unrelated implementation of Vulkan
distinguishes those two.

  * lavapipe **exact** -> the module is fine, and NVIDIA's shader compiler is
    the fault. That makes it reportable upstream, with a 140-line reproducer.
  * lavapipe **wrong too** -> the module is wrong, inspection has been missing
    it, and the search goes back to the emitter.

The reproducer is the minimal one: `identity` over a plain rank-3 array,
`Broadcast.preprocess` applied, narrow index. Rank 2 runs alongside as a control,
because it must be exact everywhere — if it fails here, the driver is not
comparable and the result means nothing.
"""

using Lava, KernelAbstractions
const KA = KernelAbstractions

@kernel function narrow_index!(dest, bc, cis, n)
    I = @index(Global, Linear)
    if I <= n
        J = @inbounds cis[I % Int32]
        @inbounds dest[I] = bc[J]
    end
end

function exact(be, sz)
    h = reshape(collect(1f0:prod(sz)), sz)
    n = prod(sz)
    A = KA.allocate(be, Float32, sz...); copyto!(A, h)
    dummy = KA.allocate(be, Float32, sz...)
    bc = Broadcast.preprocess(dummy, Broadcast.instantiate(
             Broadcast.broadcasted(identity, A)))
    out = KA.allocate(be, Float32, n); fill!(out, 0f0)
    narrow_index!(be, 64)(out, bc, CartesianIndices(sz), n; ndrange = n)
    KA.synchronize(be)
    reshape(Array(out), sz) ≈ h
end

if abspath(PROGRAM_FILE) == @__FILE__
    be = LavaBackend()
    println("device: ", Lava.vk_context().device_name)
    r2 = exact(be, (5, 5))
    r3 = exact(be, (5, 5, 3))
    println("  rank 2 (control, must be exact) : ", r2 ? "exact" : "WRONG")
    println("  rank 3 (the bug)                : ", r3 ? "exact" : "WRONG")
    println()
    # Report only what this run shows. The conclusion comes from comparing two
    # drivers, not from either alone, and a version that guessed which one it was
    # on printed the wrong verdict on the first driver it met.
    println(!r2 ? "control FAILED — this driver is not comparable; the rank-3 result means nothing" :
            "rank 3 is $(r3 ? "exact" : "WRONG") here. Run the other ICD and compare:\n" *
            "  exact on one and WRONG on the other  => the module is fine, that driver miscompiles it\n" *
            "  WRONG on both                        => the module itself is wrong")
    println("\nRecorded 2026-07-31: NVIDIA RTX 4000 Ada WRONG, lavapipe (llvmpipe) exact.")
end
