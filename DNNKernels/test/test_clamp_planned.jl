using Test, DNNKernels, Lava, KernelAbstractions
const KA = KernelAbstractions

# `clamp.default` shipped as a bare dotted broadcast:
#
#     clamp.(lhs(ctx, op), Float32(lo), Float32(hi))
#
# which allocates its output through `similar` — outside the static plan — and,
# because the bounds are `Float32`, PROMOTES: `clamp.(::Float16, ::Float32,
# ::Float32)` has eltype `Float32`, so an fp16 op wrote an unplanned fp32 buffer
# and needed a second pass to come back. `opdouble` on Whisper's encode priced it
# at 4.22 ms against 0.29 ms for `mul.Tensor` over the same 32 calls, the same
# `(1500, 1280)` fp16 shape and the same 245.8 MB — 58 GB/s against 845.
#
# THIS FILE DOES NOT GUARD `runop!`, and saying so is the point. The bug was
# invisible to values (`coerce` converted the promoted buffer back, so results
# were correct and the suite was green for as long as it shipped), and a
# source-text audit of the 75 `runop!` bodies turned out not to be a clean
# invariant either: roughly a dozen methods legitimately do not take a planned
# destination — reductions build their own result, `rand`/`randn_like` generate
# one, `_assert_tensor_metadata` has none. An allowlist over those would rot.
#
# So what is left is a backstop on the elementwise BACKEND: clamping must cost
# about what multiplying costs at the same shape. It times the broadcasts
# directly, so it would still pass if `runop!` regressed — it is here because it
# is the measurement that exposed the bug, and it keeps the clamp kernel honest
# if the backend is ever rewritten.
#
#     clamp.(a, f32, f32)      1186.6 us    10 GB/s   <- the shipped form
#     d .= clamp.(a, f32, f32)    68.9 us   111 GB/s
#     d .= a * 0.125f0            60.1 us   128 GB/s  <- the reference
@testset "a planned clamp costs about what a planned mul costs" begin
    back = LavaBackend()
    a = KA.allocate(back, Float16, 1280, 1500, 1)
    d = KA.allocate(back, Float16, 1280, 1500, 1)
    fill!(a, Float16(0.5))

    function per(f, iters = 100)
        f(); KA.synchronize(back)
        t0 = time_ns()
        for _ in 1:iters; f(); end
        KA.synchronize(back)
        (time_ns() - t0) / iters
    end

    tmul   = per(() -> (d .= a .* Float16(0.125)))
    tclamp = per(() -> (d .= clamp.(a, -1.0f0, 1.0f0)))
    @info "clamp vs mul" mul_us = tmul / 1e3 clamp_us = tclamp / 1e3 ratio = tclamp / tmul
    @test tclamp < 3 * tmul        # the shipped form missed this by 6x

    # The promotion that caused it, stated directly so the reason survives even
    # if the threshold above is ever loosened.
    @test eltype(clamp.(a, -1.0f0, 1.0f0)) === Float32       # Float32 bounds promote
    @test eltype(d .= clamp.(a, -1.0f0, 1.0f0)) === Float16  # a planned dest does not
end
