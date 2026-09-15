# The one-kernel norm path, and the predicate that unlocks it.
#
# `native_layer_norm` and `_fused_rms_norm` each have two forms: one kernel that
# reads its tensor once, and a fallback written as an expression that is six
# passes over it. The gate between them is a LAYOUT question — the kernel indexes
# linearly over a dense buffer, so a lazy `Broadcasted`, a `SubArray` or a
# permuted view has to go the other way, and so does a host `Array`, because the
# kernel is `cpu=false`.
#
# It was spelled `a isa Mantle.LavaArray` until 2026-09-14, which is that
# predicate with one backend's concrete type standing in for it. The cost was
# invisible on the backend it named and large everywhere else: on AMDGPU a
# `ROCArray` failed the test, so SAM 2.1's encoder replayed its 96 layer norms as
# 854 graph passes instead of 96 — 8.9 launches apiece for a form whose whole
# point is to be one — and the encoder ran 459 ms against 335 once the gate
# admitted them. It also moved the encoder CLOSER to PyTorch, `add_129` from
# 4.244 to 1.398, because the fallback's accumulation is not the kernel's.
#
# So what is pinned here is the predicate itself, against whatever array the
# backend under test hands out, plus the kernel's answer on that backend.

using Test, DNNKernels, Mantle, KernelAbstractions
import GPUArrays

@testset "the norm gate admits this backend's arrays and no wrapper" begin
    DK = DNNKernels
    backend = Mantle.LavaBackend()
    x = DK.toback(backend, reshape(collect(Float32, 1:32), 8, 4))
    # What the gate must accept: this backend's dense array.
    @test x isa GPUArrays.AbstractGPUArray
    # …and what it must still refuse, because their linear order is not the
    # buffer's. These are the reason the fallback exists.
    @test !(view(x, 1:4, :) isa GPUArrays.AbstractGPUArray)
    @test !(PermutedDimsArray(x, (2, 1)) isa GPUArrays.AbstractGPUArray)
    @test !(Broadcast.broadcasted(+, x, 1f0) isa GPUArrays.AbstractGPUArray)
    # A host array is refused for a different reason: `layernorm_kernel!` is
    # `cpu=false`, so there is no method to launch.
    @test !(Array(x) isa GPUArrays.AbstractGPUArray)
end

@testset "the one-kernel layer norm computes what the expression does" begin
    DK = DNNKernels
    backend = Mantle.LavaBackend()
    ctx = (; backend)                      # `layernorm!` asks for nothing else
    C, groups = 12, 5
    h = reshape(Float32.(1:(C * groups)) .* 0.25f0 .- 3.0f0, C, groups)
    γh = Float32.(1:C) ./ C
    βh = Float32.(C:-1:1) ./ 2C
    for (γ, β) in ((nothing, nothing), (γh, nothing), (γh, βh))
        a = DK.toback(backend, h)
        out = similar(a)
        μ = DK.toback(backend, zeros(Float32, groups))
        r = DK.toback(backend, zeros(Float32, groups))
        DK.layernorm!(ctx, out, μ, r, a,
                      γ === nothing ? nothing : DK.toback(backend, γ),
                      β === nothing ? nothing : DK.toback(backend, β),
                      C, 1.0f-5)
        # The reference is the fallback's arithmetic, in the same accumulator.
        m = vec(sum(h; dims = 1) ./ C)
        v = vec(sum(abs2, h .- reshape(m, 1, groups); dims = 1) ./ C)
        want = (h .- reshape(m, 1, groups)) ./ reshape(sqrt.(v .+ 1.0f-5), 1, groups)
        γ === nothing || (want = want .* γ)
        β === nothing || (want = want .+ β)
        @test Array(out) ≈ want rtol = 1.0f-5
        @test Array(μ) ≈ m rtol = 1.0f-5
        @test Array(r) ≈ 1.0f0 ./ sqrt.(v .+ 1.0f-5) rtol = 1.0f-5
    end
end
