# The float-to-integer conversions, and why they are not `trunc`/`floor`.
#
# `safetrunc` and `fastfloor` end in `unsafe_trunc`, which has no `InexactError`
# branch. That is a performance and a portability property rather than a
# numerical one: a throw inside a GPU kernel has to ALLOCATE its exception, and
# on a device that allocation is a hostcall — AMDGPU.jl reports "Global
# hostcalls detected" and keeps a host thread servicing `malloc` for the rest of
# the session, while Lava's `replace_unreachable!` warns about lowering the same
# dead path. SAM 2.1's encoder had four of them: `safetrunc` reached through a
# dynamic dispatch (its target type travelled as a broadcast argument, so on
# AMDGPU it arrived as an untyped `Ref`), and three conversions whose own guards
# had already made the throw unreachable.
#
# Dropping the branch is only safe if the guards really do cover everything
# `trunc` would have raised on, which is what this pins: every in-range value
# agrees with the throwing version, and each out-of-range one saturates the way
# torch does instead of raising.

using Test
import DNNKernels
const ST = DNNKernels.safetrunc

@testset "safetrunc saturates where trunc would throw" begin
    for T in (Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64)
        @test ST(T, NaN32) == 0
        @test ST(T, NaN) == 0
        @test ST(T, Inf32) == typemax(T)
        @test ST(T, -Inf32) == typemin(T)
        @test ST(T, Float32(typemax(T))) == typemax(T)
        @test ST(T, Float32(typemin(T))) == typemin(T)
        # Past the ends, where `trunc` raises.
        @test ST(T, 1.0f30) == typemax(T)
        @test ST(T, -1.0f30) == typemin(T)
        # …and in range, where it must agree with it exactly.
        for v in Float32.(-8:0.5:8)
            typemin(T) < v < typemax(T) || continue
            @test ST(T, v) == trunc(T, v)
        end
    end
    # The callable form is what a broadcast gets, so that the target type is a
    # type PARAMETER and the kernel needs no dynamic dispatch to find it.
    @test DNNKernels.SafeTrunc{Int32}()(2.7f0) === Int32(2)
    @test DNNKernels.SafeTrunc{Int32}()(Inf32) === typemax(Int32)
    @test isbitstype(typeof(DNNKernels.SafeTrunc{Int32}()))
    @test sizeof(DNNKernels.SafeTrunc{Int32}()) == 0
end

@testset "fastfloor agrees with floor where the caller has bounded it" begin
    for v in Float32.(-8:0.25:8)
        @test DNNKernels.fastfloor(v) == floor(Int, v)
    end
    @test DNNKernels.fastfloor(0.0f0) == 0
    @test DNNKernels.fastfloor(-0.0f0) == 0
end

# `clamp.default` keeps its bounds in the operand's own type when that type is an
# integer, so the store is not a `convert(Int, ::Float32)` with a throw path.
# Float bounds on an integer operand are a legitimate call the Wan VAE makes and
# must stay promoting.
@testset "clamp bounds stay in the operand's type only when integral" begin
    # `NamedTuple()` is the `dims` a graph with no symbolic extents has; the
    # bounds themselves are the two `nothing`s. This passed `nothing` for dims,
    # which the signature has never accepted — `numattr(dims, …)` needs the
    # table to resolve a bound named by a symbol.
    cb = DNNKernels.clampbounds
    @test cb(Int32, NamedTuple(), nothing, nothing) ===
          (typemin(Int32), typemax(Int32))
    @test cb(Float32, NamedTuple(), nothing, nothing) === (-Inf32, Inf32)
    # And a float bound on an integer operand stays promoting: `clamp(::Int,
    # 0.25, …)` is a legitimate call the Wan VAE makes.
    @test cb(Int32, NamedTuple(), 0.25, 4) === (0.25f0, 4.0f0)
    @test cb(Int32, NamedTuple(), 0, 4) === (Int32(0), Int32(4))
end
