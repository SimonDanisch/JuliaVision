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

# The third conversion rule, and the one that had a throw path in a KERNEL.
#
# `Bool <: Integer` in Julia, so an `Int64 -> Bool` cast fell through to
# `convert`, which raises `InexactError` for anything but 0 and 1 — and torch's
# `.to(torch.bool)` is `v != 0` for every source type, so `convert` was both the
# wrong answer and a `gpu_gc_pool_alloc` in the shader. Lava's
# `replace_unreachable!` warned about exactly that on kokorotext, whose BERT half
# casts an all-ones `Int64` mask: the path was never taken, so no number was
# ever wrong and nothing failed.
#
# Asserted on `castfn`'s CHOICE as well as on the values, because the defect was
# a rule selecting the wrong callable and not a callable computing the wrong
# thing. And on both routes reading the same rule: `runop!` and the emit each
# carried a copy of the `if`.
@testset "castfn is torch's cast for each destination" begin
    cf = DNNKernels.castfn
    # A Bool destination is a comparison against zero, from either family.
    @test cf(Bool, Int64) === DNNKernels.ToBool()
    @test cf(Bool, Float32) === DNNKernels.ToBool()
    @test cf(Bool, Bool) === DNNKernels.ToBool()
    @test cf(Bool, Int64)(2) === true            # `convert` throws here
    @test cf(Bool, Int64)(-1) === true
    @test cf(Bool, Int64)(0) === false
    @test cf(Bool, Float32)(0.5f0) === true      # `trunc` answers `false` here
    @test cf(Bool, Float32)(-3.7f0) === true
    @test cf(Bool, Float32)(0.0f0) === false
    # A float truncated to an integer saturates.
    @test cf(Int32, Float32) === DNNKernels.SafeTrunc{Int32}()
    @test cf(Int32, Float32)(Inf32) === typemax(Int32)
    # Everything else converts, including integer to integer.
    @test cf(Float16, Float32) === DNNKernels.ToType{Float16}()
    @test cf(Int64, Int32) === DNNKernels.ToType{Int64}()
    @test cf(ComplexF32, Float32) === DNNKernels.ToType{ComplexF32}()
    # Four methods over two axes: the pair that overlaps has to resolve rather
    # than be ambiguous, and an ambiguity inside a kernel reads as "method
    # lookup failure" with nothing about dispatch in it.
    @test length(methods(cf)) == 4
    for T in (Bool, Int32, Int64, Float16, Float32), S in (Bool, Int32, Float32)
        # Resolves to exactly one method, and the callable it picks lands in the
        # destination type. An ambiguity throws at the `cf` call.
        @test length(Base.methods(cf, Tuple{Type{T},Type{S}})) == 1
        @test cf(T, S)(one(S)) isa T
    end
    # Every one of them still costs no kernel slot.
    for f in (DNNKernels.ToBool(), DNNKernels.ToType{Float16}(),
              DNNKernels.SafeTrunc{Int32}())
        @test isbitstype(typeof(f))
        @test sizeof(f) == 0
    end
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
