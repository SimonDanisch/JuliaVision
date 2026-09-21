"""
A unary op on an INTEGER operand does not reach for `Float64`.

Every float-valued entry in `UNARY_FUSED` — `sqrt`, `log`, `exp`, `tanh`, `sin`,
`cos`, `inv`, `rsqrt_` — promotes an integer argument through `float(x)`, and
`float(::Int64)` is `Float64`. A GPU kernel handed that computes in double
precision. Where the device has no `double` it does not compile at all: Kokoro
takes `rsqrt` of an `Int64` token count, and the failure is an `InvalidIRError`,
"unsupported use of double value", pointing at `math.jl` rather than at anything
naming a model. Where the device does have one it silently pays for a precision
the destination cannot hold.

Torch gives such an op the RESULT dtype, which is what converting the operand
first does, so `infloat` is about being right as much as about compiling.
"""

using Test, DNNKernels

const DKU = DNNKernels

@testset "infloat picks the destination's float type" begin
    # An integer operand with a float destination is the case.
    f = DKU.infloat(sqrt, Float32, Int64)
    @test f isa DKU.InFloat
    @test f(4) === 2.0f0                      # Float32, NOT Float64
    @test typeof(f(4)) === Float32
    @test DKU.infloat(DKU.rsqrt_, Float32, Int32)(4) === 0.5f0
    @test typeof(DKU.infloat(sqrt, Float16, Int64)(4)) === Float16

    # …and nothing else is. A float operand already computes in its own type,
    # and wrapping it would be a needless second kernel type.
    @test DKU.infloat(sqrt, Float32, Float32) === sqrt
    @test DKU.infloat(sqrt, Float32, Float16) === sqrt
    @test DKU.infloat(-, Int64, Int64) === (-)
    # `Bool` is excluded on purpose: `!` is in the same table and `!(1.0f0)` is
    # not a function.
    @test DKU.infloat(!, Bool, Bool) === (!)
    @test DKU.infloat(!, Float32, Bool) === (!)
end

@testset "InFloat is one type per (T, f)" begin
    # The reason it is a named callable and not `x -> f(T(x))`: an anonymous
    # function gets a fresh type per definition site, so the same arithmetic
    # would compile and freeze as many kernels as there are places that wrote it.
    @test typeof(DKU.InFloat{Float32}(sqrt)) === typeof(DKU.InFloat{Float32}(sqrt))
    @test typeof(DKU.InFloat{Float32}(sqrt)) !== typeof(DKU.InFloat{Float16}(sqrt))
    @test typeof(DKU.InFloat{Float32}(sqrt)) !== typeof(DKU.InFloat{Float32}(log))
    # And that the parameters are in the order `InFloat{T}` needs: with them the
    # other way round this built `InFloat{typeof(sqrt),typeof(sqrt)}`, whose call
    # is `sqrt(sqrt(x))` — a `MethodError` that on a GPU appears as a throw the
    # compiler cannot emit rather than as an error anybody can read.
    @test DKU.InFloat{Float32}(sqrt)(9) === 3.0f0
end
