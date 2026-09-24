"""
Backend-agnostic GPU image processing built on KernelAbstractions.

All functions work on `AbstractMatrix{<:AbstractRGB}` of any KA backend —
`Matrix` (CPU), `LavaArray` (Vulkan), `CuArray`, … — dispatching kernels via
`KernelAbstractions.get_backend(img)`. Computation happens in `RGB{Float32}`
and is stored back in the array's element type.

Kernels are asynchronous; call `KernelAbstractions.synchronize(backend)`
(or fetch results with `Array`) before reading on the host.
"""
module GPUFiltering

using ColorTypes
using FixedPointNumbers
using GeometryBasics
using KernelAbstractions
import KernelAbstractions as KA
import Statistics: median

export coloradjust!, gaussianblur!, unsharpmask!, warp!, gaussianweights
export channellinear!, channelstats
export ColorAdjustments
export cropmatrix, fitmatrix, translationmatrix, areadownscale!
export grayscale!, gradients!, opticalflow!, flowwarp!, bilinearresize!, fitaffine,
       fitsimilarity, ransacsimilarity, ransactranslation, fithomography, FlowWorkspace
export samplewindow!, nccpeak, peakmargin, PatchTracker, matchpatches!
export goodfeatures
export lucaskanade!
export nv12torgb!, lumatogray!
export blend!
export pointwise!, stencil!
export lut3d!
export resizeplanar!
export tofloat, topixel, alphaof, straight, premul, AnyRGB
export paramvalue
export WarpWrite, Replace, Over

"""
    paramvalue(x) -> value

A kernel's scalar parameter, whether it arrived as a constant or as a device ref.

A kernel called directly gets the value; one DECLARED in a graph is packed with
its arguments once, when the plan records, so a value that changes between runs
arrives as a one-element device array and is read out of it here.

Three methods, because "is this a ref" cannot be asked of `AbstractArray` alone:
a parameter is as often a struct (`ColorAdjustments`) as a number — so `Number`
is too narrow — and a `Mat3f` or a `Vec4` passed BY VALUE is an `AbstractArray`
too, so that is too wide. A warp matrix went down the ref path and came back as
its first `Float32`.

A STATIC array is the value case: it carries its size in its type, which is
exactly what a one-element device array a `GPURef` resolves to does not. All
three resolve at compile time, so a kernel written against this costs nothing
either way and stays callable from a test with a plain value.
"""
@inline paramvalue(x::GeometryBasics.StaticArrays.StaticArray) = x
@inline paramvalue(x::AbstractArray) = @inbounds x[1]
@inline paramvalue(x) = x

"""
A pixel these kernels work on: RGB, or RGB with an alpha channel.

WITH ALPHA THE COLOUR IS PREMULTIPLIED — the stored red of a half-covered red
pixel is 0.5, not 1.0. Every kernel here that MIXES pixels (a blur, a warp, a
cross-fade) is a weighted sum, and a weighted sum of premultiplied colours is the
right answer while a weighted sum of unpremultiplied ones is not: the latter pulls
the colour of pixels that are not there into the ones that are, which is the halo
along a matte edge. That is why this is a property of the FORMAT and not a flag on
each operation — an op that just does its arithmetic is already correct.
"""
const AnyRGB = Union{AbstractRGB, TransparentRGB}

"Convert any RGB pixel to Float32 components for in-kernel math."
@inline tofloat(c::AbstractRGB) = RGB{Float32}(Float32(red(c)), Float32(green(c)), Float32(blue(c)))
@inline tofloat(c::TransparentRGB) = RGB{Float32}(Float32(red(c)), Float32(green(c)), Float32(blue(c)))

"A pixel's coverage. One for a format that has no alpha — it is all there."
@inline alphaof(::AbstractRGB) = 1.0f0
@inline alphaof(c::TransparentRGB) = Float32(alpha(c))

"Clamp Float32 components to [0, 1] and convert to the target pixel type."
@inline function topixel(::Type{T}, r::Float32, g::Float32, b::Float32) where {T <: AbstractRGB}
    return T(clamp(r, 0.0f0, 1.0f0), clamp(g, 0.0f0, 1.0f0), clamp(b, 0.0f0, 1.0f0))
end

# N0f8 must bypass ColorTypes' checkval: its throw path formats strings,
# which cannot compile to GPU code. The clamp makes the check redundant.
@inline fixed8(x::Float32) =
    reinterpret(N0f8, unsafe_trunc(UInt8, round(clamp(x, 0.0f0, 1.0f0) * 255.0f0)))
@inline topixel(::Type{RGB{N0f8}}, r::Float32, g::Float32, b::Float32) =
    RGB{N0f8}(fixed8(r), fixed8(g), fixed8(b))

"""
    straight(c::RGB{Float32}, a::Float32) -> RGB{Float32}

A premultiplied colour back to the colour it IS, for an op whose arithmetic is
not linear — a contrast curve, a LUT lookup, an effect callback. Adding 0.1 of
brightness to a fully transparent pixel has to leave it transparent, and it does
not if the addition lands on the premultiplied value.

Fully transparent stays at zero rather than dividing by it: there is no colour
there to recover, and any value is as good as any other once alpha is zero.
"""
@inline straight(c::RGB{Float32}, a::Float32) =
    a <= 0.0f0 ? RGB{Float32}(0, 0, 0) : RGB{Float32}(c.r / a, c.g / a, c.b / a)

"""
    topixel(T, r, g, b, a) -> T

The four-component write. `a` is DROPPED for a target that has no alpha, so a
kernel says what the coverage is once and runs unchanged into either format —
which is what keeps the alpha convention out of the ops.
"""
@inline topixel(::Type{T}, r::Float32, g::Float32, b::Float32, ::Float32) where {T <: AbstractRGB} =
    topixel(T, r, g, b)
@inline topixel(::Type{T}, r::Float32, g::Float32, b::Float32, a::Float32) where {T <: TransparentRGB} =
    T(clamp(r, 0.0f0, 1.0f0), clamp(g, 0.0f0, 1.0f0), clamp(b, 0.0f0, 1.0f0),
      clamp(a, 0.0f0, 1.0f0))
@inline topixel(::Type{RGBA{N0f8}}, r::Float32, g::Float32, b::Float32, a::Float32) =
    RGBA{N0f8}(fixed8(r), fixed8(g), fixed8(b), fixed8(a))

"""
    premul(T, r, g, b, a) -> T

Write a STRAIGHT colour into a premultiplied format. The counterpart of
[`straight`](@ref), and the pair an op reaches for when its arithmetic is not a
weighted sum.
"""
@inline premul(::Type{T}, r::Float32, g::Float32, b::Float32, a::Float32) where {T} =
    topixel(T, r * a, g * a, b * a, a)

"Upload a host vector to the same backend/device as `like`."
function todevice(like::AbstractArray, host::Vector{Float32})
    dev = KA.allocate(KA.get_backend(like), Float32, length(host))
    copyto!(dev, host)
    return dev
end

include("coloradjust.jl")
include("lut3d.jl")
include("resizeplanar.jl")
include("blur.jl")
include("transform.jl")
include("flow.jl")
include("template.jl")
include("features.jl")
include("lucaskanade.jl")
include("nv12.jl")
include("blend.jl")
include("fxmap.jl")

end
