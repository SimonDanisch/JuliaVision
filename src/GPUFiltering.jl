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

export coloradjust!, gaussianblur!, unsharpmask!, warp!
export channellinear!, channelstats
export ColorAdjustments
export cropmatrix, translationmatrix, areadownscale!
export grayscale!, gradients!, opticalflow!, flowwarp!, bilinearresize!, fitaffine,
       fitsimilarity, ransacsimilarity, ransactranslation, fithomography, FlowWorkspace
export samplewindow!, nccpeak, peakmargin, PatchTracker, matchpatches!
export goodfeatures
export lucaskanade!
export nv12torgb!, lumatogray!
export blend!
export pointwise!, stencil!
export tofloat, topixel

"Convert any RGB pixel to Float32 components for in-kernel math."
@inline tofloat(c::AbstractRGB) = RGB{Float32}(Float32(red(c)), Float32(green(c)), Float32(blue(c)))

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

"Upload a host vector to the same backend/device as `like`."
function todevice(like::AbstractArray, host::Vector{Float32})
    dev = KA.allocate(KA.get_backend(like), Float32, length(host))
    copyto!(dev, host)
    return dev
end

include("coloradjust.jl")
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
