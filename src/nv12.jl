# NV12 → RGB color conversion on the GPU.
#
# The hardware H.264 decoder (Lava's Vulkan Video path) yields NV12 planes kept
# device-resident: `y` is the full-res luma (W×H, UInt8) and `uv` the half-res
# interleaved chroma (W×(H÷2), UInt8 — each row is U,V,U,V,… for W÷2 chroma
# samples). This converts them to an RGB image on-GPU (nearest-neighbour chroma
# upsampling), so a decoded frame reaches the color/tracking pipeline as ordinary
# RGB without a host round-trip.
#
# BT.709 limited-range coefficients (the common HD case); `bt601=true` switches to
# the SD matrix. Matches ffmpeg/VideoIO's YUV→RGB to within a couple of levels
# (the residual is ffmpeg's bilinear chroma upsampling vs. our nearest).

@inline function yuv_to_rgb(Yv::Float32, Uv::Float32, Vv::Float32, bt601::Bool)
    c = Yv - 16.0f0; d = Uv - 128.0f0; e = Vv - 128.0f0
    if bt601
        r = 1.164f0 * c + 1.596f0 * e
        g = 1.164f0 * c - 0.391f0 * d - 0.813f0 * e
        b = 1.164f0 * c + 2.018f0 * d
    else
        r = 1.164f0 * c + 1.793f0 * e
        g = 1.164f0 * c - 0.213f0 * d - 0.533f0 * e
        b = 1.164f0 * c + 2.112f0 * d
    end
    return (r / 255.0f0, g / 255.0f0, b / 255.0f0)
end

@kernel function nv12torgb_kernel!(out, @Const(y), @Const(uv), bt601::Bool)
    I = @index(Global, Cartesian)
    x, row = Tuple(I)                        # 1-based; x ∈ 1:W, row ∈ 1:H
    cx = (x - 1) >> 1                        # 0-based chroma column
    crow = ((row - 1) >> 1) + 1              # 1-based chroma row
    @inbounds begin
        Yv = Float32(y[x, row])
        Uv = Float32(uv[2cx + 1, crow])      # interleaved U
        Vv = Float32(uv[2cx + 2, crow])      # interleaved V
    end
    r, g, b = yuv_to_rgb(Yv, Uv, Vv, bt601)
    @inbounds out[I] = topixel(eltype(out), r, g, b)
end

"""
    nv12torgb!(out, y, uv; bt601=false) -> out

Convert NV12 planes (`y`: W×H UInt8 luma, `uv`: W×(H÷2) UInt8 interleaved chroma)
to the RGB image `out` (W×H) on the GPU. Uses BT.709 limited-range by default;
pass `bt601=true` for SD content.
"""
function nv12torgb!(out::AbstractMatrix{<:AbstractRGB},
                    y::AbstractMatrix{UInt8}, uv::AbstractMatrix{UInt8}; bt601::Bool = false)
    size(out) == size(y) || throw(DimensionMismatch("out $(size(out)) vs luma $(size(y))"))
    nv12torgb_kernel!(KA.get_backend(out))(out, y, uv, bt601; ndrange = size(out))
    return out
end
