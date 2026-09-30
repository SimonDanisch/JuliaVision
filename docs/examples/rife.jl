# The RIFERunner README picture: the frame between two ends of a camera push-in,
# next to what a cross-fade makes of the same pair.
isdefined(@__MODULE__, :ExampleCommon) || include(joinpath(@__DIR__, "common.jl"))
using .ExampleCommon
using RIFERunner, GPUFiltering, ColorTypes, FixedPointNumbers

function rife_examples()
    a = RGB{Float32}.(loadimage(media("qwenimage", "harbor.jpg")))   # (640, 357)
    W, H = size(a)
    # The second frame: 12% closer and 2 degrees turned, a fast push-in.
    b = similar(a)
    warp!(b, a, fitmatrix((0, 0, 1, 1), (W, H), (W, H); scale = 1.12, rotation = 2))
    m = rife()
    mid = similar(a)
    interpolate!(mid, m, a, b; t = 0.5)
    t = @elapsed interpolate!(mid, m, a, b; t = 0.5)
    fade = similar(a)
    blend!(fade, a, b, 0.5)
    crop = (330:569, 60:299)                                  # the mechanic, 1:1
    saveimage(media("rife", "pushin.jpg"),
              vstrip(hstrip(resizewidth.((a, fade, mid, b), 320)...),
                     hstrip(a[crop...], fade[crop...], mid[crop...], b[crop...]; gap = 88)))
    release!(m)
    return (; seconds = t, framesize = framesize(m))
end

abspath(PROGRAM_FILE) == (@__FILE__) && rife_examples()
