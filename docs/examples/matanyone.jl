# The MatAnyoneRunner README picture: SAM 2 marks the mechanic in the first frame
# of a camera push-in, and MatAnyone carries the matte through the other 23.
isdefined(@__MODULE__, :ExampleCommon) || include(joinpath(@__DIR__, "common.jl"))
using .ExampleCommon
using MatAnyoneRunner, SAM2Runner, GPUFiltering, ColorTypes, FixedPointNumbers

function matanyone_examples(; n = 24)
    still = loadimage(media("qwenimage", "harbor.jpg"))      # (640, 357) RGB{N0f8}
    W, H = size(still)
    # A push-in towards the mechanic: 30% closer and 4 degrees turned by the end.
    frames = map(0:n-1) do k
        s = k / (n - 1)
        f = similar(still)
        warp!(f, still, fitmatrix((0, 0, 1, 1), (W, H), (W, H);
                                  scale = 1 + 0.3s, position = (-0.12s, -0.05s), rotation = 4s))
        f
    end
    seed = segment(frames[1], [(0.68, 0.7), (0.675, 0.52)])    # the SAM2Runner example's clicks
    unloadmodel!()
    propagate = matanyonepropagator()
    alpha = propagate(frames, Dict(1 => seed))                 # (W, H, n) UInt8
    t = @elapsed propagate(frames, Dict(1 => seed))
    picks = (1, n ÷ 2, n)
    saveimage(media("matanyone", "pushin.jpg"),
              vstrip(hstrip((resizewidth(frames[k], 320) for k in picks)...),
                     hstrip((resizewidth(overchecker(cutout(frames[k], alpha[:, :, k])), 320)
                             for k in picks)...)))
    return (; seconds_per_frame = t / n)
end

abspath(PROGRAM_FILE) == (@__FILE__) && matanyone_examples()
