# The SAM2Runner README pictures: clicks on the Qwen-Image examples, masks tinted
# and cut out. `julia --project=<root env> docs/examples/sam2.jl` rewrites media/sam2/.
isdefined(@__MODULE__, :ExampleCommon) || include(joinpath(@__DIR__, "common.jl"))
using .ExampleCommon
using SAM2Runner, ColorTypes, FixedPointNumbers

function sam2_examples()
    fox = loadimage(media("qwenimage", "fox.jpg"))         # (W, H) RGB{N0f8}
    harbor = loadimage(media("qwenimage", "harbor.jpg"))

    shots = [
        ("fox", fox, [(0.5, 0.55)]),                          # one click on the chest
        ("mechanic", harbor, [(0.68, 0.7), (0.675, 0.52)]),    # the man: body and head
        ("airship", harbor, [(0.35, 0.1), (0.35, 0.3, false)]), # balloon in, mast out
    ]
    timings = Pair{String,Float64}[]
    for (name, img, points) in shots
        mask = segment(img, points)                           # Matrix{UInt8}, size(img)
        t = @elapsed segment(img, points)
        push!(timings, name => t)
        W, H = size(img)
        clicked = tintmask(img, mask)
        for p in points
            marker!(clicked, p[1] * W, p[2] * H;
                    color = length(p) == 3 && !p[3] ? RGB{N0f8}(0.2, 0.2, 0.2) : RGB{N0f8}(1, 0.2, 0.2))
        end
        saveimage(media("sam2", "$name.jpg"),
                  hstrip(resizewidth(clicked, 480), resizewidth(overchecker(cutout(img, mask)), 480)))
    end
    unloadmodel!()
    return timings
end

abspath(PROGRAM_FILE) == (@__FILE__) && sam2_examples()
