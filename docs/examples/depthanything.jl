# The DepthAnythingRunner README pictures: depth of the Qwen-Image examples, in
# the colour map upstream's demo uses (Spectral reversed: red is near).
isdefined(@__MODULE__, :ExampleCommon) || include(joinpath(@__DIR__, "common.jl"))
using .ExampleCommon
using DepthAnythingRunner, GPUFiltering, ColorTypes, FixedPointNumbers
import KernelAbstractions as KA
import Makie

function depthanything_examples()
    m = depthanything()
    cmap = reverse(Makie.to_colormap(:Spectral))
    timings = Pair{String,Float64}[]
    for name in ("harbor", "fox")
        img = loadimage(media("qwenimage", "$name.jpg"))  # (W, H) RGB{N0f8}
        d = depthmap!(m, img)                             # (518, 518, 1, 1) Float16, on the device
        KA.synchronize(m.backend)
        push!(timings, name => @elapsed (depthmap!(m, img); KA.synchronize(m.backend)))
        # Inverse relative depth: larger is nearer, no scale, no fixed range.
        dh = Float32.(Array(d)[:, :, 1, 1])               # the graph returns Float16
        full = zeros(Float32, size(img))
        bilinearresize!(full, dh)                         # back to the frame's size
        lo, hi = extrema(full)
        coloured = [RGB{N0f8}(Makie.interpolated_getindex(cmap, (v - lo) / (hi - lo)))
                    for v in full]
        saveimage(media("depthanything", "$name.jpg"),
                  hstrip(resizewidth(img, 480), resizewidth(coloured, 480)))
    end
    return timings
end

abspath(PROGRAM_FILE) == (@__FILE__) && depthanything_examples()
