# The NeuralLUTRunner README pictures. The model is upstream's FiveK sRGB
# enhancer, trained to turn flat, dark camera output into a retouched photo, so
# the Qwen-Image examples are first made flat and cool with `coloradjust!` and
# the predicted LUT has to undo that.
isdefined(@__MODULE__, :ExampleCommon) || include(joinpath(@__DIR__, "common.jl"))
using .ExampleCommon
using NeuralLUTRunner, GPUFiltering, ColorTypes, FixedPointNumbers
import KernelAbstractions as KA

function neurallut_examples()
    m = neurallut()
    backend = m.backend
    timings = Pair{String,Float64}[]
    for name in ("harbor", "fox")
        host = RGB{Float32}.(loadimage(media("qwenimage", "$name.jpg")))
        img = KA.allocate(backend, RGB{Float32}, size(host)...)
        copyto!(img, host)
        coloradjust!(img; brightness = -0.15, contrast = 0.7, saturation = 0.7, temperature = -0.3)
        flat = Array(img)
        out = similar(img)
        lut = predictlut(m, img)            # (33, 33, 33, 3) Float32 on the device, [r, g, b, c]
        grade!(out, img, lut)
        KA.synchronize(backend)
        push!(timings, name => @elapsed (grade!(m, out, img); KA.synchronize(backend)))
        saveimage(media("neurallut", "$name.jpg"),
                  hstrip(resizewidth(host, 320), resizewidth(flat, 320), resizewidth(Array(out), 320)))
    end
    return timings
end

abspath(PROGRAM_FILE) == (@__FILE__) && neurallut_examples()
