# The GPUFiltering README pictures, every kernel run on the Vulkan device.
isdefined(@__MODULE__, :ExampleCommon) || include(joinpath(@__DIR__, "common.jl"))
using .ExampleCommon
using GPUFiltering, ColorTypes, FixedPointNumbers, Colors
import KernelAbstractions as KA
import Mantle

todevice(backend, host) = copyto!(KA.allocate(backend, eltype(host), size(host)...), host)

"Flow as colour: hue is the direction, saturation the length against `maxlen`."
flowcolour(u, v, maxlen) =
    [RGB{N0f8}(HSV(mod(atand(v[i], u[i]), 360), clamp(hypot(u[i], v[i]) / maxlen, 0, 1), 1))
     for i in CartesianIndices(u)]

function gpufiltering_examples()
    backend = Mantle.defaultbackend()
    host = RGB{Float32}.(loadimage(media("qwenimage", "harbor.jpg")))
    img = todevice(backend, host)                               # (640, 357) on the device
    W, H = size(img)

    # Colour, blur, sharpen and a rotated crop, each one kernel launch.
    graded = copy(img)
    coloradjust!(graded; contrast = 1.15, saturation = 1.4, temperature = 0.3)
    blurred = similar(img); gaussianblur!(blurred, img, 4.0)
    sharp = similar(img); unsharpmask!(sharp, img, 2.0, 1.5)
    rotated = similar(img)
    warp!(rotated, img, fitmatrix((0.45, 0.35, 0.4, 0.5), (W, H), (W, H); scale = 1.2, rotation = 12))
    KA.synchronize(backend)
    crop = (300:619, 110:288)                                   # the mechanic, 1:1, to see the sharpening
    saveimage(media("gpufiltering", "filters.jpg"),
              vstrip(hstrip(resizewidth.((host, Array(graded), Array(blurred)), 320)...),
                     hstrip(host[crop...], Array(sharp)[crop...], resizewidth(Array(rotated), 320))))

    # Dense optical flow between the picture and a rotated, zoomed copy of it, and
    # the global transform fitted back out of the flow.
    moved = similar(img)
    warp!(moved, img, fitmatrix((0, 0, 1, 1), (W, H), (W, H); scale = 1.06, rotation = 3))
    g1 = KA.allocate(backend, Float32, W, H); g2 = similar(g1)
    grayscale!(g1, img); grayscale!(g2, moved)
    u = similar(g1); v = similar(g1)
    ws = FlowWorkspace(backend, (W, H); levels = 4)
    opticalflow!(ws, u, v, g1, g2)
    T = fitaffine(u, v)                                         # align-back sampling matrix
    aligned = similar(img)
    warp!(aligned, moved, T)
    KA.synchronize(backend)
    t = @elapsed (opticalflow!(ws, u, v, g1, g2); KA.synchronize(backend))
    uh, vh = Array(u), Array(v)
    saveimage(media("gpufiltering", "flow.jpg"),
              hstrip(resizewidth(Array(moved), 320),
                     resizewidth(flowcolour(uh, vh, maximum(hypot.(uh, vh))), 320),
                     resizewidth(Array(aligned), 320)))
    return (; flow_ms = 1000t, T)
end

abspath(PROGRAM_FILE) == (@__FILE__) && gpufiltering_examples()
