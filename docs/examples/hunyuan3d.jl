# The Hunyuan3DRunner README picture: the Qwen-Image apple (its alpha is the
# model's own) turned into a mesh, coloured by projecting the picture back onto
# it (`tools/project_texture.jl`) and path traced with RayMakie.
isdefined(@__MODULE__, :ExampleCommon) || include(joinpath(@__DIR__, "common.jl"))
using .ExampleCommon
using Hunyuan3DRunner, FileIO, ColorTypes, FixedPointNumbers, Random
using RayMakie, Hikari, Makie
isdefined(@__MODULE__, :texturize) || include(joinpath(REPO, "tools", "project_texture.jl"))

"(W, H, 4) UInt8 from a straight-alpha PNG, which is what `imagetomesh` takes."
function rgbabytes(path::AbstractString)
    img = loadimage(path)                                    # (W, H) RGBA{N0f8}
    return [reinterpret(UInt8, getfield(img[x, y], c)) for x in axes(img, 1), y in axes(img, 2),
            c in (:r, :g, :b, :alpha)]
end

function turntable(mesh, colours; eye, size = (512, 512), samples = 64)
    scene = Scene(; size, lights = [Makie.AmbientLight(RGBf(0.55, 0.55, 0.58)),
                                    Makie.DirectionalLight(RGBf(0.75, 0.75, 0.75), Vec3f(0.4, 0.9, -0.7))])
    cam3d!(scene)
    mesh!(scene, mesh; color = colours)
    update_cam!(scene, cameracontrols(scene), Vec3f(eye...), Vec3f(0, 0, 0))
    img = colorbuffer(scene; backend = RayMakie, samples, max_depth = 4)   # (H, W) RGBA{Float32}
    return [RGB{N0f8}(clamp(c.r, 0, 1), clamp(c.g, 0, 1), clamp(c.b, 0, 1)) for c in permutedims(img)]
end

"The mesh, cleaned the way upstream's postprocessors do, written to `tmp/apple.obj`."
function applemesh(input::AbstractString)
    m = hunyuan3d()
    Random.seed!(42)                                         # `randomlatents` draws from it
    t = @elapsed begin
        v, f = imagetomesh(m, rgbabytes(input); steps = 50, guidance = 5.0, octree = 384)
    end
    raw = size(f, 2)
    v, f = removefloaters(v, f)
    v, f = removedegenerate(v, f)
    v, f = decimate(v, f; maxfaces = 40_000)
    obj = joinpath(mkpath(joinpath(REPO, "tmp")), "apple.obj")
    writeobj(obj, v, f)
    return (; obj, seconds = t, faces = raw, kept = size(f, 2))
end

"The input and three views of its mesh: the front the picture saw, a side and the back."
function renderapple(obj::AbstractString, input::AbstractString)
    # Only surfaces within 60 degrees of facing the camera take the photo's
    # colour: the apple's rim is lit white in the picture, and painted, it is what
    # the hole fill spreads over the back.
    mesh, colours = texturize(obj, input; grazing = 0.5f0)
    views = [turntable(mesh, colours; eye) for eye in ((0.05, -3.0, 0.4), (2.4, -1.9, 1.0), (-1.4, 2.7, 0.8))]
    saveimage(media("hunyuan3d", "apple.jpg"),
              hstrip(resizewidth(overchecker(loadimage(input)), 384), resizewidth.(views, 384)...))
end

function hunyuan3d_examples()
    input = media("qwenimage", "apple.png")                  # 512x512 RGBA
    made = applemesh(input)
    renderapple(made.obj, input)
    return made
end

abspath(PROGRAM_FILE) == (@__FILE__) && hunyuan3d_examples()
