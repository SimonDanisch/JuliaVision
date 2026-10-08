# Marching cubes on the device, at three grid sizes: one-shot, per frame through a
# `MarchingCubes`, and the GPU time of every pass. Run in an environment that has
# Lava, Mantle and GPUMeshing.
using Lava, Mantle, GPUMeshing
using GeometryBasics: coordinates, faces
const GM = GPUMeshing

function blob(G)
    xs = range(-1, 1; length = G)
    return Float32[0.6f0 - sqrt(x^2 + y^2 + z^2) + 0.05f0 * sin(9x) * cos(7y) * sin(5z) for x in xs, y in xs, z in xs]
end

"""The frame's passes in a plan that times them."""
function profiled(mc::MarchingCubes)
    g = Mantle.Graph(mc.dev)
    for (kernel, args, threads, group, name) in GM.passes(mc)
        Mantle.dispatch!(g, kernel, args, threads; group, name)
    end
    return Mantle.record!(Mantle.Plan(g; profile = true))
end

function bench(dev, G; frames = 60)
    field = Mantle.Buffer(dev, blob(G))
    marchingcubes(dev, field)
    oneshot = minimum(_ -> @elapsed(marchingcubes(dev, field)), 1:5)
    mc = MarchingCubes(dev, field)
    levels = range(-0.05, 0.05; length = frames)
    foreach(l -> marchingcubes!(mc; level = l), levels)       # grown to the largest frame
    times = sort([(@elapsed marchingcubes!(mc; level = l)) for l in levels])
    m = marchingcubes!(mc; level = 0)
    nv, nf = length(coordinates(m)), length(faces(m))
    println("$(G)³: $nv vertices, $nf faces | one-shot $(round(oneshot * 1e3; digits = 2)) ms | ",
            "per frame median $(round(times[end ÷ 2] * 1e3; digits = 2)) ms")
    plan = profiled(mc)
    for _ in 1:200
        Mantle.run!(plan)
        Mantle.waitfor!(plan)
    end
    for t in Mantle.timings(plan)
        t.name == "updates" || println("    ", rpad(t.name, 36), round(t.gpu_ms; digits = 3), " ms")
    end
    foreach(Mantle.free!, (plan, mc, field))
    return nothing
end

dev = Mantle.Device()
foreach(G -> bench(dev, G), (128, 256, 385))
