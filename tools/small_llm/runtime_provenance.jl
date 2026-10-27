import Mantle, DNNKernels, BonsaiRunner
using SHA

"""Disk/source identity for a freshly started inference worker.

This is not a hash of machine code already loaded in an old session. Callers
must keep the fresh-worker handoff rule, especially after engine edits.
"""
function inference_runtime_provenance()
    packages=((Mantle,("src/vulkan/runtime/launch.jl","src/vulkan/array/ka_backend.jl",
        "src/vulkan/graph.jl","src/vulkan/graphics/api.jl","src/vulkan/raytracing/shaders.jl",
        "src/graph/kalaunch.jl")),(DNNKernels,("src/q4k.jl","src/q6k.jl",)),
        (BonsaiRunner,("src/kernels.jl",)))
    Dict(string(package)=>(;version=string(pkgversion(package)),path=pkgdir(package),
        git_head=try readchomp(`git -C $(pkgdir(package)) rev-parse HEAD`) catch; nothing end,
        source_sha256=Dict(file=>bytes2hex(sha256(read(joinpath(pkgdir(package),file)))) for file in files))
        for (package,files) in packages)
end
