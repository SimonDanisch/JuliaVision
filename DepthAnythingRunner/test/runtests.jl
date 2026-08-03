"""
Until the port runs, this asserts the two things that are true now and must stay
true: the package loads on a machine with no assets, and the asset lookup names
a real place rather than throwing something unreadable.

The latency test that matters — `frozen_stats().misses == 0` in a fresh process
— belongs here once the workload drives the real call. See SAM2Runner/test for
the shape it should take; it has to run in a subprocess because Julia's
compile-time counter is per-process.
"""

using Test, DepthAnythingRunner

@testset "DepthAnythingRunner" begin
    # No `assetdir()`. It is internal — it names where the artifact happens
    # to put things, so a test that calls it has to know the layout and a
    # re-export that moves a file breaks a test that never knew it depended
    # on that. Ask for the graph and the weights instead.
    if DepthAnythingRunner.ready()
        @info "Depth Anything V2 Small: export present"
        g = DepthAnythingRunner.depthanythinggraph()
        @test g !== nothing
        w = DepthAnythingRunner.depthanythingweights()
        @test !isempty(w)
    else
        @info "Depth Anything V2 Small: no export; run tools/export_depthanything.py"
        # The error has to name the path — a caller who has not run the exporter
        # should be told where to put it, not handed a MethodError later.
        @test_throws ArgumentError DepthAnythingRunner.depthanythinggraph()
    end
end

# ---------------------------------------------------------------- latency
#
# The test the scaffold's docstring promised, in a **subprocess**: Julia's
# compile-time counter is per-process, and "the workload covers the editor's
# path" is only a claim about a *fresh* one.
#
# It asserts `no_pipeline_compilation` — which empties `PIPELINE_CACHE` first, so
# a Julia-side hit cannot mask a cold `VkPipelineCache` — rather than
# `frozen_stats().misses`, which cannot tell the frozen cache working from the
# driver's own shader cache having served everything (`STATUS.md`, cross-project).
#
# And it asserts a **negative control** in the same process, because a green from
# an instrument that cannot fire is worth nothing: this repo has hit that class
# three times. The control's kernel body is novel per RUN, so neither Lava's
# cache nor the driver's cross-process one can serve it. If it does not refuse,
# the zero above means nothing and the test fails on that instead.
const LATENCY_SUBPROCESS = raw"""
using DepthAnythingRunner, KernelAbstractions, Lava, ColorTypes
const KA = KernelAbstractions

backend = LavaBackend()
model = depthanything(; backend)
img = KA.allocate(backend, RGB{Float32}, 256, 256)
fill!(img, RGB{Float32}(0.3f0, 0.5f0, 0.7f0))
KA.synchronize(backend)

c0 = Base.cumulative_compile_time_ns()
depth = nothing
wall = @elapsed Lava.no_pipeline_compilation() do
    global depth = depthmap!(model, img)
    KA.synchronize(backend)
end
c1 = Base.cumulative_compile_time_ns()
refused = Lava.PIPELINE_COMPILES_REFUSED[]

@kernel function _novel!(o, ::Val{K}) where {K}
    i = @index(Global)
    o[i] = o[i] * Float32(K) + Float32(K)
end
probe = KA.allocate(backend, Float32, 1024)
fill!(probe, 1.0f0); KA.synchronize(backend)
Lava.no_pipeline_compilation() do
    try
        # `time_ns()` rather than `Random`: novel per RUN either way, and the
        # test environment has no Random.
        _novel!(backend)(probe, Val(Int(time_ns() % 1_000_000)); ndrange = 1024)
        KA.synchronize(backend)
    catch
    end
end
control = Lava.PIPELINE_COMPILES_REFUSED[]

println("RESULT ", (; refused, control, compile = (c1[1] - c0[1]) / 1e9, wall,
                      finite = all(isfinite, Array(depth))))
"""

@testset "DepthAnythingRunner: a fresh process compiles no pipelines" begin
    if !DepthAnythingRunner.ready()
        @info "no DepthAnythingRunner export; skipping the latency test"
    else
        script = tempname() * ".jl"
        write(script, LATENCY_SUBPROCESS)
        # `ignorestatus` so a crashed subprocess still hands back its output:
        # otherwise `read` throws, the output is lost, and a crash is
        # indistinguishable from "no GPU here" — which would let this testset
        # pass with zero assertions, the exact failure it exists to catch.
        # stderr to a file and read it back: `read(cmd, String)` captures stdout
        # only, so a subprocess that dies with a stacktrace hands back "" and the
        # diagnosis is lost exactly when it is needed.
        errfile = tempname()
        stdout_ = read(pipeline(ignorestatus(
            `$(Base.julia_cmd()) --project=$(Base.active_project()) $script`),
            stderr = errfile), String)
        out = stdout_ * (isfile(errfile) ? read(errfile, String) : "")
        lines = split(out, '\n')
        i = findfirst(l -> startswith(l, "RESULT "), lines)
        nodevice = occursin("init_vulkan", out) || occursin("no Vulkan", out) ||
                   occursin("VK_ERROR_INCOMPATIBLE_DRIVER", out)
        if i === nothing && nodevice
            @info "no working device; skipping the latency test"
        elseif i === nothing
            @error "latency subprocess produced no RESULT" out
            @test false
        else
            r = eval(Meta.parse(lines[i][8:end]))
            @info "Depth Anything, first call in a fresh process" r
            @test r.finite                 # the result is still real
            @test r.control > 0            # …and the instrument can fire at all
            @test r.refused == 0           # every pipeline was already cached
            @test r.compile < 5.0          # the package image is being used
        end
    end
end
