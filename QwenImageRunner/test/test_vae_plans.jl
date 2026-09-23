"""
The VAE decoder always plans, and plans per grid.

`qwenimagevae` used to take `record::Bool`, and `record = false` — the default
whenever no `latent` was named — fell through to `DNNKernels.execute!`, which
allocates one buffer per op and frees none. `examples/generate.jl` and
`tools/parity_qwenimage_vae.jl` both called it that way. Measured here on an
8060S, one 256x256 decode:

    old default (interpreted)   7025 MiB peak
    now (planned)               1439 MiB peak

with the output byte-identical to the old RECORDED path, 1 048 576 bytes
compared. The recorded path was always available; it just was not what a caller
who did not name a grid got, because a recording is of concrete dispatches and
the decoder had nowhere to put more than one.

It has somewhere now: it holds a `Model`, and `DNNKernels.call` keys a plan on
`(graph, dims, …)`. So the grid is an argument to `decode!` rather than to the
constructor, and decoding two sizes builds two plans.

The assertions below are shape-and-structure where they can be, because a real
decode needs the multi-gigabyte VAE checkpoint installed. The one test that does
decode is guarded and skips loudly.
"""

using Test
using QwenImageRunner
using QwenImageRunner: qwenimagevae, decode!, vaedims, VAEGRAPH, ready, vaedir
using DNNKernels
using Mantle

@testset "there is no unplanned decode path" begin
    # The keyword is gone, not defaulted. A caller passing it should hear about
    # it rather than silently get a planned decode under a name that promised
    # otherwise.
    kws = Base.kwarg_decl(only(methods(qwenimagevae)))
    @test :record ∉ kws
    @test :latent ∉ kws
    @test Set(kws) == Set([:backend, :dir, :maxpasses])

    # One method. There were two: the second dispatched on `plan::Nothing` and
    # was the door to `execute!`.
    @test length(methods(decode!)) == 1

    # And the decoder has no field that could hold "no plan".
    @test fieldnames(QwenImageRunner.QwenVAEDecoder) == (:backend, :model)
end

@testset "the grid comes from the latents" begin
    # `(width, height, 1, channels, batch)` is the Julia order.
    @test vaedims(zeros(Float16, 16, 32, 1, 64, 1)) == (h = 32, w = 16)
    @test vaedims(zeros(Float16, 64, 64, 1, 64, 2)) == (h = 64, w = 64)
end

@testset "a decoder plans per grid" begin
    if !(ready(:vae_decoder) && isfile(joinpath(vaedir(), "vae.safetensors")))
        @info "no Qwen-Image VAE checkpoint; the decode test is SKIPPED, not passing"
        @test_skip ready(:vae_decoder)
    else
        backend = Mantle.LavaBackend()
        vae = qwenimagevae(; backend)
        try
            # The recording split the old code passed to `planfor` by hand now
            # reaches the plan through the `Model`, which is what `call` reads.
            @test vae.model.record_maxpasses == Dict(VAEGRAPH => 8)
            nplans() = count(k -> k isa Tuple && first(k) === :plan,
                             keys(vae.model.scratch))
            @test nplans() == 0

            # Scaled down: raw N(0,1) is not a denoised latent and overflows
            # fp16 inside the decoder, which is a fact about the input and not
            # about the decoder. 0.3 is in range at every grid tried.
            lat(n) = DNNKernels.toback(backend,
                Float16.(0.3f0 .* randn(Float32, n, n, 1, 64, 1)))

            a = Array(decode!(vae, lat(16)))
            @test size(a) == (256, 256, 4, 1)
            @test !any(isnan, a)
            @test all(x -> -1.001f0 <= x <= 1.001f0, a)
            @test nplans() == 1

            # Same grid again: replayed, not rebuilt.
            decode!(vae, lat(16))
            @test nplans() == 1

            # A DIFFERENT grid on the SAME decoder. This is the case the old
            # code could only serve interpreted, because its single recorded
            # plan was pinned to the `latent` given at construction.
            b = Array(decode!(vae, lat(32)))
            @test size(b) == (512, 512, 4, 1)
            @test !any(isnan, b)
            @test nplans() == 2
        finally
            Mantle.release!(vae)
        end
    end
end
