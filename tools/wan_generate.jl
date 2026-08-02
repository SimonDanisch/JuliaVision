"""
Wan 2.2 text-to-video end to end on DNNKernels: encoder -> sampler -> VAE decode.

    julia> include("tools/wan_generate.jl")
    julia> pipe = wanpipeline();
    julia> frames = wangenerate(pipe; steps = 4);           # (H, W, 3, T) in 0..1
    julia> wanwrite("gen/wan_sample.mp4", frames)

The three graphs are exported separately (`tools/export_wandit.py --full`,
`export_wanvae.py`, `export_umt5.py`) and are pinned to one clip size, because
`torch.export` bakes every extent. `wanpipeline` checks that the transformer's
latent shape is the one the decoder expects rather than discovering the mismatch
16 GB into a decode.

The transformer here is the 2-layer export, not the shipped 30: the blocks are
the same module repeated, so op coverage and per-op accuracy are identical, and
30 blocks of a 5B model does not fit in this machine's 20 GB alongside a
reference. `--layers` at export time is the only thing that changes.
"""

using DNNKernels
using Lava: LavaBackend
using KernelAbstractions
using Printf
using Random: MersenneTwister
using VideoIO
using VideoIO: open_video_out
using Colors: RGB, N0f8

const GRAPHS = joinpath(@__DIR__, "..", "gen", "graphs")

"""
    wanpipeline(; ditdir, vaedir, encdir, backend) -> WanPipeline

Load the three graphs and their weights, and upload them to `backend`. The text
encoder is optional: passing `encdir = nothing` gives a pipeline that samples from
a context supplied by the caller, which is how the loop is exercised without
paying for umT5-xxl.

The GPU is the default and the CPU backend is only worth asking for as a parity
check: one VAE decode is 30 s on the card and had not finished after 17 minutes
on the CPU, where the 3-D convolutions land in a naive kernel.
"""
function wanpipeline(; ditdir = joinpath(GRAPHS, "wandit-full-fp32"),
                       vaedir = joinpath(GRAPHS, "wanvae16-fp32"),
                       encdir = nothing,
                       backend = LavaBackend())
    dit = loadgraph(joinpath(ditdir, "wan_dit.json"))
    vae = loadgraph(joinpath(vaedir, "wanvae_decoder.json"))
    ditw = readsafetensors(joinpath(ditdir, "weights.safetensors"))
    vaew = readsafetensors(joinpath(vaedir, "weights.safetensors"))
    enc, encw = if encdir === nothing
        dit, Dict{String,Any}()          # never called; see `generate(pipe, latent; context)`
    else
        (loadgraph(joinpath(encdir, "umt5_encoder.json")),
         readsafetensors(joinpath(encdir, "weights.safetensors")))
    end

    # The exported extents must agree, and the failure if they do not is a
    # DimensionMismatch a thousand ops into the decoder.
    zshape = DNNKernels.inshape(vae, DNNKernels.realinputs(vae)[1])
    xshape = DNNKernels.inshape(dit, DNNKernels.realinputs(dit)[1])
    prod(zshape) == prod(xshape) || error(
        "latent mismatch: transformer emits $xshape, decoder wants $zshape — " *
        "re-export the VAE at the transformer's frames/size")

    DNNKernels.WanPipeline(enc, encw, dit, ditw, vae, vaew, backend)
end

"""
    wangenerate(pipe; steps, guidance, seed, context, contextnull) -> (H, W, 3, T)

One generation, returned as frames in 0..1 with time last — the layout the
editor's sources use. `context` defaults to noise of the right shape so the loop
can run without a text encoder; a real prompt goes through
`generate(pipe, tokens, tokens_null, latent)`.
"""
function wangenerate(pipe; steps = 4, guidance = 5.0, seed = 0,
                     context = nothing, contextnull = nothing, progress = wanprogress)
    din = DNNKernels.realinputs(pipe.dit)
    xshape = DNNKernels.inshape(pipe.dit, din[1])
    cshape = DNNKernels.inshape(pipe.dit, din[3])
    rng = MersenneTwister(seed)
    x = randn(rng, Float32, xshape)
    ctx = context === nothing ? randn(rng, Float32, cshape) .* 0.5f0 : context
    ctxn = contextnull === nothing ? zeros(Float32, cshape) : contextnull
    out = DNNKernels.generate(pipe, x; context = ctx, contextnull = ctxn,
                           steps, guidance, progress)
    return towhct(out)
end

"""
    towhct(v) -> (H, W, 3, T)

The decoder emits `(W, H, T, 3, 1)` — torch's `(1, 3, T, H, W)` reversed. Video
here is `(H, W, C, T)` in 0..1; the VAE's range is roughly -1..1.
"""
function towhct(v::AbstractArray)
    a = Array(v)
    ndims(a) == 5 && (a = reshape(a, size(a)[1:4]))
    w, h, t, c = size(a)
    out = Array{Float32}(undef, h, w, c, t)
    for ti in 1:t, ci in 1:c, x in 1:w, y in 1:h
        out[y, x, ci, ti] = clamp((a[x, y, ti, ci] + 1) / 2, 0, 1)
    end
    out
end

function wanprogress(k, n)
    @printf("  step %2d/%d\n", k, n)
    flush(stdout)
end

"""
    wanwrite(path, frames; fps = 8) -> path

Write `(H, W, 3, T)` in 0..1 out as a video. Wan's latents carry 4 pixel frames
per latent frame at 8 fps, so the default matches what the model was trained to
produce rather than the editor's timeline rate.
"""
function wanwrite(path::AbstractString, frames::AbstractArray{<:Real,4}; fps = 8)
    mkpath(dirname(path))
    h, w, c, t = size(frames)
    c == 3 || error("expected 3 channels, got $c")
    open_video_out(path, RGB{N0f8}, (h, w); framerate = fps, target_pix_fmt = VideoIO.AV_PIX_FMT_YUV420P) do writer
        for ti in 1:t
            img = Matrix{RGB{N0f8}}(undef, h, w)
            for y in 1:h, x in 1:w
                img[y, x] = RGB{N0f8}(clamp(frames[y, x, 1, ti], 0, 1),
                                      clamp(frames[y, x, 2, ti], 0, 1),
                                      clamp(frames[y, x, 3, ti], 0, 1))
            end
            write(writer, img)
        end
    end
    return path
end
