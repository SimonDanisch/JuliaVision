"""
    DemoApp

A model picker for the JuliaVision runners: choose a model, get a GUI for it.

    using DemoApp
    server = DemoApp.serve()            # http://127.0.0.1:8080

Every model gets real controls — the inputs its entry point actually takes — and
a Run button. `run` is written against the runner's own function, so a ported
model runs for real and an unported one reports which of three things is
missing (package, weights, entry point) instead of one flat "unavailable".

The runners are loaded with `Base.require` rather than imported, so this package
stays light and works with whatever subset the active project has dev'd. Run it
from the VulkanDev environment, which has all of them.
"""
module DemoApp

using Bonito, WGLMakie, Makie, Colors, FileIO, Observables
using Bonito: Dropdown, Button, TextField, Slider, DOM

export playground, serve

assetdir() = joinpath(pkgdir(DemoApp), "assets")

# ── the roster ───────────────────────────────────────────────────────────────

"""
`kind` picks the controls, `entry` is the function that does the work, and
`assets` overrides the default `ready()` check for runners that have none
(Bonsai keeps its checkpoint outside the artifact system).
"""
struct Model
    name::String
    task::String
    mod::Symbol
    kind::Symbol
    entry::Union{Symbol,Nothing}
    run::Union{Function,Nothing}
    assets::Union{Function,Nothing}
    blurb::String
end
Model(n, t, m, k, e, r; assets = nothing, blurb = "") = Model(n, t, m, k, e, r, assets, blurb)

"""
The side this demo WANTS Qwen to generate at.

1024² is 4096 image tokens and about 9 s a denoise step; 256² is 256 tokens, and
the attention that dominates the step is quadratic in them.

Whether it can have it is a property of the shipped GRAPH, not of the runner.
The denoiser became generic in an `i` symbol and the decoder in `h`/`w`, but the
artifacts bound today were exported before that — `latents` is still
`[1, 4096, 64]` and the decoder carries no symbol at all — so asking for another
grid is an `ArgumentError` naming the exporter flag. [`qwengrid`](@ref) falls
back to the export's own grid rather than leaving the model unusable, and says
which happened.
"""
const DEMO_IMAGE_SIDE = 256

"""
    qwengrid(mod, ctx) -> (transformer, (lh, lw), note)

The denoiser at [`DEMO_IMAGE_SIDE`](@ref) if the shipped graph can serve it, and
at whatever it WAS exported for otherwise.

Asked by building it, because the bounds a `Dim` was exported with are not on the
graph: a generic denoiser still refuses a grid outside them, and only the
constructor knows. The refusal is cheap — it precedes the weight load.
"""
function qwengrid(mod, ctx::Integer)
    sf = mod.QWEN_IMAGE_21.vae_scale_factor
    want = DEMO_IMAGE_SIDE ÷ sf
    build(g) = Base.invokelatest(getfield(mod, :qwenimagetransformer);
                                 context_tokens = ctx, latent = g)
    try
        return (build((want, want)), (want, want), "")
    catch err
        err isa ArgumentError || rethrow()
        lh, lw = Base.invokelatest(mod.latentshape, Base.invokelatest(mod.assetdir))
        return (build((lh, lw)), (lh, lw),
                "  ·  $(DEMO_IMAGE_SIDE)² needs a re-exported graph; running at $(lw * sf)²")
    end
end

"Constructed models are expensive (weights + kernel compile), so build once."
const BUILT = Dict{Symbol,Any}()
built(mod, sym, args...; kw...) = get!(BUILT, Symbol(nameof(mod), :_, sym)) do
    Base.invokelatest(getfield(mod, sym), args...; kw...)
end

const MODELS = Model[
    Model("SAM 2.1", "segment anything", :SAM2Runner, :click_mask, :segment,
          (mod, i) -> (:mask, Base.invokelatest(mod.segment, i.img, [(i.row, i.col)];
                                                key = i.key, pick = :best));
          blurb = "Click a thing and it returns the mask. The encoder is 549 Mantle passes; the click is the cheap half."),

    Model("Depth Anything V2 S", "monocular depth", :DepthAnythingRunner, :image, :depthmap!,
          (mod, i) -> (:heat, Base.invokelatest(mod.depthmap!, built(mod, :depthanything), i.img));
          blurb = "Depth from a single image, 518²."),

    Model("Kokoro-82M", "text → speech", :KokoroRunner, :prompt_audio, :speak,
          (mod, i) -> (:audio, Base.invokelatest(mod.speak, built(mod, :Kokoro), i.prompt;
                                                 voice = i.voice, speed = i.speed));
          blurb = "Phonemised on the host — a lexicon lookup — then the model runs on the GPU."),

    Model("Bonsai 2", "text generation", :BonsaiRunner, :prompt_text, :generate,
          function (mod, i)
              s = get!(BUILT, :bonsai_session) do
                  Base.invokelatest(getfield(mod, :session), built(mod, :Bonsai2))
              end
              (:text, string(Base.invokelatest(mod.generate, s, i.prompt; max_tokens = i.maxtokens)))
          end;
          assets = mod -> (h = artifactshas(mod); !isempty(h) && all(indepot, h)),
          blurb = "Ternary Bonsai 2 from a PTQ1 GGUF, 262144-token native context."),

    Model("Qwen-Image 2.1", "text → image", :QwenImageRunner, :prompt_image, :denoise!,
          function (mod, i)
              dnn = isdefined(mod, :DNNKernels) ? getfield(mod, :DNNKernels) :
                    Base.require(Main, :DNNKernels)
              mantle = Base.require(Main, :Mantle)
              say(t) = (haskey(i, :status) && (i.status[] = t); nothing)

              # Each component is built and RELEASED in the order it is finished
              # with, which is what `QwenImageRunner/examples/generate.jl` does and
              # what the runner's own `release!` docstring requires: the conditioner
              # decodes to 6.9 GB on the device and the denoiser to 7.3 GB, so
              # holding both plus the decoder is 18 GB of weights before a single
              # activation. This entry used to cache all three in `BUILT` and never
              # free them, which is what made one click take the machine down.
              #
              # Nothing is cached for a second reason as well: the denoiser's
              # recorded plan is bound to THIS prompt's token count, so a cached one
              # throws `DimensionMismatch` the moment the prompt length changes.
              # On the device the three measure 6.8, 7.6 and 4.7 GB, so holding
              # them at once is 19 GB of weights before a single activation.
              # Released in turn, a 1024² generation peaks at 17.9 GB on a 32 GB
              # machine. (`Metal.currentAllocatedSize` counts Mantle's retained
              # POOL, not live memory, so that figure is a fresh-session one —
              # a second run in the same session starts near the first's high
              # water mark and says nothing about what is live.)
              # Before 6.9 GB of encoder weights, not after: `encode_prompt`
              # enforces this same limit, but only once they are resident.
              used, capacity = Base.invokelatest(mod.prompttokens, i.prompt)
              used <= capacity || error(
                  "the prompt is $used tokens and the exported encoder holds $capacity. " *
                  "Shorten it by about $(used - capacity), or re-export with " *
                  "`--component text_encoder --prompt-tokens $used`.")
              say("loading the text encoder…")
              enc = Base.invokelatest(getfield(mod, :qwenimagetextencoder))
              emb = try
                  say("encoding the prompt…")
                  Base.invokelatest(mod.encode_prompt, enc, i.prompt)
              finally
                  Base.invokelatest(mantle.release!, enc)
              end

              # Both graphs are symbolic in the resolution now — the denoiser in
              # `i` and the decoder in `h`/`w` — so the grid is a choice rather
              # than whatever the export traced at. 256² is a sixteenth of 1024²'s
              # image tokens, which is where the denoise loop's time goes.
              sf = mod.QWEN_IMAGE_21.vae_scale_factor
              ch = mod.QWEN_IMAGE_21.latent_channels

              say("loading the denoiser…")
              tr, (lh, lw), gridnote = qwengrid(mod, size(emb, 2))
              w, h = lw * sf, lh * sf
              denoised = try
                  back = tr.backend
                  ntok = Base.invokelatest(mod.image_sequence_length, w, h)
                  sched = Base.invokelatest(mod.qwen_schedule, w, h; steps = i.steps)
                  # Float16 and on the device, both: the graphs declare `Float16`
                  # for latents and embeddings, and `euler_step!` broadcasts
                  # against a device-side prediction — a host array there is a
                  # `KernelError` inside the loop rather than a copy.
                  lat = dnn.toback(back, Float16.(randn(Float32, ch, ntok, 1)))
                  embd = dnn.toback(back, Float16.(emb))
                  ts = dnn.toback(back, fill(Float16(0), 1))
                  for k in 1:i.steps
                      say("denoising: step $k of $(i.steps)…")
                      fill!(ts, convert(Float16, sched.timesteps[k] / 1000f0))
                      Base.invokelatest(mod.euler_step!, lat,
                                        Base.invokelatest(mod.denoise!, tr, lat, embd, ts),
                                        sched.sigmas[k], sched.sigmas[k + 1])
                  end
                  Base.invokelatest(mantle.waitidle, Base.invokelatest(mantle.todevice, back))
                  Array(lat)
              finally
                  Base.invokelatest(mantle.release!, tr)
              end

              # RECORDED, and therefore placed. The interpreted path allocates
              # every transient of the graph at once — 63 GiB at 1024², and still
              # a multiple of what the same decode needs at any other size. That
              # ratio is the defect, not the absolute number, so it is not a trade
              # that gets acceptable by shrinking the image.
              say("loading the decoder…")
              vae = Base.invokelatest(getfield(mod, :qwenimagevae);
                                      record = true, latent = (lh, lw))
              img = try
                  say("decoding…")
                  side = w ÷ sf
                  grid = Base.invokelatest(mod.unpacklatents, denoised, side, side)
                  Array(Base.invokelatest(mod.decode!, vae,
                        dnn.toback(vae.backend, reshape(Float16.(grid), side, side, 1, ch, 1))))
              finally
                  Base.invokelatest(mantle.release!, vae)
              end

              # RGBA in [-1, 1], and `asimage` cannot read it: the colour axis is
              # neither the third nor the last. A recorded plan returns the
              # declared output `(W, H, 4, 1)` while the interpreted path returns
              # its view root `(W, H, 1, 4, 1)`, so drop the singletons and the two
              # agree. Alpha is opaque for a text-to-image generation.
              isempty(gridnote) || say("decoding…" * gridnote)
              a = Float32.(img)
              a = reshape(a, filter(>(1), size(a))...)
              rgb = clamp.(a[:, :, 1:3] ./ 2f0 .+ 0.5f0, 0f0, 1f0)
              (:image, RGB.(view(rgb, :, :, 1), view(rgb, :, :, 2), view(rgb, :, :, 3)))
          end;
          blurb = "Qwen3-VL prompt embeddings, a FlowMatch schedule, then the VAE decode. " *
                  "1024², fixed by the export. Loads 13 GB in three stages and frees each: " *
                  "~2.5 min before the first step, then ~9 s a step."),

    Model("Neural 3D LUT", "style / mood grading", :NeuralLUTRunner, :image, :grade!,
          function (mod, i)
              out = similar(i.img)
              (:image, Base.invokelatest(mod.grade!, built(mod, :neurallut), out, i.img))
          end;
          blurb = "Predicts a 3D LUT for the picture, then applies it."),

    Model("RIFE 4.26", "frame interpolation", :RIFERunner, :two_frames, :interpolate!,
          function (mod, i)
              model = built(mod, :rife)
              w, h = Base.invokelatest(mod.framesize, model)
              a, b = fit(i.a, w, h), fit(i.b, w, h)
              out = similar(a)
              (:image, Base.invokelatest(mod.interpolate!, out, model, a, b; t = i.t))
          end;
          blurb = "Invents the frame between two frames."),

    Model("MatAnyone2", "video matting", :MatAnyoneRunner, :image_mask, :runmatanyone,
          function (mod, i)
              img = tensor(i.img, 288, 288)
              # 0..255, NOT 0..1. The model was validated against a `(0.0, 255.0)`
              # mask, and a 0/1 one is 255x too faint to register: it returns an
              # all-zero matte with no error anywhere. `MatAnyoneRunner`'s own
              # propagator documents the same scale for the same reason.
              msk = Float32.(fit(i.mask, 288, 288) .!= 0) .* 255f0
              out = Base.invokelatest(mod.runmatanyone, built(mod, :matanyonemodel), img, msk)
              # A matte that is entirely NaN is the open fault in this path, not a
              # picture: `encode_image` saturates Float16 through the library
              # convolution and everything downstream is NaN. Rendered, it is a
              # black square indistinguishable from "nothing to segment", which
              # reads as a broken port rather than a bug on one route. An
              # all-zero matte is the other degenerate answer and means the seed
              # never propagated. Both are reported rather than shown.
              host = Array(out)
              all(isfinite, host) || error(
                  "the matte came back NaN. This is the open MPSGraph convolution " *
                  "fault in `encode_image`, not your mask — running another model " *
                  "first happens to avoid it.")
              any(>(0.01), host) || error(
                  "the matte came back empty: the seed mask reached the model but " *
                  "nothing propagated from it.")
              (:image, out)
          end;
          blurb = "Propagates a matte from a seed mask. Segment a frame with SAM 2 first."),

    Model("BasicVSR++", "video upscaling", :BasicVSRRunner, :image, :upscale,
          function (mod, i)
              # the exported graph declares (64, 64, 3, 5, 1): a five-frame clip
              f = tensor(i.img, 64, 64)
              clip = repeat(reshape(f, 64, 64, 3, 1, 1), 1, 1, 1, 5, 1)
              (:image, Base.invokelatest(mod.upscale, built(mod, :basicvsrppmodel), clip))
          end;
          blurb = "The graph is exported, so this is the furthest along of the scaffolds."),

    Model("Horizon 32B", "text generation", :HorizonRunner, :prompt_text, :generate,
          (mod, i) -> (:text, string(Base.invokelatest(mod.generate, built(mod, :horizon32b), i.prompt;
                                                       maxnew = i.maxtokens)));
          blurb = "A 32B model on the same runtime."),

    Model("Whisper large-v3-turbo", "speech → text", :WhisperRunner, :audio_text, :transcribe,
          # `transcribe(w, path)` takes the path itself and decodes the container
          # through ffmpeg, so the control this GUI already had maps straight onto
          # it. This entry was `nothing`, which made the panel report NO ENTRY
          # POINT YET over a runner that exports `transcribe` and whose weights
          # are installed — the one status that claims something about the PORT
          # rather than about this machine, and it was wrong.
          function (mod, i)
              isfile(i.path) || error(
                  "no audio at $(i.path). Say something with Kokoro-82M first — it " *
                  "leaves its last utterance there — or point this at a file of your own.")
              (:text, first(Base.invokelatest(mod.transcribe, built(mod, :whisper), i.path)))
          end;
          blurb = "Encoder measured at 120.5 ms for a 30 s window. Defaults to whatever " *
                  "Kokoro last said, so the two models chain without a file to find."),
    Model("Qwen-Image-Edit", "instructed image edit", :QwenImageEditRunner, :prompt_image, nothing, nothing;
          blurb = "Edit an image by describing the change."),
    Model("FluxKlein", "text → image", :FluxKleinRunner, :prompt_image, nothing, nothing;
          blurb = "Image generation."),
    Model("ZImage", "text → image", :ZImageRunner, :prompt_image, nothing, nothing;
          blurb = "Image generation."),
    Model("Demucs", "stem separation", :DemucsRunner, :audio_audio, nothing, nothing;
          blurb = "Splits a mix into drums, bass, vocals and the rest."),
    Model("DeepFilterNet", "voice denoising", :DeepFilterRunner, :audio_audio, nothing, nothing;
          blurb = "Real-time speech denoising."),
    Model("ProPainter", "object removal", :ProPainterRunner, :image_mask, nothing, nothing;
          blurb = "Non-commercial licence (S-Lab 1.0)."),
    Model("Hunyuan3D", "image → 3D mesh", :Hunyuan3DRunner, :image, nothing, nothing;
          blurb = "A mesh from a single picture."),
]

# ── state ────────────────────────────────────────────────────────────────────

function runnermodule(m::Model)
    isdefined(Main, m.mod) && return getfield(Main, m.mod)
    Base.identify_package(string(m.mod)) === nothing && return nothing
    try
        return Base.require(Main, m.mod)
    catch err
        @error "loading $(m.mod) failed" exception = (err, catch_backtrace())
        return err
    end
end

"""
Whether any artifact this package binds is already on disk, **without
materializing one**.

This guard exists because the obvious check is a trap: a runner's `ready()` and
its `assetdir()`/`checkpointpath()` resolve `@artifact_str`, and for a `lazy`
artifact that DOWNLOADS. Calling those to populate the roster would kick off
gigabytes — Bonsai's checkpoint alone is 5.9 GB — merely because the page was
opened. This reads the depot directly, so it cannot fetch anything.
"""
function artifactshas(mod)
    dir = pkgdir(mod)
    dir === nothing && return String[]
    toml = joinpath(dir, "Artifacts.toml")
    isfile(toml) || return String[]
    [m.captures[1] for m in eachmatch(r"git-tree-sha1\s*=\s*\"([0-9a-f]{40})\"", read(toml, String))]
end

indepot(sha) = any(d -> isdir(joinpath(d, "artifacts", sha)), DEPOT_PATH)

function artifactspresent(mod)
    dir = pkgdir(mod)
    dir === nothing && return true
    toml = joinpath(dir, "Artifacts.toml")
    isfile(toml) || return true            # nothing artifact-backed to check
    # Scraped with a regex rather than parsed: TOML is a stdlib this package
    # would have to declare, and adding a dep to a dev'd package forces a
    # resolve of whatever environment loads it. The shape being matched is
    # fixed (`git-tree-sha1 = "<40 hex>"`), so a parser buys nothing here.
    shas = artifactshas(mod)
    isempty(shas) && return true
    # Read the depot directly: this cannot resolve, and so cannot fetch.
    any(indepot, shas)
end

"""
`:absent` · `:loaderror` · `:noassets` · `:noentry` (loads, has weights, but
nothing runs it) · `:ready`. These look identical from outside and mean
completely different things, which is why they are kept apart.

Side-effect free by construction: see [`artifactspresent`](@ref).
"""
function status(m::Model)
    mod = runnermodule(m)
    mod === nothing && return (:absent, nothing)
    mod isa Exception && return (:loaderror, mod)
    artifactspresent(mod) || return (:noassets, mod)
    have = try
        m.assets !== nothing ? m.assets(mod) :
        isdefined(mod, :ready) ? Base.invokelatest(getfield(mod, :ready)) : true
    catch err
        @error "assets check for $(m.name) threw" exception = (err, catch_backtrace())
        false
    end
    have || return (:noassets, mod)
    (m.entry === nothing || m.run === nothing || !isdefined(mod, m.entry)) && return (:noentry, mod)
    return (:ready, mod)
end

# ── styling ──────────────────────────────────────────────────────────────────

const INK, MUTED, ACCENT, EDGE = "#2B1E3D", "#7E7490", "#732D8D", "#D8CBE8"
const BG, PANEL, GREEN, EMBER, RED = "#F8F5FC", "#FFFFFF", "#389826", "#C25A2B", "#CB3C33"

card(children...) = DOM.div(children...;
    style = "background:$PANEL;border:1px solid $EDGE;border-radius:8px;padding:18px")
pill(text, colour) = DOM.span(text;
    style = "font:600 11px/1 ui-monospace,monospace;letter-spacing:1px;color:$colour;" *
            "border:1px solid $colour;border-radius:99px;padding:5px 10px;white-space:nowrap")
label(t) = DOM.div(t; style = "font:600 11px/1 ui-monospace,monospace;letter-spacing:1px;color:$MUTED;margin:0 0 6px")
note(t)  = DOM.div(t; style = "font-size:13px;color:$MUTED;line-height:1.5;margin-top:10px")
statusline(o) = DOM.div(o; style = "font-size:13px;color:$MUTED;margin-top:8px;min-height:18px")

const STATE_PILL = Dict(
    :ready     => ("RUNS HERE", GREEN),
    :noentry   => ("NO ENTRY POINT YET", EMBER),
    :noassets  => ("WEIGHTS NOT INSTALLED", EMBER),
    :absent    => ("NOT INSTALLED HERE", MUTED),
    :loaderror => ("FAILED TO LOAD", RED))

const WHY = Dict(
    :noentry   => "The package loads and its weights are here, but nothing in it runs the model " *
                  "end to end yet. The controls are what it will need; the wiring is what is missing.",
    :noassets  => "The package loads. Its weights are a lazy artifact and have not been fetched — " *
                  "they download on first real use, not at install time.",
    :absent    => "Not a dependency of this environment.",
    :loaderror => "The package is a dependency but failed to load.")

images() = sort(filter(f -> endswith(f, r"\.png|\.jpg"i), readdir(assetdir())))
loadimage(f) = RGB.(load(joinpath(assetdir(), f)))
imagepicker(default = "portrait.png") =
    (fs = images(); Dropdown(fs; index = something(findfirst(==(default), fs), 1)))


"""
Nearest-neighbour resize. The runners' graphs declare fixed input shapes
(MatAnyone 288², RIFE its own `framesize`), and a demo that only works when the
picture happens to be the right size is not a demo.
"""
function fit(img::AbstractMatrix, w::Integer, h::Integer)
    (size(img, 1), size(img, 2)) == (w, h) && return img
    sx, sy = size(img, 1) / w, size(img, 2) / h
    [img[clamp(ceil(Int, i * sx), 1, size(img, 1)),
         clamp(ceil(Int, j * sy), 1, size(img, 2))] for i in 1:w, j in 1:h]
end

"`(W, H, 3, 1)` Float32, which is what an exported vision graph takes."
function tensor(img::AbstractMatrix{<:Colorant}, w::Integer, h::Integer)
    f = fit(img, w, h)
    out = Array{Float32}(undef, w, h, 3, 1)
    @inbounds for j in 1:h, i in 1:w
        c = RGB{Float32}(f[i, j])
        out[i, j, 1, 1] = c.r; out[i, j, 2, 1] = c.g; out[i, j, 3, 1] = c.b
    end
    out
end


"""
Whatever a runner returns, as something Makie can show.

The runners hand back what their graph produced: a `Matrix{RGB}` from the LUT,
a device `MtlArray{Float16}` of `(W, H, C, N)` from Qwen and BasicVSR, a plain
matrix of depths. Normalising here keeps that variety out of the GUI code.
"""
function asimage(x)
    x isa AbstractMatrix{<:Colorant} && return RGB.(x)
    a = Float32.(Base.invokelatest(Array, x))          # device -> host
    # Every trailing axis goes, not just a fourth: BasicVSR returns a five-frame
    # clip `(W, H, C, T, N)`. Dropping only `N` left the result 5-D, `RGB.`
    # broadcast over it without complaint, and the pane got an `Array{RGB,5}`
    # that `permutedims` has no method for.
    ndims(a) > 3 && (a = a[:, :, :, ntuple(_ -> 1, ndims(a) - 3)...])
    if ndims(a) == 3
        lo, hi = extrema(a)
        n = hi > lo ? (a .- lo) ./ (hi - lo) : zero(a)
        return size(n, 3) >= 3 ? RGB.(n[:, :, 1], n[:, :, 2], n[:, :, 3]) :
                                 RGB.(n[:, :, 1], n[:, :, 1], n[:, :, 1])
    end
    lo, hi = extrema(a)
    n = hi > lo ? (a .- lo) ./ (hi - lo) : zero(a)
    RGB.(n, n, n)
end

"""
A depth field as a host `Matrix{Float32}`, which is what `heatmap!` takes.

Depth Anything returns `(W, H, 1, 1)` **on the device**, and `permutedims` has a
method for neither a 4-D array nor a device one — so the heat pane threw on
every run. Colour results went through [`asimage`](@ref), which normalises both;
this is the same reduction for the one output kind that bypassed it.
"""
function asheat(x)
    a = Float32.(Base.invokelatest(Array, x))          # device -> host
    ndims(a) > 2 && (a = a[:, :, ntuple(_ -> 1, ndims(a) - 2)...])
    a
end

function imagepane(obs; size = (460, 340))
    fig = Figure(; size = size, backgroundcolor = BG)
    ax = Axis(fig[1, 1]; aspect = DataAspect(), yreversed = true, backgroundcolor = BG)
    hidedecorations!(ax); hidespines!(ax)
    Makie.image!(ax, lift(a -> permutedims(a), obs))
    fig
end

function heatpane(obs; size = (460, 340))
    fig = Figure(; size = size, backgroundcolor = BG)
    ax = Axis(fig[1, 1]; aspect = DataAspect(), yreversed = true, backgroundcolor = BG)
    hidedecorations!(ax); hidespines!(ax)
    Makie.heatmap!(ax, lift(a -> permutedims(Float32.(a)), obs); colormap = :magma)
    fig
end

function overlay(img, mask; colour = RGB(0.45, 0.18, 0.55), strength = 0.55)
    out = RGB{Float64}.(img)
    @inbounds for i in eachindex(out, mask)
        mask[i] == 0 && continue
        out[i] = RGB((1 - strength) .* (red(out[i]), green(out[i]), blue(out[i])) .+
                     strength .* (red(colour), green(colour), blue(colour))...)
    end
    out
end

"""
Mono 16-bit PCM WAV, as bytes. Written by hand rather than through FFMPEG_jll:
the header is 44 bytes and this keeps the package free of a media dependency it
would otherwise need for one file.

Bytes rather than a file because that is what the page wants — see the audio
GUI — and the clamp comes BEFORE the rounding: `round(Int16, x)` on a sample
past ±1.0 is an `InexactError`, so clamping the Int16 afterwards is too late to
save a model that overshoots.
"""
function wavbytes(audio::AbstractVector{<:Real}; rate::Integer = 24000)
    pcm = round.(Int16, clamp.(Float32.(audio), -1f0, 1f0) .* 32767)
    n = length(pcm) * 2
    io = IOBuffer()
    write(io, b"RIFF"); write(io, UInt32(36 + n)); write(io, b"WAVE")
    write(io, b"fmt "); write(io, UInt32(16)); write(io, UInt16(1)); write(io, UInt16(1))
    write(io, UInt32(rate)); write(io, UInt32(rate * 2)); write(io, UInt16(2)); write(io, UInt16(16))
    write(io, b"data"); write(io, UInt32(n)); write(io, pcm)
    take!(io)
end

"The same bytes on disk."
writewav(path, audio::AbstractVector{<:Real}; rate::Integer = 24000) =
    (write(path, wavbytes(audio; rate)); path)

"""
Where the speech model leaves its last utterance.

A fixed path on purpose: it gives the transcriber a default input that exists as
soon as anything has been spoken, so the two audio models chain without the user
having to find a file. Playback does not read it — see the audio GUI.
"""
lastspeech() = joinpath(tempdir(), "demoapp_kokoro.wav")

"""
Whether anything belonging to this runner has been constructed yet.

The first click on a model pays for weights and kernel compilation — 95 s for
Kokoro, 31 s for BasicVSR — and every later one is fast. Saying which of the two
is happening is the difference between "slow" and "hung".
"""
everbuilt(mod) = any(k -> startswith(String(k), String(nameof(mod)) * "_"), keys(BUILT))

"Run `m.run`, reporting the elapsed time or the error. Errors are logged and shown, never swallowed."
function attempt(m::Model, mod, inputs, st)
    st[] = everbuilt(mod) ? "running…" :
           "first run: loading weights and compiling kernels — this can take a minute…"
    try
        t = @elapsed out = m.run(mod, inputs)
        st[] = "$(round(t * 1000; digits = 1)) ms"
        return out
    catch err
        @error "$(m.name) failed" exception = (err, catch_backtrace())
        st[] = "failed: " * first(split(sprint(showerror, err), '\n'))
        return nothing
    end
end

"Masks from SAM 2, keyed by image, so the matting models have a seed."
const LASTMASK = Dict{String,Any}()

# ── GUI kinds ────────────────────────────────────────────────────────────────

function gui(::Val{:click_mask}, m, mod, live, session)
    picker = imagepicker()
    img = Observable{Any}(loadimage(picker.value[])); shown = Observable{Any}(img[])
    st = Observable(live ? "Click the thing you want. The first click builds the model." : "Disabled — see above.")
    on(picker.value) do f; img[] = loadimage(f); shown[] = img[]; end
    fig = imagepane(shown); ax = fig.content[1]
    live && on(events(ax.scene).mousebutton) do ev
        (ev.button == Mouse.left && ev.action == Mouse.press) || return Consume(false)
        pos = Makie.mouseposition(ax.scene); h, w = size(img[])
        # `segment` takes (x, y) in *matrix* order — x indexes rows, y columns.
        col, row = pos[1] / w, pos[2] / h
        (0 <= col <= 1 && 0 <= row <= 1) || return Consume(false)
        out = attempt(m, mod, (; img = img[], row, col, key = picker.value[]), st)
        if out !== nothing
            LASTMASK[picker.value[]] = out[2]
            shown[] = overlay(img[], out[2])
        end
        Consume(true)
    end
    DOM.div(label("image"), picker, fig, statusline(st))
end

function gui(::Val{:image}, m, mod, live, session)
    picker = imagepicker()
    img = Observable{Any}(loadimage(picker.value[]))
    out = Observable{Any}(img[]); heat = Observable{Any}(zeros(Float32, size(img[])))
    isheat = Observable(false); st = Observable(live ? "Ready." : "Disabled — see above.")
    btn = Button("Run"; disabled = !live)
    on(picker.value) do f; img[] = loadimage(f); out[] = img[]; isheat[] = false; end
    on(btn.value) do _
        r = attempt(m, mod, (; img = img[]), st); r === nothing && return
        r[1] === :heat ? (heat[] = asheat(r[2]); isheat[] = true) : (out[] = asimage(r[2]); isheat[] = false)
    end
    DOM.div(label("image"), picker, DOM.div(btn; style = "margin:12px 0"),
            DOM.div(lift(h -> h ? heatpane(heat) : imagepane(out), isheat)), statusline(st))
end

function gui(::Val{:two_frames}, m, mod, live, session)
    fs = images(); a = Dropdown(fs; index = 1); b = Dropdown(fs; index = min(2, length(fs)))
    t = Slider(0:0.05:1; value = 0.5); out = Observable{Any}(loadimage(a.value[]))
    st = Observable(live ? "Ready." : "Disabled — see above."); btn = Button("Run"; disabled = !live)
    on(btn.value) do _
        r = attempt(m, mod, (; a = loadimage(a.value[]), b = loadimage(b.value[]), t = t.value[]), st)
        r !== nothing && (out[] = asimage(r[2]))
    end
    DOM.div(label("frame a"), a, DOM.div(label("frame b"), b; style = "margin-top:10px"),
            DOM.div(label("t"), t; style = "margin-top:10px"),
            DOM.div(btn; style = "margin:12px 0"), imagepane(out), statusline(st))
end

function gui(::Val{:image_mask}, m, mod, live, session)
    picker = imagepicker(); out = Observable{Any}(loadimage(picker.value[]))
    st = Observable(live ? "Needs a seed mask — segment this image with SAM 2 first." : "Disabled — see above.")
    btn = Button("Run"; disabled = !live)
    on(btn.value) do _
        mask = get(LASTMASK, picker.value[], nothing)
        if mask === nothing
            st[] = "No seed mask for this image yet — pick it under SAM 2.1 and click the subject."
            return
        end
        r = attempt(m, mod, (; img = loadimage(picker.value[]), mask), st)
        r !== nothing && (out[] = asimage(r[2]))
    end
    DOM.div(label("image"), picker, DOM.div(btn; style = "margin:12px 0"), imagepane(out), statusline(st))
end

function gui(::Val{:prompt_text}, m, mod, live, session)
    prompt = TextField("Explain what a finite element mesh is, in two sentences.")
    n = Slider(8:8:256; value = 64); out = Observable("")
    st = Observable(live ? "Ready." : "Disabled — see above."); btn = Button("Generate"; disabled = !live)
    on(btn.value) do _
        r = attempt(m, mod, (; prompt = prompt.value[], maxtokens = n.value[]), st)
        r !== nothing && (out[] = r[2])
    end
    DOM.div(label("prompt"), prompt, DOM.div(label("max tokens"), n; style = "margin-top:10px"),
            DOM.div(btn; style = "margin:12px 0"),
            DOM.pre(out; style = "background:$BG;border:1px solid $EDGE;border-radius:6px;padding:12px;" *
                                 "min-height:120px;white-space:pre-wrap;font-size:13px;color:$INK"),
            statusline(st))
end

function gui(::Val{:prompt_audio}, m, mod, live, session)
    prompt = TextField("Ferrite meshes, rendered at last.")
    voice = TextField("af_heart"); speed = Slider(0.5:0.1:2.0; value = 1.0)
    src = Observable("")
    st = Observable(live ? "Ready." : "Disabled — see above."); btn = Button("Speak"; disabled = !live)
    on(btn.value) do _
        r = attempt(m, mod, (; prompt = prompt.value[], voice = voice.value[], speed = speed.value[]), st)
        r === nothing && return
        # `Bonito.url` has no single-argument method, so `url(Asset(path))` was
        # a `MethodError` on every click and no audio ever reached the page. It
        # takes the session's asset server, which serves a `BinaryAsset` under
        # its own `mime` with range support — what `<audio>` needs to play and
        # seek. Bytes rather than a temp file also key the URL by content hash,
        # so a second utterance cannot be served from the cache of the first,
        # which one fixed path would have guaranteed.
        src[] = Bonito.url(session, Bonito.BinaryAsset(wavbytes(r[2]), "audio/wav"))
        writewav(lastspeech(), r[2])        # so the transcriber has something to read
        st[] = st[] * "  ·  $(round(length(r[2]) / 24000; digits = 2)) s of audio"
    end
    DOM.div(label("text"), prompt,
            DOM.div(label("voice"), voice; style = "margin-top:10px"),
            DOM.div(label("speed"), speed; style = "margin-top:10px"),
            DOM.div(btn; style = "margin:12px 0"),
            DOM.audio(; controls = true, src = src, style = "width:100%"),
            statusline(st))
end

function gui(::Val{:prompt_image}, m, mod, live, session)
    prompt = TextField("a brushed steel bracket under raking light")
    steps = Slider(4:2:50; value = 16)
    out = Observable{Any}(fill(RGB(0.97, 0.96, 0.99), 64, 64))
    st = Observable(live ? "Ready." : "Disabled — see above."); btn = Button("Generate"; disabled = !live)
    on(btn.value) do _
        # `status` rides along so a model that runs for minutes can say which
        # stage it is in; `attempt` overwrites it with the elapsed time at the
        # end, and a run that ignores the field is unaffected.
        r = attempt(m, mod, (; prompt = prompt.value[], steps = steps.value[], status = st), st)
        r !== nothing && (out[] = asimage(r[2]))
    end
    DOM.div(label("prompt"), prompt, DOM.div(label("steps"), steps; style = "margin-top:10px"),
            DOM.div(btn; style = "margin:12px 0"), imagepane(out), statusline(st))
end

function audiogui(m, mod, live, session, verb, outnote)
    path = TextField(lastspeech())
    out = Observable("")
    st = Observable(live ? "Ready." : "Disabled — see above."); btn = Button(verb; disabled = !live)
    # The result was dropped here, so a model that worked looked identical to one
    # that did nothing.
    on(btn.value) do _
        r = attempt(m, mod, (; path = path.value[]), st)
        r !== nothing && (out[] = r[2] isa AbstractString ? r[2] : string(r[2]))
    end
    DOM.div(label("audio file"), path, DOM.div(btn; style = "margin:12px 0"),
            DOM.pre(out; style = "background:$BG;border:1px solid $EDGE;border-radius:6px;padding:12px;" *
                                 "min-height:80px;white-space:pre-wrap;font-size:13px;color:$INK"),
            note(outnote), statusline(st))
end
gui(::Val{:audio_text}, m, mod, live, session)  = audiogui(m, mod, live, session, "Transcribe", "Output is a transcript.")
gui(::Val{:audio_audio}, m, mod, live, session) = audiogui(m, mod, live, session, "Process", "Output is one file per stem.")
gui(::Val{k}, m, mod, live, session) where {k}  = DOM.div(note("No controls defined for kind :$k yet."))

# ── panel & app ──────────────────────────────────────────────────────────────

function panel(session, m::Model)
    state, mod = status(m)
    text, colour = STATE_PILL[state]
    live = state === :ready
    head = DOM.div(pill(text, colour), DOM.span(m.blurb; style = "font-size:14px;color:$INK;margin-left:12px");
                   style = "display:flex;align-items:center;margin-bottom:6px;flex-wrap:wrap")
    why = live ? DOM.div() :
          state === :loaderror ?
              DOM.pre(sprint(showerror, mod);
                      style = "font-size:12px;color:$MUTED;white-space:pre-wrap;max-height:160px;overflow:auto") :
              note(WHY[state])
    card(head, why, DOM.div(gui(Val(m.kind), m, mod, live, session);
                            style = "margin-top:14px;border-top:1px solid $EDGE;padding-top:14px"))
end

function playground()
    App() do session
        picker = Dropdown([m.name for m in MODELS])
        body = Observable{Any}(panel(session, MODELS[1]))
        on(picker.option_index) do i; body[] = panel(session, MODELS[i]); end
        counts = Dict{Symbol,Int}()
        for m in MODELS; s = first(status(m)); counts[s] = get(counts, s, 0) + 1; end
        tally = join(["$v $(lowercase(first(STATE_PILL[k])))" for (k, v) in sort(collect(counts), by = first)], " · ")
        header = DOM.div(
            DOM.div("JuliaVision"; style = "font:600 12px/1 ui-monospace,monospace;letter-spacing:2px;color:$ACCENT"),
            DOM.h1("$(length(MODELS)) models, one stack";
                   style = "font:700 28px/1.2 system-ui;color:$INK;margin:8px 0 6px"),
            DOM.div(tally; style = "font-size:13px;color:$MUTED"), style = "margin-bottom:22px")
        side = DOM.div(label("model"), picker,
                       DOM.div(lift(i -> MODELS[i].task, picker.option_index);
                               style = "font-size:13px;color:$MUTED;margin-top:12px"),
                       style = "width:250px;flex:none")
        DOM.div(header, DOM.div(side, DOM.div(body; style = "flex:1;min-width:0");
                                style = "display:flex;gap:28px;align-items:flex-start"),
                style = "font-family:system-ui;background:$BG;padding:32px;min-height:100vh")
    end
end

serve(; host = "127.0.0.1", port = 8080) = Bonito.Server(playground(), host, port)

end # module
