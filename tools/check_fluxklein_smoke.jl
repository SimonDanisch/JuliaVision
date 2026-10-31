# The FLUX.2-klein smoke graphs (tools/export_fluxklein.py --smoke) on the device,
# against the torch references written next to them. The first thing a port
# answers: does every op the export produced run here, and to what precision.
#
#   using Mantle; Mantle.defaultdevice!("NVIDIA")   # not the 7900 XTX, it drives the desktop
#   include("tools/check_fluxklein_smoke.jl")
#   checksmoke("/tmp/claude-1000/fluxklein_smoke")

using DNNKernels, Mantle
using DNNKernels: loadgraph, readsafetensors, Model, call, toback

"Which graph reads which reference tensors, and what it is checked against."
const SMOKE = (
    encoder = (weights = "encoder_smoke_weights", reference = "encoder_reference",
               inputs = ("inputs_embeds", "cos", "sin", "mask"), output = "prompt_embeds", dims = (;)),
    transformer = (weights = "transformer_smoke_weights", reference = "transformer_reference",
                   inputs = ("latents", "prompt_embeddings", "timestep_proj", "rope_cos", "rope_sin"),
                   output = "output", dims = (i = 6, t = 5, tj = 11)),
    vae_decoder = (weights = "vae_decoder_smoke_weights", reference = "vae_decoder_reference",
                   inputs = ("latents",), output = "image", dims = (h = 3, w = 2)),
    vae_encoder = (weights = "vae_encoder_smoke_weights", reference = "vae_encoder_reference",
                   inputs = ("image",), output = "latents", dims = (h = 3, w = 2)),
)

"Run graph `name` of the smoke export in `dir`; returns `(got, want, maxabs, relative)`."
function checkgraph(dir::AbstractString, name::Symbol; backend = Mantle.defaultbackend())
    s = SMOKE[name]
    graph = loadgraph(joinpath(dir, "fluxklein_$name.json"))
    weights = readsafetensors(joinpath(dir, "$(s.weights).safetensors"))
    ref = readsafetensors(joinpath(dir, "$(s.reference).safetensors"))
    model = Model(Dict("fluxklein_$name" => graph), weights; backend)
    args = [toback(backend, ref[k]) for k in s.inputs]
    got = Array(only(call(model, "fluxklein_$name", args...; dims = s.dims)))
    want = ref[s.output]
    maxabs = maximum(abs.(Float32.(got) .- Float32.(want)))
    relative = maxabs / maximum(abs.(Float32.(want)))
    @info "fluxklein smoke" name size(got) size(want) maxabs relative
    return got, want, maxabs, relative
end

checksmoke(dir::AbstractString; parts = (:encoder, :transformer, :vae_decoder, :vae_encoder)) =
    Dict(p => checkgraph(dir, p)[3:4] for p in parts)
