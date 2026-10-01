"""
TRELLIS.2's graphs and weights, the artifacts bound in this package's
`Artifacts.toml`, downloaded on first use. A re-export reaches the runner by
re-binding them with `tools/make_artifacts.jl`, never by reading `gen/`.

A part is `"cond"` (DINOv3), one of the five [`FLOWS`](@ref), `"ssdec"` (the
structure decoder), `"shapedec"` or `"texdec"`. Its graph is
`trellis2_<part>.json` in its [`graphdir`](@ref) (DINOv3's at 1024 px is part
`"cond-1024"`), and its weights are the tensors of every file in
[`weightfiles`](@ref) together. A part's artifacts are fetched when the part is
loaded, so the 512 pipeline never downloads the 1024 models.

The artifacts: `trellis2`, the graph tree laid out like `gen/graphs` (with the
structure decoder's weights); `trellis2-cond-w`, DINOv3's weights; each flow's
weights in two shards; and upstream's two sparse-decoder checkpoints.
"""

"""The five flow models, whose weights are two shards each."""
const FLOWS = ("ss", "shape512", "shape1024", "tex512", "tex1024")

"""Each part's weights: `artifact => file inside it`."""
const WEIGHTS = Dict{String,Vector{Pair{String,String}}}(
    (p => ["trellis2-$p-w$i" => "weights-$(i)of2.safetensors" for i in 1:2] for p in FLOWS)...,
    "cond" => ["trellis2-cond-w" => "weights-1of1.safetensors"],
    "ssdec" => ["trellis2" => "trellis2-ssdec/weights.safetensors"],
    "shapedec" => ["trellis2-shapedec" => "shape_dec_next_dc_f16c32_fp16.safetensors"],
    "texdec" => ["trellis2-texdec" => "tex_dec_next_dc_f16c32_fp16.safetensors"])

const ARTIFACTS_TOML = normpath(joinpath(@__DIR__, "..", "Artifacts.toml"))

"""The directory of artifact `name`, downloading it if it is not in the store yet."""
artifact(name::AbstractString) = @artifact_str(name)

"""The export directory of `part`, holding its graph (and `sampling.json`)."""
graphdir(part::AbstractString) = joinpath(artifact("trellis2"), "trellis2-$part")

"""The safetensors files whose tensors together are `part`'s weights."""
weightfiles(part::AbstractString) = [joinpath(artifact(n), f) for (n, f) in WEIGHTS[part]]

"""`part`'s weights. The shards partition the tensors, so merging them is the checkpoint."""
readweights(part::AbstractString) = merge((readsafetensors(f) for f in weightfiles(part))...)

"""The parts a pipeline loads: the 512 one, or with `cascade` upstream's `1024_cascade`."""
parts(cascade::Bool) = cascade ? ["cond", "ss", "ssdec", "shape512", "shape1024", "tex1024", "shapedec", "texdec"] :
                                 ["cond", "ss", "ssdec", "shape512", "tex512", "shapedec", "texdec"]

"""
    ready(; cascade = true) -> Bool

Whether every artifact the pipeline needs is already in the store. Does not go
through `@artifact_str`, which downloads: up to 15 GB to answer a question.
"""
function ready(; cascade::Bool = true)
    names = unique(vcat(["trellis2"], [first(w) for part in parts(cascade) for w in WEIGHTS[part]]))
    return all(names) do n
        h = artifact_hash(n, ARTIFACTS_TOML)
        h !== nothing && artifact_exists(h)
    end
end
