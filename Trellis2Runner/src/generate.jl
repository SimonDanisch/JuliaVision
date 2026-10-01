"""
    Cascade(shape)

What `1024_cascade` adds to the 512 models: the 1024 shape flow, which samples
the latent again on the 64^3 tokens the 512 latent's decoder subdivides into.
DINOv3 at 1024 px is the conditioner's second graph ([`conditioner`](@ref)).
"""
struct Cascade
    shape::FlowStage
end

"""
    Trellis2(; backend = Mantle.defaultbackend(), cascade = true)

TRELLIS.2 loaded, as upstream's default `1024_cascade` pipeline or, with
`cascade = false`, the `512` one: the DINOv3 conditioner ([`conditioner`](@ref)),
the sparse-structure flow and its decoder, the 512 shape flow, the texture flow
of the output resolution, the two sparse decoders and, for the cascade, its
[`Cascade`](@ref) model. The artifacts download on first use (see
[`ready`](@ref)).
"""
struct Trellis2{C<:Union{Nothing,Cascade}}
    cond::Model
    ss::FlowStage
    ssdec::Model
    shape::FlowStage
    tex::FlowStage
    shapedec::SparseDecoder
    texdec::SparseDecoder
    cascade::C
end

function Trellis2(; backend = Mantle.defaultbackend(), cascade::Bool = true)
    ss = FlowStage("ss"; backend)
    dev = ss.model.device
    res = cascade ? 1024 : 512
    return Trellis2(conditioner(; cascade, backend), ss, loadmodel("ssdec"; backend),
                    FlowStage("shape512"; backend), FlowStage("tex$res"; backend),
                    SparseDecoder(dev, readweights("shapedec"); out = 7, subdiv = true),
                    SparseDecoder(dev, readweights("texdec"); out = 6, subdiv = false),
                    cascade ? Cascade(FlowStage("shape1024"; backend)) : nothing)
end

"""The output grid: 512, or 1024 for the cascade."""
resolution(p::Trellis2) = resolution(p.cascade)
resolution(::Nothing) = 512
resolution(::Cascade) = 1024

"""
    Noise(ss, shape, tex)

The three stages' starting noise: `ss` `(16, 16, 16, 8, 1)`, and a function of
the voxel count for the two latents, `(32, N, 1)` each.
"""
struct Noise{S,T,X}
    ss::S
    shape::T
    tex::X
end

"""Standard normal noise for every stage from one seeded generator."""
function Noise(seed::Integer)
    rng = Random.Xoshiro(seed)
    ss = randn(rng, Float32, 16, 16, 16, 8, 1)
    shape = n -> randn(rng, Float32, 32, n, 1)
    tex = n -> randn(rng, Float32, 32, n, 1)
    return Noise(ss, shape, tex)
end

"""
    shapelatent(p, rgba, cond, coords, noise) -> (shape, coords, cond)

The shape latent, the voxels it is on and the condition the texture flow takes.
At 512 one flow over the sparse structure. The cascade samples that latent too,
subdivides it with the shape decoder ([`upsample`](@ref)), and samples the 1024
latent on the resulting 64^3 tokens ([`cascadecoords`](@ref)) from DINOv3's
1024 px features.
"""
shapelatent(p::Trellis2, rgba, cond, coords, noise) = shapelatent(p, p.cascade, rgba, cond, coords, noise)

shapelatent(p::Trellis2, ::Nothing, rgba, cond, coords, noise) =
    (shapeslat(p.shape, cond, coords, noise.shape(size(coords, 2))), coords, cond)

function shapelatent(p::Trellis2, c::Cascade, rgba, cond, coords, noise)
    lr = shapeslat(p.shape, cond, coords, noise.shape(size(coords, 2)))
    hr = cascadecoords(upsample(p.shapedec, Array(Mantle.storage(lr))[:, :, 1], coords[2:4, :]);
                       resolution = resolution(c))
    cond1024 = Array(encodeimage(p.cond, dinoinput(rgba; size = 1024); graph = "trellis2_cond_1024"))
    return shapeslat(c.shape, cond1024, hr, noise.shape(size(hr, 2))), hr, cond1024
end

"""
    generate(p, rgba; noise = Noise(42), texture = 4096, decimation = 500_000) -> MetaMesh

`Trellis2ImageTo3DPipeline.run` for an image with an alpha channel (`(4, W, H)`
UInt8), at the pipeline `p` was loaded as, then `o_voxel.postprocess.to_glb`
as upstream's example calls it: the conditioning, the sparse structure, the
shape latent ([`shapelatent`](@ref)) and the texture latent, both decoders, the
mesh (small holes filled, rebuilt as a closed shell by [`remesh`](@ref), then
simplified towards `decimation` faces) and its PBR textures at `texture`^2,
sampled on the mesh before remeshing (see [`texturedmesh`](@ref)).
`save("model.glb", generate(p, rgba))` writes it.
"""
function generate(p::Trellis2, rgba::AbstractArray{UInt8,3}; noise = Noise(42), texture::Int = 4096,
                  decimation::Int = 500_000)
    cond = Array(encodeimage(p.cond, dinoinput(rgba)))
    coords = sparsestructure(p.ss, p.ssdec, cond, noise.ss)
    shape, coords, cond = shapelatent(p, rgba, cond, coords, noise)
    tex = texslat(p.tex, cond, coords, shape, noise.tex(size(coords, 2)))
    voxels = coords[2:4, :]
    res = resolution(p) ÷ 16
    geometry = decode(p.shapedec, Array(Mantle.storage(shape))[:, :, 1], voxels; res)
    material = decode(p.texdec, Array(Mantle.storage(tex))[:, :, 1], voxels; res, guide = geometry)
    level = geometry.levels[end]
    dev = p.shapedec.dev
    V, F = dualgridmesh(dev, level, geometry.out)
    # Filled twice, as upstream does: in `decode_latent` and again at the start of
    # `to_glb`. The first pass's fans close off boundary chains that were not
    # holes yet; the second fills those (47 on the 1024 example).
    V, F = fillholes(dev, fillholes(dev, V, F)...)
    # The filled mesh is what the remesh's distances and the bake's projection
    # query; both bucket it on the host ([`SurfaceGrid`](@ref)).
    surface = (download(V), download(F))
    foreach(Mantle.free!, (V, F))
    # `to_glb(remesh = True, remesh_band = 1)`: a grid 3 voxels wider than the
    # cube, so the shell around the outermost voxels fits.
    V, F = simplify(dev, remesh(dev, surface...; resolution = level.res, scale = (level.res + 3) / level.res)...,
                    decimation)
    V, F = removedegenerate(dev, V, F)
    vertices, faces = download(V), download(F)
    foreach(Mantle.free!, (V, F))
    attrs = clamp.(material.out .* 0.5f0 .+ 0.5f0, 0f0, 1f0)
    return texturedmesh(dev, vertices, faces, level, attrs; surface, size = texture, doublesided = false)
end
