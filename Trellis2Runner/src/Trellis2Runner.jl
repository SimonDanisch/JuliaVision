"""
TRELLIS.2 image-to-3D on Lava.

`Trellis2ImageTo3DPipeline.run` with `pipeline_type` 512 or 1024_cascade:
DINOv3 conditions a dense flow over a 16^3 latent, whose decoded occupancy gives
the voxels; SLat flow models sample a shape latent and a texture latent on those
voxels, and two sparse decoders turn them into a mesh with PBR attributes.

The flow models and the conditioner are graphs exported by
`tools/export_trellis2.py`, each with its `sampling.json`; the sampler, the voxel
bookkeeping and (to come) the sparse decoders are written here.
"""
module Trellis2Runner

using Lava, DNNKernels
import Mantle
import JSON3
import KernelInterface as KI
import Atomix
import Random
import DataStructures
import SparseArrays
using GeometryBasics: GeometryBasics, Point3f, Vec, Vec2f, Vec3, Vec3f, GLTriangleFace
using ColorTypes: RGB, RGBA, N0f8
using LinearAlgebra: cross, dot, norm, normalize
using DNNKernels: loadgraph, readsafetensors, Model, call, linspace, emitgraph!

export FlowSampler, tschedule, guidanceruns, sampleflow!
export occupiedcoords, ropetable
export Normalization, FlowStage, loadmodel, encodeimage
export sparsestructure, shapeslat, texslat
export SparseDecoder, decode, upsample, cascadecoords, dualgridmesh, remesh, orientfaces, fillholes, simplify, cleanmesh, texturedmesh
export cropobject, dinoinput
export Trellis2, Cascade, Noise, generate

include("sampler.jl")
include("structure.jl")
include("pipeline.jl")
include("decoder.jl")
include("mesh.jl")
include("image.jl")
include("simplify.jl")
include("cleanup.jl")
include("lscm.jl")
include("piecewise.jl")
include("pack.jl")
include("surface.jl")
include("remesh.jl")
include("texture.jl")
include("generate.jl")

end # module
