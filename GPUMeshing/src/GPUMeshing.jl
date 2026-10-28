"""
Volume to mesh on the GPU, through Mantle.

Two extractors and the device pieces they are built from:

- [`marchingcubes`](@ref): the iso-surface of a dense scalar grid, on the device
  or the host, the same code either way. [`MarchingCubes`](@ref) keeps everything
  it needs for a field that changes every frame.
- [`dualcontour`](@ref): the zero set of a field sampled at the corners of a
  sparse set of voxels, one vertex per voxel. [`edgequads`](@ref) is its quad
  stage on its own, for a caller that places the vertices itself, and
  [`meshdual`](@ref) the whole mesh from such vertices, as Hermite dual contouring
  places them with [`qefsolve`](@ref), keeping sharp edges and corners.
- [`dualmc`](@ref): dual marching cubes with Hermite data, one vertex per patch of a voxel:
  sharp edges and corners, and a closed two-manifold however the signs fall.
- [`simplify`](@ref): parallel edge collapses, to a face count or within a distance.
- [`SurfaceGrid`](@ref): closest points and [`distances`](@ref) to a triangle mesh.
- [`areavolume`](@ref) and [`closedmanifold`](@ref): a mesh's invariants, checked on the device.
- Connectivity ([`Edges`](@ref), [`components`](@ref)) and cumesh's clean-up: [`fillholes`](@ref),
  [`removeduplicates`](@ref), [`removedegenerate`](@ref), [`removeunreferenced`](@ref),
  [`orientfaces`](@ref), [`splitnonmanifold`](@ref), [`removesmallcomponents`](@ref).
- Tetrahedral meshes: their faces and edges and which bound the mesh ([`tetboundary`](@ref)),
  and which way a boundary face points ([`faceorientation`](@ref)).
- [`volumemoments`](@ref) and [`tetvolumemoments`](@ref): volume, first and second moments of
  a solid from its closed surface or its tetrahedra.
- [`joinparts`](@ref) and [`splitparts`](@ref): several meshes as one and back.

The pieces underneath are public for the packages that build their own passes on
them: a GPU [`HashMap`](@ref) over grid coordinates, prefix sums, stream
compaction and runs of sorted keys, and [`runonce!`](@ref).
"""
module GPUMeshing

import Mantle
import KernelInterface as KI
import Atomix
import AcceleratedKernels as AK
import GeometryBasics
using GeometryBasics: Point3f, Vec3f, Vec, Mat, GLTriangleFace, GLIndex, coordinates, faces, normals
using LinearAlgebra: cross, dot, norm

export marchingcubes, marchingcubes!, MarchingCubes, dualcontour, SparseGrid, voxelcorners, edgequads, HashMap
export meshdual, dualmc, DualPatches, qefsolve, simplify, SurfaceGrid, distances, areavolume, closedmanifold, windingnumbers, edgecrossings
export Edges, edgelist, components, orientedcomponents, manifoldfaceadjacency, edgeswithfaces, VertexBoundaries, manifoldboundaryadjacency
export fillholes, removeduplicates, removedegenerate, removeunreferenced, orientfaces, splitnonmanifold, removesmallcomponents
export TetFaces, TetBoundary, tetboundary, faceorientation, VolumeMoments, volumemoments, tetvolumemoments, joinparts, splitparts
public runonce!, prefixsum, lastentry, select, compactcolumns, compactcols!, copycols!, iota!, runs, runstarts!, vertex, point
public edgekey, edgelow, edgehigh, edgeface, edgeindex, facetet, sorted3
public hashlookup, lookupcoord, gridkey, aroundoffset, MCTABLE, DMCTABLE, cellkey, patchof, patchcount, CORNER, EDGE, Sym3, symeigen, nearest, closestpoint

include("device.jl")
include("hashmap.jl")
include("marchingcubes.jl")
include("dualgrid.jl")
include("qef.jl")
include("dualmc.jl")
include("surface.jl")
include("simplify.jl")
include("checks.jl")
include("intersect.jl")
include("topology.jl")
include("cleanup.jl")
include("tetmesh.jl")
include("moments.jl")
include("parts.jl")

end # module
