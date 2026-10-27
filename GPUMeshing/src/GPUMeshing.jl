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

The pieces underneath are public for the packages that build their own passes on
them: a GPU [`HashMap`](@ref) over grid coordinates, prefix sums and stream
compaction, and [`runonce!`](@ref).
"""
module GPUMeshing

import Mantle
import KernelInterface as KI
import Atomix
import AcceleratedKernels as AK
import GeometryBasics
using GeometryBasics: Point3f, Vec3f, GLTriangleFace, GLIndex, coordinates, faces, normals
using LinearAlgebra: cross, dot, norm

export marchingcubes, marchingcubes!, MarchingCubes, dualcontour, SparseGrid, voxelcorners, edgequads, HashMap
export meshdual, dualmc, DualPatches, qefsolve, simplify, SurfaceGrid, distances, areavolume, closedmanifold, windingnumbers, edgecrossings
public runonce!, prefixsum, lastentry, select, compactcolumns, compactcols!, vertex
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

end # module
