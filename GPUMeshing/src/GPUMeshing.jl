"""
Volume to mesh on the GPU, through Mantle.

Two extractors and the device pieces they are built from:

- [`marchingcubes`](@ref): the iso-surface of a dense scalar grid, on the device
  or the host, the same code either way. [`MarchingCubes`](@ref) keeps everything
  it needs for a field that changes every frame.
- [`dualcontour`](@ref): the zero set of a field sampled at the corners of a
  sparse set of voxels, one vertex per voxel. [`edgequads`](@ref) is its quad
  stage on its own, for a caller that places the vertices itself.

The pieces underneath are public for the packages that build their own passes on
them: a GPU [`HashMap`](@ref) over grid coordinates, prefix sums and stream
compaction, and [`runonce!`](@ref).
"""
module GPUMeshing

import Mantle
import KernelInterface as KI
import Atomix
import GeometryBasics
using GeometryBasics: Point3f, Vec3f, GLTriangleFace, GLIndex, coordinates, faces, normals
using LinearAlgebra: cross, dot

export marchingcubes, marchingcubes!, MarchingCubes, dualcontour, SparseGrid, voxelcorners, edgequads, HashMap
public runonce!, prefixsum, lastentry, select, compactcolumns, compactcols!, vertex
public hashlookup, lookupcoord, gridkey, aroundoffset, MCTABLE

include("device.jl")
include("hashmap.jl")
include("marchingcubes.jl")
include("dualgrid.jl")

end # module
