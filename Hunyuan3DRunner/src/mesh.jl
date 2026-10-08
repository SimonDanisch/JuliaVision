"""
The implicit field to a triangle mesh.

`MCSurfaceExtractor` upstream is `skimage.measure.marching_cubes(..., method =
"lewiner")` followed by a rescale from grid indices into the sampling box. This is
the same two steps: [`GPUMeshing.marchingcubes`](@ref) on the device, where
[`occupancy`](@ref) left the field, then [`meshbox`](@ref).
"""

"""
    meshbox(vertices, grid, octree; box_v = 1.01) -> vertices

`MCSurfaceExtractor.run`'s rescale from grid indices into the sampling box:

    vertices = vertices / grid_size * bbox_size + bbox_min

**Divided by `grid_size`, which is `octree + 1`, not by `octree`.** The grid's
last sample sits exactly on the box's far face, so mapping index `octree` back to
`box_v` would divide by `octree`. Upstream divides by one more than that, which
shrinks the mesh by a factor of `octree / (octree + 1)` — 0.26% at the default
384 — and shifts it. It is reproduced because it is what the reference mesh is,
not because it is right.
"""
function meshbox(verts::AbstractArray{<:Real,2}, octree::Integer; box_v::Real = 1.01)
    G = Float32(octree + 1)
    lo, size_ = Float32(-box_v), Float32(2box_v)
    return @. Float32(verts) / G * size_ + lo
end

"""
    latents2mesh(m, latents; box_v, octree, level, progress) -> (vertices, faces)

`ShapeVAE.latents2mesh`: sweep the geometry decoder over the grid, then extract
the surface. `vertices` is `(3, V)` in the sampling box, `faces` is `(3, F)`.

The extraction runs on the device, on the field where `occupancy` left it; only
the mesh comes back to the host.

**The axes are reversed on the way out.** [`occupancy`](@ref) returns the field in
the runtime's layout, where `field[k, j, i]` is upstream's `grid_logits[i, j, k]`,
so marching cubes over it produces vertices ordered `(z, y, x)`. Reversing the
three components is what puts them back in the `(x, y, z)` the box mapping
expects; skipping it yields a mesh that is a transpose of the right one, which is
a plausible-looking object rather than an error.

Reversing three components is a *reflection*, not a rotation, so every triangle
turns inside out with it. Swapping two indices of each face undoes that. Leaving
it out has a recognisable symptom — geometry that is exactly right and renders
black, or lit from within.
"""
function latents2mesh(m::Hunyuan3D, latents; box_v::Real = 1.01, octree::Integer = 384,
                      level::Real = 0.0, progress = nothing)
    field = occupancy(m, latents; box_v, octree, progress)
    surface = marchingcubes(Mantle.todevice(m.backend), field; level)
    Mantle.free!(field)
    verts = reinterpret(reshape, Float32, Array(coordinates(surface)))
    tris = Int32.(reinterpret(reshape, UInt32, Array(faces(surface)))) .+ Int32(1)   # zero-based on the device
    return meshbox(reverse(verts; dims = 1), octree; box_v), tris[[1, 3, 2], :]
end
