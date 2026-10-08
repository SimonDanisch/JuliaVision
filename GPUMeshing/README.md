# GPUMeshing.jl

Volume to mesh on the GPU, through Mantle: marching cubes over a dense field,
dual contouring over a sparse set of voxels, and the GPU hash map, scans and
compaction they are built from.

Hunyuan3DRunner extracts its occupancy field with `marchingcubes`, on the device
where the decoder left it. Trellis2Runner's `remesh` is `dualcontour` over a
narrow band, and its `dualgridmesh` uses the quad stage, `edgequads`.

## Marching cubes

`marchingcubes` returns a `GeometryBasics.Mesh` of `Point3f` positions,
`GLTriangleFace` faces and `Vec3f` normals, so Makie draws it as it is:

```julia
using Lava, Mantle, GPUMeshing, GeometryBasics
m = marchingcubes(sdf; level = 0, origin = Vec3f(-1), spacing = Vec3f(2 / (size(sdf, 1) - 1)))
mesh!(ax, m)

dev = Mantle.Device()
field = Mantle.Buffer(dev, sdf)                # (nx, ny, nz) Float32
m = marchingcubes(dev, field; level = 0)       # the same mesh, its arrays on the device
```

From the device, the mesh's arrays are `LavaArray`s and the mesh owns them. For a
field that changes every frame, set the extraction up once and rerun it:

```julia
mc = MarchingCubes(dev, field; origin, spacing)
surface = Observable(marchingcubes!(mc; level = 0))
mesh!(ax, surface)
for frame in frames
    simulate!(field, frame)                    # writes the field on the device
    surface[] = marchingcubes!(mc; level = 0)
end
```

A frame is one submission of a plan recorded once, and one read of the two
counts. `marchingcubes!` returns device views of the workspace's own buffers, no
copy, valid until the next call. The buffers grow when a frame needs more room
and are reused otherwise.

Grid point `(i, j, k)` sits at `origin + spacing * (i - 1, j - 1, k - 1)`; the
defaults give grid index space, zero-based, as `skimage.measure.marching_cubes`
returns vertices. There is one vertex per crossed grid edge, shared by every cell
around it. Faces point toward decreasing values, and so do the normals: the
field's gradient by central differences, interpolated along each edge, scaled by
`spacing` and normalised.

The case table is derived from the cube's face contours when the package
compiles, not transcribed. Ambiguous faces are decided by the asymptotic decider,
so the table has a column for every sign pattern and every answer on the six
faces. Winding comes from the table too, which keeps it consistent across cells.

### Drawing it with RayMakie

In RayMakie's default traced mode, `mesh!` takes the device mesh, and the
picture is the same as from host arrays. The BLAS is built on the GPU, but what
feeds it is prepared on the host: the mesh is decomposed and its per-triangle
shading data built in a CPU loop,
so a new mesh costs RayMakie a download and a rebuild. On a Radeon RX 7900 XTX,
with the mesh changing every frame:

| grid | vertices | extraction | RayMakie, new mesh and one frame | RayMakie, one frame |
|---|---|---|---|---|
| 128³ | 29 000 | 0.5 ms | 100 ms | 26 ms |
| 256³ | 117 000 | 1 ms | 200 to 240 ms | 23 to 47 ms |

About half of a new-mesh frame is that rebuild, in Mantle's
`hwtlas_add_geometry!`, Hikari's `push!` and RayMakie's `push_to_scene`.
Keeping the mesh on the device needs a device path there: the BLAS built from
the mesh's own buffers, and the per-triangle data built by a kernel.

RayMakie's raster mode (`rasterize = true`) keeps the mesh on the device:
positions, faces and normals are copied into the stage's buffers on the GPU
(`raster_faces`, `raster_normals` and `cornerbounds` in RayMakie's
`plots/mesh.jl`, and a `decompose` that passes device faces through, in its
`gpu_arrays.jl`). Measured before the faces and normals stayed on the device, a
frame with a new mesh, marching cubes included, was 5.5 ms at 128³ and 256³ and
10 ms at 385³, against 2 to 4.5 ms for an unchanged one; their round trip was
about 1 ms of that at 256³.

### How the device path is fast

Each workgroup covers a brick of 32×4×2 grid points. One pass counts every
brick's vertices, reading each value of the field about once. Only bricks near
the surface go further: those with vertices, and those whose upper neighbours
have some. A small scan lists them on the device, and the passes that count
triangles and write the mesh run on that list alone, sized on the device with no
round trip to the host. There is no scan over the whole grid.

On a Radeon RX 7900 XTX, with a sphere-like surface:

| grid | vertices | per frame | one-shot |
|---|---|---|---|
| 128³ | 27 832 | 0.44 ms | about 2 ms |
| 256³ | 112 160 | 0.84 ms | about 3 ms |
| 385³ | 254 478 | 1.5 ms | 6 to 9 ms |

At 385³ the count pass is 0.55 ms of that, about the cost of reading the 228 MB
field once, and the normals add 0.1 ms to the vertex pass. Up to 256³ the field fits in the card's 96 MB Infinity Cache, and the
count drops to 0.1 ms. The one-shot form also allocates its scratch and records
its plan twice, once to count and once at the exact size. The host path takes 0.22 s
at 385³ on 24 threads. `benchmark/marchingcubes.jl` measures all of this and
prints the time of every pass.

Hunyuan3DRunner's previous CPU version took 0.35 s at 385³, plus 0.19 s to
download the field. It joined the wrong diagonal pair on ambiguous faces, and it
wound each triangle by the local gradient. On noisy fields the second left about
a thousand inconsistently wound edges per 14×11×9 grid. Both are fixed here, and
`test/reference.jl` keeps its loop walk, with the decider fixed, as an
independent check of the vertices and triangles.

## Dual contouring

```julia
grid = SparseGrid(dev, coords, n, res)    # n active voxels, (3, n) Int32, zero-based in res³
corners, m = voxelcorners(dev, grid)      # their corners, (3, m) Int32
values = ...                              # the field at each corner, (m,) Float32, yours to compute
V, F = dualcontour(dev, grid, corners, values, m; origin, cell)
```

This is cumesh's `simple_dual_contour_kernel`. Each voxel gets one vertex at the
mean of its edges' crossings. Each crossed edge whose four voxels are all present
gets a quad, wound to face the positive side. `edgequads` is that quad stage on
its own, for a caller that places the vertices itself.

## Tests

`test/runtests.jl` checks that every case of the table closes, and that the
decider agrees with a flood fill of the bilinear interpolant. It checks a sphere
for watertightness, winding, area and volume, and noisy fields for closed,
consistent winding. The host path is compared with the reference loop walk, and
the device path with the host path face for face. A workspace is run over a
field rewritten between frames, through an empty frame and through growth. Dual
contouring is checked on a sphere's distance over a narrow band.
