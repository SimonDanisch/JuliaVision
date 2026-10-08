"""
    remesh(dev, vertices, faces; resolution, scale, center = Vec3f(0), band = 1) -> (V, F)

cumesh's `remesh_narrow_band_dc`, what `to_glb(remesh = True)` rebuilds the
mesh with, on the device: the surface `band` voxels away from the mesh, the zero
set of its unsigned distance minus `band` voxels, extracted by dual contouring
on a `resolution`^3 grid over the cube of side `scale` around `center`. Only
voxels near that surface are visited, those whose centre is within 0.87 voxels
of it. `Mantle.Buffer`s out, `(3, V)` Float32 and `(3, F)` Int32.

Holes and slits narrower than about two voxels close, and the result is a
closed shell wound outward whatever the input was: the dual-grid mesh's open
sheets and edges of three or more faces are gone. A thin part becomes a shell
two voxels thick.

The active voxels: cumesh refines an octree from 32^3, keeping at every level
the cells whose centre is within 0.87 cells of the band surface by its BVH's
exact distance. [`SurfaceGrid`](@ref) answers only within one of its cells, so
the coarse levels are one step here: the cells of four voxels whose triangle
boxes, grown by the band and both half-diagonals, hold a triangle
([`markband!`](@ref)) expand into their voxels, and those within 0.87 voxels
of the band surface are the active set — cumesh's, since the box test takes a
superset of what its coarse levels keep. The band distance at their corners goes
to [`GPUMeshing.dualcontour`](@ref), which is cumesh's dual contouring: one vertex
per voxel at the mean of its edges' crossings, a quad of the four voxels around
every crossed edge, split as cumesh splits it.
"""
function remesh(dev::Mantle.Device, vertices::AbstractMatrix{Float32}, faces::AbstractMatrix{<:Integer};
                resolution::Int, scale::Real, center::Vec3f = Vec3f(0), band::Real = 1)
    res = resolution
    ratio = 4
    res % ratio == 0 || throw(ArgumentError("remesh: the resolution $res is not a multiple of $ratio"))
    cell = Float32(scale / res)
    eps = Float32(band) * cell
    lo = center .- Float32(scale) / 2
    # Distances are exact out to a cell of this grid, four voxels; the farthest
    # asked for is a band voxel's corner, under three.
    grid = SurfaceGrid(vertices, faces; res = cld(res, 4))
    sv, sf = Mantle.Buffer(dev, grid.vertices), Mantle.Buffer(dev, grid.faces)
    ss, st = Mantle.Buffer(dev, grid.start), Mantle.Buffer(dev, grid.tris)
    nf = size(faces, 2)
    # Coarse cells that may hold a band voxel, and their voxels.
    cres = res ÷ ratio
    ccell = cell * ratio
    ncell = cres^3
    mask = fill!(Mantle.Buffer(dev, Int32, ncell), 0)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, markband!, (mask, sv, sf, cres, lo[1], lo[2], lo[3], ccell,
                                    eps + 0.87f0 * cell + 0.87f0 * ccell, nf), nf; name = "remesh/band")
    runonce!(g)
    cpos = prefixsum(dev, mask)
    ncand = lastentry(cpos) * ratio^3
    cand = Mantle.Buffer(dev, Int32, (3, max(ncand, 1)))
    d = Mantle.Buffer(dev, Float32, max(ncand, 1))
    flags = Mantle.Buffer(dev, Int32, max(ncand, 1))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, expandcells!, (cand, mask, cpos, Int32(cres), Int32(ratio), ncell), ncell; name = "remesh/expand")
    Mantle.dispatch!(g, coorddistance!, (d, cand, 0.5f0, 0f0, cell, lo[1], lo[2], lo[3], sv, sf, ss, st, grid.res, ncand),
                     ncand; name = "remesh/centre distance")
    Mantle.dispatch!(g, nearband!, (flags, d, eps, 0.87f0 * cell, ncand), ncand; name = "remesh/active")
    runonce!(g)
    coords, n = compactcolumns(dev, cand, flags, ncand)
    foreach(Mantle.free!, (mask, cpos, cand, d, flags))
    # The voxels, and the band distance at their corners.
    vox = SparseGrid(dev, coords, n, res)
    corners, ncorner = voxelcorners(dev, vox)
    value = Mantle.Buffer(dev, Float32, ncorner)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, coorddistance!, (value, corners, 0f0, eps, cell, lo[1], lo[2], lo[3], sv, sf, ss, st, grid.res, ncorner),
                     ncorner; name = "remesh/corner distance")
    runonce!(g)
    foreach(Mantle.free!, (sv, sf, ss, st))
    V, F = dualcontour(dev, vox, corners, value, ncorner; origin = lo, cell)
    foreach(Mantle.free!, (vox, corners, value))
    return V, F
end

"""Mark, in the `res`^3 `mask`, every cell whose centre lies in a triangle's
bounding box grown by `reach`; the grid's corner is at `lo`, cells `cell` wide."""
function markband!(mask, vertices, faces, res, lox, loy, loz, cell, reach, nf)
    f = Int32(KI.get_global_id().x)
    f <= nf || return nothing
    @inbounds begin
        ia, ib, ic = faces[1, f], faces[2, f], faces[3, f]
        # Cell units, cell centres at the integers.
        gx(v) = (vertices[1, v] - lox) / cell - 0.5f0
        gy(v) = (vertices[2, v] - loy) / cell - 0.5f0
        gz(v) = (vertices[3, v] - loz) / cell - 0.5f0
        r = reach / cell
        x0 = max(Int32(0), unsafe_trunc(Int32, ceil(min(gx(ia), gx(ib), gx(ic)) - r)))
        x1 = min(Int32(res - 1), unsafe_trunc(Int32, floor(max(gx(ia), gx(ib), gx(ic)) + r)))
        y0 = max(Int32(0), unsafe_trunc(Int32, ceil(min(gy(ia), gy(ib), gy(ic)) - r)))
        y1 = min(Int32(res - 1), unsafe_trunc(Int32, floor(max(gy(ia), gy(ib), gy(ic)) + r)))
        z0 = max(Int32(0), unsafe_trunc(Int32, ceil(min(gz(ia), gz(ib), gz(ic)) - r)))
        z1 = min(Int32(res - 1), unsafe_trunc(Int32, floor(max(gz(ia), gz(ib), gz(ic)) + r)))
        for z in z0:z1, y in y0:y1, x in x0:x1
            mask[x + Int32(res) * (y + Int32(res) * z) + Int32(1)] = Int32(1)
        end
    end
    return nothing
end

"""The `ratio`^3 voxels of every marked cell, cell by cell; cell `c` covers
voxels `ratio (cx, cy, cz)` onwards."""
function expandcells!(cand, mask, pos, cres, ratio, ncell)
    c = Int32(KI.get_global_id().x)
    c <= ncell || return nothing
    @inbounds begin
        mask[c] == Int32(1) || return nothing
        c0 = c - Int32(1)
        cx = c0 % cres
        cy = (c0 ÷ cres) % cres
        cz = c0 ÷ (cres * cres)
        j = (pos[c] - Int32(1)) * ratio * ratio * ratio
        for z in Int32(0):(ratio - Int32(1)), y in Int32(0):(ratio - Int32(1)), x in Int32(0):(ratio - Int32(1))
            j += Int32(1)
            cand[1, j] = cx * ratio + x
            cand[2, j] = cy * ratio + y
            cand[3, j] = cz * ratio + z
        end
    end
    return nothing
end

"""The unsigned distance from grid point `coords[:, i] + offset` (in voxels of
`cell`, from `lo`) to the surface of the grid, minus `minus`; `Inf` beyond the
grid's reach (see [`nearest`](@ref))."""
function coorddistance!(out, coords, offset, minus, cell, lox, loy, loz, sv, sf, ss, st, sres, n)
    i = Int32(KI.get_global_id().x)
    i <= n || return nothing
    @inbounds begin
        p = Vec3f(lox + (Float32(coords[1, i]) + offset) * cell,
                  loy + (Float32(coords[2, i]) + offset) * cell,
                  loz + (Float32(coords[3, i]) + offset) * cell)
        _, d2 = nearest(p, sv, sf, ss, st, sres)
        out[i] = sqrt(d2) - minus
    end
    return nothing
end

function nearband!(flags, d, eps, tol, n)
    i = Int32(KI.get_global_id().x)
    i <= n || return nothing
    @inbounds flags[i] = abs(d[i] - eps) < tol ? Int32(1) : Int32(0)
    return nothing
end
