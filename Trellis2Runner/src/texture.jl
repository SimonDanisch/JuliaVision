"""
    Charts

A UV atlas's charts over a triangle mesh: `chart[f]` for every face, and the
number of charts. See [`segmentcharts`](@ref).
"""
struct Charts
    chart::Vector{Int32}
    count::Int
end

"""
    facegeometry(vertices, faces) -> (normals, areas)

Unit normal and area of every face (`(3, V)` Float32 vertices, `(3, F)` faces).
"""
function facegeometry(vertices::AbstractMatrix{Float32}, faces::AbstractMatrix{<:Integer})
    nf = size(faces, 2)
    normals = Vector{Vec3f}(undef, nf)
    areas = Vector{Float32}(undef, nf)
    for f in 1:nf
        a, b, c = (Vec3f(view(vertices, :, faces[k, f])) for k in 1:3)
        n = cross(b - a, c - a)
        l = norm(n)
        areas[f] = l / 2
        normals[f] = l > 0 ? n / l : Vec3f(0, 0, 1)
    end
    return normals, areas
end

"""
    faceneighbours(faces, normals) -> Vector{NTuple{3,Int32}}

For every face, the face across each of its three edges, zero on a boundary.
An edge of more than two faces (where sheets of the dual-grid surface meet)
links each of them to the one whose normal is closest to its own: for a chart,
crossing such an edge is no different from crossing any other, and treating it
as a border cut the surface into hundreds of thousands of pieces.
"""
function faceneighbours(faces::AbstractMatrix{<:Integer}, normals::Vector{Vec3f})
    nf = size(faces, 2)
    owners = Dict{Tuple{Int32,Int32},Vector{Int32}}()
    sizehint!(owners, 2nf)
    edge(a, b) = a < b ? (Int32(a), Int32(b)) : (Int32(b), Int32(a))
    for f in 1:nf, k in 1:3
        push!(get!(Vector{Int32}, owners, edge(faces[k, f], faces[k % 3 + 1, f])), f)
    end
    nbr = fill((Int32(0), Int32(0), Int32(0)), nf)
    for f in 1:nf, k in 1:3
        fs = owners[edge(faces[k, f], faces[k % 3 + 1, f])]
        length(fs) >= 2 || continue
        best, g = -Inf32, Int32(0)
        for h in fs
            h == f && continue
            d = dot(normals[f], normals[h])
            d > best && ((best, g) = (d, h))
        end
        nbr[f] = Base.setindex(nbr[f], g, k)
    end
    return nbr
end

"""
    segmentcharts(normals, areas, nbr; cone = 60) -> Charts

Region growing inside a normal cone, cumesh's `compute_charts` idea: a chart
starts at the first face nobody owns and takes, most aligned first, every
neighbouring face within `cone` degrees of the chart's area-weighted normal.
Signed: taking faces either way round lets a chart run from the front of a thin
part to its back through the edges where the sheets meet, which no map
flattens. The faces have to be wound consistently first ([`orientfaces`](@ref)). With `subset`, only those faces are charted
(numbered from one), and `chart` is `-1` elsewhere.
"""
function segmentcharts(normals::Vector{Vec3f}, areas::Vector{Float32},
                       nbr::Vector{NTuple{3,Int32}}; cone::Real = 60,
                       subset::AbstractVector{<:Integer} = 1:length(normals))
    nf = length(normals)
    cosmax = cosd(Float32(cone))
    # Faces outside `subset` count as owned already, so nothing grows into them.
    chart = fill(Int32(-1), nf)
    chart[subset] .= 0
    count = 0
    heap = DataStructures.BinaryMinHeap{Tuple{Float32,Int32}}()
    for seed in subset
        chart[seed] == 0 || continue
        count += 1
        chart[seed] = count
        axis = normals[seed] * areas[seed]
        push!(heap, (0f0, Int32(seed)))
        # Faces are claimed when they enter the heap (so they enter it once);
        # a claim is dropped again if, by the time it is popped, the chart's
        # normal has turned away from it.
        while !isempty(heap)
            _, f = pop!(heap)
            n = normalize(axis)
            if f != seed && dot(normals[f], n) < cosmax
                chart[f] = 0
                continue
            end
            f == seed || (axis += normals[f] * areas[f])
            for g in nbr[f]
                (g == 0 || chart[g] != 0) && continue
                d = dot(normals[g], n)
                d >= cosmax || continue
                chart[g] = count
                push!(heap, (1f0 - d, g))
            end
        end
    end
    return Charts(chart, count)
end

"""
    ChartLayout

Where every chart went: the atlas's vertices (a mesh vertex once per chart it
is in, `vmap` back to the mesh), their texel positions `(x, y)` with `y` from
the top row, and the faces over them.
"""
struct ChartLayout
    vmap::Vector{Int32}
    texel::Matrix{Float32}
    faces::Matrix{Int32}
    size::Int
end

"""
The area-weighted normal of a chart whose faces may be wound either way: each
face counted on the side of the largest one.
"""
function chartnormal(normals::Vector{Vec3f}, areas::Vector{Float32}, members)
    ref = normals[members[argmax(i -> areas[members[i]], eachindex(members))]]
    return normalize(sum(sign(dot(normals[f], ref)) * normals[f] * areas[f] for f in members))
end

"""
    flatten!(pieces, pp, members, normals, areas) -> pieces

Map one chart into the plane: by [`lscm`](@ref), unless that folds it or lays
it over itself ([`selfoverlaps`](@ref)); then, as xatlas does, the chart is
grown into pieces face by face ([`PiecewiseParam`](@ref)).
Pushes `(faces, vids, uv)` per piece.
"""
function flatten!(pieces, pp::PiecewiseParam, members::Vector{Int32}, normals::Vector{Vec3f},
                  areas::Vector{Float32})
    if length(members) > 1
        r = lscm(pp.vertices, pp.faces, members, normals, areas)
        if r !== nothing && !selfoverlaps(r..., pp.faces, members)
            push!(pieces, (members, r...))
            return pieces
        end
    end
    return piecewise!(pieces, pp, members)
end

"""
    layoutcharts(vertices, faces, charts, normals, areas, nbr; size) -> ChartLayout

Flatten every chart ([`flatten!`](@ref): LSCM, or pieces grown face by face
where that folds or overlaps), turn every piece to its principal axes and pack
them ([`packcharts`](@ref)) into a `size`^2 atlas at one texel density for all:
xatlas's estimate for three quarters of the atlas used first, then scaled by
how far the packed atlas over- or undershoots `size` until it fits.
"""
function layoutcharts(vertices::AbstractMatrix{Float32}, faces::AbstractMatrix{<:Integer},
                      charts::Charts, normals::Vector{Vec3f}, areas::Vector{Float32},
                      nbr::Vector{NTuple{3,Int32}}; size::Int = 2048)
    members = [Int32[] for _ in 1:charts.count]
    for f in eachindex(charts.chart)
        push!(members[charts.chart[f]], f)
    end
    pieces = Tuple{Vector{Int32},Vector{Int32},Matrix{Float32}}[]   # (faces, vids, uv)
    pp = PiecewiseParam(vertices, faces, nbr)
    for m in members
        flatten!(pieces, pp, m, normals, areas)
    end
    # Principal axes of each piece, and its triangles over its own vertices.
    tris = Vector{Matrix{Int32}}(undef, length(pieces))
    uvarea = 0.0
    for (c, (fs, vids, uv)) in enumerate(pieces)
        m = sum(eachcol(uv)) / Base.size(uv, 2)
        d = uv .- m
        cxx, cyy, cxy = sum(abs2, view(d, 1, :)), sum(abs2, view(d, 2, :)), dot(view(d, 1, :), view(d, 2, :))
        θ = 0.5f0 * atan(2cxy, cxx - cyy)
        cs, sn = cos(θ), sin(θ)
        for i in axes(uv, 2)
            x, y = d[1, i], d[2, i]
            uv[1, i], uv[2, i] = cs * x + sn * y, -sn * x + cs * y
        end
        slot = Dict(v => Int32(i) for (i, v) in enumerate(vids))
        tris[c] = [slot[Int32(faces[k, f])] for k in 1:3, f in fs]
        for t in axes(tris[c], 2)
            a, b, e = (Vec2f(view(uv, :, tris[c][k, t])) for k in 1:3)
            uvarea += abs((b - a)[1] * (e - a)[2] - (b - a)[2] * (e - a)[1]) / 2
        end
    end
    uvs = [p[3] for p in pieces]
    density = Float32(sqrt(0.75 * size^2 / uvarea))
    packed = packcharts(uvs, tris, density; capacity = 2size)
    while packed === nothing || max(packed[2], packed[3]) > size
        density *= packed === nothing ? 0.7f0 : 0.995f0 * size / max(packed[2], packed[3])
        packed = packcharts(uvs, tris, density; capacity = 2size)
    end
    vmap = Int32[]
    texel = Float32[]
    newfaces = Matrix{Int32}(undef, 3, Base.size(faces, 2))
    for (c, (fs, vids, _)) in enumerate(pieces)
        base = Int32(length(vmap))
        append!(vmap, vids)
        append!(texel, packed[1][c])
        for (t, f) in enumerate(fs), k in 1:3
            newfaces[k, f] = base + tris[c][k, t]
        end
    end
    return ChartLayout(vmap, reshape(texel, 2, :), newfaces, size)
end

"""
    bake(dev, layout, vertices, surface, level, attrs) -> Array{Float32,3}

`to_glb`'s attribute sampling: rasterise the layout's triangles into the atlas
(every texel whose centre lies in one gets that point of the mesh), move each
point to the closest point of `surface` (the mesh before simplification, a
[`SurfaceGrid`](@ref)), and sample the attribute volume there trilinearly as
`flex_gemm`'s `grid_sample_3d` does (neighbours at the voxel centres, the
missing ones left out and the weights renormalised).

The projection is what keeps a thin part's colour its own: a simplified
triangle spanning it runs up to a voxel inside, where the trilinear footprint
takes in the voxels of the other side.

`(C + 1, size, size)`: the `C` attributes and a coverage flag, texel `(x, y)`
with `y` from the top.
"""
function bake(dev::Mantle.Device, layout::ChartLayout, vertices::AbstractMatrix{Float32},
              surface::SurfaceGrid, level::Level, attrs::AbstractMatrix{Float32})
    T = layout.size
    C = size(attrs, 1)
    nf = size(layout.faces, 2)
    g = Mantle.Graph(dev)
    grid = VoxelGrid(g, dev, level, Mantle.Buffer(dev, level.coords))
    points = Mantle.Transient.Buffer(g, Float32, 4, T * T)
    Mantle.dispatch!(g, zerofill!, (points, 4 * T * T), 4 * T * T; name = "bake/clear")
    texel = Mantle.Buffer(dev, layout.texel)
    pos = Mantle.Buffer(dev, vertices[:, layout.vmap])
    tris = Mantle.Buffer(dev, layout.faces)
    Mantle.dispatch!(g, rasterise!, (points, texel, pos, tris, T, nf, 1f0), nf; name = "bake/footprint")
    Mantle.dispatch!(g, rasterise!, (points, texel, pos, tris, T, nf, 0f0), nf; name = "bake/rasterise")
    out = fill!(Mantle.Buffer(dev, Float32, (C + 1, T * T)), 0)
    Mantle.dispatch!(g, sampletexels!, (out, points, Mantle.Buffer(dev, surface.vertices),
                                        Mantle.Buffer(dev, surface.faces), Mantle.Buffer(dev, surface.start),
                                        Mantle.Buffer(dev, surface.tris), surface.res, grid.bricks, grid.cells,
                                        Mantle.Buffer(dev, attrs), level.res, Val(C), T * T),
                     T * T; name = "bake/sample")
    runonce!(g)
    return reshape(Array(Mantle.storage(out)), C + 1, T, T)
end

"""
One triangle of the atlas: the mesh point at each texel it reaches, and a
coverage flag, into `points` `(4, T * T)`.

`reach` is half the side of the square around a texel centre that has to touch
the triangle; a texel whose centre lies outside gets the nearest point of the
triangle (the barycentrics clamped). With `reach = 1` that is the chart's
bilinear footprint ([`footprint`](@ref)), every texel filtering reads for a
point of it: left to the gap filling, such a texel at a chart's border took its
colour from whatever lay next to the chart in the atlas, a dark seam. `reach =
0` is the exact raster, run after it, so a texel inside a triangle keeps its
own point whatever triangle's footprint also reaches it.
"""
function rasterise!(points, texel, pos, faces, T, nf, reach)
    f = KI.get_global_id().x
    f <= nf || return nothing
    @inbounds begin
        ia, ib, ic = faces[1, f], faces[2, f], faces[3, f]
        ax, ay = texel[1, ia], texel[2, ia]
        bx, by = texel[1, ib], texel[2, ib]
        cx, cy = texel[1, ic], texel[2, ic]
        area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        abs(area) > 1f-12 || return nothing
        # How far each barycentric can grow inside the square from its centre.
        g0x, g0y = (by - cy) / area, (cx - bx) / area
        g1x, g1y = (cy - ay) / area, (ax - cx) / area
        r0 = reach * (abs(g0x) + abs(g0y))
        r1 = reach * (abs(g1x) + abs(g1y))
        r2 = reach * (abs(g0x + g1x) + abs(g0y + g1y))
        # Texel coordinates lie in the atlas by construction; `unsafe_trunc` keeps
        # the conversion free of `floor(Int, x)`'s throw path, which a kernel
        # cannot take.
        x0 = max(Int32(0), unsafe_trunc(Int32, floor(min(ax, bx, cx) - reach)))
        x1 = min(Int32(T - 1), unsafe_trunc(Int32, floor(max(ax, bx, cx) + reach)))
        y0 = max(Int32(0), unsafe_trunc(Int32, floor(min(ay, by, cy) - reach)))
        y1 = min(Int32(T - 1), unsafe_trunc(Int32, floor(max(ay, by, cy) + reach)))
        for y in y0:y1, x in x0:x1
            px, py = x + 0.5f0, y + 0.5f0
            (min(ax, bx, cx) - reach <= px <= max(ax, bx, cx) + reach &&
             min(ay, by, cy) - reach <= py <= max(ay, by, cy) + reach) || continue
            w0 = ((bx - px) * (cy - py) - (by - py) * (cx - px)) / area
            w1 = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) / area
            w2 = 1f0 - w0 - w1
            (w0 >= -r0 && w1 >= -r1 && w2 >= -r2) || continue
            w0, w1, w2 = max(w0, 0f0), max(w1, 0f0), max(w2, 0f0)
            s = w0 + w1 + w2
            t = x + T * y + 1
            for k in 1:3
                points[k, t] = (w0 * pos[k, ia] + w1 * pos[k, ib] + w2 * pos[k, ic]) / s
            end
            points[4, t] = 1f0
        end
    end
    return nothing
end

"""One texel: its point moved onto the surface ([`project`](@ref)) and the
attributes sampled there."""
function sampletexels!(out, points, sv, sf, start, tris, sres, bricks, cells, attrs, res, ::Val{C}, n) where {C}
    t = KI.get_global_id().x
    t <= n || return nothing
    @inbounds begin
        points[4, t] > 0f0 || return nothing
        p = project(Vec3f(points[1, t], points[2, t], points[3, t]), sv, sf, start, tris, sres)
        # In voxel units of the `res`^3 grid over [-0.5, 0.5]^3.
        qx, qy, qz = (p[1] + 0.5f0) * res, (p[2] + 0.5f0) * res, (p[3] + 0.5f0) * res
        acc = ntuple(_ -> 0f0, Val(C))
        wsum = 0f0
        for corner in 0:7
            nx = unsafe_trunc(Int32, qx + ((corner & 1) - 0.5f0))
            ny = unsafe_trunc(Int32, qy + ((corner >> 1 & 1) - 0.5f0))
            nz = unsafe_trunc(Int32, qz + ((corner >> 2 & 1) - 0.5f0))
            (0 <= nx < res && 0 <= ny < res && 0 <= nz < res) || continue
            v = voxelat(bricks, cells, res, nx, ny, nz)
            v > 0 || continue
            w = (1f0 - abs(nx + 0.5f0 - qx)) * (1f0 - abs(ny + 0.5f0 - qy)) *
                (1f0 - abs(nz + 0.5f0 - qz))
            wsum += w
            acc = addweighted(acc, attrs, w, v)
        end
        wsum > 0f0 || return nothing
        for k in 1:C
            out[k, t] = acc[k] / wsum
        end
        out[C + 1, t] = 1f0
    end
    return nothing
end

"""`acc + w attrs[:, v]` as a tuple. A function of its own: written inline, the
closure captures `acc` while the loop reassigns it, which boxes it."""
@inline addweighted(acc::NTuple{C,Float32}, attrs, w::Float32, v) where {C} =
    ntuple(k -> acc[k] + w * @inbounds(attrs[k, v]), Val(C))

"""
    pushpull!(tex)

Fill every uncovered texel (`tex[end, x, y] == 0`) from the covered ones around
it: average down a pyramid of coverage-weighted halvings, then fill each level's
holes from the one below. Stands in for upstream's `cv2.inpaint`; what it has to
do is the same — no black seams where bilinear filtering reaches past a chart.
"""
function pushpull!(tex::Array{Float32,3})
    C = size(tex, 1) - 1
    size(tex, 2) <= 1 && return tex
    w, h = size(tex, 2), size(tex, 3)
    half = zeros(Float32, C + 1, cld(w, 2), cld(h, 2))
    for y in 1:h, x in 1:w
        c = tex[C + 1, x, y]
        c > 0 || continue
        X, Y = (x + 1) ÷ 2, (y + 1) ÷ 2
        for k in 1:C
            half[k, X, Y] += c * tex[k, x, y]
        end
        half[C + 1, X, Y] += c
    end
    for Y in axes(half, 3), X in axes(half, 2)
        c = half[C + 1, X, Y]
        c > 0 || continue
        for k in 1:C
            half[k, X, Y] /= c
        end
        half[C + 1, X, Y] = 1
    end
    pushpull!(half)
    for y in 1:h, x in 1:w
        tex[C + 1, x, y] > 0 && continue
        X, Y = (x + 1) ÷ 2, (y + 1) ÷ 2
        for k in 1:C
            tex[k, x, y] = half[k, X, Y]
        end
    end
    return tex
end

"""
    texturedmesh(dev, vertices, faces, level, attrs; surface = (vertices, faces), size = 2048, cone = 85) -> MetaMesh

`o_voxel.postprocess.to_glb`'s texturing: charts, the atlas, the attribute
volume baked into it at the closest points of `surface` (the mesh before
simplification, see [`bake`](@ref)) and gaps filled, as one PBR material in
MeshIO's form; `save("model.glb", mesh)` writes it.

`attrs` is the texture decoder's output mapped to [0, 1] (`out * 0.5 + 0.5`),
`(6, N)` over `level`'s voxels, laid out `base_color` (1:3), `metallic`,
`roughness`, `alpha`. The base colour texture carries alpha; the
metallic-roughness texture is glTF's (roughness in green, metallic in blue).
"""
function texturedmesh(dev::Mantle.Device, vertices::AbstractMatrix{Float32},
                      faces::AbstractMatrix{<:Integer}, level::Level, attrs::AbstractMatrix{Float32};
                      surface::Tuple{AbstractMatrix{Float32},AbstractMatrix{<:Integer}} = (vertices, faces),
                      size::Int = 2048, cone::Real = 85)
    normals, areas = facegeometry(vertices, faces)
    nbr = faceneighbours(faces, normals)
    charts = segmentcharts(normals, areas, nbr; cone)
    layout = layoutcharts(vertices, faces, charts, normals, areas, nbr; size)
    # Cells of two voxels: a query reaches two voxels at least.
    grid = SurfaceGrid(surface...; res = level.res ÷ 2)
    tex = pushpull!(bake(dev, layout, vertices, grid, level, attrs))
    # Smooth normals of the mesh, shared by a vertex's copies in every chart.
    vn = zeros(Vec3f, Base.size(vertices, 2))
    for f in eachindex(normals), k in 1:3
        vn[faces[k, f]] += normals[f] * areas[f]
    end
    q(x) = N0f8(clamp(x, 0f0, 1f0))
    basecolor = [RGBA{N0f8}(q(tex[1, x, y]), q(tex[2, x, y]), q(tex[3, x, y]), q(tex[6, x, y]))
                 for y in 1:size, x in 1:size]
    metalrough = [RGB{N0f8}(0, q(tex[5, x, y]), q(tex[4, x, y])) for y in 1:size, x in 1:size]
    points = [Point3f(view(vertices, :, v)) for v in layout.vmap]
    uv = [Vec2f(layout.texel[1, i] / size, 1f0 - layout.texel[2, i] / size) for i in eachindex(layout.vmap)]
    nrm = [normalize(vn[v]) for v in layout.vmap]
    fcs = [GLTriangleFace(layout.faces[1, f], layout.faces[2, f], layout.faces[3, f])
           for f in 1:Base.size(layout.faces, 2)]
    mesh = GeometryBasics.Mesh(points, fcs; normal = nrm, uv)
    material = Dict{String,Any}("diffuse map" => Dict{String,Any}("image" => basecolor),
                                "metallic roughness map" => Dict{String,Any}("image" => metalrough),
                                "metallic" => 1f0, "roughness" => 1f0,
                                "alpha mode" => "OPAQUE", "double sided" => true)
    return GeometryBasics.MetaMesh(mesh, Dict{Symbol,Any}(
        :materials => Dict{String,Any}("trellis2" => material), :material_names => ["trellis2"]))
end
