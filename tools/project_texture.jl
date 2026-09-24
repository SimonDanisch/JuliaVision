"""
Colour a Hunyuan3D mesh from the image it was generated from.

    include("tools/project_texture.jl")
    mesh, colours = texturize("teapot.obj", "teapot-cutout.png")
    render(mesh, colours, "teapot.png"; eye = (0.05, -1.6, 0.15))

**This is not the texture branch.** `hunyuan3d-paintpbr-v2-1` is a separate
multi-view PBR painter with its own checkpoint and its own pipeline, and it is
not ported — see `plans/models-to-port.md`. What this does is the oldest trick
there is: project the one photograph the shape came from onto the shape, and
invent the rest. That buys the side the camera saw, at full photographic detail,
and nothing else. The back is a diffusion of the front's colours over the mesh
graph, which for a white porcelain teapot reads as plain white and for anything
with a patterned back is a lie.

What it is good for: a turntable that faces the camera, a thumbnail, checking
that a reconstruction lines up with its input at all. What it cannot do: PBR
maps, view-dependent shading, or any colour on a surface the photo never saw.

## The registration is measured, not assumed

The mesh comes back in the grid's own axes and nothing in the file says which
one faces the camera. [`orientfit`](@ref) settles it by silhouette: project the
vertices along each candidate axis pair, rasterise, and score IoU against the
alpha of the image `prepareimage` actually fed the model. On the demo teapot the
winner is `x = +axis1, y = -axis2` at **IoU 0.915**, with the runner-up at 0.791
— a margin worth having, and the reason this is a function rather than a
constant.

Two details that are easy to get wrong and produce a plausible-looking
misregistration rather than an error:

  * score against the **recentred** image (`Hunyuan3DRunner.recenter`), not the
    cut-out you started from. The model never saw your framing.
  * scale the projection so the object fills `1 - BORDER_RATIO` of the square,
    which is what `recenter` does. Filling the frame instead cost 0.25 of IoU
    and left the ranking a four-way tie.
"""

using FileIO, MeshIO, GeometryBasics, ImageCore, LinearAlgebra
import Hunyuan3DRunner as H

"""
    orientfit(P, alpha; n = 256) -> (iou, ax, sx, ay, sy)

Which mesh axes the image's x and y are, by silhouette overlap. `P` is `3 x N`
vertex positions and `alpha` the `(S, S, 1)` placed alpha `recenter` returns.
"""
function orientfit(P::AbstractMatrix, alpha::AbstractArray{UInt8,3}; n::Int = 256)
    S = size(alpha, 1)
    tgt = falses(n, n)
    for j in 1:n, i in 1:n
        x = clamp(round(Int, (j - 0.5) * S / n), 1, S)
        y = clamp(round(Int, (i - 0.5) * S / n), 1, S)
        tgt[i, j] = alpha[x, y, 1] > 0x7f
    end
    best = (-1.0, 1, 1, 2, -1)
    for ax in 1:3, ay in 1:3
        ax == ay && continue
        for sx in (1, -1), sy in (1, -1)
            g = silhouette(P, ax, sx, ay, sy, n)
            s = count(g .& tgt) / count(g .| tgt)
            s > best[1] && (best = (s, ax, sx, ay, sy))
        end
    end
    best
end

"""The point cloud's silhouette on an `n x n` grid, framed as `recenter` frames
the image: aspect preserved, object filling `1 - BORDER_RATIO` of the square."""
function silhouette(P, ax, sx, ay, sy, n; border = H.BORDER_RATIO)
    x = sx .* view(P, ax, :); y = sy .* view(P, ay, :)
    xlo, xhi = extrema(x); ylo, yhi = extrema(y)
    s = max(xhi - xlo, yhi - ylo) / (1 - border)
    cx = (xhi + xlo) / 2; cy = (ylo + yhi) / 2
    g = falses(n, n)
    @inbounds for k in eachindex(x)
        px = round(Int, (x[k] - cx) / s * n + n / 2)
        py = round(Int, (y[k] - cy) / s * n + n / 2)
        (1 <= px <= n && 1 <= py <= n) && (g[py, px] = true)
    end
    g
end

"""Area-weighted vertex normals, `3 x N`."""
function vertexnormals(P, faces)
    nrm = zeros(Float32, 3, size(P, 2))
    for (a, b, c) in faces
        e1 = (P[1,b]-P[1,a], P[2,b]-P[2,a], P[3,b]-P[3,a])
        e2 = (P[1,c]-P[1,a], P[2,c]-P[2,a], P[3,c]-P[3,a])
        nf = (e1[2]*e2[3]-e1[3]*e2[2], e1[3]*e2[1]-e1[1]*e2[3], e1[1]*e2[2]-e1[2]*e2[1])
        for v in (a, b, c), i in 1:3
            nrm[i, v] += nf[i]
        end
    end
    for k in axes(nrm, 2)
        l = sqrt(nrm[1,k]^2 + nrm[2,k]^2 + nrm[3,k]^2)
        l > 0 && (nrm[:, k] ./= l)
    end
    nrm
end

"""Directed adjacency in CSR form, so the hole fill is a sweep and not a
`Vector{Vector{Int}}` per vertex."""
function adjacency(nv, faces)
    deg = zeros(Int32, nv)
    for (a, b, c) in faces; deg[a] += 2; deg[b] += 2; deg[c] += 2; end
    ptr = Vector{Int32}(undef, nv + 1); ptr[1] = 1
    for i in 1:nv; ptr[i+1] = ptr[i] + deg[i]; end
    adj = Vector{Int32}(undef, ptr[end] - 1); pos = copy(ptr)
    put!(a, b) = (adj[pos[a]] = b; pos[a] += 1)
    for (a, b, c) in faces
        put!(a,b); put!(a,c); put!(b,a); put!(b,c); put!(c,a); put!(c,b)
    end
    ptr, adj
end

"""
    fillholes!(col, seen, ptr, adj; smooth = 8) -> col

Grow the painted colours outward over the mesh graph until every vertex has one,
then smooth ONLY the invented ones. A vertex the camera never saw takes the mean
of whatever neighbours already have a colour, which is a Laplace fill in
everything but name.
"""
function fillholes!(col, seen, ptr, adj; smooth::Int = 8, maxrounds::Int = 400)
    filled = copy(seen); c2 = copy(col); front = findall(!, seen); r = 0
    while !isempty(front) && r < maxrounds
        r += 1; still = Int[]
        for k in front
            s = (0f0, 0f0, 0f0); n = 0
            for e in ptr[k]:ptr[k+1]-1
                j = adj[e]; filled[j] || continue
                s = (s[1]+col[1,j], s[2]+col[2,j], s[3]+col[3,j]); n += 1
            end
            n > 0 ? (c2[1,k] = s[1]/n; c2[2,k] = s[2]/n; c2[3,k] = s[3]/n) : push!(still, k)
        end
        col .= c2
        for k in front; k in still || (filled[k] = true); end
        front = still
    end
    for _ in 1:smooth
        for k in eachindex(seen)
            seen[k] && continue
            s = (0f0, 0f0, 0f0); n = 0
            for e in ptr[k]:ptr[k+1]-1
                j = adj[e]; s = (s[1]+col[1,j], s[2]+col[2,j], s[3]+col[3,j]); n += 1
            end
            n > 0 && (c2[1,k] = s[1]/n; c2[2,k] = s[2]/n; c2[3,k] = s[3]/n)
        end
        col .= c2
    end
    col
end

"""
    texturize(objpath, rgbapath; facing = -1, grazing = 0.02) -> (mesh, colours)

The whole thing: read the mesh and the RGBA cut-out, recentre the image the way
the model saw it, fit the orientation, project, fill.

`facing` selects which half of the surface the camera sees — the sign of the
depth axis, which a silhouette cannot determine because a silhouette is the same
from both sides. If the pattern comes out on the wrong half (the render looks
like a smudge rather than the photograph), flip it.

`grazing` is the smallest `|n . view|` that still takes colour. Lower catches
more of the spout at the cost of stretching the texture along it.
"""
function texturize(objpath::AbstractString, rgbapath::AbstractString;
                   facing::Int = -1, grazing::Float32 = 0.02f0)
    msh = load(objpath)
    V = coordinates(msh)
    P = [Float32(V[i][c]) for c in 1:3, i in eachindex(V)]
    faces = [(Int(GeometryBasics.value(f[1])), Int(GeometryBasics.value(f[2])),
              Int(GeometryBasics.value(f[3]))) for f in GeometryBasics.faces(msh)]

    cut = load(rgbapath)
    w, h = size(cut, 2), size(cut, 1)
    rgba = Array{UInt8}(undef, w, h, 4)
    for y in 1:h, x in 1:w
        c = cut[y, x]
        rgba[x,y,1] = reinterpret(UInt8, red(c));   rgba[x,y,2] = reinterpret(UInt8, green(c))
        rgba[x,y,3] = reinterpret(UInt8, blue(c));  rgba[x,y,4] = reinterpret(UInt8, alpha(c))
    end
    rgb, alpha8 = H.recenter(rgba)
    score, ax, sx, ay, sy = orientfit(P, alpha8)
    @info "silhouette fit" iou=round(score; digits=3) x="$(sx > 0 ? "+" : "-")axis$ax" y="$(sy > 0 ? "+" : "-")axis$ay"

    xs = sx .* view(P, ax, :); ys = sy .* view(P, ay, :)
    xlo, xhi = extrema(xs); ylo, yhi = extrema(ys)
    s = max(xhi - xlo, yhi - ylo) / (1 - H.BORDER_RATIO)
    cx = (xhi + xlo) / 2; cy = (ylo + yhi) / 2
    depth = 6 - ax - ay                      # the axis neither image axis uses
    nrm = vertexnormals(P, faces)
    S = size(rgb, 1)
    col = zeros(Float32, 3, size(P, 2)); seen = falses(size(P, 2))
    for k in axes(P, 2)
        facing * nrm[depth, k] > grazing || continue
        u = (xs[k] - cx) / s + 0.5f0; v = (ys[k] - cy) / s + 0.5f0
        (0 <= u <= 1 && 0 <= v <= 1) || continue
        xi = clamp(round(Int, u * S), 1, S); yi = clamp(round(Int, v * S), 1, S)
        alpha8[xi, yi, 1] > 0x7f || continue
        for c in 1:3; col[c, k] = rgb[xi, yi, c] / 255f0; end
        seen[k] = true
    end
    @info "projected" painted=count(seen) of=size(P, 2)
    ptr, adj = adjacency(size(P, 2), faces)
    fillholes!(col, seen, ptr, adj)

    # Makie is z-up and the mesh is y-up.
    pts = [Point3f(P[1,k], P[3,k], P[2,k]) for k in axes(P, 2)]
    tri = [TriangleFace{Int}(a, b, c) for (a, b, c) in faces]
    (GeometryBasics.normal_mesh(pts, tri),
     [RGBf(col[1,k], col[2,k], col[3,k]) for k in axes(col, 2)])
end

"""One render of a coloured mesh. `eye` is in Makie's frame, so y is the depth
axis and z is up."""
function render(mesh, colours, path; eye = (0.05, -1.6, 0.15), size = (900, 900))
    Makie = Base.require(Base.PkgId(Base.UUID("e9467ef8-e4e7-5192-8a1a-b1aee30e663a"), "GLMakie"))
    f = Makie.Figure(; size, backgroundcolor = :white)
    l = Makie.LScene(f[1, 1]; show_axis = false,
        scenekw = (lights = [Makie.AmbientLight(RGBf(0.55, 0.55, 0.58)),
                             Makie.DirectionalLight(RGBf(0.75, 0.75, 0.75),
                                                    Makie.Vec3f(0.4, 0.9, -0.7))],))
    Makie.mesh!(l, mesh; color = colours, shading = true)
    Makie.update_cam!(l.scene, Makie.cameracontrols(l.scene),
                      Makie.Vec3f(eye...), Makie.Vec3f(0, 0, 0))
    Makie.save(path, f)
    path
end
