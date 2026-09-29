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
vertices along each candidate view (every axis as up, turned about it),
rasterise, and score IoU against the alpha of the image `prepareimage` actually
fed the model. On the demo teapot the winner is `x = +axis1, y = -axis2` at
**IoU 0.915**, with the runner-up at 0.791 — a margin worth having, and the
reason this is a function rather than a constant.

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
    orientfit(P, alpha; n = 256, step = 2) -> (iou, R)

The direction the image looks at the mesh from, by silhouette overlap. `R` is a
3x3 rotation whose rows are the image's x, its y (pointing down) and the depth,
in mesh coordinates. `P` is `3 x N` vertex positions and `alpha` the `(S, S, 1)`
placed alpha `recenter` returns.

Every mesh axis is tried as up, and the view is turned about it in `step`
degrees. Trying only the axis-aligned views is enough for a teapot, but the
image does not always look down a mesh axis: for a character in a
three-quarter pose, the best axis-aligned view reached IoU 0.40, and it painted
the texture on from the side.
"""
function orientfit(P::AbstractMatrix, alpha::AbstractArray{UInt8,3}; n::Int = 256, step::Int = 2)
    S = size(alpha, 1)
    tgt = falses(n, n)
    for j in 1:n, i in 1:n
        x = clamp(round(Int, (j - 0.5) * S / n), 1, S)
        y = clamp(round(Int, (i - 0.5) * S / n), 1, S)
        tgt[i, j] = alpha[x, y, 1] > 0x7f
    end
    # A silhouette needs far fewer points than a full-resolution mesh has.
    Q = P[:, 1:max(1, size(P, 2) ÷ 50_000):end]
    best = (-1.0, Matrix{Float32}(I, 3, 3))
    for k in 1:3, su in (1f0, -1f0)
        u = zeros(Float32, 3); u[k] = su
        a = zeros(Float32, 3); a[mod1(k + 1, 3)] = 1f0
        b = cross(u, a)
        for θ in 0:step:360-step
            h = cosd(θ) .* a .+ sind(θ) .* b
            # Depth is `h x u`: with up = +axis2 and x = +axis1 that is +axis3,
            # the depth the axis-aligned search used, so `facing` means the same.
            R = Matrix{Float32}(vcat(h', -u', cross(h, u)'))
            g = silhouette(Q, R, n)
            s = count(g .& tgt) / count(g .| tgt)
            s > best[1] && (best = (s, R))
        end
    end
    best
end

"""The point cloud's silhouette on an `n x n` grid, seen through the view `R`
and framed as `recenter` frames the image: aspect preserved, object filling
`1 - BORDER_RATIO` of the square."""
function silhouette(P, R::AbstractMatrix, n; border = H.BORDER_RATIO)
    x = vec(R[1:1, :] * P); y = vec(R[2:2, :] * P)
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

"""
    frontdepth(u, v, z, faces, S) -> S x S Matrix{Float32}

The nearest surface per image pixel: every triangle rasterised at its projected
`(u, v)` in `[0, 1]`, keeping the largest `z` (the camera looks down `-z`).
Without it a normal test alone paints every surface that faces the camera,
including the ones behind the one the photo shows: hair behind a face took the
face a second time.
"""
function frontdepth(u, v, z, faces, S::Int)
    zb = fill(-Inf32, S, S)
    @inbounds for (a, b, c) in faces
        xa, ya = u[a] * S, v[a] * S; xb, yb = u[b] * S, v[b] * S; xc, yc = u[c] * S, v[c] * S
        det = (xb - xa) * (yc - ya) - (xc - xa) * (yb - ya)
        abs(det) < 1f-12 && continue
        x0 = max(1, floor(Int, min(xa, xb, xc))); x1 = min(S, ceil(Int, max(xa, xb, xc)))
        y0 = max(1, floor(Int, min(ya, yb, yc))); y1 = min(S, ceil(Int, max(ya, yb, yc)))
        for yi in y0:y1, xi in x0:x1
            px = Float32(xi); py = Float32(yi)    # pixel `xi` is where `round(u * S)` lands
            wb = ((px - xa) * (yc - ya) - (xc - xa) * (py - ya)) / det
            wc = ((xb - xa) * (py - ya) - (px - xa) * (yb - ya)) / det
            wa = 1 - wb - wc
            (wa >= 0 && wb >= 0 && wc >= 0) || continue
            d = wa * z[a] + wb * z[b] + wc * z[c]
            d > zb[xi, yi] && (zb[xi, yi] = d)
        end
    end
    zb
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
    texturize(objpath, rgbapath; facing = -1, grazing = 0.02, occlusion = 3) -> (mesh, colours)

The whole thing: read the mesh and the RGBA cut-out, recentre the image the way
the model saw it, fit the orientation, project, fill.

`facing` selects which half of the surface the camera sees — the sign of the
depth axis, which a silhouette cannot determine because a silhouette is the same
from both sides. If the pattern comes out on the wrong half (the render looks
like a smudge rather than the photograph), flip it.

`grazing` is the smallest `|n . view|` that still takes colour. Lower catches
more of the spout at the cost of stretching the texture along it.

`occlusion` is how far behind the nearest surface, in pixels of depth, a vertex
may lie and still take colour. Surfaces closer together than that behind one
another both get the front one's colour.
"""
function texturize(objpath::AbstractString, rgbapath::AbstractString;
                   facing::Int = -1, grazing::Float32 = 0.02f0, occlusion::Float32 = 3f0)
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
    score, R = orientfit(P, alpha8)
    @info "silhouette fit" iou=round(score; digits=3) x=round.(R[1, :]; digits=3) up=-R[2, :]

    xs = vec(R[1:1, :] * P); ys = vec(R[2:2, :] * P)
    xlo, xhi = extrema(xs); ylo, yhi = extrema(ys)
    s = max(xhi - xlo, yhi - ylo) / (1 - H.BORDER_RATIO)
    cx = (xhi + xlo) / 2; cy = (ylo + yhi) / 2
    nrm = vec(R[3:3, :] * vertexnormals(P, faces))    # each normal along the depth
    S = size(rgb, 1)
    us = (xs .- cx) ./ s .+ 0.5f0; vs = (ys .- cy) ./ s .+ 0.5f0
    zs = facing .* vec(R[3:3, :] * P)
    zb = frontdepth(us, vs, zs, faces, S)
    # A vertex sits up to half a pixel off the centre its depth is compared at,
    # so the front surface is anything within `occlusion` pixels of depth of it.
    tol = occlusion * s / S
    col = zeros(Float32, 3, size(P, 2)); seen = falses(size(P, 2)); hidden = 0
    for k in axes(P, 2)
        facing * nrm[k] > grazing || continue
        u = us[k]; v = vs[k]
        (0 <= u <= 1 && 0 <= v <= 1) || continue
        xi = clamp(round(Int, u * S), 1, S); yi = clamp(round(Int, v * S), 1, S)
        alpha8[xi, yi, 1] > 0x7f || continue
        zs[k] >= zb[xi, yi] - tol || (hidden += 1; continue)
        for c in 1:3; col[c, k] = rgb[xi, yi, c] / 255f0; end
        seen[k] = true
    end
    @info "projected" painted=count(seen) hidden of=size(P, 2)
    ptr, adj = adjacency(size(P, 2), faces)
    fillholes!(col, seen, ptr, adj)

    # Makie is z-up and the mesh is y-up.
    pts = [Point3f(P[1,k], P[3,k], P[2,k]) for k in axes(P, 2)]
    tri = [TriangleFace{Int}(a, b, c) for (a, b, c) in faces]
    (GeometryBasics.normal_mesh(pts, tri),
     [RGB{Float32}(col[1,k], col[2,k], col[3,k]) for k in axes(col, 2)])
end

"""One render of a coloured mesh. `eye` is in Makie's frame, so y is the depth
axis and z is up."""
function render(mesh, colours, path; eye = (0.05, -1.6, 0.15), size = (900, 900))
    # GLMakie is loaded here rather than at the top so `texturize` alone does not
    # pay for it. Its methods are then newer than this call, so the drawing has to
    # run in the latest world; calling it directly works only when GLMakie was
    # already loaded before this function ran.
    Makie = Base.require(Base.PkgId(Base.UUID("e9467ef8-e4e7-5192-8a1a-b1aee30e663a"), "GLMakie"))
    return Base.invokelatest(drawmesh, Makie, mesh, colours, path, eye, size)
end

function drawmesh(Makie, mesh, colours, path, eye, size)
    f = Makie.Figure(; size, backgroundcolor = :white)
    l = Makie.LScene(f[1, 1]; show_axis = false,
        scenekw = (lights = [Makie.AmbientLight(Makie.RGBf(0.55, 0.55, 0.58)),
                             Makie.DirectionalLight(Makie.RGBf(0.75, 0.75, 0.75),
                                                    Makie.Vec3f(0.4, 0.9, -0.7))],))
    Makie.mesh!(l, mesh; color = colours, shading = true)
    Makie.update_cam!(l.scene, Makie.cameracontrols(l.scene),
                      Makie.Vec3f(eye...), Makie.Vec3f(0, 0, 0))
    Makie.save(path, f)
    path
end
