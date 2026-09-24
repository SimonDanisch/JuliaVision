"""
What has to stay true without an artifact installed.

Two kinds of thing are pinned here, and neither needs the 7.37 GB checkpoint.

**The arithmetic contract with PyTorch.** Four separate places in the sampler
round differently from the obvious reading of upstream's source, and every one of
them was found by comparing against a dumped reference rather than by reading —
they are invisible in a single step and change the trajectory over fifty. The
cases below are the specific values that discriminate: each asserts the reference
semantics *and* names what the wrong one would give, so a regression reports the
answer it produced rather than "not equal".

**The surface extractor.** `marchingcubes` is built from face contours instead of
a case table, so the property to check is that it closes — every one of the 256
sign patterns, then a sphere against its analytic area and volume.

There is deliberately no parity check against the local `gen/` tree.
`DNNKernels/src/assets.jl` removed that fallback on the grounds that a test whose
assets come from a working copy passes for whoever ran the exporter and is
unreachable for everyone else — which is how the one gate that catches a
fast-but-wrong kernel came to run on a single machine. The measured parity is
recorded in the module docstring; it belongs in this file the moment the export
travels with the repository.

The latency gate — `Mantle.no_pipeline_compilation` reporting 0 refusals in a fresh
subprocess — belongs here once the workload drives a real call. See SAM2Runner's
suite for the shape.
"""

using Test, Hunyuan3DRunner, DNNKernels
import Mantle
import Artifacts
const H = Hunyuan3DRunner

@testset "Hunyuan3DRunner" begin
    @testset "loads, and says whether the four parts are here" begin
        # `ready` must answer WITHOUT resolving an artifact: it is what
        # `@setup_workload` branches on, and `@artifact_str` downloads. A guard
        # that fetched 6.8 GB to answer "do we have it" would be worse than none.
        @test Hunyuan3DRunner.ready() isa Bool
        @test Hunyuan3DRunner.KERNELS_VERSION == DNNKernels.KERNELS_VERSION
    end

    @testset "each part resolves to its own artifact" begin
        # Four artifacts, not one tree: the parts run at wildly different rates,
        # so asking for the geometry decoder must not fetch the 5.7 GiB
        # denoiser. Skipped rather than failed where they are not bound — the
        # bindings live in this package's `Artifacts.toml` and a fresh clone
        # downloads them on first use.
        if Hunyuan3DRunner.ready()
            for (d, f) in ((H.conddir(), "hunyuan3d_cond.json"),
                           (H.vaedir(),  "hunyuan3d_vae.json"),
                           (H.geodir(),  "hunyuan3d_geo.json"))
                @test isdir(d)
                @test isfile(joinpath(d, f))
                @test isfile(joinpath(d, "weights.safetensors"))
            end
            # The 5.7-GiB denoiser is intentionally different: its graph has a
            # small artifact of its own and its weights are four release-sized
            # shards, merged by `hunyuan3dweights`. Requiring a monolithic file
            # beside the graph contradicts the loader and the artifact layout.
            @test isfile(joinpath(H.ditdir(), "hunyuan3d_dit.json"))
            for (artifact, file) in H.DIT_SHARDS
                hash = Artifacts.artifact_hash(artifact, H.ARTIFACTS_TOML)
                @test hash !== nothing
                @test isfile(joinpath(Artifacts.artifact_path(hash), file))
            end
            # The four are distinct trees, which is the whole claim.
            @test length(unique([H.conddir(), H.ditdir(), H.vaedir(), H.geodir()])) == 4
        else
            @info "Hunyuan3D parts not in the artifact store; skipping"
        end
    end

    @testset "an explicit dir that has nothing names the path" begin
        d = mktempdir()
        e = try
            Hunyuan3DRunner.hunyuan3dgraph(; dir = d)
            nothing
        catch err
            err
        end
        @test e isa ArgumentError
        @test occursin(d, e.msg)
    end

    @testset "the exported shapes match the checkpoint config" begin
        # From `hunyuan3d-dit-v2-1/config.yaml`, not from `HunYuanDiTPlain`'s
        # signature — the defaults there are a different model (1024 latents, 4
        # channels, depth 24) and reading them instead is how the condition
        # length silently becomes 257.
        @test Hunyuan3DRunner.LATENTS == 4096
        @test Hunyuan3DRunner.LATENT_CHANNELS == 64
        @test Hunyuan3DRunner.COND_DIM == 1024
        # 518/14 = 37 patches a side, plus the class token.
        @test Hunyuan3DRunner.COND_TOKENS == 37 * 37 + 1
    end

    # ------------------------------------------------------------ the schedule

    @testset "the sigma schedule is upstream's, in upstream's precision" begin
        s = H.flowsigmas(50)
        @test length(s) == 51                     # `steps` plus the appended 1.0
        @test eltype(s) === Float32               # `.to(dtype=torch.float32)`
        @test s[1] === 0.0f0                      # "we start from 0"
        @test s[50] === 1.0f0
        @test s[51] === 1.0f0                     # so the last step moves nothing
        @test s[2] === 0.020408163f0
        # fp32 before the differences, not fp64 narrowed after: the two disagree
        # in the last bit and that bit is taken fifty times.
        @test s[2] - s[1] === Float32(1 / 49)

        t = H.flowtimesteps(s)
        @test length(t) == 50
        @test eltype(t) === Float32
        @test t[1] === 0.0f0
        @test t[50] === 1000.0f0
    end

    @testset "the timestep is narrowed BEFORE it is divided" begin
        t = H.flowtimesteps(H.flowsigmas(50))
        # `timestep.to(latents.dtype)` then `/ num_train_timesteps`, not the
        # other way round. Nine of the fifty steps land on a different fp16
        # value, and the result is an input to a 754-op forward.
        @test H.conditioningtime(Float16, t[10]) === Float16(0.1836)
        @test Float16(Float64(t[10]) / 1000) === Float16(0.1837)   # the wrong order
        differ = count(k -> H.conditioningtime(Float16, t[k]) != Float16(Float64(t[k]) / 1000), 1:50)
        @test differ == 9
    end

    # ------------------------------------------------ the fp16 rounding contract

    @testset "guidance rounds after every operation" begin
        # PyTorch evaluates `u + g * (c - u)` as three fp16 ops. Fused into one
        # expression Lava keeps the product in fp32 and rounds once at the store,
        # which is the more accurate of the two and not the one being reproduced.
        # These two values differ between the readings.
        pred = reshape(Float16[-0.0603, -0.08575], 1, 1, 2)
        @test H.cfg(pred, 5.0)[1] === Float16(0.04144)
        @test Float16(Float32(-0.08575f0) +
                      5.0f0 * Float32(Float16(-0.0603) - Float16(-0.08575))) === Float16(0.0415)
    end

    @testset "the Euler step size is fp16, not fp32" begin
        s = H.flowsigmas(50)
        # `(sigma_next - sigma)` is a 0-dim fp32 tensor and `model_output` is a
        # dimensioned fp16 one; PyTorch gives the dimensioned operand priority,
        # so the SCALAR narrows and the product is fp16. The effective step is
        # Float16(1/49) = 0.0204, not 0.020408163.
        @test Float16(s[2] - s[1]) === Float16(0.0204)
        @test H.eulerstep(Float16[-0.4294], Float16[-3.25], s[1], s[2])[1] === Float16(-0.4956)
        # what an fp32 step size would have given, for the failure message
        @test Float16(Float32(Float16(-0.4294)) +
                      (s[2] - s[1]) * Float32(Float16(-3.25))) === Float16(-0.4958)
        # the final step is a no-op by construction
        @test H.eulerstep(Float16[1.5], Float16[9.0], s[50], s[51])[1] === Float16(1.5)
    end

    @testset "the VAE scale factor is the checkpoint's" begin
        # Small enough that dropping it produces a slightly wrong mesh rather
        # than a broken one, which is exactly why it is pinned.
        @test H.SCALE_FACTOR == 1.0039506158752403
        @test H.TRAIN_TIMESTEPS == 1000
    end

    # ------------------------------------------------------------ preprocessing

    @testset "recentring is exact integer geometry" begin
        # A 100x80 frame whose alpha is a 41x41 block. Upstream's arithmetic:
        # S = 100, extents are max-min = 40 (not 41), desired = int(100*0.85) =
        # 85, scale = 85/40, so w2 = h2 = 85 and the offset is (100-85)/2 = 7.
        rgba = zeros(UInt8, 100, 80, 4)
        rgba[20:60, 10:50, 1:3] .= 0x40
        rgba[20:60, 10:50, 4] .= 0xff
        rgb, alpha = H.recenter(rgba)
        @test size(rgb) == (100, 100, 3)
        @test size(alpha) == (100, 100, 1)
        cols = findall(i -> any(@view(alpha[i, :, 1]) .!= 0), 1:100)
        rows = findall(j -> any(@view(alpha[:, j, 1]) .!= 0), 1:100)
        @test extrema(cols) == (8, 92)
        @test extrema(rows) == (8, 92)
        # everything outside the object is composited onto white
        @test all(rgb[1:7, :, :] .== 0xff)
        @test all(rgb[93:100, :, :] .== 0xff)
        # an image with no alpha at all is an error, not an empty mesh later
        @test_throws ArgumentError H.recenter(zeros(UInt8, 8, 8, 4))
    end

    @testset "prepareimage lands in (-1, 1) at the configured size" begin
        rgba = zeros(UInt8, 64, 64, 4)
        rgba[10:50, 10:50, :] .= 0xff
        img, msk = H.prepareimage(rgba)
        @test size(img) == (H.PROC_RES, H.PROC_RES, 3, 1)
        @test size(msk) == (H.PROC_RES, H.PROC_RES, 1, 1)
        @test all(-1 .<= img .<= 1)
        # white object on a white background: every pixel is 255 -> +1
        @test all(img .== 1.0f0)
    end

    @testset "normalizeimage is fp16 all the way, constants included" begin
        img = fill(0.0f0, H.PROC_RES, H.PROC_RES, 3, 1)     # midpoint of (-1,1) -> 0.5
        out = H.normalizeimage(img)
        @test size(out) == (H.DINO_RES, H.DINO_RES, 3, 1)
        @test eltype(out) === Float16
        for c in 1:3
            # `F.normalize` builds mean/std at the INPUT's dtype, so the divisor
            # is Float16(0.229) and not 0.229.
            want = (Float16(0.5) - Float16(H.DINO_MEAN[c])) / Float16(H.DINO_STD[c])
            @test out[1, 1, c, 1] === want
        end
        @test_throws ArgumentError H.normalizeimage(fill(0.0f0, 8, 4, 3, 1))
    end

    # -------------------------------------------------------- surface extraction

    @testset "every one of the 256 sign patterns closes" begin
        bad = Int[]
        for pat in 0:255
            f = Array{Float32}(undef, 2, 2, 2)
            for c in 1:8
                dx, dy, dz = H.CORNER[c]
                f[dx + 1, dy + 1, dz + 1] = ((pat >> (c - 1)) & 1) == 1 ? 1.0f0 : -1.0f0
            end
            v, t = H.marchingcubes(f; level = 0.0)
            used = zeros(Int, size(v, 2))
            for j in 1:size(t, 2), r in 1:3
                used[t[r, j]] += 1
            end
            # no vertex may be produced and then left unreferenced: that is a
            # loop that failed to close.
            (size(v, 2) == count(!=(0), used)) || push!(bad, pat)
            # empty only for the two uniform patterns
            (pat == 0 || pat == 255) && (size(t, 2) == 0 || push!(bad, pat))
        end
        @test isempty(bad)
    end

    @testset "a sphere comes out watertight and the right size" begin
        n, R = 48, 0.7
        xs = range(-1, 1; length = n)
        sph = Float32[R^2 - (x^2 + y^2 + z^2) for x in xs, y in xs, z in xs]
        v, t = H.marchingcubes(sph; level = 0.0)
        @test size(v, 1) == 3 && size(t, 1) == 3
        @test size(t, 2) > 1000

        edges = Dict{Tuple{Int32,Int32},Int}()
        for j in 1:size(t, 2), r in 1:3
            edges[minmax(t[r, j], t[mod1(r + 1, 3), j])] = 1 +
                get(edges, minmax(t[r, j], t[mod1(r + 1, 3), j]), 0)
        end
        # watertight: every edge belongs to exactly two triangles
        @test all(==(2), values(edges))
        # and it is a single sphere, not a sphere plus debris
        @test size(v, 2) - length(edges) + size(t, 2) == 2

        h = Float32(2 / (n - 1))
        vw = @. Float32(v) * h - 1.0f0
        area = 0.0
        vol = 0.0
        for j in 1:size(t, 2)
            p1 = @view vw[:, t[1, j]]
            p2 = @view vw[:, t[2, j]]
            p3 = @view vw[:, t[3, j]]
            a, b = p2 .- p1, p3 .- p1
            cr = (a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1])
            area += 0.5 * sqrt(sum(abs2, cr))
            vol += sum(p1 .* cr) / 6
        end
        @test isapprox(area, 4π * R^2; rtol = 0.01)
        @test isapprox(vol, 4 / 3 * π * R^3; rtol = 0.01)
        # positive signed volume: normals point outward, i.e. toward decreasing
        # field, which is `skimage`'s `gradient_direction = "descent"`.
        @test vol > 0
    end

    # -------------------------------------------------------- mesh clean-up

    # A triangle strip: face `i` is `(i, i+1, i+2)`, so it shares an edge with
    # face `i-1`. Two of these at disjoint index ranges are two components.
    strip(n, off) = reshape(reduce(vcat, [Int32[off + i, off + i + 1, off + i + 2]
                                          for i in 1:n]), 3, n)

    @testset "components join across edges, not vertices" begin
        # MeshLab's `ConnectedComponents` walks face-face adjacency, which is
        # across shared EDGES. Two triangles meeting at one vertex are two
        # components — verified against pymeshlab, which keeps both at a ratio
        # that would remove them if it saw one component of two faces.
        F = Int32[1 1; 2 4; 3 5]
        _, n = H.facecomponents(F, 5)
        @test n == 2
        # and a strip is one
        @test H.facecomponents(strip(10, 0), 12)[2] == 1
    end

    @testset "the floater threshold is truncated, against the largest component" begin
        F = hcat(strip(1024, 0), strip(4, 1026))
        V = zeros(Float32, 3, 1032)
        @test H.facecomponents(F, 1032)[2] == 2

        # `faces < trunc(ratio * largest)`. With largest = 1024 and a 4-face
        # speck the flip is at exactly 5/1024, NOT 4/1024 — both sides checked
        # against pymeshlab.
        keeps(r) = size(H.removefloaters(V, F; nbfaceratio = r)[2], 2)
        @test keeps(4 / 1024) == 1028            # trunc = 4, and 4 < 4 is false
        @test keeps(4.9 / 1024) == 1028          # trunc = 4 still
        @test keeps(5 / 1024) == 1024            # trunc = 5, speck drops
        @test keeps(H.FLOATER_RATIO) == 1024     # 0.005 -> trunc = 5

        # Against the LARGEST, not the total: with three big components the
        # speck is 0.13% of the mesh yet survives a 0.2% threshold, because it
        # is 0.39% of the biggest one.
        F3 = hcat(strip(1024, 0), strip(1024, 1026), strip(1024, 2052), strip(4, 3078))
        V3 = zeros(Float32, 3, 3084)
        @test size(H.removefloaters(V3, F3; nbfaceratio = 0.002)[2], 2) == 3076
        @test size(H.removefloaters(V3, F3; nbfaceratio = 0.005)[2], 2) == 3072
    end

    @testset "degenerate removal welds first, and stays closed" begin
        # Two triangles sharing an edge, with a third vertex duplicated at the
        # same position — what marching cubes produces when the field is exactly
        # at the level on a grid vertex.
        V = Float32[0 1 0 0; 0 0 1 1; 0 0 0 0]
        F = Int32[1 1; 2 2; 3 4]                 # verts 3 and 4 coincide
        v, f = H.weld(V, F)
        @test size(v, 2) == 3                    # welded to three distinct points
        v, f = H.removedegenerate(V, F)
        # both faces become (1,2,w) after welding, so one survives as a triangle
        # and the duplicate is dropped rather than leaving a torn edge
        @test size(f, 2) == 2
        @test size(v, 2) == 3
    end

    # --------------------------------------------------------- decimation

    @testset "decimation hits the target and keeps the surface" begin
        n, R = 32, 0.7
        xs = range(-1, 1; length = n)
        sph = Float32[R^2 - (x^2 + y^2 + z^2) for x in xs, y in xs, z in xs]
        v, f = H.marchingcubes(sph; level = 0.0)
        v = Float32.(v) .* Float32(2 / (n - 1)) .- 1.0f0
        nf = size(f, 2)
        @test nf > 2000

        dv, df = H.decimate(v, f; maxfaces = 1000)
        @test size(df, 2) == 1000                # exactly the target

        # still closed, and still a sphere rather than a sphere with a bite out
        edges = Dict{Tuple{Int32,Int32},Int}()
        for j in 1:size(df, 2), r in 1:3
            k = minmax(df[r, j], df[mod1(r + 1, 3), j])
            edges[k] = get(edges, k, 0) + 1
        end
        @test all(==(2), values(edges))
        @test size(dv, 2) - length(edges) + size(df, 2) == 2

        area(V, Fc) = sum(1:size(Fc, 2); init = 0.0) do j
            a, b, c = V[:, Fc[1, j]], V[:, Fc[2, j]], V[:, Fc[3, j]]
            u, w = b .- a, c .- a
            0.5 * sqrt(sum(abs2, (u[2] * w[3] - u[3] * w[2],
                                  u[3] * w[1] - u[1] * w[3],
                                  u[1] * w[2] - u[2] * w[1])))
        end
        # a decimator that hits its face count by collapsing the wrong edges
        # still hits its face count; the area is what catches it.
        @test isapprox(area(dv, df), area(v, f); rtol = 0.01)

        # below the target it is a no-op, as `reduce_face` is
        v2, f2 = H.decimate(v, f; maxfaces = nf + 1)
        @test size(f2, 2) == nf
    end

    @testset "the geometry sweep writes its own queries" begin
        # The one part of this file that runs a graph, and it needs only the
        # 25 MB geometry decoder — not the 5.7 GiB denoiser. What it pins is the
        # structure `occupancy` depends on: the query pass is declared INTO the
        # decoder's graph, so a chunk is one submission, and which points it
        # writes is a `GPURef` the host stores per run. Both of those are silent
        # when wrong — a stale ref gives the previous chunk's points, which is a
        # plausible-looking mesh — so the grid is checked against the host
        # formula it is supposed to reproduce.
        if !Hunyuan3DRunner.ready()
            @info "Hunyuan3D parts not in the artifact store; skipping the sweep"
        else
            dir = H.geodir()
            geo = DNNKernels.Model(
                Dict("hunyuan3d_geo" => DNNKernels.loadgraph(
                         joinpath(dir, "hunyuan3d_geo.json"))),
                DNNKernels.readsafetensors(joinpath(dir, "weights.safetensors"));
                backend = Mantle.defaultbackend())
            chunk = Int(geo.graphs["hunyuan3d_geo"].buffers["queries"].shape[2])
            dev = Mantle.todevice(geo.backend)
            # The four models are one object here: `occupancy` reads `geo`,
            # `chunk` and the ref, and giving it the same model four times keeps
            # this test off the denoiser. That it CONSTRUCTS is half the
            # assertion — the field types named `Model{B}`, which is `Model`'s
            # device parameter, so no four models could satisfy them at all.
            m = H.Hunyuan3D(geo.backend, geo, geo, geo, geo, chunk,
                            Mantle.GPURef(dev, H.GridChunk(0, 0, 0.0, 0.0, 0)))
            latents = Mantle.storage(Mantle.Buffer(dev, Float16, (1024, 4096, 1)))
            fill!(latents, Float16(0.01))

            hostgrid(G, first0) = begin
                N = G^3
                step = 2.02 / (G - 1)
                q = zeros(Float16, 3, chunk)
                for c in 1:chunk
                    n = min(first0 + c - 1, N - 1)      # 0-based, clamped tail
                    k = n % G; j = (n ÷ G) % G; i = n ÷ (G * G)
                    q[1, c] = Float16(-1.01 + step * i)
                    q[2, c] = Float16(-1.01 + step * j)
                    q[3, c] = Float16(-1.01 + step * k)
                end
                q
            end
            queries() = Array(reshape(
                first(DNNKernels.recordedplan(geo, "hunyuan3d_geo").inputs), 3, chunk))

            field = H.occupancy(m, latents; octree = 32)
            @test size(field) == (33, 33, 33)
            @test all(isfinite, Array(field))
            # The LAST chunk, because that is the one with a clamped tail.
            @test queries() == hostgrid(33, (cld(33^3, chunk) - 1) * chunk)

            # A second sweep at a different resolution replays the SAME plan —
            # the grid is in the ref, not baked into the recording.
            nplans = count(v -> v isa DNNKernels.RecordedPlan, values(geo.scratch))
            f2 = H.occupancy(m, latents; octree = 16)
            @test size(f2) == (17, 17, 17)
            @test queries() == hostgrid(17, 0)
            @test count(v -> v isa DNNKernels.RecordedPlan, values(geo.scratch)) == nplans

            DNNKernels.releaseplans!(geo)
            DNNKernels.releaseweights!(geo.weights)
        end
    end

    @testset "the box mapping divides by octree + 1" begin
        # Upstream's `vertices / grid_size * bbox_size + bbox_min` with
        # `grid_size = octree + 1`. Index 0 maps to -box_v exactly; index
        # `octree` lands SHORT of +box_v, by one grid step. Reproduced because it
        # is what the reference mesh is, not because it is right.
        v = Float32[0 384; 0 384; 0 384]
        w = H.meshbox(v, 384; box_v = 1.01)
        @test w[1, 1] ≈ -1.01f0
        @test w[1, 2] ≈ -1.01f0 + 384 / 385 * 2.02f0
        @test w[1, 2] < 1.01f0
    end
end
