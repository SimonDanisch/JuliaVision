# The operations that serve page and model pipelines: pixel formats, turns,
# components, morphology, histograms, quantiles, thresholds and resampling,
# each against a direct reference on the CPU backend.

using GPUFiltering: luma, Bilinear, Area, quantilesizes

turnref(img, d) = d == 0 ? img : d == 90 ? reverse(permutedims(img); dims = 1) :
                  d == 180 ? reverse(img) : reverse(permutedims(img); dims = 2)

@testset "bilinear, areamean" begin
    # Singleton borders replicate instead of reading outside.
    @test GPUFiltering.bilinear(fill(0.3f0, 1, 1), -10, 20) == 0.3f0
    @test GPUFiltering.bilinear(reshape(Float32[0, 1], 1, 2), 1, 1.5) == 0.5f0
    @test GPUFiltering.bilinear(reshape(Float32[0, 1], 2, 1), 1.5, 1) == 0.5f0
    @test GPUFiltering.bilinear([Gray(0.0f0) Gray(1.0f0)], 1, 1.25) == RGB{Float32}(0.25, 0.25, 0.25)
    img = Float32[1 2; 3 4]
    @test GPUFiltering.areamean(img, 0.0f0, 2.0f0, 0.0f0, 2.0f0) == 2.5f0
    @test GPUFiltering.areamean(img, 1.0f0, 2.0f0, 0.0f0, 1.0f0) == 3.0f0
end

@testset "convertpixels!, grayscale!, rotate90!" begin
    img = rand(RGB{N0f8}, 13, 7)
    f = convertpixels!(similar(img, RGB{Float32}), img)
    @test f == RGB{Float32}.(img)
    @test convertpixels!(similar(img), f) == img
    g = rand(Gray{Float32}, 5, 4)
    @test convertpixels!(Matrix{RGB{Float32}}(undef, 5, 4), g) == RGB{Float32}.(g)
    @test grayscale!(Matrix{Gray{Float32}}(undef, size(img)), img) ≈ Gray{Float32}.(luma.(img))
    for d in (0, 90, 180, 270)
        out = similar(img, d in (90, 270) ? reverse(size(img)) : size(img))
        @test rotate90!(out, img, d) == turnref(img, d)
    end
    @test_throws ArgumentError rotate90!(similar(img), img, 45)
    @test_throws DimensionMismatch rotate90!(similar(img), img, 90)
end

@testset "components!" begin
    mask = zeros(Int32, 9, 6)
    mask[2:3, 2:3] .= 1          # a square
    mask[4, 4] = 1               # touching it at a corner only: same component
    mask[7:8, 1:5] .= 1          # a bar
    labels = components!(zeros(Int32, size(mask)), mask)
    @test all(iszero, labels[mask .== 0])
    square = labels[2, 2]
    @test all(==(square), labels[[CartesianIndex(2, 2), CartesianIndex(3, 3), CartesianIndex(4, 4)]])
    @test all(==(labels[7, 1]), labels[7:8, 1:5])
    @test labels[7, 1] != square
    @test square == LinearIndices(mask)[2, 2]   # the component's first pixel
    # A predicate picks the pixels: here the values above a threshold.
    prob = Float32.(mask) .* 0.8f0
    @test components!(zeros(Int32, size(mask)), prob, >(0.3f0)) == labels
end

@testset "erode!, dilate!, opening!" begin
    img = Int32.(rand(0:1, 23, 17))
    window(f, a, r) = [f(a[max(1, x - r):min(end, x + r), max(1, y - r):min(end, y + r)]) for x in axes(a, 1), y in axes(a, 2)]
    for r in (1, 2)
        @test erode!(similar(img), img, r) == window(minimum, img, r)
        @test dilate!(similar(img), img, r) == window(maximum, img, r)
        @test opening!(similar(img), img, r) == window(maximum, window(minimum, img, r), r)
    end
end

@testset "histogram!, otsu!, selectquantile!" begin
    v = Float32[fill(0.2f0, 300); fill(0.8f0, 200)]
    img = reshape(v[randperm(length(v))], 25, 20)
    counts = histogram!(zeros(Int32, 256, 64), img, identity)
    @test sum(counts) == length(img)
    @test sum(counts[52, :]) == 300 && sum(counts[205, :]) == 200
    t = otsu!(zeros(Float32, 2), counts)
    @test 0.2f0 < t[1] < 0.8f0
    @test t[2] ≈ 0.6f0 atol = 0.01
    # One class only: no separation.
    @test otsu!(zeros(Float32, 2), histogram!(zeros(Int32, 256, 64), fill(0.5f0, 10, 10), identity))[2] == 0
    vals = rand(Float32, 37, 29)
    for (p, stride) in ((0.005, 1), (0.5, 3), (1.0, 1))
        q = selectquantile!(zeros(Float32, 1), vals, p, identity, stride)
        s = sort(vec(vals)[1:stride:end])
        @test q[1] == s[clamp(round(Int, p * length(s)), 1, length(s))]
    end
    @test quantilesizes(37 * 29; stride = 3) == ((256, cld(length(1:3:37 * 29), 256)), (256,), (2,))
end

@testset "sauvola!" begin
    img = rand(RGB{Float32}, 40, 30)
    lum = luma.(img)
    ref = [begin
               win = lum[max(1, x - 7):min(end, x + 7), max(1, y - 7):min(end, y + 7)]
               m = sum(win) / length(win)
               sd = sqrt(max(sum(abs2, win) / length(win) - m^2, 0))
               lum[x, y] > m * (1 + 0.2f0 * (sd / 0.5f0 - 1))
           end for x in 1:40, y in 1:30]
    out = sauvola!(Matrix{Gray{Bool}}(undef, 40, 30), img, 15)
    @test count(gray.(out) .!= ref) <= 2   # float summation order at the threshold
end

@testset "gridsample!" begin
    img = [RGB{Float32}(x / 9, y / 7, (x + y) / 16) for x in 1:9, y in 1:7]
    grid(f, gw, gh) = (g = Array{Float32}(undef, gw, gh, 2, 1);
        for j in 1:gh, i in 1:gw
            u, v = -1 + 2(i - 1) / (gw - 1), -1 + 2(j - 1) / (gh - 1)
            g[i, j, 1, 1], g[i, j, 2, 1] = f(u, v)
        end; g)
    near(a, b) = maximum(abs, reinterpret(Float32, a) .- reinterpret(Float32, b)) < 2.0f-6
    @test near(gridsample!(similar(img), img, grid((u, v) -> (u, v), 4, 5)), img)
    @test near(gridsample!(similar(img), img, grid((u, v) -> (-u, v), 4, 5)), img[end:-1:1, :])
    @test all(iszero ∘ red, gridsample!(similar(img), img, grid((u, v) -> (3.0f0, 3.0f0), 2, 2)))
end

@testset "sampleplanar!, resizeplanar!, areadownscale! by blocks" begin
    img = rand(RGB{Float32}, 12, 8)
    # Area sampling at a factor of two is the mean of 2x2 blocks.
    dst = zeros(Float32, 6, 4, 3, 1)
    sampleplanar!(dst, img, PlanarMap((2, 2), (0, 0), (6, 4)), Area())
    blocks = [sum(img[2x-1:2x, 2y-1:2y]) / 4 for x in 1:6, y in 1:4]
    @test dst[:, :, 1, 1] ≈ red.(blocks) && dst[:, :, 3, 1] ≈ blue.(blocks)
    small = areadownscale!(Matrix{RGB{Float32}}(undef, 6, 4), img, Blocks(2))
    @test small ≈ blocks
    @test areadownscale!(Matrix{RGB{Float32}}(undef, 2, 2), img, Blocks(3, (3, 1))) ≈
          [sum(img[3+3x-2:3+3x, 1+3y-2:1+3y]) / 9 for x in 1:2, y in 1:2]
    @test_throws DimensionMismatch areadownscale!(Matrix{RGB{Float32}}(undef, 7, 4), img, Blocks(2))
    # Outside the extent: the fill, normalised.
    sampleplanar!(dst, img, PlanarMap((2, 2), (0, 0), (3, 4)), Area(), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 1)
    @test all(==(1.0f0), dst[4:6, :, :, 1])
    @test dst[1:3, :, 1, 1] ≈ (red.(blocks[1:3, :]) .- 0.5f0) ./ 0.5f0
    # A batch: one map per slot, as lines cut out of a page; an empty extent is all fill.
    maps = [PlanarMap((2, 2), (0, 0), (6, 4)), PlanarMap((1, 1), (4, 2), (6, 4)), PlanarMap((1, 1), (0, 0), (0, 0))]
    batch = zeros(Float32, 6, 4, 3, 3)
    sampleplanar!(batch, img, maps, Bilinear())
    for k in 1:2
        one = zeros(Float32, 6, 4, 3, 1)
        sampleplanar!(one, img, maps[k], Bilinear())
        @test batch[:, :, :, k] == one[:, :, :, 1]
    end
    @test all(iszero, batch[:, :, :, 3])
    @test_throws DimensionMismatch sampleplanar!(batch, img, maps[1:2], Bilinear())
    # resizeplanar! is torch's bilinear upsample with align_corners=False.
    up = zeros(Float32, 20, 11, 3, 1)
    resizeplanar!(up, img)
    torch(u, n, N) = max((u - 0.5f0) * n / N - 0.5f0, 0.0f0)
    ref = [begin
               fx, fy = torch(u, 12, 20), torch(v, 8, 11)
               x0, y0 = floor(Int, fx), floor(Int, fy)
               dx, dy = fx - x0, fy - y0
               x1, y1 = min(x0 + 2, 12), min(y0 + 2, 8); x0 += 1; y0 += 1
               (1 - dx) * (1 - dy) * red(img[x0, y0]) + dx * (1 - dy) * red(img[x1, y0]) +
               (1 - dx) * dy * red(img[x0, y1]) + dx * dy * red(img[x1, y1])
           end for u in 1:20, v in 1:11]
    @test up[:, :, 1, 1] ≈ ref
end

@testset "boxrows!, fillboxes!, componentboxes!" begin
    img = rand(RGB{Float32}, 20, 12)
    mask = rand(Bool, 20, 12)
    boxes = [Int32.((2, 3, 9, 7)), Int32.((15, 1, 20, 12)), Int32.((1, 1, 1, 1))]
    stats = fill(RowStats(), 12, 3)
    boxrows!(stats, img, mask, boxes, identity, 2)
    for k in 1:2, r in 1:12
        x0, y0, x1, y1 = boxes[k]
        y = y0 + r - 1
        st = stats[r, k]
        if y > y1
            @test st.count == 0 && st.runs == 0 && st.first == 0
        else
            row = mask[x0:x1, y]
            @test st.count == count(row)
            @test st.runs == count(i -> row[i] && (i == 1 || !row[i - 1]), eachindex(row))
            @test st.first == (any(row) ? x0 + findfirst(row) - 1 : 0)
            @test st.last == (any(row) ? x0 + findlast(row) - 1 : 0)
            @test st.sum ≈ sum(img[x0:x1, y][row]; init = RGB{Float32}(0, 0, 0))
        end
    end
    # Boxes past the count are left out.
    @test all(st -> st.count == 0, stats[:, 3])

    filled = fillboxes!(copy(mask), boxes, false, 2)
    inside(x, y) = any(b -> b[1] <= x <= b[3] && b[2] <= y <= b[4], boxes[1:2])
    @test all(filled[x, y] == (inside(x, y) ? false : mask[x, y]) for x in 1:20, y in 1:12)

    blobs = zeros(Int32, 10, 8)
    blobs[2:3, 2:4] .= 1; blobs[7:9, 6:8] .= 1; blobs[6, 1] = 1
    labels = components!(zeros(Int32, 10, 8), blobs)
    found = zeros(Int32, 5, 4); count1 = zeros(Int32, 1)
    componentboxes!(found, count1, labels, 2)
    @test count1[1] == 2
    @test Set(eachcol(found[:, 1:2])) == Set([Int32[2, 2, 3, 4, 6], Int32[7, 6, 9, 8, 9]])
end
