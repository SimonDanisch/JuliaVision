using Test, Random, BonsaiRunner
import Mantle

# Decode's layer boundary as one pass: the residual add, the RMS norm and the
# signed transform of the norm. Its workgroups each compute the norm over the
# whole row, so a wrong reduction shows up as a block-wise scale error; the
# residual goes to a second buffer, so an in-place write would show up here as a
# doubled branch in some blocks.
@testset "decode boundary matches add, norm and transform" begin
    dev = Mantle.Device()
    sub = Mantle.caps(dev).subgroup
    rng = MersenneTwister(13)
    n = 5120
    xh = randn(rng, Float32, n); bh = randn(rng, Float32, n)
    wh = rand(rng, Float32, n) .+ 0.5f0; sh = rand(rng, Float32[-1, 1], n)
    function fwht(v)
        v = Float64.(v); h = 1
        while h < length(v)
            for i in 1:2h:length(v), j in i:i+h-1
                a, b = v[j], v[j+h]; v[j] = a + b; v[j+h] = a - b
            end
            h *= 2
        end
        v ./ sqrt(length(v))
    end
    for hasbranch in (true, false)
        v = hasbranch ? Float64.(xh) .+ bh : Float64.(xh)
        nv = v ./ sqrt(sum(abs2, v) / n + 1e-6) .* wh
        wanthad = reduce(vcat, [fwht(nv[i:i+1023] .* sh[i:i+1023]) for i in 1:1024:n])
        xin, br, w, sg = map(a -> Mantle.Buffer(dev, a), (xh, bh, wh, sh))
        xout, norm, had = (Mantle.Buffer(dev, fill(NaN32, n)) for _ in 1:3)
        g = Mantle.Graph(dev)
        Mantle.dispatch!(g, BonsaiRunner.add_rmsnorm_hadamard_kernel!,
            (hasbranch ? xout : xin, norm, had, xin, hasbranch ? br : xin, w, sg, Int32(n), 1f-6,
             Val(hasbranch), Val(sub)), (n ÷ 1024) * 256; group = 256, name = "boundary")
        Mantle.runonce!(g)
        @test Array(Mantle.storage(norm)) ≈ nv rtol = 2e-6
        @test Array(Mantle.storage(had)) ≈ wanthad rtol = 2e-5 atol = 1e-5
        hasbranch && @test Array(Mantle.storage(xout)) == xh .+ bh
        @test Array(Mantle.storage(xin)) == xh
        foreach(Mantle.free!, (xin, br, w, sg, xout, norm, had))
    end
end

# The greedy pick on the device. It has to be Julia's `argmax` exactly, since the
# host scan it replaced was: the first of equal maxima, and NaN above every number.
@testset "device greedy pick is argmax" begin
    dev = Mantle.Device()
    rng = MersenneTwister(17)
    n = 248320
    cases = Vector{Float32}[randn(rng, Float32, n)]
    tie = randn(rng, Float32, n); tie[1000] = 50f0; tie[200000] = 50f0; push!(cases, tie)
    nan = randn(rng, Float32, n); nan[777] = NaN32; nan[5] = 1f6; push!(cases, nan)
    odd = randn(rng, Float32, 1001); odd[end] = 99f0; push!(cases, odd)
    # Fewer values than the first pass has parts: most parts are empty.
    short = randn(rng, Float32, 100); short[3] = -Inf32; push!(cases, short)
    push!(cases, fill(-Inf32, 300))    # all equal: the first
    for x in cases
        xb = Mantle.Buffer(dev, x); best = Mantle.Buffer(dev, Int32, (1,))
        g = Mantle.Graph(dev)
        BonsaiRunner.declare_argmax!(g, best, xb; name = "greedy")
        Mantle.runonce!(g)
        @test only(Array(Mantle.storage(best))) == argmax(x) - 1
        foreach(Mantle.free!, (xb, best))
    end
end

# alpha and beta in one launch, from the checkpoint's raw BF16 words. The words
# are the top halves of Float32 values, so the reference is exact in Float64.
@testset "two BF16 projections in one launch" begin
    dev = Mantle.Device()
    sub = Mantle.caps(dev).subgroup
    rng = MersenneTwister(19)
    K, M = 5120, 48
    words = (rand(rng, UInt16, K, M) .& 0x3fff, rand(rng, UInt16, K, M) .& 0x3fff)
    xh = randn(rng, Float32, K)
    val(wd) = Float64.(reinterpret.(Float32, UInt32.(wd) .<< 16))
    w1, w2, x = Mantle.Buffer(dev, words[1]), Mantle.Buffer(dev, words[2]), Mantle.Buffer(dev, xh)
    o1, o2 = Mantle.Buffer(dev, fill(NaN32, M)), Mantle.Buffer(dev, fill(NaN32, M))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, BonsaiRunner.dense2_gemv_kernel!,
        (o1, o2, w1, w2, x, Int32(K), Int32(M), Val(sub)), 2M * 256; group = 256, name = "ab")
    Mantle.runonce!(g)
    for (o, wd) in ((o1, words[1]), (o2, words[2]))
        want = transpose(val(wd)) * Float64.(xh)
        @test Array(Mantle.storage(o)) ≈ want rtol = 1e-5
    end
    foreach(Mantle.free!, (w1, w2, x, o1, o2))
end
