using Test, Random, BonsaiRunner, DNNKernels
import Mantle

# Decode attention in two passes: chunks of the context, then their merge. The
# single-pass kernel it replaced walked every position serially per head and was
# 35 ms of a 104 ms token at 2048 positions on an M5. Pinned against the
# definition in Float64, at positions on both sides of a chunk boundary and well
# past one, with a capacity larger than the position so the chunks past it must
# be skipped.
@testset "split-context decode attention matches the definition" begin
    dev = Mantle.Device()
    sub = Mantle.caps(dev).subgroup
    rng = MersenneTwister(5)
    cap = 2 * BonsaiRunner.ATTN_CHUNK + 32
    nc = cld(cap, BonsaiRunner.ATTN_CHUNK)
    qh = randn(rng, Float32, 6144); gh = randn(rng, Float32, 6144)
    kh = rand(rng, Int8, 256, 4, cap); vh = rand(rng, Int8, 256, 4, cap)
    ksh = rand(rng, Float32, 4, cap) .* 0.05f0; vsh = rand(rng, Float32, 4, cap) .* 0.05f0
    bufs = map(a -> Mantle.Buffer(dev, a), (qh, gh, kh, vh, ksh, vsh))
    qd, gd, kd, vd, ksd, vsd = bufs
    for posv in (0, BonsaiRunner.ATTN_CHUNK - 1, BonsaiRunner.ATTN_CHUNK, cap - 1)
        pos = Mantle.Buffer(dev, Int32[posv])
        out = Mantle.Buffer(dev, fill(NaN32, 6144))
        pm = Mantle.Buffer(dev, fill(NaN32, 24nc)); ps = Mantle.Buffer(dev, fill(NaN32, 24nc))
        pa = Mantle.Buffer(dev, fill(NaN32, 24nc * 256))
        g = Mantle.Graph(dev)
        Mantle.dispatch!(g, BonsaiRunner.attention_partial_kernel!,
            (pm, ps, pa, qd, kd, vd, ksd, vsd, pos, Int32(nc), Val(sub)), 4nc * 256;
            group = 256, name = "attention_partial")
        Mantle.dispatch!(g, BonsaiRunner.attention_combine_kernel!,
            (out, pm, ps, pa, gd, pos, Int32(nc)), 24 * 256;
            group = 256, name = "attention_combine")
        Mantle.runonce!(g)
        got = Array(Mantle.storage(out))
        want = similar(got, Float64)
        for h in 0:23
            kvh = h ÷ 6
            sc = [sum(Float64(qh[h*256+d+1]) * kh[d+1, kvh+1, p+1] * ksh[kvh+1, p+1] for d in 0:255) / 16
                  for p in 0:posv]
            w = exp.(sc .- maximum(sc)); w ./= sum(w)
            for d in 0:255
                want[h*256+d+1] = sum(w[p+1] * vh[d+1, kvh+1, p+1] * vsh[kvh+1, p+1] for p in 0:posv) *
                                  gh[h*256+d+1]
            end
        end
        @test !any(isnan, got)
        @test maximum(abs, got .- want) / maximum(abs, want) < 2e-6
        foreach(Mantle.free!, (pos, out, pm, ps, pa))
    end
    foreach(Mantle.free!, bufs)
end

# The prefill's attention: groups of ATTN_TOKENS tokens per KV head with an online
# softmax over the context, against the same definition. Nine tokens starting
# past the first position: each token's causal limit differs, some straddle a
# chunk boundary, and the last group holds one token.
@testset "prefill attention matches the definition" begin
    dev = Mantle.Device()
    rng = MersenneTwister(9)
    cap = 3 * BonsaiRunner.ATTN_CHUNK
    T = 9
    p0 = BonsaiRunner.ATTN_CHUNK - 4
    qh = randn(rng, Float32, 6144T); gh = randn(rng, Float32, 6144T)
    kh = rand(rng, Int8, 256, 4, cap); vh = rand(rng, Int8, 256, 4, cap)
    ksh = rand(rng, Float32, 4, cap) .* 0.05f0; vsh = rand(rng, Float32, 4, cap) .* 0.05f0
    bufs = map(a -> Mantle.Buffer(dev, a), (qh, gh, kh, vh, ksh, vsh, Int32[p0], fill(NaN32, 6144T)))
    qd, gd, kd, vd, ksd, vsd, pos, out = bufs
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, BonsaiRunner.attention_prefill_kernel!,
        (out, qd, gd, kd, vd, ksd, vsd, pos, Int32(T)), cld(T, BonsaiRunner.ATTN_TOKENS) * 4 * 256;
        group = 256, name = "attention_prefill")
    Mantle.runonce!(g)
    got = Array(Mantle.storage(out))
    want = similar(got, Float64)
    for tk in 0:T-1, h in 0:23
        posv = p0 + tk; kvh = h ÷ 6; qb = tk * 6144 + h * 256
        sc = [sum(Float64(qh[qb+d+1]) * kh[d+1, kvh+1, p+1] * ksh[kvh+1, p+1] for d in 0:255) / 16
              for p in 0:posv]
        w = exp.(sc .- maximum(sc)); w ./= sum(w)
        for d in 0:255
            want[qb+d+1] = sum(w[p+1] * vh[d+1, kvh+1, p+1] * vsh[kvh+1, p+1] for p in 0:posv) * gh[qb+d+1]
        end
    end
    @test !any(isnan, got)
    @test maximum(abs, got .- want) / maximum(abs, want) < 2e-6
    foreach(Mantle.free!, bufs)
end
