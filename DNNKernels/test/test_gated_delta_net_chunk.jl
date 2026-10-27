# The chunk Gated DeltaNet kernel against `n` launches of the one-token kernel:
# outputs of every token and the final state, in the native and the tiled
# value-head order, for chunks of one token and more. To rounding, not bitwise:
# the driver may fuse multiplies and adds differently in the two shaders. A wrong
# head, token column or state run is off by the values' own size.
module GatedDeltaNetChunkTests
using Test, Random, DNNKernels
import DNNKernels: Mantle

const DK = DNNKernels

# The one-token kernel reads q and k already repeated to every value head.
keyhead(h, nh, nkh, tiled) = tiled ? h % nkh : h ÷ (nh ÷ nkh)

function run(device)
    rng = MersenneTwister(11)
    L = DK.GDN_LANES
    @testset "chunk Gated DeltaNet is n one-token launches" begin
        for (nh, nkh) in ((32, 16), (48, 16), (16, 16)), tiled in (false, true), n in (1, 5, 17)
            channels = 256nkh + 128nh; vd = 128nh
            conv = randn(rng, Float32, channels, n)
            alpha = randn(rng, Float32, nh, n); beta = randn(rng, Float32, nh, n)
            z = randn(rng, Float32, vd, n)
            dt = randn(rng, Float32, nh); a = -rand(rng, Float32, nh) .- 0.1f0
            nw = 1f0 .+ 0.1f0 .* randn(rng, Float32, 128)
            state0 = 0.1f0 .* randn(rng, Float32, 128 * 128 * nh)
            owned = Any[]; plan = nothing
            up(x) = (b = Mantle.Buffer(device, x); push!(owned, b); b)
            try
                bconv, balpha, bbeta, bz, bdt, ba, bnw = up(conv), up(alpha), up(beta), up(z), up(dt), up(a), up(nw)
                chunkstate = up(copy(state0)); chunkout = up(fill(NaN32, vd, n))
                tokenstate = up(copy(state0)); tokenout = [up(fill(NaN32, vd)) for _ in 1:n]
                rq = [up(reduce(vcat, [conv[keyhead(h, nh, nkh, tiled)*128+1:keyhead(h, nh, nkh, tiled)*128+128, j] for h in 0:nh-1])) for j in 1:n]
                rk = [up(reduce(vcat, [conv[128nkh+keyhead(h, nh, nkh, tiled)*128+1:128nkh+keyhead(h, nh, nkh, tiled)*128+128, j] for h in 0:nh-1])) for j in 1:n]
                g = Mantle.Graph(device)
                Mantle.dispatch!(g, DK.gated_delta_net_chunk_kernel!,
                    (chunkout, chunkstate, bconv, balpha, bbeta, bdt, ba, bz, bnw, 1f-6,
                     Int32(nh), Int32(nkh), Int32(n), Val(L), Val(tiled)),
                    nh * 128 * L; group = 128 * L, name = "chunk")
                for j in 1:n
                    v = Mantle.viewof(bconv, (vd,); offset = (j - 1) * channels + 256nkh)
                    Mantle.dispatch!(g, DK.gated_delta_net_kernel!,
                        (tokenout[j], tokenstate, rq[j], rk[j], v,
                         Mantle.viewof(balpha, (nh,); offset = (j - 1) * nh), Mantle.viewof(bbeta, (nh,); offset = (j - 1) * nh),
                         bdt, ba, Mantle.viewof(bz, (vd,); offset = (j - 1) * vd), bnw, 1f-6, Int32(nh), Int32(nh), Val(L)),
                        nh * 128 * L; group = 128 * L, name = "token.$j")
                end
                plan = Mantle.record!(Mantle.Plan(g; budget = Inf)); Mantle.run!(plan); Mantle.waitidle(device)
                got = Array(chunkout); want = reduce(hcat, Array.(tokenout))
                @test all(isfinite, got)
                @test maximum(abs.(got .- want)) <= 1f-5 * maximum(abs.(want))
                @test maximum(abs.(Array(chunkstate) .- Array(tokenstate))) <= 1f-6 * maximum(abs.(Array(tokenstate)))
                @test maximum(abs.(Array(chunkstate) .- state0)) > 1f-2
            finally
                plan === nothing || Mantle.free!(plan)
                foreach(Mantle.free!, reverse(owned))
            end
        end
    end
end
end
