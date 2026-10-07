using Test, DNNKernels, Mantle, Random, LinearAlgebra

@testset "masked attention with partial key tiles" begin
    dev = Mantle.Device()
    caps = Mantle.caps(dev)
    if caps.coopmat && caps.coopmatsubgroup == 32
        rng = MersenneTwister(83)
        for lq in (8, 16, 24, 64), lk in (16, 33)
            e, h = 128, 2
            qh = randn(rng, Float16, e, lq, h, 1) .* Float16(0.2)
            kh = randn(rng, Float16, e, lk, h, 1) .* Float16(0.2)
            vh = randn(rng, Float16, e, lk, h, 1)
            maskh = [s <= min(r, lk) ? Float16(0) : -floatmax(Float16)
                     for s in 1:lk, r in 1:lq, _ in 1:1, _ in 1:1]
            # Poison storage beyond the mask so a padded-key read cannot pass silently.
            parent = Mantle.Buffer(dev, vcat(vec(maskh), fill(Float16(NaN), 64)))
            mask = Mantle.viewof(parent, size(maskh))
            q, k, v = map(a -> Mantle.Buffer(dev, a), (qh, kh, vh))
            out = Mantle.Buffer(dev, Float16, size(qh))
            scale = Float32(inv(sqrt(e)))
            flash = DNNKernels.flashcm_plan(caps, q, k, v, nothing;
                clamp=true, BR=16, BC=32, NW=8, rego=false, split=false)
            @test flash isa DNNKernels.FlashCMPlan
            g = Mantle.Graph(dev)
            DNNKernels.flash_dispatch!(g, caps, out, flash, q, k, v, scale,
                                      nothing, nothing; mask)
            plan = Mantle.record!(Mantle.Plan(g))
            try
                Mantle.run!(plan)
                got = Float32.(Array(out))
                want = similar(got)
                for head in 1:h
                    scores = Float16.(Float16.(Float32.(kh[:,:,head,1])' * Float32.(qh[:,:,head,1])) .* Float16(scale))
                    scores = Float32.(Float16.(scores .+ maskh[:,:,1,1]))
                    p = exp.(scores .- maximum(scores; dims=1))
                    p = Float16.(p ./ sum(p; dims=1))
                    want[:,:,head,1] = Float32.(vh[:,:,head,1]) * Float32.(p)
                end
                @test all(isfinite, got)
                @test maximum(abs, got .- want) < 0.005
            finally
                Mantle.free!(plan)
                foreach(Mantle.free!, (parent, q, k, v, out))
            end
        end
    else
        @test_skip false
    end
end
