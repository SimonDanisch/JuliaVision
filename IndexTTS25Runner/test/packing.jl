# Small model-free regression: pointer selection in stackmu! previously wrote
# zeros on RTX 4000 Ada. Distinct inputs and poisoned outputs check every frame,
# including the concatenation boundary and the unconditional guidance batch.
function testpacking(dev)
    @testset "semantic conditioning packing" begin
        for (np, nc) in ((1, 1), (2, 3), (210, 350))
            prompt = reshape(Float32.(1:512np), 512, np, 1)
            cond = reshape(-Float32.(1:512nc), 512, nc, 1)
            expected = cat(cat(prompt, cond; dims=2), zeros(Float32, 512, np+nc, 1); dims=3)
            buffers = map(a -> Mantle.Buffer(dev, a),
                          (prompt, cond, fill(-Inf32, 512, np+nc, 2)))
            a, b, out = buffers
            try
                g = Mantle.Graph(dev)
                Mantle.dispatch!(g, R.stackmu!, (out, a, b, Int32(np), Int32(np+nc)), 512(np+nc))
                Mantle.runonce!(g)
                @test Array(out) == expected
            finally
                foreach(Mantle.free!, buffers)
            end
        end
    end
end
