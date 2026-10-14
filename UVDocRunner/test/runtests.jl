using Test, UVDocRunner, ColorTypes, DNNKernels

rgbarray(a) = [RGB{Float32}(a[x,y,1,1], a[x,y,2,1], a[x,y,3,1]) for x in axes(a,1), y in axes(a,2)]
maxerror(a,b) = maximum(max(abs(red(x)-red(y)), abs(green(x)-green(y)), abs(blue(x)-blue(y))) for (x,y) in zip(a,b))

@testset "UVDocRunner" begin
    @testset "grid coordinates and padding" begin
        # Unequal sides and distinct channels catch transposition/channel swaps.
        img = [RGB{Float32}(x/9, y/7, (x+y)/16) for x in 1:9, y in 1:7]
        grid = cat(Float32[-1 -1; 1 1], Float32[-1 1; -1 1]; dims=3)
        @test maxerror(unwarp(img,grid), img) < 2e-6
        @test maxerror(unwarp(img,grid[2:-1:1,:,:]), img[end:-1:1,:]) < 2e-6
        @test all(iszero ∘ red, unwarp(img,fill(3f0,2,2,2)))
        grid[:,:,1] .= -1.125f0 # source x=0.5: half first column, half padding
        grid[:,:,2] .= -1f0
        @test red(unwarp(img,grid)[1,1]) ≈ red(img[1,1])/2
        @test size(unwarp(img,grid; output_size=(1,1))) == (1,1)
        @test_throws ArgumentError unwarp(img,zeros(Float32,2,2,3))
        @test_throws ArgumentError unwarp(img,fill(NaN32,2,2,2))
        @test_throws ArgumentError unwarp(img,grid; output_size=(0,3))
        @test_throws ArgumentError UVDocRunner.prepare(img[1:0,:])
    end

    refpath = joinpath(@__DIR__, "..", "..", "gen", "graphs", "uvdoc", "reference.safetensors")
    if isfile(refpath)
        ref = readsafetensors(refpath)
        img = rgbarray(ref["original"])
        @testset "upstream preprocessing and resampling" begin
            @test maximum(abs.(UVDocRunner.prepare(img) .- ref["input"])) < 2e-5
            @test maxerror(unwarp(img,ref["grid"][:,:,:,1]), rgbarray(ref["unwarped"])) < 1e-4
        end
        if UVDocRunner.ready()
            @testset "GPU against PyTorch, including background tasks" begin
                m = uvdoc()
                try
                    grid = deformation_grid(m,img)
                    @test size(grid) == (31,45,2)
                    @test maximum(abs.(grid .- ref["grid"][:,:,:,1])) < 2e-4
                    # Model outputs must be copied before the next request
                    # reuses its buffers; GPU submission stays on its owner.
                    grids = fetch.([Threads.@spawn deformation_grid(m,img) for _ in 1:2])
                    @test all(g -> maximum(abs.(g .- grid)) < 1e-6, grids)
                    @test maxerror(unwarp(m,img), rgbarray(ref["unwarped"])) < 5e-3
                finally
                    close(m)
                end
            end
        else
            @test_skip "UVDoc artifact not installed"
        end
    else
        @test_skip "Export references with tools/export_uvdoc.py to test against PyTorch"
    end
end
