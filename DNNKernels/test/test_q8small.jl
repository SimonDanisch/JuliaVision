using Test, DNNKernels, Mantle, Random

@testset "packed INT8 short batches" begin
    dev = Mantle.Device()
    rng = MersenneTwister(84)
    @test DNNKernels.q8small_applicable(5120,8192,2)
    @test DNNKernels.q8small_applicable(5120,26624,8)
    @test !DNNKernels.q8small_applicable(53248,5120,4)
    @test !DNNKernels.q8small_applicable(10240,5120,2)
    @test !DNNKernels.q8small_applicable(5120,8192,1)
    @test !DNNKernels.q8small_applicable(5120,8192,9)
    @test !DNNKernels.q8small_applicable(256,512,4)
    # Ragged rows and reduction, signed bytes, and unequal row scales.
    for (m, k) in ((37, 131), (256, 512)), n in 2:8, T in (Float16, Float32)
        mg = cld(m, 4)
        q = rand(rng, UInt32, mg, k)
        scale = rand(rng, Float32, m) .* 0.01f0 .+ 0.002f0
        b = T.(randn(rng, Float32, k, n))
        bias = randn(rng, Float32, m)
        a = DNNKernels.QInt8Matrix(Mantle.Buffer(dev, q), Mantle.Buffer(dev, scale), m)
        x, c, bi = Mantle.Buffer(dev, b), Mantle.Buffer(dev, T, (m, n)), Mantle.Buffer(dev, bias)
        weights = Float32[DNNKernels.q8byte(q[cld(i,4),j], (i-1)%4) * scale[i]
                          for i in 1:m, j in 1:k]
        for splits in (1, 3), hasbias in (false, true)
            g = Mantle.Graph(dev)
            parts = Mantle.Transient.Buffer(g, Float32, (4mg, n, splits))
            DNNKernels.q8small_dispatch!(g, c, a, x, parts, hasbias ? bi : nothing, x -> max(x, 0f0))
            plan = Mantle.record!(Mantle.Plan(g))
            try
                for _ in 1:2
                    Mantle.run!(plan)
                    got = Float32.(Array(c))
                    want = max.(weights * Float32.(b) .+ (hasbias ? bias : zeros(Float32,m)), 0f0)
                    @test all(isfinite, got)
                    @test isapprox(got, want; rtol = T === Float16 ? 0.001 : 2e-5, atol = 2e-5)
                end
            finally
                Mantle.free!(plan)
            end
        end
        foreach(Mantle.free!, (a.q, a.scale, x, c, bi))
    end
end

@testset "short batches through an exported graph" begin
    dev = Mantle.Device()
    rng = MersenneTwister(85)
    m,k = 512,2048
    a = DNNKernels.QInt8Matrix(Mantle.Buffer(dev,rand(rng,UInt32,m÷4,k)),
                              Mantle.Buffer(dev,fill(0.002f0,m)),m)
    q,scale = Array(a.q),Array(a.scale)
    w = Float32[DNNKernels.q8byte(q[cld(i,4),j],(i-1)%4)*scale[i] for i in 1:m,j in 1:k]
    for n in (2,3,8,9)
        buf(id,shape;kind=:transient) = DNNKernels.Buffer(id,kind,Any[shape...],Float16,
            id=="w" ? "w" : "",(0,0),"","",Dict{String,Any}())
        buffers = Dict(b.id=>b for b in (buf("x",(n,k);kind=:external),buf("w",(m,k);kind=:weight),buf("y",(n,m)),buf("z",(n,m))))
        ops = [DNNKernels.Op("y","mm.default",["x","w"],"y",Dict{String,Any}()),
               DNNKernels.Op("z","clamp.default",["y"],"z",Dict{String,Any}("arg1"=>0,"arg2"=>1000))]
        graph = DNNKernels.Graph("q8short",String[],["x"],["z"],buffers,collect(keys(buffers)),ops)
        plan = DNNKernels.planfor(dev,graph,Dict{String,Any}("w"=>a),(;))
        try
            for _ in 1:2
                bh = Float16.(randn(rng,Float32,k,n))
                b = Mantle.Buffer(dev,bh)
                got = Float32.(Array(first(DNNKernels.replay!(plan,"q8short",(b,)))))
                @test size(got) == (m,n)
                @test isapprox(got,max.(w*Float32.(bh),0f0);rtol=0.002,atol=0.002)
                Mantle.free!(b)
            end
        finally
            Mantle.free!(plan)
        end
    end
    foreach(Mantle.free!,(a.q,a.scale))
end
