using Test, Random, LinearAlgebra
using .DeltaChunkReference

function test_delta_chunk_reference()
    rng=MersenneTwister(1935)
    @testset "Parallel delta chunk equals recurrent state update" begin
        for n in (1,2,7,32,64), gates in (:normal,:reset,:retain,:no_write)
            q=randn(rng,16,n);q./=sqrt.(sum(abs2,q;dims=1).+1e-6)
            k=randn(rng,16,n);k./=sqrt.(sum(abs2,k;dims=1).+1e-6)
            v=randn(rng,8,n);s=randn(rng,16,8)
            decay=0.8 .+ 0.2.*rand(rng,n);beta=rand(rng,n)
            gates===:reset && (decay[cld(n,2)]=0)
            gates===:retain && fill!(decay,1)
            gates===:no_write && fill!(beta,0)
            reference=DeltaChunkReference.recurrent(q,k,v,decay,beta,s)
            got=DeltaChunkReference.chunk(q,k,v,decay,beta,s)
            @test got.output ≈ reference.output atol=1e-12 rtol=1e-12
            @test got.state ≈ reference.state atol=1e-12 rtol=1e-12
            if n>1
                split=n÷2
                first=DeltaChunkReference.chunk(q[:,1:split],k[:,1:split],v[:,1:split],decay[1:split],beta[1:split],s)
                last=DeltaChunkReference.chunk(q[:,split+1:end],k[:,split+1:end],v[:,split+1:end],decay[split+1:end],beta[split+1:end],first.state)
                @test hcat(first.output,last.output) ≈ got.output atol=1e-12 rtol=1e-12
                @test last.state ≈ got.state atol=1e-12 rtol=1e-12
            end
        end
        q=ones(1,8);v=ones(1,8);s=ones(1,1)
        tiny=fill(1e-200,8);beta=fill(0.5,8)
        @test DeltaChunkReference.chunk(q,q,v,tiny,beta,s).output ≈
              DeltaChunkReference.recurrent(q,q,v,tiny,beta,s).output
        @test_throws ArgumentError DeltaChunkReference.chunk(q,q,v,fill(-0.1,8),beta,s)
    end
    nothing
end
