# The products a wide prompt chunk takes against the ones the bit-exact chunks
# and decode run: a dense Float32 weight through the tiled GEMM against a
# Float64 product, and a Q5_K weight through the one-expert grouped kernel
# against the single-column kernel, on extents the tiles do not divide.
isdefined(@__MODULE__,:Qwen36GGUF) || include("qwen36_gguf.jl")
using Test,Random,SHA,JSON3
import Mantle

function test_chunk_products(device)
    G=Qwen36GGUF;Q=G.Q;DK=G.DK;rng=MersenneTwister(31)
    owned=Any[];up(x)=(b=Mantle.Buffer(device,x);push!(owned,b);b)
    function run(g)
        plan=Mantle.record!(Mantle.Plan(g;budget=Inf))
        try
            Mantle.run!(plan);Mantle.waitidle(device)
        finally
            Mantle.free!(plan)
        end
    end
    @testset "wide-chunk products" begin
        try
            for (m,k,n) in ((256,2048,256),(1,2048,40),(70,512,17),(130,256,100))
                w=randn(rng,Float32,k,m);x=randn(rng,Float32,k,n)
                A=Q.DenseWeight(up(w),m,k);out=up(fill(NaN32,m,n))
                g=Mantle.Graph(device);Q.linear_batch!(g,out,A,up(x),n;name="dense");run(g)
                want=permutedims(Float64.(w))*Float64.(x)
                @test maximum(abs.(Array(out).-want))<=1e-5*maximum(abs.(want))
            end
            folder=joinpath(@__DIR__,"gguf_q5_reference")
            reference=JSON3.read(read(joinpath(folder,"reference.json"),String))
            @assert bytes2hex(open(sha256,joinpath(folder,"reference.safetensors")))==reference.tensor_sha256
            tensors=DK.readsafetensors(joinpath(folder,"reference.safetensors"))
            for k in (256,512,1024),m in (17,40),n in (17,40)
                rowbytes=(k÷256)*176;src=vec(tensors["q5.$k.bytes"]);nrows=length(src)÷rowbytes
                bytes=reduce(vcat,[src[(r-1)*rowbytes+1:r*rowbytes] for r in rand(rng,1:nrows,m)])
                A=G.Q5KMatrix(up(reinterpret(DK.UVec4,bytes)),m,k)
                x=randn(rng,Float32,k,n);bx=up(x)
                wide=up(fill(NaN32,m,n));columns=[up(fill(NaN32,m)) for _ in 1:n]
                g=Mantle.Graph(device)
                Q.linear_batch!(g,wide,A,bx,n;name="wide")
                for j in 1:n
                    Q.linear!(g,columns[j],A,up(x[:,j]);name="column.$j")
                end
                run(g)
                got=Array(wide);want=reduce(hcat,Array.(columns))
                @test all(isfinite,got)
                @test maximum(abs.(got.-want))<=1f-5*maximum(abs.(want))
            end
        finally
            foreach(Mantle.free!,reverse(owned))
        end
    end
end
