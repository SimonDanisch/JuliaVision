# Routed experts grouped by expert against the decode kernels' (token, expert)
# pairs: Q4_K and Q5_K rows, gate/up-shaped (one column a token) and
# down-shaped (one column a slot) products, 512 and 2048 wide, with a refused
# token, row counts the tiles do not divide, and experts with no slot and with
# more slots than one pass takes. To rounding: the two shaders may fuse
# multiplies and adds differently.
isdefined(@__MODULE__,:Qwen36GGUF) || include("qwen36_gguf.jl")
using Test,Random,SHA,JSON3
import Mantle

# Each token's experts distinct, the way the router picks them; expert 0 gets
# no slot and expert 1 gets more than sixteen.
function grouped_selection(rng,ne,topk,tokens)
    sel=zeros(Int32,topk,tokens)
    for t in 1:tokens
        rest=shuffle(rng,2:ne-1)
        sel[:,t]=t<=20 ? Int32.([1;rest[1:topk-1]]) : Int32.(rest[1:topk])
    end
    sel
end

function grouped_against_routed(device,W,k,m,ne;per_expert,rng,topk=4,tokens=24)
    R=Qwen36MoEExperiment;Q=R.Q
    owned=Any[];up(x)=(b=Mantle.Buffer(device,x);push!(owned,b);b);plan=nothing
    try
        selected=up(grouped_selection(rng,ne,topk,tokens))
        status=zeros(Int32,tokens);status[5]=1;status=up(status)
        x=up(randn(rng,Float32,per_expert ? (k,topk,tokens) : (k,tokens)))
        want=up(fill(NaN32,m,topk,tokens));got=up(fill(NaN32,m,topk,tokens))
        g=Mantle.Graph(device)
        R.routed_opt!(g,want,W,x,selected,status;per_expert,name="pairs",optimizations=(:narrow_experts,))
        order,offsets=R.declare_expert_buckets!(g,selected,status,ne;name="buckets")
        R.grouped_routed!(g,got,W,x,order,offsets,selected,status;per_expert,name="grouped")
        plan=Mantle.record!(Mantle.Plan(g;budget=Inf));Mantle.run!(plan);Mantle.waitidle(device)
        a=Array(got);b=Array(want);ok=trues(tokens);ok[5]=false
        (maximum(abs.(a[:,:,ok].-b[:,:,ok]))/maximum(abs.(b[:,:,ok])),all(isnan,a[:,:,5]),all(isfinite,a[:,:,ok]))
    finally
        plan===nothing||Mantle.free!(plan);foreach(Mantle.free!,reverse(owned))
    end
end

function test_grouped_experts(device)
    G=Qwen36GGUF;M=G.M;DK=G.DK;rng=MersenneTwister(21)
    folder=joinpath(@__DIR__,"gguf_q5_reference")
    reference=JSON3.read(read(joinpath(folder,"reference.json"),String))
    @assert bytes2hex(open(sha256,joinpath(folder,"reference.safetensors")))==reference.tensor_sha256
    tensors=DK.readsafetensors(joinpath(folder,"reference.safetensors"))
    owned=Any[];up(x)=(b=Mantle.Buffer(device,x);push!(owned,b);b)
    @testset "routed experts grouped by expert" begin
        try
            ne=8
            for k in (512,2048),m in (18,1024),per_expert in (false,true)
                W=M.Q4Experts(DK.Q4KMatrix(up(DK.q4k_quantize(randn(rng,Float32,m*ne,k))),m*ne,k),m,ne)
                err,refused,finite=grouped_against_routed(device,W,k,m,ne;per_expert,rng)
                @test err<1f-5
                @test refused
                @test finite
            end
            # Q5_K rows: the reference's rows, drawn into every expert.
            for k in (512,1024),m in (18,40),per_expert in (false,true)
                rowbytes=(k÷256)*176;src=vec(tensors["q5.$k.bytes"]);nrows=length(src)÷rowbytes
                bytes=reduce(vcat,[src[(r-1)*rowbytes+1:r*rowbytes] for r in rand(rng,1:nrows,m*ne)])
                W=G.Q5Experts(G.Q5KMatrix(up(reinterpret(DK.UVec4,bytes)),m*ne,k),m,ne)
                err,refused,finite=grouped_against_routed(device,W,k,m,ne;per_expert,rng)
                @test err<1f-5
                @test refused
                @test finite
            end
        finally
            foreach(Mantle.free!,reverse(owned))
        end
    end
end
