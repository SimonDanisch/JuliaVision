# The chunk-wide causal attention against the Float64 reference, token by token:
# chunks that fill their tiles and chunks that do not (the slid-back last tile),
# chunks at position 0 and deep in a cache, Float32, Float16, q8_0 and q4_0 caches,
# one, two and four tiles a workgroup, two and four subgroups a tile, V staged
# as it is and transposed, and the key split with its merge: one split,
# the device's choice, and counts that leave later splits with no keys at all
# or with none of a row's (the chunk's own keys, for its first tokens). Operands are Float16 on the matrix units, so
# the bound is Float16's, relative to the output's scale.
isdefined(@__MODULE__,:attention_reference) || include("test_flash_decode.jl")
using Test,Random

function prefill_attention_device(device,q,kc,vc,gate,start,nkv,nqh;np,qs=2,vt=false,splits=nothing)
    Q=Qwen36MoEExperiment.Q;n=size(q,2)
    owned=Any[];up(x)=(b=Mantle.Buffer(device,x);push!(owned,b);b);plan=nothing
    try
        cache(c)=c isa Tuple ? (size(c[1],1)==64 ? Q.Q8KV : Q.Q4KV)(up(c[1]),up(c[2])) : up(c)
        bk,bv=cache(kc),cache(vc)
        bq,bg=up(q),up(gate);out=up(zeros(Float32,size(q)))
        pos=Mantle.GPURef(device,Int32(start));push!(owned,pos)
        g=Mantle.Graph(device)
        Q.declare_prefill_attention!(g,out,bq,bg,bk,bv,pos,nkv,nqh,n;np,qs,vt,splits,name="prefill_attention")
        plan=Mantle.record!(Mantle.Plan(g;budget=Inf));Mantle.run!(plan);Mantle.waitidle(device)
        Float64.(Array(out))
    finally
        plan===nothing||Mantle.free!(plan);foreach(Mantle.free!,reverse(owned))
    end
end

function prefill_attention_error(device,T,n,start;nkv=2,nqh=16,np=2,qs=2,vt=false,splits=nothing,rng=MersenneTwister(n+start))
    capacity=start+n
    k=randn(rng,Float32,256,nkv,capacity);v=randn(rng,Float32,256,nkv,capacity)
    q=randn(rng,Float32,256nqh,n);gate=rand(rng,Float32,256nqh,n)
    kc,vc,kref,vref=if T===Int8||T===Qwen36MoEExperiment.Q.Q4_0
        rows=T===Int8 ? q8rows : q4rows
        kw,ks,kd=rows(k);vw,vs,vd=rows(v);(kw,ks),(vw,vs),kd,vd
    else
        kx,vx=T.(k),T.(v);kx,vx,kx,vx
    end
    got=prefill_attention_device(device,q,kc,vc,gate,start,nkv,nqh;np,qs,vt,splits)
    want=reduce(hcat,[attention_reference(q[:,t],kref,vref,gate[:,t],start+t-1,nkv,nqh) for t in 1:n])
    maximum(abs.(got.-want))/maximum(abs.(want))
end

function test_prefill_attention(device)
    @testset "chunk-wide causal attention" begin
        for T in (Float32,Float16,Int8,Qwen36MoEExperiment.Q.Q4_0),(n,start) in ((2,0),(3,0),(16,5),(37,100),(64,4000)),(np,qs,vt) in ((2,2,false),(4,2,false),(1,4,false),(4,4,false),(2,2,true),(1,4,true))
            err=prefill_attention_error(device,T,n,start;np,qs,vt,splits=1)
            @test err<2f-3
        end
        for T in (Float16,Int8,Qwen36MoEExperiment.Q.Q4_0),(n,start) in ((2,0),(4,90),(16,5),(37,100),(4,4000)),splits in (nothing,3,7,64),(np,qs,vt) in ((4,4,false),(2,2,false),(2,2,true))
            err=prefill_attention_error(device,T,n,start;splits,np,qs,vt)
            @test err<2f-3
        end
    end
end

# The chunk's prepare kernels against the per-token ones: every token's query,
# gate, and cache row at its own position, for Float32, Float16, q8_0 and q4_0.
function test_prepare_chunk(device;nqh=16,nkv=2,n=5,start=7)
    Q=Qwen36MoEExperiment.Q;rng=MersenneTwister(9);theta=1f7;eps=1f-6
    @testset "chunk prepare kernels" begin
        for T in (Float32,Float16,Int8,Q.Q4_0)
            owned=Any[];up(x)=(b=Mantle.Buffer(device,x);push!(owned,b);b);plan=nothing
            try
                capacity=start+n
                alloc(E,dims)=up(zeros(E,dims))
                cache()=Q.kvcache(alloc,T,nkv,capacity)
                kc1,vc1,kc2,vc2=cache(),cache(),cache(),cache()
                qfull=up(randn(rng,Float32,512nqh,n));k=up(randn(rng,Float32,256nkv,n));v=up(randn(rng,Float32,256nkv,n))
                qn=up(1f0.+0.1f0.*randn(rng,Float32,256));kn=up(1f0.+0.1f0.*randn(rng,Float32,256))
                positions=up(Int32.(start:start+n-1))
                aq1,g1,aq2,g2=(up(fill(NaN32,256nqh,n)) for _ in 1:4)
                g=Mantle.Graph(device)
                Q.declare_prepare_chunk!(g,aq1,g1,qfull,qn,kc1,vc1,k,v,kn,positions,nqh,nkv,theta,eps,n;name="chunk")
                col(a,w,j)=Mantle.viewof(a,(w,);offset=j*w)
                for j in 0:n-1
                    pos=col(positions,1,j)
                    Q.dispatch(g,Q.BR.prepare_q_kernel!,(col(aq2,256nqh,j),col(g2,256nqh,j),col(qfull,512nqh,j),qn,pos,theta,eps),nqh*256;name="q.$j")
                    Q.declare_prepare_kv!(g,kc2,vc2,col(k,256nkv,j),col(v,256nkv,j),kn,pos,nkv,theta,eps;name="kv.$j")
                end
                plan=Mantle.record!(Mantle.Plan(g;budget=Inf));Mantle.run!(plan);Mantle.waitidle(device)
                close(a,b)=maximum(abs.(Float64.(a).-Float64.(b)))<=1f-5*max(1,maximum(abs.(Float64.(b))))
                @test close(Array(aq1),Array(aq2))
                @test close(Array(g1),Array(g2))
                if T===Int8||T===Q.Q4_0
                    # Rounding can step one way or the other at a tie; one step
                    # in a thousand values is the bound.
                    rows=T===Int8 ? q8rows : q4rows
                    for (a,b) in ((kc1,kc2),(vc1,vc2))
                        @test close(Array(a.scales),Array(b.scales))
                        qa=reshape(storedints(rows,Array(a.q)),256,nkv,capacity);qb=reshape(storedints(rows,Array(b.q)),256,nkv,capacity)
                        @test maximum(abs.(qa.-qb))<=1
                        @test count(qa.!=qb)<=length(qa)÷1000
                        @test count(!iszero,Array(a.scales)[:,:,start+1:end])>0
                    end
                else
                    @test close(Array(kc1),Array(kc2))
                    @test close(Array(vc1),Array(vc2))
                    @test maximum(abs.(Float64.(Array(kc1)[:,:,start+1:end])))>0
                end
            finally
                plan===nothing||Mantle.free!(plan);foreach(Mantle.free!,reverse(owned))
            end
        end
    end
end
