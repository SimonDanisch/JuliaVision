using Test, Random

function test_qwen36_fusion(device)
    R=Qwen36MoEExperiment;Q=R.Q;M=R.M;DK=R.DK
    supported=Mantle.caps(device).subgroup>=32&&Mantle.caps(device).subgroup%32==0
    @testset "Shared SwiGLU/Q8 fusion" begin
        if !supported
            @test_throws ArgumentError R.prepare_swiglu_q8!(Mantle.Graph(device),zeros(Float32,256),zeros(Float32,256),zeros(Float32,256),256,1;name="unsupported")
        else
            rng=MersenneTwister(36072)
            for k in (256,512,1024),packed in (false,true)
                owned=Any[];plan=nothing
                upload(x)=begin b=Mantle.Buffer(device,x);push!(owned,b);b;end
                try
                    columns=packed ? 4 : 1
                    gate=randn(rng,Float32,packed ? 2k*columns : k*columns)
                    gate[1:8].=Float32[-100,100,0,-0.0,1f-20,-1f-20,12,-12]
                    up=randn(rng,Float32,k*columns)
                    packed&&(gate[4k+1:end].=NaN32)
                    gb=upload(gate);ub=upload(up);status=packed ? upload(Int32[0,1]) : nothing
                    old=upload(fill(NaN32,k*columns));new=upload(fill(NaN32,k*columns))
                    g=Mantle.Graph(device)
                    if packed
                        Q.dispatch(g,M.swiglu_packed!,(old,gb,Int32(k),Int32(k*columns)),k*columns;name="separate_swiglu")
                    else
                        Q.dispatch(g,Q.swiglu!,(old,gb,ub,Int32(k)),k;name="separate_swiglu")
                    end
                    qold=R.prepare_q8!(g,old,k,columns;status,topk=2,per_expert=packed,name="separate_q8",parallel=true)
                    qnew=R.prepare_swiglu_q8!(g,new,gb,ub,k,columns;packed,status,topk=2,per_expert=packed,name="fused")
                    captured=Any[]
                    for src in (qold.q,qnew.q,qold.ds,qnew.ds)
                        dst=Mantle.Buffer(device,eltype(src),(length(src),));push!(owned,dst)
                        Q.dispatch(g,R.BR.copy_kernel!,(dst,src,Int32(length(src))),length(src);name="capture")
                        push!(captured,dst)
                    end
                    plan=Mantle.record!(Mantle.Plan(g;budget=Inf));Mantle.waitidle(device)
                    before=Mantle.submissions(device);Mantle.run!(plan);Mantle.waitidle(device)
                    @test Mantle.submissions(device)-before==1
                    valid=packed ? 2k : k
                    @test reinterpret(UInt32,Array(new)[1:valid])==reinterpret(UInt32,Array(old)[1:valid])
                    @test reinterpret(UInt32,Array(captured[1]))==reinterpret(UInt32,Array(captured[2]))
                    @test reinterpret(UInt32,Array(captured[3]))==reinterpret(UInt32,Array(captured[4]))
                    packed&&(@test all(iszero,Array(new)[valid+1:end]))
                finally
                    plan===nothing||Mantle.free!(plan)
                    foreach(Mantle.free!,reverse(owned))
                end
            end
        end
    end
    @testset "Capability-selected dense rows preserve reduction order" begin
        owned=Any[];plans=Any[];rng=MersenneTwister(36073)
        upload(x)=begin b=Mantle.Buffer(device,x);push!(owned,b);b;end
        try
            for k in (256,2048),m in (1,17,257)
                w=upload(randn(rng,Float32,k,m));x=upload(randn(rng,Float32,k))
                a=upload(fill(NaN32,m));b=upload(fill(NaN32,m));g=Mantle.Graph(device)
                weight=Q.DenseWeight(w,m,k)
                base=(:small_dense,:dense_unroll32)
                R.linear_opt!(g,a,weight,x;name="original64",optimizations=base)
                R.linear_opt!(g,b,weight,x;name="capability",optimizations=(base...,:capability_dense))
                p=Mantle.record!(Mantle.Plan(g;budget=Inf));push!(plans,p)
                Mantle.run!(p);Mantle.waitidle(device)
                @test all(isfinite,Array(b))
                @test reinterpret(UInt32,Array(a))==reinterpret(UInt32,Array(b))
            end
        finally
            foreach(Mantle.free!,reverse(plans));foreach(Mantle.free!,reverse(owned))
        end
    end
    @testset "Paired expert rows, tails, masks and invalid IDs" begin
        if Mantle.caps(device).subgroup<32
            @test_skip false
        else
            rng=MersenneTwister(36074)
            for k in (1024,2048),m in (34,128)
                owned=Any[];plan=nothing
                upload(x)=begin b=Mantle.Buffer(device,x);push!(owned,b);b;end
                try
                    ne=4;topk=2;tokens=3
                    packed=DK.q4k_quantize(randn(rng,Float32,m*ne,k))
                    weight=M.Q4Experts(DK.Q4KMatrix(upload(packed),m*ne,k),m,ne)
                    input=randn(rng,Float32,k,tokens);input[:,3].=NaN32
                    x=upload(input);selected=upload(Int32[0 3 1;2 1 2]);status=upload(Int32[0,0,1])
                    a=upload(fill(NaN32,m,topk,tokens));b=upload(fill(NaN32,m,topk,tokens))
                    g=Mantle.Graph(device)
                    qx=R.prepare_q8!(g,x,k,tokens;status,topk,name="input",parallel=true)
                    base=(:narrow_experts,)
                    R.routed_opt!(g,a,weight,qx,selected,status;name="original",optimizations=base)
                    R.routed_opt!(g,b,weight,qx,selected,status;name="paired",optimizations=(base...,:paired_expert_rows))
                    plan=Mantle.record!(Mantle.Plan(g;budget=Inf));Mantle.run!(plan);Mantle.waitidle(device)
                    @test reinterpret(UInt32,vec(Array(a)))==reinterpret(UInt32,vec(Array(b)))
                    @test all(iszero,Array(b)[:,:,3])
                    copyto!(selected,fill(Int32(ne),topk,tokens));copyto!(status,zeros(Int32,tokens));copyto!(x,zeros(Float32,k,tokens))
                    Mantle.run!(plan);Mantle.waitidle(device)
                    @test all(isnan,Array(b))
                finally
                    plan===nothing||Mantle.free!(plan);foreach(Mantle.free!,reverse(owned))
                end
            end
        end
    end
end
