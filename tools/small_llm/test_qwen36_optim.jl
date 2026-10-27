using Test,Random,JSON3,SHA

function test_qwen36_optim(device)
    R=Qwen36MoEExperiment;G=Qwen36GGUF;Q=R.Q;M=R.M;DK=R.DK
    owned=Any[];plans=Any[];rng=MersenneTwister(3601)
    upload(x)=begin b=Mantle.Buffer(device,x);push!(owned,b);b;end
    record(graph)=begin p=Mantle.record!(Mantle.Plan(graph;budget=Inf));push!(plans,p);p;end
    run(graph)=begin p=record(graph);Mantle.run!(p);Mantle.waitidle(device);p;end
    clean()=begin foreach(Mantle.free!,reverse(plans));empty!(plans);foreach(Mantle.free!,reverse(owned));empty!(owned);end
    @testset "Shared producer extent rejection" begin
        graph=Mantle.Graph(device)
        @test_throws DimensionMismatch R.prepare_q8!(graph,zeros(Float32,255),256,1;name="short")
        @test_throws DimensionMismatch R.prepare_q8!(graph,zeros(Float32,256),256,1;status=Int32[],name="missing_status")
        @test_throws DimensionMismatch R.prepare_q8!(graph,zeros(Float32,768),256,3;status=Int32[0],topk=2,per_expert=true,name="partial_expert")
    end
    @testset "Parallel Q8 exact bytes, FP32 scales and invalid-token masks" begin
        for k in (256,512,2048),checked in (false,true),per in (false,true)
            try
                topk=2;tokens=3;columns=per ? topk*tokens : tokens
                data=randn(rng,Float32,k,columns);data[1:32,:].=0f0;data[33:64,:].*=1f20;data[65:96,:].*=1f-20
                if checked
                    data[:,per ? (3:4) : (2:2)].=NaN32
                end
                x=upload(data);status=checked ? upload(Int32[0,1,0]) : nothing
                graph=Mantle.Graph(device)
                old=R.prepare_q8!(graph,x,k,columns;status,topk,per_expert=per,name="original_q8")
                new=R.prepare_q8!(graph,x,k,columns;status,topk,per_expert=per,name="parallel_q8",parallel=true)
                captured=Any[]
                for (name,src,T) in (("oldq",old.q,DK.UVec4),("newq",new.q,DK.UVec4),("oldds",old.ds,Float32),("newds",new.ds,Float32))
                    dst=Mantle.Buffer(device,T,(length(src),));push!(owned,dst);push!(captured,dst)
                    Q.dispatch(graph,R.BR.copy_kernel!,(dst,src,Int32(length(src))),length(src);name)
                end
                run(graph)
                @test reinterpret(UInt32,Array(captured[1]))==reinterpret(UInt32,Array(captured[2]))
                @test reinterpret(UInt32,Array(captured[3]))==reinterpret(UInt32,Array(captured[4]))
            finally
                clean()
            end
        end
    end
    @testset "Small FP32 projections and output coverage" begin
        for k in (256,2048),m in (1,17,32,256,257)
            try
                W=randn(rng,Float32,k,m);x=randn(rng,Float32,k)
                weight=Q.DenseWeight(upload(W),m,k);xb=upload(x);out=upload(fill(NaN32,m));original=upload(fill(NaN32,m));tuned=upload(fill(NaN32,m))
                graph=Mantle.Graph(device)
                R.linear_opt!(graph,out,weight,xb;name="small_f32",optimizations=(:small_dense,))
                Q.linear!(graph,original,weight,xb;name="original_f32")
                R.linear_opt!(graph,tuned,weight,xb;name="small_tuned",optimizations=(:small_dense,:router_wg64,:dense_unroll32))
                p=record(graph);Mantle.waitidle(device)
                before=Mantle.submissions(device);Mantle.run!(p);Mantle.waitidle(device)
                @test Mantle.submissions(device)-before==1
                @test all(isfinite,Array(out))
                @test Array(out)≈transpose(W)*x atol=5f-4 rtol=3f-5
                @test reinterpret(UInt32,Array(out))==reinterpret(UInt32,Array(original))
                @test reinterpret(UInt32,Array(tuned))==reinterpret(UInt32,Array(original))
            finally
                clean()
            end
        end
    end
    @testset "Bit-exact fused residual/next-input normalization" begin
        for n in (128,256,2048,4096),add in (false,true)
            try
                x=randn(rng,Float32,n);branch=randn(rng,Float32,n);weights=randn(rng,Float32,n)
                xb=upload(x);oldx=upload(x);bb=upload(branch);wb=upload(weights)
                out=upload(fill(NaN32,n));old=upload(fill(NaN32,n));graph=Mantle.Graph(device)
                add&&Q.dispatch(graph,Q.add!,(oldx,bb,Int32(n)),n;name="separate_add")
                Q.dispatch(graph,R.BR.rmsnorm_batch_kernel!,(old,oldx,wb,Int32(n),1f-6),256;name="separate_norm")
                Q.dispatch(graph,R.boundary_norm!,(out,xb,bb,wb,Int32(n),1f-6,Val(add),Val(Mantle.caps(device).subgroup)),256;name="fused_boundary")
                q=upload(fill(ntuple(_->VecElement(UInt32(0)),4),n÷16));ds=upload(zeros(Float32,n÷16))
                oldq=upload(fill(ntuple(_->VecElement(UInt32(0)),4),n÷16));oldds=upload(zeros(Float32,n÷16))
                # Separate input copy prevents applying the residual twice.
                qx=upload(x);qout=upload(fill(NaN32,n))
                Q.dispatch(graph,R.boundary_norm_q8!,(qout,qx,bb,wb,q,ds,Int32(n),1f-6,Val(add),Val(Mantle.caps(device).subgroup),Val(n)),256;name="boundary_q8")
                Q.dispatch(graph,DK.q8act_kernel!,(oldq,oldds,old,Int32(n),Int32(n÷32)),n÷32;name="separate_q8")
                run(graph)
                @test Array(xb)==Array(oldx)
                @test all(isfinite,Array(out))
                @test reinterpret(UInt32,Array(out))==reinterpret(UInt32,Array(old))
                @test reinterpret(UInt32,Array(q))==reinterpret(UInt32,Array(oldq))
                @test reinterpret(UInt32,Array(ds))==reinterpret(UInt32,Array(oldds))
            finally
                clean()
            end
        end
    end
    @testset "Narrow Q4/Q5 experts, masking, and reused inputs" begin
        refs=DK.readsafetensors(joinpath(@__DIR__,"gguf_q5_reference/reference.safetensors"))
        for kind in (:q4,:q5),k in (256,512,1024),per in (false,true)
            try
                m,ne,topk,tokens=17,4,2,3
                W=if kind==:q4
                    quant=DK.q4k_quantize(randn(rng,Float32,m*ne,k))
                    M.Q4Experts(DK.Q4KMatrix(upload(quant),m*ne,k),m,ne)
                else
                    packed=upload(reinterpret(DK.UVec4,vec(refs["q5.$k.bytes"])))
                    G.Q5Experts(G.Q5KMatrix(packed,m*ne,k),m,ne)
                end
                x=upload(randn(rng,Float32,per ? (k,topk,tokens) : (k,tokens)))
                selected=upload(Int32[0 3 1;2 1 2]);status=upload(Int32[0,0,1])
                old=upload(fill(NaN32,m,topk,tokens));new=upload(fill(NaN32,m,topk,tokens))
                graph=Mantle.Graph(device)
                M.declare_routed_linear!(graph,old,W,x,selected,status;per_expert=per,name="original")
                qx=R.prepare_q8!(graph,x,k,per ? topk*tokens : tokens;status,topk,per_expert=per,name="shared.q8")
                R.routed_opt!(graph,new,W,qx,selected,status;per_expert=per,name="narrow",optimizations=(:narrow_experts,))
                alternatives=Any[]
                for wg in (64,128,512)
                    alt=upload(fill(NaN32,m,topk,tokens));push!(alternatives,alt);lanes=min(32,k÷32)
                    args=(alt,W.packed.q,qx.q,qx.ds,selected,status,Int32(m),Int32(ne),Int32(topk),Int32(length(alt)),Int32(k÷256),Val(per))
                    if kind==:q4
                        Q.dispatch(graph,R.routed_q4_tuned!,(args...,Val(lanes),Val(Mantle.caps(device).subgroup),Val(wg)),cld(length(alt),wg÷lanes)*wg;group=wg,name="tuned.$wg")
                    else
                        Q.dispatch(graph,G.q5_gemv_tuned!,(args...,Val(true),Val(Mantle.caps(device).subgroup),Val(lanes),Val(wg)),cld(length(alt),wg÷lanes)*wg;group=wg,name="tuned.$wg")
                    end
                end
                run(graph)
                @test all(isfinite,Array(new))
                @test Array(new)≈Array(old) atol=2f-5 rtol=2f-5
                @test all(iszero,Array(new)[:,:,3])
                for alt in alternatives
                    @test reinterpret(UInt32,Array(alt))==reinterpret(UInt32,Array(new))
                end
                if !per
                    # Invalid IDs with valid status must remain visibly invalid.
                    copyto!(selected,fill(Int32(ne),topk,tokens));copyto!(status,zeros(Int32,tokens))
                    run(graph)
                    @test all(isnan,Array(new))
                end
            finally
                clean()
            end
        end
    end
    @testset "Shared Q8 producers distinguish overwritten input versions" begin
        try
            k,m=256,17;W=randn(rng,Float32,m,k);quant=DK.q4k_quantize(W)
            weight=DK.Q4KMatrix(upload(quant),m,k)
            x=upload(randn(rng,Float32,k));replacement=upload(randn(rng,Float32,k))
            a=upload(zeros(Float32,m));b=upload(zeros(Float32,m));c=upload(zeros(Float32,m));d=upload(zeros(Float32,m))
            graph=Mantle.Graph(device)
            first=R.prepare_q8!(graph,x,k,1;name="version1.q8")
            R.linear_opt!(graph,a,weight,first;name="reuse1")
            Q.linear!(graph,b,weight,x;name="original1")
            Q.dispatch(graph,R.BR.copy_kernel!,(x,replacement,Int32(k)),k;name="overwrite_input")
            second=R.prepare_q8!(graph,x,k,1;name="version2.q8")
            R.linear_opt!(graph,c,weight,second;name="reuse2")
            Q.linear!(graph,d,weight,x;name="original2")
            run(graph)
            @test first.q!==second.q
            @test reinterpret(UInt32,Array(a))==reinterpret(UInt32,Array(b))
            @test reinterpret(UInt32,Array(c))==reinterpret(UInt32,Array(d))
            @test Array(a)!=Array(c)
        finally
            clean()
        end
        try
            x=upload(fill(NaN32,256));status=upload(Int32[1]);graph=Mantle.Graph(device)
            qx=R.prepare_q8!(graph,x,256,1;status,name="guarded_shared.q8")
            # Copy transient outputs to persistent buffers for inspection.
            q=upload(fill(ntuple(_->VecElement(UInt32(0xffffffff)),4),16));ds=upload(fill(NaN32,16))
            Q.dispatch(graph,R.BR.copy_kernel!,(q,qx.q,Int32(16)),16;name="inspect_q")
            Q.dispatch(graph,R.BR.copy_kernel!,(ds,qx.ds,Int32(16)),16;name="inspect_ds")
            run(graph)
            @test all(iszero,reinterpret(UInt32,vec(Array(q))))
            @test all(iszero,Array(ds))
        finally
            clean()
        end
    end
    @testset "Two-row Q4 down projection, expert boundaries and masks" begin
        for m in (18,2048)
            try
                k,ne,topk,tokens=512,4,2,3
                packed=DK.q4k_quantize(randn(rng,Float32,m*ne,k))
                W=M.Q4Experts(DK.Q4KMatrix(upload(packed),m*ne,k),m,ne)
                x=upload(randn(rng,Float32,k,topk,tokens));selected=upload(Int32[0 3 1;2 1 2]);status=upload(Int32[0,0,1])
                old=upload(fill(NaN32,m,topk,tokens));new=upload(fill(NaN32,m,topk,tokens));graph=Mantle.Graph(device)
                R.routed_opt!(graph,old,W,x,selected,status;per_expert=true,name="one_row",optimizations=(:narrow_experts,:down_wg64))
                R.routed_opt!(graph,new,W,x,selected,status;per_expert=true,name="two_rows",optimizations=(:narrow_experts,:down_wg64,:down_rows2))
                run(graph)
                @test all(isfinite,Array(new))
                @test reinterpret(UInt32,Array(new))==reinterpret(UInt32,Array(old))
                @test all(iszero,Array(new)[:,:,3])
                copyto!(selected,fill(Int32(ne),topk,tokens));copyto!(status,zeros(Int32,tokens));run(graph)
                @test all(isnan,Array(new))
            finally
                clean()
            end
        end
    end
    @testset "Quantized full tiny model, shared producer and fusion parity" begin
        model=load_qwen36_moe_fixture(R,joinpath(@__DIR__,"qwen36_moe_tiny_reference"),device)
        baseline=nothing;optimized=nothing
        try
            for (name,weight) in collect(model.weights)
                if weight isa Q.DenseWeight&&weight.m>=128&&!endswith(name,"mlp.gate.weight")
                    dense=Mantle.Buffer(device,permutedims(Array(weight.data)))
                    try
                        packed=name=="lm_head" ? DK.quantizeq6k(device,dense) : DK.quantizeq4k(device,dense)
                        append!(model.owned,DK.kqarrays(packed));model.weights[name]=packed
                    finally
                        Mantle.free!(dense)
                    end
                elseif occursin(".experts.",name)
                    k,rows,ne=size(weight);dense=Mantle.Buffer(device,permutedims(reshape(Array(weight),k,rows*ne)))
                    try
                        packed=DK.quantizeq4k(device,dense);push!(model.owned,packed.q)
                        model.weights[name]=M.Q4Experts(packed,rows,ne)
                    finally
                        Mantle.free!(dense)
                    end
                end
            end
            R.prepare_shared_gateup!(model)
            baseline=R.session(model;capacity=64,optimizations=())
            optimized=R.session(model;capacity=64)
            @test length(optimized.plan.passes)<length(baseline.plan.passes)
            # Setup kernels are not part of either recorded inference graph.
            Mantle.waitidle(device)
            for token in (3,19,8,42,2,17,89,9)
                before=Mantle.submissions(device)
                R.step!(baseline,token);R.step!(optimized,token);Mantle.waitidle(device)
                @test Mantle.submissions(device)-before==2
                @test Array(optimized.logits)≈Array(baseline.logits) atol=2f-4 rtol=2f-4
                @test only(Array(optimized.best))==only(Array(baseline.best))
            end
            input=upload(fill(NaN32,256));out=upload(zeros(Float32,256))
            selected=upload(zeros(Int32,2,1));routing=upload(zeros(Float32,2,1));status=upload(Int32[0])
            graph=Mantle.Graph(device)
            R.declare_ffn!(graph,model,out,input,0,selected,routing,status;
                optimizations=R.DEFAULT_OPTIMIZATIONS)
            run(graph)
            @test only(Array(status))==1
            @test all(isnan,Array(out))
        finally
            clean()
            baseline===nothing||close(baseline);optimized===nothing||close(optimized);close(model)
        end
    end
end
