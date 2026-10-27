using Test,Random,LinearAlgebra
import Mantle,KernelAbstractions,DNNKernels

function reference_q8_activations(x,status,topk,per_expert)
    k=size(x,1);columns=length(x)÷k
    a=reshape(x,k,columns);out=zeros(Float32,k,columns)
    for col in 1:columns
        token=per_expert ? (col-1)÷topk+1 : col
        status[token]!=0 && continue
        for b in 0:k÷32-1
            ids=32b+1:32b+32;values=a[ids,col]
            d=maximum(abs,values)/127f0
            invd=d==0f0 ? 0f0 : inv(d)
            out[ids,col].=Float32.(round.(Int32,values.*invd)).*d
        end
    end
    out
end

function test_qwen35_quant_experts(device)
    M=Qwen35MoE;DK=DNNKernels;rng=MersenneTwister(3531);rows=Any[]
    @testset "Selected Q4_K experts with shared Q8 activation arithmetic" begin
        for (k,m,ne,topk) in ((256,17,8,2),(512,9,256,8)), per_expert in (false,true)
            owned=Any[];plan=nothing
            upload(x)=begin a=Mantle.Buffer(device,x);push!(owned,a);a;end
            try
                tokens=3
                W=0.1f0*randn(rng,Float32,m*ne,k)
                packed=DK.q4k_quantize(W)
                decoded=DK.q4k_dequantize(packed,m*ne,k)
                weight=M.Q4Experts(DK.Q4KMatrix(upload(packed),m*ne,k),m,ne)
                shape=per_expert ? (k,topk,tokens) : (k,tokens)
                x=randn(rng,Float32,shape)
                per_expert ? (x[:,:,3].=NaN32) : (x[:,3].=NaN32)
                codes=Int32[0,0,1]
                selected=hcat([Int32.(randperm(rng,ne)[1:topk].-1) for _ in 1:tokens]...)
                xbuf=upload(x);ids=upload(selected);status=upload(codes)
                out=upload(fill(NaN32,m,topk,tokens))
                g=Mantle.Graph(device)
                M.declare_routed_linear!(g,out,weight,xbuf,ids,status;per_expert)
                plan=Mantle.record!(Mantle.Plan(g;budget=Inf))
                rounded=reference_q8_activations(x,codes,topk,per_expert)
                for replay in 1:2
                    before=Mantle.submissions(device)
                    Mantle.run!(plan);KernelAbstractions.synchronize(device)
                    @test Mantle.submissions(device)-before==1
                    want=zeros(Float64,m,topk,tokens)
                    for token in 1:2, rank in 1:topk
                        expert=Int(selected[rank,token]);wr=expert*m+1:expert*m+m
                        col=per_expert ? (token-1)*topk+rank : token
                        want[:,rank,token]=Float64.(decoded[wr,:])*Float64.(rounded[:,col])
                    end
                    got=Array(out)
                    @test all(isfinite,got)
                    @test got ≈ want atol=5e-5 rtol=5e-5
                    @test all(iszero,got[:,:,3])
                    push!(rows,(;k,m,experts=ne,topk,per_expert,replay,maxabs=maximum(abs,got-want)))
                    selected.=ne.-1 .-selected;ids[:]=vec(selected)
                end
                selected[1,1]=Int32(-1);ids[:]=vec(selected)
                Mantle.run!(plan)
                @test all(isnan,Array(out)[:,1,1])
            finally
                plan===nothing || Mantle.free!(plan)
                foreach(Mantle.free!,owned)
            end
        end
    end
    test_qwen35_quant_block(device)
    rows
end

function test_qwen35_quant_block(device)
    @testset "Complete recorded Q4 expert block" begin
        M=Qwen35MoE;DK=DNNKernels;rng=MersenneTwister(3533)
        k=f=256;ne=8;topk=2;tokens=2;sf=32
        owned=Any[];plan=nothing
        upload(x)=begin a=Mantle.Buffer(device,x);push!(owned,a);a;end
        function packed(rows,reduction)
            bytes=DK.q4k_quantize(0.02f0*randn(rng,Float32,rows*ne,reduction))
            decoded=DK.q4k_dequantize(bytes,rows*ne,reduction)
            M.Q4Experts(DK.Q4KMatrix(upload(bytes),rows*ne,reduction),rows,ne),decoded
        end
        try
            gu,gu_ref=packed(2f,k);down,down_ref=packed(k,f)
            wh=(;router=0.02f0*randn(rng,Float32,k,ne),
                shared_gate_up=0.02f0*randn(rng,Float32,k,2sf),
                shared_down=0.02f0*randn(rng,Float32,sf,k),shared_gate=0.02f0*randn(rng,Float32,k,1))
            W=(;router=upload(wh.router),gate_up=gu,down,
                shared_gate_up=upload(wh.shared_gate_up),shared_down=upload(wh.shared_down),shared_gate=upload(wh.shared_gate))
            xh=randn(rng,Float32,k,tokens);x=upload(xh);out=upload(fill(NaN32,k,tokens))
            selected=upload(zeros(Int32,topk,tokens));routing=upload(zeros(Float32,topk,tokens))
            status=upload(zeros(Int32,tokens))
            g=Mantle.Graph(device);M.declare_block!(g,out,x,W,selected,routing,status)
            plan=Mantle.record!(Mantle.Plan(g;budget=Inf))
            for replay in 1:2
                logits=wh.router'*xh;probs=exp.(logits.-maximum(logits;dims=1));probs./=sum(probs;dims=1)
                ids=hcat([sortperm(1:ne;by=i->(-probs[i,t],i))[1:topk] for t in 1:tokens]...)
                scores=hcat([probs[ids[:,t],t]./sum(probs[ids[:,t],t]) for t in 1:tokens]...)
                xr=reference_q8_activations(xh,zeros(Int32,tokens),topk,false)
                acts=zeros(Float32,f,topk,tokens)
                for t in 1:tokens, rank in 1:topk
                    e=ids[rank,t]-1
                    a=gu_ref[e*2*f+1:(e+1)*2*f,:]*xr[:,t]
                    acts[:,rank,t]=(a[1:f]./(1f0.+exp.(-a[1:f]))).*a[f+1:2f]
                end
                ar=reference_q8_activations(acts,zeros(Int32,tokens),topk,true)
                want=zeros(Float32,k,tokens)
                for t in 1:tokens, rank in 1:topk
                    e=ids[rank,t]-1
                    want[:,t].+=scores[rank,t].*(down_ref[e*k+1:(e+1)*k,:]*ar[:,(t-1)*topk+rank])
                end
                shared_gu=wh.shared_gate_up'*xh
                shared_act=(shared_gu[1:sf,:]./(1f0.+exp.(-shared_gu[1:sf,:]))).*shared_gu[sf+1:2sf,:]
                want.+=(wh.shared_down'*shared_act)./(1f0.+exp.(-(wh.shared_gate'*xh)))
                before=Mantle.submissions(device)
                Mantle.run!(plan);KernelAbstractions.synchronize(device)
                @test Mantle.submissions(device)-before==1
                @test all(iszero,Array(status))
                @test Array(selected)==ids.-1
                @test Array(out) ≈ want atol=5f-5 rtol=2f-4
                xh=reverse(xh;dims=2);x[:]=vec(xh)
            end
        finally
            plan===nothing || Mantle.free!(plan)
            foreach(Mantle.free!,owned)
        end
    end
end
