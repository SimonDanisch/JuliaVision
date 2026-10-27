using JSON3,SHA,Statistics

function tune_qwen36_down(checkpoint,config,output)
    ispath(output)&&error("Preserve tuning receipt")
    G=Qwen36GGUF;R=G.R;Q=R.Q;M=R.M
    first(split(read(`sha256sum $checkpoint`,String)))==G.CHECKPOINT_SHA256||error("Checkpoint")
    report=Dict{String,Any}("complete"=>false,"scope"=>"40 actual expert-down weight sets, fixed inputs; kernel screening, not end-to-end decode", "variants"=>Any[])
    save()=open(io->JSON3.pretty(io,report),output,"w")
    let
        d=Mantle.Device("radeon");model=nothing;owned=Any[];plan=nothing;expected=nothing
        upload(x)=begin b=Mantle.Buffer(d,x);push!(owned,b);b;end
        try
            model=G.loadmodel(checkpoint,config,d;verify_hash=false)
            k=Int(model.config.moe_intermediate_size);m=Int(model.config.hidden_size);topk=Int(model.config.num_experts_per_tok)
            x=upload(reshape(sin.(Float32.(1:k*topk)),k,topk,1));selected=upload(reshape(Int32[0,17,39,70,123,171,212,255],topk,1));status=upload(Int32[0]);out=upload(fill(NaN32,m,topk,1))
            for wg in (256,64,128,512,256)
                graph=Mantle.Graph(d);qx=R.prepare_q8!(graph,x,k,topk;status,topk,per_expert=true,name="q8")
                for il in 0:Int(model.config.num_hidden_layers)-1
                    w=model.weights["layers.$il.mlp.experts.down_proj"];lanes=min(32,k÷32)
                    common=(out,w.packed.q,qx.q,qx.ds,selected,status,Int32(m),Int32(256),Int32(topk),Int32(length(out)),Int32(k÷256),Val(true))
                    if w isa M.Q4Experts
                        Q.dispatch(graph,R.routed_q4_tuned!,(common...,Val(lanes),Val(Mantle.caps(d).subgroup),Val(wg)),cld(length(out),wg÷lanes)*wg;group=wg,name="down.$il")
                    else
                        Q.dispatch(graph,G.q5_gemv_tuned!,(common...,Val(true),Val(Mantle.caps(d).subgroup),Val(lanes),Val(wg)),cld(length(out),wg÷lanes)*wg;group=wg,name="down.$il")
                    end
                end
                plan=Mantle.record!(Mantle.Plan(graph;budget=Inf));Mantle.run!(plan);Mantle.waitidle(d)
                actual=Array(out);expected===nothing&&(expected=actual)
                same=reinterpret(UInt32,actual)==reinterpret(UInt32,expected);same||error("Tuned workgroup changed results")
                samples=Float64[]
                for _ in 1:20
                    t=time_ns();Mantle.run!(plan);Mantle.waitidle(d);push!(samples,(time_ns()-t)/1e6)
                end
                row=(;workgroup=wg,median_ms=median(samples),samples,bitidentical=same)
                push!(report["variants"],row);save();println(row);flush(stdout)
                Mantle.free!(plan);plan=nothing
            end
            report["complete"]=true
        finally
            plan===nothing||Mantle.free!(plan);foreach(Mantle.free!,reverse(owned));model===nothing||close(model)
            finish_device!(d);report["normal_final_waitidle"]=true;save()
        end
    end
    report
end
