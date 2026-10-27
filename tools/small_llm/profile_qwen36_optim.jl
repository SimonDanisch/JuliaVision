using JSON3,SHA

function profile_qwen36_optim(checkpoint,config,output;optimizations)
    ispath(output)&&error("Preserve prior profile")
    G=Qwen36GGUF;R=G.R
    first(split(read(`sha256sum $checkpoint`,String)))==G.CHECKPOINT_SHA256||error("Checkpoint hash")
    report=Dict{String,Any}("complete"=>false,"checkpoint_sha256"=>G.CHECKPOINT_SHA256,
        "optimizations"=>optimizations,"scope"=>"Instrumented per-pass GPU costs; not an end-to-end speed measurement",
        "source_sha256"=>Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in
            ("qwen36_moe_model.jl","qwen36_optim.jl","qwen36_gguf_kernels.jl","profile_qwen36_optim.jl")))
    save()=open(io->JSON3.pretty(io,report),output,"w")
    let
        d=Mantle.Device("radeon");m=nothing;s=nothing
        try
            m=G.loadmodel(checkpoint,config,d;verify_hash=false);R.prepare_shared_gateup!(m)
            s=R.session(m;capacity=256,head_layout=:tiled,kv_type=Float32,optimizations,profile=true)
            for token in (27175,667,17839,958,3145,63368,5638,13);R.step!(s,token);end
            for _ in 1:6;R.decode!(s,1);Mantle.waitidle(d);end
            passes=[(;name=t.name,gpu_ms=t.gpu_ms,samples=t.samples) for t in Mantle.timings(s.plan)]
            report["passes"]=passes;report["pass_count"]=length(passes)
            report["graph"]=(;peak=s.plan.peak,naive=s.plan.naive,
                barrier_passes=count(p->p.barrier!==nothing,s.plan.passes),
                alias_handovers=sum(count(t->t.resource==0,p.pre) for p in s.plan.passes))
            report["first_layer_usages"]=[(;name=p.pass.name,
                usages=[(;resource=Int(u.first),usage=string(u.second)) for u in p.pass.usages],
                transitions=[(;resource=t.resource,from=string(t.from),to=string(t.to)) for t in p.pre])
                for p in s.plan.passes if startswith(p.pass.name,"layers.0.")]
            report["top_passes"]=sort(passes;by=t->t.gpu_ms,rev=true)[1:min(30,length(passes))]
            report["complete"]=true
        catch err
            report["error"]=sprint(showerror,err);rethrow()
        finally
            s===nothing||close(s);m===nothing||close(m)
            report["normal_final_waitidle"]=false
            try
                finish_device!(d);report["normal_final_waitidle"]=true
            finally
                save()
            end
        end
    end
    report
end
