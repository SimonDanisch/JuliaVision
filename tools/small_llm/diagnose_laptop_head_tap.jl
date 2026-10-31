include("laptop_native_datastore.jl")

function diagnose_laptop_head_tap(;seconds=210)
    seconds>0||throw(ArgumentError("time budget"))
    directory=joinpath(@__DIR__,"laptop_head_diagnostic")
    ispath(directory)&&!isempty(readdir(directory))&&error("preserve prior diagnostic")
    mkpath(directory)
    reference=JSON3.read(read(joinpath(@__DIR__,"conversation_native_reference.json"),String))
    ids=Int.(only(filter(r->r.id=="baseline_short",reference.rows)).julia_ids)
    Q=SmallLLMChecks.Qwen35Experiment
    let
        started=time();device=Mantle.Device("radeon");model=nothing;query=nothing
        report=Dict{String,Any}("complete"=>false,"scope"=>"Read-only head-capture failure diagnosis, not an accepted gate or task-quality experiment",
            "runtime_provenance"=>inference_runtime_provenance(),"source_sha256"=>bytes2hex(open(sha256,@__FILE__)),"observations"=>Any[])
        save()=open(io->JSON3.pretty(io,report),joinpath(directory,"report.json"),"w")
        observations=Any[]
        try
            save();model=load_portable_qwen35(Q,"gen/qwen35-4b-portable",device)
            query=Mantle.Buffer(device,zeros(Float32,Int(model.config.hidden_size)))
            function observe(label)
                session=Q.session(model;capacity=256,kv_type=Float32,attention=:split)
                try
                    Q.prefill!(session,ids;chunk=8);Mantle.waitidle(device);pre=Array(session.logits)
                    before=Mantle.submissions(device);Q.decode!(session,8);Mantle.waitidle(device)
                    post=Array(session.logits);tokens=Int.(Array(session.predictions)[length(ids):length(ids)+7])
                    submissions=Mantle.submissions(device)-before
                    for (suffix,values) in (("pre",pre),("post",post))
                        open(io->write(io,values),joinpath(directory,label*"_"*suffix*".f32"),"w")
                    end
                    item=(;label,tokens,submissions,finite=all(isfinite,pre)&&all(isfinite,post),elapsed_seconds=time()-started)
                    push!(report["observations"],item);save()
                    (;pre,post,item)
                finally close(session) end
            end
            for label in ("direct1","direct2","tapped","restored")
                time()<started+seconds-30||error("diagnostic cleanup margin")
                observation=label=="tapped" ? with_head_input_tap(()->observe(label),model,query) : observe(label)
                push!(observations,observation)
            end
            a=first(observations)
            report["comparisons"]=[(;label=b.item.label,pre_exact=a.pre==b.pre,post_exact=a.post==b.post,
                max_pre_abs=maximum(abs.(a.pre.-b.pre)),max_post_abs=maximum(abs.(a.post.-b.post)),
                tokens_equal=a.item.tokens==b.item.tokens,submissions_equal=a.item.submissions==b.item.submissions) for b in observations[2:end]]
            report["captured_hidden_finite"]=all(isfinite,Array(query))
            save()
        finally
            try query===nothing||Mantle.free!(query);model===nothing||close(model)
            finally finish_device!(device) end
        end
        report["normal_final_waitidle"]=true;report["complete"]=true;save();report
    end
end
