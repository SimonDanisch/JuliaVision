# A/B only the shared Mantle recorded-buffer entry optimization. No global
# method overrides: each owned plan is recompiled using the normal refit path.
using Mantle, JSON3, Statistics, SHA

function sync_ablation!(pl::Mantle.Plan, automatic::Bool)
    dev=pl.graph.dev
    Mantle.waitidle(dev)
    Mantle.invalidate!(pl)
    am=pl.args
    am===nothing || Mantle.retire!(Mantle.pool(dev),dev,am.store)
    c=Mantle.Compile(pl.graph;alias=pl.alias,policy=pl.policy,
        ordered_entry=automatic && Mantle.buffer_head_ordered(dev) &&
            Mantle.recordsplans(dev) && Mantle.recordable(dev,pl))
    Mantle.install!(pl,Mantle.compile!(c))
    Mantle.record!(pl)
    pl
end

function benchmark_horizon_sync(h;output,tokens=32,repeats=6)
    ispath(output) && error("Refusing to overwrite $output")
    DK=DNNKernels; HR=HorizonRunner
    be=h.backend; dev=h.model.device
    k,v=HR.newcache(h)
    tokh=zeros(Int64,1,1);posh=zeros(Int64,1)
    tokd=DK.toback(be,tokh);posd=DK.toback(be,posh)
    kv=HR.attentionbucket(h,tokens+8)
    maskh=HR.maskbuffer(h,1;kv);maskd=DK.toback(be,maskh)
    plans()=begin
        out=Any[]
        function collectplans(m)
            for x in values(m.scratch)
                x isa DK.RecordedPlan && push!(out,x.plan)
                x isa DK.Model && collectplans(x)
            end
        end
        collectplans(h.model);out
    end
    function forward(i)
        tokh[1]=mod(100+i,1000);posh[1]=i-1
        HR.causalmask!(maskh,posh)
        copyto!(tokd,tokh);copyto!(posd,posh);copyto!(maskd,maskh)
        first(HR.bucketcall(h,"horizon32b_decode_bucket",tokd,k,v,posd,maskd;kv))
    end
    rows=Any[];reference=nothing
    try
        forward(1);Mantle.waitidle(dev)
        @assert !isempty(plans())
        for trial in 0:repeats
            for automatic in (iseven(trial) ? (false,true) : (true,false))
                foreach(p->sync_ablation!(p,automatic),plans())
                fill!(k,0);fill!(v,0)
                for i in 1:8;forward(i);end
                Mantle.waitidle(dev)
                ids=Int[];final=nothing
                before=Mantle.submissions(dev)
                elapsed=@elapsed begin
                    for i in 9:tokens+8
                        final=Array(forward(i))
                        push!(ids,argmax(vec(final))-1)
                    end
                    Mantle.waitidle(dev)
                end
                submissions=Mantle.submissions(dev)-before
                @assert all(isfinite,final)
                bits=collect(reinterpret(UInt16,vec(final)))
                if reference===nothing
                    reference=(ids,bits)
                else
                    @assert ids==reference[1] && bits==reference[2]
                end
                trial==0 && continue
                ps=plans()
                push!(rows,(;trial,automatic,elapsed,tokens_per_second=tokens/elapsed,
                    submissions,passes=sum(length(p.passes) for p in ps),
                    barrier_passes=sum(count(x->!isempty(x.pre),p.passes) for p in ps)))
                println(last(rows));flush(stdout)
                open(io->JSON3.pretty(io,(;complete=false,rows)),output,"w")
            end
        end
        old=median([r.tokens_per_second for r in rows if !r.automatic])
        new=median([r.tokens_per_second for r in rows if r.automatic])
        result=(;name="Horizon32B Q4_K_M",tokens,prefix_tokens=8,complete=true,
            old_tps=old,new_tps=new,percent=100*(new/old-1),
            tokens_and_logits_bitidentical=true,rows,
            scope="Shared Mantle synchronization only; fixed teacher-forced sequence; original bucketcall and per-token host logit readback")
        open(io->JSON3.pretty(io,result),output,"w")
        result
    finally
        DK.releasedevice!(k);DK.releasedevice!(v)
        DK.releasedevice!(tokd);DK.releasedevice!(posd);DK.releasedevice!(maskd)
    end
end

function benchmark_bonsai_sync(model;output,tokens=32,repeats=6)
    ispath(output) && error("Refusing to overwrite $output")
    s=BonsaiRunner.session(model;kv_page=256)
    rows=Any[]; reference=nothing
    prompt=[27175,667,17839,958,3145,63368,5638,13]
    try
        # Warm all kernels before alternating conditions. Keep host feedback
        # exactly as this runner implements it; do not compare to Qwen's loop.
        for trial in 0:repeats
            for automatic in (iseven(trial) ? (false,true) : (true,false))
                sync_ablation!(s.plan,automatic)
                for (hist,state) in values(s.recurrent)
                    fill!(hist,0f0);fill!(state,0f0)
                end
                s.position=0
                for token in prompt; BonsaiRunner.step!(s,token); end
                Mantle.waitidle(s.plan.graph.dev)
                ids=Int[]
                before=Mantle.submissions(s.plan.graph.dev)
                elapsed=@elapsed begin
                    for _ in 1:tokens
                        token=BonsaiRunner._greedy(s,s.logits)
                        push!(ids,token)
                        BonsaiRunner.step!(s,token)
                    end
                    Mantle.waitidle(s.plan.graph.dev)
                end
                submissions=Mantle.submissions(s.plan.graph.dev)-before
                logits=Array(s.logits)
                @assert all(isfinite,logits)
                if reference===nothing
                    reference=(ids,copy(reinterpret(UInt32,logits)))
                else
                    @assert ids==reference[1]
                    @assert reinterpret(UInt32,logits)==reference[2]
                end
                trial==0 && continue
                push!(rows,(;trial,automatic,elapsed,tokens_per_second=tokens/elapsed,
                    submissions,passes=length(s.plan.passes),
                    barrier_passes=count(p->!isempty(p.pre),s.plan.passes)))
                println(last(rows));flush(stdout)
                open(io->JSON3.pretty(io,(;complete=false,rows)),output,"w")
            end
        end
        old=median([r.tokens_per_second for r in rows if !r.automatic])
        new=median([r.tokens_per_second for r in rows if r.automatic])
        result=(;name="Bonsai2 27B PTQ1",tokens,prompt,complete=true,old_tps=old,
            new_tps=new,percent=100*(new/old-1),tokens_and_logits_bitidentical=true,rows,
            scope="Shared Mantle synchronization only; original Bonsai kernels and host greedy feedback")
        open(io->JSON3.pretty(io,result),output,"w")
        result
    finally
        close(s)
    end
end

function benchmark_qwen35_sync(model; name, output, tokens=64, repeats=6)
    ispath(output) && error("Refusing to overwrite $output")
    s=Qwen35Experiment.session(model;capacity=256)
    rows=Any[]; reference=nothing
    prompt=[27175,667,17839,958,3145,63368,5638,13]
    try
        for trial in 0:repeats
            for automatic in (iseven(trial) ? (false,true) : (true,false))
                sync_ablation!(s.plan,automatic)
                reset_qwen35!(s)
                for token in prompt; Qwen35Experiment.step!(s,token); end
                Mantle.waitidle(model.device)
                before=Mantle.submissions(model.device)
                elapsed=@elapsed begin
                    Qwen35Experiment.decode!(s,tokens)
                    Mantle.waitidle(model.device)
                end
                submissions=Mantle.submissions(model.device)-before
                ids=Array(s.predictions)[length(prompt):length(prompt)+tokens-1]
                logits=Array(s.logits)
                @assert all(isfinite,logits) && submissions==tokens
                if reference===nothing
                    reference=(ids,copy(reinterpret(UInt32,logits)))
                else
                    @assert ids==reference[1]
                    @assert reinterpret(UInt32,logits)==reference[2]
                end
                trial==0 && continue
                push!(rows,(;trial,automatic,elapsed,tokens_per_second=tokens/elapsed,
                    submissions,passes=length(s.plan.passes),
                    barrier_passes=count(p->!isempty(p.pre),s.plan.passes)))
                println(last(rows));flush(stdout)
                open(io->JSON3.pretty(io,(;name,tokens,prompt,complete=false,rows)),output,"w")
            end
        end
        old=median([r.tokens_per_second for r in rows if !r.automatic])
        new=median([r.tokens_per_second for r in rows if r.automatic])
        result=(;name,tokens,prompt,complete=true,old_tps=old,new_tps=new,
            percent=100*(new/old-1),tokens_and_logits_bitidentical=true,rows,
            scope="Shared Mantle synchronization only; unchanged Qwen3.5 Q4_K kernels; device-resident feedback",
            model_source_sha256=bytes2hex(sha256(read(joinpath(@__DIR__,"qwen35.jl")))))
        open(io->JSON3.pretty(io,result),output,"w")
        result
    finally
        close(s)
    end
end
