# Speculative greedy decoding with drafts from the MTP layer, against plain
# decoding and n-gram drafts, from one saved state per prompt and depth. The
# prompts: the rename task (it copies its prompt, n-gram drafts' best case) and
# code written from scratch (nothing to copy). Each variant runs once to record
# and compile, then `rounds` times in turn; the fastest round counts.
isdefined(@__MODULE__,:spec_prompt) || include("test_speculative.jl")
isdefined(@__MODULE__,:basecorpus) || include("benchmark_long_context.jl")
isdefined(@__MODULE__,:test_mtp) || include("test_mtp.jl")

fresh_prompt(tokenizer)=SmallLLMChecks.QwenText.encode(tokenizer,
    "<|im_start|>user\nWrite a Julia function `primes_upto(n)` that returns all primes up to `n` with a sieve of Eratosthenes, then a short test set for it. Output only the code.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")

function mtp_profile(model,tokenizer;depths=(0,65536),n=192,kv_type=Int8,ks=(1,2,3),rounds=2,deep=16384,
                     prompts=(rename=spec_prompt(tokenizer),fresh=fresh_prompt(tokenizer)))
    Q36.hasmtp(model) || Q36.loadmtp!(model,QWEN36_MTP)
    rows=[]
    for depth in depths,(label,prompt) in pairs(prompts)
        ids=[depth>0 ? basecorpus(tokenizer,depth) : Int[];prompt]
        s=Q36.session(model;capacity=length(ids)+n+64,attention=:flash,kv_type)
        sp=nothing;ng=nothing;rw=nothing
        try
            sp=Q36.MTPSpeculator(s;kmax=maximum(ks),deep);ng=Q36.Speculator(s;kmax=8,deep);rw=Rewind(sp)
            Q36.prefill!(sp,ids;chunk=512);Mantle.waitidle(model.device)
            P=s.inner.position;first=Int(only(Array(s.inner.best)));save!(rw)
            foreach(k->Q36.prepare!(sp;k),ks);Q36.prepare!(ng)
            reset!()=(restore!(rw);setposition!(s,P);Mantle.waitidle(model.device))
            function plain()
                Q36.step!(s,first);generate!(s,n-2);Mantle.waitidle(model.device)
                Int.(Array(s.inner.predictions)[P+1:P+n-1])
            end
            variants=Pair{Symbol,Any}[:plain=>(plain,nothing),:ngram=>(()->Q36.speculative_decode!(ng,ids,n;first)[2:end],ng)]
            for k in ks
                push!(variants,Symbol("mtp_k$k")=>(()->Q36.speculative_decode!(sp,ids,n;first,k)[2:end],sp))
            end
            best=Dict(k=>Inf for (k,_) in variants);outputs=Dict{Symbol,Any}();stats=Dict{Symbol,Any}()
            for (k,(f,_)) in variants;reset!();outputs[k]=f();end
            for r in 1:rounds,(k,(f,spec)) in circshift(variants,r)
                reset!()
                before=spec===nothing ? nothing : (spec.steps,spec.drafted,spec.accepted)
                t=@elapsed f();best[k]=min(best[k],t)
                spec===nothing || (stats[k]=(steps=spec.steps-before[1],drafted=spec.drafted-before[2],accepted=spec.accepted-before[3]))
            end
            ref=outputs[:plain]
            for (k,_) in variants
                o=outputs[k];same=something(findfirst(i->o[i]!=ref[i],1:min(length(o),length(ref))),min(length(o),length(ref))+1)-1
                st=get(stats,k,(steps=n-1,drafted=0,accepted=0))
                row=(;depth,prompt=label,variant=k,tok_s=(n-1)/best[k],agree_tokens=same,st...)
                push!(rows,row)
                println(rpad(string(depth),7),rpad(string(label),8),rpad(string(k),9),lpad(round(row.tok_s,digits=1),6)," tok/s, agrees for ",
                    same,", ",st.steps," rounds, ",st.accepted,"/",st.drafted," drafts accepted")
            end
        finally
            rw===nothing||close(rw);sp===nothing||close(sp);ng===nothing||close(ng);close(s);GC.gc(true);Mantle.trim_gpu_pool!()
        end
    end
    rows
end

# Each plan of a round alone, from the state after the prompt: a decode step,
# MTP rounds of k drafts (drafts and verify in one plan), the n-gram verify of
# the same width (no drafting), and the replay after a refusal. The minimum of
# `reps` interleaved runs, each with its host wait, so a GPU shared with other
# work still gives each plan's own cost. The state is scratch afterwards.
function mtp_plan_times(model,tokenizer;depth=0,kv_type=Int8,ks=(1,2,3),reps=25,deep=16384,prompt=fresh_prompt(tokenizer))
    Q36.hasmtp(model) || Q36.loadmtp!(model,QWEN36_MTP)
    ids=[depth>0 ? basecorpus(tokenizer,depth) : Int[];prompt]
    s=Q36.session(model;capacity=length(ids)+64,attention=:flash,kv_type);sp=nothing;ng=nothing
    try
        sp=Q36.MTPSpeculator(s;kmax=maximum(ks),deep);ng=Q36.Speculator(s;kmax=maximum(ks),deep)
        Q36.prefill!(sp,ids;chunk=512);Mantle.waitidle(model.device)
        P=s.inner.position;first=Int(only(Array(s.inner.best)));chunk=P>=deep
        plans=Pair{String,Any}["decode"=>()->(s.inner.token[]=Int32(first);s.inner.posref[]=Int32(P);Mantle.run!(s.inner.plan);Array(s.inner.best))]
        for k in ks
            r=Q36.roundplan!(sp,k,1,chunk);v=Q36.verifyplan!(ng,k+1,chunk)
            push!(plans,"mtp_round_k$k"=>()->(sp.buffers.best[1:1]=Int32[first];s.inner.posref[]=Int32(P);Mantle.run!(r.plan);Array(sp.buffers.best)))
            push!(plans,"verify_$(k+1)"=>()->(v.tokens[:]=fill(Int32(first),k+1);s.inner.posref[]=Int32(P);Mantle.run!(v.plan);Array(ng.buffers.best)))
        end
        replay=Q36.replayplan!(sp,0)
        push!(plans,"replay"=>()->(Mantle.run!(replay);Mantle.waitidle(model.device)))
        best=Dict(name=>Inf for (name,_) in plans)
        foreach(((_,f),)->f(),plans)
        for r in 1:reps,(name,f) in circshift(plans,r)
            best[name]=min(best[name],@elapsed f())
        end
        for (name,_) in plans;println(rpad(name,16),round(1e3best[name],digits=2)," ms");end
        Dict(name=>1e3best[name] for (name,_) in plans)
    finally
        sp===nothing||close(sp);ng===nothing||close(ng);close(s);GC.gc(true);Mantle.trim_gpu_pool!()
    end
end
