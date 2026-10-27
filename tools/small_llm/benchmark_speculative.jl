# Speculative greedy decoding against plain decoding, from one saved state per
# depth: a prompt asking for code back with a rename, alone and after Julia Base
# sources. Plain decoding queues its steps with no host wait; speculation reads
# every verify's argmaxes back. Each variant runs once to record and compile,
# then `rounds` times in turn; the fastest round counts.
isdefined(@__MODULE__,:spec_prompt) || include("test_speculative.jl")
isdefined(@__MODULE__,:basecorpus) || include("benchmark_long_context.jl")

function speculative_profile(model,tokenizer;depths=(0,65536),n=192,kv_type=Int8,kmax=8,rounds=2)
    rows=[]
    for depth in depths
        ids=[depth>0 ? basecorpus(tokenizer,depth) : Int[];spec_prompt(tokenizer)]
        s=Q36.session(model;capacity=length(ids)+n+64,attention=:flash,kv_type)
        rw=Rewind(s);speculators=Dict(c=>Q36.Speculator(s;kmax,deep=c ? 0 : typemax(Int)) for c in (false,true))
        try
            ingest!(s,ids;chunk=512);Mantle.waitidle(model.device)
            P=s.position;first=Int(only(Array(s.inner.best)));save!(rw)
            foreach(Q36.prepare!,values(speculators))
            reset!()=(restore!(rw);setposition!(s,P);Mantle.waitidle(model.device))
            function plain()
                # The tokens after `first`: one step consumes it, n-2 more follow.
                Q36.step!(s,first);generate!(s,n-2);Mantle.waitidle(model.device)
                Int.(Array(s.inner.predictions)[P+1:P+n-1])
            end
            variants=Dict(:plain=>plain,
                :verify_decode_attention=>()->Q36.speculative_decode!(speculators[false],ids,n;first)[2:end],
                :verify_chunk_attention=>()->Q36.speculative_decode!(speculators[true],ids,n;first)[2:end])
            best=Dict(k=>Inf for k in keys(variants));outputs=Dict{Symbol,Any}()
            for k in keys(variants);reset!();outputs[k]=variants[k]();end
            for r in 1:rounds,k in circshift(collect(keys(variants)),r)
                reset!();t=@elapsed variants[k]();best[k]=min(best[k],t)
            end
            ref=outputs[:plain]
            for k in keys(variants)
                o=outputs[k];same=something(findfirst(i->o[i]!=ref[i],1:min(length(o),length(ref))),min(length(o),length(ref))+1)-1
                row=(;depth,variant=k,tok_s=(n-1)/best[k],agree_tokens=same)
                push!(rows,row)
                println(rpad(string(depth),7),rpad(string(k),26),round(row.tok_s,digits=1)," tok/s, agrees with plain for ",same," tokens")
            end
            for (c,sp) in speculators
                println("    chunk_attention=$c: ",sp.steps," verifies, ",sp.drafted," drafted, ",sp.accepted," accepted (all rounds)")
            end
        finally
            foreach(close,values(speculators));close(rw);close(s);GC.gc(true);Mantle.trim_gpu_pool!()
        end
    end
    rows
end
