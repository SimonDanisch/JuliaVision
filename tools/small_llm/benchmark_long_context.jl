# Long-context throughput and retrieval for a text backbone. One session ingests
# a long document of real Julia source in segments; at each milestone the state
# is saved, a question about a word stated at the start is answered greedily
# (decode speed at that position, and whether attention still reaches the
# start), and the state is restored before ingestion continues.
isdefined(@__MODULE__,:tool_julia_answer) || include("tool_julia.jl")
using Printf

const NEEDLE_WORD="kumquat-cobalt-73"

"""Julia's Base sources, concatenated in a fixed order, as `n` tokens or more."""
function basecorpus(tokenizer,n)
    dir=joinpath(Sys.BINDIR,"..","share","julia","base")
    io=IOBuffer();ids=Int[]
    for (root,_,files) in walkdir(dir),f in sort(files)
        endswith(f,".jl") || continue
        print(io,"\n# file: base/",relpath(joinpath(root,f),dir),"\n",read(joinpath(root,f),String))
        if io.size>4n
            append!(ids,SmallLLMChecks.QwenText.encode(tokenizer,String(take!(io))))
            length(ids)>=n && break
        end
    end
    length(ids)<n && append!(ids,SmallLLMChecks.QwenText.encode(tokenizer,String(take!(io))))
    length(ids)>=n || error("Base sources give only $(length(ids)) tokens")
    ids[1:n]
end

# GPU memory in use, where the driver publishes it (amdgpu sysfs); `nothing` elsewhere.
function gttused()
    f=filter(isfile,["/sys/class/drm/card$i/device/mem_info_gtt_used" for i in 0:3])
    isempty(f) ? nothing : parse(Int,readchomp(first(f)))/2^30
end

# Qwen3.6 has 262144 positions; the last milestone leaves room for the question.
function long_context_profile(model,tokenizer;milestones=(4096,16384,65536,131072,261120),decode=64,attention=:grouped,kv_type=Float32,chunk=INGEST_CHUNK[])
    T=SmallLLMChecks.QwenText
    head=T.encode(tokenizer,"<|im_start|>user\nRemember this: the secret code word is $NEEDLE_WORD. Below is a long excerpt of Julia's Base sources.\n\n")
    question=T.encode(tokenizer,"\n\nQuestion: what is the secret code word stated at the very beginning? Answer with the word only.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    total=maximum(milestones)
    corpus=basecorpus(tokenizer,total-length(head))
    document=[head;corpus]
    capacity=total+length(question)+decode+64
    s=Q36.session(model;capacity,attention,kv_type);rw=Rewind(s);rows=Any[]
    try
        done=0
        for L in milestones
            t=@elapsed (ingest!(s,document[done+1:L];chunk);Mantle.waitidle(model.device))
            segment=L-done;done=L
            save!(rw);P=s.position
            tq=@elapsed (ingest!(s,question;chunk);Mantle.waitidle(model.device))
            pos=s.position;first_logits=Array(s.logits)
            td=@elapsed (generate!(s,decode);Mantle.waitidle(model.device))
            answer=T.decode(tokenizer,Int.(Array(s.predictions)[pos:pos+decode]))
            restore!(rw);setposition!(s,P)
            row=(;tokens=L,segment,prefill_seconds=t,prefill_tok_s=segment/t,question_seconds=tq,
                decode_tok_s=decode/td,retrieved=occursin(NEEDLE_WORD,answer),answer=first(answer,120),
                gtt_gib=gttused(),first_logits)
            push!(rows,row)
            @printf("%7d tokens: prefill %6.1f tok/s over the last %d, decode %5.1f tok/s, retrieved=%s, GTT %s GiB\n",
                L,row.prefill_tok_s,segment,row.decode_tok_s,row.retrieved,row.gtt_gib===nothing ? "-" : round(row.gtt_gib;digits=1));flush(stdout)
        end
    finally
        close(rw);close(s)
    end
    rows
end

"""KV formats against a reference format, teacher-forced: one session per format
ingests the same `depth` tokens of Julia's Base sources, then steps through the
next `positions` tokens of it; at every step the KL of the reference's next-token
distribution to each format's, and whether the argmax agrees."""
function kv_format_divergence(model,tokenizer;depth=32768,positions=64,formats=(Int8,Qwen36MoEExperiment.Q.Q4_0),reference=Float16,chunk=512)
    doc=basecorpus(tokenizer,depth+positions)
    function trace(kv_type)
        s=Q36.session(model;capacity=depth+positions+8,attention=:flash,kv_type)
        try
            ingest!(s,doc[1:depth];chunk);Mantle.waitidle(model.device)
            logits=Vector{Vector{Float32}}()
            for t in depth+1:depth+positions
                push!(logits,Array(s.logits))
                feed!(s,doc[t])
            end
            logits
        finally
            close(s);GC.gc(true);Mantle.trim_gpu_pool!()
        end
    end
    ref=trace(reference)
    rows=map(formats) do T
        got=trace(T)
        kls=[kl(a,b) for (a,b) in zip(ref,got)]
        (;kv_type=T,mean_kl=sum(kls)/length(kls),max_kl=maximum(kls),argmax_agree=count(argmax(a)==argmax(b) for (a,b) in zip(ref,got))/positions)
    end
    rows
end
