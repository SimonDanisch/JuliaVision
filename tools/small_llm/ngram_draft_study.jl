# How much prompt-lookup drafting would save on our own generations, offline:
# every saved tool-loop answer replayed against its context (system, prompt and
# the tool responses it saw), with the drafts an n-gram lookup would have made
# and the tokens the model actually wrote. The time per step comes from the
# measured decode profile, with the routed experts' bytes growing by the experts
# a verify chunk's tokens touch between them (independent routing assumed).
using JSON3,Printf

"""Receipts with a prompt and an answer: `(context, answer)` strings."""
function draft_corpus(dirs)
    out=Tuple{String,String}[]
    for d in dirs,f in sort(readdir(d;join=true))
        endswith(f,".json")&&!endswith(f,"_score.json")||continue
        x=JSON3.read(read(f,String))
        x isa JSON3.Object&&haskey(x,:prompt)&&haskey(x,:answer)||continue
        responses=haskey(x,:calls) ? join((c.response for c in x.calls if c.response!==nothing),"\n") : ""
        push!(out,(string(get(x,:system,""),"\n",x.prompt,"\n",responses),String(x.answer)))
    end
    out
end

"""The `k` tokens after the latest earlier occurrence of the history's last `n`
tokens, for the largest `n` from `nmax` down to `nmin` that occurs; empty if none."""
function ngram_draft(history,k;nmax=4,nmin=2)
    L=length(history)
    for n in min(nmax,L-1):-1:nmin
        tail=view(history,L-n+1:L)
        for s in L-n:-1:1
            if view(history,s:s+n-1)==tail
                return history[s+n:min(s+n+k-1,L)]
            end
        end
    end
    Int[]
end

"""Replay `answer` after `context` (token ids): per step the draft length and the
tokens it accepted."""
function replay_drafts(context,answer;k=5,nmax=4,nmin=2)
    history=copy(context);steps=Tuple{Int,Int}[];i=1
    while i<=length(answer)
        draft=ngram_draft(history,k;nmax,nmin)
        a=0
        while a<length(draft)&&i+a<=length(answer)&&draft[a+1]==answer[i+a];a+=1;end
        advance=min(a+1,length(answer)-i+1)
        append!(history,answer[i:i+advance-1]);i+=advance
        push!(steps,(length(draft),a))
    end
    steps
end

# Milliseconds of one decode or verify step of `m` tokens at `depth`, from the
# decode profile (12.8 ms at short depth, 3.36 of it routed experts) and a q4_0
# cache read at 200 GB/s (640 bytes a position in each of 10 layers).
function step_ms(m,depth;base=12.8,experts=3.36,kvbytes=640*10)
    touched=(1-(1-8/256)^m)/(8/256)
    (base-experts)+experts*touched+depth*kvbytes/200e9*1e3
end

function draft_study(tokenizer,dirs;ks=(2,3,5,8),depths=(4096,65536,131072),nmin=2)
    T=SmallLLMChecks.QwenText;corpus=draft_corpus(dirs)
    pairs=[(T.encode(tokenizer,c),T.encode(tokenizer,a)) for (c,a) in corpus]
    tokens=sum(length(a) for (_,a) in pairs)
    @printf("%d answers, %d generated tokens\n",length(pairs),tokens)
    rows=[]
    for k in ks
        steps=reduce(vcat,[replay_drafts(c,a;k,nmin) for (c,a) in pairs])
        drafted=count(s->s[1]>0,steps);accepted=sum(s[2] for s in steps)
        for depth in depths
            plain=tokens*step_ms(1,depth)
            spec=sum(step_ms(s[1]+1,depth) for s in steps)
            push!(rows,(;k,depth,steps=length(steps),drafted,accepted,tokens_per_step=tokens/length(steps),speedup=plain/spec))
        end
        @printf("k=%d: %d steps (%d drafted), %.2f tokens a step, %d drafted tokens accepted; speedup %s\n",
            k,length(steps),drafted,tokens/length(steps),accepted,
            join(("$(d÷1024)k $(round(r.speedup,digits=2))x" for (d,r) in zip(depths,rows[end-length(depths)+1:end]))," "))
    end
    rows
end
