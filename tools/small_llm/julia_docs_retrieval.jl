module JuliaDocsRetrieval
using SHA

const STOP=Set(split("a an and are as at be by code do for from function in including is it no not of on only or return the to using with write your julia"))
terms(text)=filter(t->length(t)>1 && !(t in STOP),[lowercase(m.match) for m in eachmatch(r"[\p{L}_][\p{L}\p{N}_!]*",text)])

"""Index public package docs only. No benchmark tests, references or generated
answers enter this index. Retrieval never depends on a model-issued tool call.
"""
function build_index(;root="/home/sim/Programmieren/VulkanDev")
    chunks=Any[]
    for directory in ("dev/Makie/docs/src","dev/KernelAbstractions/docs/src",
                      "dev/KernelAbstractions/lib/KernelInterface/src","dev/Makie/ComputePipeline/src")
        for (folder,_,files) in walkdir(joinpath(root,directory)), name in sort(files)
            (endswith(name,".md") || endswith(name,".jl")) || continue
            path=joinpath(folder,name);lines=readlines(path)
            digest=bytes2hex(sha256(read(path)))
            for firstline in 1:32:length(lines)
                lastline=min(firstline+47,length(lines))
                body=join(lines[firstline:lastline],"\n")
                counts=Dict{String,Int}()
                for word in terms(relpath(path,root)*"\n"*body)
                    counts[word]=get(counts,word,0)+1
                end
                isempty(counts) && continue
                push!(chunks,(;source=relpath(path,root),sha256=digest,firstline,lastline,body,counts,nwords=sum(values(counts))))
            end
        end
    end
    sort!(chunks;by=c->(c.source,c.firstline))
    df=Dict{String,Int}()
    for chunk in chunks,word in keys(chunk.counts); df[word]=get(df,word,0)+1; end
    (;chunks,df,avg_length=sum(c.nwords for c in chunks)/length(chunks))
end

function retrieve(index,query;topk=2,chars_per_chunk=2200,bonus=nothing,unique_sources=false,required_sources=String[])
    words=unique(terms(query));n=length(index.chunks)
    scores=map(index.chunks) do chunk
        sum(words;init=0.0) do word
            tf=get(chunk.counts,word,0);df=get(index.df,word,0)
            tf==0 ? 0.0 : log1p((n-df+0.5)/(df+0.5))*tf*2.2/(tf+1.2*(0.25+0.75*chunk.nwords/index.avg_length))
        end
    end
    if bonus!==nothing
        for i in eachindex(scores);scores[i]+=bonus(index.chunks[i]);end
    end
    selected=Any[];order=sortperm(eachindex(scores);by=i->(-scores[i],i))
    function append_chunk(i)
        c=index.chunks[i]
        push!(selected,(;source=c.source,sha256=c.sha256,firstline=c.firstline,lastline=c.lastline,
            score=scores[i],body=first(c.body,chars_per_chunk)))
    end
    # Cover distinct explicitly requested reference pages before adding generic
    # examples. Otherwise three legend examples can crowd out the band API.
    for i in order
        scores[i]>0 || break
        c=index.chunks[i]
        c.source in required_sources || continue
        any(s->s.source==c.source,selected) && continue
        append_chunk(i)
        length(selected)==topk && return selected
    end
    for i in order
        scores[i]>0 || break
        c=index.chunks[i]
        # Do not spend both snippets on overlapping copies of the same lines.
        any(s->s.source==c.source && (unique_sources||abs(s.firstline-c.firstline)<48),selected) && continue
        append_chunk(i)
        length(selected)==topk && break
    end
    selected
end

"""Second retrieval policy: preserve requested API anchors and source diversity.

Unlike plain BM25 over a long exception trace, the query is the original task.
Actual call sites of explicitly requested APIs and matching reference filenames
receive a documented bonus. Does not inspect hidden tests or reference code.
"""
function retrieve_api(index,prompt;topk=3)
    # An explicitly excluded named backend is not a positive retrieval anchor.
    query=replace(prompt,r"\bnot\s+[A-Z][A-Za-z_0-9]*"=>"")
    # An explicitly named package is a scope signal: a plotting question must
    # not fill its remaining snippet budget with similarly named kernel APIs.
    if occursin(r"\bMakie\b",query) && !occursin(r"\b(?:KernelInterface|KernelAbstractions)\b",query)
        index=(;index...,chunks=filter(c->startswith(c.source,"dev/Makie/"),index.chunks))
    end
    words=Set(terms(query))
    generic=Set(["makeplot","solve!","solve","figure","axis","length","abs","float64","collect"])
    calls=unique([lowercase(m.captures[1]) for m in eachmatch(r"([\p{L}_][\p{L}\p{N}_!]+)\s*\(",query)
        if !(lowercase(m.captures[1]) in generic)])
    append!(calls,[lowercase(m.captures[1]) for m in eachmatch(r"\b(?:Makie|KernelInterface)\.([\p{L}_][\p{L}\p{N}_!]*)",query)])
    unique!(calls)
    function path_match(chunk)
        stem=lowercase(first(splitext(basename(chunk.source))))
        !(stem in generic) && (stem in words || any(c->endswith(rstrip(c,'!'),stem),calls))
    end
    bonus=function(chunk)
        pathmatch=path_match(chunk)
        compact=replace(lowercase(chunk.body),r"\s+"=>"")
        exactcalls=count(c->occursin(c*"(",compact),calls)
        (pathmatch ? 16.0 : 0.0)+24.0*exactcalls
    end
    required_sources=unique([c.source for c in index.chunks if path_match(c)])
    retrieve(index,query;topk,chars_per_chunk=3500,bonus,unique_sources=true,required_sources)
end

format_evidence(chunks)=join(("Local package documentation: $(c.source):$(c.firstline)-$(c.lastline); sha256=$(c.sha256)\n$(c.body)" for c in chunks),"\n\n")
end
