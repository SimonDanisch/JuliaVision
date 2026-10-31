"""Task-independent sparse BM25 index. Scoring is shared host/device arithmetic.

Tokenization/index construction are offline host work. Retrieval is evidence
selection, not proof, and never executes document text or generated programs.
"""
module PublicDocumentRetrieval

struct Index
    names::Vector{String}
    texts::Vector{String}
    vocabulary::Dict{String,Int32}
    offsets::Vector{Int32}
    terms::Vector{Int32}
    counts::Vector{Int32}
    lengths::Vector{Int32}
    idf::Vector{Float32}
    average_length::Float32
end

function words(text)
    # Keep exact API identifiers as well as camel-case pieces. Fixed for all
    # documents/queries; no task-specific keyword additions or stemming rules.
    spaced=replace(String(text),r"([a-z])([A-Z])"=>s"\1 \2")
    [lowercase(m.match) for m in eachmatch(r"[\pL_][\pL\pN_!]*",spaced)]
end

function Index(rows)
    isempty(rows)&&throw(ArgumentError("empty documentation"))
    names=String[String(r.qualified) for r in rows]
    all(!isempty,names)||throw(ArgumentError("empty document name"))
    length(unique(names))==length(names)||throw(ArgumentError("duplicate document name"))
    texts=String[String(r.text) for r in rows]
    documents=[words(names[i]*"\n"*texts[i]) for i in eachindex(names)]
    dictionary=sort!(unique(reduce(vcat,documents;init=String[])))
    length(dictionary)<typemax(Int32)||throw(ArgumentError("vocabulary too large"))
    vocabulary=Dict(w=>Int32(i) for (i,w) in enumerate(dictionary))
    offsets=Int32[1];terms=Int32[];counts=Int32[];lengths=Int32[]
    frequency=zeros(Int32,length(dictionary))
    for document in documents
        length(document)<=typemax(Int32)||throw(ArgumentError("document too large"))
        tally=Dict{Int32,Int32}()
        for word in document
            id=vocabulary[word];tally[id]=get(tally,id,Int32(0))+Int32(1)
        end
        for id in sort!(collect(keys(tally)))
            push!(terms,id);push!(counts,tally[id]);frequency[id]+=Int32(1)
        end
        length(terms)<typemax(Int32)||throw(ArgumentError("index too large"))
        push!(offsets,Int32(length(terms)+1));push!(lengths,Int32(length(document)))
    end
    n=length(names)
    idf=Float32[log(1+(n-f+0.5)/(f+0.5)) for f in frequency]
    Index(names,texts,vocabulary,offsets,terms,counts,lengths,idf,Float32(sum(Int64,lengths)/n))
end

function query_ids(index::Index,text)
    sort!(unique(Int32[index.vocabulary[w] for w in words(text) if haskey(index.vocabulary,w)]))
end

"""Shared allocation-free BM25 arithmetic over sparse sorted document terms."""
@inline function score(offsets,terms,counts,lengths,idf,query,document::Int32,
        average_length::Float32,k1::Float32,b::Float32,query_length::Int32=Int32(length(query)))
    result=0f0
    @inbounds normalization=k1*(1f0-b+b*Float32(lengths[document])/average_length)
    for qi in Int32(1):query_length
        @inbounds q=query[qi]
        @inbounds lo=offsets[document]
        @inbounds hi=offsets[document+Int32(1)]-Int32(1)
        while lo<=hi
            mid=lo+((hi-lo)>>1)
            @inbounds term=terms[mid]
            if term==q
                @inbounds count=Float32(counts[mid])
                @inbounds result+=idf[q]*count*(k1+1f0)/(count+normalization)
                break
            elseif term<q
                lo=mid+Int32(1)
            else
                hi=mid-Int32(1)
            end
        end
    end
    result
end

function retrieve(index::Index,text;limit=4,k1=1.2f0,b=0.75f0)
    limit isa Integer && !(limit isa Bool) && limit>=0 || throw(ArgumentError("invalid limit"))
    k1=Float32(k1);b=Float32(b)
    isfinite(k1)&&k1>0&&isfinite(b)&&0<=b<=1||throw(ArgumentError("invalid BM25 parameters"))
    query=query_ids(index,text)
    scores=[score(index.offsets,index.terms,index.counts,index.lengths,index.idf,
        query,Int32(d),index.average_length,k1,b) for d in eachindex(index.names)]
    order=sortperm(eachindex(scores);by=d->(-scores[d],index.names[d]))
    positive=filter(d->scores[d]>0,order)
    chosen=first(positive,min(limit,length(positive)))
    [(;qualified=index.names[d],text=index.texts[d],score=scores[d]) for d in chosen]
end
end
