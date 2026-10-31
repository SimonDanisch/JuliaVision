isdefined(@__MODULE__,:PublicDocumentRetrieval)||include("public_document_retrieval.jl")

"""Experimental source-only alias grouping and two-field lexical scores.

Group only actual captured binding identity AND byte-identical documentation.
Scoring reuses the frozen allocation-free BM25 core on CPU/GPU; no task labels,
model-generated query, hidden tests or semantic equivalence inference.
"""
module AliasDocumentRetrieval
import ..PublicDocumentRetrieval
const PDR=PublicDocumentRetrieval

struct Index
    body::PDR.Index
    titles::PDR.Index
    aliases::Vector{Vector{String}}
    binding_ids::Vector{Int64}
end

function Index(rows)
    isempty(rows)&&throw(ArgumentError("empty binding corpus"))
    groups=Dict{Tuple{Int64,String},Vector{String}}();seen=Set{String}()
    for row in rows
        id=row.binding_id
        id isa Integer&&!(id isa Bool)&&0<id<=typemax(Int64)||throw(ArgumentError("invalid binding identity"))
        name=String(row.qualified);isempty(name)&&throw(ArgumentError("empty public name"))
        name in seen&&throw(ArgumentError("duplicate public name"));push!(seen,name)
        push!(get!(groups,(Int64(id),String(row.text)),String[]),name)
    end
    grouped=[(;binding_id=key[1],text=key[2],aliases=sort!(names)) for (key,names) in groups]
    sort!(grouped;by=r->first(r.aliases))
    body=PDR.Index([(;qualified=first(r.aliases),text=r.text) for r in grouped])
    titles=PDR.Index([(;qualified=first(r.aliases),text=join(r.aliases,"\n")) for r in grouped])
    Index(body,titles,[r.aliases for r in grouped],Int64[r.binding_id for r in grouped])
end

"""Same flat numeric scoring function for host and device callers."""
@inline function score(bo,bt,bc,bl,bi,bq,ba::Float32,to,tt,tc,tl,ti,tq,ta::Float32,
        document::Int32,weight::Float32,k1::Float32,b::Float32,
        body_queries::Int32=Int32(length(bq)),title_queries::Int32=Int32(length(tq)))
    body=PDR.score(bo,bt,bc,bl,bi,bq,document,ba,k1,b,body_queries)
    title=PDR.score(to,tt,tc,tl,ti,tq,document,ta,k1,b,title_queries)
    body+weight*title
end

function retrieve(index::Index,text;limit=4,name_weight=2f0,k1=1.2f0,b=0.75f0)
    limit isa Integer&&!(limit isa Bool)&&limit>=0||throw(ArgumentError("invalid limit"))
    weight=Float32(name_weight);k1=Float32(k1);b=Float32(b)
    isfinite(weight)&&0<=weight<=16&&isfinite(k1)&&k1>0&&isfinite(b)&&0<=b<=1||throw(ArgumentError("invalid score parameters"))
    body=index.body;titles=index.titles;bq=PDR.query_ids(body,text);tq=PDR.query_ids(titles,text)
    scores=[score(body.offsets,body.terms,body.counts,body.lengths,body.idf,bq,body.average_length,
        titles.offsets,titles.terms,titles.counts,titles.lengths,titles.idf,tq,titles.average_length,
        Int32(i),weight,k1,b) for i in eachindex(body.names)]
    order=sortperm(eachindex(scores);by=i->(-scores[i],body.names[i]))
    positive=filter(i->scores[i]>0,order);chosen=first(positive,min(limit,length(positive)))
    [(;qualified=body.names[i],aliases=copy(index.aliases[i]),binding_id=index.binding_ids[i],text=body.texts[i],score=scores[i]) for i in chosen]
end
end
