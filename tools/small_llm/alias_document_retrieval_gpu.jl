isdefined(@__MODULE__,:AliasDocumentRetrieval)||include("alias_document_retrieval.jl")
module AliasDocumentRetrievalGPU
import ..AliasDocumentRetrieval
import KernelInterface as KI

function score_documents!(out,bo,bt,bc,bl,bi,bq,ba::Float32,to,tt,tc,tl,ti,tq,ta::Float32,
        body_queries::Int32,title_queries::Int32,weight::Float32,k1::Float32,b::Float32,n::Int32)
    document=Int32(KI.get_global_id().x)
    if document<=n
        @inbounds out[document]=AliasDocumentRetrieval.score(bo,bt,bc,bl,bi,bq,ba,
            to,tt,tc,tl,ti,tq,ta,document,weight,k1,b,body_queries,title_queries)
    end
    nothing
end
end
