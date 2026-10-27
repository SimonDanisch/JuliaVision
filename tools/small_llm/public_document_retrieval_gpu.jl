isdefined(@__MODULE__,:PublicDocumentRetrieval)||include("public_document_retrieval.jl")
module PublicDocumentRetrievalGPU
import ..PublicDocumentRetrieval
import KernelInterface as KI
import Mantle

"""One writer per document, same sparse BM25 arithmetic as CPU scoring."""
function score_documents!(out,offsets,terms,counts,lengths,idf,query,nquery::Int32,
        average::Float32,k1::Float32,b::Float32,n::Int32)
    document=Int32(KI.get_global_id().x)
    if document<=n
        @inbounds out[document]=PublicDocumentRetrieval.score(offsets,terms,counts,
            lengths,idf,query,document,average,k1,b,nquery)
    end
    nothing
end
end
