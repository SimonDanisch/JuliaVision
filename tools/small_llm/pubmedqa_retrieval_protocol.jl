include("pubmedqa_reading_protocol.jl")
isdefined(@__MODULE__,:PublicDocumentRetrieval)||include("public_document_retrieval.jl")
module PubMedQARetrievalProtocol
import ..PubMedQAReadingProtocol, ..PublicDocumentRetrieval
using JSON3,SHA
const ARMS=("question_only","corresponding_context","mismatched_context","retrieved_context")

"""Automatic body-only BM25, unchanged public scorer math, no model query.
All evidence arms use the SAME character cap/one-string paragraph format.
Zero-match queries remain in the cohort with an empty source pack.
"""
function jobs(tasks,corpus;characters=8000)
    characters isa Integer&&!(characters isa Bool)&&0<characters<=8000||throw(ArgumentError("Character cap"))
    all(t->Set(String.(keys(t)))==Set(["pmid","question","abstract_context"]),tasks)||throw(ArgumentError("Public fields only"))
    all(r->Set(String.(keys(r)))==Set(["qualified","text"])&&occursin(r"^[0-9]+$",String(r.qualified)),corpus)||throw(ArgumentError("Body-only corpus schema"))
    normalized=[(;pmid=String(t.pmid),question=String(t.question),
        abstract_context=[first(join(String.(t.abstract_context),"\n\n"),characters)]) for t in tasks]
    base=PubMedQAReadingProtocol.jobs(normalized)
    index=PublicDocumentRetrieval.Index(corpus)
    rows=Any[]
    for (i,task) in enumerate(normalized)
        query=String(task.question)
        found=PublicDocumentRetrieval.retrieve(index,query;limit=1,k1=1.2f0,b=0.75f0)
        snippets=isempty(found) ? String[] : [first(only(found).text,characters)]
        retrieved_id=isempty(found) ? nothing : only(found).qualified
        template=only(filter(j->j.task==task.pmid&&j.arm=="corresponding_context",base))
        request="Research question (JSON string):\n"*JSON3.write(query)*"\n\nRetrieved abstract context (JSON string array; conclusion withheld):\n"*JSON3.write(snippets)
        retrieved=(;id="pubmed_"*task.pmid*"_retrieved_context",task=task.pmid,arm="retrieved_context",context_pmid=retrieved_id,
            messages=[first(template.messages),"user"=>request])
        for offset in 0:3
            arm=ARMS[mod1(i+offset,4)]
            job=arm=="retrieved_context" ? retrieved : only(filter(j->j.task==task.pmid&&j.arm==arm,base))
            push!(rows,(;job...,query_sha256=bytes2hex(sha256(query)),
                retrieval_score=arm=="retrieved_context"&&!isempty(found) ? only(found).score : nothing,
                retrieval_found=arm=="retrieved_context" ? !isempty(found) : nothing))
        end
    end
    rows
end
end
