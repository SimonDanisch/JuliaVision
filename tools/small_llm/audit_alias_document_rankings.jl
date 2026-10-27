using JSON3, SHA, Test
isdefined(@__MODULE__,:AliasDocumentContext)||include("alias_document_context.jl")

"""Descriptive ranking audit of already-used PUBLIC requests; no relevance labels.

Not a held-out test, retrieval metric or neural evaluation. Never reads candidate
outputs, grader results, hidden tests or reference programs.
"""
function audit_alias_document_rankings(;destination=joinpath(@__DIR__,"alias_document_ranking_audit.json"))
    isfile(destination)&&error("Preserve ranking audit")
    corpus_path=joinpath(@__DIR__,"julia_public_binding_docs.json")
    public_path=normpath(joinpath(@__DIR__,"../../../../gen/bigcodebench-x-followup-20261025/public_prompts.jsonl"))
    corpus=JSON3.read(read(corpus_path,String));corpus.complete||error("Incomplete corpus")
    original=PublicDocumentRetrieval.Index(corpus.rows)
    grouped=AliasDocumentRetrieval.Index(corpus.rows)
    identity=Dict(String(r.qualified)=>(Int64(r.binding_id),String(r.text)) for r in corpus.rows)
    requests=JSON3.read.(readlines(public_path));rows=Any[]
    checks=@testset "Public-only ranking audit integrity" begin
        @test length(requests)==8
        for request in requests
            query=String(request.prompt)
            baseline=PublicDocumentRetrieval.retrieve(original,query)
            dedup=AliasDocumentContext.build(grouped,query;characters_per_document=1200,name_weight=0f0)
            named=AliasDocumentContext.build(grouped,query;characters_per_document=1200,name_weight=2f0)
            @test dedup.query==named.query==query
            @test length(unique(identity[r.qualified] for r in dedup.rows))==length(dedup.rows)
            @test length(unique(identity[r.qualified] for r in named.rows))==length(named.rows)
            @test all(r->all(alias->identity[alias]==identity[r.qualified],r.aliases),named.rows)
            push!(rows,(;task=String(request.task_id),
                original_names=[r.qualified for r in baseline],
                original_unique_binding_text_groups=length(unique(identity[r.qualified] for r in baseline)),
                grouped_name_weight_zero=dedup.rows,grouped_name_weight_two=named.rows))
        end
    end
    c=Test.get_test_counts(checks)
    packet=(;complete=true,scope="POSTHOC descriptive source-ranking audit on eight already-used public requests. No human relevance labels, candidate outputs, hidden tests, scores or reference programs read. Not intelligence evidence; neither grouped variant has been evaluated by a model.",
        score_definition="Frozen index constructor includes canonical name in each field: body indexes canonical name plus doc text; name field indexes canonical name plus all aliases. Combined score=body+weight*name. Weight0 changes grouping/IDF, not just final deduplication.",
        corpus_sha256=bytes2hex(sha256(read(corpus_path))),public_requests_sha256=bytes2hex(sha256(read(public_path))),
        source_sha256=bytes2hex(sha256(read(@__FILE__))),checks=c.passes+c.cumulative_passes,rows)
    open(destination,"w") do io;JSON3.write(io,packet);end
    println("Descriptive public ranking audit checks=",packet.checks);packet
end
