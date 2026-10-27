include("test_alias_document_retrieval_gpu.jl")
include("lint_document_context.jl")
using Test,JSON3,SHA
function qualify_julia_docs_v2(device;destination=joinpath(@__DIR__,"julia_docs_v2_validation.json"))
    isfile(destination)&&error("Preserve expanded corpus qualification")
    occursin("llvmpipe",lowercase(string(device)))||error("ROOT software-only")
    path=joinpath(@__DIR__,"julia_public_binding_docs_v2.json");c=JSON3.read(read(path,String))
    c.complete&&c.gpu_nodes_visible===false||error("Capture incomplete or not CPU-only")
    bytes2hex(sha256(read(joinpath(@__DIR__,"capture_julia_docs_v2.py"))))==c.source_sha256||error("Capture source drift")
    index=AliasDocumentRetrieval.Index(c.rows);original=Dict(String(r.qualified)=>r for r in c.rows)
    old=JSON3.read(read(joinpath(@__DIR__,"julia_public_binding_docs.json"),String))
    tests=@testset "Expanded docs: source fidelity, effective names, bounded contexts" begin
        @test length(c.modules)==17
        @test length(original)==length(c.rows)
        @test all(r->haskey(original,String(r.qualified)),old.rows)
        @test sum(length,index.aliases)==length(c.rows)
        @test all(eachindex(index.aliases)) do i
            all(name->original[name].binding_id==index.binding_ids[i]&&original[name].text==index.body.texts[i],index.aliases[i])
        end
        @test haskey(original,"Dates.DateTime")
        @test "parse" in c.bindings.JSON3 # effective inherited names are not necessarily package-specific methods
        @test isempty(QualifiedBindingDiagnosticsV2.analyze("JSON3.parse(x)",c.bindings).references_to_unlisted_bindings)
        for query in ("parse JSON values into Julia structures","date and time formatted as text","histogram numeric vector","multiply a sparse matrix")
            pack=AliasDocumentContext.build(index,query;documents=4,characters_per_document=1200)
            @test length(pack.rows)<=4
            @test all(r->r.text==first(original[r.qualified].text,1200),pack.rows)
            @test all(r->isfinite(r.score)&&r.score>0,pack.rows)
        end
        lint=LintDocumentContext.build(index,c.bindings,"decode JSON","JSON3.definitely_unlisted_candidate_name(x)")
        @test lint.diagnostic.unlisted_qualified_names==["JSON3.definitely_unlisted_candidate_name"]
        @test all(r->r.text==first(original[r.qualified].text,1200),lint.rows)
    end
    software=test_alias_document_retrieval_gpu(device,index)
    counts(t)=let x=Test.get_test_counts(t);x.passes+x.cumulative_passes end
    changed_old=[String(r.qualified) for r in old.rows if original[String(r.qualified)].text!=r.text]
    result=(;complete=true,scope="Expanded PUBLIC corpus/grouping/effective namespace and shared CPU/software-Vulkan score qualification ONLY, no new neural quality study or changes to frozen corpus",records=length(c.rows),grouped_records=length(index.body.names),
        corpus_checks=counts(tests),software_checks=counts(software),prior_record_text_changes=changed_old,
        capture_seconds=c.capture_seconds,corpus_sha256=bytes2hex(sha256(read(path))),
        sources=Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in
            ("qualify_julia_docs_v2.jl","capture_julia_docs_v2.py","alias_document_retrieval.jl","alias_document_retrieval_gpu.jl","public_document_retrieval.jl","qualified_binding_diagnostics_v2.jl","lint_document_context.jl")))
    open(io->JSON3.pretty(io,result),destination,"w");println("Expanded corpus qualified: ",result.records," records, ",result.software_checks," shared-math software checks");result
end
