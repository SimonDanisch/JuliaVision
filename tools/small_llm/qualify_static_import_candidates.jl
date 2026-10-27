include("test_static_import_candidates.jl")
using JSON3,SHA
function qualify_static_import_candidates(;destination=joinpath(@__DIR__,"static_import_candidates_validation.json"))
    isfile(destination)&&error("Preserve qualification receipt")
    unit=test_static_import_candidates()
    corpuspath=joinpath(@__DIR__,"julia_public_binding_docs_v2.json")
    corpus=JSON3.read(read(corpuspath,String))
    live=@testset "Import hints from actual captured public namespaces" begin
        check(s)=StaticImportCandidates.analyze(s,corpus)
        @test "Dates" in only(filter(x->x.name=="DateTime",check("DateTime(x)").suggestions)).possible_modules
        @test isempty(check("using Dates;DateTime(x)").suggestions)
        @test "Dates" in only(filter(x->x.name=="DateTime",check("import Dates;DateTime(x)").suggestions)).possible_modules
        @test isempty(check("using JSON3;JSON3.read(x);println(x)").suggestions)
        @test isempty(check("f(DateTime)=DateTime(x)").suggestions)
        @test check("@eval using Dates;DateTime(x)").opaque_metaprogramming
    end
    counts(t)=let x=Test.get_test_counts(t);x.passes+x.cumulative_passes end
    root=joinpath(@__DIR__,"alias_transfer3_results")
    conditions=JSON3.read(read(joinpath(root,"conditions.json"),String))
    rows=map(conditions.jobs) do job
        path=joinpath(root,String(job.id)*".json")
        receipt=JSON3.read(read(path,String));observations=StaticImportCandidates.analyze(String(receipt.code),corpus)
        (;id=job.id,generation_sha256=bytes2hex(sha256(read(path))),observations)
    end
    result=(;complete=true,unit_checks=counts(unit),captured_namespace_checks=counts(live),
        scope="POSTHOC static source inspection of ALL16 completed original external answers, no score/test-trace inputs and no candidate execution. Hints are not demonstrated failures or repairs. NOT used in frozen lint revision prompts; no model-quality claim.",
        corpus_sha256=bytes2hex(sha256(read(corpuspath))),rows,
        sources=Dict(file=>bytes2hex(sha256(read(joinpath(@__DIR__,file)))) for file in
            ("static_import_candidates.jl","test_static_import_candidates.jl","qualify_static_import_candidates.jl")))
    open(io->JSON3.pretty(io,result),destination,"w");result
end
