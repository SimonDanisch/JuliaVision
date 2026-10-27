using JSON3, SHA, Test
include("test_alias_document_retrieval.jl")
include("test_alias_document_context.jl")
include("test_alias_document_retrieval_gpu.jl")

"""Host/full-corpus and software-only arithmetic qualification, not quality."""
function qualify_alias_document_retrieval(device;destination=joinpath(@__DIR__,"alias_document_retrieval_validation.json"))
    isfile(destination)&&error("Preserve existing qualification receipt")
    occursin("llvmpipe",lowercase(string(device)))||error("Software-only qualification requires llvmpipe")
    parent_path=joinpath(@__DIR__,"julia_public_docs.json")
    captured_path=joinpath(@__DIR__,"julia_public_binding_docs.json")
    parent=JSON3.read(read(parent_path,String));captured=JSON3.read(read(captured_path,String))
    index=AliasDocumentRetrieval.Index(captured.rows)
    unit=test_alias_document_retrieval();context=test_alias_document_context()
    corpus=@testset "Captured corpus grouping fidelity" begin
        @test parent.complete && captured.complete
        @test captured.parent_docs_sha256==bytes2hex(sha256(read(parent_path)))
        @test length(parent.rows)==length(captured.rows)
        @test all(zip(parent.rows,captured.rows)) do (old,new)
            old.qualified==new.qualified && old.text==new.text
        end
        @test sum(length,index.aliases)==length(captured.rows)
        @test all(length(names)==length(unique(names)) for names in index.aliases)
        original=Dict(String(row.qualified)=>row for row in captured.rows)
        @test all(eachindex(index.aliases)) do i
            all(name->original[name].binding_id==index.binding_ids[i] &&
                original[name].text==index.body.texts[i],index.aliases[i])
        end
        @test length(index.body.names)<=length(captured.rows)
        for query in ("scatter points histogram bins","save figure PNG image","JSON parse dictionary","nonexistentwordxyz")
            pack=AliasDocumentContext.build(index,query)
            @test length(pack.rows)<=4
            @test all(r->length(r.text)<=1800,pack.rows)
            @test all(r->isfinite(r.score)&&r.score>0,pack.rows)
        end
    end
    software=test_alias_document_retrieval_gpu(device,index)
    count(t)=let c=Test.get_test_counts(t);c.passes+c.cumulative_passes end
    files=("alias_document_retrieval.jl","alias_document_retrieval_gpu.jl","alias_document_context.jl",
        "public_document_retrieval.jl","test_alias_document_retrieval.jl",
        "test_alias_document_retrieval_gpu.jl","test_alias_document_context.jl",
        "qualify_alias_document_retrieval.jl","capture_julia_binding_identity.py")
    packet=(;complete=true,scope="Infrastructure only: source fidelity, grouping, quoted context, full-corpus CPU/software-Vulkan arithmetic. No neural model or intelligence evaluation.",
        parent_sha256=bytes2hex(sha256(read(parent_path))),corpus_sha256=bytes2hex(sha256(read(captured_path))),
        original_records=length(captured.rows),grouped_records=length(index.body.names),
        unit_checks=count(unit),context_checks=count(context),corpus_checks=count(corpus),software_checks=count(software),
        device=string(device),sources=Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in files))
    open(destination,"w") do io;JSON3.write(io,packet);end
    println(JSON3.write(packet));packet
end
