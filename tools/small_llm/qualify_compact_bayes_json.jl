include("test_compact_bayes_json.jl")
include("portable_evidence.jl")
import Mantle
using SHA

function qualify_compact_bayes_json(device)
    occursin("llvmpipe",lowercase(sprint(show,device)))||error("Explicit software device required")
    packing=test_compact_bayes_json();rows=Any[]
    numerical=@testset "Compact reference inputs through unchanged shared solver" begin
        for (i,r) in enumerate(bayes_extraction_references())
            nodes=map(r.network.nodes) do n
                n.states==["no","yes"] ? (;name=n.name,parents=n.parents,table=[row.p[2] for row in n.rows]) :
                    (;name=n.name,states=n.states,parents=n.parents,table=[row.p for row in n.rows])
            end
            packed=CompactBayesJSON.expand(JSON3.write((;nodes,query=r.network.query,observed=r.network.observed))).packed
            e=PortableEvidence.prepare_bayes(device,packed.factors,packed.query,packed.observed)
            try
                before=Mantle.submissions(device);PortableEvidence.run!(e);Mantle.waitidle(device)
                submissions=Mantle.submissions(device)-before
                target=i==8 ? "faulty" : "yes";index=findfirst(==(target),packed.query_states)
                posterior=Float64(Array(e.outputs.posterior)[index]);status=only(Array(e.outputs.status))
                @test submissions==1
                @test status==0
                @test isapprox(posterior,Float64(r.answer);atol=1e-6,rtol=0)
                push!(rows,(;id=i,posterior,expected=Float64(r.answer),status,submissions))
            finally close(e) end
        end
    end
    counts=[Test.get_test_counts(t) for t in (packing,numerical)]
    result=(;complete=true,checks=sum(c.passes+c.cumulative_passes for c in counts),rows,
        scope="Private reference frontend/shared-kernel qualification ONLY. No neural generations or improved extraction demonstrated.",
        source_sha256=Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in
            ("compact_bayes_json.jl","bayes_json.jl","portable_evidence.jl","test_compact_bayes_json.jl","qualify_compact_bayes_json.jl","bayes_extraction_references.jl")))
    open(io->JSON3.pretty(io,result),joinpath(@__DIR__,"compact_bayes_reference_validation.json"),"w");result
end
