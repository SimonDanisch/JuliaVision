include("compact_bayes_json.jl")
include("bayes_extraction_references.jl")
include("bayes_extraction_admission.jl")
using Test,JSON3

function test_compact_bayes_json()
    @testset "Compact CPT packing, no inference implementation" begin
        for r in bayes_extraction_references()
            # Private reference conversion ONLY in this test. No reference data
            # enters the production frontend or any model prompt.
            nodes=map(r.network.nodes) do n
                if n.states==["no","yes"]
                    (;name=n.name,parents=n.parents,table=[row.p[2] for row in n.rows])
                else
                    (;name=n.name,states=n.states,parents=n.parents,table=[row.p for row in n.rows])
                end
            end
            text=JSON3.write((;nodes,query=r.network.query,observed=r.network.observed))
            out=CompactBayesJSON.expand(text)
            @test bayes_extraction_faithful(JSON3.read(JSON3.write(out.network)),JSON3.read(JSON3.write(r.network)))
            old=BayesJSON.compile(JSON3.write(r.network);max_states=256)
            @test out.packed.query==old.query&&out.packed.observed==old.observed
            @test all(a.variables==b.variables&&a.values==b.values for (a,b) in zip(out.packed.factors,old.factors))
            @test_throws ArgumentError CompactBayesJSON.expand(text*" trailing")
        end
        root=(;name="a",parents=String[],table=[0.2])
        good=(;nodes=[root],query="a",observed=Dict())
        @test CompactBayesJSON.expand(JSON3.write(good)).packed.nstates==2
        for nodes in ([merge(root,(;table=[true]))],[merge(root,(;table=[1.2]))],
                [merge(root,(;table=[0.1,0.2]))],[merge(root,(;parents=["a"]))],
                [root,root],[merge(root,(;states=["x","y"],table=[0.2]))],
                [merge(root,(;states=["x","y"],table=[[0.2,0.2]]))])
            @test_throws ArgumentError CompactBayesJSON.expand(JSON3.write((;good...,nodes)))
        end
        @test_throws ArgumentError CompactBayesJSON.expand(JSON3.write(good);max_states=1)
        @test_throws ArgumentError CompactBayesJSON.expand(JSON3.write((;good...,unknown=1)))
        @test_throws ArgumentError CompactBayesJSON.expand(JSON3.write((;good...,nodes=[merge(root,(;code="evil()"))])))
        @test_throws ArgumentError CompactBayesJSON.expand("{\"nodes\":[],\"query\":\"a\",\"observed\":{}} {}")
        mixed=(;nodes=[root,(;name="b",parents=String[],states=["low","mid","high"],table=[[0.2,0.3,0.5]]),
            (;name="c",parents=["a","b"],table=[0.01,0.02,0.03,0.04,0.05,0.06])],query="c",observed=Dict("b"=>"mid"))
        packed=CompactBayesJSON.expand(JSON3.write(mixed)).packed
        @test packed.nstates==12
        factor=last(packed.factors)
        @test factor.variables==[:a,:b,:c]
        for a in 1:2,b in 1:3
            @test factor.values[a,b,2]==mixed.nodes[3].table[(a-1)*3+b]
            @test factor.values[a,b,1]==1-mixed.nodes[3].table[(a-1)*3+b]
        end
        cyclic=(;nodes=[(;name="a",parents=["b"],table=[.2,.3]),(;name="b",parents=["a"],table=[.4,.5])],query="a",observed=Dict())
        @test_throws ArgumentError CompactBayesJSON.expand(JSON3.write(cyclic))
        @test_throws ArgumentError CompactBayesJSON.expand(JSON3.write((;good...,observed=Dict("a"=>"unknown"))))
        @test_throws ArgumentError CompactBayesJSON.expand(JSON3.write((;good...,query="unknown")))
        @test_throws ArgumentError CompactBayesJSON.expand(JSON3.write((;good...,nodes=[merge(root,(;parents=["unknown"]))])))
    end
end
