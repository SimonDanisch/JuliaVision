include("finite_choice_constraint.jl")
using Test
function test_finite_choice_constraint()
    G=FiniteChoiceConstraint
    codec=s->Int32.(codeunits(s));decode=ids->String(UInt8.(ids))
    @testset "Finite canonical support, not semantic correctness" begin
        g=G.Grammar(codec,decode,["a","ab","λ"],512,511)
        for path in g.sequences
            state=(Int32(1),Int32(0),Int32(0))
            for token in path
                @test isfinite(G.constrained_value(g.trie.offsets,g.trie.labels,g.trie.targets,state[1],state[2],g.eos,token,1f0))
                state=G.transition(g,state,token)
                @test state[3]==0
            end
            @test state[2]==1
            @test G.transition(g,state,Int32(511))==state
            @test G.transition(g,state,Int32(97))[3]==1
            @test G.constrained_value(g.trie.offsets,g.trie.labels,g.trie.targets,state[1],state[2],g.eos,Int32(97),2f0)==-Inf32
        end
        @test G.transition(g,(Int32(1),Int32(0),Int32(0)),Int32(511))[3]==1
        @test G.transition(g,(Int32(-1),Int32(0),Int32(1)),Int32(97))[3]==1
        @test_throws ArgumentError G.Grammar(codec,decode,String[],512,511)
        @test_throws ArgumentError G.Grammar(codec,decode,["a","a"],512,511)
        @test_throws ArgumentError G.Grammar(codec,decode,[""],512,511)
        @test_throws ArgumentError G.Grammar(codec,decode,["a"],512,true)
        @test_throws ArgumentError G.Grammar(codec,decode,["a"],512,511;max_nodes=1)
        @test_throws ArgumentError G.Grammar(s->[511],decode,["a"],512,511)
        @test_throws ArgumentError G.Grammar(codec,ids->"wrong",["a"],512,511)
        @test_throws ArgumentError G.Grammar(s->[true],decode,["a"],512,511)
        @test_throws ArgumentError G.Grammar(codec,decode,["a"],512,511;max_choices=true)
    end
end
