include("rpn_prefix_grammar.jl")
using Test,Random
function test_rpn_prefix_grammar()
    G=RPNPrefixGrammar
    encode=s->Int32.(codeunits(s))
    feed(g,state,action)=foldl((s,t)->G.transition(g,s,t),g.sequences[action];init=state)
    function allowed(depth,operations,maxdepth,budget)
        operations>=budget&&return depth==1 ? Int[G.FINISH] : findall(==(2),G.ARITIES)
        [a for a in eachindex(G.ARITIES) if G.ARITIES[a]==3 ? depth==1 :
            depth>=G.ARITIES[a] && !(G.ARITIES[a]==0&&depth==maxdepth)]
    end
    @testset "Finite postfix grammar; no arithmetic oracle" begin
        g=G.Grammar(encode,512,511;max_depth=3,budget=3)
        @test length(g.roots)==7
        @test length(g.offsets)==length(g.terminal_action)+1
        @test length(g.labels)==length(g.targets)
        @test last(g.offsets)==length(g.labels)+1
        start=G.initial(g)
        @test feed(g,start,12)[5]==1 # binary cannot consume an empty stack
        @test feed(g,start,G.FINISH)[5]==1
        @test feed(g,start,G.EOS_ACTION)[5]==1
        state=feed(g,start,1)
        state=feed(g,state,1)
        state=feed(g,state,11) # square, then budget exhausted at depth two
        @test state[2:3]==(Int32(2),Int32(3))
        @test feed(g,state,1)[5]==1
        @test feed(g,state,10)[5]==1
        state=feed(g,state,12) # model-selected addition reduces depth
        @test state[2:3]==(Int32(1),Int32(4))
        state=feed(g,state,G.FINISH)
        @test state[4]==0 && state[1]==g.roots[7]
        state=feed(g,state,G.EOS_ACTION)
        @test state[4:5]==(Int32(1),Int32(0))
        @test G.transition(g,state,0)==state
        bad=(start[1],Int32(0),Int32(0),Int32(0),Int32(1))
        @test G.transition(g,bad,Int32('x'))[5]==1
        for badstate in ((start[1],-1,0,0,0),(start[1],4,0,0,0),
                         (start[1],0,-1,0,0),(start[1],0,10,0,0),(start[1],0,0,2,0))
            @test G.transition(g,Int32.(badstate),0)[5]==1
        end
        for kwargs in ((;budget=0),(;budget=true),(;budget=1025),(;max_depth=1),(;max_depth=65),(;max_depth=true))
            @test_throws ArgumentError G.Grammar(encode,512,511;kwargs...)
        end
        @test_throws ArgumentError G.Grammar(encode,true,0)
        @test_throws ArgumentError G.Grammar(encode,512,true)
        @test_throws ArgumentError G.Grammar(encode,512,512)
        @test_throws ArgumentError G.Grammar(s->[511],512,511)
        @test_throws ArgumentError G.Grammar(s->[1],512,511)
        @test_throws ArgumentError G.Grammar(s->Int[],512,511)
        # Exhaustive action admission at every reachable arity/budget class.
        for depth in 0:3,operations in 0:3
            depth==0&&operations==3&&continue
            root=G.root_for(g.roots,Int32(depth),Int32(operations),g.max_depth,g.budget)
            s=(root,Int32(depth),Int32(operations),Int32(0),Int32(0))
            valid=allowed(depth,operations,3,3)
            for action in eachindex(G.ARITIES)
                @test (feed(g,s,action)[5]==0)==(action in valid)
            end
        end
        # Random complete paths must finish within the proven action bound.
        rng=MersenneTwister(20261025)
        for maxdepth in (2,3,8),budget in (1,3,32),rep in 1:64
            grammar=G.Grammar(encode,512,511;max_depth=maxdepth,budget)
            s=G.initial(grammar);steps=0
            while s[1]!=grammar.roots[7]
                choices=allowed(s[2],s[3],maxdepth,budget)
                action=rand(rng,choices)
                s=feed(grammar,s,action);steps+=1
                @test s[5]==0 && 0<=s[2]<=maxdepth
                steps<=budget+maxdepth||error("grammar failed to terminate")
            end
            @test s[2]==1 && s[3]<=budget+maxdepth-1
            @test feed(grammar,s,G.EOS_ACTION)[4]==1
        end
    end
end
