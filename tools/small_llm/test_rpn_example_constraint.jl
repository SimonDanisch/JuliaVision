include("rpn_example_constraint.jl")
using Test
function test_rpn_example_constraint()
    R=RPNPrefixGrammar;C=RPNExampleConstraint
    @testset "Public-example finish pruning retains shared token prefixes" begin
        # Every action shares its first token. Final token selects the action.
        codec(text)=Int32[1,findfirst(==(strip(text)),R.ACTIONS)+1]
        grammar=R.Grammar(codec,32,31)
        nf=C.nonfinish_nodes(grammar)
        for token in Int32(0):Int32(31)
            @test C.token_allowed(grammar.offsets,grammar.labels,grammar.targets,nf,Int32(0),token,false)
            @test C.committed_action(grammar.offsets,grammar.labels,grammar.targets,grammar.terminal_action,Int32(0),token)==0
        end
        for root in grammar.roots
            for first_token in Int32(0):Int32(31)
                allowed=C.token_allowed(grammar.offsets,grammar.labels,grammar.targets,nf,root,first_token,true)
                target=TokenPrefixTrie.edge_target(grammar.offsets,grammar.labels,grammar.targets,root,first_token)
                @test allowed==(target>0)
                pruned=C.token_allowed(grammar.offsets,grammar.labels,grammar.targets,nf,root,first_token,false)
                @test !pruned||allowed
            end
        end
        root=grammar.roots[2] # depth1, includes pushes/unaries/finish
        shared=TokenPrefixTrie.edge_target(grammar.offsets,grammar.labels,grammar.targets,root,Int32(1))
        @test C.token_allowed(grammar.offsets,grammar.labels,grammar.targets,nf,root,Int32(1),false)
        @test !C.token_allowed(grammar.offsets,grammar.labels,grammar.targets,nf,shared,Int32(18),false)
        @test C.token_allowed(grammar.offsets,grammar.labels,grammar.targets,nf,shared,Int32(18),true)
        @test C.committed_action(grammar.offsets,grammar.labels,grammar.targets,grammar.terminal_action,root,Int32(1))==0
        @test C.committed_action(grammar.offsets,grammar.labels,grammar.targets,grammar.terminal_action,shared,Int32(18))==R.FINISH
        @test C.committed_action(grammar.offsets,grammar.labels,grammar.targets,grammar.terminal_action,root,Int32(30))==-1
        forced=grammar.roots[5]
        @test !C.token_allowed(grammar.offsets,grammar.labels,grammar.targets,nf,forced,Int32(1),false)
        @test C.token_allowed(grammar.offsets,grammar.labels,grammar.targets,nf,forced,Int32(1),true)
        stack=zeros(Float32,16);stack[1]=1;stack[9]=2
        @test C.finish_allowed(stack,Float32[1,2],Int32[0,0],Int32(1),Int32(8),Int32(2),0f0,0f0)
        @test !C.finish_allowed(stack,Float32[1,3],Int32[0,0],Int32(1),Int32(8),Int32(2),0f0,0f0)
        @test !C.finish_allowed(stack,Float32[1,2],Int32[0,1],Int32(1),Int32(8),Int32(2),0f0,0f0)
        @test !C.finish_allowed(stack,Float32[],Int32[],Int32(1),Int32(8),Int32(0),0f0,0f0)
        @test !C.finish_allowed(stack,Float32[1,2],Int32[0,0],Int32(2),Int32(8),Int32(2),0f0,0f0)
    end
end
