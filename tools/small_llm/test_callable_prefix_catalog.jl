include("callable_prefix_catalog.jl")
using Test
function test_callable_prefix_catalog()
    C=CallablePrefixCatalog
    inventory=(;complete=true,rows=[(;qualified="Makie.Figure",kind="type"),(;qualified="CairoMakie.activate!",kind="function")])
    @testset "Whole-header token catalog" begin
        catalog=C.build(inventory,x->Int.(codeunits(x)),256)
        @test catalog.names==["CairoMakie.activate!","Makie.Figure"]
        @test catalog.headers==["API: CairoMakie.activate!\n","API: Makie.Figure\n"]
        for sequence in catalog.sequences
            node=Int32(1)
            for (i,token) in enumerate(sequence)
                node=TokenPrefixTrie.next_node(catalog.trie,node,token)
                @test i==length(sequence) ? node==0 : node>0
            end
        end
        @test_throws ArgumentError C.build((;complete=false,rows=inventory.rows),identity,256)
        @test_throws ArgumentError C.build((;complete=true,rows=[(;qualified="Makie.bad()",kind="function")]),identity,256)
        @test_throws ArgumentError C.build((;complete=true,rows=[(;qualified="Makie.Figure",kind="value")]),identity,256)
        @test_throws ArgumentError C.build((;complete=true,rows=[inventory.rows[1],inventory.rows[1]]),identity,256)
        @test_throws ArgumentError C.build(inventory,x->[0],256) # tokenizer collision
    end
end
