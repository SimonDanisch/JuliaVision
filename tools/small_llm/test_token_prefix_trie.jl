include("token_prefix_trie.jl")
using Test, Random
function test_token_prefix_trie()
    T = TokenPrefixTrie
    @testset "Sparse prefix constraints; not semantic knowledge" begin
        t = T.Trie([[0, 2, 3], [0, 2, 4], [5, 1]], 8)
        @test length(t.labels) == length(t.terminal) - 1
        @test T.next_node(t, 1, 7) == -1
        @test T.next_node(t, -1, 0) == -1
        @test T.next_node(t, 99, 0) == -1
        @test T.next_node(t, 0, 7) == 0
        n = T.next_node(t, 1, 0)
        n = T.next_node(t, n, 2)
        @test n > 0
        @test T.next_node(t, n, 3) == 0
        @test T.next_node(t, n, 4) == 0
        @test T.next_node(t, n, 2) == -1
        @test T.constrained_logit(t.offsets,t.labels,t.targets,Int32(1),Int32(0),2f0) == 2f0
        @test T.constrained_logit(t.offsets,t.labels,t.targets,Int32(1),Int32(1),2f0) == -Inf32
        @test isnan(T.constrained_logit(t.offsets,t.labels,t.targets,Int32(0),Int32(1),NaN32))
        for bad in ([], [[]], [[1],[1]], [[1],[1,2]], [[1,2],[1]], [[-1]], [[8]], [[true]], [[1.5]])
            @test_throws ArgumentError T.Trie(bad,8)
        end
        @test_throws ArgumentError T.Trie([[0]],true)
        @test_throws ArgumentError T.Trie([[0]],0)
        # Exhaustive agreement with a simple prefix-list oracle at each depth.
        rng = MersenneTwister(20261025)
        paths = unique([rand(rng,0:30,4) for _ in 1:150])
        trie = T.Trie(paths,31)
        for path in paths
            node = Int32(1)
            for depth in 1:4
                for token in 0:30
                    allowed = any(p -> p[1:depth-1] == path[1:depth-1] && p[depth] == token,paths)
                    found = T.edge_target(trie.offsets,trie.labels,trie.targets,node,Int32(token)) > 0
                    @test found == allowed
                end
                node = T.next_node(trie,node,path[depth])
            end
            @test node == 0
        end
    end
end
