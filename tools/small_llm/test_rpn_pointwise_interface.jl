include("rpn_pointwise_interface.jl")
using Test
function test_rpn_pointwise_interface()
    R=RPNPointwiseInterface
    @testset "Finite postfix program admission, no arithmetic repair" begin
        for action in RPNPrefixGrammar.ACTIONS[1:9]
            r=R.lower(action*"\nfinish\n")
            @test r.accepted
            @test r.certificate.operations==1
        end
        for action in ("neg","square")
            @test R.lower("x0\n"*action*"\nfinish").accepted
        end
        for action in ("add","sub","mul","min","max")
            @test R.lower("x_prev_periodic\nx0\n"*action*"\nfinish").accepted
        end
        left=R.lower("x0\nx_prev_periodic\nsub\nfinish")
        right=R.lower("x_prev_periodic\nx0\nsub\nfinish")
        @test left.accepted && right.accepted
        @test left.code!=right.code # never canonicalize/reverse subtraction
        @test occursin("mod1",left.code)
        @test R.lower("x0\nfinish\n<|im_end|>unscored suffix").code==R.lower("x0\nfinish").code
        for bad in ("","finish","x0","x0\nx0\nfinish","x0\nfinish\nx0\nfinish",
                    "add\nfinish","x0\nunknown\nfinish","read(secret)\nfinish","```julia\nx0\nfinish\n```",
                    "x0\nx0\nx0\nadd\nadd\nfinish")
            @test !R.lower(bad;max_depth=2).accepted
        end
        @test !R.lower("x0\nneg\nfinish";budget=1).accepted
        @test R.lower("x0\nx0\nsub\nfinish";budget=2).accepted # final reduction permitted
        @test !R.lower("x0\nfinish";budget=true).accepted
        @test !R.lower("x0\nfinish";max_depth=1).accepted
        @test !R.lower(repeat("x",16385)).accepted
    end
end
