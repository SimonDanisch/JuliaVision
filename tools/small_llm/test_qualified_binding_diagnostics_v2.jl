include("qualified_binding_diagnostics_v2.jl")
using Test
function test_qualified_binding_diagnostics_v2()
    inventory=Dict("JSON3"=>["read","write","Object"],"Base64"=>["base64encode"])
    check(source)=QualifiedBindingDiagnosticsV2.analyze(source,inventory)
    @testset "Shadowed module names are unknown, not missing global APIs" begin
        for code in ("using JSON3;JSON3.parse(x)","f(x::JSON3.Object)=JSON3.parse(x)")
            r=check(code)
            @test r.source_parse_ok
            @test r.references_to_unlisted_bindings==["JSON3.parse"]
        end
        for code in ("f(JSON3)=JSON3.parse(x)","f(JSON3)::Any=JSON3.parse(x)",
                "f(x) where JSON3 = JSON3.parse(x)","let JSON3;JSON3.parse(x);end",
                "let JSON3=other;JSON3.parse(x);end","import Foo as JSON3;JSON3.parse(x)",
                "using Foo: JSON3;JSON3.parse(x)","import Foo: JSON3;JSON3.parse(x)",
                "import Foo.JSON3;JSON3.parse(x)","struct JSON3 end;JSON3.parse(x)",
                "abstract type JSON3 end;JSON3.parse(x)","primitive type JSON3 8 end;JSON3.parse(x)",
                "function f(;JSON3=other);JSON3.parse(x);end")
            r=check(code)
            @test r.source_parse_ok
            @test isempty(r.references_to_unlisted_bindings)
            @test "JSON3" in r.conservatively_skipped_modules
        end
        r=check("f(JSON3)=JSON3.parse(x);Base64.encode(x)")
        @test r.references_to_unlisted_bindings==["Base64.encode"]
        @test isempty(check("JSON3.read(x);Base64.base64encode(x)").references_to_unlisted_bindings)
        r=check("function (")
        @test !r.source_parse_ok
        @test !isempty(r.conservatively_skipped_modules)
        @test_throws ArgumentError check(repeat("x",262145))
        @test_throws ArgumentError check(repeat("begin\n",130)*"JSON3.parse(x)\n"*repeat("end\n",130))
    end
end
