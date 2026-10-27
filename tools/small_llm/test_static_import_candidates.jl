include("static_import_candidates.jl")
using Test
function test_static_import_candidates()
    corpus=(;modules=["Base","Dates","Statistics","Other"],
        bindings=(;Base=["println","sin","Vector","eval","include","include_string"]),
        rows=[(;qualified=x) for x in ("Base.sin","Dates.DateTime","Statistics.mean","Other.mean")])
    check(s)=StaticImportCandidates.analyze(s,corpus)
    @testset "Static import candidates are conservative, non-executing hints" begin
        @test only(check("DateTime(x)").suggestions)==(;name="DateTime",possible_modules=["Dates"])
        @test only(check("mean(x)").suggestions).possible_modules==["Other","Statistics"]
        for source in ("using Dates;DateTime(x)","using Dates: DateTime;DateTime(x)",
                "import Dates: DateTime;DateTime(x)","using Dates: DateTime as DT;DT(x)",
                "import Dates as D;D.DateTime(x)","import Dates;Dates.DateTime(x)",
                "sin(x);println(x);Vector(x)","unknown_local_function(x)",
                "DateTime(x)=x;DateTime(y)","function DateTime(x);x;end;DateTime(y)",
                "f(DateTime)=DateTime(x)","f(;DateTime=x)=DateTime(y)",
                "DateTime=callable;DateTime(x)","let DateTime=x;DateTime(y);end",
                "DateTime::Any=callable;DateTime(x)","global DateTime;DateTime(x)",
                "function f();local DateTime;DateTime(x);end",
                "try f();catch DateTime;DateTime(x);end",
                "struct DateTime end;DateTime(x)","abstract type DateTime end;DateTime(x)",
                "struct DateTime{T} end;DateTime(x)",
                "(DateTime->DateTime(x))(y)","f(x) where DateTime=DateTime(x)",
                "\"DateTime(x)\"",":(DateTime(x))","# DateTime(x)\nsin(y)")
            result=check(source)
            @test result.source_parse_ok
            @test isempty(result.suggestions)
        end
        @test length(check("import Dates;DateTime(x)").suggestions)==1 # import isn't using
        for source in ("using Uncaptured;DateTime(x)","using Foo.Dates;DateTime(x)",
                "using .Dates;DateTime(x)","using Uncaptured: f;DateTime(x)",
                "import Uncaptured as U;DateTime(x)")
            result=check(source)
            @test !isempty(result.unknown_imports)
            @test isempty(result.suggestions)
        end
        for source in ("@some_macro DateTime(x)","eval(x);DateTime(x)",
                "include(x);DateTime(x)","include_string(m,x);DateTime(x)",
                "Core.eval(m,x);DateTime(x)","Base.include(m,x);DateTime(x)",
                "module M;DateTime(x);end")
            result=check(source)
            @test result.opaque_metaprogramming
            @test isempty(result.suggestions)
        end
        result=check("function (")
        @test !result.source_parse_ok
        @test isempty(result.suggestions)
        @test !check("using Dates as D;D.DateTime(x)").source_parse_ok
        @test_throws ArgumentError check(repeat("x",262145))
        @test_throws ArgumentError check(repeat("begin\n",130)*"DateTime(x)\n"*repeat("end\n",130))
    end
end
