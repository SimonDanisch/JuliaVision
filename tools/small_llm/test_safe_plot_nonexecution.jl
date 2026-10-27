include("safe_plot_julia_source.jl")
using Test

function test_safe_plot_nonexecution()
    @testset "Restricted plot compiler rejects active code without executing it" begin
        marker=Ref(0)
        base="""
        function makeplot(data)
            fig=Figure(size=(720,480),fontsize=16)
            ax=Axis(fig[1,1])
            lines!(ax,data["x"],data["a"];color="#0072B2",linewidth=2.5)
            return fig
        end
        """
        attacks=("exit()", "error(\"candidate executed\")", "run(`false`)",
            "Core.eval(Main, :(error(\"executed\")))", "Base.include(Main, \"missing.jl\")",
            "ccall(:abort,Cvoid,())", "open(\"/tmp/never-plot-write\",\"w\")",
            "global marker=1", "marker[]=1", "@eval error(\"executed\")",
            "@async error(\"executed\")", "while true; end", "for i=1:100; end",
            "try error(\"x\") catch; end", "let; error(\"x\"); end",
            "[error(\"x\") for i=1:2]", "getfield(Main,:eval)(:(error(\"x\")))")
        @test SafePlotJuliaSource.lower(base).accepted
        for attack in attacks
            @test !SafePlotJuliaSource.lower(replace(base,"return fig"=>attack*";return fig")).accepted
            @test marker[]==0
        end
        for depth in (65,128,256)
            nested=repeat("(",depth)*"data"*repeat(",)",depth)
            @test !SafePlotJuliaSource.lower(replace(base,"return fig"=>"return "*nested)).accepted
        end
        @test !SafePlotJuliaSource.lower(replace(base,"data[\"a\"]"=>"data[Symbol(\"a\")]")).accepted
        @test !SafePlotJuliaSource.lower(replace(base,"lines!"=>"Base.invokelatest")).accepted
    end
end
