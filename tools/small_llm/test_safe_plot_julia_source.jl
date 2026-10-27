include("safe_plot_julia_source.jl")
using Test
function test_safe_plot_julia_source()
    source="""
    using CairoMakie
    function makeplot(data)
        fig=Figure(size=(720,480),fontsize=16)
        ax=Axis(fig[1,1],title="Signal",xlabel="x",ylabel="y")
        lines!(ax,data["x"],data["a"];color="#0072B2",linewidth=2.5,label="A")
        axislegend(ax;position=:rt,framevisible=false)
        return fig
    end
    """
    @testset "Julia plot source becomes checked IR, never eval" begin
        result=SafePlotJuliaSource.lower(source)
        @test result.accepted
        @test result.spec.size==(720,480)
        @test result.spec.axes[1].title=="Signal"
        @test result.spec.series[1].y=="a"
        @test result.spec.legend
        for invalid in (replace(source,"lines!"=>"run"),replace(source,"data[\"a\"]"=>"read(\"/private\")"),
                replace(source,"fontsize=16"=>"fontsize=100000"),replace(source,"linewidth=2.5"=>"linewidth=4"),
                replace(source,"color=\"#0072B2\""=>"color=:blue"),
                replace(source,"return fig"=>"return Main"),replace(source,"data)"=>"data::Dict)"),
                replace(source,"return fig"=>"@eval evil();return fig"),
                replace(source,"return fig"=>"for i=1:2;end;return fig"),
                replace(source,"fig=Figure"=>"Base=Figure"))
            @test !SafePlotJuliaSource.lower(invalid).accepted
        end
        @test !SafePlotJuliaSource.lower(repeat("x",32769)).accepted
        @test !SafePlotJuliaSource.lower("function (").accepted
    end
    source
end
