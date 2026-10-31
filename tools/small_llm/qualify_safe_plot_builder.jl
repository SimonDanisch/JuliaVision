isdefined(@__MODULE__,:JuliaTaskEval)||include("julia_eval.jl")
include("julia_render.jl")
using JSON3,SHA

"""CPU-only, networkless trusted-reference checks/render; NO model attempts."""
function qualify_safe_plot_builder(;destination=joinpath(@__DIR__,"safe_plot_builder_validation.json"))
    isfile(destination)&&error("Preserve reference qualification")
    spec=(;size=[960,480],legend=true,linkx=true,
        axes=[(;title="Signal and envelope",xlabel="Time",ylabel="Amplitude",xscale="linear",yscale="linear",legend_position="rt"),
            (;title="Positive response",xlabel="Time",ylabel="Response (log scale)",xscale="linear",yscale="log10",legend_position="lt")],
        series=[(;kind="band",axis=1,x="x",low="lo",high="hi",label="Envelope",color="blue"),
            (;kind="line",axis=1,x="x",y="a",label="Signal",color="blue"),
            (;kind="scatter",axis=2,x="x",y="b",label="Response",color="orange")])
    literal=repr(JSON3.write(spec))
    source="import Makie\n"*read(joinpath(@__DIR__,"safe_plot_spec.jl"),String)*"\n"*read(joinpath(@__DIR__,"safe_plot_builder.jl"),String)*"\n"*"""
    function demo_data()
        x=collect(range(0.1,2pi;length=64));a=sin.(x)
        Dict("x"=>x,"a"=>a,"lo"=>a.-0.15,"hi"=>a.+0.15,"b"=>exp.(x./4))
    end
    function reference_plot()
        SafePlotBuilder.build(SafePlotSpec.parse_spec($literal),demo_data()).figure
    end
    """
    tests="""
    data=demo_data();original=deepcopy(data)
    spec=SafePlotSpec.parse_spec($literal)
    built=SafePlotBuilder.build(spec,data)
    @test length(built.axes)==2
    @test length(built.plots)==3
    @test built.axes[1].title[]=="Signal and envelope"
    @test built.axes[2].ylabel[]=="Response (log scale)"
    @test built.axes[2].yscale[]===log10
    @test built.plots[1] isa Makie.Band
    @test built.plots[2] isa Makie.Lines
    @test built.plots[3] isa Makie.Scatter
    @test built.plots[2].label[]=="Signal"
    @test data==original
    bad=deepcopy(data);bad["b"][1]=0.0
    @test_throws ArgumentError SafePlotBuilder.build(spec,bad)
    """
    task=(;id="safe_plot_builder_reference",category="makie",tests)
    semantic=JuliaTaskEval.evaluate(task,source)
    semantic.passed||error("Reference semantics failed: "*semantic.diagnostic)
    image=joinpath(@__DIR__,"safe_plot_builder_artifacts/reference.png")
    render=render_julia_candidate(source,"reference_plot()";artifact_path=image)
    render.rendered||error("Reference render failed: "*render.diagnostic)
    render.dimensions==(;width=960,height=480)||error("Unexpected image dimensions")
    render.layout!==nothing&&length(render.layout.axes)==2&&all(a->a.within_canvas,render.layout.axes)||error("Reference decorations clipped")
    result=(;complete=true,semantic_checks=11,render,semantic,
        scope="Trusted declarative Makie builder CPU-only semantic/reference-render qualification, NOT neural answer quality, arbitrary Julia admission or an aesthetic score",
        public_reference_spec=spec,
        sources=Dict(file=>bytes2hex(sha256(read(joinpath(@__DIR__,file)))) for file in
            ("safe_plot_spec.jl","safe_plot_builder.jl","qualify_safe_plot_builder.jl","julia_render.jl","julia_eval.jl","julia_sandbox.sh")))
    open(io->JSON3.pretty(io,result),destination,"w");result
end
