isdefined(@__MODULE__,:JuliaTaskEval) || include("julia_eval.jl")
using JSON3, SHA

"""Capture generic runtime interface facts, not solutions or expected task outputs.

Run trusted fixture code only in the headless sandbox. Record every input name,
not a hand-picked subset matching the failed task. Keep separate from frozen
source-only/runtime-error studies; this artifact has no model-quality result.
"""
function capture_makie_public_interface()
    output=joinpath(@__DIR__,"makie_public_interface.json")
    isfile(output) && error("public interface already captured; version new captures explicitly")
    program="""
        f=Figure();a=Axis(f[1,1]);b=Axis(f[1,2]);c=Axis(f[2,1])
        plots=(scatter!(a,[2.,4.],[3.,7.];color=[0.2,0.8]),
            lines!(b,[2.,4.],[3.,7.]),heatmap!(c,[1. 2.;3. 4.]))
        rows=map(plots) do plot
            graph=plot.attributes
            (;plot_type=string(typeof(plot)),
                mutable_inputs=sort!([(;name=String(key),storage_type=string(typeof(node))) for (key,node) in graph.inputs];by=row->row.name),
                readable_nodes=sort!(String.(collect(keys(graph.outputs)))))
        end
        println("PUBLIC_MAKIE_INTERFACE=",JSON3.write(rows))
        """
    fixture=(;id="public_makie_interface",category="makie",setup="using CairoMakie, JSON3",tests=program)
    result=JuliaTaskEval.evaluate(fixture,"")
    result.passed || error("public interface capture failed: "*result.diagnostic)
    prefix="PUBLIC_MAKIE_INTERFACE="
    lines=filter(line->startswith(line,prefix),split(result.diagnostic,'\n'))
    length(lines)==1 || error("missing/ambiguous public interface packet")
    rows=JSON3.read(only(lines)[length(prefix)+1:end])
    length(rows)==3 && all(row->!isempty(row.mutable_inputs),rows) || error("empty interface capture")
    report=(;scope="trusted generic Scatter/Lines/Heatmap runtime interface facts, no model output, task assertions, reference solution or correctness labels",
        observation="ComputePipeline _update! checks graph.inputs; readable derived nodes need not be mutable input keywords",
        source_url="https://github.com/MakieOrg/Makie.jl",
        fixture_sha256=bytes2hex(sha256(fixture.setup*program)),
        capture_source_sha256=bytes2hex(sha256(read(@__FILE__))),
        sandbox_sha256=bytes2hex(sha256(read(joinpath(@__DIR__,"julia_sandbox.sh")))),rows)
    open(io->JSON3.pretty(io,report),output,"w");report
end
