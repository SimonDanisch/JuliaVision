isdefined(@__MODULE__,:SafePlotSpec)||include("safe_plot_spec.jl")
module SafePlotBuilder
import ..SafePlotSpec
import Makie

"""Trusted shared Makie construction; rendering backend chosen by caller.
No generated Julia code is executed. Safety is the narrow schema, not a
claim that the requested plot is scientifically correct or attractive.
"""
function build(spec::SafePlotSpec.Spec,data)
    SafePlotSpec.validate_data(spec,data)
    figure=Makie.Figure(;size=spec.size,fontsize=16)
    axes=[Makie.Axis(figure[1,i];title=a.title,xlabel=a.xlabel,ylabel=a.ylabel,
        xscale=a.xscale=="log10" ? log10 : identity,yscale=a.yscale=="log10" ? log10 : identity)
        for (i,a) in enumerate(spec.axes)]
    plots=Any[]
    for series in spec.series
        axis=axes[series.axis];x=data[series.x];color=SafePlotSpec.PALETTE[series.color]
        plot=if series.kind=="line"
            Makie.lines!(axis,x,data[series.y];label=series.label,color,linewidth=2.5)
        elseif series.kind=="scatter"
            Makie.scatter!(axis,x,data[series.y];label=series.label,color,markersize=9)
        else
            Makie.band!(axis,x,data[series.low],data[series.high];label=series.label,color=(color,0.25))
        end
        push!(plots,plot)
    end
    spec.linkx&&Makie.linkxaxes!(axes...)
    for (i,axis) in enumerate(axes)
        spec.legend&&any(s->s.axis==i&&!isempty(s.label),spec.series)&&Makie.axislegend(axis;position=spec.axes[i].legend_position,framevisible=false)
    end
    (;figure,axes,plots)
end
end
