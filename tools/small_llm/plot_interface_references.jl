isdefined(@__MODULE__,:SafePlotSpec)||include("safe_plot_spec.jl")
using JSON3
# PRIVATE expected specifications. Never transfer this file to model workers.
function plot_interface_references()
    axis(title,xlabel,ylabel;xs="linear",ys="linear",position="rt")=
        (;title,xlabel,ylabel,xscale=xs,yscale=ys,legend_position=position)
    series(kind,y,label,color;panel=1)=kind=="band" ?
        (;kind,axis=panel,x="x",low="lo",high="hi",label,color) : (;kind,axis=panel,x="x",y,label,color)
    spec(size,axes,series;legend=true,linkx=false)=(;size,axes,series,legend,linkx)
    [
        spec([720,480],[axis("Calibration","Time (s)","Value")],[series("line","a","Sensor A","blue")]),
        spec([800,500],[axis("Samples","Time","Score";position="lt")],[series("scatter","b","Measured B","green")]),
        spec([900,500],[axis("Envelope check","Time","Signal")],[series("band","","Range","blue"),series("line","a","Center","orange")]),
        spec([1000,480],[axis("Two sensors","Time","Value"),axis("Replicates","Time","Reading";position="lt")],
            [series("line","a","A","blue"),series("line","b","B","red"),series("scatter","b","Cloud","green";panel=2)];linkx=true),
        spec([760,500],[axis("Positive signals","Time","Response";ys="log10",position="lb")],[series("line","b","B first","green"),series("line","a","A second","red")]),
        spec([1000,500],[axis("Log-x A","Input","Value";xs="log10",position="lb"),axis("Log-x B","Input","Value";xs="log10")],
            [series("scatter","a","Raw A","blue"),series("line","b","Raw B","orange";panel=2)];linkx=true),
        spec([720,480],[axis("No legend","x","Reading")],[series("band","","","blue"),series("line","a","","red")];legend=false),
        spec([800,500],[axis("Double log","Input","Positive output";xs="log10",ys="log10",position="rb")],
            [series("scatter","a","Samples","orange"),series("line","b","Trend","green")])
    ]
end

function plot_specs_equivalent(a,b)
    a.size==b.size&&a.linkx==b.linkx&&length(a.axes)==length(b.axes)&&length(a.series)==length(b.series)||return false
    for i in eachindex(a.axes)
        x,y=a.axes[i],b.axes[i]
        (x.title,x.xlabel,x.ylabel,x.xscale,x.yscale)==(y.title,y.xlabel,y.ylabel,y.xscale,y.yscale)||return false
        visible_a=a.legend&&any(s->s.axis==i&&!isempty(s.label),a.series)
        visible_b=b.legend&&any(s->s.axis==i&&!isempty(s.label),b.series)
        visible_a==visible_b||return false
        visible_a&&x.legend_position!=y.legend_position&&return false
    end
    fields(s)=(s.kind,s.axis,s.x,s.y,s.low,s.high,s.label,s.color)
    all(fields(x)==fields(y) for (x,y) in zip(a.series,b.series))
end

"""Trusted reference emitter for qualification ONLY, never shown to the model."""
function plot_interface_native_reference(spec)
    io=IOBuffer();println(io,"using CairoMakie\nfunction makeplot(data)\n fig=Figure(size=",repr(spec.size),",fontsize=16)")
    for (i,axis) in enumerate(spec.axes)
        println(io," ax",i,"=Axis(fig[1,",i,"];title=",repr(axis.title),",xlabel=",repr(axis.xlabel),",ylabel=",repr(axis.ylabel),
            ",xscale=",axis.xscale=="log10" ? "log10" : "identity",",yscale=",axis.yscale=="log10" ? "log10" : "identity",")")
    end
    for s in spec.series
        name=s.kind=="line" ? "lines!" : s.kind=="scatter" ? "scatter!" : "band!"
        arguments=s.kind=="band" ? [s.x,s.low,s.high] : [s.x,s.y]
        print(io," ",name,"(ax",s.axis,",",join(["data["*repr(x)*"]" for x in arguments],","),";label=",repr(s.label),",color=")
        color=SafePlotSpec.PALETTE[s.color]
        print(io,s.kind=="band" ? "("*repr(color)*",0.25)" : repr(color))
        s.kind=="line"&&print(io,",linewidth=2.5")
        s.kind=="scatter"&&print(io,",markersize=9")
        println(io,")")
    end
    if spec.legend
        for (i,a) in enumerate(spec.axes)
            any(s->s.axis==i&&!isempty(s.label),spec.series)&&println(io," axislegend(ax",i,";position=:",a.legend_position,",framevisible=false)")
        end
    end
    spec.linkx&&println(io," linkxaxes!(ax1,ax2)")
    println(io," return fig\nend");String(take!(io))
end
