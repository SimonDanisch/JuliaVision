"""Small declarative plotting interface, not an arbitrary Julia interpreter.
No eval, imports, callbacks, function names or expressions come from the input.
This development interface is NOT yet a measured model-quality improvement.
"""
module SafePlotSpec
using JSON3
const PALETTE=Dict("blue"=>"#0072B2","orange"=>"#E69F00","green"=>"#009E73","red"=>"#D55E00")
struct AxisSpec
    title::String
    xlabel::String
    ylabel::String
    xscale::String
    yscale::String
    legend_position::Symbol
end
struct SeriesSpec
    kind::String
    axis::Int
    x::String
    y::String
    low::String
    high::String
    label::String
    color::String
end
struct Spec
    size::Tuple{Int,Int}
    axes::Vector{AxisSpec}
    series::Vector{SeriesSpec}
    legend::Bool
    linkx::Bool
end

function fields(value,required,optional=String[])
    value isa JSON3.Object||throw(ArgumentError("JSON object required"))
    keys=Set(String.(Base.keys(value)))
    Set(required)⊆keys&&keys⊆Set(vcat(required,optional))||throw(ArgumentError("Unknown or missing fields"))
end
function text(value)
    value isa AbstractString&&ncodeunits(value)<=512&&all(!iscntrl,value)||throw(ArgumentError("Bounded plain label required"))
    String(value)
end
function reference(value)
    value isa AbstractString&&occursin(r"^[A-Za-z][A-Za-z0-9_]{0,31}$",value)||throw(ArgumentError("Data reference name"))
    String(value)
end

function parse_spec(source::AbstractString)
    ncodeunits(source)<=32768||throw(ArgumentError("Bounded JSON source"))
    depth=0;quoted=false;escaped=false
    for character in source
        if quoted
            if escaped
                escaped=false
            elseif character=='\\'
                escaped=true
            elseif character=='"'
                quoted=false
            end
        elseif character=='"'
            quoted=true
        elseif character in ('{','[')
            depth+=1;depth<=16||throw(ArgumentError("Bounded JSON nesting"))
        elseif character in ('}',']')
            depth-=1;depth>=0||throw(ArgumentError("Unbalanced JSON"))
        end
    end
    object=JSON3.read(source);fields(object,["size","axes","series","legend"],["linkx"])
    object.size isa AbstractVector&&length(object.size)==2&&all(x->x isa Integer&&!(x isa Bool),object.size)||throw(ArgumentError("Integer size pair"))
    width,height=Int.(object.size)
    400<=width<=1600&&300<=height<=1200||throw(ArgumentError("Bounded figure size"))
    object.axes isa AbstractVector&&1<=length(object.axes)<=2||throw(ArgumentError("One or two horizontal axes"))
    axes=map(object.axes) do axis
        fields(axis,["title","xlabel","ylabel","xscale","yscale"],["legend_position"])
        axis.xscale in ("linear","log10")&&axis.yscale in ("linear","log10")||throw(ArgumentError("Known scales only"))
        position=get(axis,:legend_position,"rt")
        position in ("rt","lt","rb","lb")||throw(ArgumentError("Known legend position"))
        AxisSpec(text(axis.title),text(axis.xlabel),text(axis.ylabel),String(axis.xscale),String(axis.yscale),Symbol(position))
    end
    object.series isa AbstractVector&&1<=length(object.series)<=8||throw(ArgumentError("One to eight series"))
    series=map(object.series) do row
        common=["kind","axis","x","label","color"]
        row isa JSON3.Object&&haskey(row,:kind)&&row.kind in ("line","scatter","band")||throw(ArgumentError("Known plot kind"))
        fields(row,vcat(common,row.kind=="band" ? ["low","high"] : ["y"]))
        row.axis isa Integer&&!(row.axis isa Bool)&&1<=row.axis<=length(axes)||throw(ArgumentError("Axis index"))
        row.color isa AbstractString&&haskey(PALETTE,row.color)||throw(ArgumentError("Known palette entry"))
        SeriesSpec(String(row.kind),Int(row.axis),reference(row.x),row.kind=="band" ? "" : reference(row.y),
            row.kind=="band" ? reference(row.low) : "",row.kind=="band" ? reference(row.high) : "",text(row.label),String(row.color))
    end
    object.legend isa Bool&&get(object,:linkx,false) isa Bool||throw(ArgumentError("Boolean legend/linkx"))
    linkx=get(object,:linkx,false)
    !linkx||length(axes)==2||throw(ArgumentError("Linking requires two axes"))
    !linkx||axes[1].xscale==axes[2].xscale||throw(ArgumentError("Linked axes require matching x scales"))
    Spec((width,height),axes,series,object.legend,linkx)
end

"""Validate supplied HOST data without modifying it or evaluating expressions."""
function validate_data(spec::Spec,data)
    vector(name)=begin
        value=get(data,name,nothing)
        value isa AbstractVector&&1<=length(value)<=100000||throw(ArgumentError("Bounded nonempty host vector: $name"))
        all(x->x isa Real&&!(x isa Bool)&&isfinite(x),value)||throw(ArgumentError("Finite real data: $name"))
        value
    end
    for series in spec.series
        axis=spec.axes[series.axis];x=vector(series.x)
        ys=series.kind=="band" ? (vector(series.low),vector(series.high)) : (vector(series.y),)
        all(y->length(y)==length(x),ys)||throw(DimensionMismatch("Series data lengths"))
        axis.xscale=="log10"&&any(v->v<=0,x)&&throw(ArgumentError("Nonpositive logarithmic x"))
        axis.yscale=="log10"&&any(y->any(v->v<=0,y),ys)&&throw(ArgumentError("Nonpositive logarithmic y"))
        series.kind=="band"&&any(t->t[1]>t[2],zip(ys...))&&throw(ArgumentError("Band low exceeds high"))
    end
    nothing
end
end
