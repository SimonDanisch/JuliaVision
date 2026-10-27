module MakieSceneAdapter
using JSON3

# A deliberately small declarative API, not arbitrary Julia evaluation. The
# compiler supplies Makie syntax; the model still chooses data, layout and style.
const SCHEMA = """
Return a JSON object only, with these fields:
size: [width,height]; axes: array of {id,row,col,title,xlabel,ylabel};
plots: ordered array of {kind,axis,x,y,color?,alpha?,label?} for kind lines or scatter,
or {kind:"band",axis,x,lower,upper,color?,alpha?,label?}.
Data expressions are input names, finite numbers, {op:"abs",arg:expr}, or
{op:"+"|"-"|"*"|"/",left:expr,right:expr}; operations are elementwise.
links: array of {dimension:"x"|"y",axes:[axis ids]};
legends: array of {axis,position:"lt"|"rt"|"lb"|"rb"};
returns: ordered array of "fig" and axis ids.
Use only supplied input names. Colors: blue,red,black,green,orange,gray,purple.
No extra fields. Optional text attributes may be omitted. No Julia code.
"""

fail(msg)=throw(ArgumentError(msg))
function fields(o,allowed,required=())
    o isa AbstractDict || fail("expected object")
    all(k->String(k) in allowed,keys(o)) || fail("unknown field")
    all(k->haskey(o,k),required) || fail("missing field")
end
function text(x)
    x isa AbstractString && length(x)<=256 || fail("expected bounded string")
    String(x)
end
function items(x,n)
    x isa AbstractVector && length(x)<=n || fail("array limit")
    x
end
function integer(x,lo,hi)
    x isa Integer && !(x isa Bool) && lo<=x<=hi || fail("integer range")
    Int(x)
end
function dataexpr(x,inputs,depth=0,budget=Ref(32))
    budget[]-=1
    depth<=4 && budget[]>=0 || fail("expression limit")
    if x isa AbstractString
        x in inputs || fail("unknown input")
        return String(x)
    elseif x isa Real && !(x isa Bool)
        isfinite(x) || fail("nonfinite literal")
        return repr(Float64(x))
    end
    fields(x,("op","arg","left","right"),("op",))
    op=x["op"]
    if op=="abs"
        fields(x,("op","arg"),("op","arg"))
        return "abs.("*dataexpr(x["arg"],inputs,depth+1,budget)*")"
    end
    op in ("+","-","*","/") || fail("unknown operator")
    fields(x,("op","left","right"),("op","left","right"))
    "("*dataexpr(x["left"],inputs,depth+1,budget)*" ."*op*" "*dataexpr(x["right"],inputs,depth+1,budget)*")"
end

function compile_scene(raw;inputs)
    ncodeunits(raw)<=16384 || fail("spec too large")
    length(inputs) in 1:8 && all(x->occursin(r"^[a-z][a-z0-9_]*$",x),inputs) || fail("invalid trusted inputs")
    length(unique(inputs))==length(inputs) || fail("duplicate inputs")
    o=JSON3.read(raw)
    fields(o,("size","axes","plots","links","legends","returns"),("size","axes","plots","returns"))
    sz=items(o["size"],2);length(sz)==2 || fail("size length")
    w,h=(integer(x,200,2000) for x in sz)
    lines=["using Makie", "function makeplot("*join(inputs,",")*")", "fig=Figure(size=($w,$h))"]
    ids=Dict{String,String}();cells=Set{Tuple{Int,Int}}()
    for (i,a) in enumerate(items(o["axes"],4))
        fields(a,("id","row","col","title","xlabel","ylabel"),("id","row","col"))
        id=text(a["id"])
        (haskey(ids,id) || id=="fig") && fail("duplicate/reserved axis")
        row=integer(a["row"],1,4);col=integer(a["col"],1,4)
        (row,col) in cells && fail("duplicate cell");push!(cells,(row,col))
        name="ax$i";ids[id]=name
        attrs=[k*"="*repr(text(a[k])) for k in ("title","xlabel","ylabel") if haskey(a,k)]
        push!(lines,name*"=Axis(fig[$row,$col]"*(isempty(attrs) ? "" : ";"*join(attrs,","))*")")
    end
    axis(id)=get(ids,text(id)) do;fail("unknown axis");end
    for p in items(o["plots"],16)
        fields(p,("kind","axis","x","y","lower","upper","color","alpha","label"),("kind","axis","x"))
        kind=p["kind"];kind in ("lines","scatter","band") || fail("unknown plot")
        names=kind=="band" ? ("x","lower","upper") : ("x","y")
        all(k->haskey(p,k),names) || fail("missing data")
        any(k->haskey(p,k),kind=="band" ? ("y",) : ("lower","upper")) && fail("extraneous data")
        args=[axis(p["axis"]);[dataexpr(p[k],inputs) for k in names]]
        attrs=String[]
        if haskey(p,"color") || haskey(p,"alpha")
            color=get(p,"color","blue");color in ("blue","red","black","green","orange","gray","purple") || fail("unknown color")
            c=repr(Symbol(color))
            if haskey(p,"alpha")
                alpha=p["alpha"];alpha isa Real && !(alpha isa Bool) && isfinite(alpha) && 0<=alpha<=1 || fail("invalid alpha")
                c="("*c*","*repr(Float64(alpha))*")"
            end
            push!(attrs,"color="*c)
        end
        haskey(p,"label") && push!(attrs,"label="*repr(text(p["label"])))
        push!(lines,String(kind)*"!("*join(args,",")*(isempty(attrs) ? "" : ";"*join(attrs,","))*")")
    end
    for link in items(get(o,"links",[]),8)
        fields(link,("dimension","axes"),("dimension","axes"))
        dim=link["dimension"];dim in ("x","y") || fail("unknown link")
        axes=items(link["axes"],4);length(axes)>=2 && length(unique(axes))==length(axes) || fail("invalid link")
        push!(lines,"link"*dim*"axes!("*join(axis.(axes),",")*")")
    end
    for legend in items(get(o,"legends",[]),4)
        fields(legend,("axis","position"),("axis","position"))
        position=legend["position"];position in ("lt","rt","lb","rb") || fail("unknown position")
        push!(lines,"axislegend("*axis(legend["axis"])*";position="*repr(Symbol(position))*")")
    end
    returns=items(o["returns"],5);isempty(returns) && fail("empty return")
    values=[r=="fig" ? "fig" : axis(r) for r in returns]
    push!(lines,"("*join(values,",")*",)");push!(lines,"end")
    join(lines,"\n")
end
end
