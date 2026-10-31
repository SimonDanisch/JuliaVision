isdefined(@__MODULE__,:SafePlotSpec)||include("safe_plot_spec.jl")
module SafePlotJuliaSource
import ..SafePlotSpec
using JSON3
struct FigureRef end
struct CellRef
    column::Int
end
struct AxisRef
    index::Int
end
struct DataRoot end
struct DataRef
    name::String
end
const CALLS=Set((:Figure,:Axis,:lines!,:scatter!,:band!,:axislegend,:linkxaxes!))

"""Compile a deliberately restricted ordinary-looking Julia plot program.

NEVER evals/loads candidate source. Supports one untyped makeplot(data), simple
assignments/calls/return, literal figure cells, supplied data refs and the seven
listed Makie constructors. No loops, macros, reflection, I/O or arbitrary Julia.
Only exact supported styles lower; do not silently repair API names/arithmetic.
This is an interface experiment, not a general Julia interpreter or quality gain.
"""
function lower(source::AbstractString)
    reject(reason)=(;accepted=false,reason,spec=nothing)
    ncodeunits(source)<=32768||return reject("bounded source required")
    try
        tree=Meta.parseall(source);stack=Any[(tree,0)];nodes=0
        while !isempty(stack)
            node,depth=pop!(stack);node isa Expr||continue
            nodes+=1;nodes<=3000&&depth<=64||throw(ArgumentError("bounded AST"))
            node.head in (:error,:incomplete,:macrocall,:quote,:module,:baremodule)&&throw(ArgumentError("unsupported syntax"))
            append!(stack,[(x,depth+1) for x in node.args if x isa Expr])
        end
        statements(x)=x isa Expr&&x.head in (:toplevel,:block) ? filter(y->!(y isa LineNumberNode),x.args) : Any[x]
        top=Any[]
        for node in statements(tree)
            if node isa Expr&&node.head in (:using,:import)
                sprint(Base.show_unquoted,node) in ("using CairoMakie","import CairoMakie","using Makie","import Makie")||throw(ArgumentError("unsupported import"))
            else
                push!(top,node)
            end
        end
        length(top)==1||throw(ArgumentError("one function required"))
        definition=only(top)
        definition isa Expr&&definition.head==:function||throw(ArgumentError("long-form function required"))
        signature,body=definition.args
        signature isa Expr&&signature.head==:call&&length(signature.args)==2&&signature.args[1]==:makeplot&&signature.args[2] isa Symbol||throw(ArgumentError("untyped makeplot(data) required"))
        argument=signature.args[2]
        reserved=union(CALLS,Set((:Makie,:CairoMakie,:Base,:Core,:Main,:nothing,:identity,:log10)))
        argument∉reserved||throw(ArgumentError("reserved argument name"))
        env=Dict{Symbol,Any}(argument=>DataRoot());axes=Any[];series=Any[];legends=Dict{Int,Symbol}()
        size=nothing;linked=false;returned=nothing
        function callee(node)
            node isa Symbol&&node in CALLS&&return node
            if node isa Expr&&node.head==Symbol(".")&&length(node.args)==2&&node.args[1] in (:Makie,:CairoMakie)
                name=node.args[2]
                name isa QuoteNode&&name.value in CALLS&&return name.value
            end
            throw(ArgumentError("unknown constructor/call"))
        end
        allowed(kw,names)=all(k->k in names,keys(kw))||throw(ArgumentError("unsupported keyword"))
        plain(x)=x isa AbstractString ? String(x) : throw(ArgumentError("plain string required"))
        dataref(x)=x isa DataRef ? x.name : throw(ArgumentError("supplied data reference required"))
        axisref(x)=x isa AxisRef ? x.index : throw(ArgumentError("declared axis required"))
        function color(x,band)
            if band
                x isa Tuple&&length(x)==2&&x[2] isa Real&&x[2]==0.25||throw(ArgumentError("explicit band alpha0.25 required"))
                x=x[1]
            end
            matches=[name for (name,value) in SafePlotSpec.PALETTE if x==value]
            length(matches)==1||throw(ArgumentError("exact published palette hex color required"))
            only(matches)
        end
        function value(node)
            node isa Union{AbstractString,Real}&&return node
            node isa QuoteNode&&node.value isa Symbol&&return node.value
            if node isa Symbol
                node==:nothing&&return nothing
                node in (:identity,:log10)&&return node
                haskey(env,node)||throw(ArgumentError("unbound local"))
                return env[node]
            end
            node isa Expr||throw(ArgumentError("unsupported value"))
            node.head==:tuple&&return Tuple(value(x) for x in node.args)
            if node.head==:ref
                base=value(first(node.args));indices=[value(x) for x in node.args[2:end]]
                base isa DataRoot&&length(indices)==1&&indices[1] isa AbstractString&&return DataRef(plain(indices[1]))
                base isa FigureRef&&length(indices)==2&&indices[1]===1&&indices[2] isa Int&&1<=indices[2]<=2&&return CellRef(indices[2])
                throw(ArgumentError("only declared figure cells/data references"))
            end
            node.head==:call||throw(ArgumentError("unsupported expression"))
            name=callee(first(node.args));pos=Any[];kw=Dict{Symbol,Any}()
            function keyword(x)
                x isa Expr&&x.head==:kw&&length(x.args)==2&&first(x.args) isa Symbol||throw(ArgumentError("literal keyword required"))
                key=first(x.args);haskey(kw,key)&&throw(ArgumentError("duplicate keyword"));kw[key]=value(last(x.args))
            end
            for x in node.args[2:end]
                if x isa Expr&&x.head==:parameters
                    foreach(keyword,x.args)
                elseif x isa Expr&&x.head==:kw
                    keyword(x)
                else
                    push!(pos,value(x))
                end
            end
            if name==:Figure
                allowed(kw,(:size,:fontsize));isempty(pos)&&size===nothing||throw(ArgumentError("one Figure required"))
                get(kw,:fontsize,16)==16||throw(ArgumentError("fixed font16 interface"))
                size=get(kw,:size,nothing);size isa Tuple&&length(size)==2||throw(ArgumentError("explicit size pair required"))
                return FigureRef()
            elseif name==:Axis
                allowed(kw,(:title,:xlabel,:ylabel,:xscale,:yscale))
                length(pos)==1&&pos[1] isa CellRef&&pos[1].column==length(axes)+1&&length(axes)<2||throw(ArgumentError("left-to-right axes required"))
                xs=get(kw,:xscale,:identity);ys=get(kw,:yscale,:identity)
                xs in (:identity,:log10)&&ys in (:identity,:log10)||throw(ArgumentError("supported scales only"))
                push!(axes,Dict("title"=>plain(get(kw,:title,"")),"xlabel"=>plain(get(kw,:xlabel,"")),"ylabel"=>plain(get(kw,:ylabel,"")),
                    "xscale"=>xs==:log10 ? "log10" : "linear","yscale"=>ys==:log10 ? "log10" : "linear"))
                return AxisRef(length(axes))
            elseif name in (:lines!,:scatter!,:band!)
                band=name==:band!;length(pos)==(band ? 4 : 3)&&length(series)<8||throw(ArgumentError("plot arity/count"))
                allowed(kw,(:label,:color,:linewidth,:markersize))
                name==:lines!&&get(kw,:linewidth,nothing)!=2.5&&throw(ArgumentError("explicit linewidth2.5 required"))
                name==:scatter!&&get(kw,:markersize,nothing)!=9&&throw(ArgumentError("explicit markersize9 required"))
                name!=:lines!&&haskey(kw,:linewidth)&&throw(ArgumentError("irrelevant style keyword"))
                name!=:scatter!&&haskey(kw,:markersize)&&throw(ArgumentError("irrelevant style keyword"))
                row=Dict{String,Any}("kind"=>name==:lines! ? "line" : band ? "band" : "scatter","axis"=>axisref(pos[1]),
                    "x"=>dataref(pos[2]),"label"=>plain(get(kw,:label,"")),"color"=>color(get(kw,:color,nothing),band))
                if band;row["low"]=dataref(pos[3]);row["high"]=dataref(pos[4]);else;row["y"]=dataref(pos[3]);end
                push!(series,row);return nothing
            elseif name==:axislegend
                allowed(kw,(:position,:framevisible));length(pos)==1||throw(ArgumentError("legend arity"))
                get(kw,:framevisible,nothing)===false||throw(ArgumentError("explicit frameless legend required"))
                index=axisref(pos[1]);haskey(legends,index)&&throw(ArgumentError("duplicate legend"))
                position=get(kw,:position,:rt);position in (:rt,:lt,:rb,:lb)||throw(ArgumentError("legend position"))
                legends[index]=position;return nothing
            else
                isempty(kw)&&length(pos)==2&&axisref(pos[1])!=axisref(pos[2])||throw(ArgumentError("link both axes once"))
                linked&&throw(ArgumentError("duplicate link"));linked=true;return nothing
            end
        end
        for statement in statements(body)
            if statement isa Expr&&statement.head==:(=)
                name=first(statement.args);name isa Symbol&&name∉reserved&&name!=argument||throw(ArgumentError("simple local binding required"))
                returned=value(last(statement.args));env[name]=returned
            elseif statement isa Expr&&statement.head==:return
                length(statement.args)==1||throw(ArgumentError("one return value"))
                returned=value(only(statement.args));break
            else
                returned=value(statement)
            end
        end
        returned isa FigureRef&&size!==nothing||throw(ArgumentError("return declared Figure"))
        labeled=Set(Int(s["axis"]) for s in series if !isempty(s["label"]))
        isempty(legends)||Set(keys(legends))==labeled||throw(ArgumentError("all labeled axes or no legends"))
        for (i,position) in legends;axes[i]["legend_position"]=String(position);end
        packet=(;size=collect(size),axes,series,legend=!isempty(legends),linkx=linked)
        spec=SafePlotSpec.parse_spec(JSON3.write(packet))
        (;accepted=true,reason="admitted restricted Julia-to-plot constructor program",spec)
    catch error
        error isa InterruptException&&rethrow()
        reject(sprint(showerror,error))
    end
end
end
