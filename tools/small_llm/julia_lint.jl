"""Small transparent static diagnostics for the Julia pilot.

Parses, never evaluates generated code. These are conservative task-specific
heuristics, NOT a replacement for Julia inference, JET, or a language server.
They expose multiple likely errors rather than stopping at the first exception.
"""
module JuliaPilotLint

function analyze(code;requires_views=false)
    ncodeunits(code)<=65536 || throw(ArgumentError("candidate too large"))
    ast=Meta.parseall(code)
    findings=NamedTuple[];bindings=Dict{Symbol,Symbol}()
    emit(kind,message)=push!(findings,(;kind,message))
    function constructor(e)
        e isa Expr && e.head==:call || return nothing
        f=first(e.args)
        f isa Symbol && return f
        f isa Expr && f.head==:curly && return first(f.args)
        nothing
    end
    function bind(e)
        e isa Expr || return
        if e.head==:(=) && length(e.args)==2 && first(e.args) isa Symbol
            c=constructor(last(e.args))
            c in (:Axis,:Figure) && (bindings[first(e.args)]=c)
        end
        foreach(bind,e.args)
    end
    bind(ast)
    function walk(e)
        e isa Expr || return
        if e.head in (:error,:incomplete)
            emit("syntax",sprint(show,e));return
        elseif e.head==:(::) && length(e.args)==2 && requires_views
            t=last(e.args)
            if t===:Vector || (t isa Expr && t.head==:curly && first(t.args)===:Vector)
                emit("input_domain","A Vector annotation excludes SubArray views explicitly allowed by the task. AbstractVector accepts both ordinary vectors and views; an untyped argument can also accept both.")
            end
        elseif e.head==:call && !isempty(e.args)
            f=first(e.args)
            if f isa Expr && f.head==:. && length(f.args)==2
                object,field=f.args
                if object isa Symbol && get(bindings,object,nothing)===:Axis && field isa QuoteNode
                    name=field.value
                    if name in (:plot!,:lines!,:scatter!,:band!)
                        emit("module_function","$object is constructed as Axis. $name is a Makie module-level plotting function, not an Axis method field; the axis is passed as its first argument.")
                    elseif name===:legend
                        emit("module_function","$object is constructed as Axis; it has no legend method field. Makie's module-level axislegend function takes an Axis as an argument.")
                    end
                end
            elseif f===:Figure
                for arg in e.args[2:end]
                    children=arg isa Expr && arg.head==:parameters ? arg.args : Any[arg]
                    for kw in children
                        if kw isa Expr && kw.head==:kw && first(kw.args)===:resolution
                            emit("deprecated_api","Figure's resolution keyword is deprecated in this Makie checkout; size=(width,height) is the current keyword. A scalar resolution is not a two-dimensional figure size.")
                        end
                    end
                end
            end
        elseif e.head in (:(=),:(+=),:(-=)) && !isempty(e.args)
            target=first(e.args)
            if target isa Expr && target.head==:. && last(target.args) isa QuoteNode
                object,field=target.args
                if field.value in (:first,:second) && occursin("Pair",code)
                    emit("immutable_field","This assigns to .$(field.value) while the code constructs Pairs. Pair is immutable: its fields cannot be reassigned; updating a collection entry and mutating a Pair field are different operations.")
                elseif object isa Symbol && get(bindings,object,nothing)===:Figure && field.value===:title
                    emit("wrong_object","Figure is a layout/container, not the upper Axis. An Axis owns the title attribute requested for that subplot.")
                end
            elseif target isa Expr && target.head==:ref && length(target.args)==2
                object,key=target.args
                if object isa Symbol && key isa String && get(bindings,object,nothing) in (:Figure,:Axis)
                    emit("wrong_indexing","String indexing $object[$(repr(key))] is not Makie's API for setting axis labels/titles.")
                end
            end
        end
        foreach(walk,e.args)
    end
    walk(ast)
    unique(findings)
end
end
