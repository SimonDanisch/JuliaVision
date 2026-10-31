module CompactBayesJSON
using JSON3
include("bayes_json.jl")

"""Pack compact CPTs into the existing BayesJSON/shared-kernel interface.

This is ONLY a declarative frontend, not an inference engine. Binary states
default to [no,yes]; a scalar row is P(yes), with the complement computed by
the packer. Other state sets require explicit probability vectors. Row order
is the Cartesian product of parents' states, RIGHTMOST parent changing fastest.
No inferred edges, missing-row filling, probability renormalization, reference
problem lookup or generated-code evaluation is allowed.
"""
function expand(text::AbstractString;max_states=256)
    ncodeunits(text)<=65536||throw(ArgumentError("JSON budget"))
    max_states isa Integer&&!(max_states isa Bool)&&1<=max_states<=1_000_000||throw(ArgumentError("State budget"))
    # Bound depth and require the complete single object before JSON3, which
    # otherwise accepts a valid object followed by trailing non-JSON bytes.
    source=strip(text);isempty(source)&&throw(ArgumentError("Empty JSON"))
    first(source)=='{'||throw(ArgumentError("One JSON object"))
    stack=Char[];quoted=false;escaped=false;finished=false
    for c in source
        if finished
            c in (' ','\t','\n','\r')||throw(ArgumentError("Trailing bytes"))
        elseif quoted
            if escaped;escaped=false
            elseif c=='\\';escaped=true
            elseif c=='"';quoted=false
            end
        elseif c=='"';quoted=true
        elseif c in ('{','[')
            push!(stack,c);length(stack)<=16||throw(ArgumentError("Depth budget"))
        elseif c in ('}',']')
            isempty(stack)&&throw(ArgumentError("Unbalanced JSON"))
            pop!(stack)==(c=='}' ? '{' : '[')||throw(ArgumentError("Unbalanced JSON"))
            isempty(stack)&&(finished=true)
        end
    end
    finished&&!quoted&&isempty(stack)||throw(ArgumentError("Incomplete JSON"))
    p=JSON3.read(source)
    exact(x,required,optional=String[])=begin
        x isa JSON3.Object||throw(ArgumentError("Object required"))
        fieldnames=String.(keys(x))
        length(fieldnames)==length(unique(fieldnames))&&all(k->k in fieldnames,required)&&
            all(k->k in required||k in optional,fieldnames)||throw(ArgumentError("Exact fields required"))
    end
    exact(p,["nodes","query","observed"])
    p.nodes isa JSON3.Array&&1<=length(p.nodes)<=16||throw(ArgumentError("Node budget"))
    names=String[];states=Vector{String}[]
    total=1
    for n in p.nodes
        exact(n,["name","parents","table"],["states"])
        name=String(n.name)
        occursin(r"^[A-Za-z][A-Za-z0-9_]{0,31}$",name)&&!(name in names)||throw(ArgumentError("Unique node identifiers"))
        choices=hasproperty(n,:states) ? String.(n.states) : ["no","yes"]
        1<=length(choices)<=8&&length(choices)==length(unique(choices))&&all(s->1<=ncodeunits(s)<=64,choices)||throw(ArgumentError("States"))
        total<=max_states÷length(choices)||throw(ArgumentError("Enumeration budget"))
        total*=length(choices);push!(names,name);push!(states,choices)
    end
    lookup=Dict(name=>i for (i,name) in enumerate(names))
    nodes=map(enumerate(p.nodes)) do (i,n)
        n.parents isa JSON3.Array&&n.table isa JSON3.Array||throw(ArgumentError("Parent/table arrays"))
        parents=String.(n.parents)
        length(unique(parents))==length(parents)&&all(k->haskey(lookup,k)&&lookup[k]!=i,parents)||throw(ArgumentError("Parent identifiers"))
        dims=[length(states[lookup[k]]) for k in parents]
        count=prod(dims;init=1)
        length(n.table)==count||throw(ArgumentError("Complete table required"))
        rows=map(1:count) do row
            remainder=row-1;assignment=Vector{String}(undef,length(parents))
            for k in reverse(eachindex(parents))
                assignment[k]=states[lookup[parents[k]]][mod(remainder,dims[k])+1]
                remainder=div(remainder,dims[k])
            end
            value=n.table[row]
            probabilities=if value isa Real&&!(value isa Bool)
                states[i]==["no","yes"]&&isfinite(value)&&0<=value<=1||throw(ArgumentError("Scalar row requires binary no,yes states"))
                [1-Float64(value),Float64(value)]
            elseif value isa JSON3.Array
                all(v->v isa Real&&!(v isa Bool),value)||throw(ArgumentError("Numeric vector row"))
                Float64.(value)
            else
                throw(ArgumentError("Probability row"))
            end
            (;given=assignment,p=probabilities)
        end
        (;name=names[i],states=states[i],parents,rows)
    end
    p.observed isa JSON3.Object||throw(ArgumentError("Observation object"))
    observed_keys=String.(keys(p.observed))
    length(observed_keys)==length(unique(observed_keys))||throw(ArgumentError("Duplicate observation"))
    expanded=(;nodes,query=String(p.query),observed=Dict(String(k)=>String(v) for (k,v) in pairs(p.observed)))
    packed=BayesJSON.compile(JSON3.write(expanded);max_states)
    (;network=expanded,packed)
end
end
