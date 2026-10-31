module BayesJSON
using JSON3

"""Compile bounded declarative Bayesian-network JSON to shared-solver factors.

This is host-side input validation/packing, NOT a second CPU inference engine.
The returned factors feed PortableEvidence.prepare_bayes on either backend.
Node rows give parent STATE NAMES in parent order and probabilities in the
node's declared state order. No Julia expressions or code are accepted.
"""
function compile(text::AbstractString;max_states=1_000_000)
    ncodeunits(text)<=65536 || throw(ArgumentError("JSON budget exceeded"))
    data=JSON3.read(text)
    exactkeys(x,allowed)=Set(String.(keys(x)))==Set(allowed)
    exactkeys(data,["nodes","query","observed"]) || throw(ArgumentError("network fields"))
    1<=length(data.nodes)<=16 || throw(ArgumentError("node budget"))
    nodes=data.nodes;names=String[];states=Vector{String}[]
    for node in nodes
        exactkeys(node,["name","states","parents","rows"]) || throw(ArgumentError("node fields"))
        name=String(node.name)
        occursin(r"^[A-Za-z][A-Za-z0-9_]{0,31}$",name) || throw(ArgumentError("node identifier"))
        name in names && throw(ArgumentError("duplicate node"))
        choices=String.(node.states)
        1<=length(choices)<=8 && length(unique(choices))==length(choices) || throw(ArgumentError("states"))
        all(s->1<=ncodeunits(s)<=64,choices) || throw(ArgumentError("state name budget"))
        push!(names,name);push!(states,choices)
    end
    lookup=Dict(n=>i for (i,n) in enumerate(names));parent_ids=Vector{Int}[]
    nstates=1
    for choices in states
        nstates<=max_states÷length(choices) || throw(ArgumentError("enumeration budget exceeded"))
        nstates*=length(choices)
    end
    for (i,node) in enumerate(nodes)
        parents=String.(node.parents)
        length(unique(parents))==length(parents) && all(p->haskey(lookup,p)&&lookup[p]!=i,parents) || throw(ArgumentError("invalid parents"))
        push!(parent_ids,[lookup[p] for p in parents])
    end
    # A DAG is required: rejecting cycles avoids silently treating an arbitrary
    # cyclic product of normalized conditionals as a Bayesian network.
    resolved=falses(length(nodes))
    for _ in eachindex(nodes)
        for i in eachindex(nodes)
            all(p->resolved[p],parent_ids[i]) && (resolved[i]=true)
        end
    end
    all(resolved) || throw(ArgumentError("cyclic network"))
    factors=Any[]
    for (i,node) in enumerate(nodes)
        parents=parent_ids[i];dimensions=Tuple(length(states[j]) for j in vcat(parents,i))
        expected=prod(dimensions[1:end-1];init=1)
        length(node.rows)==expected || throw(ArgumentError("incomplete conditional table"))
        values=fill(NaN,dimensions);seen=Set{Tuple}()
        for row in node.rows
            exactkeys(row,["given","p"]) || throw(ArgumentError("row fields"))
            length(row.given)==length(parents) || throw(ArgumentError("parent assignment arity"))
            assignment=Tuple(map(eachindex(parents)) do k
                state=findfirst(==(String(row.given[k])),states[parents[k]])
                state===nothing && throw(ArgumentError("unknown parent state"))
                state
            end)
            assignment in seen && throw(ArgumentError("duplicate conditional row"));push!(seen,assignment)
            all(p->p isa Real && !(p isa Bool),row.p) || throw(ArgumentError("numeric probabilities required"))
            probabilities=Float64.(row.p)
            length(probabilities)==length(states[i]) && all(p->isfinite(p)&&0<=p<=1,probabilities) || throw(ArgumentError("invalid probabilities"))
            isapprox(sum(probabilities),1;atol=1e-6,rtol=0) || throw(ArgumentError("unnormalized conditional row"))
            for s in eachindex(probabilities);values[assignment...,s]=probabilities[s];end
        end
        all(isfinite,values) || throw(ArgumentError("missing table cell"))
        push!(factors,(;variables=Symbol.(names[vcat(parents,i)]),values))
    end
    query=String(data.query);haskey(lookup,query) || throw(ArgumentError("unknown query"))
    observed=Dict{Symbol,Int}()
    for (name,value) in pairs(data.observed)
        key=String(name);haskey(lookup,key) || throw(ArgumentError("unknown observation"))
        state=findfirst(==(String(value)),states[lookup[key]])
        state===nothing && throw(ArgumentError("unknown observed state"))
        observed[Symbol(key)]=state
    end
    (;factors,query=Symbol(query),observed,query_states=states[lookup[query]],nstates)
end
end
