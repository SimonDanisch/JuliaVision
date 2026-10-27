# Independent host oracle for tests. Runtime inference is implemented once in
# ../portable_evidence.jl and runs unchanged on CPU and GPU through Mantle.
module StructuredEvidence

struct Factor
    variables::Vector{Symbol}
    values::Array{Float64}
    function Factor(variables,values)
        names=Symbol.(variables);data=Float64.(values)
        length(names)==ndims(data) || throw(ArgumentError("one axis required per variable"))
        length(unique(names))==length(names) || throw(ArgumentError("repeated factor variable"))
        all(d->d>0,size(data)) || throw(ArgumentError("empty factor axis"))
        all(x->isfinite(x)&&x>=0,data) || throw(ArgumentError("invalid factor probability"))
        new(names,data)
    end
end

logadd(a,b)=a==-Inf ? b : b==-Inf ? a : max(a,b)+log1p(exp(-abs(a-b)))

"""Exact finite-factor inference by log-space enumeration, with a state budget.

Variables use one-based discrete states. Factors may be unnormalized potentials;
their product is normalized here. This reference is for small experiments, not
scalable general-purpose Bayesian inference.
"""
function infer(factors::AbstractVector{Factor},query::Symbol,observations=Dict{Symbol,Int}();max_states=1_000_000)
    cards=Dict{Symbol,Int}()
    for f in factors, (axis,v) in enumerate(f.variables)
        card=size(f.values,axis)
        haskey(cards,v) && cards[v]!=card && throw(ArgumentError("inconsistent cardinality for $v"))
        cards[v]=card
    end
    haskey(cards,query) || throw(ArgumentError("unknown query variable"))
    for (v,state) in observations
        haskey(cards,v) && 1<=state<=cards[v] || throw(ArgumentError("invalid observed state for $v"))
    end
    states=1
    for card in values(cards)
        states<=max_states÷card || throw(ArgumentError("exact enumeration exceeds state budget"))
        states*=card
    end
    variables=sort!(collect(keys(cards));by=string)
    axes=Dict(v=>i for (i,v) in enumerate(variables))
    dims=Tuple(cards[v] for v in variables)
    logprior=fill(-Inf,cards[query]);logjoint=copy(logprior)
    for assignment in CartesianIndices(dims)
        logweight=0.0
        for f in factors
            p=f.values[Tuple(assignment[axes[v]] for v in f.variables)...]
            if p==0
                logweight=-Inf;break
            end
            logweight+=log(p)
        end
        state=assignment[axes[query]]
        logprior[state]=logadd(logprior[state],logweight)
        if all(assignment[axes[v]]==state for (v,state) in observations)
            logjoint[state]=logadd(logjoint[state],logweight)
        end
    end
    logz=foldl(logadd,logprior;init=-Inf)
    isfinite(logz) || throw(ArgumentError("factor system has zero total probability"))
    loge=foldl(logadd,logjoint;init=-Inf)
    prior=exp.(logprior.-logz)
    posterior=loge==-Inf ? nothing : exp.(logjoint.-loge)
    # Undefined conditionals are explicit. They must not become hard masks.
    likelihood=Union{Missing,Float64}[a==-Inf ? missing : b-a for (a,b) in zip(logprior,logjoint)]
    (;prior,posterior,log_evidence=loge-logz,log_likelihood=likelihood,enumerated_states=states)
end

"""Map P(observations | hypothesis) to single-token candidate log scores.

Do not add the posterior to model logits: that would count its prior twice.
These scores are Bayesian only if the model distribution represents the intended
hypothesis prior. Multi-token candidate labels require a token trie/span scorer.
"""
function token_likelihood_scores(result,token_ids::AbstractVector{<:Integer})
    result.posterior===nothing && throw(ArgumentError("observations have zero probability"))
    length(token_ids)==length(result.log_likelihood) || throw(ArgumentError("candidate cardinality mismatch"))
    length(unique(token_ids))==length(token_ids) || throw(ArgumentError("candidate tokens must be distinct"))
    all(t->t>=0,token_ids) || throw(ArgumentError("token IDs must be nonnegative"))
    any(ismissing,result.log_likelihood) && throw(ArgumentError("conditional likelihood undefined for zero-prior hypothesis"))
    Dict(Int(t)=>Float32(l) for (t,l) in zip(token_ids,result.log_likelihood))
end

"""Return a verified binary constraint, or nothing when no truth constraint is justified.

The caller may install returned scores with default=-Inf on the evidence buffer.
Unknown and inconsistent proofs require abstention or a soft/search policy.
"""
function proof_token_scores(result,affirmative::Integer,negative::Integer)
    affirmative>=0 && negative>=0 && affirmative!=negative || throw(ArgumentError("distinct nonnegative tokens required"))
    result.status===:entailed && return Dict(Int(affirmative)=>0f0)
    result.status===:refuted && return Dict(Int(negative)=>0f0)
    result.status in (:unknown,:inconsistent) && return nothing
    throw(ArgumentError("unrecognized proof status"))
end

struct Rule
    head::Symbol
    premises::Vector{Symbol}
end
Rule(head,premises)=Rule(Symbol(head),Symbol.(premises))

"""Positive ground Horn reasoning with explicit negative facts and open-world status.

No failure-to-prove assumption is made. A contradiction produces :inconsistent,
which callers must not silently translate into a hard token exclusion.
"""
function prove(facts,rules::AbstractVector{Rule},query;negative_facts=Symbol[])
    known=Set(Symbol.(facts));negative=Set(Symbol.(negative_facts))
    proof=Dict{Symbol,Union{Nothing,Int}}(f=>nothing for f in known)
    changed=true
    while changed
        changed=false
        for (i,rule) in enumerate(rules)
            if !(rule.head in known) && all(p->p in known,rule.premises)
                push!(known,rule.head);proof[rule.head]=i;changed=true
            end
        end
    end
    q=Symbol(query)
    status=q in known ? (q in negative ? :inconsistent : :entailed) :
                       (q in negative ? :refuted : :unknown)
    (;status,derived=known,negative,proof,contradictions=intersect(known,negative))
end

end
