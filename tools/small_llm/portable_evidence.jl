module PortableEvidence
import Mantle
import KernelInterface as KI

# Kernel bodies below are identical for Lavapipe (CPU) and hardware GPUs.
# Host-side packing translates symbols/variable axes into indexed flat arrays.
@inline logadd(a::Float32,b::Float32)=a==-Inf32 ? b : b==-Inf32 ? a : max(a,b)+log1p(exp(-abs(a-b)))
@inline assignment(i,strides,cards,v)=((i-Int32(1))÷strides[v])%cards[v]+Int32(1)

function enumerate_factors!(prior,joint,cards,strides,observed,offsets,variables,fstrides,voffsets,logvalues,nstates::Int32,nfactors::Int32,nvars::Int32)
    i=Int32(KI.get_global_id().x)
    if i<=nstates
        weight=0f0;compatible=true
        @inbounds for v in Int32(1):nvars
            state=assignment(i,strides,cards,v)
            compatible &= observed[v]==0 || observed[v]==state
        end
        @inbounds for f in Int32(1):nfactors
            index=voffsets[f]
            for j in offsets[f]:offsets[f+Int32(1)]-Int32(1)
                index+=(assignment(i,strides,cards,variables[j])-Int32(1))*fstrides[j]
            end
            weight+=logvalues[index]
        end
        @inbounds begin prior[i]=weight;joint[i]=compatible ? weight : -Inf32;end
    end
    nothing
end

function marginalize!(prior_mass,joint_mass,prior,joint,nstates::Int32,qstride::Int32,nquery::Int32)
    q=Int32(KI.get_global_id().x)
    if q<=nquery
        p=-Inf32;j=-Inf32
        @inbounds for i in Int32(1):nstates
            if ((i-Int32(1))÷qstride)%nquery+Int32(1)==q
                p=logadd(p,prior[i]);j=logadd(j,joint[i])
            end
        end
        @inbounds begin prior_mass[q]=p;joint_mass[q]=j;end
    end
    nothing
end

function normalize!(prior,posterior,likelihood,defined,status,log_evidence,pm,jm,n::Int32)
    if KI.get_global_id().x==1
        z=-Inf32;e=-Inf32
        @inbounds for i in Int32(1):n
            z=logadd(z,pm[i]);e=logadd(e,jm[i])
        end
        @inbounds begin
            # 0: valid, 1: impossible observations, 2: zero total factor mass.
            status[1]=z==-Inf32 ? Int32(2) : e==-Inf32 ? Int32(1) : Int32(0)
            log_evidence[1]=z==-Inf32 ? -Inf32 : e-z
            for i in Int32(1):n
                prior[i]=z==-Inf32 ? 0f0 : exp(pm[i]-z)
                posterior[i]=e==-Inf32 ? 0f0 : exp(jm[i]-e)
                defined[i]=pm[i]==-Inf32 ? Int32(0) : Int32(1)
                likelihood[i]=pm[i]==-Inf32 ? -Inf32 : jm[i]-pm[i]
            end
        end
    end
    nothing
end

function initialize_proof!(known,proof,facts,n::Int32)
    i=Int32(KI.get_global_id().x)
    if i<=n
        @inbounds begin known[i]=facts[i];proof[i]=facts[i]==1 ? Int32(0) : Int32(-1);end
    end
    nothing
end

function horn_round!(out,oproof,known,proof,heads,offsets,premises,nrules::Int32,n::Int32)
    atom=Int32(KI.get_global_id().x)
    if atom<=n
        @inbounds begin
            value=known[atom];origin=proof[atom]
            if value==0
                for r in Int32(1):nrules
                    if heads[r]==atom
                        satisfied=true
                        for j in offsets[r]:offsets[r+Int32(1)]-Int32(1)
                            satisfied &= known[premises[j]]==1
                        end
                        if satisfied
                            value=Int32(1);origin=r;break
                        end
                    end
                end
            end
            out[atom]=value;oproof[atom]=origin
        end
    end
    nothing
end

function proof_status!(positive,provenance,status,known,proof,negative,n::Int32)
    i=Int32(KI.get_global_id().x)
    if i<=n
        @inbounds begin
            positive[i]=known[i];provenance[i]=proof[i]
            # 0 unknown, 1 entailed, 2 refuted, 3 inconsistent. No explosion.
            status[i]=known[i]+Int32(2)*negative[i]
        end
    end
    nothing
end

function truth_mask!(bias,status,query::Int32,yes::Int32,no::Int32,n::Int32)
    i=Int32(KI.get_global_id().x)
    if i<=n
        @inbounds begin
            result=status[query]
            target=result==1 ? yes : no
            bias[i]=(result==1 || result==2) && i-Int32(1)!=target ? -Inf32 : 0f0
        end
    end
    nothing
end

function status_mask!(bias,status,query::Int32,labels::NTuple{4,Int32},enabled,n::Int32)
    i=Int32(KI.get_global_id().x)
    if i<=n
        @inbounds begin
            id=i-Int32(1)
            allowed=if enabled[1]==1
                id==labels[status[query]+Int32(1)]
            else
                id==labels[1] || id==labels[2] || id==labels[3] || id==labels[4]
            end
            bias[i]=allowed ? 0f0 : -Inf32
        end
    end
    nothing
end

function add_likelihood!(logits,likelihood,defined,status,tokens,nquery::Int32,nvocab::Int32,::Val{RESTRICT}) where RESTRICT
    i=Int32(KI.get_global_id().x)
    if i<=nvocab
        @inbounds begin
            usable=status[1]==0
            for q in Int32(1):nquery
                usable &= defined[q]==1
            end
            if usable
                matched=false;score=0f0
                for q in Int32(1):nquery
                    if tokens[q]==i-Int32(1)
                        matched=true;score=likelihood[q]
                    end
                end
                logits[i]=RESTRICT && !matched ? -Inf32 : logits[i]+score
            end
        end
    end
    nothing
end

struct Execution{D,I,O,M,P}
    device::D
    inputs::I
    outputs::O
    metadata::M
    plan::P
    owned::Vector{Any}
end
run!(e::Execution)=(Mantle.run!(e.plan);e.outputs)
function Base.close(e::Execution)
    Mantle.free!(e.plan);foreach(Mantle.free!,e.owned);empty!(e.owned);nothing
end

function compile_factors(factors,query,observations;max_states=1_000_000)
    cards=Dict{Symbol,Int}()
    for f in factors
        length(f.variables)==ndims(f.values) || throw(ArgumentError("factor axes mismatch"))
        length(unique(f.variables))==length(f.variables) || throw(ArgumentError("duplicate variable"))
        all(x->isfinite(x)&&x>=0,f.values) || throw(ArgumentError("invalid factor value"))
        for (axis,v) in enumerate(f.variables)
            k=size(f.values,axis)
            k>0 && (!haskey(cards,v)||cards[v]==k) || throw(ArgumentError("inconsistent cardinality"))
            cards[v]=k
        end
    end
    haskey(cards,query) || throw(ArgumentError("unknown query"))
    names=sort!(collect(keys(cards));by=string);lookup=Dict(v=>i for (i,v) in enumerate(names))
    counts=Int32[cards[v] for v in names];strides=Int32[];nstates=1
    for count in counts
        nstates<=min(max_states,typemax(Int32))÷count || throw(ArgumentError("enumeration budget exceeded"))
        push!(strides,Int32(nstates));nstates*=count
    end
    observed=zeros(Int32,length(names))
    for (v,state) in observations
        haskey(cards,v) && 1<=state<=cards[v] || throw(ArgumentError("invalid observed state"))
        observed[lookup[v]]=Int32(state)
    end
    offsets=Int32[1];variables=Int32[];fstrides=Int32[];voffsets=Int32[];logvalues=Float32[]
    for f in factors
        stride=1;push!(voffsets,Int32(length(logvalues)+1))
        logs=log.(Float64.(vec(f.values)))
        peak=maximum(logs)
        # A factor's common scale cancels in all normalized results and
        # P(observations). Remove it before Float32 conversion, not afterward.
        isfinite(peak) && (logs .-= peak)
        append!(logvalues,Float32.(logs))
        for (axis,v) in enumerate(f.variables)
            push!(variables,Int32(lookup[v]));push!(fstrides,Int32(stride));stride*=size(f.values,axis)
        end
        push!(offsets,Int32(length(variables)+1))
    end
    (;names,cards=counts,strides,observed,offsets,variables,fstrides,voffsets,logvalues,
      nstates,nfactors=length(factors),query=lookup[query],nquery=cards[query])
end

function declare_bayes!(g,i,o,m)
    prior=Mantle.Transient.Buffer(g,Float32,(m.nstates,));joint=Mantle.Transient.Buffer(g,Float32,(m.nstates,))
    pm=Mantle.Transient.Buffer(g,Float32,(m.nquery,));jm=Mantle.Transient.Buffer(g,Float32,(m.nquery,))
    Mantle.dispatch!(g,enumerate_factors!,
        (prior,joint,i.cards,i.strides,i.observed,i.offsets,i.variables,i.fstrides,i.voffsets,i.logvalues,
         Int32(m.nstates),Int32(m.nfactors),Int32(length(m.names))),m.nstates;group=256,name="evidence.enumerate")
    Mantle.dispatch!(g,marginalize!,
        (pm,jm,prior,joint,Int32(m.nstates),m.strides[m.query],Int32(m.nquery)),m.nquery;group=256,name="evidence.marginalize")
    Mantle.dispatch!(g,normalize!,
        (o.prior,o.posterior,o.log_likelihood,o.defined,o.status,o.log_evidence,pm,jm,Int32(m.nquery)),
        1;group=1,name="evidence.normalize")
    o
end

function prepare_bayes(device,factors,query,observations=Dict{Symbol,Int}();max_states=1_000_000)
    m=compile_factors(factors,query,observations;max_states)
    owned=Any[];plan=nothing
    upload(x)=begin a=Mantle.Buffer(device,x);push!(owned,a);a;end
    output(T,n)=upload(zeros(T,n))
    try
        fields=(:cards,:strides,:observed,:offsets,:variables,:fstrides,:voffsets,:logvalues)
        inputs=NamedTuple{fields}(Tuple(upload(getproperty(m,k)) for k in fields))
        outputs=(;prior=output(Float32,m.nquery),posterior=output(Float32,m.nquery),
            log_likelihood=output(Float32,m.nquery),defined=output(Int32,m.nquery),
            status=output(Int32,1),log_evidence=output(Float32,1))
        g=Mantle.Graph(device);declare_bayes!(g,inputs,outputs,m)
        plan=Mantle.Plan(g;budget=Inf);Mantle.record!(plan)
        Execution(device,inputs,outputs,m,plan,owned)
    catch
        plan===nothing || Mantle.free!(plan)
        foreach(Mantle.free!,owned);rethrow()
    end
end

function compile_horn(facts,rules,query,negative_facts)
    names=sort!(unique(vcat(Symbol.(facts),Symbol.(negative_facts),[Symbol(query)],
        [Symbol(r.head) for r in rules],[Symbol(p) for r in rules for p in r.premises]));by=string)
    lookup=Dict(v=>i for (i,v) in enumerate(names));n=length(names)
    initial=zeros(Int32,n);negative=zeros(Int32,n)
    for f in facts;initial[lookup[Symbol(f)]]=1;end
    for f in negative_facts;negative[lookup[Symbol(f)]]=1;end
    heads=Int32[];offsets=Int32[1];premises=Int32[]
    for r in rules
        push!(heads,Int32(lookup[Symbol(r.head)]))
        append!(premises,Int32[lookup[Symbol(p)] for p in r.premises])
        push!(offsets,Int32(length(premises)+1))
    end
    # Empty input lists still get a valid descriptor address; their count is zero.
    nrules=length(heads)
    isempty(heads) && push!(heads,0)
    isempty(premises) && push!(premises,0)
    (;names,initial,negative,heads,offsets,premises,n,nrules,query=lookup[Symbol(query)])
end

function declare_horn!(g,i,o,m)
    scratch()=Mantle.Transient.Buffer(g,Int32,(m.n,))
    a=scratch();ap=scratch();b=scratch();bp=scratch()
    Mantle.dispatch!(g,initialize_proof!,(a,ap,i.initial,Int32(m.n)),m.n;group=256,name="proof.initialize")
    # At most n synchronous rounds derive every atom in a finite positive program.
    for round in 1:m.n
        Mantle.dispatch!(g,horn_round!,
            (b,bp,a,ap,i.heads,i.offsets,i.premises,Int32(m.nrules),Int32(m.n)),
            m.n;group=256,name="proof.round.$round")
        a,b=b,a;ap,bp=bp,ap
    end
    Mantle.dispatch!(g,proof_status!,
        (o.positive,o.provenance,o.status,a,ap,i.negative,Int32(m.n)),m.n;group=256,name="proof.status")
    o
end

function prepare_horn(device,facts,rules,query;negative_facts=Symbol[])
    m=compile_horn(facts,rules,query,negative_facts);owned=Any[];plan=nothing
    upload(x)=begin a=Mantle.Buffer(device,x);push!(owned,a);a;end
    try
        fields=(:initial,:negative,:heads,:offsets,:premises)
        inputs=NamedTuple{fields}(Tuple(upload(getproperty(m,k)) for k in fields))
        outputs=(;positive=upload(zeros(Int32,m.n)),provenance=upload(fill(Int32(-1),m.n)),status=upload(zeros(Int32,m.n)))
        g=Mantle.Graph(device);declare_horn!(g,inputs,outputs,m)
        plan=Mantle.Plan(g;budget=Inf);Mantle.record!(plan)
        Execution(device,inputs,outputs,m,plan,owned)
    catch
        plan===nothing || Mantle.free!(plan)
        foreach(Mantle.free!,owned);rethrow()
    end
end

function declare_truth_mask!(g,bias,proof::Execution,yes::Integer,no::Integer)
    0<=yes<length(bias) && 0<=no<length(bias) && yes!=no || throw(ArgumentError("invalid truth token IDs"))
    Mantle.dispatch!(g,truth_mask!,
        (bias,proof.outputs.status,Int32(proof.metadata.query),Int32(yes),Int32(no),Int32(length(bias))),
        length(bias);group=256,name="proof.truth_mask")
    nothing
end

"""A four-way proof-status readout for experiments, including unknown/conflict.

Labels are ordered (unknown, entailed, refuted, inconsistent). This classifies a
formal proof state; use declare_truth_mask! for an open-world yes/no decision.
"""
function declare_status_mask!(g,bias,proof::Execution,labels::NTuple{4,<:Integer},enabled)
    length(unique(labels))==4 && all(id->0<=id<length(bias),labels) || throw(ArgumentError("invalid status labels"))
    Mantle.dispatch!(g,status_mask!,
        (bias,proof.outputs.status,Int32(proof.metadata.query),Int32.(labels),enabled,Int32(length(bias))),
        length(bias);group=256,name="proof.status_mask")
    nothing
end

"""Apply computed conditional likelihoods to logits in the same graph.

token_ids is a device buffer containing distinct zero-based vocabulary IDs in
query-state order. Invalid evidence or undefined conditionals leave logits
unchanged; inspect outputs.status/defined before interpreting a Bayesian result.
The optional restriction conditions generation on this finite candidate set.
"""
function declare_bayesian_logits!(g,logits,e::Execution,token_ids;restrict_candidates=false)
    length(token_ids)==e.metadata.nquery || throw(ArgumentError("candidate cardinality mismatch"))
    Mantle.dispatch!(g,add_likelihood!,
        (logits,e.outputs.log_likelihood,e.outputs.defined,e.outputs.status,token_ids,
         Int32(e.metadata.nquery),Int32(length(logits)),Val(restrict_candidates)),
        length(logits);group=256,name="evidence.apply_likelihood")
    nothing
end

end
