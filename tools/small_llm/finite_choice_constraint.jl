isdefined(@__MODULE__,:TokenPrefixTrie)||include("token_prefix_trie.jl")

"""Finite canonical-token support, not a truth/relevance or whole-language solver.

Accepts exactly one supplied choice followed by EOS. Supports prefix-sharing
names by including EOS in each path. It does NOT admit every alternate BPE
segmentation of a string: encode/decode round trips define the canonical paths.
The same transition/mask helpers run on CPU and device.
"""
module FiniteChoiceConstraint
import ..TokenPrefixTrie

struct Grammar
    trie::TokenPrefixTrie.Trie
    choices::Vector{String}
    sequences::Vector{Vector{Int32}}
    eos::Int32
end

function Grammar(encode,decode,choices,vocabulary,eos;max_choices=4096,max_nodes=1_000_000)
    vocabulary isa Integer&&!(vocabulary isa Bool)&&0<vocabulary<=typemax(Int32)||throw(ArgumentError("Vocabulary"))
    eos isa Integer&&!(eos isa Bool)&&0<=eos<vocabulary||throw(ArgumentError("EOS"))
    max_choices isa Integer&&!(max_choices isa Bool)&&1<=max_choices<=4096||throw(ArgumentError("Choice budget"))
    max_nodes isa Integer&&!(max_nodes isa Bool)&&1<=max_nodes<=1_000_000||throw(ArgumentError("Node budget"))
    1<=length(choices)<=max_choices||throw(ArgumentError("Nonempty bounded choices"))
    all(c->c isa AbstractString&&0<ncodeunits(c)<=1024,choices)||throw(ArgumentError("Bounded choice strings"))
    names=String.(choices);length(unique(names))==length(names)||throw(ArgumentError("Duplicate choices"))
    sequences=Vector{Int32}[];extent=1
    for name in names
        ids=encode(name)
        1<=length(ids)<=1024&&all(t->t isa Integer&&!(t isa Bool)&&0<=t<vocabulary&&t!=eos,ids)||throw(ArgumentError("Canonical token path"))
        tokens=Int32.(ids);decode(tokens)==name||throw(ArgumentError("Exact codec roundtrip required"))
        # Upper bound includes the accepting EOS edge; preflight BEFORE trie allocation.
        extent<=max_nodes-length(tokens)-1||throw(ArgumentError("Node budget exceeded"))
        extent+=length(tokens)+1;push!(sequences,vcat(tokens,Int32(eos)))
    end
    trie=TokenPrefixTrie.Trie(sequences,vocabulary)
    Grammar(trie,names,sequences,Int32(eos))
end

@inline function constrained_value(offsets,labels,targets,node::Int32,done::Int32,eos::Int32,token::Int32,value)
    if done==1
        return token==eos ? value : oftype(value,-Inf)
    end
    TokenPrefixTrie.constrained_logit(offsets,labels,targets,node,token,value)
end

@inline function transition(offsets,labels,targets,terminal,node::Int32,done::Int32,rejected::Int32,eos::Int32,token::Int32)
    rejected!=0&&return (Int32(-1),done,Int32(1))
    if done==1
        return token==eos ? (node,done,Int32(0)) : (Int32(-1),done,Int32(1))
    end
    done==0&&node>=1||return (Int32(-1),done,Int32(1))
    target=TokenPrefixTrie.edge_target(offsets,labels,targets,node,token)
    target>=1||return (Int32(-1),done,Int32(1))
    if @inbounds terminal[target]!=0
        return token==eos ? (target,Int32(1),Int32(0)) : (Int32(-1),done,Int32(1))
    end
    (target,Int32(0),Int32(0))
end
transition(g::Grammar,state,token::Int32)=transition(g.trie.offsets,g.trie.labels,g.trie.targets,g.trie.terminal,state...,g.eos,token)
end
