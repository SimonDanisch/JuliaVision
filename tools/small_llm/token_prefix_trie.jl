"""Sparse token-prefix constraints. This restricts syntax, not truth or API relevance.

Token IDs are zero-based; nodes are one-based. State 0 means unconstrained,
state -1 means rejected. A terminal node exits the constraint permanently.
The same binary-search transition is used by host tests and device kernels.
"""
module TokenPrefixTrie

struct Trie
    offsets::Vector{Int32}
    labels::Vector{Int32}
    targets::Vector{Int32}
    terminal::Vector{Int32}
    vocabulary::Int32
end

function Trie(sequences, vocabulary::Integer)
    vocabulary isa Bool && throw(ArgumentError("integer vocabulary required"))
    0 < vocabulary <= typemax(Int32) || throw(ArgumentError("invalid vocabulary"))
    children = [Dict{Int32,Int32}()]
    terminal = Int32[0]
    seen = Set{Tuple}()
    for sequence in sequences
        isempty(sequence) && throw(ArgumentError("empty prefix"))
        ids = Int32[]
        for token in sequence
            token isa Integer && !(token isa Bool) && 0 <= token < vocabulary ||
                throw(ArgumentError("invalid token ID"))
            push!(ids, Int32(token))
        end
        key = Tuple(ids)
        key in seen && throw(ArgumentError("duplicate prefix"))
        push!(seen, key)
        node = Int32(1)
        for token in ids
            terminal[node] == 0 || throw(ArgumentError("prefixes must be prefix-free"))
            if !haskey(children[node], token)
                length(children) < typemax(Int32) || throw(ArgumentError("too many nodes"))
                push!(children, Dict{Int32,Int32}()); push!(terminal, 0)
                children[node][token] = Int32(length(children))
            end
            node = children[node][token]
        end
        isempty(children[node]) || throw(ArgumentError("prefixes must be prefix-free"))
        terminal[node] = 1
    end
    isempty(seen) && throw(ArgumentError("no prefixes"))
    offsets = Int32[1]; labels = Int32[]; targets = Int32[]
    for edges in children
        for token in sort!(collect(keys(edges)))
            push!(labels, token); push!(targets, edges[token])
        end
        push!(offsets, Int32(length(labels) + 1))
    end
    Trie(offsets, labels, targets, terminal, Int32(vocabulary))
end

@inline function edge_target(offsets, labels, targets, node::Int32, token::Int32)
    # Invalid state never indexes buffers. Zero is the unconstrained sentinel.
    1 <= node < length(offsets) || return Int32(-1)
    @inbounds lo = offsets[node]
    @inbounds hi = offsets[node + Int32(1)] - Int32(1)
    while lo <= hi
        mid = lo + ((hi - lo) >> 1)
        @inbounds label = labels[mid]
        if label == token
            @inbounds return targets[mid]
        elseif label < token
            lo = mid + Int32(1)
        else
            hi = mid - Int32(1)
        end
    end
    Int32(-1)
end

@inline function next_node(offsets, labels, targets, terminal, node::Int32, token::Int32)
    node == 0 && return Int32(0)
    target = edge_target(offsets, labels, targets, node, token)
    target < 1 && return Int32(-1)
    @inbounds terminal[target] != 0 ? Int32(0) : target
end
next_node(t::Trie, node::Integer, token::Integer) =
    next_node(t.offsets, t.labels, t.targets, t.terminal, Int32(node), Int32(token))

@inline function constrained_logit(offsets, labels, targets, node::Int32, token::Int32, value)
    node == 0 && return value
    edge_target(offsets, labels, targets, node, token) > 0 ? value : oftype(value, -Inf)
end
end
