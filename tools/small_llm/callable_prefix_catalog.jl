isdefined(@__MODULE__,:TokenPrefixTrie) || include("token_prefix_trie.jl")
module CallablePrefixCatalog
import ..TokenPrefixTrie

"""Build a sparse lexicon from public callable metadata and a caller's tokenizer.

Each allowed prefix is a complete `API: Module.name\n` header. After it the
constraint deactivates. It does not enforce signatures or later code use.
Tokenize the WHOLE header, not independent name pieces (BPE is not compositional).
"""
function build(inventory, encode, vocabulary)
    inventory.complete || throw(ArgumentError("incomplete inventory"))
    names=String[]
    for row in inventory.rows
        row.kind in ("function","type") || throw(ArgumentError("non-callable metadata"))
        name=String(row.qualified)
        pieces=split(name,'.')
        length(pieces)==2 && all(Base.isidentifier,pieces) || throw(ArgumentError("invalid qualified name"))
        push!(names,name)
    end
    length(unique(names))==length(names) || throw(ArgumentError("duplicate name"))
    sort!(names)
    headers=["API: "*name*"\n" for name in names]
    sequences=[encode(header) for header in headers]
    trie=TokenPrefixTrie.Trie(sequences,vocabulary)
    (;names,headers,sequences,trie)
end
end
