module QwenText
using JSON3, Unicode
import BonsaiRunner

struct Tokenizer
    bpe::BonsaiRunner.BonsaiTokenizer
end

"""Load native text-tokenizer tables, reusing the existing Qwen byte-BPE engine.

The wrapper applies the checkpoint's NFC normalization. Chat rendering supports
plain text system/user/assistant messages; tool and multimodal templates need
separate parity fixtures before being supported.
"""
function Tokenizer(directory::AbstractString)
    data=JSON3.read(read(joinpath(directory,"tokenizer.json"),String))
    data.model.type=="BPE" && data.normalizer.type=="NFC" || throw(ArgumentError("unsupported tokenizer configuration"))
    vocab=Dict(String(s)=>Int(i) for (s,i) in pairs(data.model.vocab))
    for added in data.added_tokens
        for flag in (:single_word,:lstrip,:rstrip)
            get(added,flag,false) && throw(ArgumentError("unsupported added-token matching flag"))
        end
        vocab[String(added.content)]=Int(added.id)
    end
    tokens=fill("",maximum(values(vocab))+1)
    for (s,i) in vocab
        isempty(tokens[i+1]) || throw(ArgumentError("duplicate token ID"))
        tokens[i+1]=s
    end
    all(!isempty,tokens) || throw(ArgumentError("non-contiguous vocabulary"))
    ranks=Dict{Tuple{String,String},Int}()
    for (i,merge) in enumerate(data.model.merges)
        pieces=merge isa AbstractString ? split(String(merge),' ';limit=2) : String.(merge)
        length(pieces)==2 || throw(ArgumentError("invalid BPE merge"))
        ranks[(pieces[1],pieces[2])]=i-1
    end
    enc,dec=BonsaiRunner._bytemap()
    all(c->haskey(vocab,string(c)),enc) || throw(ArgumentError("missing byte token"))
    types=ones(Int,length(tokens))
    special=Pair{String,Int}[]
    for added in data.added_tokens
        push!(special,String(added.content)=>Int(added.id))
        added.special && (types[Int(added.id)+1]=3)
    end
    sort!(special;by=p->(-ncodeunits(first(p)),first(p)))
    bpe=BonsaiRunner.BonsaiTokenizer(tokens,vocab,ranks,types,enc,dec,special,
        vocab["<|endoftext|>"],vocab["<|im_end|>"])
    Tokenizer(bpe)
end

encode(t::Tokenizer,text::AbstractString)=BonsaiRunner.encode(t.bpe,Unicode.normalize(text,:NFC))
decode(t::Tokenizer,ids;skipspecial=true)=BonsaiRunner.decode(t.bpe,ids;skipspecial)

"""Render the pinned Qwen3.5 template's plain-text role subset.

Historical reasoning is removed before the last genuine user query; assistant
messages after that query retain an explicit reasoning block. Tool-role and
multimodal messages remain unsupported until independently qualified.
"""
function render_chat(messages;thinking=false)
    isempty(messages) && throw(ArgumentError("no messages"))
    parsed=map(messages) do m
        role=string(m isa Pair ? first(m) : m.role)
        role in ("system","user","assistant") || throw(ArgumentError("only text chat roles supported"))
        content=m isa Pair ? last(m) : m.content
        content isa AbstractString || throw(ArgumentError("plain text content required"))
        reasoning=m isa Pair || !hasproperty(m,:reasoning_content) ? nothing : m.reasoning_content
        (;role,content=String(strip(content)),reasoning)
    end
    lastquery=findlast(parsed) do m
        m.role=="user" && !(startswith(m.content,"<tool_response>") && endswith(m.content,"</tool_response>"))
    end
    lastquery===nothing && throw(ArgumentError("no user query"))
    io=IOBuffer()
    for (i,m) in enumerate(parsed)
        m.role=="system" && i!=1 && throw(ArgumentError("system message must be first"))
        content=m.content
        print(io,"<|im_start|>",m.role,'\n')
        if m.role=="assistant"
            reasoning=""
            if m.reasoning isa AbstractString
                reasoning=String(m.reasoning)
            elseif occursin("</think>",content)
                parts=split(content,"</think>";keepempty=true)
                reasoning=String(lstrip(last(split(rstrip(first(parts),'\n'),"<think>";keepempty=true)),'\n'))
                content=String(lstrip(last(parts),'\n'))
            end
            i>lastquery && print(io,"<think>\n",strip(reasoning),"\n</think>\n\n")
        end
        print(io,content,"<|im_end|>\n")
    end
    print(io,"<|im_start|>assistant\n<think>\n")
    thinking || print(io,"\n</think>\n\n")
    String(take!(io))
end

chat_ids(t::Tokenizer,messages;thinking=false)=encode(t,render_chat(messages;thinking))

end
