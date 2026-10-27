# Member names inside the answer, constrained to what a module defines. Declared
# into the decode graph around GPUSampling: no token or logit readback.
isdefined(@__MODULE__,:GPUSampling) || include("gpu_sampling.jl")

"""After `</think>`, wherever the answer has just written the name of a listed
module and a `.` (`KI.`, `Atomix.`), the next characters must spell a name that
module defines, macros included: `Atomix.@atomic` can be sampled and
`Atomix.atomic_max` cannot. Everything outside a member position is untouched.

The mask walks a CHARACTER trie through every candidate token's bytes, so each
BPE split of a legal name stays legal; a token-path trie would accept only the
canonical split. A legal name is not a correct one: this removes nonexistent
names and nothing else. Each constrained step logs the share of the sampling
distribution (at the sampler's temperature) the mask removed, which says how
strongly the model believed in names that do not exist.
"""
module MemberNameConstraint
import ..GPUSampling
import Mantle
import KernelInterface as KI

const NOCHILD=Int32(-1)
const COMPLETE=Int32(-2)   # the token finished a legal name and left the member position
const ASCII=Int32(128)

"""Host tables: member and module-name tries over ASCII, and every token's bytes."""
struct Tables
    trans::Vector{Int32}      # 128 per member-trie node, 0-based child or -1
    terminal::Vector{Int32}   # 1 where a member name ends
    modtrans::Vector{Int32}   # 128 per module-trie node; node 0 is the root
    modroot::Vector{Int32}    # per module-trie node: the member-trie root of the module named there, or -1
    bytes::Vector{Int32}      # token bytes, one per entry
    offsets::Vector{Int32}    # token t (1-based) is bytes[offsets[t]+1:offsets[t+1]]
    special::Vector{Int32}    # control tokens and ids past the tokenizer's table
    think_end::Int32
    vocabulary::Int
    members::Vector{Pair{String,Vector{String}}}
end

"""Names `M.name` resolves for a module `M`: identifiers and macros, ASCII only."""
membernames(M::Module)=sort!(unique!([String(n) for n in names(M;all=true,imported=true)
    if occursin(r"^@?[A-Za-z_][A-Za-z0-9_!]*$",String(n)) && isdefined(M,n)]))

function trie!(trans,terminal,root,word)
    node=root
    for b in codeunits(word)
        slot=node*ASCII+b+1
        if trans[slot]==NOCHILD
            append!(trans,fill(NOCHILD,ASCII));push!(terminal,0)
            trans[slot]=Int32(length(terminal)-1)
        end
        node=trans[slot]
    end
    terminal[node+1]=1
    node
end

"""Tables for `modules`, pairs of the name the answer writes and the module, so
an alias like `KI` is listed as its own name; `vocabulary` is the logit count."""
function Tables(tokenizer,modules::AbstractVector{<:Pair{String,Module}},vocabulary::Integer)
    bpe=tokenizer.bpe;n=length(bpe.tokentostr)
    n<=vocabulary || throw(ArgumentError("tokenizer larger than the logits"))
    trans=Int32[];terminal=Int32[];roots=Dict{Module,Int32}();members=Pair{String,Vector{String}}[]
    for (_,M) in modules
        haskey(roots,M) && continue
        append!(trans,fill(NOCHILD,ASCII));push!(terminal,0);roots[M]=Int32(length(terminal)-1)
        names=membernames(M);push!(members,string(M)=>names)
        foreach(w->trie!(trans,terminal,roots[M],w),names)
    end
    modtrans=fill(NOCHILD,ASCII);modterminal=Int32[0];ends=Pair{Int32,Int32}[]
    for (name,M) in modules
        occursin(r"^[A-Za-z_][A-Za-z0-9_]*$",name) || throw(ArgumentError("module name $name"))
        push!(ends,trie!(modtrans,modterminal,Int32(0),name)=>roots[M])
    end
    modroot=fill(NOCHILD,length(modterminal))
    for (node,root) in ends;modroot[node+1]=root;end
    bytes=Int32[];offsets=Int32[0];special=zeros(Int32,vocabulary)
    for t in 1:vocabulary
        if t<=n
            bpe.types[t]==3 && (special[t]=1)
            for c in bpe.tokentostr[t]
                b=get(bpe.bytedecoder,c,nothing)
                b===nothing ? append!(bytes,codeunits(string(c))) : push!(bytes,b)
            end
        else
            special[t]=1
        end
        push!(offsets,Int32(length(bytes)))
    end
    think_end=Int32(bpe.strtotoken["</think>"])
    Tables(trans,terminal,modtrans,modroot,bytes,offsets,special,think_end,Int(vocabulary),members)
end

# ── The arithmetic, shared by the device kernels and the host reference ──────
#
# The automaton state is `(phase, lex, modnode, previdentifier, membernode)`:
# `phase` is 1 after `</think>`; `lex` packs comment, string and escape flags,
# the current backtick run and the number of code fences seen; `modnode` walks
# the module-name trie along the current identifier; `membernode` is the member
# trie node while a member name is being written, -1 otherwise.

@inline isident(b)=(Int32('a')<=b<=Int32('z'))|(Int32('A')<=b<=Int32('Z'))|(Int32('0')<=b<=Int32('9'))|
    (b==Int32('_'))|(b==Int32('!'))

"""Whether text at lexer state `lex` is code: not a comment or string literal,
and inside a code fence or before any (an answer may be bare code). Prose and
comments may write `Atomix.jl`; code may not."""
@inline function incode(lex::Int32)
    fences=lex>>Int32(5)
    (lex&Int32(3))==0 && (fences==0 || isodd(fences))
end

"""The lexer state after byte `b`: `#` comments to the end of the line, `"`
strings with backslash escapes, and three backticks outside them as a fence."""
@inline function lexstep(lex::Int32,b::Int32)
    comment=lex&Int32(1);str=(lex>>Int32(1))&Int32(1);esc=(lex>>Int32(2))&Int32(1)
    ticks=(lex>>Int32(3))&Int32(3);fences=lex>>Int32(5)
    if comment!=0
        b==Int32('\n') && (comment=Int32(0))
    elseif str!=0
        if esc!=0
            esc=Int32(0)
        elseif b==Int32('\\')
            esc=Int32(1)
        elseif b==Int32('"')
            str=Int32(0)
        end
    elseif b==Int32('#')
        comment=Int32(1)
    elseif b==Int32('"')
        str=Int32(1)
    end
    if b==Int32('`') && (comment|str)==0
        ticks+=Int32(1)
        ticks==Int32(3) && (ticks=Int32(0);fences=min(fences+Int32(1),Int32(1023)))
    else
        ticks=Int32(0)
    end
    comment|(str<<Int32(1))|(esc<<Int32(2))|(ticks<<Int32(3))|(fences<<Int32(5))
end

"""One byte at member-trie `node`: the child, `COMPLETE` when a finished name is
followed by a non-identifier byte, `NOCHILD` when the byte breaks every name."""
@inline function memberstep(trans,terminal,node::Int32,b::Int32)
    child=b<ASCII ? (@inbounds trans[node*ASCII+b+Int32(1)]) : NOCHILD
    child>=0 && return child
    (@inbounds terminal[node+Int32(1)]!=0) && !isident(b) ? COMPLETE : NOCHILD
end

"""One byte of the answer: the next `(lex, modnode, previdentifier, membernode)`,
and whether the byte broke a member name."""
@inline function bytestep(trans,terminal,modtrans,modroot,phase::Int32,lex::Int32,modnode::Int32,prev::Int32,node::Int32,b::Int32)
    if node>=0
        child=memberstep(trans,terminal,node,b)
        child==NOCHILD && return (lex,modnode,prev,node,true)
        child>=0 && return (lex,modnode,prev,child,false)
        node=NOCHILD     # the name is complete; this byte is ordinary text again
    end
    code=phase!=0 && incode(lex)
    lex=lexstep(lex,b)
    @inbounds if code && isident(b)
        from=prev!=0 ? modnode : Int32(0)
        modnode=from>=0 && b<ASCII ? modtrans[from*ASCII+b+Int32(1)] : NOCHILD
        prev=Int32(1)
    else
        if code && b==Int32('.') && prev!=0 && modnode>=0 && modroot[modnode+Int32(1)]>=0
            node=modroot[modnode+Int32(1)]
        end
        modnode=NOCHILD;prev=Int32(0)
    end
    (lex,modnode,prev,node,false)
end

"""Token `t` (1-based) from `(phase, lex, modnode, previdentifier, membernode)`:
the next state and whether any of its bytes broke a member name. The whole
token is walked, so a trigger and an illegal name inside ONE token
(`.atomic_max`) are caught, not only tokens that start at a member position."""
@inline function tokenstep(trans,terminal,modtrans,modroot,bytes,offsets,special,think_end::Int32,
                           phase::Int32,lex::Int32,modnode::Int32,prev::Int32,node::Int32,t::Int32)
    # `</think>` is an added token without the special flag in Qwen's tables.
    t-Int32(1)==think_end && return (Int32(1),Int32(0),NOCHILD,Int32(0),NOCHILD,node>=0)
    @inbounds first=offsets[t]
    @inbounds last=offsets[t+Int32(1)]
    (@inbounds special[t]!=0)|(first==last) && return (phase,lex,NOCHILD,Int32(0),NOCHILD,node>=0)
    j=first
    while j<last
        lex,modnode,prev,node,broke=bytestep(trans,terminal,modtrans,modroot,phase,lex,modnode,prev,node,@inbounds bytes[j+Int32(1)])
        broke && return (phase,lex,modnode,prev,node,true)
        j+=Int32(1)
    end
    (phase,lex,modnode,prev,node,false)
end

# ── Device state and kernels ─────────────────────────────────────────────────

"""One generation's constraint. `state` is the automaton state followed by
`[member steps, other refusing steps]`. `removed` logs the removed share of each
step that starts at a member position, the first `capacity` of them; `other`
holds the largest share removed at any other answer step, where only tokens
carrying a trigger and an illegal name together are refused."""
struct State
    state
    removed
    other
    trans
    terminal
    modtrans
    modroot
    bytes
    offsets
    special
    think_end::Int32
    vocabulary::Int
    capacity::Int
end

initialstate(thinking::Bool)=(Int32(thinking ? 0 : 1),Int32(0),NOCHILD,Int32(0),NOCHILD)

function State(device,t::Tables;thinking::Bool=true,capacity::Int=512)
    owned=Any[];upload(x)=(b=Mantle.Buffer(device,x);push!(owned,b);b)
    try
        State(upload(Int32[initialstate(thinking)...,0,0]),upload(zeros(Float32,capacity)),upload(Float32[0]),upload(t.trans),
            upload(t.terminal),upload(t.modtrans),upload(t.modroot),upload(t.bytes),upload(t.offsets),
            upload(t.special),t.think_end,t.vocabulary,capacity)
    catch
        foreach(Mantle.free!,reverse(owned));rethrow()
    end
end
Base.close(s::State)=(foreach(Mantle.free!,(s.state,s.removed,s.other,s.trans,s.terminal,s.modtrans,s.modroot,s.bytes,s.offsets,s.special));nothing)

"""What the generation's constraint did: constrained steps and their removed shares."""
function report(s::State)
    st=Array(s.state);steps=Int(st[6])
    (;member_steps=steps,removed_share=Float64.(Array(s.removed)[1:min(steps,s.capacity)]),
     removed_log_truncated=steps>s.capacity,other_refusing_steps=Int(st[7]),
     other_max_share=Float64(only(Array(s.other))),answered=st[1]!=0)
end

function allowed!(allowed,state,trans,terminal,modtrans,modroot,bytes,offsets,special,active,think_end::Int32,n::Int32)
    i=Int32(KI.get_global_id().x)
    if i<=n
        @inbounds refuse=active[1]!=0 && state[1]!=0 &&
            tokenstep(trans,terminal,modtrans,modroot,bytes,offsets,special,think_end,
                state[1],state[2],state[3],state[4],state[5],i)[6]
        @inbounds allowed[i]=Int32(refuse ? 0 : 1)
    end
    nothing
end

# One workgroup. The share is at the sampler's temperature, before presence.
function removed_share!(state,removed,other,logits,allowed,active,n::Int32,temperature::Float32,capacity::Int32)
    t=Int32(KI.get_local_id().x)
    shared=KI.localmemory(Float32,Val((256,)),Val(1))
    cut=KI.localmemory(Float32,Val((256,)),Val(2))
    live=@inbounds active[1]!=0 && state[1]!=0
    m=-Inf32;i=t
    while live && i<=n
        m=max(m,@inbounds logits[i]);i+=Int32(256)
    end
    @inbounds shared[t]=m
    KI.barrier()
    stride=Int32(128)
    while stride>0
        t<=stride && (@inbounds shared[t]=max(shared[t],shared[t+stride]))
        KI.barrier();stride>>=1
    end
    m=@inbounds shared[1]
    KI.barrier()
    total=0f0;gone=0f0;i=t
    while live && i<=n
        e=exp((@inbounds(logits[i])-m)/temperature);total+=e
        (@inbounds allowed[i])==0 && (gone+=e);i+=Int32(256)
    end
    @inbounds begin shared[t]=total;cut[t]=gone;end
    KI.barrier()
    stride=Int32(128)
    while stride>0
        t<=stride && (@inbounds begin shared[t]+=shared[t+stride];cut[t]+=cut[t+stride];end)
        KI.barrier();stride>>=1
    end
    if t==1 && live
        @inbounds begin
            share=cut[1]/shared[1]
            if state[5]>=0
                k=state[6]
                k<capacity && (removed[k+Int32(1)]=share)
                state[6]=k+Int32(1)
            elseif cut[1]>0f0
                state[7]+=Int32(1);other[1]=max(other[1],share)
            end
        end
    end
    nothing
end

function apply!(logits,allowed,n::Int32)
    i=Int32(KI.get_global_id().x)
    i<=n && (@inbounds allowed[i])==0 && (@inbounds logits[i]=-Inf32)
    nothing
end

function advance!(state,trans,terminal,modtrans,modroot,bytes,offsets,special,selected,status,active,think_end::Int32)
    if KI.get_global_id().x==1
        @inbounds if active[1]!=0 && status[1]==0
            p,l,mo,pr,no,_=tokenstep(trans,terminal,modtrans,modroot,bytes,offsets,special,think_end,
                state[1],state[2],state[3],state[4],state[5],selected[1]+Int32(1))
            state[1]=p;state[2]=l;state[3]=mo;state[4]=pr;state[5]=no
        end
    end
    nothing
end

"""Mask, sample and advance inside the decode recording. Inactive sampling
(prompt chunks) leaves every pass a no-op."""
function declare!(g,logits,s::State,sampler)
    n=s.vocabulary
    length(logits)==n==sampler.vocabulary || throw(DimensionMismatch("constraint vocabulary"))
    allowed=Mantle.Transient.Buffer(g,Int32,(n,))
    Mantle.dispatch!(g,allowed!,(allowed,s.state,s.trans,s.terminal,s.modtrans,s.modroot,s.bytes,s.offsets,s.special,
        sampler.active,s.think_end,Int32(n)),n;group=256,name="members.allowed")
    Mantle.dispatch!(g,removed_share!,(s.state,s.removed,s.other,logits,allowed,sampler.active,Int32(n),sampler.temperature,Int32(s.capacity)),
        256;group=256,name="members.removed_share")
    Mantle.dispatch!(g,apply!,(logits,allowed,Int32(n)),n;group=256,name="members.mask")
    GPUSampling.declare!(g,logits,sampler)
    Mantle.dispatch!(g,advance!,(s.state,s.trans,s.terminal,s.modtrans,s.modroot,s.bytes,s.offsets,s.special,
        sampler.selected,sampler.status,sampler.active,s.think_end),1;group=1,name="members.advance")
    nothing
end

# ── Host reference ───────────────────────────────────────────────────────────

hoststep(t::Tables,st,id)=tokenstep(t.trans,t.terminal,t.modtrans,t.modroot,t.bytes,t.offsets,t.special,t.think_end,st...,Int32(id+1))

"""Feed `ids` (0-based) on the host from the start of the answer (or of the
thinking): the first token the mask would refuse (1-based position, or
`nothing`) and how many tokens started at a member position."""
function trace(t::Tables,ids;thinking::Bool=false)
    st=initialstate(thinking);constrained=0
    for (k,id) in enumerate(ids)
        st[5]>=0 && (constrained+=1)
        r=hoststep(t,st,id);st=r[1:5]
        r[6] && return (;refused=k,constrained)
    end
    (;refused=nothing,constrained)
end

"""The host automaton state after `ids`, as the device keeps it."""
hoststate(t::Tables,ids;thinking::Bool=false)=foldl((st,id)->hoststep(t,st,id)[1:5],ids;init=initialstate(thinking))

"""Which tokens the mask allows from host state `st`."""
allowedset(t::Tables,st)=[!(st[1]!=0 && hoststep(t,st,i-1)[6]) for i in 1:t.vocabulary]
end
