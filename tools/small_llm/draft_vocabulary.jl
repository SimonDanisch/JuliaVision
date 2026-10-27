# PARKED (2026-10-27), not on the main branches: with `Q36.draftvocabulary!(model,
# read_draft_vocabulary())` a draft costs 0.7-1.2 ms instead of ~2.5 ms, and the
# gathered head's rows are bit-identical to the full head's, but MTP acceptance
# fell from 82% to 64% (k=3) and 90% to 72% (k=2) on the benchmark prompts, far
# more than the 0.5% coverage loss explains. The cause is not known; find it
# before using this.
# The tokens Qwen3.6's MTP drafts are restricted to (`Q36.draftvocabulary!`):
# the most frequent in what the model writes, counted over its own answers in
# the study results (reasoning and Julia code) and over Julia Base's sources,
# ties in token id order (BPE merge order), and every special token. A third of
# the kernel-tool answers counted, the other two thirds held out: 16384 tokens
# cover 99.49% of the held-out tokens, 8192 cover 99.19%, 32768 cover 99.75%.
isdefined(@__MODULE__,:basecorpus) || include("benchmark_long_context.jl")
using JSON3

const QWEN36_DRAFT_VOCABULARY=joinpath(@__DIR__,"qwen36_draft_vocabulary.txt")
const DRAFT_SOURCES=("kernel_tools_v2_results","julia_tasks_results","bigcodebench_x_results")

"""How often each token (index id + 1) occurs in the answers of `sources` and in
the first `base` tokens of Julia Base."""
function count_draft_tokens(tokenizer;sources=DRAFT_SOURCES,base=300_000)
    T=SmallLLMChecks.QwenText;counts=zeros(Int,length(tokenizer.bpe.tokentostr))
    for d in sources,file in sort(readdir(joinpath(@__DIR__,d);join=true))
        endswith(file,".json") || continue
        r=JSON3.read(read(file,String))
        r isa JSON3.Object&&get(r,:answer,nothing) isa AbstractString || continue
        for id in T.encode(tokenizer,String(r.answer));counts[id+1]+=1;end
    end
    for id in basecorpus(tokenizer,base);counts[id+1]+=1;end
    counts
end

"""The `n` draft tokens: the special ones, then the most frequent."""
function draft_tokens(tokenizer,counts;n=16384)
    special=sort!(last.(tokenizer.bpe.special))
    n>length(special) || throw(ArgumentError("$n draft tokens leave no room after $(length(special)) special ones"))
    rest=filter(id->!(id in special),sortperm(counts;rev=true,alg=Base.Sort.DEFAULT_STABLE).-1)
    sort!([special;rest[1:n-length(special)]])
end

write_draft_vocabulary(tokenizer;n=16384,path=QWEN36_DRAFT_VOCABULARY,kw...)=
    (write(path,join(draft_tokens(tokenizer,count_draft_tokens(tokenizer;kw...);n),'\n'),'\n');path)
read_draft_vocabulary(path=QWEN36_DRAFT_VOCABULARY)=parse.(Int,filter(!isempty,readlines(path)))
