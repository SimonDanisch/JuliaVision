using JSON3, SHA, Test

function lexical_budget_metrics(text::AbstractString;n=6)
    words=split(text);counts=Dict{Tuple,Int}()
    for i in 1:max(0,length(words)-n+1)
        key=Tuple(words[i:i+n-1]);counts[key]=get(counts,key,0)+1
    end
    windows=max(0,length(words)-n+1)
    repeated=sum((count-1 for count in values(counts));init=0)
    (;characters=length(text),words=length(words),ngram_width=n,ngram_windows=windows,
        repeated_window_fraction=windows==0 ? 0.0 : repeated/windows,
        highest_window_frequency=maximum(values(counts);init=0))
end

function audit_generation_budgets()
    @test lexical_budget_metrics("one two three four five six").repeated_window_fraction==0
    @test lexical_budget_metrics(join(fill("one two three four five six",20)," ")).repeated_window_fraction>0.8
    @test lexical_budget_metrics("").highest_window_frequency==0
    rows=Any[]
    for directory in ("vulkan_kernel_results","kernel_transfer_results","kernel_transfer_budget_results")
        path=joinpath(@__DIR__,directory);isdir(path) || continue
        for name in sort(readdir(path))
            endswith(name,".json") && filesize(joinpath(path,name))<=16777216 || continue
            packet=JSON3.read(read(joinpath(path,name),String))
            hasproperty(packet,:answer) && hasproperty(packet,:budget_tokens) || continue
            push!(rows,(;file=joinpath(directory,name),generation_sha256=bytes2hex(sha256(read(joinpath(path,name)))),
                task=packet.task,arm=packet.arm,budget=packet.budget_tokens,generated=packet.generated_tokens,
                truncated=packet.truncated,finished_thinking=packet.finished_thinking,code_bytes=ncodeunits(String(packet.code)),
                lexical=lexical_budget_metrics(String(packet.answer))))
        end
    end
    report=(;scope="descriptive budget/lexical audit ONLY; no hidden thinking text reproduced, no correctness or loop classifier",
        limitation="lexical repetition can be legitimate; truncation does not establish quantization error, missing API knowledge, or inability with a larger budget",
        model_card=(;url="https://huggingface.co/Qwen/Qwen3.5-4B#best-practices",checked="2026-10-24",
            precise_coding=(;temperature=0.6,top_p=0.95,top_k=20,presence_penalty=0.0,repetition_penalty=1.0),
            recommended_most_queries_max_output=32768,
            conclusion="current sampled coding parameters match the card;2048 and6144 token caps do not reproduce its recommended output budget"),
        rows,source_sha256=bytes2hex(sha256(read(@__FILE__))))
    open(io->JSON3.pretty(io,report),joinpath(@__DIR__,"generation_budget_audit.json"),"w");report
end
