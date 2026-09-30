# The HorizonRunner README example. Needs the local export (`gen/graphs/horizon32b`,
# 65 GB, not an artifact) and the repo's Python environment for the tokenizer,
# which `tools/horizon_tokenize.py` wraps.
isdefined(@__MODULE__, :ExampleCommon) || include(joinpath(@__DIR__, "common.jl"))
using .ExampleCommon
using HorizonRunner   # `generate` qualified: BonsaiRunner exports one too
import JSON3

"One request to `tools/horizon_tokenize.py`: `{\"chat\": text}`, `{\"decode\": ids}`."
function horizontokens(request::Pair)
    python = joinpath(REPO, ".venv", "bin", "python")
    cmd = addenv(`$python $(joinpath(REPO, "tools", "horizon_tokenize.py"))`,
                 "VIDEOEDIT_ROOT" => normpath(joinpath(REPO, "..", "..")))
    out = IOBuffer()
    run(pipeline(cmd; stdin = IOBuffer(JSON3.write(Dict(request))), stdout = out))
    return JSON3.read(String(take!(out)))
end

function horizon_examples(; maxnew = 160)
    dir = joinpath(REPO, "gen", "graphs", "horizon32b")
    HorizonRunner.ready(; dir) || error("no Horizon export at $dir")
    tload = @elapsed m = horizon32b(; dir)
    prompt = horizontokens("chat" => "Explain in two sentences what a KV cache is for.")
    ids = collect(Int, prompt.ids)
    HorizonRunner.generate(m, ids; maxnew = 4, stop = (1, 250019))      # compile the paths
    t = @elapsed out = HorizonRunner.generate(m, ids; maxnew, stop = (1, 250019))
    text = horizontokens("decode" => out).text
    return (; answer = text, load_seconds = tload, prompt_tokens = length(ids),
            tokens = length(out), tok_s = length(out) / t)
end

abspath(PROGRAM_FILE) == (@__FILE__) && horizon_examples()
