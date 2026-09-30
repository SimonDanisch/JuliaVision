# The BonsaiRunner README example: one question, greedy, with the prompt and
# decode rates measured separately.
# Explicit names: WhisperRunner also exports `encode`, and Mantle `release!`.
using BonsaiRunner: BonsaiRunner, Bonsai2, session, chatprompt, prefill!, step!, generate, release!

const BONSAI_QUESTION = "In three sentences: why does a Vulkan compute shader need a pipeline barrier " *
                        "between two dispatches that share a buffer?"

function bonsai_examples(; max_tokens = 200)
    model = Bonsai2(; context = 4096)
    tk = model.tokenizer
    ids = BonsaiRunner.encode(tk, chatprompt(["user" => BONSAI_QUESTION]; thinking = false))
    s = session(model)
    generate(s, ids; max_tokens = 4)                  # compile and warm both paths
    release!(s)

    s = session(model)
    tprefill = @elapsed logits = prefill!(s, ids)
    out = Int[]
    tdecode = @elapsed for _ in 1:max_tokens
        id = BonsaiRunner._greedy(s, logits)
        push!(out, id)
        id == tk.eos && break
        logits = step!(s, id)
    end
    release!(s)
    return (; answer = BonsaiRunner.decode(tk, out), prompt_tokens = length(ids), tokens = length(out),
            prefill_tok_s = length(ids) / tprefill, decode_tok_s = length(out) / tdecode)
end

abspath(PROGRAM_FILE) == (@__FILE__) && bonsai_examples()
