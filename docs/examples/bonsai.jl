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
    # Up to the first token: `prefill!` returns once the prompt is queued, and the
    # pick is what waits for it. Timing `prefill!` alone put the prompt's GPU time
    # into the first decode step, a second of it on an M5.
    tprefill = @elapsed begin
        logits = prefill!(s, ids)
        out = [BonsaiRunner._greedy(s, logits)]
    end
    tdecode = @elapsed while length(out) < max_tokens && out[end] != tk.eos
        logits = step!(s, out[end])
        push!(out, BonsaiRunner._greedy(s, logits))
    end
    release!(s)
    return (; answer = BonsaiRunner.decode(tk, out), prompt_tokens = length(ids), tokens = length(out),
            prefill_tok_s = length(ids) / tprefill, decode_tok_s = (length(out) - 1) / tdecode)
end

abspath(PROGRAM_FILE) == (@__FILE__) && bonsai_examples()
