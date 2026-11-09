# Resident line-protocol worker for bonsai_server.py.
using BonsaiRunner, Mantle, JSON3

mutable struct CachedConversation
    session::BonsaiSession
    tokens::Vector{Int}
    logits::Any
    used::Int
end
mutable struct BonsaiService
    model::Bonsai2
    conversations::Vector{CachedConversation}
    clock::Int
end
emit(x) = (println("BONSAI_EVENT:", JSON3.write(x)); flush(stdout))

function reset!(entry::CachedConversation)
    s = entry.session
    KernelAbstractions = BonsaiRunner.KernelAbstractions
    KernelAbstractions.synchronize(s.model.backend)
    for buffers in values(s.recurrent), buffer in buffers
        fill!(buffer, zero(eltype(buffer)))
    end
    s.position = 0
    empty!(entry.tokens)
    entry.logits = nothing
    entry
end

function choose_conversation!(service::BonsaiService, ids; reuse=true)
    # Recurrent state cannot be rewound by changing a position: reuse only an
    # entire resident prefix. Divergent/shortened histories reset all recurrence.
    matches = if reuse
        filter(service.conversations) do entry
        n = length(entry.tokens)
        n > 0 && n <= length(ids) && entry.tokens == view(ids, 1:n)
        end
    else
        CachedConversation[]
    end
    entry = if !isempty(matches)
        matches[argmax(length(e.tokens) for e in matches)]
    elseif length(service.conversations) < 2
        e = CachedConversation(session(service.model), Int[], nothing, 0)
        push!(service.conversations, e)
        e
    else
        reset!(service.conversations[argmin(e.used for e in service.conversations)])
    end
    service.clock += 1
    entry.used = service.clock
    entry
end

function prompt_ids(model, messages)
    # Completed assistant turns must render the same empty thinking block as
    # the generation prompt; otherwise every next turn diverges at that header.
    normalized = map(messages) do message
        role, text = String(message.role), String(message.content)
        if role == "assistant" && !startswith(text, "<think>")
            text = "<think>\n\n</think>\n\n" * text
        end
        role => text
    end
    BonsaiRunner.encode(model.tokenizer, chatprompt(normalized; thinking=false))
end

function answer!(service::BonsaiService, req; send=emit)
    model = service.model
    ids = prompt_ids(model, req.messages)
    n = Int(req.max_tokens)
    length(ids)+n <= model.maxcontext || throw(ArgumentError("Prompt and output exceed 16384 tokens; shorten the conversation"))
    entry = choose_conversation!(service, ids; reuse=get(req, :cache_prompt, true))
    cached = length(entry.tokens)
    send((;type="start", prompt_tokens=length(ids), cached_tokens=cached))
    s = entry.session
    try
        started = time_ns()
        if cached < length(ids)
            entry.logits = prefill!(s, view(ids, cached+1:length(ids)))
            entry.tokens = copy(ids)
        end
        id = BonsaiRunner._greedy(s, entry.logits) # wait for prompt GPU work
        prefill_seconds = (time_ns()-started)/1e9
        started = time_ns()
        out = Int[]; sent = ""; reason = "length"
        for i in 1:n
            if id == model.tokenizer.eos
                reason = "stop"; break
            end
            push!(out, id)
            text = BonsaiRunner.decode(model.tokenizer, out)
            if isvalid(text)
                delta = text[ncodeunits(sent)+1:end]
                isempty(delta) || send((;type="token",text=delta))
                sent = text
            end
            if i < n
                entry.logits = step!(s,id)
                push!(entry.tokens, id)
                id = BonsaiRunner._greedy(s, entry.logits)
            end
        end
        # The final emitted token remains pending; the next prompt ingests it
        # along with the closing assistant delimiter and new user turn.
        decode_seconds = (time_ns()-started)/1e9
        send((;type="done", completion_tokens=length(out), finish_reason=reason,
               cached_tokens=cached, prefill_seconds, decode_seconds))
    catch
        # An interrupted/failed graph may leave recurrence partially advanced.
        reset!(entry)
        rethrow()
    end
end

function main()
    model = Bonsai2(device=Mantle.Device("radeon"), context=16384)
    service = BonsaiService(model, CachedConversation[], 0)
    try
        # Keep the warmup session and its compiled plans in the resident pool.
        answer!(service, (;messages=[(;role="user",content="Say ready.")],max_tokens=2); send=_ -> nothing)
        emit((;type="ready"))
        for line in eachline(stdin)
            try
                answer!(service, JSON3.read(line))
            catch err
                showerror(stderr,err,catch_backtrace()); println(stderr)
                emit((;type="error",message=sprint(showerror,err)))
            end
        end
    finally
        foreach(e -> close(e.session), service.conversations)
        close(model)
    end
end
main()
