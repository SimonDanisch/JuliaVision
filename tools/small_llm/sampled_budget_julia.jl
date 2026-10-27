isdefined(@__MODULE__,:sampled_julia_answer) || include("sampled_julia.jl")
isdefined(@__MODULE__,:ReasoningBudget) || include("reasoning_budget.jl")

function sampled_budget_julia_answer(model,tokenizer,messages;max_tokens=2048,thinking=true,thinking_budget=1024,
        seed=UInt32(1),topk=20,temperature=0.6f0,top_p=0.95f0,presence=0f0,output_chunk=128)
    thinking || error("budget controller requires an explicitly open thinking segment")
    Q=SmallLLMChecks.QwenText
    end_ids=Q.encode(tokenizer,"</think>")
    length(end_ids)==1 && Q.decode(tokenizer,end_ids;skipspecial=false)=="</think>" || error("closing marker must be exactly one native token")
    sampler=nothing;budget=nothing
    try
        sampler=GPUSampling.State(model.device,Int(model.config.vocab_size);
            seed,topk,temperature,top_p,presence,fallback=tokenizer.bpe.eos)
        budget=ReasoningBudget.State(model.device,Int(model.config.vocab_size),only(end_ids);limit=thinking_budget)
        GPUSampling.activate!(sampler,false)
        result=generate_julia_answer(model,tokenizer,messages;max_tokens,thinking=true,output_chunk,
            logit_hook=(graph,logits)->begin
                ReasoningBudget.declare!(graph,logits,budget,sampler)
                GPUSampling.declare!(graph,logits,sampler)
            end,
            before_last_prompt=()->GPUSampling.activate!(sampler,true),
            after_output=()->(only(Array(sampler.status))==0 || error("sampler rejected generation")))
        draws=Int(only(Array(sampler.counter)))
        draws==result.decode_steps_executed+1 || error("prompt consumed unexpected draws")
        forced=Int(only(Array(budget.forced_count)));0<=forced<=1 || error("controller forced more than one closing token")
        (;result...,thinking_budget=Int(thinking_budget),forced_closures=forced,
            sampling=(;kind="exact top-k/nucleus with separately declared thinking-budget control",seed=Int(seed),topk,temperature,top_p,presence,draws),
            scope="budget steering, not knowledge or proof of improved reasoning")
    finally
        budget===nothing || close(budget)
        sampler===nothing || close(sampler)
    end
end
