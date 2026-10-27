isdefined(@__MODULE__,:generate_julia_answer)||include("run_julia_tasks.jl")
isdefined(@__MODULE__,:RPNPrefixGrammarGPU)||include("rpn_prefix_grammar_gpu.jl")
isdefined(@__MODULE__,:RPNPointwiseInterface)||include("rpn_pointwise_interface.jl")

"""Opt-in finite program generation; does not evaluate the generated program.

Enabled/disabled arms use the SAME graph and sampler, changing one device flag.
All grammar transitions are device resident. Final readbacks verify the run;
they never feed a next token. Full-model identity/hardware gates are required
before using this experimental helper for quality claims.
"""
function sampled_rpn_program(model,tokenizer,messages;enabled=true,max_tokens=512,
        operation_budget=32,max_depth=8,seed=UInt32(20261024),topk=20,temperature=0.6f0,
        top_p=0.95f0,presence=0f0,output_chunk=128)
    enabled isa Bool||throw(ArgumentError("Boolean grammar control required"))
    max_tokens isa Integer && !(max_tokens isa Bool) && 0<max_tokens<=8192||throw(ArgumentError("invalid token budget"))
    operation_budget isa Integer && !(operation_budget isa Bool) && 1<=operation_budget<=1024||throw(ArgumentError("invalid operation budget"))
    max_depth isa Integer && !(max_depth isa Bool) && 2<=max_depth<=64||throw(ArgumentError("invalid stack bound"))
    grammar=RPNPrefixGrammar.Grammar(s->SmallLLMChecks.QwenText.encode(tokenizer,s),
        Int(model.config.vocab_size),Int(tokenizer.bpe.eos);max_depth,budget=operation_budget)
    any(ids->tokenizer.bpe.bos in ids,grammar.sequences)&&throw(ArgumentError("grammar contains a generation stop/BOS token"))
    bound=maximum(length,grammar.sequences)*(operation_budget+max_depth+1)
    enabled&&max_tokens<bound&&throw(ArgumentError("token budget cannot cover the finite grammar bound"))
    sampler=GPUSampling.State(model.device,Int(model.config.vocab_size);
        seed,topk,temperature,top_p,presence,fallback=tokenizer.bpe.eos)
    state=nothing
    try
        state=RPNPrefixGrammarGPU.State(model.device,grammar;enabled)
        GPUSampling.activate!(sampler,false)
        answer=generate_julia_answer(model,tokenizer,messages;max_tokens,thinking=false,output_chunk,
            logit_hook=(g,l)->RPNPrefixGrammarGPU.declare!(g,l,state,sampler),
            before_last_prompt=()->GPUSampling.activate!(sampler,true),
            after_output=()->begin
                only(Array(sampler.status))==0||error("sampler rejected raw/masked logits")
                only(Array(state.rejected))==0||error("grammar transition rejected selected token")
            end)
        draws=Int(only(Array(sampler.counter)))
        draws==answer.decode_steps_executed+1||error("prompt consumed sampler draws")
        snapshot=(;enabled,node=only(Array(state.node)),depth=only(Array(state.depth)),
            operations=only(Array(state.operations)),done=only(Array(state.done)),rejected=only(Array(state.rejected)),
            operation_budget,max_depth,finite_token_bound=bound)
        if enabled
            snapshot.done==1&&!answer.truncated||error("finite grammar did not finish within its registered bound")
            parsed=RPNPointwiseInterface.parse_program(answer.code;budget=operation_budget,max_depth)
            parsed.accepted&&length(parsed.actions)-1==snapshot.operations||error("decoded program disagrees with device grammar state")
        end
        (;answer...,grammar=snapshot,
            sampling=(;kind="exact top-k/nucleus",seed=Int(seed),topk,temperature,top_p,presence,draws,
                policy="GPU-resident grammar mask/stack transition and sampler; completed-output polling only, no token/state feedback from CPU"))
    finally
        state===nothing||close(state)
        close(sampler)
    end
end
