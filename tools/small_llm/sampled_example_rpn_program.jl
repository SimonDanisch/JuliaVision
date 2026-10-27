isdefined(@__MODULE__,:sampled_rpn_program)||include("sampled_rpn_program.jl")
isdefined(@__MODULE__,:RPNExampleConstraintGPU)||include("rpn_example_constraint_gpu.jl")

"""Experimental grammar + PUBLIC-example finish admission in one decode graph.

Both arms must receive the SAME examples in their prompt. enabled=false only
disables the numeric mask, not the grammar, numeric tracking or graph passes.
Empty compatible-token sets are retained as failed attempts. No EOS/arithmetic
repair, retry or private grading feedback. Physical and full-model identity
qualification are required before a task-quality comparison.
"""
function sampled_example_rpn_program(model,tokenizer,messages,bank::RPNExampleSemantics.Bank;
        enabled=true,max_tokens=512,operation_budget=32,seed=UInt32(20261024),
        topk=20,temperature=0.6f0,top_p=0.95f0,presence=0f0,output_chunk=128)
    enabled isa Bool||throw(ArgumentError("Boolean numeric control required"))
    max_tokens isa Integer&&!(max_tokens isa Bool)&&0<max_tokens<=8192||throw(ArgumentError("invalid token budget"))
    operation_budget isa Integer&&!(operation_budget isa Bool)&&1<=operation_budget<=1024||throw(ArgumentError("invalid operation budget"))
    output_chunk isa Integer&&!(output_chunk isa Bool)&&output_chunk>0||throw(ArgumentError("invalid output polling chunk"))
    grammar=RPNPrefixGrammar.Grammar(s->SmallLLMChecks.QwenText.encode(tokenizer,s),
        Int(model.config.vocab_size),Int(tokenizer.bpe.eos);max_depth=Int(bank.max_depth),budget=operation_budget)
    any(ids->tokenizer.bpe.bos in ids,grammar.sequences)&&throw(ArgumentError("grammar contains BOS"))
    bound=maximum(length,grammar.sequences)*(operation_budget+Int(bank.max_depth)+1)
    max_tokens>=bound||throw(ArgumentError("token budget cannot cover finite grammar bound"))
    sampler=GPUSampling.State(model.device,Int(model.config.vocab_size);
        seed,topk,temperature,top_p,presence,fallback=tokenizer.bpe.eos)
    state=nothing;examples=nothing
    try
        state=RPNPrefixGrammarGPU.State(model.device,grammar;enabled=true)
        examples=RPNExampleConstraintGPU.State(model.device,grammar,bank;enabled)
        GPUSampling.activate!(sampler,false)
        answer=generate_julia_answer(model,tokenizer,messages;max_tokens,thinking=false,output_chunk,
            logit_hook=(g,l)->RPNExampleConstraintGPU.declare!(g,l,state,examples,sampler),
            before_last_prompt=()->GPUSampling.activate!(sampler,true))
        draws=Int(only(Array(sampler.counter)))
        draws==answer.decode_steps_executed+1||error("prompt consumed sampling draws")
        raw_status=only(Array(examples.raw_status));sampler_status=only(Array(sampler.status))
        rejected=only(Array(state.rejected));done=only(Array(state.done));depth=only(Array(state.depth))
        matched=RPNExampleConstraint.finish_allowed(Array(examples.stack),bank.expected,Array(examples.status),
            depth,bank.max_depth,Int32(length(bank.expected)),bank.atol,bank.rtol)
        parsed=RPNPointwiseInterface.parse_program(answer.answer;budget=operation_budget,max_depth=Int(bank.max_depth))
        complete=raw_status==0&&sampler_status==0&&rejected==0&&done==1&&!answer.truncated&&parsed.accepted
        reason=raw_status!=0 ? "invalid_raw_neural_logits" : sampler_status!=0 ?
            "no_admissible_token_or_sampler_rejection" : !complete ? "incomplete_or_invalid_program" :
            enabled&&!matched ? "numeric_state_disagreement" : "completed"
        accepted=complete&&(!enabled||matched)
        (;answer...,constraint=(;enabled,accepted,reason,public_examples_matched=matched,
            raw_status,sampler_status,rejected,done,depth,operations=only(Array(state.operations)),
            operation_budget,max_depth=Int(bank.max_depth),finite_token_bound=bound,
            examples_sha256=bytes2hex(sha256(JSON3.write((;bank.features,bank.expected,bank.case_offsets,
                bank.max_depth,bank.atol,bank.rtol))))),
            sampling=(;seed=Int(seed),topk,temperature,top_p,presence,draws,
                policy="GPU-resident grammar and public-example mask/state; completed-output polling only; no next-token host feedback. Rejection retained, not repaired."))
    finally
        examples===nothing||close(examples)
        state===nothing||close(state)
        close(sampler)
    end
end
