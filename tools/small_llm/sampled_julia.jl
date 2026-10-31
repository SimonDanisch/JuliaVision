# Inclusion is CPU-only; creating a state requires an explicitly supplied model
# device.
isdefined(@__MODULE__,:GPUSampling) || include("gpu_sampling.jl")

function sampled_julia_answer(model,tokenizer,messages;max_tokens=2048,thinking=true,
        seed=UInt32(1),topk=20,temperature=0.6f0,top_p=0.95f0,presence=0f0,output_chunk=128)
    state=GPUSampling.State(model.device,Int(model.config.vocab_size);
        seed,topk,temperature,top_p,presence,fallback=tokenizer.bpe.eos)
    try
        GPUSampling.activate!(state,false)
        answer=generate_julia_answer(model,tokenizer,messages;max_tokens,thinking,output_chunk,
            logit_hook=(g,l)->GPUSampling.declare!(g,l,state),
            before_last_prompt=()->GPUSampling.activate!(state,true),
            after_output=()->(only(Array(state.status))==0 || error("sampler rejected nonfinite/all-masked logits")))
        draws=Int(only(Array(state.counter)))
        draws==answer.decode_steps_executed+1 || error("unexpected sampler counter: prompt draws must be disabled")
        (;answer...,sampling=(;kind="exact top-k/nucleus",seed=Int(seed),topk,temperature,top_p,presence,draws,
            policy="GPU-resident RNG/history/selection inside the recorded decode graph; output polling only at batch boundaries"))
    finally
        close(state)
    end
end
