using JSON3,Statistics,LinearAlgebra

function attention_logit_metrics(reference,candidate,target::Integer)
    length(reference)==length(candidate) || throw(DimensionMismatch("vocabulary differs"))
    0<=target<length(reference) || throw(ArgumentError("target token outside vocabulary"))
    all(isfinite,reference) && all(isfinite,candidate) || throw(ArgumentError("nonfinite logits"))
    # FP64 log-softmax keeps tiny probabilities from corrupting KL. This is
    # deliberate host readback/scoring for evaluation, never the decode path.
    a=Float64.(reference);b=Float64.(candidate)
    a.-=maximum(a);b.-=maximum(b)
    a.-=log(sum(exp,a));b.-=log(sum(exp,b))
    kl=sum(exp(a[i])*(a[i]-b[i]) for i in eachindex(a))
    (;reference_nll=-a[target+1],candidate_nll=-b[target+1],kl,
      top1_agrees=argmax(reference)==argmax(candidate),
      maxabs=maximum(abs,reference-candidate),relrms=norm(reference-candidate)/max(norm(reference),eps(Float32)))
end

"""Teacher-forced distribution comparison over a caller-supplied token corpus.

Use a separate unquantized reference_model to measure total quantization error,
or the same model to isolate cache/kernel changes. This is not a throughput or
reasoning benchmark. Input IDs must be fixed before comparing variants.
"""
function compare_qwen35_attention_quality(model,ids;reference_model=model,output=nothing,
                                         prefix_ids=Int[],deadline=Inf)
    Q=Qwen35Experiment
    length(ids)>=2 || throw(ArgumentError("need input and target tokens"))
    all(t->0<=t<Int(model.config.vocab_size),ids) &&
        all(t->0<=t<Int(model.config.vocab_size),prefix_ids) || throw(ArgumentError("invalid token ID"))
    model.config.vocab_size==reference_model.config.vocab_size || throw(DimensionMismatch("vocabulary differs"))
    capacity=nextpow(2,length(ids)+length(prefix_ids))
    rows=Any[]
    result(complete)=(;scope="teacher-forced distribution drift, not reasoning quality; explicit readbacks excluded from performance benchmarks",
        complete,quantize=string(model.quantize),reference_quantize=string(reference_model.quantize),
        same_model_reference=reference_model===model,input_ids=Int.(ids),prefix_ids=Int.(prefix_ids),rows)
    for (T,mode) in ((Float32,:split),(Float32,:grouped),(Float16,:split),(Float16,:grouped))
        time()<deadline || error("quality window exhausted before next variant")
        reference=Q.session(reference_model;capacity,kv_type=Float32,attention=:split)
        candidate=nothing
        try
            candidate=Q.session(model;capacity,kv_type=T,attention=mode)
            if !isempty(prefix_ids)
                Q.prefill!(reference,prefix_ids;chunk=32)
                Q.prefill!(candidate,prefix_ids;chunk=32)
            end
            scores=Any[]
            for i in 1:length(ids)-1
                time()<deadline || error("quality window exhausted before next token")
                a=Array(Q.step!(reference,ids[i]))
                b=Array(Q.step!(candidate,ids[i]))
                push!(scores,attention_logit_metrics(a,b,ids[i+1]))
            end
            rnll=mean(s.reference_nll for s in scores);cnll=mean(s.candidate_nll for s in scores)
            push!(rows,(;cache=string(T),attention=string(mode),tokens=length(scores),
                reference_nll=rnll,candidate_nll=cnll,nll_delta=cnll-rnll,
                reference_perplexity=exp(rnll),candidate_perplexity=exp(cnll),
                mean_kl=mean(s.kl for s in scores),max_kl=maximum(s.kl for s in scores),
                top1_agreement=mean(s.top1_agrees for s in scores),
                max_logit_abs_error=maximum(s.maxabs for s in scores),
                mean_logit_relrms=mean(s.relrms for s in scores)))
            println(last(rows));flush(stdout)
            output===nothing || open(io->JSON3.pretty(io,result(false)),output,"w")
        finally
            candidate===nothing || close(candidate)
            close(reference)
        end
    end
    output===nothing || open(io->JSON3.pretty(io,result(true)),output,"w")
    result(true)
end
