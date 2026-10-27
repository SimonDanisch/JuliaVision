# An inference interface for external evidence. The forward pass stays in
# Qwen35Experiment. declare! adds evidence to the same decode graph before
# selection; Decoder is a standalone arithmetic fixture for small tests.
module KnowledgeBias
import Mantle, BonsaiRunner
import KernelInterface as KI

struct Decoder{S,B,P}
    session::S
    bias::B
    plan::P
end

struct Evidence{B}
    bias::B
end
Evidence(device,vocabulary::Integer)=Evidence(Mantle.Buffer(device,zeros(Float32,vocabulary)))
Base.close(e::Evidence)=(Mantle.free!(e.bias);nothing)

function add_log_evidence!(logits,bias,n::Int32)
    i=Int32(KI.get_global_id().x)
    i<=n && (@inbounds logits[i]+=bias[i])
    nothing
end

"""Declare evidence into a model session's graph through its logit_hook.

    evidence = Evidence(model.device, model.config.vocab_size)
    s = Qwen35Experiment.session(model;
        logit_hook=(g, logits)->declare!(g, logits, evidence))

Keep evidence alive until after closing s. Updating evidence queues a host store
in the next decode submission; applying it requires no separate graph replay.
"""
function declare!(g,logits,e::Evidence)
    length(logits)==length(e.bias) || throw(ArgumentError("evidence vocabulary mismatch"))
    Mantle.dispatch!(g,add_log_evidence!,(logits,e.bias,Int32(length(logits))),
                     length(logits);group=256,name="external_log_evidence")
    nothing
end

function Decoder(s)
    bias=Mantle.Buffer(s.model.device,zeros(Float32,length(s.logits)))
    try
        g=Mantle.Graph(s.model.device)
        Mantle.dispatch!(g,add_log_evidence!,(s.logits,bias,Int32(length(s.logits))),
                         length(s.logits);group=256,name="external_log_evidence")
        BonsaiRunner.declare_argmax!(g,s.best,s.logits;name="conditioned_argmax")
        Decoder(s,bias,Mantle.record!(Mantle.Plan(g)))
    catch
        Mantle.free!(bias);rethrow()
    end
end

"""
    set_evidence!(decoder, scores; default=0)

Add sparse, zero-based token log scores to the model's logits on the next
decode (or standalone fixture `apply!`). Exact log likelihoods implement a
Bayesian update over this finite token hypothesis space. Arbitrary scores are
steering, not calibrated inference.
Use `default=-Inf32` and zero scores on allowed IDs for a hard constraint.
The evidence must be valid for the current prefix; multi-token values need an
incremental parser or token trie rather than a single allowed-ID set.
"""
function set_evidence!(d::Union{Decoder,Evidence},scores;default=0f0)
    host=fill(Float32(default),length(d.bias))
    for (id,score) in scores
        0<=id<length(host) || throw(ArgumentError("evidence token outside vocabulary"))
        host[Int(id)+1]=Float32(score)
    end
    all(x->isfinite(x)||x===-Inf32,host) || throw(ArgumentError("scores must be finite or -Inf"))
    any(isfinite,host) || throw(ArgumentError("evidence excludes every token"))
    d.bias[:]=host
    d
end

# Apply once to fresh forward logits. Applying twice counts evidence twice.
apply!(d::Decoder)=(Mantle.run!(d.plan);d.session.logits)
function Base.close(d::Decoder)
    Mantle.free!(d.plan);Mantle.free!(d.bias);nothing
end

end
