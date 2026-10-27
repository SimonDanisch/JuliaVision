# Controlled evidence-interface experiment. The oracle arm is deliberately named:
# injecting an exact answer verifies integration, not general reasoning gains.
using Random, JSON3, Test

function logic_cases(;seed=2751,per_status=4)
    rng=MersenneTwister(seed);cases=Any[]
    for status in (:entailed,:refuted,:unknown,:inconsistent), rep in 1:per_status
        depth=rand(rng,2:8)
        names=[Symbol("p",rand(rng,10000:99999),"_",i) for i in 1:depth+1]
        enabled=Symbol("enabled_",rand(rng,10000:99999))
        rules=[(;head=names[i+1],premises=[names[i],enabled]) for i in 1:depth]
        # An ungrounded cycle must not prove its own premises.
        append!(rules,[(;head=:loop_a,premises=[:loop_b]),
                       (;head=:loop_b,premises=[:loop_a])])
        facts=Symbol[enabled,:distractor]
        status in (:entailed,:inconsistent) && push!(facts,names[1])
        negative=status in (:refuted,:inconsistent) ? [names[end]] : Symbol[]
        query=names[end]
        push!(cases,(;id=length(cases)+1,depth,facts,rules,negative,query,expected=status))
    end
    shuffle!(rng,cases)
end

function logic_prompt(case;with_evidence)
    policy="Classify the query using only the supplied positive facts, explicit negative facts, and positive Horn rules. "*
        "Do not assume that failure to prove means false, and do not infer arbitrary facts from contradictions. "*
        "Answer exactly A if proven and not explicitly negative; B if explicitly negative and not proven; "*
        "C if neither; D if both."
    source=if with_evidence
        facts=join(string.(case.facts),", ")
        negative=isempty(case.negative) ? "none" : join(string.(case.negative),", ")
        rules=join([join(string.(r.premises)," AND ")*" -> "*string(r.head) for r in case.rules],"\n")
        "Positive facts: $facts\nExplicit negative facts: $negative\nRules:\n$rules\n"
    else
        "The knowledge base is held externally; its contents are not supplied in this message.\n"
    end
    ["system"=>policy,"user"=>source*"Query: $(case.query)\nReply with one letter."]
end

"""Compare missing evidence, textual evidence and an exact solver oracle.

The solver arm receives the same structured facts/rules rendered in the text
arm. It bypasses language grounding and forces the exact status label, making
its score an integration ceiling rather than an intelligence result.
"""
function run_logic_experiment(model,tokenizer,cases;output=nothing)
    labels=Dict(:entailed=>"A",:refuted=>"B",:unknown=>"C",:inconsistent=>"D")
    token_ids=Dict{Symbol,Int}()
    for (status,label) in labels
        ids=QwenText.encode(tokenizer,label)
        length(ids)==1 || throw(ArgumentError("status label must occupy one token"))
        token_ids[status]=only(ids)
    end
    rows=Any[]
    statuses=(:unknown,:entailed,:refuted,:inconsistent)
    label_ids=Tuple(token_ids[status] for status in statuses)
    for case in cases
        proof=PortableEvidence.prepare_horn(model.device,case.facts,case.rules,case.query;negative_facts=case.negative)
        evidence=nothing;enabled=nothing;s=nothing
        try
            evidence=KnowledgeBias.Evidence(model.device,Int(model.config.vocab_size))
            enabled=Mantle.GPURef(model.device,Int32(0))
            hook=function(g,logits)
                PortableEvidence.declare_horn!(g,proof.inputs,proof.outputs,proof.metadata)
                PortableEvidence.declare_status_mask!(g,evidence.bias,proof,label_ids,enabled)
                KnowledgeBias.declare!(g,logits,evidence)
            end
            s=Qwen35Experiment.session(model;capacity=2048,logit_hook=hook)
            for mode in (:no_evidence,:text,:solver_kernel_oracle)
                ids=QwenText.chat_ids(tokenizer,logic_prompt(case;with_evidence=mode===:text))
                reset_qwen35!(s)
                enabled[]=Int32(mode===:solver_kernel_oracle)
                elapsed=@elapsed begin
                    Qwen35Experiment.prefill!(s,ids;chunk=8)
                    KernelAbstractions.synchronize(model.device)
                end
                selected=Qwen35Experiment.greedy(s)
                prediction=only(status for (status,id) in token_ids if id==selected)
                proof_status=mode===:solver_kernel_oracle ?
                    statuses[Int(Array(proof.outputs.status)[proof.metadata.query])+1] : nothing
                row=(;id=case.id,mode=String(mode),depth=case.depth,
                    expected=String(case.expected),prediction=String(prediction),
                    correct=prediction===case.expected,prompt_tokens=length(ids),seconds=elapsed,proof_status)
                push!(rows,row);println(row);flush(stdout)
            end
        finally
            s===nothing || close(s)
            enabled===nothing || Mantle.free!(enabled)
            evidence===nothing || close(evidence)
            close(proof)
        end
    end
    summary=[(;mode,count=count(r->r.mode==mode,rows),
        accuracy=sum(r.correct for r in rows if r.mode==mode)/count(r->r.mode==mode,rows))
        for mode in ("no_evidence","text","solver_kernel_oracle")]
    result=(;scope="finite ground Horn status; shared solver kernels in neural graph; exact status readout ceiling; no learned latent adapter",summary,rows)
    output===nothing || open(io->JSON3.pretty(io,result),output,"w")
    result
end
