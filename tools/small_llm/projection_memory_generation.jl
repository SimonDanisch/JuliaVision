isdefined(@__MODULE__,:load_projection_datastore) || include("projection_datastore.jl")
using Random

"""Memory lookup and probability mixing inside the recorded decode graph.

No model-selected lookup request or host token feedback. Shuffling is a negative
control, not additional knowledge. This pilot still needs task-level evidence.
"""
function generate_projection_memory_answer(model,tokenizer,messages,data;
        weight=0.1f0,shuffled=false,max_tokens=512,thinking=false)
    Int(model.config.vocab_size)==data.vocabulary || error("datastore/model vocabulary")
    labels=shuffled ? data.labels[randperm(MersenneTwister(20261024),length(data.labels))] : data.labels
    memory=nothing;features=nothing;query=nothing
    try
        memory=KNNMemory.Memory(model.device,data.keys,labels,data.vocabulary;
            topk=min(8,length(labels)),temperature=32f0,weight)
        features=LogitFeatures.Features(model.device,data.vocabulary,data.feature_ids)
        query=Mantle.Buffer(model.device,zeros(Float32,length(data.feature_ids)))
        hook=(graph,logits)->begin
            LogitFeatures.declare!(graph,query,logits,features)
            KNNMemory.declare!(graph,logits,query,memory)
        end
        result=generate_julia_answer(model,tokenizer,messages;max_tokens,thinking,
            output_chunk=128,logit_hook=hook)
        status=only(Array(memory.status))
        merge(result,(;memory_weight=Float32(weight),memory_temperature=32f0,memory_topk=min(8,length(labels)),
            shuffled_labels=shuffled,memory_status=status,
            datastore_registration_sha256=data.registration_sha256,
            scope="published-source output-projection retrieval, not a trained latent-memory bridge"))
    finally
        # generate_julia_answer closes its graph before these owned arguments.
        memory===nothing || close(memory)
        features===nothing || close(features)
        query===nothing || Mantle.free!(query)
    end
end
