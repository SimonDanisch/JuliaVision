isdefined(@__MODULE__,:with_head_input_tap) || include("head_input_tap.jl")
isdefined(@__MODULE__,:KNNMemory) || include("knn_memory.jl")
isdefined(@__MODULE__,:generate_julia_answer) || include("run_julia_tasks.jl")
isdefined(@__MODULE__,:declare_memory_status_history!) || include("memory_status_history.jl")
using Random

"""Native hidden-state retrieval inside the existing recorded decode graph.

Keep the head tap installed until generation closes every session plan. There is
no host lookup request or CPU token feedback. Keep both the final status and a
sticky maximum across every query, including prompt ingestion. This diagnoses
validity, not retrieved neighbors or factual correctness.
"""
function generate_native_hidden_answer(model,tokenizer,messages,data;
        weight=0.1f0,temperature=32f0,shuffled=false,max_tokens=512,thinking=false)
    Int(model.config.hidden_size)==data.width && Int(model.config.vocab_size)==data.vocabulary ||
        error("native datastore/model dimensions")
    labels=shuffled ? data.labels[randperm(MersenneTwister(20261024),length(data.labels))] : data.labels
    memory=nothing;query=nothing;status_history=nothing
    try
        memory=KNNMemory.Memory(model.device,data.keys,labels,data.vocabulary;
            topk=min(8,length(labels)),temperature,weight)
        query=Mantle.Buffer(model.device,zeros(Float32,data.width))
        status_history=Mantle.Buffer(model.device,Int32[0])
        result=with_head_input_tap(model,query) do
            generate_julia_answer(model,tokenizer,messages;max_tokens,thinking,output_chunk=128,
                logit_hook=(graph,logits)->begin
                    KNNMemory.declare!(graph,logits,query,memory)
                    declare_memory_status_history!(graph,status_history,memory.status)
                end)
        end
        merge(result,(;memory_weight=Float32(weight),memory_temperature=Float32(temperature),
            memory_topk=min(8,length(labels)),shuffled_labels=shuffled,
            final_memory_status=only(Array(memory.status)),
            worst_memory_status=only(Array(status_history)),
            datastore_registration_sha256=data.registration_sha256,
            scope="native final-RMSNorm hidden-state retrieval and probability mixing, not a trained latent bridge"))
    finally
        memory===nothing || close(memory)
        query===nothing || Mantle.free!(query)
        status_history===nothing || Mantle.free!(status_history)
    end
end
