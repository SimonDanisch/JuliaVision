# Metadata only: no model weights, device creation or remote model code.
using Downloads,JSON3,SHA,TOML

function audit_moe_checkpoint(;output=nothing)
    entries=TOML.parsefile(joinpath(@__DIR__,"models.toml"))["models"]
    entry=only(filter(e->e["id"]=="Qwen/Qwen3.6-35B-A3B",entries))
    root="https://huggingface.co/$(entry["id"])/resolve/$(entry["revision"])"
    raw_config=read(Downloads.download(root*"/config.json"))
    raw_index=read(Downloads.download(root*"/model.safetensors.index.json"))
    c=JSON3.read(String(copy(raw_config))).text_config
    index=JSON3.read(String(copy(raw_index)))
    keys=Set(String.(collect(Base.keys(index.weight_map))))
    layers=Int(c.num_hidden_layers);width=Int(c.hidden_size)
    f=Int(c.moe_intermediate_size);experts=Int(c.num_experts);active=Int(c.num_experts_per_tok)
    required=["lm_head.weight","model.language_model.embed_tokens.weight","model.language_model.norm.weight"]
    for layer in 0:layers-1
        prefix="model.language_model.layers.$layer.mlp."
        append!(required,prefix.*["experts.gate_up_proj","experts.down_proj","gate.weight",
            "shared_expert.gate_proj.weight","shared_expert.up_proj.weight",
            "shared_expert.down_proj.weight","shared_expert_gate.weight"])
    end
    missing=sort!(collect(setdiff(Set(required),keys)))
    isempty(missing) || error("native checkpoint layout differs: $missing")
    result=(;id=entry["id"],revision=entry["revision"],
        config_sha256=bytes2hex(sha256(raw_config)),index_sha256=bytes2hex(sha256(raw_index)),
        checkpoint_bytes=Int(index.metadata.total_size),shards=length(unique(values(index.weight_map))),
        width,layers,experts,active_experts=active,expert_ffn=f,
        tied_embeddings=Bool(c.tie_word_embeddings),output_head_key="lm_head.weight",
        grouped_query_ratio=Int(c.num_attention_heads)÷Int(c.num_key_value_heads),
        routed_expert_parameters=3*layers*width*f*experts,
        active_routed_expert_parameters=3*layers*width*f*active,
        q4_routed_expert_bytes=3*layers*width*f*experts÷256*144,
        max_expert_gateup_q4_vectors=2*f*experts*width÷256*9,
        required_native_keys_checked=length(required),
        exclusions=["vision tensors","mtp.* tensors"],
        scope="native layout/size audit, not complete model loading or throughput verification")
    output===nothing || open(io->JSON3.pretty(io,result),output,"w")
    result
end
