# Run in an environment containing JSON3. Downloads only pinned config.json
# files; no weights, remote code, or Python inference framework is loaded.
using Downloads, JSON3, TOML

function inspect_models(; bandwidth_gbps=220.0)
    rows = []
    for entry in TOML.parsefile(joinpath(@__DIR__, "models.toml"))["models"]
        url = "https://huggingface.co/$(entry["id"])/resolve/$(entry["revision"])/config.json"
        config = JSON3.read(read(Downloads.download(url), String))
        c = config.text_config
        full = count(==("full_attention"), c.layer_types)
        @assert full > 0
        @assert length(c.layer_types) == c.num_hidden_layers
        # Q8 matches BonsaiRunner's layout: two Int8 vectors plus two Float32
        # scales per KV head per attention layer per context position.
        fp16 = full * 2 * Int(c.num_key_value_heads) * Int(c.head_dim) * 2
        q8 = full * 2 * Int(c.num_key_value_heads) * (Int(c.head_dim) + 4)
        # GDN state is Float32, one key_dim × value_dim matrix per value head.
        # The small convolution history and allocator/workspace overhead are
        # deliberately separate from this recurrent-matrix calculation.
        recurrent = (Int(c.num_hidden_layers) - full) *
            Int(c.linear_num_value_heads) * Int(c.linear_key_head_dim) *
            Int(c.linear_value_head_dim) * 4
        contexts = [
            (; tokens=n, fp16_kv_gib=n*fp16/2.0^30, q8_kv_gib=n*q8/2.0^30,
               q8_cache_scan_only_tps=bandwidth_gbps*1e9/(n*q8))
            for n in (4096, 32768, 131072, 262144, 1_000_000)
        ]
        push!(rows, (; id=entry["id"], revision=entry["revision"],
                     role=entry["role"], config_url=url,
                     native_context=Int(c.max_position_embeddings),
                     full_attention_layers=full, fp16_bytes_per_token=fp16,
                     q8_bytes_per_token=q8, recurrent_matrix_mib=recurrent/2.0^20,
                     grouped_query_ratio=Int(c.num_attention_heads)÷Int(c.num_key_value_heads),
                     tied_embeddings=Bool(c.tie_word_embeddings),
                     experts=Int(get(c,:num_experts,0)),
                     experts_per_token=Int(get(c,:num_experts_per_tok,0)),
                     assumed_bandwidth_gbps=bandwidth_gbps, contexts))
    end
    rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    JSON3.pretty(stdout, inspect_models())
    println()
end
